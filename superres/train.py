"""Training loops: multi-scale VQ-VAE, conditional VAR, SR generation, taggers."""
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader

from . import metrics as M
from .data import CacheStore, SRDataset, TaggerDataset, target_transform
from .io_utils import RunDir
from .models import VQVAE, ConditionalVAR, Tagger


# ------------------------------------------------------------------ helpers
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Amp:
    """bf16 autocast where supported (A100/L4), fp16 + GradScaler otherwise (T4)."""

    def __init__(self, cfg):
        self.on = bool(cfg["amp"]) and torch.cuda.is_available()
        self.dtype = torch.bfloat16 if self.on and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.on and self.dtype == torch.float16)

    def ctx(self):
        return torch.autocast("cuda", dtype=self.dtype, enabled=self.on)

    def step(self, loss, opt, params, clip=1.0):
        opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, clip)
        self.scaler.step(opt)
        self.scaler.update()


def loader(ds, bs, shuffle, cfg, drop_last=False):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last and len(ds) > bs,
                      num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available())


def set_lr(opt, step, total, base, warmup):
    if step < warmup:
        lr = base * (step + 1) / warmup
    else:
        lr = base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


def scale_tensor(store, dev):
    return torch.as_tensor(store.scale, device=dev).view(1, -1, *([1] * len(store.hr_shape)))


def to_energy(z, s):
    """normalised log-energy -> linear energy."""
    return s * torch.expm1(z.float().clamp(0, 30))


def _maybe_prune(trial, value, epoch):
    if trial is None:
        return
    import optuna
    trial.report(value, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()


def _decay_groups(model, wd):
    decay, no = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and "pos" not in n and "embed" not in n else no).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no, "weight_decay": 0.0}]


class EarlyStop:
    def __init__(self, patience, minimize=True):
        self.best, self.bad, self.patience, self.sign = math.inf, 0, patience, 1 if minimize else -1

    def update(self, value):
        v = self.sign * value
        if v < self.best:
            self.best, self.bad = v, 0
            return True
        self.bad += 1
        return False

    @property
    def stop(self):
        return self.bad >= self.patience

    @property
    def best_value(self):
        return self.sign * self.best


def _resume(run, model, opt):
    if run is None or not run.exists("last.pt"):
        return 0, None
    ck = torch.load(run.file("last.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    print(f"resumed {run.name} from epoch {ck['epoch'] + 1}")
    return ck["epoch"] + 1, ck.get("stopper")


def _save(run, name, model, opt, epoch, stopper, extra):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict() if opt else None,
                "epoch": epoch, "stopper": stopper.__dict__ if stopper else None, **extra}, run.file(name))


# ------------------------------------------------------------------ VQ-VAE
def build_vqvae(store, vq_cfg, circ):
    return VQVAE(len(store.hr_shape), store.C, vq_cfg, circ)


def load_vqvae(cfg, store, dev):
    run = RunDir(cfg, "vqvae", "hr", cfg["seed"])
    ck = torch.load(run.file("best.pt"), map_location="cpu", weights_only=False)
    model = build_vqvae(store, ck["vq_cfg"], cfg["data"]["circular_dims"])
    model.load_state_dict(ck["model"])
    return model.to(dev).eval().requires_grad_(False), ck["vq_cfg"]


@torch.no_grad()
def _eval_vqvae(model, dl, s, dev, amp, thresh):
    model.eval()
    mse, eerr, n = 0.0, [], 0
    for b in dl:
        x = b["hr"].to(dev, non_blocking=True)
        with amp.ctx():
            rec, _, _ = model(x)
        rec = rec.float().clamp_min(0)
        mse += F.mse_loss(rec, x, reduction="sum").item() / x[0].numel()
        dims = tuple(range(1, x.dim()))
        et, ep = to_energy(x, s).sum(dims), to_energy(rec, s).sum(dims)
        eerr.append(((ep - et).abs() / et.clamp_min(1e-9)).cpu())
        n += len(x)
    eerr = torch.cat(eerr).numpy()
    return {"val_rec_mse": mse / n, "val_energy_err_median": float(np.median(eerr)),
            "val_energy_err_mean": float(np.mean(np.clip(eerr, 0, 10))),
            "codebook_usage": float((model.vq.usage > thresh).float().mean())}


def train_vqvae(cfg, trial=None, epochs=None, max_train=None, save=True, resume=True):
    set_seed(cfg["seed"])
    dev, amp, store = device(), Amp(cfg), CacheStore(cfg)
    vc = cfg["vqvae"]
    epochs = epochs or vc["epochs"]
    tr = SRDataset(store, "train", max_n=max_train)
    va = SRDataset(store, "val", max_n=None if max_train is None else max(256, max_train // 8))
    dl_tr = loader(tr, vc["batch_size"], True, cfg, drop_last=True)
    dl_va = loader(va, vc["batch_size"], False, cfg)
    model = build_vqvae(store, vc, cfg["data"]["circular_dims"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=vc["lr"], betas=(0.9, 0.95), weight_decay=vc["weight_decay"])
    s = scale_tensor(store, dev)
    run = RunDir(cfg, "vqvae", "hr", cfg["seed"]) if save else None
    start, st = _resume(run, model, opt) if resume else (0, None)
    stopper = EarlyStop(vc["patience"])
    if st:
        stopper.__dict__.update(st)
    if run is not None:
        run.save_config({"n_params": sum(p.numel() for p in model.parameters())})
        if start == 0:
            run.reset_history()
    total, step = epochs * len(dl_tr), start * len(dl_tr)
    warm = min(500, total // 10 + 1)
    if start == 0:  # data-driven codebook init
        with torch.no_grad(), amp.ctx():
            model.vq.init_from(model.encode(next(iter(dl_tr))["hr"].to(dev)).float())
    epoch = start - 1
    for epoch in range(start, epochs):
        if stopper.stop:  # resumed a run that had already early-stopped
            print("already early-stopped; nothing to train")
            break
        model.train()
        t0, run_loss, f_last = time.time(), 0.0, None
        for b in dl_tr:
            lr = set_lr(opt, step, total, vc["lr"], warm)
            x = b["hr"].to(dev, non_blocking=True)
            with amp.ctx():
                rec, vq_loss, f_last = model(x)
            rec = rec.float()
            dims = tuple(range(1, x.dim()))
            e_true = to_energy(x, s).sum(dims)
            e_pred = to_energy(rec.clamp(max=30), s).sum(dims)
            e_loss = (torch.log(e_pred + 1e-3) - torch.log(e_true + 1e-3)).abs().mean()
            loss = F.mse_loss(rec, x) + vq_loss.float() + vc["energy_weight"] * e_loss
            amp.step(loss, opt, model.parameters())
            run_loss += loss.item()
            step += 1
        used = float((model.vq.usage > vc["dead_thresh"]).float().mean())  # before the reset below
        dead = model.vq.reinit_dead(f_last, vc["dead_thresh"])
        val = _eval_vqvae(model, dl_va, s, dev, amp, vc["dead_thresh"])
        val["codebook_usage"] = used
        objective = val["val_rec_mse"] + val["val_energy_err_median"]
        improved = stopper.update(objective)
        row = {"epoch": epoch, "lr": lr, "train_loss": run_loss / len(dl_tr), **val,
               "objective": objective, "dead_codes_reset": dead, "sec": round(time.time() - t0, 1)}
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
        if run is not None:
            run.log(row)
            extra = {"vq_cfg": vc}
            if improved:
                _save(run, "best.pt", model, None, epoch, None, extra)
            _save(run, "last.pt", model, opt, epoch, stopper, extra)
        _maybe_prune(trial, objective, epoch)
        if stopper.stop:
            print("early stop")
            break
    if run is not None:
        run.save_metrics({"best_objective": stopper.best_value, "last_epoch": epoch})
    return stopper.best_value


# --------------------------------------------------------------------- VAR
def build_var(store, vq_cfg, var_cfg, circ):
    return ConditionalVAR(len(store.hr_shape), store.C, store.hr_shape, vq_cfg, var_cfg, circ)


def load_var(cfg, store, level, dev):
    run = RunDir(cfg, "var", level, cfg["seed"])
    ck = torch.load(run.file("best.pt"), map_location="cpu", weights_only=False)
    model = build_var(store, ck["vq_cfg"], ck["var_cfg"], cfg["data"]["circular_dims"])
    model.load_state_dict(ck["model"])
    return model.to(dev).eval().requires_grad_(False)


def _var_batch(vq, b, dev, amp):
    x = b["hr"].to(dev, non_blocking=True)
    lr = b["lr"].to(dev, non_blocking=True)
    with torch.no_grad(), amp.ctx():
        idx = vq.img_to_idx(x)
        x_in = vq.vq.idx_to_var_input(idx)
    gt = torch.cat([i.flatten(1) for i in idx], 1)
    return lr, x_in, gt


@torch.no_grad()
def _eval_var(model, vq, dl, dev, amp):
    model.eval()
    ce, correct, correct_last, n_tok, n_last = 0.0, 0, 0, 0, 0
    b_last, _ = model.begin_ends[-1]
    for b in dl:
        lr, x_in, gt = _var_batch(vq, b, dev, amp)
        with amp.ctx():
            logits = model(lr, x_in)
        logits = logits.float()
        ce += F.cross_entropy(logits.reshape(-1, model.V), gt.reshape(-1), reduction="sum").item()
        pred = logits.argmax(-1)
        correct += (pred == gt).sum().item()
        correct_last += (pred[:, b_last:] == gt[:, b_last:]).sum().item()
        n_tok += gt.numel()
        n_last += gt[:, b_last:].numel()
    return {"val_ce": ce / n_tok, "val_token_acc": correct / n_tok, "val_token_acc_last_scale": correct_last / n_last}


def train_var(cfg, level, trial=None, epochs=None, max_train=None, save=True, resume=True):
    set_seed(cfg["seed"])
    dev, amp, store = device(), Amp(cfg), CacheStore(cfg)
    vc = cfg["var"]
    epochs = epochs or vc["epochs"]
    vq, vq_cfg = load_vqvae(cfg, store, dev)
    tr = SRDataset(store, "train", level, max_n=max_train)
    va = SRDataset(store, "val", level, max_n=None if max_train is None else max(256, max_train // 8))
    dl_tr = loader(tr, vc["batch_size"], True, cfg, drop_last=True)
    dl_va = loader(va, vc["batch_size"], False, cfg)
    model = build_var(store, vq_cfg, vc, cfg["data"]["circular_dims"]).to(dev)
    opt = torch.optim.AdamW(_decay_groups(model, vc["weight_decay"]), lr=vc["lr"], betas=(0.9, 0.95))
    run = RunDir(cfg, "var", level, cfg["seed"]) if save else None
    start, st = _resume(run, model, opt) if resume else (0, None)
    stopper = EarlyStop(vc["patience"])
    if st:
        stopper.__dict__.update(st)
    if run is not None:
        run.save_config({"level": level, "n_params": sum(p.numel() for p in model.parameters()),
                         "tokens_per_image": model.L})
        if start == 0:
            run.reset_history()
    total, step = epochs * len(dl_tr), start * len(dl_tr)
    warm = min(1000, total // 10 + 1)
    epoch = start - 1
    for epoch in range(start, epochs):
        if stopper.stop:  # resumed a run that had already early-stopped
            print("already early-stopped; nothing to train")
            break
        model.train()
        t0, run_loss = time.time(), 0.0
        for b in dl_tr:
            lr_now = set_lr(opt, step, total, vc["lr"], warm)
            lr, x_in, gt = _var_batch(vq, b, dev, amp)
            with amp.ctx():
                logits = model(lr, x_in)
            loss = F.cross_entropy(logits.float().reshape(-1, model.V), gt.reshape(-1),
                                   label_smoothing=vc["label_smoothing"])
            amp.step(loss, opt, model.parameters())
            run_loss += loss.item()
            step += 1
        val = _eval_var(model, vq, dl_va, dev, amp)
        improved = stopper.update(val["val_ce"])
        row = {"epoch": epoch, "lr": lr_now, "train_ce": run_loss / len(dl_tr), **val,
               "sec": round(time.time() - t0, 1)}
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
        if run is not None:
            run.log(row)
            extra = {"vq_cfg": vq_cfg, "var_cfg": vc, "level": level}
            if improved:
                _save(run, "best.pt", model, None, epoch, None, extra)
            _save(run, "last.pt", model, opt, epoch, stopper, extra)
        _maybe_prune(trial, val["val_ce"], epoch)
        if stopper.stop:
            print("early stop")
            break
    if run is not None:
        run.save_metrics({"best_val_ce": stopper.best_value, "last_epoch": epoch})
    return stopper.best_value


@torch.no_grad()
def generate_sr(cfg, level, splits=None, decodes=None):
    """Run VAR on LR events and cache the output exactly as generated (no energy correction).

    decodes: 'sample' draws each token from the predicted distribution (var.temperature, var.top_k);
             'greedy' always takes the most likely token."""
    dev, amp, store = device(), Amp(cfg), CacheStore(cfg)
    vc = cfg["var"]
    splits = splits or vc.get("gen_splits") or ("train", "val", "test")
    decodes = decodes or vc.get("decodes") or [vc.get("decode", "sample")]
    vq, _ = load_vqvae(cfg, store, dev)
    model = load_var(cfg, store, level, dev)
    s = scale_tensor(store, dev)
    dtype = np.float16 if vc.get("sr_dtype", "float32") == "float16" else np.float32
    for decode in decodes:
        gen = torch.Generator(device=dev).manual_seed(cfg["seed"])
        for split in splits:
            n = store.meta["n"][split]
            if vc["gen_max_per_split"]:
                n = min(n, vc["gen_max_per_split"])
            path = store.sr_path(split, level, decode)
            if os.path.exists(path) and not vc.get("gen_overwrite", False):
                have = len(np.load(path, mmap_mode="r"))
                if have >= n:
                    print(f"  {level} {split} [{decode}]: {have} events already generated - skipping "
                          f"(set var.gen_overwrite=true after retraining the model)")
                    continue
            ds = SRDataset(store, split, level, max_n=n)
            out = open_memmap(path + ".tmp", "w+", dtype, (n, store.C, *store.hr_shape))
            i, t0 = 0, time.time()
            for b in loader(ds, vc["gen_batch_size"], False, cfg):
                with amp.ctx():
                    z = model.generate(b["lr"].to(dev), vq, vc["temperature"], vc["top_k"], decode == "greedy", gen)
                E = to_energy(z.float(), s).cpu().numpy()
                out[i:i + len(E)] = E.astype(dtype)
                i += len(E)
                print(f"\r  {level} {split} [{decode}]: {i}/{n}", end="")
            out.flush()
            del out
            os.replace(path + ".tmp", path)
            print(f"  ({time.time() - t0:.0f}s)")


@torch.no_grad()
def reconstruct_with_tokenizer(cfg, store, hr, batch_size=64):
    """HR -> tokens -> HR with the tokenizer alone (no transformer). hr: linear energies (n, C, *grid).
    This is the best any token-predicting model could do with this tokenizer."""
    from .data import normalize
    dev, amp = device(), Amp(cfg)
    vq, _ = load_vqvae(cfg, store, dev)
    s = scale_tensor(store, dev)
    out = np.empty(hr.shape, dtype=np.float32)
    for i in range(0, len(hr), batch_size):
        x = torch.from_numpy(normalize(np.asarray(hr[i:i + batch_size], np.float32), store.scale, batch=True)).to(dev)
        with amp.ctx():
            rec, _, _ = vq(x)
        out[i:i + len(x)] = to_energy(rec.float(), s).cpu().numpy()
    return out


# ------------------------------------------------------------------ taggers
def build_tagger(cfg, store, kind, tcfg=None):
    tcfg = tcfg or cfg["tagger"]
    return Tagger(len(store.hr_shape), store.C, store.shape(kind), tcfg["widths"], 1, tcfg["dropout"],
                  cfg["data"]["circular_dims"], n_global=store.C)


@torch.no_grad()
def predict(model, dl, dev, amp):
    model.eval()
    out, ts = [], []
    for x, g, t in dl:
        with amp.ctx():
            p = model(x.to(dev, non_blocking=True), g.to(dev, non_blocking=True))
        out.append(p.float().squeeze(1).cpu())
        ts.append(t)
    return torch.cat(out).numpy(), torch.cat(ts).numpy()


def _task_metrics(cfg, pred, t_std, e_raw=None, stats=None):
    if cfg["tagger"]["task"] == "classification":
        return M.classification_metrics(t_std, pred)
    e_pred = np.exp(pred * stats[1] + stats[0])
    return M.response_metrics(e_raw, e_pred)


def fit_classifier(cfg, store, kind, arrays, targets, epochs, seed, tcfg=None, dev=None, amp=None):
    """Generic fit used by the tagger and the classifier two-sample test.
    arrays/targets: dict split -> array-like / np.ndarray (train, val[, test])."""
    tcfg = tcfg or cfg["tagger"]
    set_seed(seed)
    dev, amp = dev or device(), amp or Amp(cfg)
    ds = {k: TaggerDataset(store, arrays[k], targets[k], kind) for k in arrays}
    dls = {k: loader(v, tcfg["batch_size"], k == "train", cfg, drop_last=k == "train") for k, v in ds.items()}
    model = build_tagger(cfg, store, kind, tcfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total, step = epochs * len(dls["train"]), 0
    best, best_state, bad = -math.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for x, g, t in dls["train"]:
            set_lr(opt, step, total, tcfg["lr"], min(200, total // 10 + 1))
            with amp.ctx():
                p = model(x.to(dev, non_blocking=True), g.to(dev, non_blocking=True)).float().squeeze(1)
            loss = F.binary_cross_entropy_with_logits(p, t.to(dev))
            amp.step(loss, opt, model.parameters())
            step += 1
        pv, tv = predict(model, dls["val"], dev, amp)
        auc = M.classification_metrics(tv, pv, n_boot=0)["auc"]
        if auc > best or best_state is None:
            best, best_state, bad = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, dls


def train_tagger(cfg, kind, seed, trial=None, epochs=None, max_train=None, save=True):
    """Train the downstream model on one input representation.
    qg: quark/gluon classifier. calo: incident-energy regressor."""
    set_seed(seed)
    dev, amp, store = device(), Amp(cfg), CacheStore(cfg)
    tc = cfg["tagger"]
    epochs = epochs or tc["epochs"]
    max_train = max_train or tc["max_train"]
    classify = tc["task"] == "classification"
    raw = {s: store.target(s) for s in ("train", "val", "test")}
    t_tr, stats = target_transform(cfg, raw["train"])
    tgt = {"train": t_tr, "val": target_transform(cfg, raw["val"], stats)[0],
           "test": target_transform(cfg, raw["test"], stats)[0]}
    max_val = None if trial is None else max(512, (max_train or 0) // 4) or None
    ds = {"train": TaggerDataset(store, store.array("train", kind), tgt["train"], kind, max_train),
          "val": TaggerDataset(store, store.array("val", kind), tgt["val"], kind, max_val),
          "test": TaggerDataset(store, store.array("test", kind), tgt["test"], kind)}
    dls = {k: loader(v, tc["batch_size"], k == "train", cfg, drop_last=k == "train") for k, v in ds.items()}
    model = build_tagger(cfg, store, kind).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    variant = kind.replace(":", "-")
    run = RunDir(cfg, "tagger", variant, seed) if save else None
    if run is not None:
        run.save_config({"kind": kind, "n_train": len(ds["train"]), "target_stats": stats})
        run.reset_history()
    stopper = EarlyStop(tc["patience"], minimize=not classify)
    total, step, best_state = epochs * len(dls["train"]), 0, None
    for epoch in range(epochs):
        model.train()
        t0, run_loss = time.time(), 0.0
        for x, g, t in dls["train"]:
            lr_now = set_lr(opt, step, total, tc["lr"], min(500, total // 10 + 1))
            with amp.ctx():
                p = model(x.to(dev, non_blocking=True), g.to(dev, non_blocking=True)).float().squeeze(1)
            t = t.to(dev)
            loss = F.binary_cross_entropy_with_logits(p, t) if classify else F.huber_loss(p, t, delta=1.0)
            amp.step(loss, opt, model.parameters())
            run_loss += loss.item()
            step += 1
        pv, tv = predict(model, dls["val"], dev, amp)
        if classify:
            vm = M.classification_metrics(tv, pv, n_boot=0)
            objective = vm["auc"]
        else:
            vm = M.response_metrics(raw["val"][:len(pv)], np.exp(pv * stats[1] + stats[0]))
            vm.pop("binned")
            objective = vm["mean_binned_resolution"]
        if stopper.update(objective) or best_state is None:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        row = {"epoch": epoch, "lr": lr_now, "train_loss": run_loss / len(dls["train"]),
               **{f"val_{k}": v for k, v in vm.items()}, "sec": round(time.time() - t0, 1)}
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
        if run is not None:
            run.log(row)
        _maybe_prune(trial, objective, epoch)
        if stopper.stop:
            break
    model.load_state_dict(best_state)
    if run is None:
        return stopper.best_value
    torch.save({"model": best_state, "tagger_cfg": tc, "kind": kind, "target_stats": stats}, run.file("best.pt"))
    pt, tt = predict(model, dls["test"], dev, amp)
    test = _task_metrics(cfg, pt, tt, raw["test"][:len(pt)], stats)
    np.savez_compressed(run.file("test_predictions.npz"), pred=pt, target=tt, target_raw=raw["test"][:len(pt)])
    run.save_metrics({"kind": kind, "seed": seed, "val_objective": stopper.best_value,
                      "n_train": len(ds["train"]), "n_test": len(pt), "test": test})
    print(f"[{run.name}] test:", {k: v for k, v in test.items() if k != "binned"})
    return stopper.best_value
