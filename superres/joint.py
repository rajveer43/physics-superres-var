"""Shower-level stages of the CaloGAN dataset ("calogan"), run after the three layer datasets
(calogan_l0 / _l1 / _l2) have been prepared, trained and generated.

    evaluate_joint     reassemble HR / LR / SR / baselines into whole showers, shower observables per particle
                       and per E_inc bin, inter-layer correlations, energy closure, energy response and the
                       resolution fit sigma/E = a/sqrt(E) (+) b, hit-threshold sensitivity
    train_joint_tagger LayerStackTagger on one input representation: 3-class PID or incident-energy regression
    joint_cross_eval   the 2x3 table: a tagger trained on HR applied to every input vs a tagger trained on each
    cross_class_report SR trained on e+ / gamma only, looked at on pi+ (does pi+ become EM-like?)
    summarize_joint    tables and a report

Nothing here feeds E_inc or the overflow to an image input. E_inc is the regression target and the E bin of
the evaluation; overflow is only used in closure checks.
"""
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from . import io_utils  # noqa: E402
from . import metrics as M  # noqa: E402
from .config import apply_overrides, layer_configs  # noqa: E402
from .data import CacheStore, normalize  # noqa: E402
from .evaluate import (KIND, ORDER, _comparable, _hist_with_ratio, _plot_2x3, _style, _table_2x3,  # noqa: E402
                       by_class_table, has_values)
from .models import LayerStackTagger  # noqa: E402
from .observables import calogan_shower_observables  # noqa: E402
from .physics_ops import sum_pool  # noqa: E402

PAIRS = {"e_vs_gamma": (0, 1), "e_vs_pi": (0, 2), "gamma_vs_pi": (1, 2)}   # (signal, background) labels
KEY_OBSERVABLES = ("e_vis_ratio", "f_0", "f_1", "f_2", "depth_mm", "n_hits", "eta_width_l0", "eratio_l0",
                   "core2d_80mm_l1", "eta_width_l1", "phi_width_l1", "eta_width_l2")
CORR_OBSERVABLES = ("f_0", "f_2", "depth_mm", "eta_width_l0", "eta_width_l1", "phi_width_l1", "core2d_80mm_l1",
                    "n_hits")
METHODS_FOR_TAGGER_A = ("hr", "uniform", "sr", "srproj", "template", "ridge")


# ---------------------------------------------------------------- the three layer caches
class JointStore:
    """The caches and SR outputs of the three layers, which share events (same split, same order)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.layer_cfgs = layer_configs(cfg)
        self.names = sorted(self.layer_cfgs)
        self.stores = {n: CacheStore(c) for n, c in self.layer_cfgs.items()}
        s0 = self.stores[self.names[0]]
        self.levels = list(s0.levels)
        self.meta = s0.meta
        for n in self.names[1:]:   # the whole point: identical events in every layer
            a, b = s0, self.stores[n]
            if a.meta["n"] != b.meta["n"] or not np.array_equal(a.particle("test"), b.particle("test")):
                raise RuntimeError(f"layer caches {self.names[0]} and {n} do not hold the same events")
        self.dcfgs = {n: c["data"] for n, c in self.layer_cfgs.items()}

    def n(self, split):
        return self.meta["n"][split]

    def target(self, split):
        """E_inc in MeV."""
        return self.stores[self.names[0]].target(split)

    def particle(self, split):
        return self.stores[self.names[0]].particle(split)

    def overflow(self, split):
        return self.stores[self.names[0]].overflow(split)

    def kind_ok(self, kind, split="test"):
        method, level = kind.partition(":")[0], kind.partition(":")[2]
        method = {"var": "srproj", "varraw": "sr"}.get(method, method)
        for st in self.stores.values():
            if method in ("hr", "lr", "uniform"):
                continue
            if method in ("template", "ridge"):
                if not st.has_baseline(split, method, level):
                    return False
            elif method == "vqrec":
                if not os.path.exists(st.vqrec_path(split, level)):
                    return False
            elif not st.has_sr(split, level, st.sr_decode(method)):
                return False
        return True

    def arrays(self, split, kind):
        return {n: st.array(split, kind) for n, st in self.stores.items()}

    def factors(self, kind):
        return {n: st.factor(kind) for n, st in self.stores.items()}

    def shapes(self, kind):
        return [self.stores[n].shape(kind) for n in self.names]

    def n_common(self, split, levels):
        return min(st.n_common(split, levels) for st in self.stores.values())


def layer_energies(arrs):
    """name -> (N, 1, H, W) float32 arrays held in memory."""
    return {n: np.asarray(a, np.float32) for n, a in arrs.items()}


# ---------------------------------------------------------------- shower-level evaluation
def fit_resolution(e_gev, sigma_rel):
    """sigma/E = a / sqrt(E) (+) b  ->  (a, b), with a in sqrt(GeV)."""
    from scipy.optimize import curve_fit
    e, s = np.asarray(e_gev, float), np.asarray(sigma_rel, float)
    ok = np.isfinite(s) & (s > 0)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    try:
        (a, b), _ = curve_fit(lambda x, a, b: np.sqrt(a ** 2 / x + b ** 2), e[ok], s[ok], p0=(0.1, 0.01), maxfev=5000)
    except (RuntimeError, ValueError):
        return float("nan"), float("nan")
    return abs(float(a)), abs(float(b))


def energy_response_table(vis, e_inc_gev, particle, particles, method, level, n_bins):
    """Calorimetric response E_vis / E_inc per particle and E bin, and the stochastic / constant fit."""
    edges = np.linspace(1.0, 100.0, n_bins + 1)
    rows = []
    R = vis / (e_inc_gev * 1000.0)
    for pi, pname in enumerate(particles):
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (particle == pi) & (e_inc_gev >= lo) & (e_inc_gev < hi + (1e-9 if hi == 100.0 else 0))
            if m.sum() < 20:
                continue
            med = float(np.median(R[m]))
            bins.append((0.5 * (lo + hi), med - 1.0, M.robust_sigma(R[m]) / max(med, 1e-9), int(m.sum())))
        a, b = fit_resolution([x[0] for x in bins], [x[2] for x in bins])
        for e, bias, res, n in bins:
            rows.append({"level": level, "method": method, "particle": pname, "e_gev": e, "n": n,
                         "response_bias": bias, "resolution": res, "fit_a": a, "fit_b": b})
    return rows


def correlation_table(obs, ref, particle, particles, method, level):
    """Frobenius distance between the HR and method correlation matrices of CORR_OBSERVABLES, per particle."""
    names = [k for k in CORR_OBSERVABLES if k in ref and has_values(obs.get(k)) and has_values(ref[k])]
    rows = []
    for pi, pname in [(None, "all"), *enumerate(particles)]:
        m = np.ones(len(particle), bool) if pi is None else particle == pi
        if m.sum() < 30 or len(names) < 2:
            continue

        def corr(o):
            x = np.stack([np.asarray(o[k], float)[m] for k in names], 1)
            ok = np.isfinite(x).all(1)
            c = np.corrcoef(x[ok].T) if ok.sum() > 10 else np.full((len(names),) * 2, np.nan)
            return np.nan_to_num(c)

        c_ref, c_m = corr(ref), corr(obs)
        rows.append({"level": level, "method": method, "particle": pname, "n": int(m.sum()),
                     "frobenius_distance": float(np.linalg.norm(c_m - c_ref)),
                     **{f"corr_{a}_{b}": float(c_m[i, j]) for i, a in enumerate(names) for j, b in enumerate(names)
                        if j > i and {a, b} in ({"f_0", "f_2"}, {"f_0", "depth_mm"}, {"eta_width_l0", "eta_width_l1"})},
                     **{f"corr_ref_{a}_{b}": float(c_ref[i, j]) for i, a in enumerate(names)
                        for j, b in enumerate(names)
                        if j > i and {a, b} in ({"f_0", "f_2"}, {"f_0", "depth_mm"}, {"eta_width_l0", "eta_width_l1"})}})
    return rows


def threshold_scan(js, layers, ref_layers, particle, particles, method, level):
    """Occupancy and hits per layer for several readout thresholds, method vs HR (hit-count agreement)."""
    rows = []
    for name in js.names:
        for thr in js.dcfgs[name]["threshold_scan_mev"]:
            h_ref, h = ref_layers[name][:, 0] > thr, layers[name][:, 0] > thr
            for pi, pname in [(None, "all"), *enumerate(particles)]:
                m = np.ones(len(particle), bool) if pi is None else particle == pi
                if not m.any():
                    continue
                n_ref, n_m = h_ref[m].sum((1, 2)).astype(float), h[m].sum((1, 2)).astype(float)
                rows.append({"level": level, "method": method, "layer": name, "particle": pname,
                             "threshold_mev": thr, "occupancy_ref": float(h_ref[m].mean()),
                             "occupancy": float(h[m].mean()),
                             "mean_hits_ref": float(n_ref.mean()), "mean_hits": float(n_m.mean()),
                             "hits_pearson_r": float(np.corrcoef(n_ref, n_m)[0, 1])
                             if n_ref.std() > 0 and n_m.std() > 0 else float("nan")})
    return rows


def shower_closure(js, layers, lr_layers, ref_layers, overflow, e_inc, method, level):
    """LR closure per layer and energy closure of the reassembled shower against HR."""
    out = {}
    for name in js.names:
        f = js.stores[name].levels[level]
        lr = lr_layers[name]
        c = np.abs(sum_pool(layers[name], f) - lr).reshape(len(lr), -1).sum(1) / np.maximum(
            lr.reshape(len(lr), -1).sum(1), 1e-12)
        out[f"lr_closure_median_{name[-2:]}"] = float(np.median(c))
    vis = sum(layers[n].reshape(len(layers[n]), -1).sum(1).astype(np.float64) for n in js.names)
    vis_ref = sum(ref_layers[n].reshape(len(ref_layers[n]), -1).sum(1).astype(np.float64) for n in js.names)
    d = (vis - vis_ref) / np.maximum(vis_ref, 1e-12)
    out.update({"shower_energy_diff_median": float(np.median(d)), "shower_energy_diff_mean": float(d.mean()),
                "shower_energy_diff_abs_p95": float(np.percentile(np.abs(d), 95))})
    if method == "hr":
        out["visible_plus_overflow_over_einc_median"] = float(np.median((vis + overflow.sum(1)) / e_inc))
    return out


def evaluate_joint(cfg, level):
    js = JointStore(cfg)
    ev = cfg["eval"]
    particles = cfg["data"]["particles"]
    run = io_utils.RunDir(cfg, "sreval", level)
    pfx = f"{io_utils.prefix(cfg)}_{level}"
    primary = js.stores[js.names[0]].cfg["var"].get("decode", "greedy")

    def ok(m):
        if m == "lr" or m == "uniform":
            return True
        if m in ("sr-sample", "sr-greedy") and js.stores[js.names[0]].sr_decode(KIND[m]) == primary:
            return False
        return js.kind_ok(f"{KIND.get(m, m)}:{level}")

    main = [m for m in ev["methods"] if ok(m)]
    diag = [m for m in ev.get("diagnostics", []) if m not in main and ok(m)]
    n = js.n("test")
    for m in main + diag:
        if m.startswith("sr"):
            n = min(n, js.n_common("test", [level]))
    n = min(n, ev["max_events"])
    e_inc = js.target("test")[:n]
    e_gev = e_inc / 1000.0
    particle = js.particle("test")[:n]
    overflow = js.overflow("test")[:n]
    ref_layers = layer_energies({k: v[:n] for k, v in js.arrays("test", "hr").items()})
    lr_layers = layer_energies({k: v[:n] for k, v in js.arrays("test", f"lr:{level}").items()})
    ones = {k: (1, 1) for k in js.names}
    ref_s, ref_p = calogan_shower_observables(ref_layers, ones, js.dcfgs, e_inc)
    ref_s = {k: v for k, v in ref_s.items() if has_values(v)}

    obs, profs, rows, paired, by_class, corr, resp, thr_rows, closure = {}, {}, [], [], [], [], [], [], {}
    resp += energy_response_table(sum(ref_layers[k].reshape(n, -1).sum(1) for k in js.names).astype(float),
                                  e_gev, particle, particles, "hr", level, ev["n_energy_bins"])
    closure["hr"] = shower_closure(js, ref_layers, lr_layers, ref_layers, overflow, e_inc, "hr", level)
    for m in main + diag:
        layers = layer_energies({k: v[:n] for k, v in js.arrays("test", f"{KIND.get(m, m)}:{level}").items()})
        factors = js.factors(f"{KIND.get(m, m)}:{level}")
        s, p = calogan_shower_observables(layers, factors, js.dcfgs, e_inc)
        obs[m], profs[m] = s, p
        role = "main" if m in main else "diagnostic"
        comparable = {k: v for k, v in s.items() if _comparable(m, k)}
        by_class += by_class_table(ref_s, comparable, particle, e_gev, list(ref_s), "calogan", level, m,
                                   ev["n_energy_bins"], particles)
        for name in ref_s:
            if name in comparable and has_values(comparable[name]):
                rel = name.startswith(("e_vis", "e_ratio"))
                rows.append({"dataset": "calogan", "level": level, "method": m, "role": role, "observable": name,
                             **M.compare_distributions(ref_s[name], comparable[name], rel)})
                paired.append({"dataset": "calogan", "level": level, "method": m, "role": role, "observable": name,
                               **M.paired_metrics(ref_s[name], comparable[name])})
        corr += correlation_table(comparable, ref_s, particle, particles, m, level)
        resp += energy_response_table(sum(layers[k].reshape(n, -1).sum(1) for k in js.names).astype(float),
                                      e_gev, particle, particles, m, level, ev["n_energy_bins"])
        if m != "lr":
            thr_rows += threshold_scan(js, layers, ref_layers, particle, particles, m, level)
            closure[m] = shower_closure(js, layers, lr_layers, ref_layers, overflow, e_inc, m, level)
        del layers
    pd.DataFrame(rows).to_csv(run.file("observables.csv"), index=False)
    pd.DataFrame(paired).to_csv(run.file("paired.csv"), index=False)
    pd.DataFrame(by_class).to_csv(run.file("observables_by_class.csv"), index=False)
    pd.DataFrame(corr).to_csv(run.file("correlations.csv"), index=False)
    pd.DataFrame(resp).to_csv(run.file("energy_response.csv"), index=False)
    pd.DataFrame(thr_rows).to_csv(run.file("threshold_scan.csv"), index=False)
    run.save_config({"level": level, "n_events": n, "main_methods": main, "diagnostic_methods": diag,
                     "layers": js.names})
    run.save_metrics({"level": level, "n_events": n, "closure": closure})

    diag_dir = os.path.join(run.fig_dir, "diagnostics")
    for pi, pname in enumerate(particles):
        sel = particle == pi
        if sel.sum() < 30:
            continue
        for name in [k for k in KEY_OBSERVABLES if k in ref_s]:
            def pick(methods):
                return {m: obs[m][name][sel] for m in methods if name in obs[m] and _comparable(m, name)
                        and has_values(obs[m][name][sel])}
            _hist_with_ratio(ref_s[name][sel], pick(main), name, ev["hist_bins"], run.fig_dir, f"{pfx}_{pname}")
            if diag:
                _hist_with_ratio(ref_s[name][sel], pick(main + diag), name, ev["hist_bins"], diag_dir,
                                 f"{pfx}_{pname}")
    _plot_response(pd.DataFrame(resp), level, run.fig_dir, pfx)
    _plot_f0_f2(ref_s, obs, main, particle, particles, level, run.fig_dir, pfx)
    print(f"[{run.name}] {n} showers; main: {main}; diagnostics: {diag}; figures in {run.fig_dir}")
    return pd.DataFrame(rows)


def _plot_response(resp, level, fig_dir, pfx):
    """Resolution sigma/E of E_vis versus E_inc for each particle, with the a/sqrt(E) (+) b fit of HR."""
    if resp.empty:
        return
    particles = list(dict.fromkeys(resp["particle"]))
    fig, axes = plt.subplots(1, len(particles), figsize=(4.2 * len(particles), 3.4), squeeze=False)
    for ax, p in zip(axes[0], particles):
        for m, g in resp[resp["particle"] == p].groupby("method"):
            ax.plot(g["e_gev"], g["resolution"], "o-", ms=3, label=f"{_style(m)['label']} (a={g['fit_a'].iloc[0]:.2f}, "
                    f"b={g['fit_b'].iloc[0]:.3f})", color=_style(m)["color"])
        ax.set_title(p, fontsize=9)
        ax.set_xlabel("E_inc [GeV]")
        ax.set_ylabel("sigma_eff(E_vis / E_inc)")
        ax.set_yscale("log")
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    io_utils.save_figure(fig, fig_dir, f"{pfx}_energy-resolution")
    plt.close(fig)


def _plot_f0_f2(ref, obs, methods, particle, particles, level, fig_dir, pfx):
    """The longitudinal correlation: layer-0 fraction against layer-2 fraction, HR next to each method."""
    cols = ["hr"] + [m for m in methods if m != "lr" and "f_2" in obs[m]]
    fig, axes = plt.subplots(len(particles), len(cols), figsize=(2.6 * len(cols), 2.4 * len(particles)),
                             squeeze=False, sharex=True, sharey=True)
    for i, pname in enumerate(particles):
        sel = particle == i
        for j, m in enumerate(cols):
            o = ref if m == "hr" else obs[m]
            axes[i, j].hist2d(np.asarray(o["f_0"])[sel], np.asarray(o["f_2"])[sel], bins=30,
                              range=[[0, 0.6], [0, 0.6]], cmap="viridis", norm=plt.matplotlib.colors.LogNorm())
            if i == 0:
                axes[i, j].set_title(_style(m)["label"] if m != "hr" else "HR (truth)", fontsize=7)
        axes[i, 0].set_ylabel(f"{pname}\nf_2", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("f_0", fontsize=8)
    fig.tight_layout()
    io_utils.save_figure(fig, fig_dir, f"{pfx}_f0-vs-f2")
    plt.close(fig)


# ---------------------------------------------------------------- joint taggers
def pid_metrics(y, logits, n_boot=0):
    """3-class metrics: accuracy, one-vs-rest macro AUC, and for each pair the AUC (+ bootstrap error) and the
    background rejection at 30 / 50 / 70 % signal efficiency, from the log-probability ratio."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    z = np.asarray(logits, np.float64)
    lp = z - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) - z.max(1, keepdims=True)
    out = {"accuracy": float((lp.argmax(1) == y).mean())}
    try:
        out["auc_ovr_macro"] = float(roc_auc_score(y, np.exp(lp), multi_class="ovr"))
    except ValueError:
        out["auc_ovr_macro"] = float("nan")
    aucs = []
    for name, (s, b) in PAIRS.items():
        m = (y == s) | (y == b)
        mm = M.classification_metrics((y[m] == s).astype(int), lp[m, s] - lp[m, b], n_boot=n_boot)
        out[f"auc_{name}"] = mm["auc"]
        aucs.append(mm["auc"])
        if "auc_boot_err" in mm:
            out[f"auc_{name}_boot_err"] = mm["auc_boot_err"]
        for e in M.SIG_EFFS:
            out[f"rej{int(e * 100)}_{name}"] = mm[f"rej_at_eff{int(e * 100)}"]
    out["auc_pair_mean"] = float(np.mean(aucs))
    return out


class JointTaggerDataset(Dataset):
    """(three normalised layer images, log layer energies, target) for the same events as every other input."""

    def __init__(self, js, split, kind, targets, n):
        self.js, self.kind, self.t, self.n = js, kind, targets, n
        self.arrs = [js.stores[k].array(split, kind) for k in js.names]
        self.vol = [js.stores[k].vol(kind) for k in js.names]
        self.scale = [js.stores[k].scale for k in js.names]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        xs, g = [], []
        for a, vol, sc in zip(self.arrs, self.vol, self.scale):
            E = np.asarray(a[i], np.float32)
            xs.append(torch.from_numpy(normalize(E, sc, vol)))
            g.append(np.log1p(E.sum()) / 5.0)   # layer energy in MeV, order 1
        return xs, torch.tensor(g, dtype=torch.float32), torch.as_tensor(self.t[i])


def _flip_joint(xs):
    """One random eta / phi reflection per shower, applied to all three layers together. Valid because the beam
    hits the middle of the calorimeter, so every grid is symmetric about it (see reflection_asymmetry)."""
    B = xs[0].size(0)
    for d in (-1, -2):
        m = (torch.rand(B, device=xs[0].device) < 0.5).view(B, 1, 1, 1)
        xs = [torch.where(m, x.flip(d), x) for x in xs]
    return xs


def reflection_asymmetry(js, split="train", n=20000):
    """|mean profile - its mirror image| / mean profile, per layer and axis, for HR: the symmetry that licenses the
    flip augmentation. Small (statistical) values mean the augmentation does not change the physics."""
    out = {}
    for k in js.names:
        X = np.asarray(js.stores[k].array(split, "hr")[:n], np.float64)[:, 0].mean(0)
        for ax, name in ((0, "phi"), (1, "eta")):
            p = X.sum(1 - ax)
            out[f"{k}_{name}"] = float(np.abs(p - p[::-1]).sum() / max(p.sum(), 1e-12))
    return out


def joint_tagger_settings(cfg):
    tc = dict(cfg["tagger"])
    tc["classify"] = tc["task"] == "pid"
    return tc


def _joint_targets(cfg, js, tc):
    raw = {s: js.target(s) for s in ("train", "val", "test")}
    if tc["classify"]:
        return {s: js.particle(s).astype(np.int64) for s in raw}, raw, None
    lt = np.log(raw["train"])
    stats = (float(lt.mean()), float(lt.std()))
    return {s: ((np.log(raw[s]) - stats[0]) / stats[1]).astype(np.float32) for s in raw}, raw, stats


@torch.no_grad()
def _predict(model, dl, dev, amp):
    model.eval()
    outs, ts = [], []
    for xs, g, t in dl:
        with amp.ctx():
            p = model([x.to(dev, non_blocking=True) for x in xs], g.to(dev, non_blocking=True))
        outs.append(p.float().cpu())
        ts.append(t)
    return torch.cat(outs).numpy(), torch.cat(ts).numpy()


def _task_metrics(tc, pred, target, e_raw, stats, particle=None, n_boot=0):
    if tc["classify"]:
        return pid_metrics(target, pred, n_boot)
    e_pred = np.exp(pred.squeeze(1) * stats[1] + stats[0])
    m = M.response_metrics(e_raw, e_pred)
    binned = m.pop("binned")
    a, b = fit_resolution([math.sqrt(x["e_lo"] * x["e_hi"]) / 1000 for x in binned], [x["resolution"] for x in binned])
    m.update({"fit_a": a, "fit_b": b})
    if particle is not None:
        for pi, pname in enumerate(("eplus", "gamma", "piplus")):
            sel = particle == pi
            if sel.sum() > 40:
                mp = M.response_metrics(e_raw[sel], e_pred[sel], n_bins=5)
                m[f"response_bias_{pname}"], m[f"resolution_{pname}"] = mp["response_bias"], mp["resolution"]
    return m, binned


def joint_tagger_done(cfg, kind, seed):
    path = io_utils.run_path(cfg, "tagger", kind.replace(":", "-"), seed)
    return os.path.exists(os.path.join(path, "metrics.json")) and os.path.exists(os.path.join(path, "best.pt"))


def train_joint_tagger(cfg, kind, seed, save=True):
    from .train import Amp, EarlyStop, device, loader, set_lr, set_seed
    set_seed(seed)
    dev, amp, js = device(), Amp(cfg), JointStore(cfg)
    tc = joint_tagger_settings(cfg)
    targets, raw, stats = _joint_targets(cfg, js, tc)
    levels = list(cfg["data"]["levels"])
    cap = {s: min(tc["max_train"] or math.inf, js.n_common(s, levels)) if s == "train" else js.n_common(s, levels)
           for s in ("train", "val", "test")}
    ds = {s: JointTaggerDataset(js, s, kind, targets[s], int(cap[s])) for s in cap}
    dls = {k: loader(v, tc["batch_size"], k == "train", cfg, drop_last=k == "train") for k, v in ds.items()}
    model = LayerStackTagger(js.shapes(kind), tc["widths"], 3 if tc["classify"] else 1, tc["dropout"],
                             n_global=len(js.names)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    variant = kind.replace(":", "-")
    run = io_utils.RunDir(cfg, "tagger", variant, seed) if save else None
    if run is not None:
        run.save_config({"kind": kind, "n_train": len(ds["train"]), "target_stats": stats,
                         "reflection_asymmetry": reflection_asymmetry(js)})
        run.reset_history()
    stopper = EarlyStop(tc["patience"], minimize=not tc["classify"])
    total, step, best_state = tc["epochs"] * len(dls["train"]), 0, None
    for epoch in range(tc["epochs"]):
        model.train()
        t0, run_loss, n_right, n_seen = time.time(), 0.0, 0, 0
        for xs, g, t in dls["train"]:
            lr_now = set_lr(opt, step, total, tc["lr"], min(300, total // 10 + 1))
            xs = [x.to(dev, non_blocking=True) for x in xs]
            if tc.get("augment", False):
                xs = _flip_joint(xs)
            t = t.to(dev)
            with amp.ctx():
                p = model(xs, g.to(dev, non_blocking=True)).float()
            loss = F.cross_entropy(p, t) if tc["classify"] else F.huber_loss(p.squeeze(1), t, delta=1.0)
            amp.step(loss, opt, model.parameters())
            run_loss += loss.item()
            if tc["classify"]:
                n_right += (p.argmax(1) == t).sum().item()
                n_seen += len(t)
            step += 1
        pv, tv = _predict(model, dls["val"], dev, amp)
        if tc["classify"]:
            vm = pid_metrics(tv, pv)
            vm["loss"] = F.cross_entropy(torch.from_numpy(pv), torch.from_numpy(tv)).item()
            objective = vm["auc_pair_mean"]
        else:
            vm, _ = _task_metrics(tc, pv, tv, raw["val"][:len(pv)], stats)
            objective = vm["mean_binned_resolution"]
        if stopper.update(objective) or best_state is None:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        row = {"epoch": epoch, "lr": lr_now, "train_loss": run_loss / len(dls["train"]),
               **({"train_accuracy": n_right / max(n_seen, 1)} if tc["classify"] else {}),
               **{f"val_{k}": v for k, v in vm.items() if not isinstance(v, (list, dict))},
               "best_so_far": stopper.bad == 0, "sec": round(time.time() - t0, 1)}
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
        if run is not None:
            run.log(row)
        if stopper.stop:
            break
    model.load_state_dict(best_state)
    if run is None:
        return stopper.best_value
    torch.save({"model": best_state, "tagger_cfg": tc, "kind": kind, "target_stats": stats,
                "shapes": js.shapes(kind)}, run.file("best.pt"))
    pt, tt = _predict(model, dls["test"], dev, amp)
    particle = js.particle("test")[:len(pt)]
    test = _task_metrics(tc, pt, tt, raw["test"][:len(pt)], stats, particle, n_boot=100)
    binned = None
    if not tc["classify"]:
        test, binned = test
    np.savez_compressed(run.file("test_predictions.npz"), pred=pt, target=tt, e_inc_mev=raw["test"][:len(pt)],
                        particle=particle)
    run.save_metrics({"kind": kind, "seed": seed, "val_objective": stopper.best_value, "n_train": len(ds["train"]),
                      "n_val": len(ds["val"]), "n_test": len(pt), "same_events_all_inputs": True, "test": test,
                      "binned": binned})
    print(f"[{run.name}] test:", {k: round(v, 4) for k, v in test.items() if isinstance(v, float)})
    return stopper.best_value


def joint_tagger_kinds(cfg, levels=None):
    kinds = ["hr"]
    for lvl in levels or cfg["data"]["levels"]:
        kinds += [f"lr:{lvl}", f"sr:{lvl}", f"template:{lvl}"]
    return kinds


def train_joint_taggers(cfg, levels=None, kinds=None, seeds=None):
    tc = cfg["tagger"]
    out = {}
    js = JointStore(cfg)
    for kind in kinds or joint_tagger_kinds(cfg, levels):
        if not js.kind_ok(kind, "test") or not js.kind_ok(kind, "train"):
            print(f"=== tagger on {kind}: input not available yet - skipped")
            continue
        for seed in seeds or tc["seeds"]:
            if not tc.get("retrain", False) and joint_tagger_done(cfg, kind, seed):
                print(f"=== tagger on {kind}, seed {seed}: already trained - skipped (tagger.retrain=true to redo)")
                continue
            print(f"\n=== tagger on {kind}, seed {seed} ===")
            out[(kind, seed)] = train_joint_tagger(cfg, kind, seed)
    return out


def joint_cross_eval(cfg, level, seeds=None):
    """2x3 check, as for the other datasets: column A = one tagger trained on HR, applied unchanged to every input;
    column B = a tagger trained and tested on each input."""
    from .train import Amp, device, loader
    seeds = seeds or cfg["tagger"]["seeds"]
    tc0 = joint_tagger_settings(cfg)
    js = JointStore(cfg)
    dev, amp = device(), Amp(cfg)
    n = js.n_common("test", list(cfg["data"]["levels"]))
    a_inputs = {"HR": "hr", "LR": f"uniform:{level}", "SR": f"sr:{level}", "SR + energy constraint": f"srproj:{level}",
                "k-NN template": f"template:{level}", "Ridge": f"ridge:{level}"}
    a_inputs = {k: v for k, v in a_inputs.items() if js.kind_ok(v)}
    b_inputs = {"HR": "hr", "LR": f"lr:{level}", "SR": f"sr:{level}", "k-NN template": f"template:{level}"}
    rows = []
    for seed in seeds:
        path = io_utils.run_path(cfg, "tagger", "hr", seed)
        ck_path = os.path.join(path, "best.pt")
        if os.path.exists(ck_path):
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            tc = ck["tagger_cfg"]
            targets, raw, _ = _joint_targets(cfg, js, tc)
            if not tc["classify"]:
                targets["test"] = ((np.log(raw["test"]) - ck["target_stats"][0]) / ck["target_stats"][1]).astype(np.float32)
            for row_name, kind in a_inputs.items():
                model = LayerStackTagger(js.shapes(kind), tc["widths"], 3 if tc["classify"] else 1, tc["dropout"],
                                         n_global=len(js.names)).to(dev)
                model.load_state_dict(ck["model"])
                ds = JointTaggerDataset(js, "test", kind, targets["test"], n)
                p, t = _predict(model, loader(ds, tc["batch_size"], False, cfg), dev, amp)
                m = _task_metrics(tc, p, t, raw["test"][:len(p)], ck["target_stats"], js.particle("test")[:len(p)],
                                  n_boot=50)
                m = m[0] if isinstance(m, tuple) else m
                rows.append({"level": level, "seed": seed, "column": "A: trained on HR", "evaluated_on": row_name,
                             "input": kind, "n_test": len(p), **{k: v for k, v in m.items() if not isinstance(v, list)}})
        else:
            print(f"  [{level}] no HR tagger for seed {seed} - column A skipped")
        for row_name, kind in b_inputs.items():
            mpath = os.path.join(io_utils.run_path(cfg, "tagger", kind.replace(":", "-"), seed), "metrics.json")
            if os.path.exists(mpath):
                m = io_utils.load_json(mpath)
                rows.append({"level": level, "seed": seed, "column": "B: trained on each input",
                             "evaluated_on": row_name, "input": kind, "n_test": m["n_test"],
                             **{k: v for k, v in m["test"].items() if not isinstance(v, (list, dict))}})
    if not rows:
        print(f"  [{level}] no taggers yet")
        return None
    long = pd.DataFrame(rows)
    pfx, sdir = io_utils.prefix(cfg), io_utils.summary_dir(cfg)
    tt = io_utils.tagger_tag(cfg)
    long.to_csv(os.path.join(sdir, f"{pfx}_{tt}-2x3-runs_{level}.csv"), index=False)
    keys = ([f"auc_{p}" for p in PAIRS] if tc0["classify"] else ["mean_binned_resolution"])
    tables = {}
    for key in keys:
        if key in long:
            tables[key] = _table_2x3(long, key, list(a_inputs))
            tables[key].to_csv(os.path.join(sdir, f"{pfx}_{tt}-2x3-{key.replace('_', '-')}_{level}.csv"))
            _plot_2x3(long, key, key, list(a_inputs), level, cfg, extra=f"-{key.replace('_', '-')}")
    return tables


# ---------------------------------------------------------------- cross-class test
def cross_class_report(cfg, levels=None):
    """pi+ seen by an SR model that was trained on e+ and gamma only (this experiment, data.train_particles = [0, 1])
    against the SR model trained on all three (experiment joint.cross_ref_tag). Per level: the mean of f_0, f_2,
    Eratio and the width in HR pi+, HR e+, SR of the reference, SR of this experiment, and the W1 distance to HR pi+."""
    ref_tag = cfg["joint"]["cross_ref_tag"]
    if not ref_tag:
        print("cross_class: set joint.cross_ref_tag to the run_tag of the experiment trained on all particles")
        return None
    ref_cfg = apply_overrides(cfg, {"run_tag": ref_tag})
    rows = []
    for level in levels or list(cfg["data"]["levels"]):
        paths = {"this": io_utils.run_path(cfg, "sreval", level), "ref": io_utils.run_path(ref_cfg, "sreval", level)}
        tabs = {}
        for k, p in paths.items():
            f = os.path.join(p, "observables_by_class.csv")
            if not os.path.exists(f):
                print(f"  [{level}] {f} missing")
                tabs = None
                break
            t = pd.read_csv(f)
            tabs[k] = t[(t["method"] == "sr") & t["e_lo"].isna() & (t["level"] == level)]
        if not tabs:
            continue
        for obs in ("f_0", "f_2", "eratio_l0", "eta_width_l1", "depth_mm", "e_vis_ratio"):
            def one(k, particle):
                g = tabs[k][(tabs[k]["particle"] == particle) & (tabs[k]["observable"] == obs)]
                return g.iloc[0] if len(g) else None
            pi_this, pi_ref, e_ref = one("this", "piplus"), one("ref", "piplus"), one("ref", "eplus")
            if pi_this is None or pi_ref is None:
                continue
            rows.append({"level": level, "observable": obs,
                         "mean_HR_piplus": pi_ref["mean_ref"], "mean_HR_eplus": None if e_ref is None else e_ref["mean_ref"],
                         "mean_SR_trained_on_all": pi_ref["mean_pred"], "mean_SR_trained_on_e_gamma": pi_this["mean_pred"],
                         "w1_over_sigma_trained_on_all": pi_ref["w1_over_sigma"],
                         "w1_over_sigma_trained_on_e_gamma": pi_this["w1_over_sigma"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(io_utils.summary_dir(cfg), f"{io_utils.prefix(cfg)}_cross-class-piplus.csv"), index=False)
    return df


# ---------------------------------------------------------------- summary
def summarize_joint(cfg):
    pfx, sdir = io_utils.prefix(cfg), io_utils.summary_dir(cfg)
    levels = list(cfg["data"]["levels"])
    tt, title = io_utils.tagger_tag(cfg), io_utils.tagger_title(cfg)
    sections = []
    rows = []
    import glob
    for mpath in sorted(glob.glob(os.path.join(io_utils.run_glob(cfg, "tagger"), "metrics.json"))):
        m = io_utils.load_json(mpath)
        method, _, level = m["kind"].partition(":")
        rows.append({"run": os.path.basename(os.path.dirname(mpath)), "input": m["kind"], "method": method,
                     "level": level or "native", "seed": m["seed"], "n_train": m["n_train"],
                     **{k: v for k, v in m["test"].items() if not isinstance(v, (list, dict))}})
    if rows:
        tag = pd.DataFrame(rows)
        tag.to_csv(os.path.join(sdir, f"{pfx}_{tt}-runs.csv"), index=False)
        num = [c for c in tag.columns if tag[c].dtype.kind in "fi" and c not in ("seed", "n_train")]
        agg = tag.groupby(["method", "level"])[num].agg(["mean", "std"])
        agg.columns = [f"{a} ({'mean' if b == 'mean' else 'spread over seeds'})" for a, b in agg.columns]
        agg.insert(0, "n seeds", tag.groupby(["method", "level"])["seed"].nunique())
        agg = agg.reset_index()
        agg.to_csv(os.path.join(sdir, f"{pfx}_{tt}-summary.csv"), index=False)
        sections.append((f"Downstream {title}. PID: AUC of the pairs (e+ vs gamma is the hard one), one-vs-rest macro AUC "
                         "and background rejection at fixed signal efficiency; 'boot_err' resamples the test set, "
                         "'spread over seeds' needs two or more seeds. Reading guide: SR cannot hold more information "
                         "than LR (data processing), so SR close to LR is the expected outcome.", agg))
    for level in levels:
        for path in sorted(glob.glob(os.path.join(sdir, f"{pfx}_{tt}-2x3-*_{level}.csv"))):
            if "-2x3-runs_" in path:
                continue
            key = os.path.basename(path)[len(pfx) + len(tt) + 6:-len(level) - 5]
            sections.append((f"{level}: {title} {key}, A = trained on HR only and applied to each input, "
                             "B = trained and tested on each input", pd.read_csv(path, index_col=0)))
    for level in levels:
        rdir = io_utils.run_path(cfg, "sreval", level)
        obs_p = os.path.join(rdir, "observables.csv")
        if os.path.exists(obs_p):
            obs = pd.read_csv(obs_p)
            piv = obs.pivot_table(index="observable", columns="method", values="w1_over_sigma")
            piv = piv[[c for c in ORDER if c in piv.columns] + [c for c in piv.columns if c not in ORDER]]
            piv.to_csv(os.path.join(sdir, f"{pfx}_shower-w1_{level}.csv"))
            sections.append((f"{level}: shower observables, W1 / sigma_HR (0 = identical to HR), reassembled showers", piv))
        bc_p = os.path.join(rdir, "observables_by_class.csv")
        if os.path.exists(bc_p):
            bc = pd.read_csv(bc_p)
            allE = bc[bc["e_lo"].isna() & bc["method"].isin(["sr", "template", "ridge"])]
            piv = allE.pivot_table(index=["particle", "observable"], columns="method", values="w1_over_sigma")
            piv.to_csv(os.path.join(sdir, f"{pfx}_shower-w1-by-particle_{level}.csv"))
            sections.append((f"{level}: W1 / sigma_HR per particle (all energies). The k-NN template and ridge columns "
                             "are the bar a generative model has to clear on fluctuations and tails", piv))
        cl = os.path.join(rdir, "metrics.json")
        if os.path.exists(cl):
            c = pd.DataFrame(io_utils.load_json(cl).get("closure", {})).T
            c.to_csv(os.path.join(sdir, f"{pfx}_closure_{level}.csv"))
            sections.append((f"{level}: LR closure per layer and energy closure of the reassembled shower vs HR", c))
        cr = os.path.join(rdir, "correlations.csv")
        if os.path.exists(cr):
            cc = pd.read_csv(cr)
            sections.append((f"{level}: distance between HR and method correlation matrices of the shower observables",
                             cc[cc["particle"] == "all"].set_index("method")[["frobenius_distance"]]))
    cross = cross_class_report(cfg, levels) if cfg["joint"].get("cross_ref_tag") else None
    if cross is not None:
        sections.append(("pi+ seen by SR trained on e+ and gamma only, against SR trained on all three particles", cross))
    from .evaluate import _write_report
    _write_report(cfg, sections, sdir)
    return dict(sections)
