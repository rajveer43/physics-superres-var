"""Non-generative reference points for super-resolution (calogan layers), written next to the SR outputs.

With a beam fixed at the detector centre the showers of one particle type are nearly identical in shape, so a
"mean shape" already looks like a good SR. A generative model only earns its keep where it beats these on the
event-by-event fluctuations and the tails:

    template   the HR images of the k nearest training events in LR space (log-energy distance), averaged, then
               rescaled block-wise so every coarse cell keeps its measured energy. Uses the LR image only, no
               particle label and no E_inc, and is trained on the mixture of particles.
    ridge      linear regression LR -> HR in log-energy space (the cheap deterministic regressor: it returns
               the conditional mean, so it blurs fluctuations), zero-suppressed at the readout threshold, then
               block-wise rescaled to the measured coarse energies.

Both have exact LR closure, like `srproj`. Files: {split}_base-{template|ridge}_{level}.npy
"""
import os
import time

import numpy as np
import torch

from .data import CacheStore, hit_threshold, normalize
from .physics_ops import project_to_lr


def _features(store, split, level, idx):
    """Normalised log-energy of the LR image, flattened (n, F)."""
    lr = store.array(split, f"lr:{level}")
    vol = store.vol(f"lr:{level}")
    x = normalize(np.asarray(lr[idx], np.float32), store.scale, vol, batch=True)
    return x.reshape(len(x), -1)


def _chunks(n, size):
    for s in range(0, n, size):
        yield slice(s, min(n, s + size))


def _train_events(store):
    return np.arange(store.meta["n"]["train"]) if store._train_idx is None else store._train_idx


def fit_ridge(store, level, alpha, max_train=200000):
    from sklearn.linear_model import Ridge
    idx = _train_events(store)[:max_train]
    hr = normalize(np.asarray(store.array("train", "hr")[idx], np.float32), store.scale, batch=True)
    model = Ridge(alpha=alpha).fit(_features(store, "train", level, idx), hr.reshape(len(idx), -1))
    return model


def ridge_predict(model, store, cfg, split, level, sl):
    n = np.arange(store.meta["n"][split])[sl]
    z = model.predict(_features(store, split, level, n)).reshape(len(n), store.C, *store.hr_shape)
    energy = store.scale.reshape(1, -1, *([1] * len(store.hr_shape))) * np.expm1(np.clip(z, 0, 30))
    energy = np.where(energy > hit_threshold(cfg), energy, 0.0).astype(np.float32)
    return energy


class KNNTemplate:
    def __init__(self, store, level, k, n_ref, seed, dev):
        rng = np.random.default_rng(seed)
        train = _train_events(store)
        self.ref = np.sort(rng.choice(train, min(n_ref, len(train)), replace=False))
        self.k, self.dev = k, dev
        self.feat = torch.from_numpy(_features(store, "train", level, self.ref)).to(dev)
        self.hr = np.asarray(store.array("train", "hr")[self.ref], np.float32)
        self.pos = np.full(store.meta["n"]["train"], -1, dtype=np.int64)
        self.pos[self.ref] = np.arange(len(self.ref))
        self.sq = (self.feat ** 2).sum(1)

    def predict(self, store, split, level, sl):
        rows = np.arange(store.meta["n"][split])[sl]
        q = torch.from_numpy(_features(store, split, level, rows)).to(self.dev)
        d = (q ** 2).sum(1, keepdim=True) - 2 * q @ self.feat.T + self.sq[None]
        if split == "train":   # an event must not be its own neighbour
            p = torch.from_numpy(self.pos[rows]).to(self.dev)
            m = p >= 0
            d[m.nonzero().squeeze(1), p[m]] = float("inf")
        nn_idx = d.topk(self.k, dim=1, largest=False).indices.cpu().numpy()
        return self.hr[nn_idx].mean(1)


def run_baselines(cfg, levels=None, overwrite=False):
    if cfg.get("family") != "calogan":
        print(f"baselines: nothing to do for {cfg['dataset']}")
        return None
    from .train import device
    store, dev = CacheStore(cfg), device()
    bc = cfg["baselines"]
    levels = levels or list(store.levels)
    for level in levels:
        f = store.levels[level]
        t0 = time.time()
        models = {}
        if "template" in bc["methods"]:
            models["template"] = KNNTemplate(store, level, bc["knn_k"], bc["knn_ref"], cfg["seed"], dev)
        if "ridge" in bc["methods"]:
            models["ridge"] = fit_ridge(store, level, bc["ridge_alpha"])
        for split in bc["splits"]:
            n = store.meta["n"][split]
            lr_all = store.array(split, f"lr:{level}")
            outs = {m: np.lib.format.open_memmap(store.baseline_path(split, m, level) + ".tmp", "w+", np.float32,
                                                 (n, store.C, *store.hr_shape))
                    for m in models if overwrite or not store.has_baseline(split, m, level)}
            for sl in _chunks(n, 1024):
                lr = np.asarray(lr_all[sl], np.float32)
                for m, out in outs.items():
                    raw = (models[m].predict(store, split, level, sl) if m == "template"
                           else ridge_predict(models[m], store, cfg, split, level, sl))
                    out[sl] = project_to_lr(raw, lr, f)
            for m, out in outs.items():
                out.flush()
                del out
                os.replace(store.baseline_path(split, m, level) + ".tmp", store.baseline_path(split, m, level))
        print(f"  baselines {level}: {list(models)} for {bc['splits']} ({time.time() - t0:.0f}s)")
    return None
