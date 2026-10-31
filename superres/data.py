"""Loading raw data, writing energy-preserving multi-resolution caches, and
torch Datasets.

Cache (local disk, numpy memmaps, linear energies, float32):
    {cache_root}/{dataset}/meta.json
    {split}_hr.npy        (N, C, *hr_shape)       zero-padded ground truth
    {split}_{level}.npy   (N, C, *hr_shape / f)   sum-pooled low resolution
    {split}_target.npy    (N,)                    qg: class label, calo: E_inc [MeV]
    {split}_{tag}_sr-{decode}_{level}.npy     VAR output (written by generate); decode = greedy | sample
    test_{tag}_vqrec_{level}.npy              tokenizer-only reconstruction (written by eval_sr)

Model outputs carry tag = {dataset}_{version}_{run_date}_{run_tag} (e.g. qg_v2_2026-10-06_multiseed), so outputs of another model or experiment
are never picked up by mistake; the HR / LR / target arrays are shared by all of them.

Network inputs use x = log1p(E / (s_c * V)), where s_c is the mean non-zero
cell energy of channel c and V the number of fine cells merged into one cell
(V = 1 at high resolution). This keeps LR and HR inputs on the same scale.
"""
import functools
import glob
import json
import math
import os

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset

from . import io_utils
from .download import find_data_files, raw_dir
from .physics_ops import project_to_lr, sum_pool, uniform_upsample

SPLITS = ("train", "val", "test")
IMG_KEYS = ("X_jets", "X", "x", "images", "image", "jet_images", "jets", "data", "inputs")
LABEL_KEYS = ("y", "Y", "label", "labels", "target", "targets")


def hit_threshold(cfg):
    """Readout threshold in data units: a cell below it counts as empty."""
    d = cfg["data"]
    if "hit_threshold_mev" in d:  # calogan layers
        return d["hit_threshold_mev"]
    return d["voxel_threshold_mev"] if cfg["dataset"] == "calo" else d["occupancy_threshold"]


def cache_dir(cfg):
    d = os.path.join(cfg["paths"]["cache_root"], cfg["dataset"])
    os.makedirs(d, exist_ok=True)
    return d


# ----------------------------------------------------------- raw jet readers
def _label_from_name(path):
    name = os.path.basename(path).lower()
    if "quark" in name:
        return 1
    if "gluon" in name:
        return 0
    return None


def _split_obj(obj):
    """Find (images, labels) inside whatever was stored in a .pt file."""
    if isinstance(obj, torch.utils.data.TensorDataset):
        obj = obj.tensors
    if isinstance(obj, dict):
        X = next((obj[k] for k in IMG_KEYS if k in obj and getattr(obj[k], "ndim", 0) >= 3), None)
        if X is None:
            cands = [v for v in obj.values() if getattr(v, "ndim", 0) >= 3]
            X = max(cands, key=lambda v: v.numel() if torch.is_tensor(v) else v.size) if cands else None
        y = next((obj[k] for k in LABEL_KEYS if k in obj), None)
        return X, y
    if isinstance(obj, (list, tuple)):
        arrays = [o for o in obj if hasattr(o, "shape")]
        if arrays and all(a.ndim == 3 for a in arrays) and len(arrays) == len(obj):
            return list(arrays), None  # list of single images
        X = max((a for a in arrays if a.ndim >= 3), key=lambda a: a.shape[0] * math.prod(a.shape[1:]), default=None)
        y = next((a for a in arrays if a.ndim <= 2 and X is not None and a.shape[0] == X.shape[0]), None)
        return X, y
    if hasattr(obj, "shape"):
        return obj, None
    return None, None


def _to_nchw(X, C):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:
        X = X[:, None]
    if X.shape[1] != C and X.shape[-1] == C:
        X = np.moveaxis(X, -1, 1)
    if X.shape[1] != C:
        raise ValueError(f"cannot interpret image block of shape {X.shape} as (N, {C}, H, W)")
    return np.ascontiguousarray(X)


class _ArraySource:
    """.pt / .npz / .npy / .h5 files holding an image array and (maybe) labels."""

    def __init__(self, path):
        self.path = path
        ext = os.path.splitext(path)[1].lower()
        self._h5 = None
        if ext in (".pt", ".pth"):
            try:
                obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            except Exception:
                obj = torch.load(path, map_location="cpu", weights_only=False)
            X, y = _split_obj(obj)
        elif ext == ".npz":
            z = np.load(path)
            X, y = _split_obj({k: z[k] for k in z.files})
        elif ext == ".npy":
            X, y = np.load(path, mmap_mode="r"), None
        else:  # h5 / hdf5
            import h5py
            self._h5 = h5py.File(path, "r")
            keys = list(self._h5.keys())
            xk = next((k for k in IMG_KEYS if k in keys), None) or max(
                (k for k in keys if self._h5[k].ndim >= 3), key=lambda k: self._h5[k].size)
            yk = next((k for k in LABEL_KEYS if k in keys), None)
            X, y = self._h5[xk], (self._h5[yk] if yk else None)
        if X is None:
            raise ValueError(f"no image array found in {path}")
        self.X = X
        if y is None:
            lab = _label_from_name(path)
            if lab is None:
                raise ValueError(f"no labels in {path} and file name has no 'quark'/'gluon'")
            self.y = None
            self.const_label = lab
        else:
            self.y = np.asarray(y.numpy() if torch.is_tensor(y) else y).reshape(len(X), -1)[:, -1]
            self.const_label = None
        self.n = len(X)

    def _take(self, idx):
        X = self.X
        if isinstance(X, list):
            return np.stack([np.asarray(X[i]) for i in idx])
        if torch.is_tensor(X):
            return X[torch.as_tensor(idx)].float().numpy()
        return np.asarray(X[idx])

    def iter_selected(self, idx, C, chunk=512):
        for s in range(0, len(idx), chunk):
            part = idx[s:s + chunk]
            X = _to_nchw(self._take(part), C)
            y = np.full(len(part), self.const_label) if self.y is None else self.y[part]
            yield X, y.astype(np.float64)


class _ParquetSource:
    """ML4Sci-style parquet (column X_jets holding nested 3x125x125 lists)."""

    def __init__(self, path):
        import pyarrow.parquet as pq
        self.path = path
        self.pf = pq.ParquetFile(path)
        self.n = self.pf.metadata.num_rows
        cols = self.pf.schema_arrow.names
        self.xcol = next(c for c in IMG_KEYS if c in cols)
        self.ycol = next((c for c in LABEL_KEYS if c in cols), None)
        self.const_label = None if self.ycol else _label_from_name(path)

    @staticmethod
    def _images(col):
        """Nested list column -> (k, *image_shape) float32 without going through Python lists."""
        import pyarrow as pa
        shape = np.asarray(col[0].as_py()).shape  # one row only, to learn the nesting
        flat = col
        while pa.types.is_list(flat.type) or pa.types.is_large_list(flat.type) \
                or pa.types.is_fixed_size_list(flat.type):
            flat = flat.flatten()
        return flat.to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(col), *shape)

    def iter_selected(self, idx, C, chunk=512):
        import pyarrow as pa
        start = 0
        cols = [self.xcol] + ([self.ycol] if self.ycol else [])
        for batch in self.pf.iter_batches(batch_size=chunk, columns=cols):
            b = batch.num_rows
            lo, hi = np.searchsorted(idx, start), np.searchsorted(idx, start + b)
            if hi > lo:
                taken = batch.take(pa.array(idx[lo:hi] - start))
                X = _to_nchw(self._images(taken.column(self.xcol)), C)
                y = (np.asarray(taken.column(self.ycol).to_pylist(), dtype=np.float64).reshape(len(X), -1)[:, -1]
                     if self.ycol else np.full(len(X), float(self.const_label)))
                yield X, y
            start += b


def _open_source(path):
    return _ParquetSource(path) if path.lower().endswith(".parquet") else _ArraySource(path)


# ----------------------------------------------------------- cache writing
def _preprocess(X, pad):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    neg = float((X < 0).mean())
    X = np.maximum(X, 0.0)
    X = np.pad(X, [(0, 0), (0, 0)] + [tuple(p) for p in pad])
    return X.astype(np.float32), neg


def _write_cache(cfg, chunks, split_id, rng, extras=None):
    """chunks yields (X_native (k,C,*native), target (k,)) in row order;
    split_id[r] in {0,1,2} assigns row r to train/val/test. extras: name -> (N_rows, ...) array stored as
    {split}_{name}.npy in the same event order (the shuffle depends only on split_id and the rng seed, so
    separate calls with the same split_id and a fresh rng of the same seed order their events identically)."""
    d = cfg["data"]
    out = cache_dir(cfg)
    C, hr = len(d["channels"]), tuple(d["hr_shape"])
    levels = {k: tuple(v) for k, v in d["levels"].items()}
    counts = {s: int((split_id == i).sum()) for i, s in enumerate(SPLITS)}
    pos = np.empty(len(split_id), dtype=np.int64)
    for i in range(len(SPLITS)):
        rows = np.where(split_id == i)[0]
        pos[rows] = rng.permutation(len(rows))  # shuffle within each split
    mm, tgt = {}, {}
    for s in SPLITS:
        mm[(s, "hr")] = open_memmap(os.path.join(out, f"{s}_hr.npy"), "w+", np.float32, (counts[s], C, *hr))
        for lvl, f in levels.items():
            shape = tuple(h // fi for h, fi in zip(hr, f))
            mm[(s, lvl)] = open_memmap(os.path.join(out, f"{s}_{lvl}.npy"), "w+", np.float32, (counts[s], C, *shape))
        tgt[s] = np.zeros(counts[s], dtype=np.float64)
    r0, negs = 0, []
    for X, t in chunks:
        k = len(X)
        X, neg = _preprocess(X, d["pad"])
        negs.append(neg)
        pooled = {lvl: sum_pool(X, f) for lvl, f in levels.items()}
        sid, p = split_id[r0:r0 + k], pos[r0:r0 + k]
        for i, s in enumerate(SPLITS):
            m = sid == i
            if m.any():
                mm[(s, "hr")][p[m]] = X[m]
                for lvl in levels:
                    mm[(s, lvl)][p[m]] = pooled[lvl][m]
                tgt[s][p[m]] = t[m]
        r0 += k
        print(f"\r  wrote {r0}/{len(split_id)}", end="")
    print()
    if r0 != len(split_id):
        raise RuntimeError(f"expected {len(split_id)} rows, got {r0}")
    for a in mm.values():
        a.flush()
    for s in SPLITS:
        np.save(os.path.join(out, f"{s}_target.npy"), tgt[s])
        for name, arr in (extras or {}).items():
            rows = np.where(split_id == SPLITS.index(s))[0]
            ordered = np.zeros((counts[s], *arr.shape[1:]), dtype=arr.dtype)
            ordered[pos[rows]] = arr[rows]
            np.save(os.path.join(out, f"{s}_{name}.npy"), ordered)
    return counts, float(np.mean(negs)) if negs else 0.0


def _finish_meta(cfg, counts, extra):
    d = cfg["data"]
    out = cache_dir(cfg)
    hr = np.load(os.path.join(out, "train_hr.npy"), mmap_mode="r")
    sample = np.asarray(hr[: min(d.get("scale_events", 2000), len(hr))])
    scales = []
    for c in range(sample.shape[1]):
        v = sample[:, c][sample[:, c] > 0]
        scales.append(float(v.mean()) if v.size else 1.0)
    meta = {
        "dataset": cfg["dataset"], "created": io_utils.now(), "n": counts,
        "channels": d["channels"], "native_shape": d["native_shape"], "hr_shape": d["hr_shape"],
        "pad": d["pad"], "levels": d["levels"], "scales": scales, **extra,
    }
    io_utils.save_json(os.path.join(out, "meta.json"), meta)
    io_utils.save_json(os.path.join(io_utils.dataset_root(cfg), "data_meta.json"), meta)
    return meta


def prepare_qg(cfg):
    d = cfg["data"]
    files = find_data_files(raw_dir(cfg))
    if not files:
        raise RuntimeError(f"no jet files in {raw_dir(cfg)} - run the download stage first")
    C = len(d["channels"])
    sources = [_open_source(p) for p in files]
    ns = np.array([s.n for s in sources])
    total = int(ns.sum())
    N = min(total, d["max_samples"] or total)
    rng = np.random.default_rng(cfg["seed"])
    sel = np.sort(rng.choice(total, N, replace=False))  # sample across all files (class files!)
    offsets = np.concatenate([[0], np.cumsum(ns)])
    fr = np.asarray(d["split"], dtype=float) / sum(d["split"])
    n_tr, n_va = int(fr[0] * N), int(fr[1] * N)
    split_id = np.full(N, 2)
    perm = rng.permutation(N)
    split_id[perm[:n_tr]] = 0
    split_id[perm[n_tr:n_tr + n_va]] = 1

    def chunks():
        for src, a, b in zip(sources, offsets[:-1], offsets[1:]):
            local = sel[(sel >= a) & (sel < b)] - a
            if len(local):
                print(f"  reading {len(local)} jets from {os.path.basename(src.path)}")
                yield from src.iter_selected(local, C)

    print(f"Preparing {N} of {total} jets from {len(files)} file(s)")
    counts, neg = _write_cache(cfg, chunks(), split_id, rng)
    labels = np.concatenate([np.load(os.path.join(cache_dir(cfg), f"{s}_target.npy")) for s in SPLITS])
    balance = {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    print("label counts:", balance)
    return _finish_meta(cfg, counts, {"source_files": files, "label_counts": balance,
                                      "negative_fraction_clipped": neg})


def prepare_calo(cfg):
    import h5py
    d = cfg["data"]
    f_train, f_test = (os.path.join(raw_dir(cfg), f) for f in d["files"])
    rng = np.random.default_rng(cfg["seed"])
    native = tuple(d["native_shape"])

    def read(path, n_max):
        h = h5py.File(path, "r")
        n = h["showers"].shape[0]
        idx = np.sort(rng.choice(n, min(n, n_max or n), replace=False))
        return h, idx

    h1, idx1 = read(f_train, d["max_samples"])
    h2, idx2 = read(f_test, d["test_max"])
    n_va = int(d["val_frac"] * len(idx1))
    split1 = np.zeros(len(idx1), dtype=int)
    split1[rng.permutation(len(idx1))[:n_va]] = 1
    split_id = np.concatenate([split1, np.full(len(idx2), 2)])

    def chunks(chunk=5000):
        for h, idx in ((h1, idx1), (h2, idx2)):
            for s in range(0, len(idx), chunk):
                part = idx[s:s + chunk]
                lo, hi = part[0], part[-1] + 1
                showers = h["showers"][lo:hi][part - lo]
                e_inc = h["incident_energies"][lo:hi][part - lo].reshape(-1)
                # CaloChallenge ordering: (layer, angular, radial)
                yield showers.reshape(len(part), 1, *native).astype(np.float32), e_inc.astype(np.float64)

    print(f"Preparing {len(idx1)} train/val showers + {len(idx2)} test showers")
    counts, neg = _write_cache(cfg, chunks(), split_id, rng)
    return _finish_meta(cfg, counts, {"source_files": [f_train, f_test], "energy_unit": "MeV",
                                      "negative_fraction_clipped": neg})


def calogan_split_ids(particle, e_inc_gev, fractions, n_bins, rng):
    """0/1/2 = train/val/test for every event, stratified by particle x E_inc bin (1-100 GeV)."""
    ebin = np.clip(((e_inc_gev - 1.0) / 99.0 * n_bins).astype(int), 0, n_bins - 1)
    fr = np.asarray(fractions, dtype=float) / sum(fractions)
    split_id = np.full(len(particle), 2)
    for p in np.unique(particle):
        for b in range(n_bins):
            rows = np.where((particle == p) & (ebin == b))[0]
            rows = rows[rng.permutation(len(rows))]
            n_tr, n_va = int(round(fr[0] * len(rows))), int(round(fr[1] * len(rows)))
            split_id[rows[:n_tr]] = 0
            split_id[rows[n_tr:n_tr + n_va]] = 1
    return split_id


def calogan_condition(layers):
    """Measured shower summary the layer models may be conditioned on: log1p(E_layer / MeV) and the layer
    fractions (N, 6). Sum-pooling conserves layer energies, so this is identical at every LR level."""
    e = np.stack([layers[k].reshape(len(layers[k]), -1).sum(1) for k in sorted(layers)], 1).astype(np.float64)
    frac = e / np.maximum(e.sum(1, keepdims=True), 1e-12)
    return np.concatenate([np.log1p(e), frac], 1).astype(np.float32)


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_calogan(cfg):
    """One read of eplus / gamma / piplus .hdf5 -> HR / LR caches of all three layers with identical splits
    (calogan_l0, _l1, _l2 under cache_root), plus per-event particle, overflow, condition vector and E_inc."""
    import h5py
    from .config import layer_configs
    d = cfg["data"]
    src = d["source_dir"] or os.path.join(cfg["paths"]["raw_root"], "calogan")
    files = [os.path.join(src, f"{p}.hdf5") for p in d["particles"]]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise RuntimeError(f"CaloGAN files not found: {missing} - run the download stage or set data.source_dir")
    cfgs = layer_configs(cfg)
    names = sorted(cfgs)                      # calogan_l0, _l1, _l2
    rng = np.random.default_rng(cfg["seed"])
    layers = {n: [] for n in names}
    particle, energy, overflow = [], [], []
    for label, path in enumerate(files):
        with h5py.File(path, "r") as h:
            n = h["energy"].shape[0]
            cap = d["max_per_particle"] or n
            idx = np.sort(rng.choice(n, min(n, cap), replace=False))
            for i, name in enumerate(names):
                layers[name].append(np.asarray(h[f"layer_{i}"][:], dtype=np.float32)[idx])
            energy.append(np.asarray(h["energy"][:], dtype=np.float64).reshape(-1)[idx])
            overflow.append(np.asarray(h["overflow"][:], dtype=np.float32)[idx])
            particle.append(np.full(len(idx), label, dtype=np.int8))
    layers = {k: np.concatenate(v) for k, v in layers.items()}
    particle, energy, overflow = np.concatenate(particle), np.concatenate(energy), np.concatenate(overflow)
    split_id = calogan_split_ids(particle, energy, d["split"], d["n_energy_bins"], rng)
    extras = {"particle": particle, "overflow": overflow, "cond": calogan_condition(layers)}
    sources = [{"path": f, "bytes": os.path.getsize(f), "sha256": _sha256(f)} for f in files]
    thr = d["threshold_scan_mev"]
    metas = {}
    for name in names:
        lc = cfgs[name]
        X = layers[name][:, None]

        def chunks(X=X, chunk=5000):
            for s in range(0, len(X), chunk):
                yield X[s:s + chunk], energy[s:s + chunk] * 1000.0     # E_inc in MeV, like the layer cells

        counts, neg = _write_cache(lc, chunks(), split_id, np.random.default_rng(cfg["seed"]), extras)
        scan = {d["particles"][p]: {str(t): float((X[particle == p] > t).mean()) for t in thr}
                for p in range(len(d["particles"]))}
        prof = X[particle == 0][:5000, 0].mean(0)
        metas[name] = _finish_meta(lc, counts, {
            "source_files": sources, "energy_unit": "MeV", "layer": name, "particles": d["particles"],
            "axes": ["phi", "eta"], "negative_fraction_clipped": neg, "occupancy_scan": scan,
            "beam_centre_bins": [[int(i) for i in np.argsort(prof.sum(1 - a))[-2:]] for a in (0, 1)]})
        print(f"  {name}: {counts}")
    return metas[names[0]]


def prepare(cfg, force=False):
    meta_path = os.path.join(cache_dir(cfg), "meta.json")
    if cfg.get("family") == "calogan":   # one read of the HDF5 files writes the caches of all three layers
        from .config import layer_configs
        metas = [os.path.join(cache_dir(c), "meta.json") for c in layer_configs(cfg).values()]
        if all(os.path.exists(m) for m in metas) and not force:
            print(f"cache exists: {metas}")
            return io_utils.load_json(metas[0])
        return prepare_calogan(cfg)
    if os.path.exists(meta_path) and not force:
        print(f"cache exists: {meta_path}")
        return io_utils.load_json(meta_path)
    return prepare_calo(cfg) if cfg["dataset"] == "calo" else prepare_qg(cfg)


# ------------------------------------------------------------- cache access
def normalize(E, scale, vol=1.0, batch=False):
    shape = (1, -1) + (1,) * (E.ndim - 2) if batch else (-1,) + (1,) * (E.ndim - 1)
    return np.log1p(E / (scale.reshape(shape) * vol)).astype(np.float32)


class _Lazy:
    """Array-like that computes rows on access from other arrays."""

    def __init__(self, fn, *arrays):
        self.fn, self.arrays = fn, arrays

    def __len__(self):
        return min(len(a) for a in self.arrays)

    def __getitem__(self, i):
        single = isinstance(i, (int, np.integer))
        parts = [np.asarray(a[[i]] if single else a[i], dtype=np.float32) for a in self.arrays]
        out = self.fn(*parts)
        return out[0] if single else out


def _same(a):
    return a


def parse_kind(kind):
    method, _, level = kind.partition(":")
    return method, (level or None)


# names used by runs made before the raw output became the primary result
LEGACY_KINDS = {"var": "srproj", "varraw": "sr"}


class CacheStore:
    """Access to cached arrays. kind is one of
    'hr'                 ground truth
    'lr:<level>'         sum-pooled image on its own coarse grid
    'sr:<level>'         model output exactly as generated (the primary result)
    'srsample:<level>' / 'srgreedy:<level>'   the two decoding modes explicitly
    'srproj:<level>'     model output rescaled so each coarse cell matches the measurement (diagnostic)
    'uniform:<level>'    coarse energy spread evenly over the fine pixels (diagnostic)
    'vqrec:<level>'      HR -> tokens -> HR by the tokenizer alone, decoded with that level's LR (diagnostic)
    'template:<level>' / 'ridge:<level>'   no-generative baselines (calogan): k-NN template and ridge regression
                         from the LR image, both with exact LR closure (written by the baselines stage)"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = cache_dir(cfg)
        path = os.path.join(self.dir, "meta.json")
        if not os.path.exists(path):
            raise RuntimeError("no cache found - run the prepare stage first")
        self.meta = io_utils.load_json(path)
        self.scale = np.asarray(self.meta["scales"], dtype=np.float32)
        self.levels = {k: tuple(v) for k, v in self.meta["levels"].items()}
        self.hr_shape = tuple(self.meta["hr_shape"])
        self.C = len(self.meta["channels"])
        self.tag = io_utils.model_tag(cfg)
        self._train_idx = None
        keep = cfg["data"].get("train_particles")
        if keep is not None and os.path.exists(self.path("train", "particle")):
            # SR trained on some particles only (cross-class test): the energy scale uses those events too
            sel = np.where(np.isin(self._mm("train", "particle"), keep))[0]
            self._train_idx = sel
            hr = self._mm("train", "hr")
            sample = np.asarray(hr[sel[: cfg["data"].get("scale_events", 2000)]])
            self.scale = np.asarray([float(sample[:, c][sample[:, c] > 0].mean()) for c in range(sample.shape[1])],
                                    dtype=np.float32)

    def n_train(self):
        return self.meta["n"]["train"] if self._train_idx is None else len(self._train_idx)

    def particle(self, split):
        return np.load(self.path(split, "particle"))

    def overflow(self, split):
        return np.load(self.path(split, "overflow"))

    def baseline_path(self, split, name, level):
        return self.path(split, f"base-{name}_{level}")

    def has_baseline(self, split, name, level):
        return os.path.exists(self.baseline_path(split, name, level))

    def path(self, split, what):
        return os.path.join(self.dir, f"{split}_{what}.npy")

    def _mm(self, split, what):
        return np.load(self.path(split, what), mmap_mode="r")

    def sr_path(self, split, level, decode="greedy"):
        """decode: 'greedy' (most likely token) or 'sample' (tokens drawn from the predicted distribution)."""
        return self.path(split, f"{self.tag}_sr-{decode}_{level}")

    def has_sr(self, split, level, decode="greedy"):
        return os.path.exists(self.sr_path(split, level, decode))

    def vqrec_path(self, split, level):
        return self.path(split, f"{self.tag}_vqrec_{level}")

    def n_common(self, split, levels):
        """Events that HR and every generated SR output of `levels` have, so that taggers on
        different inputs are trained and scored on the same events (SR may be capped by
        var.gen_max_per_split)."""
        n = self.meta["n"][split]
        for lvl in levels:
            if self.has_sr(split, lvl, self.sr_decode("sr")):
                n = min(n, len(np.load(self.sr_path(split, lvl, self.sr_decode("sr")), mmap_mode="r")))
        return n

    def sr_decode(self, method):
        """Which stored SR file a method reads. 'sr' and 'srproj' follow cfg var.decode."""
        primary = self.cfg["var"].get("decode", "greedy")
        return {"sr": primary, "srproj": primary, "srsample": "sample", "srgreedy": "greedy"}[method]

    def target(self, split):
        return np.load(self.path(split, "target"))

    def factor(self, kind):
        method, level = parse_kind(kind)
        return self.levels[level] if method == "lr" else (1,) * len(self.hr_shape)

    def vol(self, kind):
        return math.prod(self.factor(kind))

    def shape(self, kind):
        return tuple(h // f for h, f in zip(self.hr_shape, self.factor(kind)))

    def array(self, split, kind):
        method, level = parse_kind(kind)
        method = LEGACY_KINDS.get(method, method)
        if method == "hr":
            return self._mm(split, "hr")
        if method == "vqrec":  # tokenizer-only reconstruction of HR (written by evaluate)
            return _Lazy(_same, np.load(self.vqrec_path(split, level), mmap_mode="r"))
        f = self.levels[level]
        lr = self._mm(split, level)
        if method == "lr":
            return lr
        if method == "uniform":
            return _Lazy(functools.partial(uniform_upsample, f=f), lr)
        if method in ("template", "ridge"):
            return _Lazy(_same, np.load(self.baseline_path(split, method, level), mmap_mode="r"))
        if method not in ("sr", "srproj", "srsample", "srgreedy"):
            raise ValueError(f"unknown kind {kind}")
        sr = np.load(self.sr_path(split, level, self.sr_decode(method)), mmap_mode="r")
        if method == "srproj":  # SR with every coarse cell forced back to its measured energy
            return _Lazy(functools.partial(project_to_lr, f=f), sr, lr)
        return _Lazy(_same, sr)  # the model output as generated


class SRDataset(Dataset):
    """Normalised HR (and LR) pairs for VQ-VAE / VAR training.
    level: one level name -> item["lr"]; a list of names -> item["lr:<level>"] for each.
    calogan: item["g"] is the measured shower summary (layer energies and fractions) the VAR is conditioned on."""

    def __init__(self, store, split, level=None, max_n=None, all_events=False):
        """all_events: ignore data.train_particles (generation must cover every event of the split)."""
        self.store, self.split = store, split
        self.single = isinstance(level, str)
        self.levels = [level] if self.single else list(level or [])
        filtered = split == "train" and not all_events and store._train_idx is not None
        total = store.n_train() if filtered else store.meta["n"][split]
        self.n = total if max_n is None else min(max_n, total)
        self.idx = store._train_idx if filtered else None
        self._hr, self._lr, self._g = None, {}, None
        self.use_g = os.path.exists(store.path(split, "cond")) and store.cfg.get("var", {}).get("cond_dim", 0) > 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._hr is None:
            self._hr = self.store.array(self.split, "hr")
            self._lr = {lvl: self.store.array(self.split, f"lr:{lvl}") for lvl in self.levels}
            if self.use_g:
                self._g = self.store._mm(self.split, "cond")
        j = int(self.idx[i]) if self.idx is not None else i
        out = {"hr": torch.from_numpy(normalize(np.asarray(self._hr[j], np.float32), self.store.scale))}
        for lvl, arr in self._lr.items():
            vol = self.store.vol(f"lr:{lvl}")
            x = torch.from_numpy(normalize(np.asarray(arr[j], np.float32), self.store.scale, vol))
            out["lr" if self.single else f"lr:{lvl}"] = x
        if self.use_g:
            out["g"] = torch.from_numpy(np.array(self._g[j], np.float32))
        return out


def target_transform(cfg, t, stats=None):
    """qg: label as float; calo: standardised log(E_inc)."""
    if cfg["tagger"]["task"] == "classification":
        return (t == cfg["data"].get("signal_label", 1)).astype(np.float32), None
    lt = np.log(t)
    stats = stats or (float(lt.mean()), float(lt.std()))
    return ((lt - stats[0]) / stats[1]).astype(np.float32), stats


class TaggerDataset(Dataset):
    """(normalised image, global log-energy per channel, target)."""

    def __init__(self, store, arr, targets, kind, max_n=None):
        self.store, self.arr, self.t = store, arr, targets
        self.vol = store.vol(kind)
        self.n = min(len(arr), len(targets)) if max_n is None else min(max_n, len(arr), len(targets))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        E = np.asarray(self.arr[i], dtype=np.float32)
        x = normalize(E, self.store.scale, self.vol)
        g = np.log1p(E.reshape(E.shape[0], -1).sum(1) / self.store.scale).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(g), torch.tensor(self.t[i], dtype=torch.float32)
