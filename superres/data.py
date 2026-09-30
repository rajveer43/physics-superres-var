"""Loading raw data, writing energy-preserving multi-resolution caches, and
torch Datasets.

Cache (local disk, numpy memmaps, linear energies, float32):
    {cache_root}/{dataset}/meta.json
    {split}_hr.npy        (N, C, *hr_shape)       zero-padded ground truth
    {split}_{level}.npy   (N, C, *hr_shape / f)   sum-pooled low resolution
    {split}_target.npy    (N,)                    qg: class label, calo: E_inc [MeV]
    {split}_sr-var_{level}.npy                    VAR output (written by generate)

Network inputs use x = log1p(E / (s_c * V)), where s_c is the mean non-zero
cell energy of channel c and V the number of fine cells merged into one cell
(V = 1 at high resolution). This keeps LR and HR inputs on the same scale.
"""
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


def _write_cache(cfg, chunks, split_id, rng):
    """chunks yields (X_native (k,C,*native), target (k,)) in row order;
    split_id[r] in {0,1,2} assigns row r to train/val/test."""
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
    return counts, float(np.mean(negs)) if negs else 0.0


def _finish_meta(cfg, counts, extra):
    d = cfg["data"]
    out = cache_dir(cfg)
    hr = np.load(os.path.join(out, "train_hr.npy"), mmap_mode="r")
    sample = np.asarray(hr[: min(2000, len(hr))])
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


def prepare(cfg, force=False):
    meta_path = os.path.join(cache_dir(cfg), "meta.json")
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


def parse_kind(kind):
    method, _, level = kind.partition(":")
    return method, (level or None)


class CacheStore:
    """Access to cached arrays. kind is one of
    'hr' | 'lr:<level>' | 'uniform:<level>' | 'var:<level>' | 'varraw:<level>'."""

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

    def path(self, split, what):
        return os.path.join(self.dir, f"{split}_{what}.npy")

    def _mm(self, split, what):
        return np.load(self.path(split, what), mmap_mode="r")

    def sr_path(self, split, level):
        return self.path(split, f"sr-var_{level}")

    def has_sr(self, split, level):
        return os.path.exists(self.sr_path(split, level))

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
        if method == "hr":
            return self._mm(split, "hr")
        f = self.levels[level]
        lr = self._mm(split, level)
        if method == "lr":
            return lr
        if method == "uniform":
            return _Lazy(lambda a: uniform_upsample(a, f), lr)
        sr = np.load(self.sr_path(split, level), mmap_mode="r")
        if method == "varraw":
            return _Lazy(lambda a: a, sr)
        if method == "var":
            return _Lazy(lambda a, b: project_to_lr(a, b, f), sr, lr)
        raise ValueError(f"unknown kind {kind}")


class SRDataset(Dataset):
    """Normalised HR (and LR) pairs for VQ-VAE / VAR training."""

    def __init__(self, store, split, level=None, max_n=None):
        self.store, self.split, self.level = store, split, level
        self.n = store.meta["n"][split] if max_n is None else min(max_n, store.meta["n"][split])
        self._hr = self._lr = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._hr is None:
            self._hr = self.store.array(self.split, "hr")
            if self.level:
                self._lr = self.store.array(self.split, f"lr:{self.level}")
        out = {"hr": torch.from_numpy(normalize(np.asarray(self._hr[i], np.float32), self.store.scale))}
        if self.level:
            vol = self.store.vol(f"lr:{self.level}")
            out["lr"] = torch.from_numpy(normalize(np.asarray(self._lr[i], np.float32), self.store.scale, vol))
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
