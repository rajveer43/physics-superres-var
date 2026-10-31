"""Checks of the CaloGAN pieces on synthetic showers (seconds, no real data), plus checks on the real HDF5 files
when they are present (skipped otherwise).

    python -m pytest tests/test_calogan.py
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from superres.config import CALOGAN_LAYERS, get_config, layer_configs  # noqa: E402
from superres.data import calogan_condition, calogan_split_ids  # noqa: E402
from superres.observables import (calogan_shower_observables, layer_observables,  # noqa: E402
                                  strip_eratio)
from superres.physics_ops import sum_pool  # noqa: E402

REAL = os.path.join(ROOT, "dataset")


def _layer_cfg(name):
    return get_config(name)["data"]


# ------------------------------------------------------------------ geometry
def test_pooling_levels_divide_the_grid_and_keep_the_beam_symmetry():
    """Every LR level must divide its layer, and keep the beam axis (grid centre) on a cell edge or cell centre,
    otherwise the shower core falls inside one coarse cell and the level is not comparable with the others."""
    for name, spec in CALOGAN_LAYERS.items():
        for level, f in spec["levels"].items():
            for n, fi in zip(spec["shape"], f):
                assert n % fi == 0, (name, level, f)
                # the centre of the axis is a cell edge (even n) or a cell centre (odd n): pooling must not move it
                if n % 2 == 0:
                    assert (n // 2) % fi == 0, f"{name} {level}: factor {fi} straddles the beam on an axis of {n}"
                else:
                    assert fi in (1, n), f"{name} {level}: odd axis {n} pooled by {fi}"


def test_layer_configs_share_experiment_and_overrides():
    cfg = get_config("calogan", {"run_tag": "t", "vqvae.epochs": 3})
    for c in layer_configs(cfg).values():
        assert c["run_tag"] == "t" and c["vqvae"]["epochs"] == 3 and c["family"] == "calogan"


def test_tokenizer_scales_fit_the_latent_grid():
    for name, spec in CALOGAN_LAYERS.items():
        shape = list(spec["shape"])
        for st in spec["strides"]:
            shape = [(s - 1) // x + 1 for s, x in zip(shape, st)]
        assert shape == list(spec["scales"][-1]), (name, shape)
        assert len(spec["widths"]) == len(spec["strides"]) + 1


# ------------------------------------------------------------------ data
def test_sum_pool_conserves_energy_to_1e6():
    rng = np.random.default_rng(0)
    for name, spec in CALOGAN_LAYERS.items():
        E = rng.exponential(5.0, (40, 1, *spec["shape"])) * (rng.random((40, 1, *spec["shape"])) < 0.4)
        E = E.astype(np.float32)
        for f in spec["levels"].values():
            pooled = sum_pool(E, f)
            ref = E.astype(np.float64).sum((1, 2, 3))
            assert np.allclose(pooled.astype(np.float64).sum((1, 2, 3)), ref, rtol=1e-6)


def test_split_is_disjoint_stratified_and_complete():
    rng = np.random.default_rng(1)
    n = 3000
    particle = np.repeat([0, 1, 2], n // 3)
    e = rng.uniform(1, 100, n)
    sid = calogan_split_ids(particle, e, [0.8, 0.1, 0.1], 10, np.random.default_rng(0))
    assert set(np.unique(sid)) == {0, 1, 2} and len(sid) == n
    for p in range(3):   # every particle is in every split with the right share
        frac = [(sid[particle == p] == s).mean() for s in range(3)]
        assert frac == pytest.approx([0.8, 0.1, 0.1], abs=0.02)
    ebin = np.clip(((e - 1) / 99 * 10).astype(int), 0, 9)
    for b in range(10):   # and so is every energy bin
        assert (sid[ebin == b] == 0).mean() == pytest.approx(0.8, abs=0.06)
    again = calogan_split_ids(particle, e, [0.8, 0.1, 0.1], 10, np.random.default_rng(0))
    assert np.array_equal(sid, again)


def test_write_cache_keeps_labels_aligned_across_layers(tmp_path):
    """The three layer caches must hold the same events in the same order: particle, E_inc and condition vector are
    written per layer with a shuffle that depends only on split_id and the seed."""
    from superres.data import SPLITS, _write_cache
    rng = np.random.default_rng(0)
    n = 90
    particle = np.repeat([0, 1, 2], n // 3).astype(np.int8)
    e = rng.uniform(1, 100, n)
    sid = calogan_split_ids(particle, e, [0.8, 0.1, 0.1], 10, np.random.default_rng(0))
    layers = {k: rng.exponential(1.0, (n, 1, *CALOGAN_LAYERS[k]["shape"])).astype(np.float32) for k in sorted(CALOGAN_LAYERS)}
    # tag every event by putting its row number in the first cell of layer 0 (so it can be found again)
    layers["calogan_l0"][:, 0, 0, 0] = np.arange(n) + 1000.0
    extras = {"particle": particle, "cond": calogan_condition({k: v[:, 0] for k, v in layers.items()})}
    out = {}
    for name in sorted(layers):
        cfg = get_config(name, {"paths.cache_root": str(tmp_path)})
        _write_cache(cfg, iter([(layers[name], e * 1000.0)]), sid, np.random.default_rng(0), extras)
        out[name] = {s: dict(hr=np.load(tmp_path / name / f"{s}_hr.npy"), part=np.load(tmp_path / name / f"{s}_particle.npy"),
                             tgt=np.load(tmp_path / name / f"{s}_target.npy"),
                             cond=np.load(tmp_path / name / f"{s}_cond.npy")) for s in SPLITS}
    for s in SPLITS:
        for name in ("calogan_l1", "calogan_l2"):
            assert np.array_equal(out[name][s]["part"], out["calogan_l0"][s]["part"])
            assert np.array_equal(out[name][s]["tgt"], out["calogan_l0"][s]["tgt"])
            assert np.array_equal(out[name][s]["cond"], out["calogan_l0"][s]["cond"])
        row = np.rint(out["calogan_l0"][s]["hr"][:, 0, 0, 0]).astype(int) - 1000   # original row of each cached event
        assert np.array_equal(out["calogan_l0"][s]["part"], particle[row])
        assert np.allclose(out["calogan_l0"][s]["tgt"], e[row] * 1000.0)
        # condition vector = layer energies of that very event (also in the other layers)
        assert np.allclose(out["calogan_l1"][s]["cond"][:, 1], np.log1p(layers["calogan_l1"][row].sum((1, 2, 3))), rtol=1e-4)


def test_condition_vector_is_level_independent():
    rng = np.random.default_rng(0)
    layers = {k: rng.exponential(2.0, (10, *CALOGAN_LAYERS[k]["shape"])).astype(np.float32) for k in sorted(CALOGAN_LAYERS)}
    c = calogan_condition(layers)
    assert c.shape == (10, 6) and np.allclose(c[:, 3:].sum(1), 1.0, atol=1e-5)
    pooled = {k: sum_pool(v[:, None], CALOGAN_LAYERS[k]["levels"]["lr4"])[:, 0] for k, v in layers.items()}
    assert np.allclose(calogan_condition(pooled), c, rtol=1e-5)


# ------------------------------------------------------------------ observables (analytic cases)
def _strips(values):
    E = np.zeros((1, 1, 3, 96), np.float32)
    for i, v in values.items():
        E[0, 0, 1, i] = v
    return E


def test_eratio_single_peak_is_one_and_two_equal_peaks_is_zero():
    d = _layer_cfg("calogan_l0")
    assert layer_observables(_strips({47: 10.0}), (1, 1), d, np.array([1e4]))[0]["eratio"][0] == pytest.approx(1.0)
    # two equal peaks far apart (a pi0 -> gamma gamma like pattern)
    assert layer_observables(_strips({30: 5.0, 60: 5.0}), (1, 1), d, np.array([1e4]))[0]["eratio"][0] == pytest.approx(0.0)
    # unequal: (E1 - E2)/(E1 + E2) = (8 - 2) / (8 + 2)
    assert layer_observables(_strips({30: 8.0, 60: 2.0}), (1, 1), d, np.array([1e4]))[0]["eratio"][0] == pytest.approx(0.6)
    # adjacent strips are one peak, not two
    assert layer_observables(_strips({47: 5.0, 48: 5.0}), (1, 1), d, np.array([1e4]))[0]["eratio"][0] == pytest.approx(1.0)


def test_strip_observables_are_nan_on_coarse_eta_grids():
    d = _layer_cfg("calogan_l0")
    E = sum_pool(_strips({30: 8.0, 60: 2.0}), (1, 4))
    o = layer_observables(E, (1, 4), d, np.array([1e4]))[0]
    assert np.isnan(o["eratio"][0]) and np.isnan(o["max_strip_frac"][0])


def test_single_cell_shower_moments_and_core_fractions():
    d = _layer_cfg("calogan_l1")   # 12 x 12 cells of 40 mm, beam on the edge between cells 5 and 6
    E = np.zeros((1, 1, 12, 12), np.float32)
    E[0, 0, 6, 8] = 100.0          # cell centre: phi = +20 mm, eta = +100 mm
    o = layer_observables(E, (1, 1), d, np.array([1e5]))[0]
    assert o["e_layer"][0] == pytest.approx(100.0) and o["e_ratio"][0] == pytest.approx(1e-3)
    assert o["eta_centroid"][0] == pytest.approx(100.0) and o["phi_centroid"][0] == pytest.approx(20.0)
    assert o["eta_width"][0] == pytest.approx(0.0, abs=1e-6)
    assert o["n_hits"][0] == 1
    assert o["core2d_80mm"][0] == pytest.approx(0.0) and o["core_eta_160mm"][0] == pytest.approx(1.0)
    E[0, 0, 5:7, 5:7] = 25.0       # the four central cells
    E[0, 0, 6, 8] = 0.0
    o = layer_observables(E, (1, 1), d, np.array([1e5]))[0]
    assert o["core2d_40mm"][0] == pytest.approx(1.0) and o["eta_centroid"][0] == pytest.approx(0.0, abs=1e-6)
    assert o["eta_width"][0] == pytest.approx(20.0)
    # on the 2x2 pooled grid the +-40 mm window is not a cell edge any more: NaN, not a biased number
    o2 = layer_observables(sum_pool(E, (2, 2)), (2, 2), d, np.array([1e5]))[0]
    assert np.isnan(o2["core2d_40mm"][0]) and o2["core2d_80mm"][0] == pytest.approx(1.0)


def test_shower_observables_fractions_and_depth():
    rng = np.random.default_rng(0)
    cfgs = layer_configs(get_config("calogan"))
    dcfgs = {k: c["data"] for k, c in cfgs.items()}
    layers = {k: (rng.exponential(3, (6, 1, *CALOGAN_LAYERS[k]["shape"]))).astype(np.float32) for k in dcfgs}
    ones = {k: (1, 1) for k in dcfgs}
    o, _ = calogan_shower_observables(layers, ones, dcfgs, np.full(6, 1e5))
    assert np.allclose(o["f_0"] + o["f_1"] + o["f_2"], 1.0, atol=1e-6)
    assert np.all((o["depth_mm"] > 45) & (o["depth_mm"] < 458.5))
    assert np.allclose(o["e_vis_ratio"] * 1e5, o["e_vis"])


# ------------------------------------------------------------------ download
def _fake_download_cfg(tmp_path, payload):
    import hashlib
    files = {p: {"id": p, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
             for p in ("eplus", "gamma", "piplus")}
    return get_config("calogan", {"paths.raw_root": str(tmp_path), "data.files": files,
                                  "data.url": "http://example.invalid/{id}"})


def test_download_fetches_missing_files_and_verifies_sha256(tmp_path, monkeypatch):
    from superres import download
    payload = b"HDF fake payload" * 10
    calls = []

    def fake_fetch(url, dest, expected_size=None):
        calls.append(url)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(payload)

    monkeypatch.setattr(download, "fetch", fake_fetch)
    cfg = _fake_download_cfg(tmp_path, payload)
    found = download.download(cfg)
    assert sorted(found) == ["eplus", "gamma", "piplus"] and len(calls) == 3
    assert calls[0] == "http://example.invalid/eplus"
    download.download(cfg)          # complete files are not fetched again
    assert len(calls) == 3


def test_download_deletes_a_corrupt_file(tmp_path, monkeypatch):
    from superres import download

    def bad_fetch(url, dest, expected_size=None):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"X" * 160)     # right size, wrong content

    monkeypatch.setattr(download, "fetch", bad_fetch)
    cfg = _fake_download_cfg(tmp_path, b"HDF fake payload" * 10)
    with pytest.raises(RuntimeError, match="sha256"):
        download.download(cfg)
    assert not os.path.exists(os.path.join(str(tmp_path), "calogan", "eplus.hdf5"))


def test_download_with_source_dir_never_fetches(tmp_path, monkeypatch):
    from superres import download
    monkeypatch.setattr(download, "fetch", lambda *a, **k: pytest.fail("must not download"))
    cfg = get_config("calogan", {"data.source_dir": str(tmp_path)})
    with pytest.raises(RuntimeError, match="not found"):
        download.download(cfg)


def test_configured_hashes_match_the_local_files():
    import hashlib
    if not os.path.exists(os.path.join(REAL, "eplus.hdf5")):
        pytest.skip("dataset/ not present")
    cfg = get_config("calogan")
    for particle, info in cfg["data"]["files"].items():
        path = os.path.join(REAL, f"{particle}.hdf5")
        assert os.path.getsize(path) == info["bytes"]
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        assert h.hexdigest() == info["sha256"]


# ------------------------------------------------------------------ no leakage
def test_no_truth_in_model_inputs(tmp_path):
    """What the SR model and the taggers are fed (HR / LR images and the condition vector g) is a function of the
    cell energies alone: overwriting E_inc, overflow and particle label on disk changes none of it, and g equals the
    summary recomputed from the three HR images of the same event."""
    import torch  # noqa: F401
    from superres.data import CacheStore, SRDataset, prepare
    from superres.joint import JointStore, JointTaggerDataset
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from smoke_test import fake_calogan
    raw = tmp_path / "raw" / "calogan"
    fake_calogan(str(raw), n=150)
    paths = {"paths.raw_root": str(tmp_path / "raw"), "paths.cache_root": str(tmp_path / "cache"),
             "paths.drive_root": str(tmp_path / "exp")}
    cfg = get_config("calogan", paths, smoke=True)
    prepare(cfg)
    js = JointStore(cfg)
    stores = js.stores

    def items():
        sr = {k: [SRDataset(st, "train", "lr2")[i] for i in range(5)] for k, st in stores.items()}
        tg = JointTaggerDataset(js, "train", "lr:lr2", np.zeros(js.n("train"), np.int64), 5)
        return sr, [tg[i] for i in range(5)]

    sr0, tg0 = items()
    for sr in sr0.values():
        assert set(sr[0]) == {"hr", "lr", "g"}
    hr = {k: np.asarray(st.array("train", "hr")[:5], np.float32)[:, 0] for k, st in stores.items()}
    assert np.allclose(sr0["calogan_l0"][0]["g"].numpy(), calogan_condition({k: v[:1] for k, v in hr.items()})[0], rtol=1e-5)
    for st in stores.values():   # scramble every truth file
        for split in ("train", "val", "test"):
            for what in ("target", "overflow", "particle"):
                arr = np.load(st.path(split, what))
                np.save(st.path(split, what), np.random.default_rng(0).permutation(arr))
    sr1, tg1 = items()
    for k in sr0:
        for a, b in zip(sr0[k], sr1[k]):
            assert all(torch.equal(a[key], b[key]) for key in a)
    for (xa, ga, _), (xb, gb, _) in zip(tg0, tg1):
        assert torch.equal(ga, gb) and all(torch.equal(x, y) for x, y in zip(xa, xb))


# ------------------------------------------------------------------ real files
real = pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "eplus.hdf5")), reason="dataset/ not present")


@real
def test_real_files_geometry_and_centering():
    h5py = pytest.importorskip("h5py")
    with h5py.File(os.path.join(REAL, "eplus.hdf5"), "r") as f:
        shapes = {k: f[k].shape[1:] for k in ("layer_0", "layer_1", "layer_2", "overflow", "energy")}
        assert shapes == {"layer_0": (3, 96), "layer_1": (12, 12), "layer_2": (12, 6), "overflow": (3,), "energy": (1,)}
        e = f["energy"][:5000, 0]
        assert 1.0 <= e.min() and e.max() <= 100.0
        for k, spec in zip(("layer_0", "layer_1", "layer_2"), CALOGAN_LAYERS.values()):
            prof = f[k][:3000].mean(0)
            assert (f[k][:100] >= 0).all()
            for ax, n in enumerate(spec["shape"]):
                p = prof.sum(1 - ax)
                top2 = sorted(np.argsort(p)[-2:])
                if n % 2 == 0:   # the beam hits the middle edge: the two central cells carry the shower
                    assert top2 == [n // 2 - 1, n // 2], (k, ax, top2)
                else:            # odd axis (3 phi strips of layer 0): the middle strip
                    assert np.argmax(p) == n // 2


@real
def test_real_files_reflection_symmetry():
    """The flip augmentation of the taggers needs the showers to be statistically mirror-symmetric."""
    h5py = pytest.importorskip("h5py")
    for particle in ("eplus", "piplus"):
        with h5py.File(os.path.join(REAL, f"{particle}.hdf5"), "r") as f:
            for k in ("layer_0", "layer_1", "layer_2"):
                X = f[k][:20000].astype(np.float64).mean(0)
                for ax in (0, 1):
                    p = X.sum(1 - ax)
                    assert np.abs(p - p[::-1]).sum() / p.sum() < 0.05, (particle, k, ax)
