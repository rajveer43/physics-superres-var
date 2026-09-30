"""SR evaluation against the ground truth and cross-run summaries.

For each LR level, on the test set, four representations are compared with HR:
    lr       - the coarse measurement itself (on its own grid)
    uniform  - energy spread evenly inside each coarse cell (no-learning baseline)
    var-raw  - VAR output as generated
    var      - VAR output projected to reproduce every coarse cell energy exactly
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from . import io_utils  # noqa: E402
from . import metrics as M  # noqa: E402
from .data import CacheStore  # noqa: E402
from .observables import cell_spectrum, jet_observables, shower_observables  # noqa: E402
from .physics_ops import sum_pool  # noqa: E402

METHODS = ("lr", "uniform", "var-raw", "var")
KIND = {"lr": "lr", "uniform": "uniform", "var-raw": "varraw", "var": "var"}
STYLE = {"hr": dict(color="0.3", label="HR (truth)"), "lr": dict(color="tab:red", label="LR"),
         "uniform": dict(color="tab:orange", label="uniform upsample"),
         "var-raw": dict(color="tab:cyan", label="VAR (raw)"), "var": dict(color="tab:blue", label="VAR (projected)")}
RELATIVE = {"qg": ("sum_", "jet_pt", "jet_mass"), "calo": ("e_total", "e_ratio")}


def _is_relative(ds, name):
    return any(name.startswith(p) for p in RELATIVE[ds])


def _observables(cfg, store, E, kind, e_inc):
    f = store.factor(kind)
    if cfg["dataset"] == "qg":
        return jet_observables(E, f, cfg["data"], cfg["eval"].get("nsub_max", 0))
    return shower_observables(E, f, cfg["data"], e_inc)


def _threshold(cfg):
    d = cfg["data"]
    return d["voxel_threshold_mev"] if cfg["dataset"] == "calo" else d["occupancy_threshold"]


def c2st_auc(cfg, store, ref, other, seed):
    """Classifier two-sample test: AUC of a CNN separating HR from SR (0.5 = indistinguishable)."""
    from .train import Amp, device, fit_classifier, predict
    n = min(len(ref), len(other), cfg["eval"]["c2st_max"])
    X = np.concatenate([np.asarray(ref[:n], np.float32), np.asarray(other[:n], np.float32)])
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.float32)
    perm = np.random.default_rng(seed).permutation(2 * n)
    a, b = int(0.5 * len(perm)), int(0.75 * len(perm))
    parts = {"train": perm[:a], "val": perm[a:b], "test": perm[b:]}
    dev, amp = device(), Amp(cfg)
    tcfg = dict(cfg["tagger"], patience=2)
    model, dls = fit_classifier(cfg, store, "hr", {k: X[v] for k, v in parts.items()},
                                {k: y[v] for k, v in parts.items()}, cfg["eval"]["c2st_epochs"], seed, tcfg, dev, amp)
    p, t = predict(model, dls["test"], dev, amp)
    return M.classification_metrics(t, p, n_boot=50)


# ----------------------------------------------------------------- plotting
def _hist_with_ratio(ref, others, name, bins, fig_dir, prefix):
    lo, hi = np.percentile(ref, [0.5, 99.5])
    if lo == hi:
        lo, hi = ref.min(), ref.max() + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(5, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    h_ref, _ = np.histogram(ref, edges, density=True)
    ax.stairs(h_ref, edges, fill=True, alpha=0.3, **STYLE["hr"])
    for m, v in others.items():
        h, _ = np.histogram(v, edges, density=True)
        ax.stairs(h, edges, lw=1.4, **STYLE[m])
        with np.errstate(divide="ignore", invalid="ignore"):
            rx.stairs(np.where(h_ref > 0, h / h_ref, np.nan), edges, color=STYLE[m]["color"], lw=1.2)
    rx.axhline(1, color="0.3", lw=0.8)
    rx.set_ylim(0.5, 1.5)
    rx.set_ylabel("ratio to HR")
    rx.set_xlabel(name)
    ax.set_ylabel("normalised")
    ax.legend(fontsize=7, frameon=False)
    io_utils.save_figure(fig, fig_dir, f"{prefix}__hist__{name}")
    plt.close(fig)


def _profiles(ref_p, other_p, fig_dir, prefix):
    for pname, (prof, centres, xlabel) in ref_p.items():
        fig, (ax, rx) = plt.subplots(2, 1, figsize=(5, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        mean_ref = prof.mean(0)
        ax.plot(centres, mean_ref, "o-", ms=3, **STYLE["hr"])
        for m, p in other_p.items():
            prof_m, cen_m, _ = p[pname]
            mean_m = prof_m.mean(0)
            ax.plot(cen_m, mean_m, "-", lw=1.3, **STYLE[m])
            if len(cen_m) == len(centres):
                with np.errstate(divide="ignore", invalid="ignore"):
                    rx.plot(centres, mean_m / mean_ref, color=STYLE[m]["color"])
        ax.set_yscale("log")
        ax.set_ylabel("mean energy fraction")
        ax.legend(fontsize=7, frameon=False)
        rx.axhline(1, color="0.3", lw=0.8)
        rx.set_ylim(0.5, 1.5)
        rx.set_xlabel(xlabel)
        io_utils.save_figure(fig, fig_dir, f"{prefix}__profile__{pname}")
        plt.close(fig)


def _examples(cfg, images, fig_dir, prefix, n):
    """images: dict method -> (N, C, *grid) linear energies. Sums channels (qg) or angle (calo)."""
    def view(E):
        return E.sum(0) if cfg["dataset"] == "qg" else E[0].sum(1).T  # calo: radial x layer
    names = list(images)
    fig, axes = plt.subplots(n, len(names), figsize=(2.3 * len(names), 2.3 * n), squeeze=False)
    for i in range(n):
        vmax = max(view(images["hr"][i]).max(), 1e-6)
        for j, m in enumerate(names):
            ax = axes[i, j]
            img = view(images[m][i])
            ax.imshow(np.clip(img, vmax * 1e-4, None), norm=LogNorm(vmax * 1e-4, vmax), origin="lower",
                      aspect="auto", cmap="viridis", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(STYLE[m]["label"], fontsize=8)
    io_utils.save_figure(fig, fig_dir, f"{prefix}__examples")
    plt.close(fig)


# --------------------------------------------------------------- evaluation
def evaluate_level(cfg, level, with_c2st=True):
    store = CacheStore(cfg)
    ds = cfg["dataset"]
    ev = cfg["eval"]
    run = io_utils.RunDir(cfg, "sreval", level)
    prefix = f"{ds}__{level}"
    n = min(ev["max_events"], store.meta["n"]["test"])
    e_inc = store.target("test")[:n] if ds == "calo" else None
    hr = np.asarray(store.array("test", "hr")[:n], np.float32)
    ref_s, ref_p = _observables(cfg, store, hr, "hr", e_inc)
    thr = _threshold(cfg)
    ref_spec = cell_spectrum(hr, thr)
    lr_true = np.asarray(store.array("test", f"lr:{level}")[:n], np.float32)
    f = store.levels[level]

    rows, consistency, obs, profs, imgs = [], {}, {}, {}, {"hr": hr[:ev["n_examples"]]}
    for m in METHODS:
        if m.startswith("var") and not store.has_sr("test", level):
            print(f"  [{level}] no VAR output yet - skipping {m}")
            continue
        kind = f"{KIND[m]}:{level}"
        E = np.asarray(store.array("test", kind)[:n], np.float32)
        imgs[m] = E[:ev["n_examples"]]
        s, p = _observables(cfg, store, E, kind, e_inc)
        obs[m], profs[m] = s, p
        for name in ref_s:
            if name in s:
                rows.append({"dataset": ds, "level": level, "method": m, "observable": name,
                             **M.compare_distributions(ref_s[name], s[name], _is_relative(ds, name))})
        if m != "lr":  # SR lives on the HR grid: cell spectrum and closure w.r.t. the measurement
            spec = cell_spectrum(E, thr)
            rows.append({"dataset": ds, "level": level, "method": m, "observable": "log10_cell_energy",
                         "w1_over_sigma": float(M.wasserstein_distance(ref_spec, spec) / (ref_spec.std() + 1e-12)),
                         "ks": float(M.ks_2samp(ref_spec, spec).statistic)})
            closure = np.abs(sum_pool(E, f) - lr_true).reshape(n, -1).sum(1) / np.maximum(
                lr_true.reshape(n, -1).sum(1), 1e-12)
            consistency[m] = {"median_lr_closure": float(np.median(closure)),
                              "mean_lr_closure": float(np.mean(closure))}
    df = pd.DataFrame(rows)
    df.to_csv(run.file("observables.csv"), index=False)

    c2st = {}
    if with_c2st:
        for m in ("uniform", "var-raw", "var"):
            if m in obs:
                print(f"  [{level}] classifier two-sample test HR vs {m}")
                E = store.array("test", f"{KIND[m]}:{level}")
                c2st[m] = c2st_auc(cfg, store, hr, E, cfg["seed"])
    run.save_config({"level": level, "n_events": n})
    run.save_metrics({"level": level, "n_events": n, "lr_closure": consistency, "c2st": c2st})

    bins = ev["hist_bins"]
    for name in ref_s:
        _hist_with_ratio(ref_s[name], {m: o[name] for m, o in obs.items() if name in o}, name, bins,
                         run.fig_dir, prefix)
    _profiles(ref_p, profs, run.fig_dir, prefix)
    _examples(cfg, {k: v for k, v in imgs.items() if k != "var-raw"}, run.fig_dir, prefix,
              min(ev["n_examples"], n))
    print(f"[{run.name}] saved {len(df)} rows, figures in {run.fig_dir}")
    return df


# ---------------------------------------------------------------- summaries
def _load_metrics(cfg, stage):
    root = os.path.join(io_utils.dataset_root(cfg), "runs")
    out = []
    for path in sorted(glob.glob(os.path.join(root, f"{cfg['dataset']}__{stage}__*", "metrics.json"))):
        out.append((os.path.basename(os.path.dirname(path)), io_utils.load_json(path)))
    return out


def _split_variant(kind):
    method, _, level = kind.partition(":")
    return method, level or "native"


def summarize(cfg):
    ds = cfg["dataset"]
    sdir, fdir = io_utils.summary_dir(cfg), io_utils.summary_dir(cfg, "figures")
    levels = list(cfg["data"]["levels"])
    tables = {}

    # taggers
    rows = []
    for name, m in _load_metrics(cfg, "tagger"):
        method, level = _split_variant(m["kind"])
        row = {"run": name, "input": m["kind"], "method": method, "level": level, "seed": m["seed"],
               "n_train": m["n_train"]}
        row.update({k: v for k, v in m["test"].items() if not isinstance(v, list)})
        rows.append(row)
    if rows:
        tag = pd.DataFrame(rows)
        tag.to_csv(os.path.join(sdir, f"{ds}__tagger_runs.csv"), index=False)
        key = "auc" if ds == "qg" else "mean_binned_resolution"
        num = [c for c in tag.columns if tag[c].dtype.kind in "fi" and c not in ("seed", "n_train")]
        agg = tag.groupby(["input", "method", "level"])[num].agg(["mean", "std"]).reset_index()
        agg.columns = ["_".join(c).strip("_") for c in agg.columns]
        agg.to_csv(os.path.join(sdir, f"{ds}__tagger_summary.csv"), index=False)
        tables["tagger"] = agg
        _plot_tagger(tag, key, levels, fdir, ds)
        if ds == "calo":
            _plot_calo_resolution(cfg, fdir)

    # SR observables
    frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(
        io_utils.dataset_root(cfg), "runs", f"{ds}__sreval__*", "observables.csv")))]
    if frames:
        obs = pd.concat(frames, ignore_index=True)
        obs.to_csv(os.path.join(sdir, f"{ds}__sr_observables.csv"), index=False)
        piv = obs.pivot_table(index="observable", columns=["level", "method"], values="w1_over_sigma")
        piv.to_csv(os.path.join(sdir, f"{ds}__sr_w1_table.csv"))
        tables["w1"] = piv
        _plot_w1(obs, levels, fdir, ds)

    # C2ST + closure
    rows = []
    for name, m in _load_metrics(cfg, "sreval"):
        for meth, r in m.get("c2st", {}).items():
            rows.append({"level": m["level"], "method": meth, "c2st_auc": r["auc"], "c2st_auc_std": r.get("auc_std")})
        for meth, r in m.get("lr_closure", {}).items():
            rows.append({"level": m["level"], "method": meth, **r})
    if rows:
        c = pd.DataFrame(rows).groupby(["level", "method"]).first().reset_index()
        c.to_csv(os.path.join(sdir, f"{ds}__c2st_closure.csv"), index=False)
        tables["c2st"] = c
    _write_report(cfg, tables, sdir)
    return tables


def _plot_tagger(tag, key, levels, fdir, ds):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    hr = tag[tag["method"] == "hr"][key]
    if len(hr):
        ax.axhspan(hr.mean() - hr.std(ddof=0), hr.mean() + hr.std(ddof=0), color="0.8")
        ax.axhline(hr.mean(), color="0.3", label="HR (truth)")
    x = np.arange(len(levels))
    for m in ("lr", "uniform", "var"):
        sub = tag[tag["method"] == m].groupby("level")[key]
        mean = [sub.mean().get(l, np.nan) for l in levels]
        std = [sub.std(ddof=0).get(l, np.nan) for l in levels]
        ax.errorbar(x, mean, yerr=std, marker="o", capsize=3, **STYLE[m])
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_xlabel("down-sampling level")
    ax.set_ylabel("tagger ROC AUC" if ds == "qg" else "energy resolution (mean over bins)")
    ax.legend(fontsize=8, frameon=False)
    io_utils.save_figure(fig, fdir, f"{ds}__tagger_{key}_vs_level")
    plt.close(fig)


def _plot_calo_resolution(cfg, fdir):
    fig, axes = plt.subplots(1, len(cfg["data"]["levels"]), figsize=(4.5 * len(cfg["data"]["levels"]), 3.6),
                             squeeze=False)
    runs = dict(_load_metrics(cfg, "tagger"))
    hr = [m for m in runs.values() if m["kind"] == "hr"]
    for ax, level in zip(axes[0], cfg["data"]["levels"]):
        curves = ([("hr", hr[0])] if hr else []) + [
            (meth, m) for m in runs.values() for meth in ("lr", "uniform", "var") if m["kind"] == f"{meth}:{level}"]
        seen = set()
        for meth, m in curves:
            if meth in seen:
                continue
            seen.add(meth)
            b = m["test"]["binned"]
            e = [np.sqrt(x["e_lo"] * x["e_hi"]) / 1e3 for x in b]
            ax.plot(e, [x["resolution"] for x in b], "o-", ms=3, **STYLE[meth])
        ax.set_xscale("log")
        ax.set_title(level, fontsize=9)
        ax.set_xlabel("E_inc [GeV]")
        ax.set_ylabel("sigma_eff(E_pred/E_inc)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    io_utils.save_figure(fig, fdir, "calo__energy_resolution_vs_E")
    plt.close(fig)


def _plot_w1(obs, levels, fdir, ds):
    sub = obs[obs["method"] != "var-raw"]
    for level in levels:
        s = sub[sub["level"] == level].pivot_table(index="observable", columns="method", values="w1_over_sigma")
        if s.empty:
            continue
        s = s[[m for m in ("lr", "uniform", "var") if m in s.columns]]
        fig, ax = plt.subplots(figsize=(0.9 * len(s.columns) + 3, 0.35 * len(s) + 1.2))
        im = ax.imshow(s.values, cmap="magma_r", aspect="auto")
        ax.set_xticks(range(len(s.columns)))
        ax.set_xticklabels(s.columns)
        ax.set_yticks(range(len(s.index)))
        ax.set_yticklabels(s.index, fontsize=8)
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                ax.text(j, i, f"{s.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k")
        fig.colorbar(im, ax=ax, label="W1 / sigma_HR (lower is better)")
        ax.set_title(f"{ds} {level}", fontsize=9)
        io_utils.save_figure(fig, fdir, f"{ds}__w1_heatmap__{level}")
        plt.close(fig)


def _md(df):
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


def _write_report(cfg, tables, sdir):
    ds = cfg["dataset"]
    lines = [f"# {ds} super-resolution summary", "", f"generated {io_utils.now()}", ""]
    names = {"tagger": "Downstream tagger / regressor (test set, mean and std over seeds)",
             "w1": "Observable agreement with HR: W1 distance / sigma_HR (0 = identical)",
             "c2st": "Classifier two-sample test AUC (0.5 = indistinguishable from HR) and LR closure"}
    for k, t in tables.items():
        lines += [f"## {names[k]}", "", _md(t.round(4)), ""]
    with open(os.path.join(sdir, f"{ds}__report.md"), "w") as f:
        f.write("\n".join(lines))
