"""SR evaluation against the ground truth and cross-run summaries.

Main comparison (figures/ and the report): HR truth, LR, and the super-resolved
image exactly as the model generates it ("sr"). Nothing is corrected afterwards.

Diagnostics (figures/diagnostics/ and extra table rows), to understand a result:
    vqrec       HR -> tokens -> HR by the tokenizer alone (decoded with the same LR image);
                the ceiling for any token model
    sr-greedy / sr-sample   the decoding mode that is not the primary one
    uniform     coarse energy spread evenly over the fine pixels (no-learning reference)
    srproj      sr rescaled so every coarse cell matches the measured energy
"""
import glob
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from . import io_utils  # noqa: E402
from . import metrics as M  # noqa: E402
from .data import LEGACY_KINDS, CacheStore, hit_threshold  # noqa: E402
from .observables import cell_spectrum, jet_observables, shower_observables  # noqa: E402
from .physics_ops import sum_pool  # noqa: E402

KIND = {"lr": "lr", "sr": "sr", "sr-sample": "srsample", "sr-greedy": "srgreedy",
        "uniform": "uniform", "srproj": "srproj"}
STYLE = {
    "hr": dict(color="0.3", label="HR (truth)"),
    "lr": dict(color="tab:red", label="LR (input)"),
    "sr": dict(color="tab:blue", label="Super-resolved (VAR)"),
    "vqrec": dict(color="tab:green", label="Tokenizer only (HR→tokens→HR, with LR)"),
    "sr-greedy": dict(color="tab:purple", label="SR, most-likely tokens"),
    "sr-sample": dict(color="tab:purple", label="SR, sampled tokens"),
    "uniform": dict(color="tab:orange", label="Uniform upsample"),
    "srproj": dict(color="tab:cyan", label="SR + energy constraint"),
}
ORDER = ["hr", "lr", "sr", "vqrec", "sr-greedy", "sr-sample", "uniform", "srproj"]
RELATIVE = {"qg": ("sum_", "jet_pt", "jet_mass"), "calo": ("e_total", "e_ratio")}


def _style(m):
    return STYLE.get(m, dict(color="tab:brown", label=m))


def _is_relative(ds, name):
    return any(name.startswith(p) for p in RELATIVE[ds])


def _observables(cfg, store, E, factor, e_inc):
    if cfg["dataset"] == "qg":
        return jet_observables(E, factor, cfg["data"], cfg["eval"].get("nsub_max", 0))
    return shower_observables(E, factor, cfg["data"], e_inc)


def _resolve_methods(cfg, store, level):
    """Which methods can be evaluated now -> (main, diagnostics)."""
    primary = cfg["var"].get("decode", "greedy")

    def available(m):
        if m in ("lr", "uniform", "vqrec"):
            return True
        decode = store.sr_decode(KIND[m])
        if m in ("sr-sample", "sr-greedy") and decode == primary:
            return False  # identical to "sr"
        return store.has_sr("test", level, decode)

    ev = cfg["eval"]
    main = [m for m in ev.get("methods", ["lr", "sr"]) if available(m)]
    diag = [m for m in ev.get("diagnostics", []) if m not in main and available(m)]
    for m in ev.get("methods", []):
        if m not in main:
            print(f"  [{level}] '{m}' has not been generated yet - skipped")
    return main, diag


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
        h, _ = np.histogram(np.clip(v, lo, hi), edges, density=True)  # out-of-range values go to the edge bins
        ax.stairs(h, edges, lw=1.4, **_style(m))
        with np.errstate(divide="ignore", invalid="ignore"):
            rx.stairs(np.where(h_ref > 0, h / h_ref, np.nan), edges, color=_style(m)["color"], lw=1.2)
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
            ax.plot(cen_m, mean_m, "-", lw=1.3, **_style(m))
            if len(cen_m) == len(centres):
                with np.errstate(divide="ignore", invalid="ignore"):
                    rx.plot(centres, mean_m / mean_ref, color=_style(m)["color"])
        ax.set_yscale("log")
        ax.set_ylabel("mean energy fraction")
        ax.legend(fontsize=7, frameon=False)
        rx.axhline(1, color="0.3", lw=0.8)
        rx.set_ylim(0.5, 1.5)
        rx.set_ylabel("ratio to HR")
        rx.set_xlabel(xlabel)
        io_utils.save_figure(fig, fig_dir, f"{prefix}__profile__{pname}")
        plt.close(fig)


def _examples(cfg, images, lr_factor, fig_dir, prefix, n):
    """images: dict method -> (n, C, *grid) linear energies. Every panel in a row uses the same
    colour scale. LR is drawn as energy per fine-pixel area, otherwise its merged cells would
    look brighter than the truth for no physical reason."""
    qg = cfg["dataset"] == "qg"

    def view(E, m):
        img = E.sum(0) if qg else E[0].sum(1).T  # qg: sum channels; calo: sum over angle -> radius x layer
        if m == "lr":
            img = img / (math.prod(lr_factor) if qg else lr_factor[0] * lr_factor[2])
        return img

    names = [m for m in ORDER if m in images]
    fig, axes = plt.subplots(n, len(names), figsize=(2.4 * len(names) + 0.8, 2.4 * n), squeeze=False)
    for i in range(n):
        vmax = max(view(images["hr"][i], "hr").max(), 1e-6)
        norm = LogNorm(vmax * 1e-4, vmax)
        for j, m in enumerate(names):
            ax = axes[i, j]
            im = ax.imshow(np.clip(view(images[m][i], m), vmax * 1e-4, None), norm=norm, origin="lower",
                           aspect="auto", cmap="viridis", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(_style(m)["label"], fontsize=8)
        fig.colorbar(im, ax=list(axes[i]), fraction=0.025, pad=0.01).set_label("energy per pixel", fontsize=7)
    fig.suptitle("One event per row; all panels in a row share the colour scale (LR shown per fine-pixel area)",
                 fontsize=9)
    io_utils.save_figure(fig, fig_dir, f"{prefix}__examples")
    plt.close(fig)


def _channel_panels(cfg, rows, lr_factor, level, fig_dir, prefix, n):
    """One figure per event: rows = HR / LR / SR (and any other method passed), columns = channels.
    A column shares one colour scale (each channel has its own energy range). LR is shown on the
    fine grid as energy per fine pixel, i.e. each coarse cell's energy spread over the pixels it covers."""
    from .physics_ops import uniform_upsample
    channels = cfg["data"]["channels"]
    log = cfg["eval"].get("channel_scale", "linear") == "log"
    names = [m for m in ORDER if m in rows]
    titles = {"hr": "HR (ground truth)", "lr": "LR (input, per fine pixel)", "sr": "SR (VAR output)",
              "vqrec": "Tokenizer only", "sr-sample": "SR, sampled tokens", "sr-greedy": "SR, most-likely tokens"}
    for i in range(n):
        fig, axes = plt.subplots(len(names), len(channels), figsize=(3.3 * len(channels) + 0.6, 3.3 * len(names)),
                                 squeeze=False)
        for c, ch in enumerate(channels):
            vmax = max(float(rows["hr"][i, c].max()), 1e-6)
            norm = LogNorm(vmax * 1e-3, vmax) if log else plt.Normalize(0, vmax)
            for r, m in enumerate(names):
                img = rows[m][i:i + 1]
                if m == "lr":
                    img = uniform_upsample(img, lr_factor)
                img = img[0, c]
                ax = axes[r, c]
                im = ax.imshow(np.clip(img, vmax * 1e-3, None) if log else img, norm=norm, cmap="inferno",
                               origin="lower", interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(ch.upper() if ch != "tracks" else "Tracks", fontsize=13, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(titles.get(m, _style(m)["label"]), fontsize=11)
            fig.colorbar(im, ax=list(axes[:, c]), fraction=0.04, pad=0.02, location="bottom").set_label(
                f"{ch} energy per pixel", fontsize=8)
        f = "×".join(str(v) for v in lr_factor)
        fig.suptitle(f"Per-channel jet image, {level} ({f} pixels merged into one, then super-resolved)\n"
                     f"test event {i}; each column shares one colour scale ({'log' if log else 'linear'})",
                     fontsize=11)
        io_utils.save_figure(fig, fig_dir, f"{prefix}__channels__event{i}")
        plt.close(fig)


# --------------------------------------------------------------- evaluation
def evaluate_level(cfg, level, with_c2st=True):
    store = CacheStore(cfg)
    ds, ev = cfg["dataset"], cfg["eval"]
    run = io_utils.RunDir(cfg, "sreval", level)
    prefix = f"{io_utils.prefix(cfg)}__{level}"
    f = store.levels[level]
    ones = (1,) * len(store.hr_shape)
    main, diag = _resolve_methods(cfg, store, level)

    n = min(ev["max_events"], store.meta["n"]["test"])
    for m in main + diag:  # SR may have been generated for fewer events
        if m in KIND and KIND[m].startswith("sr"):
            n = min(n, len(np.load(store.sr_path("test", level, store.sr_decode(KIND[m])), mmap_mode="r")))
    e_inc = store.target("test")[:n] if ds == "calo" else None
    hr = np.asarray(store.array("test", "hr")[:n], np.float32)
    lr_true = np.asarray(store.array("test", f"lr:{level}")[:n], np.float32)
    ref_s, ref_p = _observables(cfg, store, hr, ones, e_inc)
    thr = hit_threshold(cfg)
    ref_spec = cell_spectrum(hr, thr)

    def load(m):
        if m == "vqrec":
            path = store.vqrec_path("test", level)
            if os.path.exists(path) and len(np.load(path, mmap_mode="r")) >= n:
                return np.asarray(np.load(path, mmap_mode="r")[:n], np.float32)
            from .train import reconstruct_with_tokenizer
            print(f"  [{level}] running the tokenizer alone on {n} HR events")
            E = reconstruct_with_tokenizer(cfg, store, hr, lr_true, level)
            np.save(path, E.astype(np.float16 if cfg["var"].get("sr_dtype") == "float16" else np.float32))
            return E
        return np.asarray(store.array("test", f"{KIND[m]}:{level}")[:n], np.float32)

    rows, consistency, obs, profs = [], {}, {}, {}
    n_ex = min(ev["n_examples"], n)
    imgs, diag_imgs = {"hr": hr[:n_ex]}, {}
    for m in main + diag:
        E = load(m)
        if m in main:
            imgs[m] = E[:n_ex]
        elif m in ("vqrec", "sr-sample", "sr-greedy"):
            diag_imgs[m] = E[:n_ex]
        s, p = _observables(cfg, store, E, f if m == "lr" else ones, e_inc)
        obs[m], profs[m] = s, p
        role = "main" if m in main else "diagnostic"
        for name in ref_s:
            if name in s:
                rows.append({"dataset": ds, "level": level, "method": m, "role": role, "observable": name,
                             **M.compare_distributions(ref_s[name], s[name], _is_relative(ds, name))})
        if m != "lr":  # on the HR grid: cell spectrum and agreement with the coarse measurement
            spec = cell_spectrum(E, thr)
            rows.append({"dataset": ds, "level": level, "method": m, "role": role, "observable": "log10_cell_energy",
                         "w1_over_sigma": float(M.wasserstein_distance(ref_spec, spec) / (ref_spec.std() + 1e-12)),
                         "ks": float(M.ks_2samp(ref_spec, spec).statistic)})
            closure = np.abs(sum_pool(E, f) - lr_true).reshape(n, -1).sum(1) / np.maximum(
                lr_true.reshape(n, -1).sum(1), 1e-12)
            consistency[m] = {"median_lr_closure": float(np.median(closure)),
                              "mean_lr_closure": float(np.mean(closure))}
        del E
    df = pd.DataFrame(rows)
    df.to_csv(run.file("observables.csv"), index=False)

    c2st = {}
    if with_c2st:
        for m in ev.get("c2st_methods", ["sr"]):
            if m in obs:
                print(f"  [{level}] classifier two-sample test: HR vs {m}")
                c2st[m] = c2st_auc(cfg, store, hr, load(m), cfg["seed"])
    run.save_config({"level": level, "n_events": n, "main_methods": main, "diagnostic_methods": diag})
    run.save_metrics({"level": level, "n_events": n, "lr_closure": consistency, "c2st": c2st})

    bins = ev["hist_bins"]
    diag_dir = os.path.join(run.fig_dir, "diagnostics")
    for name in ref_s:
        _hist_with_ratio(ref_s[name], {m: obs[m][name] for m in main if name in obs[m]}, name, bins,
                         run.fig_dir, prefix)
        if diag:
            _hist_with_ratio(ref_s[name], {m: obs[m][name] for m in main + diag if name in obs[m]}, name, bins,
                             diag_dir, prefix)
    _profiles(ref_p, {m: profs[m] for m in main}, run.fig_dir, prefix)
    if diag:
        _profiles(ref_p, profs, diag_dir, prefix)
    imgs["lr"] = lr_true[:n_ex]
    _examples(cfg, imgs, f, run.fig_dir, prefix, n_ex)
    if diag_imgs:
        _examples(cfg, {**imgs, **diag_imgs}, f, diag_dir, prefix, n_ex)
    if ds == "qg":
        _channel_panels(cfg, imgs, f, level, run.fig_dir, prefix, n_ex)
        if diag_imgs:  # same events with the tokenizer-only and other-decoding rows added
            _channel_panels(cfg, {**imgs, **diag_imgs}, f, level, diag_dir, prefix, n_ex)
    print(f"[{run.name}] {n} events; main: {main}; diagnostics: {diag}; figures in {run.fig_dir}")
    return df


# ---------------------------------------------------------------- summaries
def _load_metrics(cfg, stage):
    root = os.path.join(io_utils.dataset_root(cfg), "runs")
    out = []
    for path in sorted(glob.glob(os.path.join(root, f"{io_utils.prefix(cfg)}__{stage}__*", "metrics.json"))):
        out.append((os.path.basename(os.path.dirname(path)), io_utils.load_json(path)))
    return out


def _split_variant(kind):
    method, _, level = kind.partition(":")
    return LEGACY_KINDS.get(method, method), level or "native"


def _sorted_methods(methods):
    return sorted(methods, key=lambda m: ORDER.index(m) if m in ORDER else len(ORDER))


def values_table(obs, level):
    """Mean of every quantity: HR truth | LR | SR | diagnostics, plus SR's distance to the truth."""
    sub = obs[(obs["level"] == level) & obs["mean_pred"].notna()]
    if sub.empty:
        return None
    t = sub.pivot_table(index="observable", columns="method", values="mean_pred")
    t = t[_sorted_methods(t.columns)]
    t.insert(0, "hr", sub.groupby("observable")["mean_ref"].first())
    t.columns = [_style(c)["label"] for c in t.columns]
    if (sub["method"] == "sr").any():
        t["SR distance to truth (W1/σ)"] = sub[sub["method"] == "sr"].set_index("observable")["w1_over_sigma"]
    return t


def summarize(cfg):
    ds, pfx = cfg["dataset"], io_utils.prefix(cfg)
    sdir, fdir = io_utils.summary_dir(cfg), io_utils.summary_dir(cfg, "figures")
    levels = list(cfg["data"]["levels"])
    sections = []

    # taggers
    rows = []
    for name, m in _load_metrics(cfg, "tagger"):
        method, level = _split_variant(m["kind"])
        row = {"run": name, "input": m["kind"], "method": method, "level": level, "seed": m["seed"],
               "n_train": m["n_train"]}
        for k, v in m["test"].items():
            if not isinstance(v, list):  # older runs called the bootstrap error "*_std"
                row[k.replace("_std", "_boot_err") if k in ("auc_std", "rej_at_eff50_std") else k] = v
        rows.append(row)
    if rows:
        tag = pd.DataFrame(rows)
        tag.to_csv(os.path.join(sdir, f"{pfx}__tagger_runs.csv"), index=False)
        key = "auc" if ds == "qg" else "mean_binned_resolution"
        num = [c for c in tag.columns if tag[c].dtype.kind in "fi" and c not in ("seed", "n_train")]
        agg = tag.groupby(["method", "level"])[num].agg(["mean", "std"])
        agg.columns = [f"{a} ({'mean' if b == 'mean' else 'spread over seeds'})" for a, b in agg.columns]
        agg.insert(0, "n seeds", tag.groupby(["method", "level"])["seed"].nunique())
        agg = agg.reset_index()
        agg.to_csv(os.path.join(sdir, f"{pfx}__tagger_summary.csv"), index=False)
        sections.append(("Downstream tagger / regressor on the test set. 'boot_err' is the uncertainty from "
                         "resampling the test set; 'spread over seeds' needs at least two seeds.", agg))
        _plot_tagger(tag, key, levels, fdir, ds, pfx)
        if ds == "calo":
            _plot_calo_resolution(cfg, fdir, pfx)

    # SR observables
    frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(
        io_utils.dataset_root(cfg), "runs", f"{pfx}__sreval__*", "observables.csv")))]
    if frames:
        obs = pd.concat(frames, ignore_index=True)
        obs["method"] = obs["method"].replace({"var": "srproj", "var-raw": "sr"})  # older runs
        obs.to_csv(os.path.join(sdir, f"{pfx}__sr_observables.csv"), index=False)
        for level in levels:
            t = values_table(obs, level)
            if t is not None:
                t.to_csv(os.path.join(sdir, f"{pfx}__values__{level}.csv"))
                sections.append((f"{level}: mean of each quantity for the truth, the coarse input and the "
                                 "super-resolved output (other columns are diagnostics)", t))
        piv = obs.pivot_table(index="observable", columns=["level", "method"], values="w1_over_sigma")
        piv.to_csv(os.path.join(sdir, f"{pfx}__sr_w1_table.csv"))
        sections.append(("Distance to the HR distribution, W1 / sigma_HR (0 = identical)", piv))
        _plot_w1(obs, levels, fdir, ds, pfx)

    # C2ST + closure
    rows = []
    for name, m in _load_metrics(cfg, "sreval"):
        for meth, r in m.get("c2st", {}).items():
            rows.append({"level": m["level"], "method": LEGACY_KINDS.get(meth.replace("-", ""), meth),
                         "c2st_auc": r["auc"], "c2st_auc_boot_err": r.get("auc_boot_err", r.get("auc_std"))})
        for meth, r in m.get("lr_closure", {}).items():
            rows.append({"level": m["level"], "method": LEGACY_KINDS.get(meth.replace("-", ""), meth), **r})
    if rows:
        c = pd.DataFrame(rows).groupby(["level", "method"]).first().reset_index()
        c.to_csv(os.path.join(sdir, f"{pfx}__c2st_closure.csv"), index=False)
        sections.append(("Two-sample test AUC (0.5 = a CNN cannot tell it from HR) and LR closure "
                         "(sum|pool(SR) - LR| / sum LR; 0 = agrees with the coarse measurement)", c))
    _write_report(cfg, sections, sdir)
    return dict(sections)


def _plot_tagger(tag, key, levels, fdir, ds, pfx):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    hr = tag[tag["method"] == "hr"][key]
    if len(hr):
        ax.axhspan(hr.mean() - hr.std(ddof=0), hr.mean() + hr.std(ddof=0), color="0.8")
        ax.axhline(hr.mean(), **STYLE["hr"])
    x = np.arange(len(levels))
    for m in _sorted_methods(set(tag["method"]) - {"hr"}):
        sub = tag[tag["method"] == m].groupby("level")[key]
        mean = [sub.mean().get(l, np.nan) for l in levels]
        std = [sub.std(ddof=0).get(l, np.nan) for l in levels]
        ax.errorbar(x, mean, yerr=std, marker="o", capsize=3, **_style(m))
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_xlabel("down-sampling level")
    ax.set_ylabel("tagger ROC AUC" if ds == "qg" else "energy resolution (mean over bins)")
    ax.legend(fontsize=8, frameon=False)
    io_utils.save_figure(fig, fdir, f"{pfx}__tagger_{key}_vs_level")
    plt.close(fig)


def _plot_calo_resolution(cfg, fdir, pfx):
    levels = list(cfg["data"]["levels"])
    fig, axes = plt.subplots(1, len(levels), figsize=(4.5 * len(levels), 3.6), squeeze=False)
    runs = [m for _, m in _load_metrics(cfg, "tagger")]
    for ax, level in zip(axes[0], levels):
        seen = set()
        for m in runs:
            method, lvl = _split_variant(m["kind"])
            if (lvl not in (level, "native")) or method in seen:
                continue
            seen.add(method)
            b = m["test"]["binned"]
            e = [np.sqrt(x["e_lo"] * x["e_hi"]) / 1e3 for x in b]
            ax.plot(e, [x["resolution"] for x in b], "o-", ms=3, **_style(method))
        ax.set_xscale("log")
        ax.set_title(level, fontsize=9)
        ax.set_xlabel("E_inc [GeV]")
        ax.set_ylabel("sigma_eff(E_pred/E_inc)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    io_utils.save_figure(fig, fdir, f"{pfx}__energy_resolution_vs_E")
    plt.close(fig)


def _plot_w1(obs, levels, fdir, ds, pfx):
    for level in levels:
        s = obs[obs["level"] == level].pivot_table(index="observable", columns="method", values="w1_over_sigma")
        if s.empty:
            continue
        s = s[_sorted_methods(s.columns)]
        fig, ax = plt.subplots(figsize=(1.3 * len(s.columns) + 3, 0.35 * len(s) + 1.6))
        im = ax.imshow(np.log10(np.clip(s.values, 1e-3, None)), cmap="magma_r", aspect="auto")
        ax.set_xticks(range(len(s.columns)))
        ax.set_xticklabels([_style(c)["label"] for c in s.columns], fontsize=7, rotation=20, ha="right")
        ax.set_yticks(range(len(s.index)))
        ax.set_yticklabels(s.index, fontsize=8)
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                v = s.values[i, j]
                ax.text(j, i, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if v > 1 else "k")
        fig.colorbar(im, ax=ax, label="log10(W1 / sigma_HR), lower is better")
        ax.set_title(f"{ds} {level}: distance to the HR distribution", fontsize=9)
        io_utils.save_figure(fig, fdir, f"{pfx}__w1_heatmap__{level}")
        plt.close(fig)


def _md(df):
    shown = df.round(4).astype(object).where(df.notna(), "–")  # blanks, not "nan"
    try:
        return shown.to_markdown()
    except ImportError:
        return "```\n" + shown.to_string() + "\n```"


def _write_report(cfg, sections, sdir):
    ds, pfx = cfg["dataset"], io_utils.prefix(cfg)
    lines = [f"# {ds} super-resolution summary ({cfg.get('version', 'v1')})", "", f"generated {io_utils.now()}", "",
             "SR means the model output exactly as generated. Columns such as 'Uniform upsample', "
             "'SR + energy constraint' and 'Tokenizer only' are diagnostics.", ""]
    for title, t in sections:
        lines += [f"## {title}", "", _md(t), ""]
    with open(os.path.join(sdir, f"{pfx}__report.md"), "w") as f:
        f.write("\n".join(lines))
