"""Result layout on Google Drive and small I/O helpers.

Layout (all names are stable, so later stages can find earlier outputs):

    {drive_root}/                        e.g. MyDrive/GSoC_SuperRes/experiments
      {dataset}/
        {version}/                       e.g. v2; one folder per model version
          {run_id}/                      one experiment: start date + tag, e.g. 2026-10-05-tokfix
            README_layout.md
            run_info.json                start time, git commit, GPU, config of every session
            data_meta.json
            runs/{dataset}__{version}__{run_id}__{stage}__{variant}__s{seed}/
                config.json  history.csv  metrics.json  best.pt  last.pt
                figures/      training curves (*__training_curves), and for sreval runs the
                              observable histograms, profiles, example events and per-channel events
                figures/diagnostics/   the same with diagnostic methods added
            optuna/{prefix}__{stage}__{variant}.db / __trials.csv / __best.json
            summary/{prefix}__*.csv|md  summary/figures/*.png|pdf

prefix  : {dataset}__{version}__{run_id}, the start of every run, figure and table name

stage   : vqvae | var | sreval | tagger
variant : hr | pool4x4 | lr-pool4x4 | sr-pool4x4 | uniform-pool4x4 | srproj-pool4x4
Every figure is written as .png and .pdf.
"""
import csv
import datetime as _dt
import json
import os
import re
import shutil

import numpy as np

LAYOUT_README = __doc__


def slug(s):
    return re.sub(r"[^A-Za-z0-9.\-]+", "-", str(s)).strip("-")


def run_id(cfg):
    return slug(cfg.get("run_id") or "untagged")


def prefix(cfg):
    """{dataset}__{version}__{run_id}: the start of every run, figure and table name."""
    return f"{slug(cfg['dataset'])}__{slug(cfg.get('version', 'v1'))}__{run_id(cfg)}"


def model_tag(cfg):
    """{version}-{run_id}: marks model outputs in the local cache."""
    return f"{slug(cfg.get('version', 'v1'))}-{run_id(cfg)}"


def run_name(cfg, stage, variant, seed=None):
    parts = [stage, variant] + ([f"s{seed}"] if seed is not None else [])
    return "__".join([prefix(cfg)] + [slug(p) for p in parts])


def now():
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def experiment_root(cfg):
    """{drive_root}/{dataset}/{version}/{run_id}: everything one experiment writes."""
    root = os.path.join(cfg["paths"]["drive_root"], cfg["dataset"], slug(cfg.get("version", "v1")), run_id(cfg))
    os.makedirs(root, exist_ok=True)
    readme = os.path.join(root, "README_layout.md")
    if not os.path.exists(readme):
        with open(readme, "w") as f:
            f.write("# Result layout\n\n```" + LAYOUT_README + "```\n")
    return root


dataset_root = experiment_root


def save_run_info(cfg, **session):
    """run_info.json: when the experiment started, plus one entry per session (code commit, GPU, config)."""
    path = os.path.join(experiment_root(cfg), "run_info.json")
    info = load_json(path) if os.path.exists(path) else {
        "experiment": prefix(cfg), "started": now(), "sessions": []}
    info["sessions"].append({"time": now(), **session, "config": cfg})
    save_json(path, info)
    return info


def summary_dir(cfg, sub=""):
    d = os.path.join(dataset_root(cfg), "summary", sub)
    os.makedirs(d, exist_ok=True)
    return d


class RunDir:
    def __init__(self, cfg, stage, variant, seed=None):
        self.cfg = cfg
        self.name = run_name(cfg, stage, variant, seed)
        self.path = os.path.join(dataset_root(cfg), "runs", self.name)
        self.fig_dir = os.path.join(self.path, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)

    def file(self, name):
        return os.path.join(self.path, name)

    def exists(self, name):
        return os.path.exists(self.file(name))

    def save_config(self, extra=None):
        save_json(self.file("config.json"),
                  {"run": self.name, "created": now(), "config": self.cfg, **(extra or {})})

    def save_metrics(self, metrics):
        save_json(self.file("metrics.json"), {"run": self.name, "saved": now(), **metrics})

    def reset_history(self):
        if self.exists("history.csv"):
            os.remove(self.file("history.csv"))

    def log(self, row):
        path = self.file("history.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def plot_history(self, groups):
        """Training curves from history.csv -> figures/{run}__training_curves.png|pdf.
        groups: list of (panel title, [columns]); columns missing from the history are skipped."""
        path = self.file("history.csv")
        if not os.path.exists(path):
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        h = pd.read_csv(path)
        groups = [(t, [c for c in cols if c in h]) for t, cols in groups]
        groups = [(t, cols) for t, cols in groups if cols]
        if h.empty or not groups:
            return
        fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.3), squeeze=False)
        for ax, (title, cols) in zip(axes[0], groups):
            for c in cols:
                ax.plot(h["epoch"], h[c], "o-", ms=3, label=c)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("epoch")
            ax.legend(fontsize=7, frameon=False)
        fig.suptitle(self.name, fontsize=9)
        fig.tight_layout()
        save_figure(fig, self.fig_dir, f"{self.name}__training_curves")
        plt.close(fig)


def study_paths(cfg, stage, variant):
    name = run_name(cfg, stage, variant)
    drive_dir = os.path.join(dataset_root(cfg), "optuna")
    local_dir = os.path.join(cfg["paths"]["cache_root"], "optuna", model_tag(cfg))
    os.makedirs(drive_dir, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)
    return {
        "name": name,
        # SQLite locking is unreliable on the Drive FUSE mount, so the live DB is
        # local and copied to Drive after every trial.
        "db_local": os.path.join(local_dir, name + ".db"),
        "db_drive": os.path.join(drive_dir, name + ".db"),
        "trials_csv": os.path.join(drive_dir, name + "__trials.csv"),
        "best_json": os.path.join(drive_dir, name + "__best.json"),
    }


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def save_figure(fig, directory, name):
    os.makedirs(directory, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(directory, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
