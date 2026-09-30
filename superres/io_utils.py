"""Result layout on Google Drive and small I/O helpers.

Layout (all names are stable, so later stages can find earlier outputs):

    {drive_root}/
      README_layout.md
      {dataset}/
        data_meta.json
        runs/{dataset}__{stage}__{variant}__s{seed}/
            config.json  history.csv  metrics.json  best.pt  last.pt  figures/
        optuna/{dataset}__{stage}__{variant}.db / __trials.csv / __best.json
        summary/{dataset}__*.csv  summary/figures/*.png|pdf

stage   : vqvae | var | sreval | tagger | c2st
variant : hr | pool4x4 | lr-pool4x4 | uniform-pool4x4 | var-pool4x4 | varraw-pool4x4
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


def run_name(dataset, stage, variant, seed=None):
    parts = [dataset, stage, variant] + ([f"s{seed}"] if seed is not None else [])
    return "__".join(slug(p) for p in parts)


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


def dataset_root(cfg):
    root = os.path.join(cfg["paths"]["drive_root"], cfg["dataset"])
    os.makedirs(root, exist_ok=True)
    readme = os.path.join(cfg["paths"]["drive_root"], "README_layout.md")
    if not os.path.exists(readme):
        with open(readme, "w") as f:
            f.write("# Result layout\n\n```" + LAYOUT_README + "```\n")
    return root


def summary_dir(cfg, sub=""):
    d = os.path.join(dataset_root(cfg), "summary", sub)
    os.makedirs(d, exist_ok=True)
    return d


class RunDir:
    def __init__(self, cfg, stage, variant, seed=None):
        self.cfg = cfg
        self.name = run_name(cfg["dataset"], stage, variant, seed)
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


def study_paths(cfg, stage, variant):
    name = run_name(cfg["dataset"], stage, variant)
    drive_dir = os.path.join(dataset_root(cfg), "optuna")
    local_dir = os.path.join(cfg["paths"]["cache_root"], "optuna")
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
