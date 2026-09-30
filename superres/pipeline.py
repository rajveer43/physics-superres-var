"""Stage runner.

    python -m superres.pipeline --dataset qg --stages download,prepare,train_vqvae
    python -m superres.pipeline --dataset calo --stages all --set vqvae.epochs=10
    python -m superres.pipeline --dataset qg --stages all --smoke        # quick end-to-end check

Stages (in order): download, prepare, tune_vqvae, train_vqvae, tune_var, train_var,
generate, eval_sr, tune_tagger, train_taggers, summarize
"""
import argparse

from . import tuning
from .config import get_config, parse_set_args

ORDER = ["download", "prepare", "tune_vqvae", "train_vqvae", "tune_var", "train_var",
         "generate", "eval_sr", "tune_tagger", "train_taggers", "summarize"]


def tagger_kinds(cfg, levels=None):
    kinds = ["hr"]
    for lvl in levels or cfg["data"]["levels"]:
        kinds += [f"lr:{lvl}", f"uniform:{lvl}", f"var:{lvl}"]
    return kinds


def run_stage(cfg, stage, levels=None, kinds=None, seeds=None):
    levels = levels or list(cfg["data"]["levels"])
    if stage == "download":
        from .download import download
        return download(cfg)
    if stage == "prepare":
        from .data import prepare
        return prepare(cfg)
    if stage == "tune_vqvae":
        return tuning.tune_vqvae(cfg)
    if stage == "train_vqvae":
        from .train import train_vqvae
        return train_vqvae(tuning.with_tuned(cfg, "vqvae", "hr"))
    if stage == "tune_var":
        return tuning.tune_var(cfg)
    if stage == "train_var":
        from .train import train_var
        c = tuning.with_tuned(cfg, "var", "all")
        return {lvl: train_var(c, lvl) for lvl in levels}
    if stage == "generate":
        from .train import generate_sr
        c = tuning.with_tuned(cfg, "var", "all")
        for lvl in levels:
            generate_sr(c, lvl)
        return None
    if stage == "eval_sr":
        from .evaluate import evaluate_level
        c = tuning.with_tuned(cfg, "tagger", "hr")  # C2ST uses the tagger CNN
        return {lvl: evaluate_level(c, lvl) for lvl in levels}
    if stage == "tune_tagger":
        return tuning.tune_tagger(cfg)
    if stage == "train_taggers":
        from .train import train_tagger
        c = tuning.with_tuned(cfg, "tagger", "hr")
        results = {}
        for kind in kinds or tagger_kinds(cfg, levels):
            for seed in seeds or cfg["tagger"]["seeds"]:
                print(f"\n=== tagger on {kind}, seed {seed} ===")
                results[(kind, seed)] = train_tagger(c, kind, seed)
        return results
    if stage == "summarize":
        from .evaluate import summarize
        return summarize(cfg)
    raise ValueError(f"unknown stage {stage}; choose from {ORDER}")


def run(dataset, stages, overrides=None, smoke=False, levels=None, kinds=None, seeds=None):
    cfg = get_config(dataset, overrides, smoke=smoke)
    stages = ORDER if stages == ["all"] else stages
    out = {}
    for s in stages:
        print(f"\n######## {dataset}: {s} ########")
        out[s] = run_stage(cfg, s, levels, kinds, seeds)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["qg", "calo"])
    p.add_argument("--stages", default="all", help=f"comma list from {ORDER} or 'all'")
    p.add_argument("--levels", default=None, help="comma list of LR levels (default: all)")
    p.add_argument("--kinds", default=None, help="comma list of tagger inputs, e.g. hr,var:pool4x4")
    p.add_argument("--seeds", default=None, help="comma list of tagger seeds")
    p.add_argument("--set", nargs="*", default=[], help="config overrides key=value (JSON values)")
    p.add_argument("--smoke", action="store_true", help="tiny settings for a quick end-to-end check")
    a = p.parse_args()
    split = lambda s: s.split(",") if s else None  # noqa: E731
    run(a.dataset, split(a.stages), parse_set_args(a.set), a.smoke, split(a.levels), split(a.kinds),
        [int(x) for x in split(a.seeds)] if a.seeds else None)


if __name__ == "__main__":
    main()
