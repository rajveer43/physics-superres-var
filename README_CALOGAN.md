# Running the CaloGAN dataset on Perlmutter

Super-resolution of the CaloGAN EM-calorimeter showers (e+, gamma, pi+; three layers on three different grids).
Design and measured numbers: [docs/calogan_design.md](docs/calogan_design.md). General Perlmutter set-up and the Slurm
job chain: [README_PERLMUTTER.md](README_PERLMUTTER.md). This page is only what is specific to `calogan`.

## How it is organised

`calogan` is three layer datasets (`calogan_l0`, `calogan_l1`, `calogan_l2`: tokenizer, VAR transformer, evaluation
per layer) plus the shower stage that puts the layers back together (shower observables, PID and energy taggers,
closure, summary). `--dataset calogan` runs every stage, looping over the layers where needed, so the Slurm job chain
is the same as for qg. One run writes four experiment folders with the same date, version and tag:

```
$ROOT/experiments/calogan_l0/<experiment>/    tokenizer, transformers, layer evaluation
$ROOT/experiments/calogan_l1/<experiment>/
$ROOT/experiments/calogan_l2/<experiment>/
$ROOT/experiments/calogan/<experiment>/       reassembled showers, taggers, summary/<name>_report.md
```

## Data

`eplus.hdf5`, `gamma.hdf5`, `piplus.hdf5` (406 MB each) from
[doi.org/10.17632/pvn3xc3wy5.1](https://doi.org/10.17632/pvn3xc3wy5.1) (Mendeley Data). **They download
automatically**: the `download` stage fetches them into `$ROOT/raw/calogan/` (resumable `wget`) and checks the sha256
that Mendeley lists, so nothing has to be copied by hand. This runs in the first Slurm job (`prepare`), which needs
outbound internet like the qg and calo downloads do. To use files you already have, set
`EXTRA_SET="data.source_dir=/path/to/files"`: nothing is downloaded then.

## Steps

1. **Get the code** (the `calogan` branch) and set up the environment as in `README_PERLMUTTER.md`:

   ```bash
   git clone --branch calogan --single-branch https://github.com/rajveer43/physics-superres-var.git
   cd physics-superres-var
   bash slurm/setup_env.sh
   ```

2. **Edit `slurm/settings.sh`**. This is the only file to edit:

   ```bash
   ACCOUNT=m1234_g          # your NERSC GPU project; the placeholder blocks submission
   DATASET=calogan
   LEVELS="lr2 lr4 lr8"
   SEEDS="42 43 44"
   RUN_TAG=first            # letters, digits, '-' and '.'
   TUNE=0                   # keep 0: the shower tagger has no Optuna study
   QOS=shared
   ```

   The wall-time limits can stay as they are; the data are small (300k showers, 504 cells each). If the compute node
   has no internet, download once on a login node:
   `python -m superres.pipeline --dataset calogan --stages download --set paths.raw_root=$SCRATCH/superres/raw`

3. **Smoke test first** (synthetic data, debug QOS). Wait for `SMOKE OK` at the end of the log:

   ```bash
   bash slurm/submit_all.sh --smoke
   ```

4. **Real run**:

   ```bash
   bash slurm/submit_all.sh --dry-run    # prints the sbatch commands, submits nothing
   bash slurm/submit_all.sh
   bash slurm/status.sh                  # progress, listed per layer
   ```

   After a failure or a timeout, submit again from the stage that failed, for example
   `bash slurm/submit_all.sh --from var`. Finished work is skipped.

5. **Results**. The last job packs everything into `$SCRATCH/superres/results_calogan_*.tar.gz` (the shower experiment
   and the three layer experiments, without checkpoints). Start with
   `calogan/<experiment>/summary/<name>_report.md`.

## Optional: cross-class test

Does an SR model trained on e+ and gamma only turn pi+ into something EM-like? Run a second experiment with, in
`settings.sh`:

```bash
RUN_TAG=eg-only
EXTRA_SET="data.train_particles=[0,1] joint.cross_ref_tag=first"
```

and submit as above. Tokenizers, transformers and SR are retrained on e+ and gamma only; the report of this run has a
table comparing pi+ with the `first` run.

## What to look at in the report

- The k-NN `template` and `ridge` columns are the bar the VAR has to clear. With a fixed beam position a mean shape
  already looks good, so judge the model on the per-particle W1 / sigma table, the correlations and the two-sample
  AUC, not on the mean shapes.
- `lr8` puts the shower core on the corner of four layer-1 cells; expect it to be dominated by the prior.
- Taggers: by the data-processing inequality SR cannot hold more information than LR, so SR close to LR is the
  expected result. The 2x3 tables show whether an HR-trained tagger can be run on SR.
- Hit threshold: the default is 15 keV. Occupancy falls quickly with the threshold (e+ layer 0: 45 % at 15 keV, 26 %
  at 1 MeV), so check `evaluation/sr_<level>/threshold_scan.csv` before trusting hit counts.

## Checks and known limits

- `python -m pytest tests/` (geometry, splits, observables, download logic, leakage; the real-file checks run when
  `dataset/` exists).
- `python tests/smoke_test.py --dataset calogan --root /tmp/superres_smoke` runs every stage on synthetic data on a CPU.
- Not yet run on Perlmutter: start with the `--smoke` job. The real download (1.2 GB) was checked only up to the first
  bytes of the file and the sha256 values; the unit tests cover the fetch/verify logic with a stand-in for `wget`.
- The qg and calo smoke tests were not re-run to completion after the CaloGAN changes. The qg smoke test runs out of
  memory at `tune_var` on a 24 GB laptop, also on the commit before these changes.
