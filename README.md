# Physics super-resolution with VAR-style next-scale prediction

ML4Sci / CMS / E2E project: map coarse calorimeter data back to high resolution
with a Visual AutoRegressive (VAR) transformer. The result is judged by physics
observables and by downstream taggers, not by SSIM/PSNR.

| Dataset | Content | HR grid | LR levels (sum-pooled) |
|---|---|---|---|
| `qg` | CMS quark/gluon jet images, channels tracks / ECAL / HCAL | 3 x 125 x 125 (padded to 128) | `pool2x2` (64²), `pool4x4` (32²), `pool8x8` (16²) |
| `calo` | CaloChallenge 2022 Dataset 2, e⁻ showers 1 GeV–1 TeV | 45 layers x 16 angle x 9 radius (padded to 48 layers) | `pool1x2x1`, `pool1x4x3`, `pool3x8x3` |

Level names give the pooling factor per axis. `pool3x8x3` merges 3 layers x 8 angular x 3 radial cells into one.

**Why only up to the native resolution:** there is no ground truth above
125 x 125 (qg) or 45 x 16 x 9 (calo). Output beyond those grids could not be
checked against anything, so the SR target is always the native HR grid, which
means 2x, 4x and 8x upscaling for jets.

## Method

1. **Energy-preserving down-sampling.** Each LR cell is the *sum* of the HR cells
   it covers, which is what a detector with larger cells would record. Total
   energy is exactly conserved.
2. **Multi-scale residual VQ-VAE** (VAR paper, Algorithms 1 and 2). It works on
   `log1p(E / s_c)` images and encodes each HR image into K token maps
   (qg: 1² … 16²; calo: 1x1x1 … 12x4x9). The codebook is shared across scales.
   LPIPS and GAN losses are replaced by a total-energy loss.
3. **Conditional VAR transformer.** Next-scale prediction with a block-causal
   mask, AdaLN and L2-normalised q/k, as in the paper. The class-label start
   token is replaced by the LR measurement. An LR encoder provides the start
   token and the AdaLN condition, and adds a spatial feature map to every
   scale's tokens. One model is trained per LR level.
4. **LR projection (optional).** The SR output is rescaled so every block
   reproduces its measured coarse-cell energy exactly. Both `var-raw` and `var`
   (projected) are evaluated.
5. **Baselines.** `lr` (the coarse grid itself) and `uniform` (each coarse
   cell's energy spread evenly over its fine cells).
6. **Downstream taggers.** One CNN architecture is tuned once on HR, then
   trained separately on every input: HR, each LR grid, each uniform grid and
   each VAR output.
   - qg: quark vs gluon classification
   - calo: incident-energy regression

The 2-D (jets) and 3-D (showers) models share code. The calorimeter angular axis
uses periodic padding.

## Metrics (chosen to be read directly by physicists)

| Question | Metric |
|---|---|
| Does SR recover the right physics distributions? | W1 distance / σ_HR, KS statistic, mean shift, for each observable |
| Is each event reconstructed correctly? | Per-event bias (median) and resolution (IQR/1.349) of SR vs true observable |
| Is SR consistent with what was measured? | LR closure: Σ\|pool(SR) − LR\| / Σ LR |
| Can a network tell SR from truth? | Classifier two-sample test AUC (0.5 = indistinguishable) |
| Does SR help an analysis? (qg) | Tagger ROC AUC and background rejection 1/ε_B at ε_S = 30/50/70%, bootstrap errors |
| Does SR help an analysis? (calo) | Energy response bias and resolution σ_eff/E, overall and per energy bin |

Observables used:
- Jets: per-channel and total energy, jet pT, jet mass, girth, p_TD, τ₂₁ (N-subjettiness), hit multiplicity, radial pT profile.
- Showers: E_dep/E_inc, shower depth centroid and width, radial centroid and width, lateral x/y centroid, energy fractions in early/mid/late layers, hit count, longitudinal and radial profiles, cell-energy spectrum.

## Running on Colab

### Get the code into Colab

Open the notebook straight from GitHub:
[colab.research.google.com/github/rajveer43/physics-superres-var/blob/main/notebooks/superres_colab.ipynb](https://colab.research.google.com/github/rajveer43/physics-superres-var/blob/main/notebooks/superres_colab.ipynb)

Its first code cell clones this repository into `/content/superres_code`. Each
later run of that cell pulls the latest commit, so to change code you push to
GitHub and re-run the cell. The runtime doesn't need restarting.

### Run it

1. Set the runtime to GPU.
2. Keep `SMOKE = True` and run all cells. This is a quick pass on a tiny subset
   with 1–2 epochs, and it writes to `superres_results_smoke/`.
   `FAKE_DATA = True` does the same on synthetic data, with no download.
3. Set `SMOKE = False` and run all cells again for the real results.
4. Change `DATASET` to `'calo'` or `'qg'` and repeat. Run one dataset per
   runtime, and delete the other dataset's cache first if disk is short.

Command-line alternative (for example on a cluster):

```bash
pip install -r requirements.txt
python -m superres.pipeline --dataset calo --stages all --set paths.drive_root=/path/results
python -m superres.pipeline --dataset qg --stages train_var,generate --levels pool8x8 --set var.epochs=20
python tests/smoke_test.py --root /tmp/superres_smoke      # synthetic end-to-end check
```

### Stages

| Stage | What it does | Output |
|---|---|---|
| `download` | Zenodo (calo, md5-checked) or CERNBox share over WebDAV (qg); resumable `wget -c` | `/content/data/raw/<ds>/` |
| `prepare` | Pad, clip negatives, sum-pool to every LR level, split into train/val/test, per-channel energy scale | `/content/data/cache/<ds>/*.npy`, `data_meta.json` on Drive |
| `tune_vqvae` | Optuna: lr, codebook size, latent channels, β, energy weight, width | `optuna/<ds>__vqvae__hr*` |
| `train_vqvae` | Full VQ-VAE training with the tuned parameters | `runs/<ds>__vqvae__hr__s42/` |
| `tune_var` | Optuna on the hardest level: depth (width = 64·depth), lr, dropout, weight decay, label smoothing | `optuna/<ds>__var__all*` |
| `train_var` | One conditional VAR per LR level | `runs/<ds>__var__<level>__s42/` |
| `generate` | VAR SR for all train/val/test events | `/content/data/cache/<ds>/*_sr-var_<level>.npy` |
| `eval_sr` | Observables, W1/KS, per-event bias and resolution, LR closure, C2ST, plots | `runs/<ds>__sreval__<level>/` |
| `tune_tagger` | Optuna on HR: lr, dropout, weight decay, width | `optuna/<ds>__tagger__hr*` |
| `train_taggers` | Tagger/regressor for `hr`, then `lr`, `uniform` and `var` at every level, for each seed | `runs/<ds>__tagger__<input>__s<seed>/` |
| `summarize` | Tables, plots and a markdown report across all runs | `summary/` |

Every training stage resumes from `last.pt` on Drive. Optuna studies resume from
their database copy on Drive. After a disconnect, re-run the notebook from the top.

### Result layout and naming on Drive

```
MyDrive/superres_results/            (…_smoke/ and …_fake/ for test runs)
  README_layout.md
  <dataset>/
    data_meta.json                    shapes, splits, energy scales, label counts
    runs/<dataset>__<stage>__<variant>__s<seed>/
        config.json                   full config + creation time
        history.csv                   one row per epoch
        metrics.json                  final / test metrics
        best.pt  last.pt              checkpoints (last.pt = resume point)
        test_predictions.npz          taggers only
        observables.csv  figures/     sreval only (png + pdf)
    optuna/<dataset>__<stage>__<variant>.db | __trials.csv | __best.json
    summary/
        <dataset>__tagger_runs.csv    one row per tagger run
        <dataset>__tagger_summary.csv mean and std over seeds
        <dataset>__sr_observables.csv all observable comparisons
        <dataset>__sr_w1_table.csv    observable x (level, method) W1 table
        <dataset>__c2st_closure.csv   two-sample-test AUC and LR closure
        <dataset>__report.md          all tables in one page
        figures/                      AUC or resolution vs level, W1 heatmaps, resolution vs E
```

- `stage` is one of `vqvae`, `var`, `sreval` or `tagger`.
- `variant` is the input or level, e.g. `hr`, `pool4x4`, `lr-pool4x4`,
  `uniform-pool4x4` or `var-pool4x4`.
- Names contain no timestamps, so later stages can find earlier outputs. The
  creation time is stored in `config.json` instead.

### Disk budget (local Colab disk; the free tier has about 100 GB)

Worked out from array sizes with the default settings:

| | qg (40k jets) | calo (100k + 100k showers) |
|---|---|---|
| raw download | size of the CERNBox share (not checked) | 2.8 GB |
| HR cache (float32) | 7.9 GB | 5.5 GB |
| LR caches | 2.6 GB | 3.3 GB |
| SR caches, 3 levels | 11.8 GB (float16) | 16.6 GB (float32) |
| **total** | **about 22 GB + raw** | **about 28 GB** |

Lower `data.max_samples` (qg) or `data.max_samples` / `data.test_max` (calo) if
disk or RAM is tight. Drive only receives checkpoints, metrics and figures, a
few hundred MB per dataset.

### Time budget

Not measured yet. Run the smoke pass first; the epoch times it prints in
`history.csv` let you scale the full settings. As a rough guide on a T4:

- VAR training is the dominant cost: one model per level, about 650–700 tokens
  per image.
- For a first full pass, `var.epochs` of 15–20 and `optuna.var_trials` of about
  5 should fit in a Colab session.
- Stages are independent, so they can run across several sessions.
- On an A100 or L4 the code uses bf16 automatically.

## Assumptions to check on the first real run

- **CERNBox access.** The public-share WebDAV endpoint was not tested (no network
  access while writing this). If `download` fails, put direct links in
  `data.extra_urls` or copy the files into `/content/data/raw/qg/`.
- **Jet file format.** The loader auto-detects `.pt` contents (a dict, tuple or
  TensorDataset; channels first or last), as well as parquet, h5 and npz. Check
  the label counts printed by `prepare`. Class `data.signal_label` (default 1)
  is the signal for 1/ε_B.
- **Jet geometry.** `data.pixel_size = 0.0174` (the ECAL crystal size in η–φ)
  sets the coordinates for jet mass, pT and τ₂₁. Comparisons between HR and SR
  do not depend on it.
- **Calo geometry.** The radial bin edges in `config.CALO_R_EDGES` are the
  Dataset 2 values from `binning_dataset_2.xml`. Voxels are ordered
  (layer, angle, radius), as in the CaloChallenge code.
- **Testing so far.** The pipeline has not been run end to end yet. Run
  `tests/smoke_test.py` or the notebook with `FAKE_DATA = True` before the long runs.

## Code map

```
superres/
  config.py        defaults for both datasets, SMOKE settings, dotted overrides
  download.py      Zenodo + CERNBox downloaders
  data.py          raw readers, cache writer, CacheStore, torch Datasets
  physics_ops.py   sum_pool, uniform_upsample, project_to_lr
  models/          nn_utils (2-D/3-D periodic conv), vqvae, var, tagger
  train.py         VQ-VAE / VAR / tagger training, SR generation
  observables.py   jet and shower observables
  metrics.py       AUC, 1/eps_B, response and resolution, W1/KS
  evaluate.py      per-level SR evaluation, C2ST, plots, summaries
  tuning.py        Optuna studies synced to Drive
  pipeline.py      stage runner (CLI + notebook)
notebooks/superres_colab.ipynb
tests/smoke_test.py
```

## References

- K. Tian et al., *Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction*, arXiv:2404.02905.
- Fast Calorimeter Simulation Challenge 2022, Dataset 2, Zenodo record 6366271.
