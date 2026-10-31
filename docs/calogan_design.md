# CaloGAN: design note

Dataset: CaloGAN EM-calorimeter showers (Nachman, de Oliveira, Paganini; DOI 10.17632/pvn3xc3wy5.1), 100k
showers each of e+, gamma and pi+, E_inc flat on 1-100 GeV. Three longitudinal layers, each on its own grid.

## What the data looks like (measured on the three files)

| | L0 | L1 | L2 |
|---|---|---|---|
| array shape | 3 x 96 | 12 x 12 | 12 x 6 |
| cell (phi x eta) | 160 x 5 mm | 40 x 40 mm | 40 x 80 mm |
| depth | 90 mm | 347 mm | 43 mm |
| mean energy fraction e+ / gamma / pi+ | 0.20 / 0.15 / 0.10 | 0.80 / 0.85 / 0.76 | 0.0015 / 0.002 / 0.14 |
| occupancy (cell > 0) e+ / gamma / pi+ | 46 / 42 / 25 % | 80 / 80 / 61 % | 28 / 31 / 41 % |

- All layers cover 480 x 480 mm. The array axes are therefore (phi, eta), not (eta, phi); L0 is the fine-eta strip
  layer. `tests/test_calogan.py::test_real_files_geometry_and_centering` checks this on the real files.
- The beam hits the centre. The shower axis is the middle cell edge (bins 47|48 of 96, 5|6 of 12, 2|3 of 6) or the
  middle strip (the 3 phi strips of L0). Centroid jitter in L1: 0.04 cell for e+, 0.3 cell for pi+. Lateral position
  is not a free variable; the task is shape and fluctuations.
- (visible + overflow) / E_inc = 0.994 for e+ and gamma; for pi+ it is 0.27 on average (5th-95th percentile
  0.004-0.74). Cell energies span 1e-3 MeV to 28 GeV.

## Decision: three per-layer 2-D problems plus a joint stage

* `calogan_l0`, `calogan_l1`, `calogan_l2` are single-channel 2-D datasets, each with its own tokenizer, VAR and LR
  levels. One `prepare_calogan` reads the HDF5 files once and writes the three caches with identical, stratified
  splits (particle x 10 E_inc bins, 80/10/10) plus per-event particle label, overflow, E_inc and a measured shower
  summary.
* Each layer's VAR gets that summary as an extra condition (log layer energies and layer fractions of all three
  layers, 6 numbers). It is measured, so it is legitimate: sum-pooling conserves each layer's energy, so the vector
  is identical at HR and at every LR level. It does carry E_vis, which for e+/gamma is E_inc to 0.6 %; E_inc itself
  is never an input.
* The dataset `calogan` is the joint stage: it reads the three layer caches and outputs, reassembles whole
  showers, and runs shower observables, PID / energy taggers, closure and the cross-class test. Stages from
  `tune_vqvae` to `baselines` loop over the layers, so the Slurm chain is unchanged (`DATASET=calogan`).

Rejected:

* **Padded 3-D canvas (layer, phi, eta)**: a (3,96) layer next to a (12,12) layer cannot share a grid without padding
  or resampling. Padding makes convolutions straddle meaningless zeros and it makes the beam axis land on different
  cell positions per layer.
* **Resampling to a common 5 mm x 40 mm grid**: invents resolution in L1/L2 (8x and 16x finer in eta than measured),
  so the "truth" would contain structure that never existed and the hit threshold loses its meaning.
* A layer-aware single model is kept as the fallback if the per-layer default fails in the full runs.

## LR levels

Sum-pool per layer, factors (phi, eta). A factor keeps the beam symmetry only if it divides half of the axis
(tested in `test_pooling_levels_divide_the_grid_and_keep_the_beam_symmetry`). A factor of 4 on the 12-cell axes,
or 2 on the 6-cell eta axis of L2, would put the shower core inside one coarse cell, so those are not used.

| level | L0 (3x96) | L1 (12x12) | L2 (12x6) |
|---|---|---|---|
| lr2 | (1,2) -> 3x48 | (2,2) -> 6x6 | (2,1) -> 6x6 |
| lr4 | (1,4) -> 3x24 | (3,3) -> 4x4 | (3,3) -> 4x2 |
| lr8 | (3,8) -> 1x12 | (6,6) -> 2x2 | (6,3) -> 2x2 |

At lr8 the beam sits on the corner of four L1 cells and the SR is largely prior-dominated; read that level against
the template baseline.

## Tokenizer geometry

| layer | latent | scales | tokens |
|---|---|---|---|
| L0 | 3 x 24 (strides 1x2, 1x2) | 1x1, 1x2, 1x4, 2x8, 3x12, 3x24 | 131 |
| L1 | 6 x 6 (stride 2x2) | 1, 2, 3, 4, 6 | 66 |
| L2 | 6 x 6 (stride 2x1) | 1, 2, 3, 6 | 50 |

No periodic padding. Codebook 512 (1024 is too many for <= 288 cells); `codebook_usage` and dead-code resets are
logged each epoch (`history.csv`).

## Baselines (in every table and plot)

`uniform`; `template` (k-NN in LR space, no label, no E_inc, trained on the mixture, exact LR closure); `ridge`
(linear LR -> HR in log-energy, zero-suppressed, exact closure); `vqrec` (tokenizer-only upper bound); `srproj`.

## What would count as a result

The SR model has to beat `template` on event-by-event fluctuations and tails (W1/sigma of the shower-shape
observables per particle and per E_inc bin; correlations; C2ST) and must not turn pi+ into e+ (cross-class test:
train on e+ and gamma only, `data.train_particles=[0,1]`). Tagger columns are reported as they come out: by the data
processing inequality SR cannot hold more information than LR, so SR ~ LR is the expected outcome and the value of
SR is that an HR-trained tool can be run on coarse data.
