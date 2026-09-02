# Why SL8LHGCN still loses to LHGCN (2026-09-02)

Status of the comparison on Amazon-CD, seed 2024, user-wise 8:1:1, full ranking:

| run | Recall@10 | note |
|---|---:|---|
| LHGCN historical acceptance band | 0.092 - 0.098 | `REPRODUCTION.md`, provisional band |
| SL8 `ambient_retract` L2, batch 32768, LR 0.005 | **0.0837 test** (0.0856 valid) | `experiment_runs/amazon_cd/results/sl8_l2_epoch120_test.json` |
| SL8 `karcher1` + LieBN, L2, batch 65536 | 0.0666 valid (PF4096) | `AMAZON_CD_SL8_LIEBN_STATUS_2026-09-02.md` |
| SL8 `karcher1` + LieBN, L4, batch 16384 | 0.0651 valid (PF4096), diverged at epoch 37 | same |
| SL8 `karcher1` + LieBN, L0, batch 65536 | 0.0335 valid (PF4096) | same |

So the gap to beat is roughly 0.084 -> 0.095, about 12% relative, and it is
*not* a numerical-health problem: the committed test manifest reports zero
orientation repairs, zero membership violations, `max_abs_output_log_determinant`
2.4e-6, and a maximum Gregory reconstruction residual of 1.2e-7.

## Finding 1: the hinge margin was transplanted from a different distance scale

`SL8LHGCN_cd.yaml` inherits the released LHGCN objective *verbatim* --
`pairwise_loss: lhgcn_hinge_squared_sum`, `loss_margin: 0.1`, `score_scale: 1.0`
with `learnable_score_scale: false` -- but the two geometries do not put `d^2`
on remotely the same scale.

Measured (`diagnose_margin_scale_transfer.py`, Part A):

| scorer | d^2 p50 | d^2 p95 | margin / d^2 p95 |
|---|---:|---:|---:|
| SL8, coord_clip 0.75 | 1.131 | 1.372 | 0.073 |
| LHGCN c=0.5, spatial norm 2 | 10.011 | 11.458 | 0.009 |
| LHGCN c=0.5, spatial norm 6 | 50.000 | 50.000 | 0.002 |

LHGCN's `sqdist` clamps at 50 and its embeddings are never radius-capped, so
`margin = 0.1` is a *tight* threshold there (0.2-0.9% of the working scale) and
the hinge saturates for well-separated triples -- the loss concentrates on
genuinely violating pairs. On SL(8) the same 0.1 is 7% of the working scale, so
essentially every triple stays inside the hinge's linear region and the
objective degenerates into "push every sampled negative away by the same
amount", with no hard-negative focus.

This is the same failure mode found in the ProCLIP reconstruction, where a
`sl_scale=0.1` factor silently multiplied the effective InfoNCE temperature by
100.

## Finding 2: recalibrating the margin is worth more than the geometry changes

Controlled synthetic bipartite CF (identical propagation, loss, optimiser,
evaluation for both models; `diagnose_margin_scale_transfer.py`, Part B):

| setup | Recall@10 | learned d^2 p95 | hinge active |
|---|---:|---:|---:|
| LHGCN (Lorentz, learnable BN gamma, margin 0.1) | 0.3325 | 2.794 | 3.9% |
| SL8 clip 0.75, margin 0.1 (current config) | 0.3421 | 0.426 | 7.6% |
| SL8 clip 3.0, margin 0.1 | 0.3642 | 0.914 | 3.3% |
| **SL8 clip 0.75, margin 0.02** | **0.4150** | 0.274 | 2.6% |
| SL8 clip 0.75, margin 0.005 | 0.4050 | 0.123 | 2.3% |
| SL8 clip 0.75, margin 0.001 | 0.3792 | 0.064 | 1.9% |
| SL8 clip 3.0, margin 0.005 | 0.4054 | 0.123 | 2.3% |

Margin recalibration alone is worth +21% relative (0.3421 -> 0.4150) and moves
the hinge-active fraction into LHGCN's regime. This synthetic task is *not*
Amazon-CD and does not by itself prove the real gap closes; it establishes that
the mechanism is real and large, and that the current margin sits on the wrong
side of the optimum.

**The searched margin grid never went low enough.** Across the queues and
tuning docs the only values ever tried are 0.05, 0.1, 0.15, 0.2, 0.25 and 0.3,
and 52 of 55 queued trials pin 0.1. The synthetic optimum is near 0.02, below
the entire searched range, and the search direction was mostly upward.

## Finding 3: `coord_clip` is not the binding constraint

The trained SL models never approach their cap: learned `d^2` p95 is 0.426 at
clip 0.75 (cap implies 2.25) and 0.914 at clip 3.0. Raising `coord_clip`, or
extending the Gregory log domain, therefore does not by itself buy capacity --
the model has no incentive to expand while the margin keeps the hinge linear.
Note the direction of the effect: lowering the margin *shrinks* the learned
diameter further while improving ranking.

## Finding 4: the LieBN / karcher1 path is currently behind the simple baseline

The best SL8 number in the repository is still plain `ambient_retract` at L2
(0.0856 valid / 0.0837 test), not the `karcher1` + LieBN path (0.0666 valid at
L2). Two confounds prevent a clean verdict -- the LieBN runs use batch 16384 or
65536 rather than 32768, and their validation is the PF4096 prefiltered
approximation rather than exact full ranking -- but the LieBN path has not yet
demonstrated an advantage and additionally diverged at L4. It should be treated
as an open hypothesis, with `ambient_retract` L2 as the baseline to beat.

## Recommended next runs

1. Margin sweep `loss_margin` in {0.2, 0.1, 0.05, 0.02, 0.01, 0.005} on the
   *current best* configuration (`ambient_retract`, L2, batch 32768, LR 0.005,
   clip 0.75). One axis, six trials; this is the cheapest test of Finding 1 on
   real data.
2. Repeat the winning margin under `karcher1` + LieBN at L2 so the geometry
   comparison is made at each path's own calibrated margin rather than at a
   margin calibrated for the Lorentz baseline.
3. Rerun the L2 LieBN control at batch 32768 with exact (not PF4096)
   validation, so Finding 4 can be settled without confounds.
4. Optional, only if 1-2 show the margin optimum pressed against the low end:
   allow a learnable score scale for the hinge (currently rejected by the
   faithful-hinge check in `sl8lhgcn.py`) as a separately labelled control, so
   the model can calibrate its own scale instead of relying on a swept constant.

Reproduce the measurements with:

```bash
.venv-slrec/bin/python slrec_experiments/diagnose_margin_scale_transfer.py
```
