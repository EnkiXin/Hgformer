# Amazon-CD SL8-LHGCN LieBN status (2026-09-02)

This note records the interrupted numerical-stability run on the Yanglab RTX
4070 Ti and the earlier controls. It contains validation results only. No
held-out test split was evaluated.

## Evaluation protocol

- dataset: `Amazon_cd`, seed 2024, user-wise random 8:1:1 split
- model: `SL8LHGCN`, 64 stored scalars per entity (`matrix_dim=8`)
- graph: normalized bipartite propagation, last-layer aggregation, no self loop
- geometry: `sl_gcn_mode=karcher1`, row-mean seed, correction disabled
- normalization: per-layer `SLLieBatchNorm`, `mean_norm` dispersion
- scorer: `group_log`, Schatten `p=2`, 12 log terms, one-sided Frobenius fast path
- evaluation: validation-only, mask-aware Frobenius prefilter with 4096
  candidates followed by exact `group_log` reranking
- training: Adam, LR 0.005, margin 0.1, coordinate clip 0.75

The PF4096 numbers are approximate full-catalog validation results and must be
labelled as such. `test_result` was not produced.

## L4 / batch 16384 interrupted run

Run tag:
`stageA_013_structure_L4_B16384_LR0p005_M0p1_C0p75`

Requested schedule: 500 maximum epochs, validation every 10 epochs, and
`stopping_step=2`. Training was manually stopped after epoch 43 because the
loss became numerically unstable.

| completed training epochs | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.053283 | 0.027758 | 0.085428 | 0.036160 | 0.148041 | 0.049165 |
| 20 | 0.061068 | 0.032828 | 0.095967 | 0.041968 | 0.162176 | 0.055797 |
| 30 | **0.065125** | **0.035222** | **0.101112** | **0.044622** | **0.166228** | **0.058269** |
| 40 | 0.006422 | 0.003614 | 0.009961 | 0.004558 | 0.017658 | 0.006207 |

The loss decreased normally from 27,955.3136 at epoch 0 to 1,616.9845 at
epoch 36. It then jumped to 159,822,091,892,443.9375 at epoch 37 and remained
around 668k--679k through epoch 43. The epoch-40 validation collapse is
therefore not a model-quality result; it is a post-divergence diagnostic.
There was no CUDA OOM, Python exception, logged NaN, or logged Inf. GPU memory
was about 9.1 GiB of 12 GiB during training.

The best pre-divergence checkpoint is local only:

- local file: `saved/SL8LHGCN-Sep-01-2026_23-27-14.pth`
- saved after the epoch-30 validation
- bytes: 128,204,344
- SHA256: `AA5F9A7DE8956A6EE9A38170FA3094C02AD04F67D4FF40522B96051383794BB8`

The checkpoint is intentionally not committed: it exceeds GitHub's normal
100 MiB file limit. Use release storage, an artifact store, or Git LFS if the
binary must be shared later.

## Earlier controls

| run | best validation epoch (one-based) | Recall@10 | NDCG@10 | status |
|---|---:|---:|---:|---|
| L0 / batch 65536 | 190 | 0.033505 | 0.018872 | completed control |
| L2 / batch 65536 | 180 | 0.066558 | 0.037217 | completed before the latest stability fix |
| L4 / batch 16384 | 30 | 0.065125 | 0.035222 | interrupted after later divergence |

The L4 result at epoch 30 nearly reaches the older L2 result much earlier in
epoch count, but it does not surpass it. The L2 result predates the latest
LieBN stability changes, so L2 should be rerun under the same code and
evaluation protocol before drawing a layer-depth conclusion.

## Current model assessment

The repaired LieBN path prevents the immediate epoch-0 NaN previously observed
for L4 and trains normally for 37 epochs. A remaining rare instability can
still catastrophically increase the objective. Before resuming a broad grid,
instrument the first non-finite or extreme gradient/parameter update around
the LieBN tangent clipping, scale parameter, group-log distance, and optimizer
step. Preserve the epoch-30 checkpoint as the valid pre-divergence artifact.
