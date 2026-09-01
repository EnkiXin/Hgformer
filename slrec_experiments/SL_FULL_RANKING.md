# SL8/SL16 full-ranking paths

The production default remains `sl_score_mode: group_log`. It evaluates the
unchanged full candidate set with negative one-sided SL group-log distance.
The `p=2`, `K=12` reproduction overlays explicitly enable
`fast_one_sided_frobenius: true`: this is the same Gregory/Frobenius formula
with one solve and a blocked polynomial. It can differ in the last floating
bits because operations are reassociated, so the unit contract checks both
score agreement and top-k agreement against the two-solve reference.

Distance-membership diagnostics are an audit, not part of the score. One
bounded sample is checked when a new full-sort representation table is
materialised; it is not repeated for every item chunk. User/item
representations, item chart norms, and item chart matrices are cached until
training resumes or a checkpoint cache is loaded/cleared.

## Exact candidate batching

Legacy RecBole interprets `eval_batch_size` as a user-item pair budget and
derives the outer user batch with integer division by `item_num`. On Amazon-CD,
the previous budget produced roughly 17 users while the SL8 scorer was sized
for 64, fragmenting one useful scorer chunk across about four evaluator calls.

`full_sort_user_batch_size` names the outer user count directly. The SL8 and
SL16 overlays set it to 64 and 16 respectively, matching
`eval_user_chunk_size`. This changes neither users, candidates, history masks,
nor metric collection. Saved-checkpoint evaluation exposes the same control:

```bash
python evaluate_recbole_gnn_checkpoint.py \
  --checkpoint-file /path/to/model.pth \
  --full-sort-user-batch-size 64 \
  --eval-user-chunk-size 64 \
  --eval-item-chunk-size 1024
```

The chosen outer batch is recorded in the result JSON. Increase it only after
a memory smoke test because the evaluator still owns a dense
`users x all_items` score matrix.

## Optional Euclidean-chart control

Apply `baseline_config_fixed/SL8LHGCN_chart_euclidean.yaml` to select
`sl_score_mode: tangent_euclidean` (the descriptive alias
`chart_euclidean_distance` is also accepted). This is a separately labelled
decoder control:

```text
score(u, i) = -scale * ||x_u - x_i||_F^2
```

- `tangent_last` and zero-layer models reuse their scaled, radius-capped,
  trace-free effective coordinates without an exp/log round trip.
- `ambient_retract` and `karcher1` first run their actual group propagation,
  then Gregory-log each final entity once and project the result trace-free.
- training and point prediction use direct squared Frobenius distance;
  full ranking uses `||x||^2 + ||y||^2 - 2 x @ y.T`, with TF32 disabled for
  the FP32 GEMM. Full sort therefore no longer constructs a matrix log per
  user-item pair.
- the chart full-sort path multiplies by the complete item table, just like
  LightGCN. `eval_item_chunk_size` applies only to `group_log`; chart mode uses
  `eval_user_chunk_size` plus the outer user batch.

This chart distance is **not** an exact algebraic rewrite of the original
`D_SL(A,B) = ||log(A^-1 B)||`. They coincide in the commuting case and agree
locally near the identity to first order; outside that regime they define
different rankings and must be reported as different models. The norms-plus-
GEMM expression and direct squared differences are equal over real arithmetic,
but float32 cancellation can affect pathological near-ties. Tests cover score
agreement and close non-tied top-k ordering; exact bitwise equality is not
claimed.

`torch.compile` remains an opt-in profiling experiment. The observed gain was
only about 1.08x and compiled kernels can introduce warm-up, shape-cache, and
numerical risks, so formal defaults do not enable it.
