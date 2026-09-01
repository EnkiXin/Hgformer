# SL(8) in the LHGCN graph framework

## What the released LHGCN actually is

There is no original `LHGCN` class in Hgformer. The released standalone path
is `HGCF` with `conv: lGCN`, now exposed without a Transformer by
`recbole_gnn/model/general_recommender/lhgcn.py`. Its graph operator uses
`D^-1/2 A D^-1/2` without self loops, applies a shared `LorentzBatchNorm`
after every layer, and returns only the last layer. Its pairwise objective is
the batch sum

```text
relu(d(u, positive)^2 - d(u, negative)^2 + margin).sum()
```

and full ranking uses negative hyperbolic distance. This differs from both
LightGCN (which averages layer outputs) and `RecFormer(no_transformer=True)`.

The paper text describes an equal-weight centroid over the node itself and its
neighbours. The released code instead has no self loop, uses symmetric degree
weights, and includes a learned scalar in `LorentzBatchNorm`. Results must be
labelled as either **released-code LHGCN** or **paper-style self-loop
ablation**, not merged.

## SL(8)-LHGCN construction

Each entity stores an unconstrained `8 x 8` matrix `X_v` (64 scalars). It is
projected to the trace-free Lie algebra and exponentiated:

```text
Theta_v = X_v - trace(X_v) I / 8
G_v^(0) = exp(Theta_v) in SL(8)
```

The canonical `ambient_retract` layer is

```text
M_v^(l+1) = sum_w A_tilde[v,w] G_w^(l)
G_v^(l+1) = Retr_SL(M_v^(l+1)).
```

For a positive determinant,
`Retr_SL(M) = M / det(M)^(1/8)`. A negative-determinant aggregate is first
orientation-repaired by reflecting its final column; a singular/non-finite
aggregate explicitly falls back to `exp(trace_free(M))`. The model records
repair and fallback counts and rates on every forward pass through
`projection_diagnostics()` and emits a warning on the first event. The shared
quick-start result also writes this dictionary as `model_diagnostics` in the
result JSON. This is an
extrinsic retraction analogous to LHGCN's ambient aggregation, **not** a
closed-form Frechet mean for the Schatten semidistance.

The `tangent_last` control instead performs

```text
Theta^(l+1) = trace_free(A_tilde Theta^(l))
G^(L) = exp(Theta^(L)).
```

It uses the same graph and final-layer policy but isolates whether direct
group aggregation helps beyond a stable tangent-space control.

The `karcher1` operator aggregates with the one-step Cartan-Schouten
exponential barycenter on **row-normalised** weights `A_hat` (the barycenter
is a weighted mean, so its weights must sum to one per row; the symmetric
weights' sub-unit row sums would otherwise contract every layer toward the
identity — note this contraction silently affects `tangent_last` too):

```text
m_v      = exp(sum_w A_hat[v,w] log G_w)                      # tangent seed
G_v^(l+1)= m_v exp(sum_w A_hat[v,w] log(m_v^-1 G_w^(l))).     # one fixed-point step
```

This is the intrinsic mean whose stationarity condition matches the model's
Schatten log semidistance (Pennec & Arsigny 2012). The output is in SL(8) by
construction: no determinant retraction, orientation repair, or singular
fallback exists on this path. Principal-log failures (non-finite or
oversized Gregory logs) are zeroed and *counted* in the layer diagnostics
instead of silently propagated. The correction costs one 8x8 log per edge
per layer; `sl_karcher_log_terms` (6), `sl_karcher_edge_chunk` (262144),
`sl_karcher_checkpoint` (true), `sl_karcher_correction` (true), and
`sl_karcher_max_log_norm` (25.0) control truncation, memory, and the guard.
Setting `sl_karcher_correction: false` gives exactly the row-normalised
tangent mean materialised in the group.

## LieBN-style layer normalisation

The released LHGCN applies a shared `LorentzBatchNorm` after every graph
layer (batch centroid centring plus a learnable Frechet-variance rescale).
`sl_layer_norm: liebn` now provides the operator-matched SL(8) analogue
following the LieBN recipe (Chen et al., ICLR 2024), with the
Cartan-Schouten exponential barycenter substituted for the Frechet mean
because SL(n) admits no bi-invariant Riemannian metric:

```text
mu   = barycenter(G_1..G_N)          # tangent mean + optional one-step refinement
xi_i = log(mu^-1 G_i)                # centring by left translation
G'_i = beta exp(gamma / (v + eps) xi_i),   v = dispersion of ||xi_i||_F
```

Defaults are chosen so the SL-vs-Lorentz comparison stays operator-matched
with the released `LorentzBatchNorm`: `liebn_dispersion: mean_norm` (mean of
tangent norms, as in the released code, rather than LieBN's variance),
`liebn_learnable_bias: false` (fixed identity bias, as the released beta is a
fixed origin), no running statistics (the released module recomputes from the
full table every forward, so training and evaluation match). The faithful
LieBN alternatives are explicit options: `liebn_dispersion: frechet`,
`liebn_learnable_bias: true`. Further keys: `liebn_mean` (`karcher1` default,
`tangent`), `liebn_eps` (1e-5), `liebn_log_terms` (8). In `tangent_last`
mode the first-order (identity-anchored) form is applied to the coordinates
after every propagation step, preserving that mode's no-materialisation
efficiency; this is exact only to O(spread^2) BCH terms and is labelled as
such in the module docstring. `sl_layer_norm: none` remains the default and
the historical control; `none`-vs-`liebn` must be reported as separate
configurations.

For user and item group elements, prediction uses

```text
D(A,B) = 0.5 * (||log(A^-1 B)||_Sp + ||log(B^-1 A)||_Sp)
score(A,B) = -scale * D(A,B).
```

`pairwise_loss: lhgcn_hinge_squared_sum` matches the released LHGCN loss
shape and reduction. `pairwise_loss: bpr_mean` is the matched BPR control used
by the existing SLRec experiments. They should be reported separately because
the hinge sum changes its effective optimisation scale with batch size.

## Configurations and local checks

The Amazon-CD starting configuration is:

```bash
python run_recbole_gnn.py \
  -m SL8LHGCN -d Amazon_cd \
  --config-files baseline_config_fixed/SL8LHGCN_cd.yaml \
  --validation-only
```

Append `baseline_config_fixed/SL8LHGCN_bpr.yaml` for the BPR control and/or
`baseline_config_fixed/SL8LHGCN_tangent.yaml` for tangent propagation. The
dataset-agnostic `baseline_config_fixed/SL8LHGCN_reproduction.yaml` can be
applied after any dataset-specific `RecFormer_*.yaml`, preserving that
dataset's filtering and split. The
released Lorentz baseline is selected by applying
`baseline_config_fixed/LHGCN_reproduction.yaml` after the corresponding
dataset-specific `RecFormer_*.yaml` protocol.

CPU regression tests:

```bash
.venv-slrec/bin/python -m pytest -q \
  tests/test_sl8lhgcn.py \
  tests/test_lhgcn_adapter.py \
  tests/test_slrecgraph_adapter.py
```

## Recommended staged grid

Keep the dataset split, one uniform negative, full-ranking validation,
`embedding_size=64`, `matrix_dim=8`, `num_factors=1`, and final-layer output
fixed. Tune in stages rather than taking a Cartesian product immediately:

1. Structural screen: `sl_gcn_mode` in `{ambient_retract, tangent_last}` and
   `gcn_layers` in `{1, 2, 3, 4, 7}`. Use `lhgcn_include_self: false` for the
   released-code comparison; test `true` only as a labelled paper-style
   ablation.
2. Optimisation screen per loss: learning rate in
   `{1e-4, 3e-4, 5e-4, 1e-3, 3e-3}`, `coord_clip` in
   `{0.25, 0.5, 0.75, 1.0}`, and `init_std` in
   `{0.005, 0.01, 0.02}` when `embedding_init: normal`. Compare that against
   `embedding_init: xavier_uniform_combined`, which matches the released
   HGCF+LGCN Adam initialisation. Keep train batch size fixed for hinge-sum
   runs.
3. Geometry screen on the best structural settings: `schatten_p` in
   `{2, 4, 8}`, `symmetric_distance` in `{false, true}`, and, if needed,
   `log_terms` in `{8, 12, 16}`.
4. Compare faithful `lhgcn_hinge_squared_sum` and matched `bpr_mean` as two
   separate result blocks. For BPR only, tune `reg_weight` and learned score
   scale.

Do not select hyperparameters on the test split. Select with validation, then
evaluate the selected checkpoint once on held-out test.

## Numerical and semantic risks

- `ambient_retract` is a new extrinsic mean hypothesis. It is not proven to
  minimise the SL(8) Schatten distance.
- A nontrivial orientation-repair rate means ambient averages regularly leave
  the positive determinant component. Treat that configuration as unstable;
  do not silently average it into the main table. Singular fallbacks beyond
  the expected reserved/isolated id rows require investigation.
- `sl_layer_norm: none` (the default) omits the released LHGCN's per-layer
  `LorentzBatchNorm` entirely and leaves the representation spread
  uncontrolled; `sl_layer_norm: liebn` is the operator-matched analogue (see
  above) but substitutes the exponential barycenter for a Frechet mean, so
  neither configuration is a layer-by-layer identity with the released model.
  Report them separately.
- Direct mode exponentiates every entity and computes determinants at every
  graph layer, so it will be slower than the current coordinate-only SLRec.
- `tangent_last` propagates the full table as 64-scalar coordinates but only
  exponentiates sampled users/items during training. It materialises all
  groups only once per cached full-sort pass. `ambient_retract` cannot use
  this optimisation because its next layer genuinely consumes group matrices.
- Full ranking remains the main bottleneck: pairwise matrix solves/logarithms
  scale with all evaluated user-item pairs. Chunk sizes control memory but do
  not remove that computation.
- `p != 2` invokes singular values; benchmark runtime before launching the
  full multi-dataset grid.

### Amazon-CD forward-cost sanity check

After 5-core filtering, Amazon-CD has about 125.2K entities and the symmetric
training adjacency has about 1.52M nonzeros. With 64 matrix coordinates and
four graph layers, each minibatch repeats roughly

```text
4 * 1.52M * 64 = 389M sparse multiply-adds
```

for graph propagation. `ambient_retract` additionally exponentiates 125.2K
initial `8 x 8` matrices and determinant-normalises 125.2K matrices at each
layer. `tangent_last` avoids those full-table dense matrix operations during
training and exponentiates only the sampled entities.

With roughly 762K training edges, the number of full-graph forwards per epoch
is approximately 93, 24, 12, and 6 for batch sizes 8,192, 32,768, 65,536, and
131,072 respectively. This is why 8,192 is a poor starting point for direct
mode. Start with a short 32,768 smoke run on a 48GB card, then try 65,536; use
the same accepted batch size for all hinge-sum comparisons because changing
it also changes that loss's effective optimisation scale.

## Faster full-ranking validation

Exhaustive `group_log` validation on Amazon-CD scores 66,317 x 58,869 = 3.9e9
pairs, each through a batched 8x8 LU solve plus the K=12 Gregory polynomial
(~8.5K flops/pair, ~3.3e13 flops/pass). The measured ~253 s corresponds to
~130 GFLOP/s — about 1% of the GPU — because millions of tiny batched
solves/matmuls are latency- and launch-bound, not compute-bound. Three
orthogonal levers, in decreasing expected impact:

1. **Experimental two-stage shortlist** (`eval_prefilter: frobenius`,
   `eval_prefilter_candidates: 2048`): one GEMM over the flattened tables
   shortlists candidates by ambient Frobenius distance; the exact SL scorer
   runs on the shortlist only (~29x fewer exact pairs on Amazon-CD). The
   candidate-internal score stays the group-log decoder, but the resulting
   ranking metric is approximate. The full-sort trainer supplies item 0 and
   seen-history exclusions to the selector before `topk`, so masked items do
   not consume shortlist capacity. This fixes candidate-budget waste but does
   not make the method exact: small synthetic-catalog checks do not establish
   containment on Amazon-CD. Keep this path out of early stopping and formal
   validation/test; use it only for screening after checking masked top-k
   containment against one exhaustive run on the same real checkpoint.
   `eval_prefilter: none` (default) is bit-identical to the historical path.
2. **TF32 for the exact stage** (`eval_tf32: true`): scoped to `group_log`
   full-sort scoring and restored afterwards; scores move at TF32 precision.
3. **Fewer/cheaper validations**: the multifidelity schedule spends most of
   its ~216 validations on the two finalists (every 10 epochs); relaxing the
   finalist cadence to 20 epochs, or validating intermediate checkpoints on a
   fixed user subsample and reserving exhaustive validation for selection
   points, cuts wall-clock roughly in half with no change to what is finally
   reported. `benchmark_sl8_compile.py` (torch.compile) and
   `profile_sl8_cd.py` (scorer vs evaluator overhead split) remain the tools
   for confirming any of this on the actual GPU.

The `sl_score_mode: tangent_euclidean` control is a different lever entirely:
it changes the model score itself, whereas `eval_prefilter` uses the group-log
score only inside an approximate candidate set. Neither is an exact algebraic
acceleration of exhaustive group-log ranking.
