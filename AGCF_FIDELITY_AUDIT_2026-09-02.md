# AGCF implementation fidelity audit (2026-09-02)

Audited `recbole_gnn/model/general_recommender/agcf.py` and
`baseline_config_fixed/AGCF_cd.yaml` against the paper text extracted from
`../tmp/pdfs/agcf_www2026/agcf_plain.txt` (WWW 2026, Qi, Liu & Zhou).

## 1. Equation-level check: faithful

Every equation the paper states is implemented as written.

| Paper | Statement | Implementation | Verdict |
|---|---|---|---|
| (6) | `P(0) = Pnet(Q(0))` | `momentum = self.pnet(position)` | ✓ |
| (7) | `K = ½ y' G(x)^-1 y` | consistent with (12) | ✓ |
| (8) | `U = ½ x' (M ⊗ S) x` | `_structural_channel_force` | ✓ |
| (9) | bipartite adjacency `A` | RecBole-GNN normalised `edge_index` | ✓ |
| — | `M = (1+δ)I + D^-1/2 A D^-1/2` | `(1+delta)*x + spmm(norm_adj, x)` | ✓ |
| (10) | `M ⊗ S` | uses the identity `(M ⊗ S) vec(Q) = M Q S`; no `Nd × Nd` matrix built | ✓ |
| (12) | `ẋ = G(x)^-1 y` | `_metric_velocity` | ✓ |
| (15) | `G_v^-1 = A_v A_v' + εI` | MLP factor + `epsilon * momentum` | ✓ |
| — | `Γ = γI` | `self.damping * old_velocity` | ✓ |
| (16) | `y_{k+1} = y_k − Δt[ ½∇_x(y'G^-1y) + α(M⊗S)x_k + Γ G^-1 y_k ]` | `0.5*geometric_force + potential_strength*interaction_force + damping*old_velocity` | ✓ **½ present** |
| (17) | `x_{k+1} = x_k + Δt G(x_k)^-1 y_{k+1}` | `_metric_velocity(position, next_momentum)` | ✓ **symplectic order correct** |
| (18) | `z = Σ_{l=0..L} x(T_l)`, `T_0 = 0` | `output_positions = [position]` then append per segment, summed | ✓ includes `t=0` |
| (20) | `max(0, d² − d² + m)` | `relu(pos − neg + margin)` | ✓ |
| (21) | `d_S² = (z_u − z_i)' S (z_u − z_i)` | `_squared_distance` | ✓ |
| (22) | `score = −d_S²` | `predict` / `_full_sort_scores` | ✓ |

The two easiest details to get wrong are both correct: the `½` on the
geometric force in (16), and the symplectic-Euler ordering in (17), which
evaluates the metric at the **old** position `x_k` but uses the **already
updated** momentum `y_{k+1}`. The full-sort expansion
`‖u‖²_S + ‖i‖²_S − 2u'Si` is algebraically exact for symmetric `S`.

## 2. One deliberate deviation, mathematically identical

The paper says the geometric force "can be computed accurately by automatic
differentiation frameworks". The implementation instead uses an exact
analytic vector-Jacobian product. Verified algebraically: with
`G^-1 = A A' + εI` and `c = A' y`,

```text
y' G^-1 y = ||c||² + ε||y||²      cotangent at A = 2 y c'
```

back-propagated through the explicit two-layer tanh MLP. The `ε||y||²` term
is x-independent and correctly drops from the gradient. This is an exact
substitution, not an approximation, and it avoids nested autograd while
still training positions, momenta, and metric weights (no `detach`).

## 3. Correction to `AGCF_PROTOCOL.md`: two datasets ARE reconstructible

`AGCF_PROTOCOL.md` states the paper "does not disclose enough filtering
detail to reconstruct that graph" and therefore treats the paper's numbers
as "context only". That is correct for Amazon-CD but **wrong for MovieLens
and Gowalla**.

**Gowalla — already exact.** `dataset/AGCF_Gowalla/AGCF_Gowalla.inter`
contains 64,115 users / 164,532 items / 2,018,421 interactions, matching the
paper's Table 2 exactly.

**MovieLens — recipe recovered.** Applying `rating ≥ 3` then an iterative
5-core on users and items to raw ML-1M, and reporting RecBole's
padding-inclusive `user_num`/`item_num`, gives

```text
6,039 users / 3,308 items / 835,789 interactions
```

which is the paper's Table 2 row exactly. (Raw distinct counts are
6,038 / 3,307; RecBole reserves index 0 as `[PAD]` in both id fields.) No
other threshold/core combination tested reproduces all three numbers.

Consequence: the `rating ≥ 3` and 5-core assumptions that the protocol doc
labels "conservative pilot choices" are **confirmed to be the paper's own
choices**. `dataset/AGCF_MovieLens/AGCF_MovieLens.inter` is currently the raw
1,000,209-interaction ML-1M file, so the filters must be applied through the
config:

```yaml
val_interval: {rating: "[3,inf)"}
user_inter_num_interval: "[5,inf)"
item_inter_num_interval: "[5,inf)"
```

The paper's numbers therefore become legitimate acceptance targets on these
two datasets:

| Dataset | R@10 | N@10 | R@20 | N@20 |
|---|---:|---:|---:|---:|
| MovieLens | 0.2086 | 0.2808 | 0.3193 | 0.2922 |
| Gowalla | 0.1351 | 0.0956 | 0.1865 | 0.1098 |

Amazon-CD (113,303 / 82,910 / 1,397,717) still does not match any tested
filtering of the local source and remains context-only.

## 4. What the paper genuinely withholds

Confirmed by reading Section 5.1: learning rate, L2 weight decay, margin `m`,
output steps `L`, integration steps `K`, potential strength `α`, and damping
`γ` are stated to be "tuned via grid search on the validation set" with
**neither the winners nor the search grids published**. Metric rank `h`,
`ε`, structural `δ`, both MLP architectures, batch size, negative sampling,
initialisation, and seed are never mentioned. `AGCF_cd.yaml`'s
`agcf_paper_unknown_pilot_fields` list is accurate and complete.

Only these are paper-fixed: RecBole, Adam, `embedding_size = 64`, 500 max
epochs, early stop on Recall@10 with patience 30, random per-user 8:1:1
split, all-ranking evaluation, and `evolution_time T = 1.0`.

## 5. Caveat on the current pilot configuration

`AGCF_cd.yaml` sets `output_steps: 1` and `integration_steps: 1`, i.e.
`L = K = 1`: a single symplectic step of size `Δt = 1.0`, with
`z = x(0) + x(T)`. This is the intended minimal correctness check, but it is
the weakest possible instantiation of the model — one Euler step cannot
express the long-range propagation the paper's Theorem 2 is about. Numbers
from this configuration must never be reported as AGCF's performance. `L`
and `K` are the first axes to sweep.

## 6. Recommended next steps

1. Build the MovieLens config with the confirmed filters above and run AGCF
   there. MovieLens is small (836K interactions, 4.2% density), so an `L × K`
   grid is cheap, and for the first time the result is comparable to a
   published number.
2. Sweep `L ∈ {1,2,3,4}`, `K ∈ {1,2,4}`, `α`, `γ`, `margin`, `lr` on
   MovieLens validation, exactly as the paper says it did.
3. Only once the reimplementation lands in a plausible band of 0.2086 R@10
   on MovieLens should it be used as a baseline for SL comparisons; before
   that, a "SL beats AGCF" claim would be a claim about an unconverged
   reimplementation.
4. `AGCFSL8Coord` should be re-run on the same MovieLens protocol after the
   SL margin recalibration (see `SL_VS_LHGCN_GAP_2026-09-02.md`), so both
   sides are compared at their own calibrated working point.
