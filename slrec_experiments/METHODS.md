# Geometry experiment map

This directory deliberately separates a controlled geometry study from the
legacy RecFormer training stack.  All models here use the same RecBole 1.2.1
data split, negative sampler, optimizer, evaluation protocol, and total latent
width.  This makes a gain attributable to the representation/decoder rather
than to a different framework implementation.

## Models

| Run name | Geometry used by the scorer | Purpose |
| --- | --- | --- |
| `BPR` | Euclidean dot product | ID-only baseline |
| `LightGCN` | Euclidean dot product after graph propagation | graph baseline |
| `MixedGeoRec-E` | Euclidean distance, 64 total coordinates | distance-decoder control |
| `MixedGeoRec-HE` | hyperbolic + Euclidean product, 64 total coordinates | controlled dual-geometry adaptation |
| `MixedGeoRec-HES` | hyperbolic + Euclidean + spherical product, 64 total coordinates | ordinary product-manifold control |
| `MixedGeoRec-HES-gated` | entity-conditioned H/E/S mixture-of-experts score | controlled curvature-routing adaptation |
| `SLRec` | intrinsic `SL_p(n)^F` product logarithmic semidistance | attached-paper adaptation |
| `SLRec-Graph` | LightGCN propagation in product Lie-algebra coordinates, then `SL_p(n)^F` scoring | geometry + graph encoder |

The mixed-geometry models are controlled adaptations, not reproductions of
CGCF, CurvGCL, or HyDRA.  In particular, they do not claim those papers'
topology augmentation, knowledge-graph contrastive objectives, or denoising
components.  A user/item-dependent gate is a mixture-of-experts scorer; it is
not itself a fixed Riemannian product metric.

## Attached-paper mapping

For an unconstrained raw matrix `X`, SLRec constructs

```text
X_bar = X - trace(X) I / n
A     = exp(step_scale * X_bar)
```

and scores a user/item pair with

```text
s(u, i) = -exp(log_score_scale) * D_SL(A_u, A_i)
D_SL(A, B) = 0.5 * (||log(A^-1 B)||_Sp + ||log(B^-1 A)||_Sp)
```

With ``F`` factors, the component distances are combined by the canonical
product metric

```text
D_product(u, i) = sqrt(sum_f D_SL(A_u,f, A_i,f)^2).
```

The optional `l1` aggregation sums component metrics; `mean` divides that sum
by `F` and is a scale-normalised ablation. The pairwise objective is BPR plus
raw-coordinate regularization. Single `SL(8)` stores 64 numbers and has 63
effective degrees of freedom. Product `SL(4)^4` also stores 64 numbers, but has
60 effective dimensions; single `SL(4)` stores 16 and has 15. This separates
matrix size from the raw entity-table budget.

For fixed matrix-log truncation length, exact symmetric product scoring costs
`O(F n^3)` per user-item pair and stores `O(F n^2)` pair intermediates. At the
same 64-coordinate budget, the cubic arithmetic proxy is 512 for `SL(8)` and
256 for `SL(4)^4`; the product therefore halves the leading matrix arithmetic,
although extra batched-kernel overhead means wall-clock speedup must be
measured. Single `SL(4)` has proxy 64 but only one quarter of the entity-table
capacity of `SL(8)`.

The implementation exposes the symmetric and one-sided distances.  The
one-sided option is the inexpensive pilot configuration; a result is not
considered geometry-validated until the symmetric/exactness diagnostic agrees
on a held-out sample.

## Claims that the experiment can and cannot support

The static CF experiment tests whether coupled mixed curvature in `SL(n)` is a
useful user/item representation.  It does **not** test the paper's deep
order-aware composition, because no ordered group multiplication occurs.  A
later sequential model must compare ordered multiplication
`g_T ... g_2 g_1` against the commutative control `exp(sum_t X_t)` to test that
claim.

Likewise, independent runs on two domains are not cross-domain
recommendation.  A CDR result requires globally aligned user identifiers,
target-only validation/test masking, and an explicit cross-domain transfer
module.

## Staged protocol

1. Smoke-test every forward/loss/full-sort path on ML-100K for one epoch.
2. Reproduce Hgformer first on Amazon-CD and Amazon-Movies with the original
   500-epoch ceiling and early stopping protocol.
3. Run matched SLRec-Graph pilots on Amazon-CD, then Movies and the Toy negative
   control.
4. Advance only stable methods that improve at least one primary metric without
   materially regressing the other to three seeds on Amazon-Toy/CD/Movies and
   one Douban domain.
5. Run CDR only after the single-domain geometry audit, using target-only,
   pooled/shared, and HCTS-style transfer baselines.

Use seeds `2024`, `2025`, and `2026`.  Keep the total effective latent width at
approximately 64, use the same per-user split, and report Recall/NDCG at
5/10/20/50, tail Recall, item coverage, wall-clock time, peak memory, and the
fraction of users harmed relative to target-only training.

The attached paper's OGBL-PPA results favor Hits@50/100 much more strongly than
Hits@20 or MRR.  Consequently, improvement at Recall@50 and tail coverage is a
reasonable hypothesis; improvement at NDCG@10 must be measured rather than
assumed.
