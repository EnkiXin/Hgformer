# GGCF clean-room reproduction contract

This repository's `GGCF` is an independent implementation of the published
formulas in [Geometric Interaction Augmented Graph Collaborative
Filtering](https://arxiv.org/abs/2208.01250) (CIKM 2023,
[DOI 10.1145/3583780.3615204](https://doi.org/10.1145/3583780.3615204)). It is
not authors' code and must be reported as **paper-faithful clean-room**.

## Formula mapping

| Paper component | Local implementation |
| --- | --- |
| `w_ui = 1 / sqrt(deg(u) deg(i))` | A symmetric, self-loop-free native sparse `D^-1/2 A D^-1/2`, rebuilt only from `train_data.dataset.get_interactions()` |
| Euclidean aggregator | Sparse Euclidean weighted sum |
| Eq. (5) | Positive weighted ambient sum followed by Lorentz-norm normalisation (closed-form Lorentz centroid) |
| Eq. (6), E side | `h_E + gamma * d_E(h_E, log_o(h_H)) * log_o(h_H)` |
| Eq. (6), H side | Exact `exp_o`, `log_o`, Lorentz distance, scalar multiplication, `P_(o->x)`, `exp_x`, and gyro-add |
| Eq. (7) | Equal mean for E and equal-weight Lorentz centroid for H across layers 0 through K |
| Decoder | Euclidean dot product plus trainable `lambda` times Lorentz inner product |
| Objective | Mean BPR loss plus configurable L2 on sampled raw E/H tangent parameters |

Curvature is exactly -1, matching the paper. Every trainable entity parameter
is Euclidean; H points are constructed with `exp_o`. The test suite checks the
Minkowski quadratic form and future-sheet condition after initialisation, each
graph layer, and final layer fusion.

## Disclosed ambiguities and clean-room choices

The manuscript reports embedding dimension 64 but does not say whether this is
per branch or shared across E/H. The local default fixes a fair **64-coordinate
total trainable budget**, E32 + H32:

```yaml
embedding_size: 64
ggcf_branch_size: 32
```

The alternative paper-per-branch interpretation is explicit rather than
silent:

```yaml
embedding_size: 128
ggcf_branch_size: 64
```

The Lorentz ambient time coordinate is derived and is not counted as a
trainable coordinate. Equal E/H widths are required because Eq. (6) directly
maps and compares the two branches.

The printed H expression in Eq. (7) normalises each layer point *inside* a sum.
Taken literally, that sum generally leaves the hyperboloid. `GGCF` uses the
equal-weight Lorentz centroid, the manifold-preserving interpretation
consistent with Eq. (5), and records this choice as
`hyperbolic_layer_fusion: lorentz_centroid`.

The paper does not publish initial values for `gamma`, `gamma_prime`, or
`lambda`, nor an embedding initialiser. These remain validation-tunable
clean-room settings. Its reported learning-rate grid is
`{1e-2, 5e-3, 1e-3, 5e-4, 1e-4}` and L2 grid is
`{0, 1e-6, 1e-5, 1e-4, 1e-3}`.

## Experiment configurations

- `recbole_gnn/properties/model/GGCF.yaml` contains dependency-safe model
  defaults.
- `baseline_config_fixed/GGCF_cd.yaml` is applied after
  `baseline_config_fixed/RecFormer_cd.yaml` to reuse the exact Hgformer
  Amazon-CD filtering, split, seed, metrics, and full-ranking evaluator.

The paper reports MovieLens-100K and LastFM, not Amazon-CD. Therefore an
Amazon-CD run is a protocol-matched transfer experiment, not a reproduction of
the paper's Table 2. Full-sort scores are exact; item chunking only bounds
memory. Validation/test edges never enter message passing.
