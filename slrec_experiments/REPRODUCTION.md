# Hgformer → SLRec-Graph reproduction protocol

## Dataset provenance

The authoritative source URLs, releases, raw filenames, byte/SHA256 checks,
experiment roles, filters, and expected graph counts are centralized in
[`../DATASETS.md`](../DATASETS.md) and
`dataset_registry.py`. Inspect the machine-readable registry without
downloading anything:

```bash
python slrec_experiments/dataset_registry.py
```

The essential provenance gates are:

- Amazon CD, Movies, Book, and Toy come from McAuley's **2014 ratings-only**
  CSVs, never Amazon Reviews 2018.
- Douban Book/Movie/Music come from the full SHA256-pinned RecBole-CDR bundle,
  never CoPD or RecBole-GNN Social-Datasets.
- Amazon Book uses the explicit iterative 8-core paper overlay.  The released
  5-core YAML is not the paper protocol.
- MovieLens-100K is a bundled smoke test, not a paper dataset.
- Record both token counts and RecBole counts; the latter include reserved ID
  zero.  `DATASETS.md` preserves the separate Douban Movie +1,000 paper typo.

Prepare or validate only the needed sources:

```bash
python slrec_experiments/prepare_amazon2014.py --list-sources
python slrec_experiments/prepare_amazon2014.py --domain Amazon_cd
python slrec_experiments/prepare_douban.py --list-sources
python slrec_experiments/prepare_douban.py --domain all
python slrec_experiments/prepare_movielens100k.py
```

For a bounded-memory structural diagnostic before choosing a geometry:

```bash
python slrec_experiments/audit_graph_geometry.py \
  dataset/Amazon_cd/Amazon_cd.inter \
  --rating-threshold 3 --k-core 5 --seed 2024 \
  --landmarks 32 --four-point-samples 4096 \
  --output experiment_results/amazon-cd-geometry.json
```

This audits the complete filtered interaction graph, not a training split. Its
sampled four-point delta, cycle rank, and branching statistics are discrete
graph diagnostics only; they are not estimates of smooth-manifold sectional
curvature and cannot by themselves establish which model will win.

## Phase 1: archival Hgformer reproduction

The fixed configurations use seed 2024, dimension 64, at most 500 epochs,
patience 30, one uniform negative, rating ≥ 3, user-wise random 8:1:1 split,
full ranking, and validation Recall@10. Run CD first:

```bash
CUDA_VISIBLE_DEVICES=0 python run_recbole_gnn.py \
  --model RecFormer \
  --config-files baseline_config_fixed/RecFormer_cd.yaml \
  --result-file experiment_results/recformer-cd-seed2024.json
```

Provisional historical acceptance bands are Recall@10 0.092–0.098 and
NDCG@10 0.053–0.058 for CD, and 0.078–0.084 / 0.048–0.053 for Movies. These
are sanity bands, not statistical claims: the old repository did not preserve
data hashes, split hashes, dependency locks, best epochs, or multiple seeds.

## Phase 2: matched SLRec-Graph run

The adapter uses the same legacy loader, split, negative sampler, evaluator,
metrics, and validation criterion:

```bash
CUDA_VISIBLE_DEVICES=0 python run_recbole_gnn.py \
  --model SLRecGraph \
  --config-files baseline_config_fixed/SLRecGraph_cd.yaml \
  --result-file experiment_results/slrecgraph-cd-seed2024.json
```

`matrix_dim=8, num_factors=1` stores 64 scalars per entity and has 63
trace-free degrees of freedom. Propagation is parameter-free normalized sparse
adjacency averaging; the exact symmetric SL score uses a truncated
differentiable matrix logarithm.

After that reference run, compare matrix size and product structure on the
same Amazon-CD split by appending exactly one override file:

```bash
# Low-cost/lower-capacity SL(4): 16 raw / 15 intrinsic dimensions.
CUDA_VISIBLE_DEVICES=0 python run_recbole_gnn.py --model SLRecGraph \
  --config-files "baseline_config_fixed/SLRecGraph_cd.yaml baseline_config_fixed/SLRecGraph_ablation_sl4.yaml" \
  --result-file experiment_results/slrecgraph-cd-sl4-seed2024.json

# Explicit single SL(8) reference: 64 raw / 63 intrinsic dimensions.
CUDA_VISIBLE_DEVICES=0 python run_recbole_gnn.py --model SLRecGraph \
  --config-files "baseline_config_fixed/SLRecGraph_cd.yaml baseline_config_fixed/SLRecGraph_ablation_sl8.yaml" \
  --result-file experiment_results/slrecgraph-cd-sl8-seed2024.json

# Equal-raw-budget product SL(4)^4: 64 raw / 60 intrinsic dimensions.
CUDA_VISIBLE_DEVICES=0 python run_recbole_gnn.py --model SLRecGraph \
  --config-files "baseline_config_fixed/SLRecGraph_cd.yaml baseline_config_fixed/SLRecGraph_ablation_sl4x4.yaml" \
  --result-file experiment_results/slrecgraph-cd-sl4x4-seed2024.json
```

The default `factor_aggregation: l2` is the product metric
`sqrt(sum_f D_f^2)`. Keep it fixed for the primary comparison; `l1` and `mean`
are decoder-scale ablations rather than extra tuning axes for the first pass.

The exact scorer costs `O(F n^3)` for every user-item pair, where `F` is
`num_factors`. Before long tuning, benchmark one validation pass. For example,
compare `--matrix-dim 8 --num-factors 1` against
`--matrix-dim 4 --num-factors 4` with `benchmark_sl_scoring.py`. If exhaustive
SL scoring is too slow, report a separate two-stage experiment (fast exhaustive
retrieval, exact SL reranking) rather than presenting candidate-set metrics as
full-sort.

## Scientific comparison after reproduction

After the archival run reaches its band, freeze a corrected protocol and run
RecFormer, LightGCN, and SLRec-Graph on identical split hashes for seeds 2024,
2025, and 2026. Select hyperparameters only by validation Recall/NDCG; evaluate
test once. Report Recall/NDCG at 5/10/20/50, time, peak memory, tail recall,
coverage, and paired per-user deltas. Include Toy as a negative control because
the historical Hgformer result was below LightGCN there.
