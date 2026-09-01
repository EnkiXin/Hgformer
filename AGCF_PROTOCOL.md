# AGCF Amazon-CD conservative pilot protocol

Paper: [Learning on Adaptive Manifolds for Graph Collaborative Filtering
(WWW 2026)](https://dl.acm.org/doi/10.1145/3774904.3792239).

## Status

This is a paper-guided local reimplementation pilot, not a claim of using the
authors' code or exactly reproducing their reported Amazon-CD result.  The
ACM article and the copy of the paper inspected on 2026-08-31 contain no code,
artifact, supplemental repository, or implementation URL, and targeted public
repository searches did not locate an official AGCF release.  If an official
repository is released later, this implementation must be compared against it
before it is described as a reproduction.

The local implementation is
`recbole_gnn/model/general_recommender/agcf.py`; its focused tests are in
`tests/test_agcf.py`.  The configuration must be composed in this order:

```text
baseline_config_fixed/RecFormer_cd.yaml
baseline_config_fixed/AGCF_cd.yaml
```

The first file owns the comparison protocol.  The second is a model-only
overlay and must not redefine the dataset, filters, split, or evaluator.

## What the paper fixes

Section 5.1 reports RecBole, Adam, embedding dimension 64, at most 500 epochs,
early stopping on Recall@10 with patience 30, a random per-user 8:1:1 split,
and all-ranking evaluation.  It reports Recall and NDCG at 10 and 20.  The
evolution time `T` is fixed at 1.0.

The paper does **not** disclose the Amazon-CD winners for `L`, `K`, potential
strength `alpha`, damping `gamma`, margin `m`, learning rate, or weight decay.
It also does not give metric rank `h`, `epsilon`, structural `delta`, the two
MLP architectures, batch size, negative-sampling details, initialization, or
seed.  Every numeric choice for those fields in `AGCF_cd.yaml` is explicitly a
conservative pilot choice.  In particular, `d=64`, `h=4`, `L=1`, `K=1`, and
`T=1` minimize the first implementation check; `alpha=0.1`, `gamma=0.01`,
`delta=0.001`, `epsilon=0.001`, and `margin=0.1` remain exposed for validation
tuning.

## Comparison contract

The effective run inherits from `RecFormer_cd.yaml`:

- dataset `Amazon_cd`, rating threshold `[3,inf)`, and iterative 5-core user
  and item filters;
- reproducible seed 2024 and random per-user 8:1:1 split;
- Recall/NDCG with the repository's `[5, 10, 20, 50]` cutoffs and validation
  selection by Recall@10;
- full-ranking validation and test candidate sets.

This deliberately prioritizes a controlled comparison with the existing
RecFormer result.  The paper's Table 2 instead reports 113,303 users, 82,910
items, and 1,397,717 interactions for Amazon-CD, and does not disclose enough
filtering detail to reconstruct that graph.  Therefore the paper's reported
Recall@10 `0.0984` and NDCG@10 `0.0579` are context only, not acceptance
thresholds for this pilot.

## Run sequence

Select or tune only on validation:

```bash
python run_recbole_gnn.py \
  --model AGCF \
  --dataset Amazon_cd \
  --config-files "baseline_config_fixed/RecFormer_cd.yaml baseline_config_fixed/AGCF_cd.yaml" \
  --validation-only
```

After choosing one configuration, evaluate its held-out test split once by
rerunning the same command without `--validation-only`.  Record both config
files and the split fingerprints emitted by the runner.  The first pilot is a
correctness and resource check; it should not be described as a faithful AGCF
reproduction until the missing paper hyperparameters and dataset construction
are resolved.
