# AGCF / AGCF-SL8Coord Amazon-CD runner

`run_agcf_cd.py` is the controlled, single-card entry point for the first
Amazon-CD experiment.  It does not contain a test-evaluation path.  Every
trial selects and saves a checkpoint from full-ranking validation only.

## First two single-point pilots

From the repository root, inspect the commands without starting training:

```bash
python slrec_experiments/run_agcf_cd.py \
  --family agcf \
  --stage pilot \
  --gpu-id 0 \
  --data-path dataset \
  --output-root outputs/agcf-cd \
  --dry-run

python slrec_experiments/run_agcf_cd.py \
  --family agcf-sl8coord \
  --stage pilot \
  --gpu-id 0 \
  --data-path dataset \
  --output-root outputs/agcf-cd \
  --dry-run
```

Remove `--dry-run` to train.  Defaults are 500 epochs, full-ranking validation
every 10 epochs, and early-stopping counter 30.  `--gpu-id` is a physical CUDA
index.  The runner writes the same physical index to the child's
`CUDA_VISIBLE_DEVICES` and RecBole `--gpu_id` setting.  This duplication is
intentional: the vendored RecBole rewrites `CUDA_VISIBLE_DEVICES` from
`gpu_id` during startup, while PyTorch still sees the selected card as its
logical `cuda:0`.  The runner holds an exclusive output-root lock and never
launches two trials concurrently.

Before either dry-run or training, the runner hashes the raw source and
refuses anything except:

```text
dataset/Amazon_cd/Amazon_cd.inter
bytes: 152,336,079
lines including header: 3,749,005
sha256: 7061471c288df93ba65bfede355aeb013e10dbdfc249db8f20a02bbf8ae031c4
```

The contract records expected post-filter RecBole cardinalities of 66,317
users (66,316 tokens), 58,869 items (58,868 tokens), and 952,547 interactions.
On the current remote layout, pass
`--data-path /storage/home/your-user/hgformer_data`; the runner then checks
`/storage/home/your-user/hgformer_data/Amazon_cd/Amazon_cd.inter`.

The two families compose configs in this fixed order:

```text
AGCF:          RecFormer_cd.yaml -> AGCF_cd.yaml
AGCF-SL8Coord: RecFormer_cd.yaml -> AGCF_cd.yaml -> AGCFSL8Coord_cd.yaml
```

The first config remains the only owner of the `[3,inf)` rating filter,
iterative 5-core user/item filters, seed 2024, random per-user 8:1:1 split,
Recall/NDCG cutoffs, Recall@10 selection, and full-ranking evaluator.  The
runner refuses an overlay that redefines those fields.

## Resume and artifacts

Re-run the exact same command to resume.  A trial is skipped only if its JSON,
split fingerprints, validation metrics, checkpoint config, complete parameter
state, and config/runtime hash all match.  Each trial has separate artifacts:

```text
outputs/agcf-cd/<family>/<stage>/
  results/<trial>.json
  logs/<trial>.log
  checkpoints/<trial>/*.pth
  summary.json
```

The JSON must contain `test_result: null`.  A result with held-out-test metrics
is rejected rather than reused for tuning.

## Optional staged search

Do the two single-point pilots first.  If their loss, geometry diagnostics,
memory, and validation runtime are sane, the optional search is:

```bash
python slrec_experiments/run_agcf_cd.py \
  --family agcf \
  --stage all \
  --gpu-id 0 \
  --data-path /storage/home/your-user/hgformer_data \
  --output-root outputs/agcf-cd \
  --max-new-trials 1
```

`--max-new-trials 1` is a useful babysitting mode: each invocation runs at
most one new trial and the next invocation resumes.  Omit it only after the
single-card resource profile is known.

The search is blocked and sequential, not a Cartesian product.  Each stage
inherits the preceding validation winner:

| Stage | New candidates | Purpose |
|---|---:|---|
| pilot | 1 | conservative paper-guided point |
| dynamics | up to 3 | `(L,K) = (1,2), (2,1), (2,2)` |
| metric | up to 3 | rank 2, rank 8, or `epsilon=0.01` |
| sl8-chart | up to 2 | SL extension only: clip `0.75` or disabled |
| forces | up to 5 | one-factor `alpha/gamma/margin/delta` checks |
| optimizer | up to 3 | one-factor learning-rate/weight-decay checks |

These are values the AGCF paper does not disclose for Amazon-CD, plus the
chart clip introduced only by our SL extension.  MLP widths, scorer semantics,
and memory chunks stay fixed in the first campaign.  The AGCF-SL8Coord scorer
is deliberately one-sided Frobenius/Schatten-2 with 12 matrix-log terms and no
jitter; it is a chart-based surrogate, not an intrinsic Hamiltonian method on
`T*SL(8)`.

For a manual one-stage continuation, pass the completed predecessor summary:

```bash
python slrec_experiments/run_agcf_cd.py \
  --family agcf \
  --stage dynamics \
  --resume-from outputs/agcf-cd/agcf/pilot/summary.json \
  --gpu-id 0 \
  --data-path /storage/home/your-user/hgformer_data \
  --output-root outputs/agcf-cd
```

Do not evaluate test while choosing stages.  After one configuration is
frozen from validation, use the separate checkpoint evaluator exactly once
and keep that test result outside all tuning summaries.
