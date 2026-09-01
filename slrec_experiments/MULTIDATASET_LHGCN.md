# Seven-dataset LHGCN reproduction and tuning

`run_multidataset_lhgcn.py` runs the standalone Light Hyperbolic GCN released
inside Hgformer. The historical implementation is exactly `HGCF` with
`conv=lGCN`; the optional `LHGCN` entry point is a naming-only adapter with
regression tests against that path.

Each `RecFormer_*.yaml` remains the authority for its dataset, rating/k-core
filters, seed 2024, user-grouped random 8:1:1 split, Recall/NDCG metrics,
top-k values, and **full-ranking** candidate set. The
`LHGCN_reproduction.yaml` overlay changes only model/training settings. Every
search trial is validation-only. This runner deliberately has no test mode.

## Single-card commands

Pass one physical GPU index to the runner; do not pass a comma-separated list.
It exports that card as the child's logical `cuda:0`, runs every subprocess
synchronously, and holds a per-user/per-physical-GPU lock for the whole run.

```bash
# Inspect the protocol and staged plan without data, disk writes, or CUDA.
python slrec_experiments/run_multidataset_lhgcn.py \
  --output-root /storage/home/your-user/lhgcn_runs \
  --dry-run --skip-data-audit --profile extended

# First obtain the requested LHGCN baseline on all seven datasets (7 jobs).
python slrec_experiments/run_multidataset_lhgcn.py \
  --data-root /storage/home/your-user/slrec_hgformer/dataset \
  --output-root /storage/home/your-user/lhgcn_runs \
  --gpu-id 0 --profile baseline --epochs 500 --eval-step 50

# Reuse those seven results, then run the staged core sweep.
python slrec_experiments/run_multidataset_lhgcn.py \
  --data-root /storage/home/your-user/slrec_hgformer/dataset \
  --output-root /storage/home/your-user/lhgcn_runs \
  --gpu-id 0 --profile core --epochs 500 --eval-step 50

# Continue through the lower-priority optimisation/initialisation stages.
python slrec_experiments/run_multidataset_lhgcn.py \
  --data-root /storage/home/your-user/slrec_hgformer/dataset \
  --output-root /storage/home/your-user/lhgcn_runs \
  --gpu-id 0 --profile extended --epochs 500 --eval-step 50
```

Use `--datasets amazon-cd` (or any comma-separated subset) for a pilot.
`--max-new-jobs N` stops cleanly after at most `N` new 500-epoch jobs. The
same command validates and skips complete JSON/checkpoint pairs, then resumes
at the first missing trial. An interrupted or stale JSON is retrained; an
artifact that contains a test result is rejected instead of overwritten.

## Search design

This is a staged one-parameter search, not a full Cartesian product. The
full-ranking validation winner is carried forward after each stage:

1. baseline: layers 4, curve 0.5, LR 0.0005, margin 0.1;
2. `gcn_layers`: every integer 1 through 8;
3. `curve`: 0.05, 0.1, 0.2, 0.5, 1.0;
4. `learning_rate`: 0.0001, 0.0003, 0.0005, 0.001, 0.003;
5. `margin`: 0.05, 0.1, 0.2, 0.3, 0.5;
6. extended `scale`: 0.01, 0.05, 0.1, 0.2;
7. extended `weight_decay`: 0, 1e-5, 1e-4, 1e-3, 0.005, 0.01;
8. extended batch size: 8192, 32768, 65536, 131072;
9. extended optimiser: Adam, RSGD, Adagrad, RMSprop, SGD.

The carried setting is not retrained. With the current baseline, the core
profile needs 20 runs per dataset (140 total); extended adds 15 per dataset
(245 total including core). `eval_step=50` performs validation at epochs
50, 100, ..., 500, while `stopping_step=11` prevents early stopping before
the fixed 500-epoch budget.

Two source-code caveats should be kept with the result interpretation:

- under `learner=adam`, HGCF applies Xavier initialisation after constructing
  the hyperbolic entity table, so the preceding `scale` initialisation is
  overwritten; equal scale-stage results are therefore an expected diagnostic;
- changing away from Adam also disables that HGCF Xavier call, so the archived
  code's optimiser comparison is partly an initialisation comparison. The
  runner preserves this released behavior rather than silently rewriting it.

## Artifacts and data gate

The runner writes `lhgcn-manifest.json`, a live cross-dataset
`lhgcn-summary.json`, per-dataset summaries, and per-stage result/log/checkpoint
directories. Resume validation covers model, dataset, seed, parameters,
config hashes, evaluation schedule, checkpoint existence, and all three split
fingerprints.

Before a real run, the shared seven-dataset registry audits every atomic file.
Pinned Amazon releases are checked by rows/hash where available; Douban must
match the full RecBole-CDR byte size and SHA256, so the small CoPD substitutes
are rejected. `--deep-data-audit` additionally instantiates the filtered
RecBole dataset and verifies post-filter users/items/interactions.
