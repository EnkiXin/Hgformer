# Seven-dataset Hgformer → SL(8) runner

`run_multidataset_sl8.py` retains the legacy seven-dataset registry for Amazon
CD, Movies, Toy, Book and Douban Book, Movie, Music.  It intentionally refuses
Amazon Book because the released YAML says 5-core; use
`run_paper_dataset_pipeline.py`, which applies the required 8-core correction,
for any Book or six-paper-dataset run.  For the remaining datasets, the fixed
`RecFormer_*.yaml` file is the protocol authority.  SL(8) inherits its rating/k-core filters, seed, random
user-wise 8:1:1 split, Recall/NDCG metrics, top-k values, and full-ranking
candidate set.  Only model, optimizer, SL geometry, and memory chunk settings
are overridden.

The training phases are deliberately validation-only.  Held-out test is a
separate explicit phase after selection:

```bash
# Inspect the complete plan without touching disk or requiring local data.
python slrec_experiments/run_multidataset_sl8.py \
  --output-root /work/hgformer-sl8 \
  --dry-run --skip-data-audit

# Train RecFormer selections first, then all 12 SL8 lr x clip trials.
# Full-ranking validation is performed at epochs 50, 100, ..., 500.
python slrec_experiments/run_multidataset_sl8.py \
  --data-root /work/data \
  --output-root /work/hgformer-sl8 \
  --gpu-id 7 \
  --phase all --tuning-profile core --epochs 500 --eval-step 50

# Only after all validation choices are frozen: evaluate the two selected
# checkpoints on test once.  This does not retrain either model.
python slrec_experiments/run_multidataset_sl8.py \
  --data-root /work/data \
  --output-root /work/hgformer-sl8 \
  --gpu-id 7 \
  --phase final-test --tuning-profile core
```

`--gpu-id` is the physical CUDA index.  Training children receive that same
index in both `CUDA_VISIBLE_DEVICES` and RecBole's `--gpu_id` argument; do not
replace the latter with logical zero.  The vendored RecBole configuration
rewrites the environment from `gpu_id` during startup.  The standalone
`tune_sl8_full_cd.py` driver follows the same physical-index contract.

The core SL8 grid is the same finite grid as `tune_sl8_full_cd.py`:

- learning rate: 0.001, 0.003, 0.006;
- coordinate cap: 0.5, 0.75, 1.0, 1.5;
- `matrix_dim=8`, `embedding_size=64`, `num_factors=1`;
- `n_layers=0` (no graph propagation);
- production one-negative BPR, one-sided Gregory-12 Schatten-2 distance.

Use `--tuning-profile paper` for only the current paper-parameter SL8 run.
Completed selection JSONs are validated against dataset, model, seed,
configuration hash, evaluation schedule, parameter values, checkpoint
existence, and split fingerprints before being resumed.

## Multiple servers

Sharding is zero-based and deterministic.  It assigns complete datasets, not
individual trials, so RecFormer prerequisites and split fingerprints remain
on the same server:

```bash
# On server k, for k = 0, 1, 2, 3:
python slrec_experiments/run_multidataset_sl8.py \
  --data-root /work/data --output-root /work/hgformer-sl8 \
  --shard-index k --shard-count 4 --phase all
```

The resulting four shards are:

- shard 0: Amazon CD, Douban Book;
- shard 1: Amazon Movies, Douban Movie;
- shard 2: Amazon Toy, Douban Music;
- shard 3: Amazon Book.

Explicit `--datasets` can be used when machines have different capacities.
`--max-new-jobs N` safely stops after N newly completed jobs; rerunning the
same command resumes completed artifacts.

## Data gate

Exact download URLs, releases, roles, sizes/hashes, filter protocols, and
expected statistics are centralized in [`../DATASETS.md`](../DATASETS.md).
Every real run audits the atomic file before producing a training command.
Amazon files must match the McAuley 2014 source row counts (and pinned hashes
where available).  The three Douban domains must match the exact byte sizes
and SHA256 digests of the full RecBole-CDR release.  The much smaller local
CoPD files are rejected before training.

`--deep-data-audit` additionally instantiates the fixed RecBole filter and
checks the post-filter user/item/edge reference counts.  It is more expensive,
especially for Amazon Book, so the exact source gate is the default.  Data
checks can only be skipped for a dry plan; `--skip-data-audit` is rejected for
real execution.
