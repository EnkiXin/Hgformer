# Amazon-CD SL8-LHGCN multi-fidelity search

`run_sl8_cd_multifidelity.py` is the resumable, single-GPU driver for the
requested 96-point SL8-LHGCN Cartesian grid. It uses the Hgformer Amazon-CD
filter, random user-grouped 8:1:1 split, Recall/NDCG metrics, and exact
full-ranking validation. It never evaluates the held-out test split.

## Search schedule

| Stage | Fresh runs | Epochs | Validation cadence | Advancement |
|---|---:|---:|---:|---|
| `screen50` | 96 | 50 | once, at epoch 50 | best four per depth (`L0/L2/L4/L6`), 16 total |
| `rerank100` | 16 | 100 | once, at epoch 100 | top 8 if 50/100 rank Spearman is below 0.5; otherwise top 4 |
| `rerank200` | 4 or 8 | 200 | once, at epoch 200 | top 2 |
| `final500` | 2 | 500 | every 10 epochs | validation winner |

Every stage starts from a new model initialization; no checkpoint is continued
into a higher-fidelity stage. Ranking is by validation Recall@10, then NDCG@10,
then the deterministic trial name. The complete workflow therefore launches
118–122 fresh training runs, consumes 8,200–9,000 trial-epochs, and performs
216–220 exact full-ranking validations. The high validation count is mostly
the two finalist runs: 50 validations each.

The initial grid is:

```text
gcn_layers       = [0, 2, 4, 6]
train_batch_size = [16384, 32768, 65536, 131072]
learning_rate    = [0.0005, 0.001, 0.005]
loss_margin      = [0.1, 0.3]
schatten_p       = [2]
```

The exact scorer uses the algebraically equivalent one-sided Frobenius fast
path and the fastest measured end-to-end full-sort layout:

```text
eval_batch_size      = 1048576
eval_user_chunk_size = 64
eval_item_chunk_size = 1024
```

For 58,869 item ids, the outer RecBole batch contains 17 users. Since the user
chunk limit is 64, it adds no further subdivision. This measured configuration
completed one exact validation in 252.970 seconds, versus 257.031 seconds for
outer-17/item-2048 and 296.994 seconds for outer-128/user-32/item-2048. The
isolated scorer microbenchmark did not predict the end-to-end evaluator winner.
This changes only batching and algebraic evaluation of the same score; it does
not use sampled candidates.

## Plan without training

```bash
cd /storage/home/your-user/Hgformer_AGCF
/storage/home/your-user/slrec_hgformer/.venv/bin/python \
  -m slrec_experiments.run_sl8_cd_multifidelity \
  --repo /storage/home/your-user/Hgformer_AGCF \
  --data-root /storage/home/your-user/hgformer_data \
  --output-root /storage/home/your-user/hgformer_results/sl8_cd_multifidelity \
  --python /storage/home/your-user/slrec_hgformer/.venv/bin/python \
  --gpu-id 7 \
  --dry-run > /tmp/sl8_cd_multifidelity_plan.json
```

## Run or resume serially on one physical GPU

```bash
cd /storage/home/your-user/Hgformer_AGCF
/storage/home/your-user/slrec_hgformer/.venv/bin/python \
  -u -m slrec_experiments.run_sl8_cd_multifidelity \
  --repo /storage/home/your-user/Hgformer_AGCF \
  --data-root /storage/home/your-user/hgformer_data \
  --output-root /storage/home/your-user/hgformer_results/sl8_cd_multifidelity \
  --python /storage/home/your-user/slrec_hgformer/.venv/bin/python \
  --gpu-id 7
```

Rerun the identical command after interruption. `--max-new-trials N` can bound
one invocation while preserving resume safety. The driver owns an exclusive
per-user/per-physical-GPU lock and launches subprocesses strictly serially. It
passes physical GPU 7 both through `CUDA_VISIBLE_DEVICES=7` and
`--gpu_id=7`, as required by this vendored RecBole configuration path.

## Artifacts and resume checks

The output root contains `manifest.json`, a live `summary.json`, and one
directory per stage. Each trial has a final annotated JSON, raw subprocess
JSON, log, isolated checkpoint directory, and optional failure record.

A final result is skipped only after all of the following are rechecked:

- model, dataset, seed, epoch budget, validation cadence, and all trial values;
- exact full-ranking split/evaluation configuration and fast scorer chunks;
- finite Recall@10/NDCG@10, with `best_valid_score == Recall@10`;
- expected train/valid/test interaction counts and all split SHA256 values;
- `test_result is null` and runner metadata says test was not evaluated;
- every-layer/final SL(8) and score-path diagnostics;
- checkpoint location, configuration, state dictionary, evaluation epoch,
  byte size, and SHA256 digest;
- parent-stage selection signature, so stale descendants cannot be reused if
  an upstream ranking changes.

An artifact that contains held-out test metrics is a hard error and is never
silently overwritten or used for model selection.
