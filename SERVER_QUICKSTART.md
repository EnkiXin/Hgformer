# Hgformer + geometry-only SL(8): server quickstart

This bundle vendors the modified RecBole 1.0.1 code used by Hgformer. Do not
install a separate RecBole package into the repository. Commands below use one
NVIDIA GPU and perform full-ranking validation on the held-out validation
split; hyperparameter-search commands do not evaluate the test split.

## 1. Environment

Use Python 3.9-3.11. Install a CUDA-enabled PyTorch build first. The currently
verified server environment is PyTorch 2.2.2 + CUDA 12.1.

```bash
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-hgformer.txt
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

## 2. Amazon-CD data

Download and convert the exact Amazon Reviews 2014 ratings-only source:

```bash
python slrec_experiments/prepare_amazon2014.py --domain Amazon_cd
```

Verify the atomic file before training:

```bash
test "$(stat -c %s dataset/Amazon_cd/Amazon_cd.inter)" -eq 152336079
test "$(wc -l < dataset/Amazon_cd/Amazon_cd.inter)" -eq 3749005
printf '%s  %s\n' \
  7061471c288df93ba65bfede355aeb013e10dbdfc249db8f20a02bbf8ae031c4 \
  dataset/Amazon_cd/Amazon_cd.inter | sha256sum -c -
```

## 3. Reproduce the current SL(8) run

This is the current geometry-only configuration: one SL(8) factor, 63
intrinsic dimensions, no graph propagation, BPR with one uniform negative,
and full-ranking validation every 50 epochs.

```bash
mkdir -p outputs/sl8-current/{checkpoints,logs,results}
CUDA_VISIBLE_DEVICES=0 python -u run_recbole_gnn.py \
  --model SLRecGraph \
  --dataset Amazon_cd \
  --config-files "baseline_config_fixed/SLRecGraph_cd.yaml baseline_config_fixed/SLRecGraph_geometry_sl8.yaml baseline_config_fixed/SLRecGraph_eval_full.yaml" \
  --epochs=500 \
  --eval_step=50 \
  --stopping_step=10 \
  --checkpoint_dir=outputs/sl8-current/checkpoints \
  --show_progress=False \
  --validation-only \
  --result-file outputs/sl8-current/results/validation.json \
  2>&1 | tee outputs/sl8-current/logs/train.log
```

## 4. Amazon-CD SL(8) finite search

Print the search without starting training:

```bash
python slrec_experiments/tune_sl8_full_cd.py \
  --output-root outputs/sl8-grid \
  --profile extended \
  --dry-run
```

Run the resumable search on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python -u slrec_experiments/tune_sl8_full_cd.py \
  --output-root outputs/sl8-grid \
  --profile extended \
  --gpu-id 0
```

The search is validation-only. Its core is the complete 3-learning-rate by
4-coordinate-clip grid. Extended stages continue from the validation winner
with initialization, regularization, metric/scale, loss, and negative-count
ablations. Re-running the command resumes and skips completed trials.

## 5. Multiple datasets and multiple servers

Inspect the multi-dataset plan:

```bash
python slrec_experiments/run_multidataset_sl8.py --help
```

The runner uses each dataset's fixed Hgformer config as the authoritative
filter/split/evaluator protocol, runs RecFormer validation before SL(8), and
supports deterministic dataset-level sharding. It rejects the small CoPD
Douban files; use the official RecBole-CDR Douban atomic files.

## 6. Single-card LHGCN baseline and staged search

The LHGCN runner first evaluates the released `HGCF + conv=lGCN` baseline on
all seven datasets, then tunes layers, curvature, learning rate, and margin.
It exposes only the physical card passed with `--gpu-id`, serializes every
job, and uses a per-card lock. Inspect the complete protocol without starting
CUDA:

```bash
python slrec_experiments/run_multidataset_lhgcn.py \
  --output-root outputs/lhgcn-seven \
  --profile extended --dry-run --skip-data-audit
```

See `slrec_experiments/MULTIDATASET_LHGCN.md` for the baseline-first command,
all grids, resume semantics, artifact layout, and archived-code caveats.

## Result discipline

- Select hyperparameters using full-ranking validation only.
- Do not compare sampled-validation Recall with full-ranking Recall.
- Evaluate the test split once, only after selecting a configuration.
- Keep each server's output directory distinct when running parallel jobs.
