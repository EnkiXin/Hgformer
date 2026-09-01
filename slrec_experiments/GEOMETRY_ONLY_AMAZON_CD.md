# Geometry-only SL(4) on Amazon-CD

The model configuration is composed in this order:

1. `baseline_config_fixed/SLRecGraph_cd.yaml` pins the Amazon-CD filtering,
   random 80/10/10 user-grouped split, seed, loss, and reported metrics.
2. `baseline_config_fixed/SLRecGraph_geometry_sl4.yaml` selects one directly
   learned SL(4) point per user/item and sets `n_layers: 0`.
3. Use `baseline_config_fixed/SLRecGraph_tune_sampled.yaml` only to select
   hyperparameters, then replace it with
   `baseline_config_fixed/SLRecGraph_eval_full.yaml` for the selected
   checkpoint's final exact ranking evaluation.

## Initial sampled-validation run

```bash
python -u run_recbole_gnn.py \
  --model SLRecGraph \
  --dataset Amazon_cd \
  --config-files "baseline_config_fixed/SLRecGraph_cd.yaml baseline_config_fixed/SLRecGraph_geometry_sl4.yaml baseline_config_fixed/SLRecGraph_tune_sampled.yaml" \
  --validation-only \
  --result-file experiment_runs/amazon_cd/results/sl4-nognn-start.json
```

`uni100` is a search proxy, not the paper metric.  The current vendored legacy
RecBole `NegSampleEvalDataLoader` must concatenate every user's sampled batch
before this command is trusted; a version that retains only `data_list[0]`
silently evaluates the wrong interactions.  Until that loader is fixed or an
official RecBole loader is used, do not launch this sampled search.

## Compact tuning order

Keep the seed and data protocol fixed.  Tune one group at a time and retain
the best `Recall@10` validation setting:

1. Learning rate: `0.0005, 0.001, 0.003`.
2. Coordinate radius: `coord_clip=0.5, 0.75, 1.0`.
3. Initialisation: `init_std=0.008, 0.012, 0.02`.
4. Regularisation: `reg_weight=0, 1e-6, 1e-5`.
5. For the best setting only, try `score_scale=0.5, 1.0, 2.0` with
   `learnable_score_scale=true`.

The defaults in `SLRecGraph_geometry_sl4.yaml` are included in every group,
so a coordinate-wise search needs ten new runs (eleven including the initial
run). Leave `matrix_dim=4`, `num_factors=1`, `n_layers=0`, `schatten_p=2`, and
`log_terms=12` fixed in this first experiment; changing them would confound
the geometry-only baseline with capacity or numerical-accuracy ablations.

After selection, load the sampled-validation checkpoint and construct the
evaluation data with `SLRecGraph_eval_full.yaml`.  Report only that full-sort
held-out test result alongside Hgformer/LightGCN results.

## Resumable staged tuner

The tuner never evaluates test, saves one best checkpoint per trial, stops at
the first failed command, and skips only a result JSON that has complete
validation metrics plus an existing checkpoint.  Preview stage 1 without
starting training:

```bash
python -u slrec_experiments/tune_slrec_geometry_cd.py \
  --stage lr \
  --output-root experiment_runs/amazon_cd \
  --existing-base-result experiment_runs/amazon_cd/results/sl4-nognn-start.json \
  --dry-run
```

Remove `--dry-run` to search `learning_rate=5e-4,1e-3,3e-3`.  The explicit
legacy-result option maps an already completed default `3e-3` run onto that
trial; it is accepted only if it used all three fixed overlays and saved a
checkpoint.

Continue each later stage from the preceding summary so every candidate keeps
the previously selected values:

```bash
python -u slrec_experiments/tune_slrec_geometry_cd.py \
  --stage coord_clip \
  --output-root experiment_runs/amazon_cd \
  --resume-from experiment_runs/amazon_cd/sl4-geometry-tuning/lr/summary.json

python -u slrec_experiments/tune_slrec_geometry_cd.py \
  --stage init_std \
  --output-root experiment_runs/amazon_cd \
  --resume-from experiment_runs/amazon_cd/sl4-geometry-tuning/coord_clip/summary.json

python -u slrec_experiments/tune_slrec_geometry_cd.py \
  --stage reg_weight \
  --output-root experiment_runs/amazon_cd \
  --resume-from experiment_runs/amazon_cd/sl4-geometry-tuning/init_std/summary.json
```

Instead of `--resume-from`, later stages accept explicit values such as
`--learning-rate 0.001 --coord-clip 0.5`.  The script refuses to start when a
preceding-stage value is missing, preventing an accidental reset to the
original static defaults.  `--values` can replace the grid for the current
stage without changing any inherited parameter.
