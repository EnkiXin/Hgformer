# Amazon-Toy SL8-LHGCN staged tuning

`run_toy_sl8lhgcn_staged.py` consumes the **complete** summary from
`run_toy_sl8lhgcn_grid.py`. It never accepts that grid's provisional winner.
The winning `gcn_layers` and `train_batch_size`, result JSON, checkpoint
configuration, split hashes, and manifold diagnostics are all checked before
the next search begins.

The fair model-selection stages all use seed 2024, 500 epochs, validation every
50 epochs, full ranking, and no test evaluation:

1. `learning_rate`: `1e-4, 3e-4, 5e-4, 1e-3`
2. `loss_margin`: `.05, .1, .2, .3`
3. `coord_clip`: `.25, .5, .75, 1.0, 1.5`
4. `schatten_p`: `1, 2, 4, 8, inf`
5. `weight_decay`: `0, 1e-5, 1e-4, 1e-3, 5e-3`
6. self loops: off, or on with weight `.1, .5, 1.0`

Each stage includes the unchanged parent candidate. That artifact is reused,
not retrained. After all other candidates finish, the best full-ranking
Recall@10 (NDCG@10 tie-break) becomes the parent of the next stage. This needs
21 new 500-epoch runs rather than all 8,000 Cartesian combinations.

Because several early trials peaked at epoch 499, the frozen final 500-epoch
winner is additionally rerun from scratch with 750 and 1000 epoch budgets.
Those two results live under `stage_07_epoch_extension` and are reported as
budget sensitivity only. They are excluded from the six-stage ranking and can
never replace the equal-budget 500-epoch winner.

Example planning command (does not create the output directory):

```bash
python -m slrec_experiments.run_toy_sl8lhgcn_staged \
  --layer-batch-summary /path/to/layer-batch-grid/summary.json \
  --output-root /path/to/toy-staged-tuning \
  --gpu-id 7 \
  --dry-run \
  --skip-data-audit
```

Example execution command:

```bash
python -m slrec_experiments.run_toy_sl8lhgcn_staged \
  --layer-batch-summary /path/to/layer-batch-grid/summary.json \
  --output-root /path/to/toy-staged-tuning \
  --data-root /path/to/hgformer_data \
  --gpu-id 7 \
  --deep-data-audit \
  --continue-on-error
```

The execution command waits while the prerequisite summary is missing or
incomplete, then shares the same filesystem lock as the layer/batch runner.
The implementation rejects every GPU index except physical GPU 7 and runs
children strictly serially. Results resume only when their exact adaptive
parent, checkpoint configuration, 8:1:1 split fingerprints, full-ranking
protocol, and all SL(8) membership diagnostics match.

## Parameter activity

The runner writes the complete classification to every dry-run plan and
summary. The most important non-tunable or inactive cases are:

- `init_std` is overwritten by `xavier_uniform_combined` and is dead here.
- `reg_weight` is used only by the BPR branch, not faithful hinge.
- `score_scale` is absent from the faithful training loss; a fixed positive
  score multiplier also cannot alter a full-ranking order.
- `factor_aggregation` is dead when `num_factors=1`.
- `n_layers` is shadowed by `gcn_layers`; it is passed only as an equal alias.
- self-loop weight is conditional-dead while self loops are off.
- `sl_scale` is active but confounded with `coord_clip`; it belongs in a later
  joint scale/clip study.
- `log_terms` is an approximation-accuracy/cost choice and should be selected
  by numerical convergence, not validation performance.
- `sl_gcn_mode`, matrix dimension, and factor count are architecture changes,
  so they remain separate ablations.

