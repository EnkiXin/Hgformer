# Paper-dataset SL-LHGCN pipeline

`run_paper_dataset_pipeline.py` runs the six datasets from the Hgformer paper,
in this order:

1. Amazon CD
2. Amazon Movies
3. Amazon Book
4. Douban Book
5. Douban Movie
6. Douban Music

All six download URLs, release identifiers, raw filenames, byte/SHA256 pins,
filter definitions, and expected counts are centralized in
[`../DATASETS.md`](../DATASETS.md). Amazon Toy is documented there separately
as a negative control, and MovieLens-100K separately as a smoke-only dataset.

It inherits each released `RecFormer_*.yaml` data protocol (rating filter,
split, seed, metrics, and full-ranking evaluation). Amazon Book is corrected
to iterative 8-core and is hard-gated before training at 211,169 users,
163,788 items, and 5,069,747 interactions (RecBole counts include one padding
token). The actual Douban Movie count, 2,552,305, is recorded as accepted; the
paper table's 2,553,305 is a known +1,000 typo.

Every training child is validation-only. The held-out test is evaluated only
by a separate `--phase final-test` command. All children receive both a
physical-GPU-7 CUDA mask and `--gpu_id=7`; one shared file lock makes execution
strictly serial. Completed jobs resume only when their protocol hashes,
arguments, checkpoint, and train/validation/test split fingerprints match.

## Recommended run

Inspect the complete plan without creating output files or touching a GPU:

```bash
.venv-slrec/bin/python slrec_experiments/run_paper_dataset_pipeline.py \
  --output-root runs/paper-sl8-sl16-practical \
  --datasets all \
  --phase sl-all \
  --sl-dims 8 16 \
  --sl-search practical \
  --dry-run --skip-data-audit
```

Run the practical two-block design for SL8 and SL16 independently:

```bash
.venv-slrec/bin/python slrec_experiments/run_paper_dataset_pipeline.py \
  --output-root runs/paper-sl8-sl16-practical \
  --datasets all \
  --phase sl-all \
  --sl-dims 8 16 \
  --sl-search practical
```

`--phase sl-all` is the recommended new-model-only phase: it executes only
SL8LHGCN and SL16LHGCN. It does not launch LightGCN, LHGCN, RecFormer, or the
capacity control. The broader legacy comparison phase remains available as
`--phase all` when those controls are explicitly wanted.

The practical design evaluates all layer x batch combinations, selects only
on validation Recall@10, then evaluates the joint effective-geometry
(`schatten_p`) x margin block at that winner. `curve` is deliberately absent:
static source auditing proves it is not read by SL8/SL16. SL16 uses 256 raw
coordinates (255 intrinsic), conservative batches/chunks, and is summarized
separately from SL8. The optional 256-dimensional LHGCN capacity control runs
only through `--phase lhgcn-capacity` or the broad `--phase all`; the recommended
`sl-all` phase never launches it.

Use `--max-new-jobs N` to stop safely after N new jobs and rerun the identical
command to resume. To run just one phase, use `--phase controls`, `grid`,
`tune`, `lhgcn-capacity`, or `lhgcn-grid`.

## Larger optional searches

Full Cartesian SL search over layer x batch x Schatten-p x margin:

```bash
.venv-slrec/bin/python slrec_experiments/run_paper_dataset_pipeline.py \
  --output-root runs/paper-sl8-sl16-full \
  --datasets all --sl-dims 8 16 \
  --phase sl-all --sl-search full-cartesian
```

Add the independent LHGCN layer x batch x active-curvature x margin control:

```bash
.venv-slrec/bin/python slrec_experiments/run_paper_dataset_pipeline.py \
  --output-root runs/paper-sl8-sl16-practical-lhgcn-grid \
  --datasets all --sl-dims 8 16 \
  --sl-search practical --lhgcn-search full-cartesian
```

The legacy staged optimizer remains available for SL8 only via
`--sl-dims 8 --sl-search staged`.

## Job counts

Counts below are validation-selection jobs and exclude the explicitly separate
final-test evaluations.

| Design | Per dataset | Six datasets |
| --- | ---: | ---: |
| SL8+SL16 practical, new models only (`sl-all`) | 68 | 408 |
| SL8+SL16 full Cartesian, new models only (`sl-all`) | 600 | 3,600 |
| SL8+SL16 practical + 3 controls + 256d capacity control | 72 | 432 |
| SL8+SL16 full Cartesian + controls | 604 | 3,624 |
| Practical plus LHGCN full control grid | 447 | 2,682 |
| Both full Cartesian designs plus LHGCN full grid | 979 | 5,874 |

Per SL dimension, practical is 15 layer/batch jobs plus at most 19 new
geometry/margin jobs (34 total); full Cartesian is 300 jobs. The independent
LHGCN grid is 375 jobs. SL16 has 4x SL8's raw entity parameters and an 8x dense
cubic-operation proxy, so it is never silently merged into SL8's budget.

## Final held-out test

After selection is complete, repeat the same output root, dimensions, and
search choices with the new-model-only final-test phase:

```bash
.venv-slrec/bin/python slrec_experiments/run_paper_dataset_pipeline.py \
  --output-root runs/paper-sl8-sl16-practical \
  --datasets all --sl-dims 8 16 \
  --sl-search practical --phase sl-final-test
```

This loads and tests only the selected SL8/SL16 checkpoints; it does not
retrain or reselect models. The broader `--phase final-test` is reserved for a
run that also selected the comparison controls.
