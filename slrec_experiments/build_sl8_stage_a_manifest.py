"""Build the deterministic SL8 Stage-A coverage design."""
from __future__ import annotations

import json
import csv
from collections import Counter, defaultdict
from pathlib import Path

LAYERS = [0, 2, 4, 6, 8]
BATCHES = [65536, 32768, 16384, 8192]
LRS = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01]
MARGINS = [0.05, 0.1, 0.2, 0.3]
EPOCHS = 500
EVAL_STEP = 10
STOPPING_STEP = 2
CLIPS = [("0.5", 0.5), ("0.75", 0.75), ("1.0", 1.0),
         ("1.5", 1.5), ("2.0", 2.0), ("disabled", 0.0)]


def build_cells():
    cells = []
    for batch in BATCHES:
        for layer in LAYERS:
            cells.append(dict(source="structure", layer=layer, batch=batch,
                              learning_rate=0.005, loss_margin=0.1,
                              coord_clip=0.75, coord_clip_label="0.75"))
    hyper = []
    for i, lr in enumerate(LRS):
        for j, (label, clip) in enumerate(CLIPS):
            hyper.append(dict(source="hparam", layer=LAYERS[(i * 6 + j) % len(LAYERS)],
                              batch=16384, learning_rate=lr,
                              loss_margin=MARGINS[(i + j) % 4],
                              coord_clip=clip, coord_clip_label=label))
    # Deterministic within-lr swap: preserve margin totals/coverage while making
    # lr=.005, clip=.75 the anchor duplicate that de-duplication removes.
    a = next(c for c in hyper if c["learning_rate"] == .005 and c["coord_clip_label"] == "0.75")
    b = next(c for c in hyper if c["learning_rate"] == .005 and c["coord_clip_label"] == "0.5")
    a["loss_margin"], b["loss_margin"] = b["loss_margin"], a["loss_margin"]
    seen = {tuple(c[k] for k in ("layer", "batch", "learning_rate", "loss_margin", "coord_clip")) for c in cells}
    removed = []
    for cell in hyper:
        key = tuple(cell[k] for k in ("layer", "batch", "learning_rate", "loss_margin", "coord_clip"))
        if key in seen:
            removed.append(cell)
        else:
            cells.append(cell); seen.add(key)
    for index, cell in enumerate(cells, 1):
        cell["control"] = cell["layer"] == 0
        cell["schatten_p"] = 2
        clip = cell["coord_clip_label"].replace(".", "p")
        lr = format(cell["learning_rate"], "g").replace(".", "p")
        margin = format(cell["loss_margin"], "g").replace(".", "p")
        cell["id"] = f"stageA_{index:03d}_{cell['source']}_L{cell['layer']}_B{cell['batch']}_LR{lr}_M{margin}_C{clip}"
    return cells, removed


def main(path: str):
    cells, removed = build_cells()
    coverage = {axis: dict(Counter(str(c[axis]) for c in cells))
                for axis in ("layer", "batch", "learning_rate", "loss_margin", "coord_clip_label")}
    lr_clip = Counter((str(c["learning_rate"]), c["coord_clip_label"]) for c in cells if c["source"] == "hparam")
    payload = {
        "protocol": {"epochs": EPOCHS, "eval_step": EVAL_STEP,
                     "stopping_step": STOPPING_STEP,
                     "eval_prefilter": "frobenius", "candidates": 4096,
                     "validation_only": True, "schatten_p": 2},
        "assignment": "margin=(lr_index+clip_index)%4 and L=(lr_index*6+clip_index)%7; swap margins within lr=.005 at clips .5/.75; exact duplicate removal",
        "cells": cells, "cell_count": len(cells), "removed_duplicates": removed,
        "coverage": coverage,
        "checks": {"unique": len(cells) == len({c["id"] for c in cells}),
                   "hparam_lr_clip_once": all(v == 1 for v in lr_clip.values()) and len(lr_clip) == 41,
                   "expected_cells": len(cells) == 61},
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = Path(path).with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["id", "source", "layer", "batch", "learning_rate",
                  "loss_margin", "coord_clip_label", "coord_clip",
                  "schatten_p", "control"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(cells)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
