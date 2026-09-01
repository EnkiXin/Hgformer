#!/usr/bin/env python3
"""Run one exact full-ranking evaluation of a saved RecBole-GNN checkpoint."""

import argparse
import json
from pathlib import Path

from recbole_gnn.quick_start import evaluate_recbole_gnn_checkpoint


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument(
        "--selection-result-file",
        type=Path,
        help="sampled-validation result JSON whose split fingerprints must match",
    )
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--skip-valid", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--eval-user-chunk-size", type=int)
    parser.add_argument("--eval-item-chunk-size", type=int)
    parser.add_argument(
        "--full-sort-user-batch-size",
        type=int,
        help="explicit outer full-ranking user batch (independent of item count)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="default: use the checkpoint preference when available",
    )
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate_recbole_gnn_checkpoint(
        args.checkpoint_file,
        evaluate_valid=not args.skip_valid,
        evaluate_test=not args.skip_test,
        eval_batch_size=args.eval_batch_size,
        eval_user_chunk_size=args.eval_user_chunk_size,
        eval_item_chunk_size=args.eval_item_chunk_size,
        full_sort_user_batch_size=args.full_sort_user_batch_size,
        device=args.device,
        show_progress=args.show_progress,
    )
    if args.selection_result_file:
        selection_path = args.selection_result_file.expanduser().resolve()
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        expected = selection.get("split_fingerprints")
        if expected is None:
            raise ValueError(
                f"selection result has no split_fingerprints: {selection_path}"
            )
        if expected != result["split_fingerprints"]:
            raise RuntimeError(
                "full-ranking data split differs from sampled-validation split: "
                f"expected={expected}, actual={result['split_fingerprints']}"
            )
        result["selection_result_file"] = str(selection_path)
        result["split_fingerprints_match"] = True
    payload = json.dumps(result, indent=2, default=_json_default) + "\n"
    if args.result_file:
        output = args.result_file.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"RESULT_JSON={output}")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
