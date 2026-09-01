#!/usr/bin/env python3
"""Micro-sweep exact SL8 full-sort scorer chunk sizes on Amazon-CD.

The script constructs the filtered dataset, split, model, and propagated SL(8)
group tables exactly once.  It then times only ``model._score_groups`` for a
grid of user/item chunk sizes.  It never takes an optimizer step and never
runs validation or test evaluation.

The full-ranking numbers are scorer-only extrapolations.  They intentionally
exclude RecBole masking, top-k selection, metric accumulation, and data-loader
overhead, so they are useful for choosing chunks rather than predicting the
complete evaluator wall time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from recbole.utils import init_seed
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation, get_model


DEFAULT_USER_CHUNKS = (8, 17, 32, 64, 128)
DEFAULT_ITEM_CHUNKS = (512, 1024, 2048, 4096)


def parse_positive_int_csv(value: str) -> Tuple[int, ...]:
    """Parse a comma-separated, duplicate-free tuple of positive integers."""

    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers: {value!r}") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("at least one positive integer is required")
    return tuple(dict.fromkeys(parsed))


def scorer_call_count(
    total_users: int,
    total_items: int,
    user_chunk: int,
    item_chunk: int,
    *,
    outer_users: int | None = None,
) -> int:
    """Return exact nested-loop scorer calls, optionally respecting outer batches.

    RecBole splits users into outer full-sort batches before the model applies
    ``eval_user_chunk_size``.  A non-divisible outer boundary can therefore add
    calls compared with chunking all users as one global matrix.
    """

    values = (total_users, total_items, user_chunk, item_chunk)
    if any(value <= 0 for value in values):
        raise ValueError("user/item totals and chunks must all be positive")
    item_calls = math.ceil(total_items / item_chunk)
    if outer_users is None:
        return math.ceil(total_users / user_chunk) * item_calls
    if outer_users <= 0:
        raise ValueError("outer_users must be positive")
    complete_outer, remainder = divmod(total_users, outer_users)
    user_calls = complete_outer * math.ceil(outer_users / user_chunk)
    if remainder:
        user_calls += math.ceil(remainder / user_chunk)
    return user_calls * item_calls


def candidate_grid(
    user_chunks: Sequence[int],
    item_chunks: Sequence[int],
    max_pairs: int,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    """Build a deterministic grid and explain combinations skipped for safety."""

    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    selected: List[Tuple[int, int]] = []
    skipped: List[Dict[str, Any]] = []
    for user_chunk in user_chunks:
        for item_chunk in item_chunks:
            pairs = user_chunk * item_chunk
            if pairs > max_pairs:
                skipped.append(
                    {
                        "user_chunk": user_chunk,
                        "item_chunk": item_chunk,
                        "pairs": pairs,
                        "reason": f"pairs exceed max_pairs={max_pairs}",
                    }
                )
            else:
                selected.append((user_chunk, item_chunk))
    # Small-to-large makes an accidental allocation regression fail before a
    # large tensor is attempted and keeps the run deterministic.
    selected.sort(key=lambda pair: (pair[0] * pair[1], pair[0], pair[1]))
    return selected, skipped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiment_runs" / "sl8_cd_chunk_sweep.json",
    )
    parser.add_argument("--gpu-id", type=int, default=3)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--train-batch-size", type=int, default=131072)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=7_535_232,
        help="RecBole score budget; 128*58,869 for filtered Amazon-CD",
    )
    parser.add_argument(
        "--user-chunks",
        type=parse_positive_int_csv,
        default=DEFAULT_USER_CHUNKS,
    )
    parser.add_argument(
        "--item-chunks",
        type=parse_positive_int_csv,
        default=DEFAULT_ITEM_CHUNKS,
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=524_288,
        help="skip a chunk shape above this many user-item pairs",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    return parser.parse_args()


def _mib(value: int) -> float:
    return value / 2**20


def _write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _benchmark_shape(
    model: Any,
    user_groups: torch.Tensor,
    item_groups: torch.Tensor,
    user_chunk: int,
    item_chunk: int,
    warmup: int,
    repeats: int,
) -> Dict[str, Any]:
    device = user_groups.device
    scorer_users = user_groups[1 : 1 + user_chunk, None, ...]
    scorer_items = item_groups[None, :item_chunk, ...]

    def score() -> torch.Tensor:
        return model._score_groups(scorer_users, scorer_items)

    with torch.inference_mode():
        for _ in range(warmup):
            warmup_result = score()
            torch.cuda.synchronize(device)
            del warmup_result

        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        durations: List[float] = []
        incremental_peaks: List[int] = []
        absolute_peaks: List[int] = []
        reserved_peaks: List[int] = []
        result = None
        for _ in range(repeats):
            allocated_before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = score()
            torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - started)
            absolute_peak = torch.cuda.max_memory_allocated(device)
            absolute_peaks.append(absolute_peak)
            incremental_peaks.append(max(absolute_peak - allocated_before, 0))
            reserved_peaks.append(torch.cuda.max_memory_reserved(device))
            finite = bool(torch.isfinite(result).all().item())
            del result
            if not finite:
                raise FloatingPointError(
                    f"non-finite scorer output for {user_chunk}x{item_chunk}"
                )

    median_seconds = statistics.median(durations)
    pairs = user_chunk * item_chunk
    return {
        "user_chunk": user_chunk,
        "item_chunk": item_chunk,
        "pairs_per_call": pairs,
        "warmup": warmup,
        "repeats": repeats,
        "durations_seconds": durations,
        "median_seconds": median_seconds,
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "pairs_per_second": pairs / median_seconds,
        "peak_incremental_allocated_mib": _mib(max(incremental_peaks)),
        "peak_absolute_allocated_mib": _mib(max(absolute_peaks)),
        "peak_reserved_mib": _mib(max(reserved_peaks)),
        "finite": True,
    }


def main() -> None:
    args = _parse_args()
    # RecBole's Config parses process argv too; profiler flags are not model
    # overrides and must not leak into that legacy parser.
    sys.argv = [sys.argv[0]]
    args.repo = args.repo.expanduser().resolve()
    args.data_path = args.data_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.gpu_id < 0:
        raise ValueError("gpu-id must be non-negative")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    combinations, skipped = candidate_grid(
        args.user_chunks, args.item_chunks, args.max_pairs
    )
    if not combinations:
        raise ValueError("all requested chunk combinations were skipped")

    config_files = [
        args.repo / "baseline_config_fixed" / "RecFormer_cd.yaml",
        args.repo / "baseline_config_fixed" / "SL8LHGCN_reproduction.yaml",
    ]
    for path in config_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    report: Dict[str, Any] = {
        "schema_version": 1,
        "purpose": "exact SL8 scorer chunk micro-sweep; no training/validation/test",
        "pid": os.getpid(),
        "config_files": [str(path) for path in config_files],
        "requested": {
            "physical_gpu_id": args.gpu_id,
            "layers": args.layers,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "user_chunks": list(args.user_chunks),
            "item_chunks": list(args.item_chunks),
            "max_pairs": args.max_pairs,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "skipped": skipped,
    }

    started = time.perf_counter()
    config = Config(
        model="SL8LHGCN",
        dataset="Amazon_cd",
        config_file_list=[str(path) for path in config_files],
        config_dict={
            "data_path": str(args.data_path),
            # RecBole itself rewrites CUDA_VISIBLE_DEVICES from gpu_id.
            "gpu_id": args.gpu_id,
            "gcn_layers": args.layers,
            "n_layers": args.layers,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "schatten_p": 2,
            "fast_one_sided_frobenius": True,
            "save_dataset": False,
            "save_dataloaders": False,
            "show_progress": False,
        },
    )
    report["setup"] = {"config_seconds": time.perf_counter() - started}
    device = torch.device(config["device"])
    if device.type != "cuda":
        raise RuntimeError(f"CUDA is required; resolved device={device}")
    report["environment"] = {
        "resolved_device": str(device),
        "requested_physical_gpu_id": args.gpu_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    print(
        f"SWEEP pid={os.getpid()} requested_physical_gpu={args.gpu_id} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"resolved={device} name={torch.cuda.get_device_name(device)}",
        flush=True,
    )

    init_seed(config["seed"], config["reproducibility"])
    started = time.perf_counter()
    dataset = create_dataset(config)
    report["setup"]["create_dataset_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    train_data, valid_data, _test_data = data_preparation(config, dataset)
    report["setup"]["data_preparation_seconds"] = time.perf_counter() - started

    init_seed(config["seed"], config["reproducibility"])
    started = time.perf_counter()
    model = get_model(config["model"])(config, train_data.dataset).to(device)
    torch.cuda.synchronize(device)
    report["setup"]["model_init_and_to_cuda_seconds"] = time.perf_counter() - started

    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        user_groups, item_groups = model.forward()
    torch.cuda.synchronize(device)
    report["setup"]["one_group_table_forward_seconds"] = time.perf_counter() - started
    # Production evaluation performs this bounded audit once per freshly built
    # table.  The sweep isolates the repeated scorer kernel, so disable it.
    model.sl_distance_membership_check = False
    model._distance_diagnostics_pending = False

    total_users = model.n_users - 1
    total_items = model.n_items
    configured_outer_users = config["full_sort_user_batch_size"]
    outer_users = (
        int(configured_outer_users)
        if configured_outer_users is not None
        else max(int(config["eval_batch_size"]) // total_items, 1)
    )
    report["dataset"] = {
        "users_including_padding": int(model.n_users),
        "evaluated_users_upper_bound": int(total_users),
        "items_including_padding": int(total_items),
        "real_candidate_pairs_excluding_padding": int(
            total_users * (total_items - 1)
        ),
        "full_sort_pairs_including_padding_item": int(total_users * total_items),
        "train_interactions": int(len(train_data.dataset.inter_feat)),
        "valid_interactions": int(len(valid_data.dataset.inter_feat)),
        "eval_batch_size": int(config["eval_batch_size"]),
        "configured_full_sort_user_batch_size": (
            int(configured_outer_users)
            if configured_outer_users is not None
            else None
        ),
        "outer_users_per_batch": int(outer_users),
        "valid_loader_batches": int(len(valid_data)),
    }
    report["model"] = {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "layers": int(model.n_layers),
        "matrix_dim": int(model.matrix_dim),
        "num_factors": int(model.num_factors),
        "log_terms": int(model.log_terms),
        "schatten_p": model.schatten_p,
        "symmetric_distance": bool(model.symmetric_distance),
        "fast_one_sided_frobenius": bool(model.fast_one_sided_frobenius),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }

    results: List[Dict[str, Any]] = []
    for user_chunk, item_chunk in combinations:
        if user_chunk > total_users or item_chunk > total_items:
            skipped.append(
                {
                    "user_chunk": user_chunk,
                    "item_chunk": item_chunk,
                    "pairs": user_chunk * item_chunk,
                    "reason": "chunk exceeds loaded dataset dimensions",
                }
            )
            continue
        try:
            result = _benchmark_shape(
                model,
                user_groups,
                item_groups,
                user_chunk,
                item_chunk,
                args.warmup,
                args.repeats,
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            skipped.append(
                {
                    "user_chunk": user_chunk,
                    "item_chunk": item_chunk,
                    "pairs": user_chunk * item_chunk,
                    "reason": f"CUDA OOM: {error}",
                }
            )
            continue

        global_calls = scorer_call_count(
            total_users, total_items, user_chunk, item_chunk
        )
        outer_calls = scorer_call_count(
            total_users,
            total_items,
            user_chunk,
            item_chunk,
            outer_users=outer_users,
        )
        total_pairs = total_users * total_items
        result.update(
            {
                "global_chunk_scorer_calls": global_calls,
                "outer_batch_scorer_calls": outer_calls,
                "global_call_extrapolated_seconds": global_calls
                * result["median_seconds"],
                "outer_call_extrapolated_seconds": outer_calls
                * result["median_seconds"],
                "throughput_extrapolated_seconds": total_pairs
                / result["pairs_per_second"],
            }
        )
        results.append(result)
        print(
            f"SWEEP u={user_chunk:3d} i={item_chunk:4d} "
            f"pairs={result['pairs_per_call']:6d} "
            f"median={result['median_seconds'] * 1e3:8.3f}ms "
            f"rate={result['pairs_per_second'] / 1e6:7.2f}Mpair/s "
            f"outer_est={result['outer_call_extrapolated_seconds']:7.2f}s "
            f"peak+={result['peak_incremental_allocated_mib']:7.1f}MiB",
            flush=True,
        )

    if not results:
        raise RuntimeError("no chunk combination completed")
    ranked = sorted(
        results,
        key=lambda item: (
            item["outer_call_extrapolated_seconds"],
            item["peak_incremental_allocated_mib"],
        ),
    )
    report["results"] = results
    report["ranking_metric"] = "outer_call_extrapolated_seconds"
    report["best"] = ranked[: min(5, len(ranked))]
    report["notes"] = [
        "No optimizer step, validation, or test evaluation was run.",
        "Timing excludes the one-time bounded SL membership audit.",
        "Extrapolations cover the exact full user-item scorer matrix including padding item 0.",
        "Evaluator masking, top-k, metric aggregation, and data-loader overhead are excluded.",
    ]
    _write_report(args.output, report)
    print(f"SWEEP_JSON={args.output}", flush=True)


if __name__ == "__main__":
    main()
