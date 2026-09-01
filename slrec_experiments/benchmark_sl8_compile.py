#!/usr/bin/env python3
"""Compare eager and ``torch.compile`` production SL8 scoring kernels.

This is an inference-only microbenchmark.  It creates a deterministic set of
valid SL(8) user/item matrices, then measures the same one-sided Frobenius
K=12 scorer used by ``SL8LHGCN`` at realistic two-dimensional full-sort chunk
shapes.  Dataset construction, training, validation, and testing are not run.

Compilation and warm-up are timed and reported separately, but excluded from
steady-state measurements.  ``fullgraph=True`` plus a Dynamo explanation make
graph breaks explicit.  Eager/compiled score and Top-K agreement are checked
for every shape.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from slrec_experiments.geometry import (
    one_sided_gregory_frobenius_distance_k12,
    to_sl,
)


def parse_positive_int_csv(value: str) -> Tuple[int, ...]:
    """Parse a duplicate-free tuple of positive comma-separated integers."""

    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers: {value!r}") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("at least one positive integer is required")
    return tuple(dict.fromkeys(values))


def requested_shapes(
    user_chunks: Sequence[int], item_chunks: Sequence[int], max_pairs: int
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    """Return deterministic safe shapes and explanations for skipped shapes."""

    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    selected: List[Tuple[int, int]] = []
    skipped: List[Dict[str, Any]] = []
    for users in user_chunks:
        for items in item_chunks:
            pairs = users * items
            if pairs > max_pairs:
                skipped.append(
                    {
                        "users": users,
                        "items": items,
                        "pairs": pairs,
                        "reason": f"pairs exceed max_pairs={max_pairs}",
                    }
                )
            else:
                selected.append((users, items))
    selected.sort(key=lambda shape: (shape[0] * shape[1], shape[0], shape[1]))
    return selected, skipped


def production_sl8_score(
    user_groups: torch.Tensor,
    item_groups: torch.Tensor,
    *,
    jitter: float = 0.0,
) -> torch.Tensor:
    """The production F=1, score-scale=1 fast SL8LHGCN scorer."""

    factor_distance = one_sided_gregory_frobenius_distance_k12(
        user_groups, item_groups, jitter=jitter
    )
    return -factor_distance.squeeze(-1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=3,
        help="physical GPU index assigned through CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--user-chunks", type=parse_positive_int_csv, default=(17,)
    )
    parser.add_argument(
        "--item-chunks", type=parse_positive_int_csv, default=(1024, 2048)
    )
    parser.add_argument("--max-pairs", type=int, default=65_536)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--coord-std", type=float, default=0.05)
    parser.add_argument("--coord-clip", type=float, default=0.75)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _physical_binding(pid: int, requested_gpu: int) -> Dict[str, Any]:
    """Resolve this CUDA process to a physical NVIDIA GPU UUID."""

    gpu_output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    physical_gpus: Dict[int, Dict[str, str]] = {}
    for line in gpu_output.splitlines():
        index, uuid, name = (part.strip() for part in line.split(",", 2))
        physical_gpus[int(index)] = {"uuid": uuid, "name": name}
    if requested_gpu not in physical_gpus:
        raise RuntimeError(f"nvidia-smi does not expose physical GPU {requested_gpu}")

    process_output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    matching_process = None
    for line in process_output.splitlines():
        uuid, process_pid, process_name, used_memory = (
            part.strip() for part in line.split(",", 3)
        )
        if int(process_pid) == pid:
            matching_process = {
                "uuid": uuid,
                "pid": int(process_pid),
                "process_name": process_name,
                "used_memory_mib": int(used_memory),
            }
            break
    if matching_process is None:
        raise RuntimeError(f"CUDA process pid={pid} was not visible in nvidia-smi")
    expected_uuid = physical_gpus[requested_gpu]["uuid"]
    if matching_process["uuid"] != expected_uuid:
        raise RuntimeError(
            f"pid={pid} is on {matching_process['uuid']}, expected {expected_uuid}"
        )
    return {
        "requested_physical_index": requested_gpu,
        "expected_uuid": expected_uuid,
        "physical_name": physical_gpus[requested_gpu]["name"],
        "process": matching_process,
    }


def _time_call(
    function: Callable[[], torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, float, int]:
    _sync(device)
    allocated_before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = function()
    _sync(device)
    seconds = time.perf_counter() - started
    peak_incremental = max(
        torch.cuda.max_memory_allocated(device) - allocated_before, 0
    )
    return result, seconds, peak_incremental


def _steady_measurements(
    eager: Callable[[], torch.Tensor],
    compiled: Callable[[], torch.Tensor],
    device: torch.device,
    repeats: int,
) -> Dict[str, Any]:
    """Alternate measurement order to reduce drift bias."""

    timings = {"eager": [], "compiled": []}
    peaks = {"eager": [], "compiled": []}
    for repeat in range(repeats):
        order = (
            (("eager", eager), ("compiled", compiled))
            if repeat % 2 == 0
            else (("compiled", compiled), ("eager", eager))
        )
        for name, function in order:
            result, seconds, peak = _time_call(function, device)
            timings[name].append(seconds)
            peaks[name].append(peak)
            del result
    output: Dict[str, Any] = {}
    for name in ("eager", "compiled"):
        output[name] = {
            "durations_seconds": timings[name],
            "median_seconds": statistics.median(timings[name]),
            "min_seconds": min(timings[name]),
            "max_seconds": max(timings[name]),
            "peak_incremental_allocated_mib": max(peaks[name]) / 2**20,
        }
    output["speedup_eager_over_compiled"] = (
        output["eager"]["median_seconds"]
        / output["compiled"]["median_seconds"]
    )
    return output


def _graph_report(explanation: Any) -> Dict[str, Any]:
    return {
        "graph_count": int(explanation.graph_count),
        "graph_break_count": int(explanation.graph_break_count),
        "break_reasons": [str(reason) for reason in explanation.break_reasons],
        "ops_per_graph": [
            [str(operation) for operation in graph_ops]
            for graph_ops in explanation.ops_per_graph
        ],
    }


def main() -> None:
    args = _parse_args()
    if args.gpu_id < 0:
        raise ValueError("gpu-id must be non-negative")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if args.topk < 1 or args.coord_std <= 0 or args.coord_clip <= 0:
        raise ValueError("topk, coord-std, and coord-clip must be positive")
    if args.jitter < 0 or args.atol < 0 or args.rtol < 0:
        raise ValueError("jitter and comparison tolerances must be non-negative")
    shapes, skipped = requested_shapes(
        args.user_chunks, args.item_chunks, args.max_pairs
    )
    if not shapes:
        raise ValueError("all requested shapes were skipped")

    # This is set before the first CUDA operation.  The physical mapping is
    # independently verified through this process's nvidia-smi UUID below.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    max_users = max(users for users, _items in shapes)
    max_items = max(items for _users, items in shapes)
    raw_users = torch.randn(
        (max_users, 1, 8, 8), generator=generator, device=device
    ) * float(args.coord_std)
    raw_items = torch.randn(
        (max_items, 1, 8, 8), generator=generator, device=device
    ) * float(args.coord_std)
    user_table = to_sl(raw_users, max_frobenius=args.coord_clip)
    item_table = to_sl(raw_items, max_frobenius=args.coord_clip)
    _sync(device)

    pid = os.getpid()
    binding = _physical_binding(pid, args.gpu_id)
    print(
        f"COMPILE_BENCH pid={pid} physical_gpu={args.gpu_id} "
        f"uuid={binding['process']['uuid']} visible={os.environ['CUDA_VISIBLE_DEVICES']} "
        f"resolved={device}",
        flush=True,
    )

    report: Dict[str, Any] = {
        "schema_version": 1,
        "purpose": "production exact SL8 K12 eager-vs-torch.compile microbenchmark",
        "pid": pid,
        "requested": {
            "user_chunks": list(args.user_chunks),
            "item_chunks": list(args.item_chunks),
            "max_pairs": args.max_pairs,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "topk": args.topk,
            "compile_mode": args.compile_mode,
            "coord_std": args.coord_std,
            "coord_clip": args.coord_clip,
            "jitter": args.jitter,
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_binding": binding,
        },
        "skipped": skipped,
        "results": [],
        "notes": [
            "No dataset, training, validation, or test evaluation was run.",
            "Compilation and warm-up times are excluded from steady-state timing.",
            "The scorer is the same algebraic K12/Frobenius fast path as production SL8LHGCN.",
        ],
    }

    for users, items in shapes:
        current_users = user_table[:users, None, ...]
        current_items = item_table[None, :items, ...]

        def score(
            left: torch.Tensor, right: torch.Tensor
        ) -> torch.Tensor:
            return production_sl8_score(left, right, jitter=args.jitter)

        eager_call = lambda: score(current_users, current_items)
        with torch.inference_mode():
            eager_reference, eager_reference_seconds, _ = _time_call(
                eager_call, device
            )

            torch._dynamo.reset()
            explain_started = time.perf_counter()
            explanation = torch._dynamo.explain(score)(
                current_users, current_items
            )
            _sync(device)
            explain_seconds = time.perf_counter() - explain_started
            graph = _graph_report(explanation)
            if graph["graph_count"] != 1 or graph["graph_break_count"] != 0:
                raise RuntimeError(f"production scorer is not one full graph: {graph}")

            torch._dynamo.reset()
            creation_started = time.perf_counter()
            compiled_score = torch.compile(
                score,
                fullgraph=True,
                dynamic=False,
                mode=args.compile_mode,
            )
            creation_seconds = time.perf_counter() - creation_started
            compiled_call = lambda: compiled_score(current_users, current_items)
            compiled_reference, first_call_seconds, _ = _time_call(
                compiled_call, device
            )

            absolute_difference = (compiled_reference - eager_reference).abs()
            denominator = eager_reference.abs().clamp_min(1e-12)
            max_abs_difference = float(absolute_difference.max().item())
            max_relative_difference = float(
                (absolute_difference / denominator).max().item()
            )
            scores_bitwise_equal = bool(
                torch.equal(compiled_reference, eager_reference)
            )
            scores_close = bool(
                torch.allclose(
                    compiled_reference,
                    eager_reference,
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
            current_topk = min(args.topk, items)
            eager_topk = torch.topk(
                eager_reference, current_topk, dim=-1
            ).indices
            compiled_topk = torch.topk(
                compiled_reference, current_topk, dim=-1
            ).indices
            topk_agreement = bool(torch.equal(eager_topk, compiled_topk))
            if not scores_close or not topk_agreement:
                raise RuntimeError(
                    f"compiled scorer disagreement for {users}x{items}: "
                    f"close={scores_close}, topk={topk_agreement}, "
                    f"max_abs={max_abs_difference}, max_rel={max_relative_difference}"
                )
            del compiled_reference

            warmup_started = time.perf_counter()
            for _ in range(args.warmup):
                warmup_result = compiled_call()
                _sync(device)
                del warmup_result
            warmup_seconds = time.perf_counter() - warmup_started
            steady = _steady_measurements(
                eager_call, compiled_call, device, args.repeats
            )
            del eager_reference

        pairs = users * items
        for name in ("eager", "compiled"):
            steady[name]["pairs_per_second"] = (
                pairs / steady[name]["median_seconds"]
            )
        shape_report = {
            "users": users,
            "items": items,
            "pairs": pairs,
            "graph": graph,
            "explain_seconds": explain_seconds,
            "compile_callable_creation_seconds": creation_seconds,
            "first_compiled_call_including_compilation_seconds": first_call_seconds,
            "compiled_warmup_seconds": warmup_seconds,
            "eager_reference_seconds": eager_reference_seconds,
            "agreement": {
                "scores_bitwise_equal": scores_bitwise_equal,
                "scores_close": scores_close,
                "max_abs_difference": max_abs_difference,
                "max_relative_difference": max_relative_difference,
                "topk": current_topk,
                "topk_indices_equal": topk_agreement,
            },
            "steady_state": steady,
        }
        report["results"].append(shape_report)
        print(
            f"COMPILE_BENCH u={users} i={items} "
            f"eager={steady['eager']['median_seconds'] * 1e3:.4f}ms "
            f"compiled={steady['compiled']['median_seconds'] * 1e3:.4f}ms "
            f"speedup={steady['speedup_eager_over_compiled']:.3f}x "
            f"graphs={graph['graph_count']} breaks={graph['graph_break_count']} "
            f"max_abs={max_abs_difference:.3e} topk_equal={topk_agreement}",
            flush=True,
        )

    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"COMPILE_BENCH_JSON={args.output}", flush=True)


if __name__ == "__main__":
    main()
