"""Micro-benchmark the exact batched SL(n) pair scorer.

This measures decoder throughput independently of data loading and graph
propagation, and extrapolates the arithmetic time for a full user-item matrix.
The extrapolation is diagnostic only: real RecBole evaluation also performs
masking, top-k selection, transfers, and chunk concatenation.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

try:
    from .geometry import (
        one_sided_gregory_frobenius_distance_k12,
        sl_semidistance,
        to_sl,
    )
except ImportError:
    from geometry import (
        one_sided_gregory_frobenius_distance_k12,
        sl_semidistance,
        to_sl,
    )


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matrix-dim", type=int, default=8)
    parser.add_argument("--num-factors", type=int, default=1)
    parser.add_argument(
        "--factor-aggregation", choices=("l2", "l1", "mean"), default="l2"
    )
    parser.add_argument("--pairs", type=int, default=65_536)
    parser.add_argument("--terms", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--coord-std", type=float, default=0.01)
    parser.add_argument("--coord-clip", type=float, default=1.0)
    parser.add_argument("--one-sided", action="store_true")
    parser.add_argument(
        "--fast-k12",
        action="store_true",
        help="benchmark the algebraically equivalent one-solve K12 path",
    )
    parser.add_argument("--users", type=int, default=66_317)
    parser.add_argument("--items", type=int, default=58_869)
    args = parser.parse_args()

    if (
        args.matrix_dim < 2
        or args.num_factors < 1
        or args.pairs < 1
        or args.terms < 1
    ):
        raise ValueError(
            "matrix-dim >= 2, num-factors >= 1, pairs >= 1, and terms >= 1 "
            "are required"
        )
    if args.fast_k12 and (not args.one_sided or args.terms != 12):
        raise ValueError("--fast-k12 requires --one-sided and --terms=12")
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(2024)
    shape = (
        args.pairs,
        args.num_factors,
        args.matrix_dim,
        args.matrix_dim,
    )
    left = torch.randn(shape, generator=generator, device=device) * args.coord_std
    right = torch.randn(shape, generator=generator, device=device) * args.coord_std
    left = to_sl(left, max_frobenius=args.coord_clip)
    right = to_sl(right, max_frobenius=args.coord_clip)

    def score() -> torch.Tensor:
        if args.fast_k12:
            factor_distances = one_sided_gregory_frobenius_distance_k12(
                left, right
            )
        else:
            factor_distances = sl_semidistance(
                left,
                right,
                p=2,
                terms=args.terms,
                symmetric=not args.one_sided,
            )
        if args.num_factors == 1:
            return factor_distances.squeeze(-1)
        if args.factor_aggregation == "l2":
            return torch.linalg.vector_norm(factor_distances, ord=2, dim=-1)
        if args.factor_aggregation == "l1":
            return factor_distances.sum(dim=-1)
        return factor_distances.mean(dim=-1)

    with torch.inference_mode():
        for _ in range(args.warmup):
            score()
        _synchronise(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = score()
            _synchronise(device)
            durations.append(time.perf_counter() - start)

    median_seconds = sorted(durations)[len(durations) // 2]
    pairs_per_second = args.pairs / median_seconds
    full_pairs = args.users * args.items
    report = {
        "device": str(device),
        "torch_version": torch.__version__,
        "matrix_dim": args.matrix_dim,
        "num_factors": args.num_factors,
        "factor_aggregation": args.factor_aggregation,
        "raw_entity_width": args.num_factors * args.matrix_dim**2,
        "intrinsic_entity_dim": args.num_factors * (args.matrix_dim**2 - 1),
        "cubic_cost_proxy_per_pair": args.num_factors * args.matrix_dim**3,
        "log_terms": args.terms,
        "symmetric": not args.one_sided,
        "fast_k12": args.fast_k12,
        "batch_pairs": args.pairs,
        "median_seconds": median_seconds,
        "pairs_per_second": pairs_per_second,
        "finite": bool(torch.isfinite(result).all()),
        "estimated_full_pairs": full_pairs,
        "estimated_decoder_hours": full_pairs / pairs_per_second / 3600.0,
    }
    if device.type == "cuda":
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
