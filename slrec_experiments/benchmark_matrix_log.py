#!/usr/bin/env python3
"""Benchmark direct eigendecomposition and Gregory matrix logarithms.

This script targets the small real matrices used by SLRec (normally n=4 or
n=8).  It measures batched CUDA forward speed, reconstruction error, agreement
with SciPy's CPU ``logm`` on a small reference sample, imaginary residuals, and
whether a simple backward pass is finite.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.linalg
import torch

try:
    from .geometry import matrix_log_gregory, trace_free
except ImportError:
    from geometry import matrix_log_gregory, trace_free


def eig_matrix_log(matrix: torch.Tensor) -> torch.Tensor:
    """Principal log through a complex eigendecomposition.

    This is mathematically direct for diagonalizable matrices whose spectrum
    avoids the closed negative real axis.  It is a benchmark candidate rather
    than a robust replacement for a Schur-Parlett ``logm``.
    """

    eigenvalues, eigenvectors = torch.linalg.eig(matrix)
    scaled_vectors = eigenvectors * torch.log(eigenvalues).unsqueeze(-2)
    return scaled_vectors @ torch.linalg.inv(eigenvectors)


def make_relative_matrices(
    batch: int,
    n: int,
    radius: float,
    device: torch.device,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    left = trace_free(torch.randn(batch, n, n, device=device, generator=generator))
    right = trace_free(torch.randn(batch, n, n, device=device, generator=generator))
    left = left / torch.linalg.matrix_norm(left, ord="fro", dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    right = right / torch.linalg.matrix_norm(right, ord="fro", dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    # Vary norms inside the requested cap instead of testing one artificial shell.
    left_scale = torch.rand(batch, 1, 1, device=device, generator=generator)
    right_scale = torch.rand(batch, 1, 1, device=device, generator=generator)
    left_group = torch.matrix_exp(left * left_scale * radius)
    right_group = torch.matrix_exp(right * right_scale * radius)
    return torch.linalg.solve(left_group, right_group)


def cuda_time_ms(
    function: Callable[[torch.Tensor], torch.Tensor],
    matrix: torch.Tensor,
    repeats: int,
) -> float:
    with torch.no_grad():
        for _ in range(3):
            function(matrix)
        torch.cuda.synchronize(matrix.device)
        start = time.perf_counter()
        for _ in range(repeats):
            function(matrix)
        torch.cuda.synchronize(matrix.device)
    return 1000.0 * (time.perf_counter() - start) / repeats


def reconstruction_error(matrix: torch.Tensor, logarithm: torch.Tensor) -> float:
    reconstructed = torch.matrix_exp(logarithm)
    numerator = torch.linalg.matrix_norm(
        reconstructed - matrix.to(reconstructed.dtype), ord="fro", dim=(-2, -1)
    )
    denominator = torch.linalg.matrix_norm(
        matrix.to(reconstructed.dtype), ord="fro", dim=(-2, -1)
    ).clamp_min(1e-12)
    return float((numerator / denominator).median().item())


def scipy_reference(matrix: torch.Tensor, sample: int) -> np.ndarray:
    arrays = matrix[:sample].detach().cpu().double().numpy()
    return np.stack([scipy.linalg.logm(value) for value in arrays], axis=0)


def relative_reference_error(estimate: torch.Tensor, reference: np.ndarray) -> float:
    estimate_array = estimate[: len(reference)].detach().cpu().numpy()
    numerator = np.linalg.norm(estimate_array - reference, axis=(-2, -1))
    denominator = np.maximum(np.linalg.norm(reference, axis=(-2, -1)), 1e-12)
    return float(np.median(numerator / denominator))


def backward_check(
    function: Callable[[torch.Tensor], torch.Tensor], matrix: torch.Tensor
) -> dict[str, object]:
    leaf = matrix[: min(256, matrix.shape[0])].detach().clone().requires_grad_(True)
    try:
        result = function(leaf)
        loss = result.real.square().mean() + result.imag.square().mean() if result.is_complex() else result.square().mean()
        loss.backward()
        gradient = leaf.grad
        assert gradient is not None
        return {
            "ok": bool(torch.isfinite(gradient).all().item()),
            "median_grad_norm": float(
                torch.linalg.matrix_norm(gradient, ord="fro", dim=(-2, -1)).median().item()
            ),
            "max_grad_norm": float(
                torch.linalg.matrix_norm(gradient, ord="fro", dim=(-2, -1)).max().item()
            ),
        }
    except Exception as error:  # Report unsupported/unstable backward explicitly.
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


def benchmark_case(
    n: int,
    radius: float,
    batch: int,
    repeats: int,
    terms: int,
    reference_sample: int,
    device: torch.device,
) -> dict[str, object]:
    matrix = make_relative_matrices(batch, n, radius, device, seed=2024 + n)
    methods: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        f"gregory_{terms}": lambda value: matrix_log_gregory(
            value, terms=terms, jitter=1e-7
        ),
        "direct_eig": eig_matrix_log,
    }
    reference_start = time.perf_counter()
    reference = scipy_reference(matrix, min(reference_sample, batch))
    reference_ms_each = (
        1000.0 * (time.perf_counter() - reference_start) / len(reference)
    )

    output: dict[str, object] = {
        "n": n,
        "radius": radius,
        "batch": batch,
        "scipy_logm_ms_each_cpu": reference_ms_each,
        "methods": {},
    }
    for name, method in methods.items():
        with torch.no_grad():
            estimate = method(matrix)
        elapsed_ms = cuda_time_ms(method, matrix, repeats)
        imaginary_ratio = 0.0
        reconstruction_log = estimate
        if estimate.is_complex():
            imaginary_ratio = float(
                (
                    torch.linalg.matrix_norm(estimate.imag, ord="fro", dim=(-2, -1))
                    / torch.linalg.matrix_norm(estimate.real, ord="fro", dim=(-2, -1)).clamp_min(1e-12)
                ).median().item()
            )
        method_output = {
            "cuda_ms_per_batch": elapsed_ms,
            "cuda_us_per_matrix": 1000.0 * elapsed_ms / batch,
            "median_reconstruction_relative_error": reconstruction_error(
                matrix, reconstruction_log
            ),
            "median_relative_error_vs_scipy": relative_reference_error(
                estimate, reference
            ),
            "median_imaginary_to_real_ratio": imaginary_ratio,
            "finite": bool(torch.isfinite(estimate).all().item()),
            "backward": backward_check(method, matrix),
        }
        output["methods"][name] = method_output
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dims", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--radii", nargs="+", type=float, default=[0.05, 0.5, 1.0])
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--terms", type=int, default=12)
    parser.add_argument("--reference-sample", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    device = torch.device("cuda")
    cases = []
    for n in args.matrix_dims:
        for radius in args.radii:
            print(f"benchmark n={n} radius={radius}", flush=True)
            cases.append(
                benchmark_case(
                    n,
                    radius,
                    args.batch,
                    args.repeats,
                    args.terms,
                    args.reference_sample,
                    device,
                )
            )
    payload = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "cases": cases,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
