r"""Differentiable utilities for the special-linear group :math:`SL(n)`.

The implementation follows the raw-coordinate parameterisation used in the
accompanying SL(n) representation-learning paper:

.. math::

   A(X) = \exp\left(X - \frac{\operatorname{tr}(X)}{n} I\right).

The matrix logarithm uses the Gregory/atanh expansion

.. math::

   \log A = 2 \sum_{k=0}^{K-1} \frac{Z^{2k+1}}{2k+1},
   \qquad Z=(A-I)(A+I)^{-1}.

Compared with the Mercator series around the identity, this transform has a
substantially larger useful convergence region.  It is intended for the
principal-log domain; keeping learned Lie-algebra coordinates in a moderate
radius is still important.  All operations are native PyTorch operations and
therefore differentiable.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch


Tensor = torch.Tensor
SchattenOrder = Union[float, int, str]


def _check_square(matrix: Tensor, name: str = "matrix") -> None:
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"{name} must have shape (..., n, n); got {tuple(matrix.shape)}"
        )


def _eye_like(matrix: Tensor) -> Tensor:
    return torch.eye(
        matrix.shape[-1], dtype=matrix.dtype, device=matrix.device
    ).expand(matrix.shape[:-2] + matrix.shape[-2:])


def _require_numerical_domain(condition: Tensor, message: str) -> None:
    """Fail fast when a batched matrix approximation leaves its safe domain.

    CUDA's asynchronous assertion avoids a host synchronisation in every
    full-sort chunk.  CPU execution raises an ordinary ``RuntimeError`` so
    unit tests and smoke runs receive the same explicit failure semantics.
    """

    condition = condition.reshape(-1).all()
    assert_async = getattr(torch, "_assert_async", None)
    if condition.device.type == "cuda" and assert_async is not None:
        assert_async(condition, message)
    elif not bool(condition):
        raise RuntimeError(message)


def _solve_checked(
    coefficient: Tensor, right_hand_side: Tensor, message: str
) -> Tensor:
    """Batched solve with fail-fast errors and no CUDA host barrier."""

    solve_ex = getattr(torch.linalg, "solve_ex", None)
    if solve_ex is None:  # Compatibility with older supported PyTorch builds.
        return torch.linalg.solve(coefficient, right_hand_side)
    solution, info = solve_ex(
        coefficient, right_hand_side, check_errors=False
    )
    _require_numerical_domain(info.eq(0), message)
    return solution


def trace_free(matrix: Tensor) -> Tensor:
    r"""Project square matrices onto the Lie algebra :math:`\mathfrak{sl}(n)`.

    Args:
        matrix: A tensor with shape ``(..., n, n)``.

    Returns:
        A tensor of the same shape whose last-two-dimensional trace is zero up
        to floating-point precision.
    """

    _check_square(matrix)
    n = matrix.shape[-1]
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return matrix - (trace / n)[..., None, None] * _eye_like(matrix)


def _cap_frobenius(matrix: Tensor, max_frobenius: Optional[float]) -> Tensor:
    """Radially cap matrices without changing their direction."""

    if max_frobenius is None or max_frobenius <= 0:
        return matrix
    norm = torch.linalg.matrix_norm(matrix, ord="fro", dim=(-2, -1), keepdim=True)
    # clamp_min avoids a 0/0 while clamp(max=1) leaves small matrices unchanged.
    factor = (float(max_frobenius) / norm.clamp_min(1e-12)).clamp(max=1.0)
    return matrix * factor


def to_sl(
    raw: Tensor,
    *,
    scale: float = 1.0,
    max_frobenius: Optional[float] = None,
) -> Tensor:
    """Map unconstrained raw coordinates to determinant-one matrices.

    The projection is applied before optional radial clipping, so clipping does
    not reintroduce a trace component.  Half/bfloat16 inputs are exponentiated
    in float32 because matrix exponentiation is poorly supported at low
    precision; the result is cast back to the input dtype.
    """

    _check_square(raw, "raw")
    original_dtype = raw.dtype
    work = raw.float() if raw.dtype in (torch.float16, torch.bfloat16) else raw
    tangent = _cap_frobenius(trace_free(work) * float(scale), max_frobenius)
    result = torch.matrix_exp(tangent)
    return result.to(original_dtype) if result.dtype != original_dtype else result


def matrix_log_gregory(
    matrix: Tensor,
    *,
    terms: int = 12,
    jitter: float = 1e-7,
    tail_tolerance: Optional[float] = None,
) -> Tensor:
    """Approximate the principal matrix logarithm with an atanh series.

    ``matrix`` should have no eigenvalue on the closed negative real axis.  The
    approximation is especially accurate for matrices produced by ``to_sl``
    with moderate coordinate norm.  ``torch.linalg.solve`` is used rather than
    explicitly forming ``(A + I)^{-1}``, improving both numerical behaviour and
    gradient quality.

    Args:
        matrix: Tensor with shape ``(..., n, n)``.
        terms: Number of odd powers in the truncated series.
        jitter: Diagonal stabiliser for the solve.  Set to zero in high-accuracy
            float64 experiments if desired.
        tail_tolerance: Optional relative upper bound on the complete omitted
            Gregory tail.  The bound uses ``q = ||Z^2||_2 < 1`` and the first
            omitted term divided by ``1-q``.  When set, inputs for which the
            truncated series has not converged sufficiently raise instead of
            returning a plausible but incorrect finite matrix log near the
            principal branch cut.
    """

    _check_square(matrix)
    if terms < 1:
        raise ValueError(f"terms must be positive; got {terms}")
    if jitter < 0:
        raise ValueError(f"jitter must be non-negative; got {jitter}")
    if tail_tolerance is not None and tail_tolerance <= 0:
        raise ValueError("tail_tolerance must be positive when enabled")

    original_dtype = matrix.dtype
    work = (
        matrix.float()
        if matrix.dtype in (torch.float16, torch.bfloat16)
        else matrix
    )
    identity = _eye_like(work)
    denominator = work + identity
    if jitter:
        denominator = denominator + float(jitter) * identity

    # Since A-I and A+I are both polynomials in A, left- and right-division
    # coincide algebraically.  Left solve is the stable batched PyTorch form.
    z = _solve_checked(
        denominator,
        work - identity,
        "Gregory matrix-log Cayley denominator is singular",
    )
    z_squared = z @ z
    power = z
    series = z
    for k in range(1, terms):
        power = power @ z_squared
        series = series + power / float(2 * k + 1)
    result = 2.0 * series
    if tail_tolerance is not None:
        first_omitted = (power @ z_squared) / float(2 * terms + 1)
        # For every j >= 0, ||Z^(2K+1) (Z^2)^j||_F is bounded by
        # ||Z^(2K+1)||_F ||Z^2||_2^j.  Ignoring the increasing odd
        # denominators gives a conservative geometric bound on the complete
        # omitted tail.  Guard calculations are diagnostics only, so detach
        # them rather than retaining one SVD per score chunk in autograd.
        with torch.no_grad():
            q = torch.linalg.matrix_norm(
                z_squared.detach(), ord=2, dim=(-2, -1)
            )
            omitted_norm = torch.linalg.matrix_norm(
                first_omitted.detach(), ord="fro", dim=(-2, -1)
            )
            series_norm = torch.linalg.matrix_norm(
                series.detach(), ord="fro", dim=(-2, -1)
            )
            one_minus_q = (1.0 - q).clamp_min(torch.finfo(work.dtype).eps)
            tail_bound_ratio = omitted_norm / (
                one_minus_q
                * series_norm.clamp_min(torch.finfo(work.dtype).eps)
            )
            valid = (
                torch.isfinite(result.detach()).all(dim=(-2, -1))
                & torch.isfinite(q)
                & q.lt(1.0)
                & torch.isfinite(tail_bound_ratio)
                & tail_bound_ratio.le(float(tail_tolerance))
            )
        _require_numerical_domain(
            valid,
            "Gregory matrix-log tail bound did not converge inside the "
            "configured principal-log domain",
        )
    return result.to(original_dtype) if result.dtype != original_dtype else result


def _matrix_log_gregory_k12_from_cayley(cayley: Tensor) -> Tensor:
    r"""Evaluate the 12-term Gregory polynomial with seven matrix products.

    Writing ``t = Z^2``, the truncated logarithm is

    .. math::

       2 Z \sum_{k=0}^{11} \frac{t^k}{2k+1}.

    The direct recurrence in :func:`matrix_log_gregory` needs twelve matrix
    products.  A four-block Paterson--Stockmeyer evaluation of exactly the
    same degree-11 polynomial needs only seven.  This helper is deliberately
    specialised to the production ``K=12`` scorer; the generic implementation
    remains the reference path for every other truncation order.
    """

    _check_square(cayley, "cayley")
    identity = torch.eye(
        cayley.shape[-1], dtype=cayley.dtype, device=cayley.device
    )
    z2 = cayley @ cayley
    z4 = z2 @ z2
    z6 = z4 @ z2

    # Four quadratic blocks in t=Z^2.  The unexpanded identity broadcasts
    # across all leading user/item/factor dimensions without allocating one
    # identity matrix per pair.
    block0 = identity + z2 / 3.0 + z4 / 5.0
    block1 = identity / 7.0 + z2 / 9.0 + z4 / 11.0
    block2 = identity / 13.0 + z2 / 15.0 + z4 / 17.0
    block3 = identity / 19.0 + z2 / 21.0 + z4 / 23.0
    polynomial = block2 + z6 @ block3
    polynomial = block1 + z6 @ polynomial
    polynomial = block0 + z6 @ polynomial
    return 2.0 * (cayley @ polynomial)


def one_sided_gregory_frobenius_distance_k12(
    left: Tensor,
    right: Tensor,
    *,
    jitter: float = 1e-7,
) -> Tensor:
    r"""Fast ``K=12``, one-sided, Schatten-2 SL distance.

    This is an algebraically equivalent evaluation of

    ``||GregoryLog(left^{-1} right)||_F``.

    It avoids first materialising ``left^{-1} right``.  If
    ``R = left^{-1} right``, the Cayley matrix used by the Gregory expansion
    satisfies

    .. math::

       (R + (1+j)I)^{-1}(R-I)
       = (right + (1+j)left)^{-1}(right-left),

    where ``j`` is the configured diagonal jitter.  Consequently only one
    batched solve is required instead of the reference path's relative-matrix
    solve followed by the Gregory solve.  The same 12-term polynomial is then
    evaluated by :func:`_matrix_log_gregory_k12_from_cayley`.

    The operation supports ordinary PyTorch broadcasting and is fully
    differentiable.  As with any reassociation of floating-point operations,
    the last few bits can differ from the two-solve reference although the
    mathematical scorer is unchanged.
    """

    _check_square(left, "left")
    _check_square(right, "right")
    if left.shape[-2:] != right.shape[-2:]:
        raise ValueError(
            "left and right matrices must have the same matrix dimensions; "
            f"got {tuple(left.shape[-2:])} and {tuple(right.shape[-2:])}"
        )
    if jitter < 0:
        raise ValueError(f"jitter must be non-negative; got {jitter}")

    original_dtype = torch.promote_types(left.dtype, right.dtype)
    promote_low_precision = original_dtype in (torch.float16, torch.bfloat16)
    left_work = left.float() if promote_low_precision else left
    right_work = right.float() if promote_low_precision else right
    denominator = right_work + (1.0 + float(jitter)) * left_work
    solve_ex = getattr(torch.linalg, "solve_ex", None)
    if solve_ex is None:  # Compatibility with older supported PyTorch builds.
        cayley = torch.linalg.solve(denominator, right_work - left_work)
    else:
        # ``solve`` synchronises a CUDA device to report factorisation errors.
        # Representations inside the configured principal-log region are
        # nonsingular by construction, so use the same LAPACK/cuSOLVER result
        # without a per-chunk host barrier. Numerical audits independently
        # catch non-finite outputs.
        cayley, solve_info = solve_ex(
            denominator, right_work - left_work, check_errors=False
        )
        solve_ok = solve_info.eq(0).all()
        assert_async = getattr(torch, "_assert_async", None)
        if assert_async is not None:
            # Preserve the reference solver's fail-fast semantics without a
            # host synchronisation for every full-sort chunk.
            assert_async(
                solve_ok,
                "fast SL Gregory solve failed because the Cayley denominator "
                "is singular",
            )
        elif not bool(solve_ok):  # pragma: no cover - old PyTorch fallback.
            raise torch.linalg.LinAlgError(
                "fast SL Gregory solve failed because the Cayley denominator "
                "is singular"
            )
    approximate_log = _matrix_log_gregory_k12_from_cayley(cayley)
    distance = torch.linalg.matrix_norm(
        approximate_log, ord="fro", dim=(-2, -1)
    )
    return (
        distance.to(original_dtype)
        if distance.dtype != original_dtype
        else distance
    )


def matrix_sqrt_denman_beavers(
    matrix: Tensor,
    *,
    iterations: int = 12,
    residual_tolerance: float = 1e-3,
) -> Tensor:
    r"""Batched principal matrix square root via Denman--Beavers iteration.

    .. math::

       Y_{k+1} = \tfrac12 (Y_k + Z_k^{-1}), \qquad
       Z_{k+1} = \tfrac12 (Z_k + Y_k^{-1}),

    with ``Y_0 = A``, ``Z_0 = I``; ``Y_k`` converges quadratically to
    ``A^{1/2}`` for matrices with no eigenvalue on the closed negative real
    axis.  Every operation is a batched solve or addition, so the iteration
    is differentiable and its gradients stay bounded where the square root
    itself is well conditioned.  A fixed iteration count keeps the autograd
    graph static; the mandatory square residual check verifies convergence
    instead of assuming every ``exp``-image pair lies in that domain.
    """

    _check_square(matrix)
    if iterations < 1:
        raise ValueError(f"iterations must be positive; got {iterations}")
    if residual_tolerance <= 0:
        raise ValueError("residual_tolerance must be positive")
    original_dtype = matrix.dtype
    work = (
        matrix.float()
        if matrix.dtype in (torch.float16, torch.bfloat16)
        else matrix
    )
    identity = _eye_like(work)
    y = work
    z = identity
    for _ in range(iterations):
        y_next = 0.5 * (
            y
            + _solve_checked(
                z,
                identity,
                "Denman--Beavers inverse factor became singular",
            )
        )
        z_next = 0.5 * (
            z
            + _solve_checked(
                y,
                identity,
                "Denman--Beavers square-root iterate became singular",
            )
        )
        y, z = y_next, z_next
    residual = torch.linalg.matrix_norm(
        y @ y - work, ord="fro", dim=(-2, -1)
    ) / torch.linalg.matrix_norm(
        work, ord="fro", dim=(-2, -1)
    ).clamp_min(torch.finfo(work.dtype).eps)
    valid = (
        torch.isfinite(y).all(dim=(-2, -1))
        & torch.isfinite(residual)
        & residual.le(float(residual_tolerance))
    )
    _require_numerical_domain(
        valid,
        "Denman--Beavers matrix square root failed its relative residual "
        "check; the relative matrix may be on or too close to the principal "
        "matrix-log branch cut",
    )
    return y.to(original_dtype) if y.dtype != original_dtype else y


def one_sided_sqrt_extended_frobenius_distance(
    left: Tensor,
    right: Tensor,
    *,
    sqrt_steps: int = 1,
    terms: int = 12,
    jitter: float = 1e-7,
    sqrt_iterations: int = 12,
    sqrt_residual_tolerance: float = 1e-3,
    log_tail_tolerance: float = 1e-3,
) -> Tensor:
    r"""One-sided Schatten-2 distance with inverse scaling-and-squaring.

    Evaluates ``||log(left^{-1} right)||_F`` as

    .. math::

       2^k \, \|\mathrm{GregoryLog}\bigl((L^{-1}R)^{1/2^k}\bigr)\|_F,

    which is algebraically identical on the principal branch but doubles the
    reliable Gregory domain with every square-root step.  The plain ``K=12``
    scorer is exact only to relative distance ~3 and *fails violently* beyond
    it (measured in fp32: at distance 5 the value inflates ~2000x and the
    gradient reaches 1e13; at 8 it is Inf/NaN).  One step extends the useful
    domain for well-conditioned pairs to roughly 6--7.  It cannot make the
    principal log continuous across the negative-real-axis branch cut, so the
    square-root residual and a conservative complete Gregory-tail bound are
    checked and an unsafe pair fails fast.  Costs ``2*sqrt_iterations + 1``
    batched solves plus one 8x8 spectral-norm check per pair on top of the
    Gregory polynomial, so enable it where training stability requires it and
    pair counts are bounded.
    """

    if sqrt_steps < 1:
        raise ValueError(f"sqrt_steps must be positive; got {sqrt_steps}")
    if left.device != right.device:
        raise ValueError(
            "left and right matrices must be on the same device; got "
            f"{left.device} and {right.device}"
        )
    original_dtype = torch.promote_types(left.dtype, right.dtype)
    work_dtype = (
        torch.float32
        if original_dtype in (torch.float16, torch.bfloat16)
        else original_dtype
    )
    left_work = left.to(dtype=work_dtype)
    right_work = right.to(dtype=work_dtype)
    relative = relative_matrix(left_work, right_work)
    for _ in range(sqrt_steps):
        relative = matrix_sqrt_denman_beavers(
            relative,
            iterations=sqrt_iterations,
            residual_tolerance=sqrt_residual_tolerance,
        )
    approximate_log = matrix_log_gregory(
        relative,
        terms=terms,
        jitter=jitter,
        tail_tolerance=log_tail_tolerance,
    )
    distance = float(2**sqrt_steps) * torch.linalg.matrix_norm(
        approximate_log, ord="fro", dim=(-2, -1)
    )
    return (
        distance.to(original_dtype)
        if distance.dtype != original_dtype
        else distance
    )


# The shorter name is convenient in model code and public experiments.
matrix_log = matrix_log_gregory


def _normalise_order(p: SchattenOrder) -> float:
    if isinstance(p, str):
        normalised = p.strip().lower()
        if normalised in {"inf", "+inf", "infinity", "+infinity"}:
            return math.inf
        try:
            p = float(normalised)
        except ValueError as exc:
            raise ValueError(f"invalid Schatten order: {p!r}") from exc
    order = float(p)
    if math.isnan(order) or order < 1.0:
        raise ValueError(f"Schatten order must be >= 1; got {p!r}")
    return order


def schatten_norm(matrix: Tensor, p: SchattenOrder = 2) -> Tensor:
    """Return the Schatten-``p`` norm over the final two dimensions.

    ``p=2`` is evaluated as the Frobenius norm and avoids an unnecessary SVD.
    Other orders use singular values.  Leading batch dimensions are preserved.
    """

    _check_square(matrix)
    order = _normalise_order(p)
    if order == 2.0:
        return torch.linalg.matrix_norm(matrix, ord="fro", dim=(-2, -1))

    original_dtype = matrix.dtype
    work = (
        matrix.float()
        if matrix.dtype in (torch.float16, torch.bfloat16)
        else matrix
    )
    singular_values = torch.linalg.svdvals(work)
    if math.isinf(order):
        norm = singular_values.amax(dim=-1)
    else:
        norm = singular_values.pow(order).sum(dim=-1).pow(1.0 / order)
    return norm.to(original_dtype) if norm.dtype != original_dtype else norm


def relative_matrix(left: Tensor, right: Tensor) -> Tensor:
    """Compute ``left^{-1} right`` by a batched linear solve."""

    _check_square(left, "left")
    _check_square(right, "right")
    if left.shape[-2:] != right.shape[-2:]:
        raise ValueError(
            "left and right matrices must have the same matrix dimensions; "
            f"got {tuple(left.shape[-2:])} and {tuple(right.shape[-2:])}"
        )
    return _solve_checked(
        left,
        right,
        "relative SL matrix solve failed because the left matrix is singular",
    )


def sl_semidistance(
    left: Tensor,
    right: Tensor,
    *,
    p: SchattenOrder = 2,
    terms: int = 12,
    jitter: float = 1e-7,
    symmetric: bool = True,
) -> Tensor:
    r"""Compute the intrinsic SL(n) pairwise semidistance.

    .. math::

       D(A,B)=\tfrac12\bigl(
       \|\log(A^{-1}B)\|_{S_p} +
       \|\log(B^{-1}A)\|_{S_p}\bigr).

    ``symmetric=False`` returns the first directed term.  It is useful as a
    faster ablation in the principal-log regime, while the default implements
    the paper's symmetric definition.
    """

    forward_log = matrix_log_gregory(
        relative_matrix(left, right), terms=terms, jitter=jitter
    )
    forward_distance = schatten_norm(forward_log, p=p)
    if not symmetric:
        return forward_distance

    reverse_log = matrix_log_gregory(
        relative_matrix(right, left), terms=terms, jitter=jitter
    )
    reverse_distance = schatten_norm(reverse_log, p=p)
    return 0.5 * (forward_distance + reverse_distance)


__all__ = [
    "matrix_log",
    "matrix_sqrt_denman_beavers",
    "one_sided_sqrt_extended_frobenius_distance",
    "matrix_log_gregory",
    "one_sided_gregory_frobenius_distance_k12",
    "relative_matrix",
    "schatten_norm",
    "sl_semidistance",
    "to_sl",
    "trace_free",
]
