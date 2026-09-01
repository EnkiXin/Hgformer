r"""Special-linear aggregation primitives for the LHGCN ablation.

The Hgformer paper's LHGCN layer takes a positive weighted sum in the
ambient Lorentz space and radially projects that sum back to the
hyperboloid.  ``SL(n)`` has no analogous closed-form Frechet mean for the
Schatten semidistance, so :func:`project_ambient_to_sl` implements an
*extrinsic retraction*, not an exact intrinsic centroid:

.. math::

   M_i = \sum_j \widetilde A_{ij} G_j, \qquad
   G'_i = M_i / \det(M_i)^{1/n}.

The formula is valid on the positive-determinant component.  A weighted
ambient sum can leave that component even when every input is in ``SL(n)``.
For a negative determinant we explicitly repair the orientation by reflecting
the last column before determinant normalisation.  A singular or non-finite
sum cannot be repaired this way; only those rows fall back to the exponential
of a trace-free matrix.  Both events are returned as diagnostics so an
experiment cannot silently hide an unstable aggregation regime.

These functions are deliberately separate from ``geometry.py``: the latter
implements the representation geometry from the SL(n) paper, whereas this
module is a new graph-aggregation hypothesis that needs an ablation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint

from .geometry import matrix_log_gregory, to_sl, trace_free


Tensor = torch.Tensor


@dataclass(frozen=True)
class SLProjectionDiagnostics:
    """Counts produced by one ambient projection step."""

    total: int
    orientation_repairs: int
    singular_fallbacks: int
    active_total: int = 0
    inactive_total: int = 0
    active_singular_fallbacks: int = 0
    inactive_singular_fallbacks: int = 0
    input_positive_determinants: int = 0
    input_negative_determinants: int = 0
    input_singular_or_nonfinite: int = 0
    input_membership_violations: int = 0
    output_nonpositive_determinants: int = 0
    output_nonfinite_log_determinants: int = 0
    output_membership_violations: int = 0
    max_abs_input_log_determinant: float = 0.0
    max_abs_output_log_determinant: float = 0.0

    @property
    def orientation_repair_rate(self) -> float:
        return self.orientation_repairs / max(1, self.total)

    @property
    def singular_fallback_rate(self) -> float:
        return self.singular_fallbacks / max(1, self.total)


def _check_square(matrix: Tensor, name: str) -> None:
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"{name} must have shape (..., n, n); got {tuple(matrix.shape)}"
        )


def project_ambient_to_sl(
    ambient: Tensor,
    *,
    fallback_clip: float = 1.0,
    collect_diagnostics: bool = True,
    active_mask: Optional[Tensor] = None,
    membership_tolerance: float = 1e-4,
    strict_membership: bool = False,
) -> Tuple[Tensor, SLProjectionDiagnostics]:
    """Retract an ambient matrix batch to ``SL(n)``.

    Positive-determinant matrices are divided by the positive real ``n``-th
    root of their determinant.  Negative-determinant matrices first have their
    last column reflected, making the determinant positive without discarding
    the rest of the aggregate.  Singular/non-finite matrices use
    ``exp(trace_free(M))`` with a Frobenius cap as an explicit last-resort
    fallback.

    The orientation repair is necessarily discontinuous at ``det(M)=0``:
    positive- and negative-determinant matrices are different connected
    components of ``GL(n, R)``.  Its frequency is therefore an important
    stability diagnostic rather than an implementation detail.
    """

    _check_square(ambient, "ambient")
    if fallback_clip <= 0:
        raise ValueError("fallback_clip must be positive")
    if membership_tolerance <= 0:
        raise ValueError("membership_tolerance must be positive")

    original_dtype = ambient.dtype
    work = (
        ambient.float()
        if ambient.dtype in (torch.float16, torch.bfloat16)
        else ambient
    )
    n = work.shape[-1]
    sign, log_abs_det = torch.linalg.slogdet(work)
    finite_matrix = torch.isfinite(work).all(dim=(-2, -1))
    finite_log_det = torch.isfinite(log_abs_det)
    nonsingular = sign.ne(0) & finite_matrix & finite_log_det
    negative = sign.lt(0) & nonsingular
    if active_mask is None:
        active = torch.ones_like(sign, dtype=torch.bool)
    else:
        active = active_mask.to(device=sign.device, dtype=torch.bool)
        while active.ndim < sign.ndim:
            active = active.unsqueeze(-1)
        try:
            active = torch.broadcast_to(active, sign.shape)
        except RuntimeError as error:
            raise ValueError(
                "active_mask cannot broadcast to the ambient matrix batch: "
                f"{tuple(active_mask.shape)} -> {tuple(sign.shape)}"
            ) from error

    # Multiplication on the right by diag(1, ..., 1, -1) reflects the last
    # column and changes determinant sign.  Broadcasting column multipliers is
    # cheaper than constructing one reflection matrix per entity.
    column_multiplier = torch.ones_like(work[..., 0, :])
    column_multiplier[..., -1] = torch.where(
        negative,
        -torch.ones_like(sign),
        torch.ones_like(sign),
    )
    oriented = work * column_multiplier.unsqueeze(-2)

    # Avoid NaN/Inf in the unselected branch of the later indexed repair.  A
    # finite nonsingular float matrix encountered in this model is comfortably
    # inside this exponent range; out-of-range rows are routed to the fallback.
    normaliser_log = -log_abs_det / float(n)
    # Use Python's float64 math here. Constructing a default float32 tensor
    # from ``torch.finfo(float64).max`` first overflows to inf and silently
    # disables the representability gate in float64 audits.
    max_log = 0.9 * math.log(torch.finfo(work.dtype).max)
    representable = nonsingular & normaliser_log.abs().le(max_log)
    safe_log = torch.nan_to_num(normaliser_log, nan=0.0, posinf=0.0, neginf=0.0)
    safe_log = safe_log.clamp(min=-max_log, max=max_log)
    projected = oriented * safe_log.exp()[..., None, None]

    invalid = ~representable
    # Do not evaluate matrix_exp for every entity: on a healthy graph only the
    # reserved id-0 rows are usually singular.  Indexed assignment keeps the
    # fallback differentiable for the affected rows.
    if bool(invalid.any()):
        projected = projected.clone()
        fallback_input = torch.nan_to_num(
            work[invalid], nan=0.0, posinf=1.0, neginf=-1.0
        )
        projected[invalid] = to_sl(
            fallback_input, max_frobenius=float(fallback_clip)
        )

    if projected.dtype != original_dtype:
        projected = projected.to(original_dtype)

    # This second slogdet is intentionally not inferred from the normalising
    # scalar: it checks the matrices that are actually returned after all
    # floating-point operations (including a possible low-precision cast).
    projected_for_check = projected.detach()
    output_work = (
        projected_for_check.float()
        if projected.dtype in (torch.float16, torch.bfloat16)
        else projected_for_check
    )
    output_sign, output_log_abs_det = torch.linalg.slogdet(output_work)
    output_finite = torch.isfinite(output_log_abs_det) & torch.isfinite(
        output_work
    ).all(dim=(-2, -1))
    output_violation = (
        output_sign.le(0)
        | ~output_finite
        | output_log_abs_det.abs().gt(float(membership_tolerance))
    )
    if collect_diagnostics:
        input_violation = (
            sign.ne(1)
            | ~finite_matrix
            | ~finite_log_det
            | log_abs_det.abs().gt(float(membership_tolerance))
        )
        valid_input_logs = nonsingular & finite_log_det
        valid_output_logs = output_sign.gt(0) & output_finite
        diagnostic_log_abs_det = log_abs_det.detach()
        # Transfer all scalar diagnostics to the host at once.  The former
        # sequence of ``.item()`` calls introduced more than a dozen GPU
        # synchronisation barriers per graph layer and per mini-batch.
        scalar_statistics = torch.stack(
            (
                negative.sum().to(torch.float64),
                invalid.sum().to(torch.float64),
                active.sum().to(torch.float64),
                (~active).sum().to(torch.float64),
                (invalid & active).sum().to(torch.float64),
                (invalid & ~active).sum().to(torch.float64),
                (sign.gt(0) & nonsingular).sum().to(torch.float64),
                negative.sum().to(torch.float64),
                (~nonsingular).sum().to(torch.float64),
                input_violation.sum().to(torch.float64),
                output_sign.le(0).sum().to(torch.float64),
                (~output_finite).sum().to(torch.float64),
                output_violation.sum().to(torch.float64),
                valid_input_logs.sum().to(torch.float64),
                torch.where(
                    valid_input_logs,
                    diagnostic_log_abs_det.abs(),
                    torch.zeros_like(diagnostic_log_abs_det),
                )
                .amax()
                .to(torch.float64),
                valid_output_logs.sum().to(torch.float64),
                torch.where(
                    valid_output_logs,
                    output_log_abs_det.abs(),
                    torch.zeros_like(output_log_abs_det),
                )
                .amax()
                .to(torch.float64),
            )
        ).detach().cpu().tolist()
        (
            orientation_repairs,
            singular_fallbacks,
            active_total,
            inactive_total,
            active_singular_fallbacks,
            inactive_singular_fallbacks,
            input_positive_determinants,
            input_negative_determinants,
            input_singular_or_nonfinite,
            input_membership_violations,
            output_nonpositive_determinants,
            output_nonfinite_log_determinants,
            output_membership_violations,
            valid_input_log_count,
            max_abs_input_log_determinant,
            valid_output_log_count,
            max_abs_output_log_determinant,
        ) = scalar_statistics
        if strict_membership and output_membership_violations:
            raise RuntimeError(
                "determinant retraction returned matrices outside SL(n): "
                f"{int(output_membership_violations)}/"
                f"{output_violation.numel()} violate sign=+1 and "
                f"|log|det||<={membership_tolerance:g}"
            )
        diagnostics = SLProjectionDiagnostics(
            total=sign.numel(),
            orientation_repairs=int(orientation_repairs),
            singular_fallbacks=int(singular_fallbacks),
            active_total=int(active_total),
            inactive_total=int(inactive_total),
            active_singular_fallbacks=int(active_singular_fallbacks),
            inactive_singular_fallbacks=int(inactive_singular_fallbacks),
            input_positive_determinants=int(input_positive_determinants),
            input_negative_determinants=int(input_negative_determinants),
            input_singular_or_nonfinite=int(input_singular_or_nonfinite),
            input_membership_violations=int(input_membership_violations),
            output_nonpositive_determinants=int(
                output_nonpositive_determinants
            ),
            output_nonfinite_log_determinants=int(
                output_nonfinite_log_determinants
            ),
            output_membership_violations=int(output_membership_violations),
            max_abs_input_log_determinant=(
                float(max_abs_input_log_determinant)
                if valid_input_log_count
                else float("inf")
            ),
            max_abs_output_log_determinant=(
                float(max_abs_output_log_determinant)
                if valid_output_log_count
                else float("inf")
            ),
        )
    else:
        if strict_membership and bool(output_violation.any()):
            raise RuntimeError(
                "determinant retraction returned matrices outside SL(n): "
                f"{int(output_violation.sum().item())}/"
                f"{output_violation.numel()} violate sign=+1 and "
                f"|log|det||<={membership_tolerance:g}"
            )
        diagnostics = SLProjectionDiagnostics(
            total=sign.numel(), orientation_repairs=0, singular_fallbacks=0
        )
    return projected, diagnostics


def ambient_sl_centroid_step(
    group: Tensor,
    adjacency: Tensor,
    *,
    fallback_clip: float = 1.0,
    collect_diagnostics: bool = True,
    active_mask: Optional[Tensor] = None,
    membership_tolerance: float = 1e-4,
    strict_membership: bool = False,
) -> Tuple[Tensor, SLProjectionDiagnostics]:
    """Apply one sparse ambient aggregation and determinant retraction.

    ``group`` is shaped ``[num_nodes, ..., n, n]``.  All dimensions after the
    node dimension are flattened for one native ``torch.sparse.mm`` call, then
    restored before projection.  The adjacency is expected to contain the
    exact propagation weights selected by the experiment.
    """

    _check_square(group, "group")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square rank-two tensor")
    if adjacency.shape[0] != group.shape[0]:
        raise ValueError(
            "adjacency node count does not match group: "
            f"{adjacency.shape[0]} != {group.shape[0]}"
        )

    flat_group = group.reshape(group.shape[0], -1)
    if adjacency.is_sparse:
        ambient = torch.sparse.mm(adjacency, flat_group)
    else:
        ambient = adjacency @ flat_group
    ambient = ambient.reshape_as(group)
    return project_ambient_to_sl(
        ambient,
        fallback_clip=fallback_clip,
        collect_diagnostics=collect_diagnostics,
        active_mask=active_mask,
        membership_tolerance=membership_tolerance,
        strict_membership=strict_membership,
    )


@dataclass(frozen=True)
class KarcherAggregationDiagnostics:
    """Counts produced by one exponential-barycenter aggregation step.

    ``nonfinite_node_logs`` and ``nonfinite_edge_logs`` count principal-log
    failures (inputs outside the Gregory log's domain).  The affected terms
    are zeroed — the node contributes the identity, the edge falls back to
    the seed — so the step stays finite and inside ``SL(n)``, but a non-zero
    count means the representation spread has left the regime in which this
    aggregation is meaningful, exactly like the ambient path's repair counts.
    """

    edges: int
    chunks: int
    correction: bool
    max_seed_tangent_norm: float
    max_correction_tangent_norm: float
    nonfinite_node_logs: int = 0
    nonfinite_edge_logs: int = 0


def row_normalise_sparse(adjacency: Tensor) -> Tensor:
    """Return the row-stochastic version of a sparse adjacency.

    The exponential barycenter is a *weighted mean*, so its weights must form
    a convex combination per node.  The symmetric ``D^-1/2 A D^-1/2`` weights
    do not sum to one per row; on the Lorentz side that global scale is
    removed by the centroid renormalisation, and the determinant retraction
    removes it too, but a barycenter (and the tangent control) would inherit
    it as a per-layer contraction toward the identity.  Rows without entries
    (reserved/isolated ids) stay empty and aggregate to the identity.
    """

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square rank-two tensor")
    if not adjacency.is_sparse:
        row_sums = adjacency.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return adjacency / row_sums
    coalesced = adjacency.coalesce()
    indices = coalesced.indices()
    values = coalesced.values()
    row_sums = torch.zeros(
        coalesced.shape[0], dtype=values.dtype, device=values.device
    )
    row_sums.index_add_(0, indices[0], values)
    normalised = values / row_sums.clamp_min(1e-12)[indices[0]]
    return torch.sparse_coo_tensor(
        indices, normalised, size=coalesced.shape, dtype=values.dtype
    ).coalesce()


def _weighted_relative_log_chunk(
    seed_rows: Tensor,
    group_cols: Tensor,
    weights: Tensor,
    log_terms: int,
    jitter: float,
) -> Tensor:
    relative = torch.linalg.solve(seed_rows, group_cols)
    logs = matrix_log_gregory(relative, terms=log_terms, jitter=jitter)
    shape = (-1,) + (1,) * (logs.ndim - 1)
    return logs * weights.reshape(shape)


def karcher_sl_centroid_step(
    group: Tensor,
    row_normalised_adjacency: Tensor,
    *,
    log_terms: int = 6,
    jitter: float = 1e-7,
    correction: bool = True,
    edge_chunk: int = 262144,
    use_checkpoint: bool = True,
    max_log_norm: float = 25.0,
) -> Tuple[Tensor, KarcherAggregationDiagnostics]:
    r"""One Cartan--Schouten exponential-barycenter aggregation step.

    For node ``v`` with convex weights ``w_vw`` this computes the one-step
    truncation of the bi-invariant mean iteration (Pennec & Arsigny 2012),
    seeded by the tangent mean:

    .. math::

       m_v = \exp\Bigl(\sum_w w_{vw} \log G_w\Bigr), \qquad
       G'_v = m_v \exp\Bigl(\sum_w w_{vw} \log(m_v^{-1} G_w)\Bigr).

    The output is in ``SL(n)`` by construction (trace-free exponentials
    composed with group products), so no determinant retraction, orientation
    repair, or singular fallback exists on this path.  The correction term
    costs one ``n x n`` log per edge; ``edge_chunk`` bounds the working set
    and ``use_checkpoint`` recomputes the chunk forward during backward
    instead of storing Gregory intermediates.  ``correction=False`` returns
    the seed, which is exactly the row-normalised ``tangent`` aggregation
    materialised in the group.

    All logs are principal-branch Gregory approximations: the step is only
    meaningful while neighbours stay inside the principal-log domain of their
    barycenter, which layer normalisation is expected to maintain.  Outside
    that domain the truncated series produces non-finite *or arbitrarily
    large finite* values, so any node/edge log that is non-finite or exceeds
    ``max_log_norm`` in Frobenius norm is zeroed (the node contributes the
    identity; the edge falls back to the seed) and counted in the
    diagnostics rather than silently propagated.
    """

    _check_square(group, "group")
    if row_normalised_adjacency.ndim != 2 or (
        row_normalised_adjacency.shape[0] != row_normalised_adjacency.shape[1]
    ):
        raise ValueError("adjacency must be a square rank-two tensor")
    if row_normalised_adjacency.shape[0] != group.shape[0]:
        raise ValueError(
            "adjacency node count does not match group: "
            f"{row_normalised_adjacency.shape[0]} != {group.shape[0]}"
        )
    if log_terms < 1:
        raise ValueError("log_terms must be positive")
    if edge_chunk < 1:
        raise ValueError("edge_chunk must be positive")
    if max_log_norm <= 0:
        raise ValueError("max_log_norm must be positive")

    original_dtype = group.dtype
    work = (
        group.float()
        if group.dtype in (torch.float16, torch.bfloat16)
        else group
    )

    node_logs = trace_free(
        matrix_log_gregory(work, terms=log_terms, jitter=jitter)
    )
    finite_nodes = torch.isfinite(node_logs).all(dim=(-2, -1)) & (
        torch.linalg.matrix_norm(
            node_logs.nan_to_num(), ord="fro", dim=(-2, -1)
        ).le(float(max_log_norm))
    )
    nonfinite_node_logs = int((~finite_nodes).sum().item())
    if nonfinite_node_logs:
        node_logs = torch.where(
            finite_nodes[..., None, None], node_logs, torch.zeros_like(node_logs)
        )
    flat_logs = node_logs.reshape(node_logs.shape[0], -1)
    if row_normalised_adjacency.is_sparse:
        seed_algebra = torch.sparse.mm(row_normalised_adjacency, flat_logs)
    else:
        seed_algebra = row_normalised_adjacency @ flat_logs
    seed_algebra = trace_free(seed_algebra.reshape_as(node_logs))
    seed = torch.matrix_exp(seed_algebra)
    max_seed_norm = float(
        torch.linalg.matrix_norm(seed_algebra.detach(), ord="fro", dim=(-2, -1))
        .max()
        .item()
    )

    if not correction:
        output = seed
        diagnostics = KarcherAggregationDiagnostics(
            edges=0,
            chunks=0,
            correction=False,
            max_seed_tangent_norm=max_seed_norm,
            max_correction_tangent_norm=0.0,
            nonfinite_node_logs=nonfinite_node_logs,
        )
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output, diagnostics

    coalesced = (
        row_normalised_adjacency.coalesce()
        if row_normalised_adjacency.is_sparse
        else row_normalised_adjacency.to_sparse().coalesce()
    )
    rows = coalesced.indices()[0]
    cols = coalesced.indices()[1]
    weights = coalesced.values().to(work.dtype)
    edge_count = int(rows.numel())

    correction_algebra = torch.zeros_like(node_logs)
    chunks = 0
    nonfinite_edge_logs = 0
    needs_grad = torch.is_grad_enabled() and work.requires_grad
    for start in range(0, edge_count, edge_chunk):
        stop = min(start + edge_chunk, edge_count)
        chunk_rows = rows[start:stop]
        seed_sel = seed[chunk_rows]
        group_sel = work[cols[start:stop]]
        weight_sel = weights[start:stop]
        if use_checkpoint and needs_grad:
            try:
                contribution = torch.utils.checkpoint.checkpoint(
                    _weighted_relative_log_chunk,
                    seed_sel,
                    group_sel,
                    weight_sel,
                    log_terms,
                    jitter,
                    use_reentrant=False,
                )
            except TypeError:  # torch builds without ``use_reentrant``
                contribution = torch.utils.checkpoint.checkpoint(
                    _weighted_relative_log_chunk,
                    seed_sel,
                    group_sel,
                    weight_sel,
                    log_terms,
                    jitter,
                )
        else:
            contribution = _weighted_relative_log_chunk(
                seed_sel, group_sel, weight_sel, log_terms, jitter
            )
        finite_edges = torch.isfinite(contribution).all(dim=(-2, -1)) & (
            torch.linalg.matrix_norm(
                contribution.nan_to_num(), ord="fro", dim=(-2, -1)
            ).le(float(max_log_norm))
        )
        chunk_nonfinite = int((~finite_edges).sum().item())
        if chunk_nonfinite:
            nonfinite_edge_logs += chunk_nonfinite
            contribution = torch.where(
                finite_edges[..., None, None],
                contribution,
                torch.zeros_like(contribution),
            )
        correction_algebra = correction_algebra.index_add(
            0, chunk_rows, contribution
        )
        chunks += 1
    correction_algebra = trace_free(correction_algebra)
    output = seed @ torch.matrix_exp(correction_algebra)
    max_correction_norm = float(
        torch.linalg.matrix_norm(
            correction_algebra.detach(), ord="fro", dim=(-2, -1)
        )
        .max()
        .item()
    )

    diagnostics = KarcherAggregationDiagnostics(
        edges=edge_count,
        chunks=chunks,
        correction=True,
        max_seed_tangent_norm=max_seed_norm,
        max_correction_tangent_norm=max_correction_norm,
        nonfinite_node_logs=nonfinite_node_logs,
        nonfinite_edge_logs=nonfinite_edge_logs,
    )
    if output.dtype != original_dtype:
        output = output.to(original_dtype)
    return output, diagnostics


__all__ = [
    "KarcherAggregationDiagnostics",
    "SLProjectionDiagnostics",
    "ambient_sl_centroid_step",
    "karcher_sl_centroid_step",
    "project_ambient_to_sl",
    "row_normalise_sparse",
]
