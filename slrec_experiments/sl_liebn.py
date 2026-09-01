r"""LieBN-style batch normalisation for ``SL(n)`` representations.

Operator-by-operator transfer of the LieBN recipe (Chen et al., "A Lie Group
Approach to Riemannian Batch Normalization", ICLR 2024) to the special-linear
group, mirroring what the released Hgformer ``LorentzBatchNorm`` does for the
Lorentz model:

1. batch mean -> Cartan--Schouten exponential barycenter (approximated by the
   tangent mean, optionally refined by one bi-invariant fixed-point step);
2. centring -> left translation ``G_i -> mu^{-1} G_i``;
3. rescaling -> in the Lie algebra, ``xi_i -> gamma / (v + eps) * xi_i`` with
   ``xi_i = log(mu^{-1} G_i)`` and dispersion ``v``;
4. biasing -> left translation to ``beta = exp(b)`` with trace-free ``b``.

Two deliberate deviations from LieBN are documented rather than hidden:

* LieBN instantiates geometries with closed-form Frechet means (SPD, SO(n),
  correlation matrices).  ``SL(n)`` is non-compact semisimple and admits no
  bi-invariant Riemannian metric, so no Frechet mean is available; the
  Cartan--Schouten exponential barycenter (Pennec & Arsigny 2012) is the
  invariant substitute, and it is exactly the stationary point of this
  repository's Schatten log semidistance objective.
* The released ``LorentzBatchNorm`` uses the *mean of tangent norms* as its
  dispersion and a fixed (non-learnable) bias at the origin.  Those are the
  defaults here (``dispersion='mean_norm'``, ``learnable_bias=False``) so that
  the SL-versus-Lorentz model comparison stays operator-matched; the faithful
  LieBN choices (``'frechet'`` variance, learnable bias) are explicit options.

**Principal-log domain guards.**  Unlike the Lorentz maps, the SL(n) group
log exists only locally, and the truncated Gregory series returns non-finite
*or arbitrarily large finite* values outside its domain.  Without guards a
single out-of-domain node poisons the batch mean and dispersion and turns the
whole table (and the loss) NaN — observed in practice from ``gcn_layers: 4``
on Amazon-CD, where the per-layer re-expansion to ``gamma`` grows the tail of
``||xi_i||`` with depth until one node crosses the divergence threshold.
Three guards make the operator total:

* statistics (mean and dispersion) are computed only over nodes whose logs
  are finite and no larger than ``max_log_norm``;
* rejected nodes pass through *unchanged* and are counted in the
  diagnostics — they no longer contaminate anyone else;
* after rescaling, output tangents are radially capped at
  ``max_tangent_norm`` (default 3.0, roughly the Gregory ``K=8..12``
  accuracy radius), so every emitted node stays inside the log domain that
  the *next* layer's statistics and aggregation depend on.  Capped nodes are
  counted.  A non-zero rejected/capped count is the signal that the
  representation spread is at the edge of the geometry, exactly like the
  ambient path's repair counts.

Like the released ``LorentzBatchNorm``, this module keeps no running
statistics: it is applied to the full entity table, which is deterministic,
so training and evaluation see identical behaviour.

``normalise_tangent`` is the first-order (identity-anchored) form of the same
operator for coordinate-space propagation: centring by algebra subtraction,
the identical dispersion rescale, and the identical output cap.  It is exact
to ``O(sigma^2)`` BCH terms and keeps the ``tangent_last`` mode's
no-materialisation efficiency.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .geometry import matrix_log_gregory, trace_free


Tensor = torch.Tensor

_MEAN_MODES = ("tangent", "karcher1")
_DISPERSIONS = ("mean_norm", "frechet")


def _combine_validity(
    values_valid: Tensor, mask: Optional[Tensor]
) -> Tensor:
    """Combine per-node/per-factor validity with an optional caller mask."""

    if mask is None:
        return values_valid
    mask = mask.to(device=values_valid.device, dtype=torch.bool)
    if mask.ndim != 1 or mask.shape[0] != values_valid.shape[0]:
        raise ValueError(
            "mask must be one-dimensional over the node axis; got "
            f"{tuple(mask.shape)} for validity {tuple(values_valid.shape)}"
        )
    return values_valid & mask[:, None]


def _valid_mean(values: Tensor, valid: Tensor) -> Tensor:
    """Per-factor mean over dim 0 restricted to ``valid`` entries.

    ``values`` is ``[N, F, ...]``; ``valid`` is ``[N, F]``.  Factors with no
    valid entry fall back to a zero mean, which corresponds to the identity
    element after exponentiation.
    """

    weights = valid.to(values.dtype)
    counts = weights.sum(dim=0).clamp_min(1.0)
    expanded = weights.reshape(
        weights.shape + (1,) * (values.ndim - weights.ndim)
    )
    return (values * expanded).sum(dim=0) / counts.reshape(
        counts.shape + (1,) * (values.ndim - weights.ndim)
    )


def _log_validity(logs: Tensor, max_log_norm: float) -> Tensor:
    """Finite and within the trusted principal-log radius, per ``[N, F]``."""

    finite = torch.isfinite(logs).all(dim=(-2, -1))
    norms = torch.linalg.matrix_norm(
        logs.nan_to_num(), ord="fro", dim=(-2, -1)
    )
    return finite & norms.le(float(max_log_norm))


def _radial_cap(
    tangents: Tensor, max_norm: Optional[float]
) -> Tuple[Tensor, Tensor]:
    """Cap tangent Frobenius norms without changing directions."""

    if max_norm is None or max_norm <= 0:
        return tangents, torch.zeros(
            tangents.shape[:-2], dtype=torch.bool, device=tangents.device
        )
    norms = torch.linalg.matrix_norm(
        tangents, ord="fro", dim=(-2, -1), keepdim=True
    )
    factor = (float(max_norm) / norms.clamp_min(1e-12)).clamp(max=1.0)
    capped = factor.squeeze(-1).squeeze(-1) < 1.0
    return tangents * factor, capped


def _relative_log(
    mean: Tensor, groups: Tensor, *, terms: int, jitter: float
) -> Tensor:
    """Trace-free principal-log approximation of ``mean^{-1} G_i``."""

    relative = torch.linalg.solve(mean.unsqueeze(0), groups)
    return trace_free(matrix_log_gregory(relative, terms=terms, jitter=jitter))


class SLLieBatchNorm(nn.Module):
    """LieBN-style normalisation over a table of ``SL(n)`` matrices.

    Args:
        matrix_dim: Group size ``n``.
        num_factors: Independent ``SL(n)`` factors; statistics, ``gamma`` and
            the optional bias are kept per factor.
        mean_mode: ``'tangent'`` uses ``exp(mean_i log G_i)``; ``'karcher1'``
            (default) refines it with one bi-invariant fixed-point step
            ``mu <- mu exp(mean_i log(mu^{-1} G_i))``.
        dispersion: ``'mean_norm'`` (LorentzBatchNorm-matched, default) uses
            the mean Frobenius norm of the centred logs; ``'frechet'`` uses
            the square root of the mean squared norm (LieBN's variance).
        eps: Additive stabiliser in the rescale denominator.
        log_terms: Gregory truncation for the batched logs.  Centred inputs
            sit near the identity, so a moderate order suffices.
        jitter: Diagonal stabiliser passed to the Gregory log.
        learnable_bias: When true, adds a learnable trace-free bias ``b`` and
            left-translates the output by ``exp(b)``.  Default false, matching
            the released ``LorentzBatchNorm``'s fixed origin.
        gamma_init: Initial value of the learnable scale.
        max_log_norm: Trust radius for the batched Gregory logs; nodes whose
            node-level or centred log is non-finite or exceeds this norm are
            excluded from the statistics and passed through unchanged.
        max_tangent_norm: Radial cap on the rescaled output tangents (the
            trust region that keeps the *next* layer inside the log domain).
            Non-positive disables the cap; the default 3.0 is roughly the
            Gregory accuracy radius measured for ``K=8..12``.
    """

    def __init__(
        self,
        matrix_dim: int,
        num_factors: int = 1,
        *,
        mean_mode: str = "karcher1",
        dispersion: str = "mean_norm",
        eps: float = 1e-5,
        log_terms: int = 8,
        jitter: float = 1e-7,
        learnable_bias: bool = False,
        gamma_init: float = 1.0,
        max_log_norm: float = 25.0,
        max_tangent_norm: float = 3.0,
    ) -> None:
        super().__init__()
        if matrix_dim < 2:
            raise ValueError("matrix_dim must be at least 2")
        if num_factors < 1:
            raise ValueError("num_factors must be positive")
        if mean_mode not in _MEAN_MODES:
            raise ValueError(
                f"mean_mode must be one of {_MEAN_MODES}; got {mean_mode!r}"
            )
        if dispersion not in _DISPERSIONS:
            raise ValueError(
                f"dispersion must be one of {_DISPERSIONS}; got {dispersion!r}"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")
        if log_terms < 1:
            raise ValueError("log_terms must be positive")
        if jitter < 0:
            raise ValueError("jitter must be non-negative")
        if max_log_norm <= 0:
            raise ValueError("max_log_norm must be positive")
        self.matrix_dim = int(matrix_dim)
        self.num_factors = int(num_factors)
        self.mean_mode = mean_mode
        self.dispersion = dispersion
        self.eps = float(eps)
        self.log_terms = int(log_terms)
        self.jitter = float(jitter)
        self.max_log_norm = float(max_log_norm)
        self.max_tangent_norm: Optional[float] = (
            float(max_tangent_norm) if max_tangent_norm > 0 else None
        )
        self.gamma = nn.Parameter(
            torch.full((self.num_factors,), float(gamma_init))
        )
        if learnable_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.num_factors, self.matrix_dim, self.matrix_dim)
            )
        else:
            self.register_parameter("bias", None)

    def _check_input(self, table: Tensor, name: str) -> None:
        if (
            table.ndim != 4
            or table.shape[1] != self.num_factors
            or table.shape[-1] != self.matrix_dim
            or table.shape[-2] != self.matrix_dim
        ):
            raise ValueError(
                f"{name} must have shape [num_nodes, {self.num_factors}, "
                f"{self.matrix_dim}, {self.matrix_dim}]; got "
                f"{tuple(table.shape)}"
            )

    def _rescale_factor(
        self, tangent_norms: Tensor, valid: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.dispersion == "mean_norm":
            spread = _valid_mean(tangent_norms, valid)
        else:
            spread = _valid_mean(
                tangent_norms.square(), valid
            ).clamp_min(0).sqrt()
        return self.gamma / (spread + self.eps), spread

    def forward(
        self, groups: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Normalise a full table of group matrices.

        Args:
            groups: ``[num_nodes, num_factors, n, n]`` matrices in ``SL(n)``.
            mask: Optional boolean node mask; statistics are computed over the
                masked rows only, while every row is transformed.  Rows whose
                logs leave the principal-log trust radius are additionally
                excluded from the statistics and passed through unchanged.
        """

        self._check_input(groups, "groups")
        original_dtype = groups.dtype
        work = (
            groups.float()
            if groups.dtype in (torch.float16, torch.bfloat16)
            else groups
        )

        node_logs = trace_free(
            matrix_log_gregory(work, terms=self.log_terms, jitter=self.jitter)
        )
        node_valid = _log_validity(node_logs, self.max_log_norm)
        stats_valid = _combine_validity(node_valid, mask)
        safe_node_logs = torch.where(
            node_valid[..., None, None], node_logs, torch.zeros_like(node_logs)
        )
        mean = torch.matrix_exp(
            trace_free(_valid_mean(safe_node_logs, stats_valid))
        )
        centred = _relative_log(
            mean, work, terms=self.log_terms, jitter=self.jitter
        )
        centred_valid = node_valid & _log_validity(centred, self.max_log_norm)
        stats_valid = _combine_validity(centred_valid, mask)
        if self.mean_mode == "karcher1":
            safe_centred = torch.where(
                centred_valid[..., None, None],
                centred,
                torch.zeros_like(centred),
            )
            mean = mean @ torch.matrix_exp(
                trace_free(_valid_mean(safe_centred, stats_valid))
            )
            centred = _relative_log(
                mean, work, terms=self.log_terms, jitter=self.jitter
            )
            centred_valid = node_valid & _log_validity(
                centred, self.max_log_norm
            )
            stats_valid = _combine_validity(centred_valid, mask)

        safe_centred = torch.where(
            centred_valid[..., None, None], centred, torch.zeros_like(centred)
        )
        tangent_norms = torch.linalg.matrix_norm(
            safe_centred, ord="fro", dim=(-2, -1)
        )
        factor, spread = self._rescale_factor(tangent_norms, stats_valid)
        scaled = safe_centred * factor[None, :, None, None]
        scaled, capped = _radial_cap(scaled, self.max_tangent_norm)
        output = torch.matrix_exp(scaled)
        if self.bias is not None:
            output = torch.matrix_exp(trace_free(self.bias)).unsqueeze(0) @ output
        # Out-of-domain rows pass through unchanged: their computed log is
        # untrustworthy, so any transform derived from it would be wrong, and
        # zero-replacing them would silently teleport them to the mean.
        output = torch.where(centred_valid[..., None, None], output, work)

        diagnostics = self._diagnostics(
            work, mean, safe_centred, spread, centred_valid, capped
        )
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output, diagnostics

    def normalise_tangent(
        self, coordinates: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """First-order (identity-anchored) form for coordinate propagation."""

        self._check_input(coordinates, "coordinates")
        work = trace_free(coordinates)
        valid = _log_validity(work, self.max_log_norm)
        stats_valid = _combine_validity(valid, mask)
        safe = torch.where(
            valid[..., None, None], work, torch.zeros_like(work)
        )
        centred = safe - _valid_mean(safe, stats_valid).unsqueeze(0)
        tangent_norms = torch.linalg.matrix_norm(
            centred, ord="fro", dim=(-2, -1)
        )
        factor, spread = self._rescale_factor(tangent_norms, stats_valid)
        output = centred * factor[None, :, None, None]
        output, capped = _radial_cap(output, self.max_tangent_norm)
        if self.bias is not None:
            output = output + trace_free(self.bias).unsqueeze(0)
        output = torch.where(valid[..., None, None], output, work)
        diagnostics = {
            "operator": "sl_liebn_tangent",
            "dispersion": [float(v) for v in spread.detach().cpu().reshape(-1)],
            "gamma": [float(v) for v in self.gamma.detach().cpu().reshape(-1)],
            "rejected_logs": int((~valid).sum().item()),
            "capped_outputs": int(capped.sum().item()),
        }
        return output, diagnostics

    def _diagnostics(
        self,
        work: Tensor,
        mean: Tensor,
        centred: Tensor,
        spread: Tensor,
        valid: Tensor,
        capped: Tensor,
    ) -> Dict[str, Any]:
        with torch.no_grad():
            sample = min(16, work.shape[0])
            relative = torch.linalg.solve(
                mean.unsqueeze(0).detach(), work[:sample].detach()
            )
            reconstruction = torch.matrix_exp(centred[:sample].detach())
            residual = torch.linalg.matrix_norm(
                reconstruction - relative, ord="fro", dim=(-2, -1)
            ) / torch.linalg.matrix_norm(
                relative, ord="fro", dim=(-2, -1)
            ).clamp_min(1e-12)
            residual = residual[valid[:sample]]
            sign, log_abs_det = torch.linalg.slogdet(mean.detach().float())
        return {
            "operator": "sl_liebn",
            "mean_mode": self.mean_mode,
            "dispersion": [float(v) for v in spread.detach().cpu().reshape(-1)],
            "gamma": [float(v) for v in self.gamma.detach().cpu().reshape(-1)],
            "max_centred_log_reconstruction_residual": (
                float(residual.max().item()) if residual.numel() else 0.0
            ),
            "mean_nonpositive_determinants": int(sign.le(0).sum().item()),
            "max_abs_mean_log_determinant": float(
                log_abs_det.abs().max().item()
            ),
            "rejected_logs": int((~valid).sum().item()),
            "capped_outputs": int(capped.sum().item()),
        }


__all__ = ["SLLieBatchNorm"]
