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
  repository's Schatten log semidistance objective.  It is only defined
  locally; the reconstruction diagnostics below make principal-log-domain
  violations observable instead of silent.
* The released ``LorentzBatchNorm`` uses the *mean of tangent norms* as its
  dispersion and a fixed (non-learnable) bias at the origin.  Those are the
  defaults here (``dispersion='mean_norm'``, ``learnable_bias=False``) so that
  the SL-versus-Lorentz model comparison stays operator-matched; the faithful
  LieBN choices (``'frechet'`` variance, learnable bias) are explicit options.

Like the released ``LorentzBatchNorm``, this module keeps no running
statistics: it is applied to the full entity table, which is deterministic,
so training and evaluation see identical behaviour.

``normalise_tangent`` is the first-order (identity-anchored) form of the same
operator for coordinate-space propagation: centring by algebra subtraction and
the identical dispersion rescale.  It is exact to ``O(sigma^2)`` BCH terms and
keeps the ``tangent_last`` mode's no-materialisation efficiency.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .geometry import matrix_log_gregory, trace_free


Tensor = torch.Tensor

_MEAN_MODES = ("tangent", "karcher1")
_DISPERSIONS = ("mean_norm", "frechet")


def _masked_mean(values: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Mean over the node axis (dim 0), restricted to ``mask`` when provided."""

    if mask is None:
        return values.mean(dim=0)
    mask = mask.to(device=values.device, dtype=torch.bool)
    if mask.ndim != 1 or mask.shape[0] != values.shape[0]:
        raise ValueError(
            "mask must be one-dimensional over the node axis; got "
            f"{tuple(mask.shape)} for values {tuple(values.shape)}"
        )
    count = int(mask.sum())
    if count == 0:
        return values.mean(dim=0)
    return values[mask].mean(dim=0)


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
        self.matrix_dim = int(matrix_dim)
        self.num_factors = int(num_factors)
        self.mean_mode = mean_mode
        self.dispersion = dispersion
        self.eps = float(eps)
        self.log_terms = int(log_terms)
        self.jitter = float(jitter)
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

    def _rescale_factor(self, tangent_norms: Tensor, mask: Optional[Tensor]) -> Tensor:
        if self.dispersion == "mean_norm":
            spread = _masked_mean(tangent_norms, mask)
        else:
            spread = _masked_mean(tangent_norms.square(), mask).clamp_min(0).sqrt()
        return self.gamma / (spread + self.eps), spread

    def forward(
        self, groups: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Normalise a full table of group matrices.

        Args:
            groups: ``[num_nodes, num_factors, n, n]`` matrices in ``SL(n)``.
            mask: Optional boolean node mask; statistics are computed over the
                masked rows only, while every row is transformed.
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
        mean = torch.matrix_exp(trace_free(_masked_mean(node_logs, mask)))
        centred = _relative_log(
            mean, work, terms=self.log_terms, jitter=self.jitter
        )
        if self.mean_mode == "karcher1":
            mean = mean @ torch.matrix_exp(
                trace_free(_masked_mean(centred, mask))
            )
            centred = _relative_log(
                mean, work, terms=self.log_terms, jitter=self.jitter
            )

        tangent_norms = torch.linalg.matrix_norm(
            centred, ord="fro", dim=(-2, -1)
        )
        factor, spread = self._rescale_factor(tangent_norms, mask)
        scaled = centred * factor[None, :, None, None]
        output = torch.matrix_exp(scaled)
        if self.bias is not None:
            output = torch.matrix_exp(trace_free(self.bias)).unsqueeze(0) @ output

        diagnostics = self._diagnostics(work, mean, centred, spread)
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output, diagnostics

    def normalise_tangent(
        self, coordinates: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """First-order (identity-anchored) form for coordinate propagation."""

        self._check_input(coordinates, "coordinates")
        work = trace_free(coordinates)
        centred = work - _masked_mean(work, mask).unsqueeze(0)
        tangent_norms = torch.linalg.matrix_norm(
            centred, ord="fro", dim=(-2, -1)
        )
        factor, spread = self._rescale_factor(tangent_norms, mask)
        output = centred * factor[None, :, None, None]
        if self.bias is not None:
            output = output + trace_free(self.bias).unsqueeze(0)
        diagnostics = {
            "operator": "sl_liebn_tangent",
            "dispersion": [float(v) for v in spread.detach().cpu().reshape(-1)],
            "gamma": [float(v) for v in self.gamma.detach().cpu().reshape(-1)],
        }
        return output, diagnostics

    def _diagnostics(
        self, work: Tensor, mean: Tensor, centred: Tensor, spread: Tensor
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
            sign, log_abs_det = torch.linalg.slogdet(mean.detach().float())
        return {
            "operator": "sl_liebn",
            "mean_mode": self.mean_mode,
            "dispersion": [float(v) for v in spread.detach().cpu().reshape(-1)],
            "gamma": [float(v) for v in self.gamma.detach().cpu().reshape(-1)],
            "max_centred_log_reconstruction_residual": float(
                residual.max().item()
            ),
            "mean_nonpositive_determinants": int(sign.le(0).sum().item()),
            "max_abs_mean_log_determinant": float(
                log_abs_det.abs().max().item()
            ),
        }


__all__ = ["SLLieBatchNorm"]
