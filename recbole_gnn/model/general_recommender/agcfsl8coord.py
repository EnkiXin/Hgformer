"""Chart-based ``SL(8)`` decoder for the clean-room AGCF dynamics.

``AGCFSL8Coord`` is deliberately labelled a **surrogate/chart ablation**.  It
runs AGCF's adaptive Hamiltonian equations in a 63-dimensional Euclidean
coefficient chart, identifies those coefficients with an orthonormal basis of
the trace-free Lie algebra ``sl(8)``, and applies ``matrix_exp`` only at the
recommendation decoder.  Pairwise relevance is the negative one-sided
Frobenius norm of ``log(U^-1 I)``.

This is not an intrinsic Hamiltonian system on ``SL(8)``: the state updates do
not evolve a group-valued position/cotangent pair, and they do not include the
coadjoint terms of left-trivialised group mechanics.  Keeping that distinction
explicit makes this model a useful, scalable control before implementing the
substantially more expensive intrinsic alternative.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from recbole_gnn.model.general_recommender.agcf import (
    AGCF,
    _as_bool,
    _config_get,
)
from slrec_experiments import geometry as sl_geometry


def _orthonormal_sl_basis(matrix_dim: int = 8) -> torch.Tensor:
    """Return a Frobenius-orthonormal basis of ``sl(matrix_dim)``.

    The off-diagonal matrix units provide ``n(n-1)`` basis elements.  The
    remaining ``n-1`` diagonal elements are the standard Helmert contrasts,
    which are mutually orthonormal and have zero trace.
    """

    if matrix_dim < 2:
        raise ValueError("matrix_dim must be at least 2")
    basis = []
    for row in range(matrix_dim):
        for column in range(matrix_dim):
            if row == column:
                continue
            element = torch.zeros(matrix_dim, matrix_dim, dtype=torch.float32)
            element[row, column] = 1.0
            basis.append(element)
    for contrast in range(1, matrix_dim):
        diagonal = torch.zeros(matrix_dim, dtype=torch.float32)
        normaliser = math.sqrt(float(contrast * (contrast + 1)))
        diagonal[:contrast] = 1.0 / normaliser
        diagonal[contrast] = -float(contrast) / normaliser
        basis.append(torch.diag(diagonal))
    return torch.stack(basis, dim=0)


class AGCFSL8Coord(AGCF):
    """AGCF in a 63-D ``sl(8)`` coefficient chart with an SL decoder.

    The dynamics and adaptive inverse metric are inherited unchanged from
    :class:`AGCF`, except that their position width is fixed to the exact
    ``8**2 - 1`` chart dimension.  ``pairwise_loss`` may be ``hinge`` (squared
    distance margin ranking) or ``bpr`` (softplus over negative distances).
    The scoring geometry is intentionally fixed to one-sided matrix-log with
    the Frobenius/Schatten-2 norm so configuration cannot silently change the
    stated ablation.
    """

    MODEL_NAME = "AGCFSL8Coord"
    MATRIX_DIM = 8
    POSITION_DIMENSION = MATRIX_DIM * MATRIX_DIM - 1
    _LOSS_ALIASES = {
        "hinge": "hinge",
        "faithful_hinge": "hinge",
        "hinge_squared": "hinge",
        "agcf_hinge": "hinge",
        "bpr": "bpr",
        "bpr_mean": "bpr",
    }

    def __init__(self, config: Any, dataset: Any) -> None:
        configured_dimension = int(
            _config_get(config, "embedding_size", self.POSITION_DIMENSION)
        )
        if configured_dimension != self.POSITION_DIMENSION:
            raise ValueError(
                f"{self.MODEL_NAME} fixes embedding_size="
                f"{self.POSITION_DIMENSION}; got {configured_dimension}"
            )
        super().__init__(config, dataset)

        matrix_dim = int(_config_get(config, "matrix_dim", self.MATRIX_DIM))
        if matrix_dim != self.MATRIX_DIM:
            raise ValueError(
                f"{self.MODEL_NAME} fixes matrix_dim={self.MATRIX_DIM}; "
                f"got {matrix_dim}"
            )
        num_factors = int(_config_get(config, "num_factors", 1))
        if num_factors != 1:
            raise ValueError(f"{self.MODEL_NAME} supports exactly one SL factor")
        self.matrix_dim = self.MATRIX_DIM
        self.num_factors = 1
        self.coordinate_dim = self.POSITION_DIMENSION
        self.intrinsic_dim = self.POSITION_DIMENSION

        self.sl_scale = float(_config_get(config, "sl_scale", 1.0))
        if self.sl_scale <= 0:
            raise ValueError("sl_scale must be positive")
        configured_clip = float(_config_get(config, "coord_clip", 1.0))
        self.coord_clip = configured_clip if configured_clip > 0 else None
        self.log_terms = int(_config_get(config, "log_terms", 12))
        self.log_jitter = float(_config_get(config, "log_jitter", 1e-7))
        if self.log_terms < 1:
            raise ValueError("log_terms must be positive")
        if self.log_jitter < 0:
            raise ValueError("log_jitter must be non-negative")

        configured_order = _config_get(config, "schatten_p", 2)
        try:
            configured_order = float(configured_order)
        except (TypeError, ValueError) as exc:
            raise ValueError("AGCFSL8Coord fixes schatten_p=2") from exc
        if configured_order != 2.0:
            raise ValueError("AGCFSL8Coord fixes schatten_p=2 (Frobenius)")
        if _as_bool(_config_get(config, "symmetric_distance", False)):
            raise ValueError(
                "AGCFSL8Coord fixes symmetric_distance=false for its "
                "one-sided matrix-log scorer"
            )
        self.schatten_p = 2
        self.symmetric_distance = False

        requested_loss = str(
            _config_get(config, "pairwise_loss", "hinge")
        ).strip().lower()
        try:
            self.pairwise_loss = self._LOSS_ALIASES[requested_loss]
        except KeyError as exc:
            raise ValueError(
                "pairwise_loss must be hinge or bpr; "
                f"got {requested_loss!r}"
            ) from exc
        self.loss_margin = float(
            _config_get(
                config,
                "loss_margin",
                _config_get(config, "margin", self.loss_margin),
            )
        )
        if self.loss_margin < 0:
            raise ValueError("loss_margin must be non-negative")

        self.eval_user_chunk_size = max(
            1, int(_config_get(config, "eval_user_chunk_size", 64))
        )
        self.eval_item_chunk_size = max(
            1, int(_config_get(config, "eval_item_chunk_size", 512))
        )
        self.sl_membership_tolerance = float(
            _config_get(config, "sl_membership_tolerance", 1e-4)
        )
        if self.sl_membership_tolerance <= 0:
            raise ValueError("sl_membership_tolerance must be positive")

        self.register_buffer(
            "sl8_chart_basis",
            _orthonormal_sl_basis(self.MATRIX_DIM),
            persistent=False,
        )
        self.restore_user_group = None
        self.restore_item_group = None
        self.other_parameter_name = ["restore_user_group", "restore_item_group"]
        self._tail_analysis = _as_bool(
            _config_get(config, "tail_analysis", False)
        )
        self._popularity_analysis = _as_bool(
            _config_get(config, "popularity_analysis", False)
        )

    def _coordinates_to_lie_algebra(
        self, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Decode independent chart coefficients to trace-free matrices."""

        if coordinates.shape[-1] != self.POSITION_DIMENSION:
            raise ValueError(
                "last coordinate dimension must be 63; "
                f"got {coordinates.shape[-1]}"
            )
        return torch.einsum(
            "...c,cij->...ij",
            coordinates,
            self.sl8_chart_basis.to(dtype=coordinates.dtype),
        )

    def _to_group(self, coordinates: torch.Tensor) -> torch.Tensor:
        lie_algebra = self._coordinates_to_lie_algebra(coordinates)
        return sl_geometry.to_sl(
            lie_algebra,
            scale=self.sl_scale,
            max_frobenius=self.coord_clip,
        )

    @staticmethod
    def _align_group_shapes(
        left: torch.Tensor, right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [B,n,n] -> [B,1,n,n] for a [B,K,n,n] negative table.
        if right.ndim == left.ndim + 1:
            left = left.unsqueeze(-3)
        return left, right

    def _group_distance(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        left, right = self._align_group_shapes(left, right)
        if self.log_terms == 12:
            # Algebraically identical to the one-sided p=2 reference below,
            # but fuses the relative/Cayley solves and evaluates the same
            # degree-11 Gregory polynomial with fewer 8x8 matrix products.
            return sl_geometry.one_sided_gregory_frobenius_distance_k12(
                left,
                right,
                jitter=self.log_jitter,
            )
        return sl_geometry.sl_semidistance(
            left,
            right,
            p=2,
            terms=self.log_terms,
            jitter=self.log_jitter,
            symmetric=False,
        )

    def _score_groups(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        return -self._group_distance(left, right)

    def _clear_full_sort_cache(self) -> None:
        self.restore_user_group = None
        self.restore_item_group = None

    def train(self, mode: bool = True) -> "AGCFSL8Coord":
        if mode:
            self._clear_full_sort_cache()
        return super().train(mode)

    def calculate_loss(self, interaction: Any) -> torch.Tensor:
        self._clear_full_sort_cache()
        all_user_coordinates, all_item_coordinates = super().forward()
        user = interaction[self.USER_ID]
        positive_item = interaction[self.ITEM_ID]
        negative_item = interaction[self.NEG_ITEM_ID]

        # The same entity appears many times in a pairwise mini-batch.  Decode
        # each selected user/item at most once so matrix_exp scales with the
        # number of unique entities rather than the number of interactions.
        unique_users, user_inverse = torch.unique(
            user.reshape(-1), sorted=False, return_inverse=True
        )
        flat_positive = positive_item.reshape(-1)
        flat_negative = negative_item.reshape(-1)
        requested_items = torch.cat((flat_positive, flat_negative), dim=0)
        unique_items, item_inverse = torch.unique(
            requested_items, sorted=False, return_inverse=True
        )
        unique_user_groups = self._to_group(all_user_coordinates[unique_users])
        unique_item_groups = self._to_group(all_item_coordinates[unique_items])
        user_group = unique_user_groups[user_inverse].reshape(
            user.shape + unique_user_groups.shape[1:]
        )
        positive_inverse = item_inverse[: flat_positive.numel()]
        negative_inverse = item_inverse[flat_positive.numel() :]
        positive_group = unique_item_groups[positive_inverse].reshape(
            positive_item.shape + unique_item_groups.shape[1:]
        )
        negative_group = unique_item_groups[negative_inverse].reshape(
            negative_item.shape + unique_item_groups.shape[1:]
        )

        has_negative_axis = negative_group.ndim == positive_group.ndim + 1
        positive_candidates = positive_group.unsqueeze(-3)
        if not has_negative_axis:
            negative_group = negative_group.unsqueeze(-3)
        candidates = torch.cat((positive_candidates, negative_group), dim=-3)
        candidate_distance = self._group_distance(
            user_group.unsqueeze(-3), candidates
        )
        positive_distance = candidate_distance[..., 0]
        negative_distance = candidate_distance[..., 1:]
        if has_negative_axis:
            positive_distance = positive_distance.unsqueeze(-1)
        else:
            negative_distance = negative_distance.squeeze(-1)

        if self.pairwise_loss == "hinge":
            return F.relu(
                positive_distance.square()
                - negative_distance.square()
                + self.loss_margin
            ).mean()
        # scores are -distance, hence -score_pos + score_neg = d_pos - d_neg.
        return F.softplus(positive_distance - negative_distance).mean()

    def predict(self, interaction: Any) -> torch.Tensor:
        all_user_coordinates, all_item_coordinates = super().forward()
        user_group = self._to_group(
            all_user_coordinates[interaction[self.USER_ID]]
        )
        item_group = self._to_group(
            all_item_coordinates[interaction[self.ITEM_ID]]
        )
        return self._score_groups(user_group, item_group)

    def _full_sort_group_tables(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.restore_user_group is None or self.restore_item_group is None:
            user_coordinates, item_coordinates = super().forward()
            self.restore_user_group = self._to_group(user_coordinates)
            self.restore_item_group = self._to_group(item_coordinates)
        return self.restore_user_group, self.restore_item_group

    def full_sort_predict(self, interaction: Any) -> torch.Tensor:
        all_user_group, all_item_group = self._full_sort_group_tables()
        requested_users = all_user_group[interaction[self.USER_ID]]
        user_chunks = []
        for user_start in range(
            0, requested_users.shape[0], self.eval_user_chunk_size
        ):
            current_users = requested_users[
                user_start : user_start + self.eval_user_chunk_size
            ]
            item_chunks = []
            for item_start in range(
                0, all_item_group.shape[0], self.eval_item_chunk_size
            ):
                current_items = all_item_group[
                    item_start : item_start + self.eval_item_chunk_size
                ]
                item_chunks.append(
                    self._score_groups(
                        current_users[:, None, ...],
                        current_items[None, ...],
                    )
                )
            user_chunks.append(torch.cat(item_chunks, dim=1))
        scores = torch.cat(user_chunks, dim=0).reshape(-1)
        if self._tail_analysis:
            return self.head_item, self.tail_item, scores
        if self._popularity_analysis:
            return (
                self.rank1item,
                self.rank2item,
                self.rank3item,
                self.rank4item,
                self.rank5item,
                scores,
            )
        return scores

    @torch.no_grad()
    def geometry_diagnostics(self, sample_nodes: int = 32) -> Dict[str, float]:
        # AGCF's SPD audit is intentionally evaluated on its node-position
        # table.  Decoder diagnostics below instead use the *actual summed
        # dynamics output*, because that is what clipping and exp see.
        diagnostics = super().geometry_diagnostics(sample_nodes=sample_nodes)
        user_coordinates, item_coordinates = super().forward()
        final_coordinates = torch.cat(
            (user_coordinates, item_coordinates), dim=0
        )
        count = min(int(sample_nodes), final_coordinates.shape[0])
        coordinates = final_coordinates[:count]
        lie_algebra = self._coordinates_to_lie_algebra(coordinates)
        groups = self._to_group(coordinates)
        trace_error = lie_algebra.diagonal(dim1=-2, dim2=-1).sum(dim=-1).abs()
        preclip_norm = torch.linalg.matrix_norm(
            lie_algebra * self.sl_scale, ord="fro", dim=(-2, -1)
        )
        if self.coord_clip is None:
            effective_norm = preclip_norm
            saturation = torch.zeros_like(preclip_norm, dtype=torch.bool)
        else:
            effective_norm = preclip_norm.clamp(max=self.coord_clip)
            saturation = preclip_norm.gt(self.coord_clip)
        sign, log_abs_det = torch.linalg.slogdet(groups.float())
        membership_violation = (
            sign.le(0)
            | ~torch.isfinite(log_abs_det)
            | log_abs_det.abs().gt(self.sl_membership_tolerance)
        )
        diagnostics.update(
            {
                "chart_dimension": self.POSITION_DIMENSION,
                "matrix_dimension": self.MATRIX_DIM,
                "decoder_sample_nodes": count,
                "decoder_preclip_frobenius_mean": float(
                    preclip_norm.mean().cpu()
                ),
                "decoder_preclip_frobenius_max": float(
                    preclip_norm.max().cpu()
                ),
                "decoder_effective_frobenius_mean": float(
                    effective_norm.mean().cpu()
                ),
                "decoder_effective_frobenius_max": float(
                    effective_norm.max().cpu()
                ),
                "decoder_clip_saturation_fraction": float(
                    saturation.float().mean().cpu()
                ),
                "max_abs_lie_trace": float(trace_error.max().cpu()),
                "max_abs_group_log_determinant": float(
                    log_abs_det.abs().max().cpu()
                ),
                "group_membership_violations": int(
                    membership_violation.sum().cpu()
                ),
            }
        )
        return diagnostics


__all__ = ["AGCFSL8Coord"]
