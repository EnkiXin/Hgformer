"""An SL(n) collaborative-filtering model for official RecBole 1.2.x.

``SLRec`` learns one or more unconstrained square matrices per user and item,
projects each matrix to the trace-free Lie algebra, and exponentiates it into
``SL(n)``.  Multiple factors form the direct product ``SL(n)^F``; relevance is
the negative product distance over the factor-wise SL(n) semidistances.
Optionally, LightGCN-style propagation is performed on the *Lie-algebra
coordinates* before the exponential map, which keeps every final
representation on the determinant-one group.

This module intentionally depends only on core RecBole and PyTorch; PyG and the
RecBole-GNN fork are not required.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType

try:  # Package import (normal use).
    from .geometry import (
        one_sided_gregory_frobenius_distance_k12,
        one_sided_sqrt_extended_frobenius_distance,
        sl_semidistance,
        to_sl,
        trace_free,
    )
except ImportError:  # Direct import from a runner located in this directory.
    from geometry import (
        one_sided_gregory_frobenius_distance_k12,
        one_sided_sqrt_extended_frobenius_distance,
        sl_semidistance,
        to_sl,
        trace_free,
    )


LOG_DOMAIN_GUARD_REVISION = "db_residual_spectral_tail_v1"


def _config_get(config: Any, key: str, default: Any) -> Any:
    """Read an optional RecBole config value without assuming dict methods."""

    try:
        value = config[key]
    except (KeyError, TypeError, AttributeError):
        return default
    return default if value is None else value


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"true", "yes", "y", "1", "on"}:
            return True
        if normalised in {"false", "no", "n", "0", "off"}:
            return False
    return bool(value)


class SLRec(GeneralRecommender):
    """Pairwise collaborative filtering in the special-linear group.

    Model-specific RecBole configuration keys:

    ``matrix_dim`` (8)
        Size ``n`` of the special-linear group.  ``n=8`` has 63 effective
        degrees of freedom and is comparable with a 64-dimensional baseline.
    ``num_factors`` (1), ``factor_aggregation`` (``l2``)
        Number ``F`` of independent SL(n) factors and their distance
        aggregation. ``l2`` is the canonical product metric,
        ``sqrt(sum_f D_f^2)``. ``l1`` sums factor distances and ``mean`` is its
        scale-normalised variant.  The raw entity width is ``F*n*n`` and the
        trace-free intrinsic dimension is ``F*(n*n-1)``.
    ``n_layers`` (0)
        Number of LightGCN propagation layers in tangent coordinates.
    ``init_std`` (0.01), ``reg_weight`` (1e-5)
        Raw-coordinate initialisation and L2 regularisation strength.
    ``sl_scale`` (1.0), ``coord_clip`` (1.0)
        Scale and optional radial Frobenius cap before matrix exponentiation.
        A non-positive ``coord_clip`` disables the cap.
    ``schatten_p`` (2), ``log_terms`` (12), ``log_jitter`` (1e-7)
        SL(n) semidistance parameters.
    ``symmetric_distance`` (True)
        Use both directed terms from the paper's semidistance.  False is a
        faster one-sided ablation.
    ``fast_one_sided_frobenius`` (False)
        Use the algebraically equivalent one-solve/blocked-polynomial path
        when ``symmetric_distance=false``, ``schatten_p=2`` and
        ``log_terms=12``.  Production speed overlays enable it explicitly;
        the base default preserves last-bit historical execution order.
    ``log_domain_sqrt_steps`` (0)
        Inverse scaling-and-squaring for the group-log distance:
        ``log R = 2^k log(R^{1/2^k})`` with a differentiable Denman--Beavers
        square root.  Each step doubles the reliable Gregory domain; the
        plain scorer is exact only to relative distance ~3 and returns
        inflated values with exploding gradients beyond it, which destroys
        training once layer normalisation pushes pair distances past ~4.
        Requires ``schatten_p: 2`` and ``symmetric_distance: false``; takes
        precedence over ``fast_one_sided_frobenius``.  The associated
        ``log_domain_sqrt_iterations`` (12),
        ``log_domain_sqrt_residual_tolerance`` (1e-3), and
        ``log_domain_tail_tolerance`` (1e-3) make the approximation fail fast
        near the principal-log branch cut instead of returning a finite but
        untrustworthy distance.  ``log_domain_guard_revision`` is a protocol
        identity and must match this implementation's spectral-tail guard.
    ``eval_log_domain_sqrt_steps`` (unset)
        Evaluation-side override of ``log_domain_sqrt_steps``; unset inherits
        the training value so both share one metric implementation.  Setting
        ``0`` restores the fused ``K=12`` full-sort path for validation
        speed while training keeps the extended domain.  This is safe for
        top-k metrics only when representations are radius-bounded (e.g. a
        LieBN trust region of 3): in-domain scores are algebraically
        identical and out-of-domain distances only inflate, so far items
        stay at the bottom.  Do not split the scorers for configurations
        without a radius bound, and note the split in reported protocols.
    ``sl_score_mode`` (``group_log``)
        ``group_log`` keeps the special-linear matrix-log decoder.  The
        opt-in ``tangent_euclidean`` control scores the squared Frobenius
        distance between the final trace-free effective coordinates.  Its
        full-sort path uses the Euclidean identity
        ``||x-y||^2 = ||x||^2 + ||y||^2 - 2 x y^T`` rather than constructing
        one matrix logarithm per candidate pair.  ``chart_euclidean_distance``
        is accepted as a descriptive alias.
    ``score_scale`` (1.0), ``learnable_score_scale`` (True)
        Initial positive multiplier on negative distances and whether it is
        learned.  ``max_score_scale`` (100) bounds its exponential form.
    ``eval_user_chunk_size`` (128), ``eval_item_chunk_size`` (1024)
        Two-dimensional chunks used by ``group_log`` full-sort prediction.
        Bounding both axes matters for the Hgformer configs, whose legacy
        ``eval_batch_size`` can place many users in one full-sort batch.  The
        ``tangent_euclidean`` path follows LightGCN and multiplies against the
        complete item table, so only its user chunk applies.
    ``eval_prefilter`` (``none``), ``eval_prefilter_candidates`` (2048)
        ``group_log`` full sort only.  ``frobenius`` shortlists each user's
        ``eval_prefilter_candidates`` nearest items by ambient Frobenius
        distance on the flattened group tables (one GEMM), then runs the
        exact SL scorer on the shortlist alone; all other items receive the
        dtype minimum score.  Unlike ``sl_score_mode: tangent_euclidean``
        this keeps the model and its reported metric the group-log decoder —
        the surrogate only selects candidates.  This is an approximate
        candidate-set evaluator, not exact full ranking: RecBole masks item 0
        and seen history after ``full_sort_predict`` returns, so those items
        can consume shortlist capacity.  Keep it disabled for early stopping
        and reported validation/test metrics unless containment is audited on
        the real checkpoint after applying the same masks.
    ``eval_tf32`` (False)
        Allow TF32 tensor-core matmuls inside ``group_log`` full-sort scoring
        only; the process setting is restored afterwards.  Scores move at
        TF32 precision, so keep one setting per results table.  The
        ``tangent_euclidean`` GEMM keeps its own explicit FP32 policy.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config: Any, dataset: Any) -> None:
        super().__init__(config, dataset)

        self.matrix_dim = int(_config_get(config, "matrix_dim", 8))
        if self.matrix_dim < 2:
            raise ValueError("matrix_dim must be at least 2")
        self.num_factors = int(_config_get(config, "num_factors", 1))
        if self.num_factors < 1:
            raise ValueError("num_factors must be positive")
        self.factor_aggregation = str(
            _config_get(config, "factor_aggregation", "l2")
        ).strip().lower()
        if self.factor_aggregation not in {"l2", "l1", "mean"}:
            raise ValueError(
                "factor_aggregation must be one of {'l2', 'l1', 'mean'}; "
                f"got {self.factor_aggregation!r}"
            )
        requested_score_mode = str(
            _config_get(config, "sl_score_mode", "group_log")
        ).strip().lower()
        score_mode_aliases = {
            "group_log": "group_log",
            "tangent_euclidean": "tangent_euclidean",
            "chart_euclidean_distance": "tangent_euclidean",
        }
        if requested_score_mode not in score_mode_aliases:
            raise ValueError(
                "sl_score_mode must be one of "
                "{'group_log', 'tangent_euclidean', "
                "'chart_euclidean_distance'}; "
                f"got {requested_score_mode!r}"
            )
        self.sl_score_mode = score_mode_aliases[requested_score_mode]
        if (
            self.sl_score_mode == "tangent_euclidean"
            and self.factor_aggregation != "l2"
        ):
            raise ValueError(
                "sl_score_mode=tangent_euclidean uses the squared canonical "
                "product Frobenius distance and therefore requires "
                "factor_aggregation: l2"
            )
        self.factor_coordinate_dim = self.matrix_dim * self.matrix_dim
        self.coordinate_dim = self.num_factors * self.factor_coordinate_dim
        self.intrinsic_dim = self.num_factors * (self.factor_coordinate_dim - 1)

        self.n_layers = int(_config_get(config, "n_layers", 0))
        if self.n_layers < 0:
            raise ValueError("n_layers must be non-negative")
        self.reg_weight = float(_config_get(config, "reg_weight", 1e-5))
        self.init_std = float(_config_get(config, "init_std", 0.01))

        self.sl_scale = float(_config_get(config, "sl_scale", 1.0))
        coord_clip = float(_config_get(config, "coord_clip", 1.0))
        self.coord_clip: Optional[float] = coord_clip if coord_clip > 0 else None
        self.schatten_p = _config_get(config, "schatten_p", 2)
        self.log_terms = int(_config_get(config, "log_terms", 12))
        self.log_jitter = float(_config_get(config, "log_jitter", 1e-7))
        self.symmetric_distance = _as_bool(
            _config_get(config, "symmetric_distance", True)
        )
        self.fast_one_sided_frobenius = _as_bool(
            # Keep the generic SLRec reference implementation opt-in stable.
            # Production SL8/SL16 overlays enable the algebraically equivalent
            # K12 path explicitly, so unrelated historical configs do not
            # silently change their last-bit floating-point execution order.
            _config_get(config, "fast_one_sided_frobenius", False)
        )
        self.log_domain_sqrt_steps = int(
            _config_get(config, "log_domain_sqrt_steps", 0)
        )
        if self.log_domain_sqrt_steps < 0:
            raise ValueError("log_domain_sqrt_steps must be non-negative")
        if self.log_domain_sqrt_steps > 0:
            try:
                order = float(self.schatten_p)
            except (TypeError, ValueError):
                order = None
            if order != 2.0 or self.symmetric_distance:
                raise ValueError(
                    "log_domain_sqrt_steps requires schatten_p: 2 and "
                    "symmetric_distance: false (the one-sided Frobenius "
                    "distance; the symmetric variant is norm-identical)"
                )
        self.log_domain_sqrt_iterations = int(
            _config_get(config, "log_domain_sqrt_iterations", 12)
        )
        self.log_domain_sqrt_residual_tolerance = float(
            _config_get(
                config, "log_domain_sqrt_residual_tolerance", 1e-3
            )
        )
        self.log_domain_tail_tolerance = float(
            _config_get(config, "log_domain_tail_tolerance", 1e-3)
        )
        requested_guard_revision = str(
            _config_get(
                config,
                "log_domain_guard_revision",
                LOG_DOMAIN_GUARD_REVISION,
            )
        )
        if (
            self.log_domain_sqrt_steps > 0
            and requested_guard_revision != LOG_DOMAIN_GUARD_REVISION
        ):
            raise ValueError(
                "log_domain_guard_revision does not match the implemented "
                f"guard: expected {LOG_DOMAIN_GUARD_REVISION!r}, got "
                f"{requested_guard_revision!r}"
            )
        self.log_domain_guard_revision = LOG_DOMAIN_GUARD_REVISION
        if self.log_domain_sqrt_iterations < 1:
            raise ValueError("log_domain_sqrt_iterations must be positive")
        if self.log_domain_sqrt_residual_tolerance <= 0:
            raise ValueError(
                "log_domain_sqrt_residual_tolerance must be positive"
            )
        if self.log_domain_tail_tolerance <= 0:
            raise ValueError("log_domain_tail_tolerance must be positive")
        eval_sqrt_steps = _config_get(config, "eval_log_domain_sqrt_steps", None)
        if eval_sqrt_steps is None:
            # Inherit the training scorer so train and evaluation share one
            # metric implementation unless the config explicitly splits them.
            self.eval_log_domain_sqrt_steps = self.log_domain_sqrt_steps
        else:
            self.eval_log_domain_sqrt_steps = int(eval_sqrt_steps)
            if self.eval_log_domain_sqrt_steps < 0:
                raise ValueError(
                    "eval_log_domain_sqrt_steps must be non-negative"
                )
            if self.eval_log_domain_sqrt_steps > 0:
                try:
                    order = float(self.schatten_p)
                except (TypeError, ValueError):
                    order = None
                if order != 2.0 or self.symmetric_distance:
                    raise ValueError(
                        "eval_log_domain_sqrt_steps requires schatten_p: 2 "
                        "and symmetric_distance: false"
                    )

        initial_score_scale = float(_config_get(config, "score_scale", 1.0))
        if initial_score_scale <= 0:
            raise ValueError("score_scale must be positive")
        self.max_score_scale = float(_config_get(config, "max_score_scale", 100.0))
        learnable_scale = _as_bool(
            _config_get(config, "learnable_score_scale", True)
        )
        self.log_score_scale = nn.Parameter(
            torch.tensor(math.log(initial_score_scale), dtype=torch.float32),
            requires_grad=learnable_scale,
        )

        self.eval_user_chunk_size = max(
            1, int(_config_get(config, "eval_user_chunk_size", 128))
        )
        self.eval_item_chunk_size = max(
            1, int(_config_get(config, "eval_item_chunk_size", 1024))
        )
        eval_prefilter = str(
            _config_get(config, "eval_prefilter", "none")
        ).strip().lower()
        if eval_prefilter not in {"none", "frobenius"}:
            raise ValueError(
                "eval_prefilter must be one of {'none', 'frobenius'}; "
                f"got {eval_prefilter!r}"
            )
        self.eval_prefilter = eval_prefilter
        self.eval_prefilter_candidates = int(
            _config_get(config, "eval_prefilter_candidates", 2048)
        )
        if self.eval_prefilter_candidates < 1:
            raise ValueError("eval_prefilter_candidates must be positive")
        self.eval_tf32 = _as_bool(_config_get(config, "eval_tf32", False))

        self.user_embedding = nn.Embedding(self.n_users, self.coordinate_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.coordinate_dim)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=self.init_std)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=self.init_std)

        if self.n_layers:
            norm_adj_matrix = self._build_normalised_adjacency(dataset).to(self.device)
        else:
            norm_adj_matrix = None
        # The graph is deterministically rebuilt from the training split.
        # Keeping it non-persistent avoids duplicating a potentially large COO
        # tensor in every legacy RecBole checkpoint while still letting
        # ``model.to(device)`` move it with the module.
        self.register_buffer(
            "norm_adj_matrix", norm_adj_matrix, persistent=False
        )

        # Full-sort evaluation caches. RecBole persists names in checkpoints in
        # the same way as its LightGCN implementation.
        self.restore_user_group: Optional[torch.Tensor] = None
        self.restore_item_group: Optional[torch.Tensor] = None
        self.restore_user_effective_coordinates: Optional[torch.Tensor] = None
        self.restore_item_effective_coordinates: Optional[torch.Tensor] = None
        # Derived item terms are deliberately transient: they are cheap to
        # rebuild from a loaded coordinate cache and need not enlarge a
        # checkpoint.  Keeping them across outer evaluator batches avoids an
        # otherwise repeated O(num_items * coordinate_dim) reduction.
        self._restore_item_coordinate_source: Optional[torch.Tensor] = None
        self._restore_item_coordinate_flat: Optional[torch.Tensor] = None
        self._restore_item_coordinate_squared_norm: Optional[torch.Tensor] = None
        self.other_parameter_name = [
            "restore_user_group",
            "restore_item_group",
            "restore_user_effective_coordinates",
            "restore_item_effective_coordinates",
        ]

    def _build_normalised_adjacency(self, dataset: Any) -> torch.Tensor:
        """Construct symmetric ``D^-1/2 A D^-1/2`` as a sparse tensor."""

        interaction = dataset.inter_matrix(form="coo")
        users = torch.as_tensor(interaction.row, dtype=torch.long)
        items = torch.as_tensor(interaction.col, dtype=torch.long) + self.n_users

        source = torch.cat((users, items), dim=0)
        target = torch.cat((items, users), dim=0)
        node_count = self.n_users + self.n_items
        degree = torch.bincount(source, minlength=node_count).to(torch.float32)
        inv_sqrt_degree = degree.clamp_min(1.0).pow(-0.5)
        values = inv_sqrt_degree[source] * inv_sqrt_degree[target]
        indices = torch.stack((source, target), dim=0)
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=(node_count, node_count),
            dtype=torch.float32,
        ).coalesce()

    def _clear_full_sort_cache(self) -> None:
        self.restore_user_group = None
        self.restore_item_group = None
        self.restore_user_effective_coordinates = None
        self.restore_item_effective_coordinates = None
        self._clear_full_sort_coordinate_terms()

    def _clear_full_sort_coordinate_terms(self) -> None:
        self._restore_item_coordinate_source = None
        self._restore_item_coordinate_flat = None
        self._restore_item_coordinate_squared_norm = None

    def load_other_parameter(self, para: Any) -> None:
        """Load persistent caches while invalidating derived chart terms."""

        # A legacy checkpoint may not contain the newer chart-cache keys.  Do
        # not let values from an earlier evaluation survive merely because a
        # key is absent from ``para``.
        self._clear_full_sort_cache()
        super().load_other_parameter(para)
        self._clear_full_sort_coordinate_terms()

    def train(self, mode: bool = True) -> "SLRec":
        # Prevent stale propagated coordinates after an optimiser step.
        if mode:
            self._clear_full_sort_cache()
        return super().train(mode)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return trace-free user/item coordinates after optional propagation."""

        all_coordinates = torch.cat(
            (self.user_embedding.weight, self.item_embedding.weight), dim=0
        )
        layer_coordinates = [all_coordinates]
        for _ in range(self.n_layers):
            if self.norm_adj_matrix is None:  # Defensive; impossible after init.
                raise RuntimeError("normalised adjacency was not initialised")
            all_coordinates = torch.sparse.mm(
                self.norm_adj_matrix, all_coordinates
            )
            layer_coordinates.append(all_coordinates)

        if len(layer_coordinates) > 1:
            all_coordinates = torch.stack(layer_coordinates, dim=0).mean(dim=0)
        all_coordinates = all_coordinates.reshape(
            -1, self.num_factors, self.matrix_dim, self.matrix_dim
        )
        all_coordinates = trace_free(all_coordinates)
        return torch.split(all_coordinates, (self.n_users, self.n_items), dim=0)

    def _to_group(self, coordinates: torch.Tensor) -> torch.Tensor:
        return to_sl(
            coordinates,
            scale=self.sl_scale,
            max_frobenius=self.coord_clip,
        )

    def _to_effective_tangent_coordinates(
        self, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Apply the exact pre-exponential trace/scale/radius transform."""

        original_dtype = coordinates.dtype
        work = (
            coordinates.float()
            if coordinates.dtype in (torch.float16, torch.bfloat16)
            else coordinates
        )
        effective = trace_free(work) * self.sl_scale
        if self.coord_clip is not None:
            norm = torch.linalg.matrix_norm(
                effective, ord="fro", dim=(-2, -1), keepdim=True
            )
            factor = (
                self.coord_clip / norm.clamp_min(1e-12)
            ).clamp(max=1.0)
            effective = effective * factor
        return (
            effective.to(original_dtype)
            if effective.dtype != original_dtype
            else effective
        )

    def _score_scale(self) -> torch.Tensor:
        scale = self.log_score_scale.exp()
        if self.max_score_scale > 0:
            scale = scale.clamp(max=self.max_score_scale)
        return scale

    @staticmethod
    def _align_pair_shapes(
        user_group: torch.Tensor, item_group: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Broadcast users over a per-user negative-sample dimension."""

        if item_group.ndim == user_group.ndim + 1:
            # Group tensors end in [factor, n, n].  Insert a negative-sample
            # axis immediately before that fixed three-dimensional suffix:
            # [B,F,n,n] -> [B,1,F,n,n].
            user_group = user_group.unsqueeze(-4)
        return user_group, item_group

    def _aggregate_factor_distances(
        self, factor_distances: torch.Tensor
    ) -> torch.Tensor:
        """Combine ``F`` component distances into an ``SL(n)^F`` distance."""

        if factor_distances.shape[-1] != self.num_factors:
            raise ValueError(
                "factor distance width does not match num_factors: "
                f"got {factor_distances.shape[-1]} and {self.num_factors}"
            )
        # Preserve the exact pre-product computation for the default F=1
        # model, including its floating-point and gradient behaviour.
        if self.num_factors == 1:
            return factor_distances.squeeze(-1)
        if self.factor_aggregation == "l2":
            return torch.linalg.vector_norm(factor_distances, ord=2, dim=-1)
        if self.factor_aggregation == "l1":
            return factor_distances.sum(dim=-1)
        return factor_distances.mean(dim=-1)

    def _score_groups(
        self, user_group: torch.Tensor, item_group: torch.Tensor
    ) -> torch.Tensor:
        user_group, item_group = self._align_pair_shapes(user_group, item_group)
        factor_distances = self._factor_distances(user_group, item_group)
        distance = self._aggregate_factor_distances(factor_distances)
        return -self._score_scale() * distance

    def _uses_fast_one_sided_frobenius(self) -> bool:
        """Whether the current geometry has the exact production fast path."""

        try:
            order = float(self.schatten_p)
        except (TypeError, ValueError):
            return False
        return (
            self.fast_one_sided_frobenius
            and getattr(self, "log_domain_sqrt_steps", 0) == 0
            and not self.symmetric_distance
            and order == 2.0
            and self.log_terms == 12
        )

    def _factor_distances(
        self, user_group: torch.Tensor, item_group: torch.Tensor
    ) -> torch.Tensor:
        """Return factor-wise distances through the selected exact formula."""

        # Training must use the domain-extended scorer wherever it is enabled
        # (the plain scorer's out-of-domain gradients destroy optimisation);
        # evaluation may drop back to the fused K=12 path when the config
        # splits them, because ranking only needs in-domain scores — the two
        # scorers are algebraically identical there — and out-of-domain
        # distances only inflate, keeping far items at the bottom.
        sqrt_steps = (
            self.log_domain_sqrt_steps
            if self.training
            else self.eval_log_domain_sqrt_steps
        )
        if sqrt_steps > 0:
            return one_sided_sqrt_extended_frobenius_distance(
                user_group,
                item_group,
                sqrt_steps=sqrt_steps,
                terms=self.log_terms,
                jitter=self.log_jitter,
                sqrt_iterations=self.log_domain_sqrt_iterations,
                sqrt_residual_tolerance=(
                    self.log_domain_sqrt_residual_tolerance
                ),
                log_tail_tolerance=self.log_domain_tail_tolerance,
            )
        if self._uses_fast_one_sided_frobenius():
            return one_sided_gregory_frobenius_distance_k12(
                user_group,
                item_group,
                jitter=self.log_jitter,
            )
        return sl_semidistance(
            user_group,
            item_group,
            p=self.schatten_p,
            terms=self.log_terms,
            jitter=self.log_jitter,
            symmetric=self.symmetric_distance,
        )

    def _score_coordinates(
        self, user_coordinates: torch.Tensor, item_coordinates: torch.Tensor
    ) -> torch.Tensor:
        return self._score_groups(
            self._to_group(user_coordinates), self._to_group(item_coordinates)
        )

    def _effective_coordinate_tables(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the final trace-free chart coordinates used by the control.

        The standalone :class:`SLRec` propagates in the Lie algebra already,
        so its ordinary forward result is the effective coordinate table.
        Group-propagating subclasses override this hook and logarithmise each
        final entity representation once.
        """

        user_coordinates, item_coordinates = self.forward()
        return (
            self._to_effective_tangent_coordinates(user_coordinates),
            self._to_effective_tangent_coordinates(item_coordinates),
        )

    def _full_sort_effective_coordinate_tables(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self.restore_user_effective_coordinates is None
            or self.restore_item_effective_coordinates is None
        ):
            (
                self.restore_user_effective_coordinates,
                self.restore_item_effective_coordinates,
            ) = self._effective_coordinate_tables()
            self._clear_full_sort_coordinate_terms()
        return (
            self.restore_user_effective_coordinates,
            self.restore_item_effective_coordinates,
        )

    @staticmethod
    def _pairwise_squared_coordinate_distance(
        left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        """Squared Frobenius distance with ordinary pair broadcasting."""

        left, right = SLRec._align_pair_shapes(left, right)
        return (left - right).square().sum(dim=(-3, -2, -1))

    def _score_effective_coordinates(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        squared_distance = self._pairwise_squared_coordinate_distance(left, right)
        return -self._score_scale() * squared_distance

    @staticmethod
    def _gemm_squared_coordinate_distance(
        left_flat: torch.Tensor,
        right_flat: torch.Tensor,
        *,
        right_squared_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """All-pairs squared Euclidean distance via norms plus one GEMM."""

        if left_flat.ndim != 2 or right_flat.ndim != 2:
            raise ValueError("flattened coordinate tables must both be matrices")
        if left_flat.shape[1] != right_flat.shape[1]:
            raise ValueError(
                "flattened coordinate widths differ: "
                f"{left_flat.shape[1]} and {right_flat.shape[1]}"
            )
        if right_squared_norm is None:
            right_squared_norm = right_flat.square().sum(dim=1)
        left_squared_norm = left_flat.square().sum(dim=1, keepdim=True)
        # TF32 changes the multiplication precision of this identity on
        # Ampere+ GPUs and can perturb close ranks relative to direct FP32
        # squared differences.  Keep this optional control in full FP32 while
        # restoring the process setting immediately afterwards.
        cuda_matmul = getattr(torch.backends, "cuda", None)
        matmul_backend = getattr(cuda_matmul, "matmul", None)
        disable_tf32 = (
            left_flat.device.type == "cuda"
            and left_flat.dtype == torch.float32
            and matmul_backend is not None
            and hasattr(matmul_backend, "allow_tf32")
        )
        previous_allow_tf32 = None
        if disable_tf32:
            previous_allow_tf32 = matmul_backend.allow_tf32
            matmul_backend.allow_tf32 = False
        try:
            inner_products = torch.matmul(
                left_flat, right_flat.transpose(0, 1)
            )
        finally:
            if disable_tf32:
                matmul_backend.allow_tf32 = previous_allow_tf32
        squared_distance = (
            left_squared_norm
            + right_squared_norm.unsqueeze(0)
            - 2.0 * inner_products
        )
        # Roundoff can make identical/near-identical rows slightly negative;
        # the direct squared-difference definition is non-negative.
        return squared_distance.clamp_min(0.0)

    def _full_sort_item_coordinate_terms(
        self, item_coordinates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._restore_item_coordinate_source is not item_coordinates:
            item_flat = item_coordinates.reshape(item_coordinates.shape[0], -1)
            self._restore_item_coordinate_source = item_coordinates
            self._restore_item_coordinate_flat = item_flat
            self._restore_item_coordinate_squared_norm = item_flat.square().sum(
                dim=1
            )
        if (
            self._restore_item_coordinate_flat is None
            or self._restore_item_coordinate_squared_norm is None
        ):  # Defensive: the source assignment above always fills both.
            raise RuntimeError("full-sort coordinate terms were not initialised")
        return (
            self._restore_item_coordinate_flat,
            self._restore_item_coordinate_squared_norm,
        )

    def calculate_loss(self, interaction: Any) -> torch.Tensor:
        self._clear_full_sort_cache()
        user = interaction[self.USER_ID]
        positive_item = interaction[self.ITEM_ID]
        negative_item = interaction[self.NEG_ITEM_ID]

        if self.sl_score_mode == "tangent_euclidean":
            all_user_coordinates, all_item_coordinates = (
                self._effective_coordinate_tables()
            )
        else:
            all_user_coordinates, all_item_coordinates = self.forward()
        user_coordinates = all_user_coordinates[user]
        positive_coordinates = all_item_coordinates[positive_item]
        negative_coordinates = all_item_coordinates[negative_item]

        if self.sl_score_mode == "tangent_euclidean":
            positive_scores = self._score_effective_coordinates(
                user_coordinates, positive_coordinates
            )
            negative_scores = self._score_effective_coordinates(
                user_coordinates, negative_coordinates
            )
        else:
            positive_scores = self._score_coordinates(
                user_coordinates, positive_coordinates
            )
            negative_scores = self._score_coordinates(
                user_coordinates, negative_coordinates
            )
        if negative_scores.ndim == positive_scores.ndim + 1:
            positive_scores = positive_scores.unsqueeze(-1)
        ranking_loss = F.softplus(negative_scores - positive_scores).mean()

        raw_user = self.user_embedding(user)
        raw_positive = self.item_embedding(positive_item)
        raw_negative = self.item_embedding(negative_item)
        regularisation = (
            raw_user.square().sum(dim=-1).mean()
            + raw_positive.square().sum(dim=-1).mean()
            + raw_negative.square().sum(dim=-1).mean()
        ) / 3.0
        return ranking_loss + self.reg_weight * regularisation

    def predict(self, interaction: Any) -> torch.Tensor:
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if self.sl_score_mode == "tangent_euclidean":
            all_user_coordinates, all_item_coordinates = (
                self._effective_coordinate_tables()
            )
            return self._score_effective_coordinates(
                all_user_coordinates[user], all_item_coordinates[item]
            )
        all_user_coordinates, all_item_coordinates = self.forward()
        return self._score_coordinates(
            all_user_coordinates[user], all_item_coordinates[item]
        )

    def _full_sort_group_tables(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.restore_user_group is None or self.restore_item_group is None:
            user_coordinates, item_coordinates = self.forward()
            self.restore_user_group = self._to_group(user_coordinates)
            self.restore_item_group = self._to_group(item_coordinates)
        return self.restore_user_group, self.restore_item_group

    def full_sort_predict(self, interaction: Any) -> torch.Tensor:
        """Score each requested user against all items in bounded-size chunks."""

        return self._full_sort_predict(
            interaction, history_index=None, apply_exclusions=False
        )

    def full_sort_predict_with_exclusions(
        self, interaction: Any, history_index: Any
    ) -> torch.Tensor:
        """Full-sort score with mask-aware candidate selection when opted in."""

        if self.eval_prefilter == "none":
            return self.full_sort_predict(interaction)
        return self._full_sort_predict(
            interaction, history_index=history_index, apply_exclusions=True
        )

    def _full_sort_predict(
        self,
        interaction: Any,
        history_index: Any = None,
        apply_exclusions: bool = False,
    ) -> torch.Tensor:

        user = interaction[self.USER_ID]
        if self.sl_score_mode == "tangent_euclidean":
            all_user_coordinates, all_item_coordinates = (
                self._full_sort_effective_coordinate_tables()
            )
            user_coordinates = all_user_coordinates[user]
            item_flat, item_squared_norm = self._full_sort_item_coordinate_terms(
                all_item_coordinates
            )
            scores = user_coordinates.new_empty(
                (user_coordinates.shape[0], self.n_items)
            )
            for user_start in range(
                0, user_coordinates.shape[0], self.eval_user_chunk_size
            ):
                user_stop = min(
                    user_start + self.eval_user_chunk_size,
                    user_coordinates.shape[0],
                )
                user_flat = user_coordinates[user_start:user_stop].reshape(
                    user_stop - user_start, -1
                )
                squared_distance = self._gemm_squared_coordinate_distance(
                    user_flat,
                    item_flat,
                    right_squared_norm=item_squared_norm,
                )
                scores[user_start:user_stop] = (
                    -self._score_scale() * squared_distance
                )
            return scores.reshape(-1)

        all_user_group, all_item_group = self._full_sort_group_tables()
        user_group = all_user_group[user]

        with self._tf32_full_sort_matmul():
            if (
                self.eval_prefilter != "none"
                and self.eval_prefilter_candidates < self.n_items
            ):
                scores = self._prefiltered_full_sort_scores(
                    user_group,
                    all_item_group,
                    history_index=history_index,
                    apply_exclusions=apply_exclusions,
                )
            else:
                scores = self._exact_full_sort_scores(
                    user_group, all_item_group
                )
        return scores.reshape(-1)

    @contextmanager
    def _tf32_full_sort_matmul(self) -> Iterator[None]:
        """Optionally allow TF32 matmuls for one ``group_log`` full sort."""

        if not self.eval_tf32 or not torch.cuda.is_available():
            yield
            return
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    def _exact_full_sort_scores(
        self, user_group: torch.Tensor, all_item_group: torch.Tensor
    ) -> torch.Tensor:
        scores = user_group.new_empty((user_group.shape[0], self.n_items))
        for user_start in range(0, user_group.shape[0], self.eval_user_chunk_size):
            user_stop = min(
                user_start + self.eval_user_chunk_size, user_group.shape[0]
            )
            current_users = user_group[user_start:user_stop]
            for item_start in range(0, self.n_items, self.eval_item_chunk_size):
                item_stop = min(item_start + self.eval_item_chunk_size, self.n_items)
                # [U,1,F,n,n] and [1,I,F,n,n] broadcast to [U,I,F,n,n].
                scores[
                    user_start:user_stop, item_start:item_stop
                ] = self._score_groups(
                    current_users[:, None, ...],
                    all_item_group[None, item_start:item_stop, ...],
                )
        return scores

    def _prefiltered_full_sort_scores(
        self,
        user_group: torch.Tensor,
        all_item_group: torch.Tensor,
        history_index: Any = None,
        apply_exclusions: bool = False,
    ) -> torch.Tensor:
        """Frobenius-GEMM shortlist, then exact rescoring of the shortlist.

        The surrogate ``||G_u - G_i||_F^2`` over all items reduces to one
        GEMM on the flattened group tables via the same Euclidean identity
        as the ``tangent_euclidean`` control; the exact SL scorer then runs
        only on each user's ``eval_prefilter_candidates`` nearest items.
        Every other item receives the dtype minimum, ranking strictly below
        all rescored candidates.  Padding item 0 is always excluded before
        ``topk``.  When the evaluator supplies history exclusions, those are
        also applied before shortlisting so they do not consume candidate
        capacity.  This path remains deliberately approximate: synthetic
        small-catalog recall does not establish containment for a trained
        checkpoint.  Do not use it for formal early stopping or reported
        validation/test metrics without an exhaustive masked containment
        audit.
        """

        user_count = user_group.shape[0]
        candidates = min(self.eval_prefilter_candidates, self.n_items)
        item_flat = all_item_group.reshape(self.n_items, -1)
        item_squared = item_flat.square().sum(dim=1)
        scores = user_group.new_full(
            (user_count, self.n_items),
            torch.finfo(user_group.dtype).min,
        )
        for user_start in range(0, user_count, self.eval_user_chunk_size):
            user_stop = min(user_start + self.eval_user_chunk_size, user_count)
            current_users = user_group[user_start:user_stop]
            current_flat = current_users.reshape(current_users.shape[0], -1)
            surrogate = self._gemm_squared_coordinate_distance(
                current_flat, item_flat, right_squared_norm=item_squared
            )
            # Item 0 is RecBole's padding token and is never recommendable,
            # even when this method is called directly rather than through
            # the mask-aware Trainer path.
            excluded = torch.zeros_like(surrogate, dtype=torch.bool)
            excluded[:, 0] = True
            if apply_exclusions and history_index is not None:
                history_rows, history_items = history_index
                history_rows = torch.as_tensor(history_rows, device=surrogate.device)
                history_items = torch.as_tensor(history_items, device=surrogate.device)
                in_chunk = (history_rows >= user_start) & (history_rows < user_stop)
                excluded[
                    history_rows[in_chunk] - user_start, history_items[in_chunk]
                ] = True
            surrogate.masked_fill_(excluded, torch.inf)
            shortlist = surrogate.topk(candidates, dim=1, largest=False).indices
            exact = current_users.new_empty(
                (current_users.shape[0], candidates)
            )
            for item_start in range(0, candidates, self.eval_item_chunk_size):
                item_stop = min(item_start + self.eval_item_chunk_size, candidates)
                chunk_groups = all_item_group[shortlist[:, item_start:item_stop]]
                # [U,F,n,n] against [U,I,F,n,n] broadcasts to [U,I].
                exact[:, item_start:item_stop] = self._score_groups(
                    current_users, chunk_groups
                )
            scores[user_start:user_stop].scatter_(1, shortlist, exact)
            scores[user_start:user_stop].masked_fill_(
                excluded, torch.finfo(user_group.dtype).min
            )
        return scores


__all__ = ["SLRec"]
