"""Paper-faithful clean-room GGCF for the legacy RecBole-GNN runner.

This is **not** the authors' official implementation.  It is an independent
implementation of equations (5)--(7) in "Geometric Interaction Augmented
Graph Collaborative Filtering" (arXiv:2208.01250; CIKM 2023).  In
particular it implements:

* symmetric ``D^-1/2 A D^-1/2`` propagation on the *training* user-item
  graph, without self loops;
* a Euclidean weighted sum and the closed-form Lorentz centroid in parallel;
* the complete dual-geometry interaction from equation (6), including the
  origin exponential/logarithmic maps, Lorentz distance, parallel transport,
  gyro-addition, and two unconstrained trainable scalars;
* equal fusion of layers zero through ``K`` and the paper's Euclidean plus
  learnably weighted Lorentz-inner-product decoder; and
* BPR with L2 regularisation of the sampled Euclidean parameters.

The paper says that the embedding dimension is 64 but does not state whether
that is a per-geometry width or a combined E/H parameter budget.  This module
therefore makes the accounting explicit: ``embedding_size`` is the combined
intrinsic coordinate budget and ``ggcf_branch_size`` is the width of each
geometry.  They must satisfy ``embedding_size == 2 * ggcf_branch_size``.
The default is E32 + H32 = 64; use 128 and 64 respectively to test the
per-branch-64 interpretation of the paper.

There is one manuscript ambiguity that no released code is available to
resolve.  The printed hyperbolic part of equation (7) normalises every point
inside a sum, which generally does not produce a point on the hyperboloid.
Here "equal fusion" is implemented as the equal-weight Lorentz centroid,
which is the manifold-preserving reading consistent with equation (5).

All trainable entity parameters are ordinary Euclidean tensors, as required
by section 3.4 of the paper.  Hyperboloid points use curvature -1 and ambient
coordinates ``(time, space...)``.  The small numerical projections below are
constraint repairs after analytic manifold operations, not learned layers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn

from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender


def _config_get(config: Any, key: str, default: Any) -> Any:
    try:
        value = config[key]
    except (KeyError, TypeError, AttributeError):
        return default
    return default if value is None else value


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "on"}:
            return True
        if lowered in {"false", "no", "0", "off"}:
            return False
    return bool(value)


class GGCF(GeneralGraphRecommender):
    """Euclidean/Lorentz graph collaborative filtering (clean-room)."""

    input_type = InputType.PAIRWISE

    def __init__(self, config: Any, dataset: Any) -> None:
        super().__init__(config, dataset)
        self.config = config

        self.embedding_size = int(_config_get(config, "embedding_size", 64))
        self.branch_size = int(
            _config_get(
                config,
                "ggcf_branch_size",
                self.embedding_size // 2,
            )
        )
        if self.embedding_size <= 0 or self.branch_size <= 0:
            raise ValueError("embedding_size and ggcf_branch_size must be positive")
        if self.embedding_size != 2 * self.branch_size:
            raise ValueError(
                "GGCF uses equal-width interacting E/H branches and explicit "
                "budget accounting: embedding_size must equal "
                "2 * ggcf_branch_size; received "
                f"{self.embedding_size} and {self.branch_size}"
            )

        self.n_layers = int(_config_get(config, "n_layers", 3))
        self.reg_weight = float(_config_get(config, "reg_weight", 1e-5))
        self.require_pow = _as_bool(_config_get(config, "require_pow", True))
        self.eval_item_chunk_size = int(
            _config_get(config, "eval_item_chunk_size", 4096)
        )
        self.lorentz_eps = float(_config_get(config, "lorentz_eps", 1e-7))
        self.gamma_init = float(_config_get(config, "gamma_init", 0.1))
        self.gamma_prime_init = float(
            _config_get(config, "gamma_prime_init", 0.1)
        )
        self.lambda_init = float(_config_get(config, "lambda_init", 1.0))
        self.init_std = float(_config_get(config, "init_std", 0.01))
        self.init_method = str(
            _config_get(config, "ggcf_init_method", "normal")
        ).strip().lower()
        self.hyperbolic_layer_fusion = str(
            _config_get(config, "hyperbolic_layer_fusion", "lorentz_centroid")
        ).strip().lower()

        if self.n_layers < 0:
            raise ValueError("n_layers must be non-negative")
        if self.reg_weight < 0:
            raise ValueError("reg_weight must be non-negative")
        if self.eval_item_chunk_size <= 0:
            raise ValueError("eval_item_chunk_size must be positive")
        if self.lorentz_eps <= 0:
            raise ValueError("lorentz_eps must be positive")
        if self.init_std <= 0:
            raise ValueError("init_std must be positive")
        if self.init_method not in {"normal", "xavier_uniform"}:
            raise ValueError(
                "ggcf_init_method must be one of {'normal', 'xavier_uniform'}"
            )
        if self.hyperbolic_layer_fusion != "lorentz_centroid":
            raise ValueError(
                "only hyperbolic_layer_fusion='lorentz_centroid' is supported; "
                "the literal typesetting of equation (7) does not preserve "
                "hyperboloid membership"
            )

        node_count = self.n_users + self.n_items
        # Each geometry owns an independent d-dimensional Euclidean parameter
        # table.  The hyperbolic table is mapped from T_o H^d by exp_o.
        self.euclidean_embedding = nn.Embedding(node_count, self.branch_size)
        self.hyperbolic_tangent_embedding = nn.Embedding(
            node_count, self.branch_size
        )
        self.gamma = nn.Parameter(torch.tensor(self.gamma_init))
        self.gamma_prime = nn.Parameter(torch.tensor(self.gamma_prime_init))
        self.geometry_lambda = nn.Parameter(torch.tensor(self.lambda_init))
        self._reset_parameters()

        # Rebuild from the dataset passed to the model.  The legacy runner
        # passes train_data.dataset here, so validation/test interactions can
        # never enter this adjacency.  Building it ourselves also makes the
        # no-self-loop and normalisation contracts auditable independently of
        # optional PyG/torch_sparse representations created by the base class.
        normalized_adjacency = self._build_training_adjacency(dataset)
        self.register_buffer(
            "normalized_adjacency", normalized_adjacency, persistent=False
        )

        self.bpr_loss = BPRLoss()
        self.embedding_regularizer = EmbLoss()

        # Standard RecBole full-ranking cache.  A representation contains d
        # Euclidean coordinates followed by d+1 Lorentz ambient coordinates.
        self.restore_user_e = None
        self.restore_item_e = None
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    @property
    def representation_size(self) -> int:
        """Final stored width; intrinsic budget plus one Lorentz time axis."""

        return 2 * self.branch_size + 1

    def _reset_parameters(self) -> None:
        for embedding in (
            self.euclidean_embedding,
            self.hyperbolic_tangent_embedding,
        ):
            if self.init_method == "normal":
                nn.init.normal_(embedding.weight, std=self.init_std)
            else:
                nn.init.xavier_uniform_(embedding.weight)

    def _build_training_adjacency(self, dataset: Any) -> torch.Tensor:
        """Return coalesced symmetric ``D^-1/2 A D^-1/2`` with no diagonal."""

        if hasattr(dataset, "get_interactions"):
            users, items = dataset.get_interactions()
            users = torch.as_tensor(users, dtype=torch.long, device=self.device)
            items = torch.as_tensor(items, dtype=torch.long, device=self.device)
        elif hasattr(dataset, "inter_matrix"):
            interaction = dataset.inter_matrix(form="coo")
            users = torch.as_tensor(
                interaction.row, dtype=torch.long, device=self.device
            )
            items = torch.as_tensor(
                interaction.col, dtype=torch.long, device=self.device
            )
        else:
            raise TypeError(
                "GGCF requires get_interactions() or inter_matrix('coo') "
                "on the training dataset"
            )

        if users.ndim != 1 or items.ndim != 1 or users.numel() != items.numel():
            raise ValueError("training user/item id vectors must be aligned and 1-D")
        if users.numel() > 0:
            if int(users.min()) < 0 or int(users.max()) >= self.n_users:
                raise ValueError("training user id outside the declared user range")
            if int(items.min()) < 0 or int(items.max()) >= self.n_items:
                raise ValueError("training item id outside the declared item range")

        item_nodes = items + self.n_users
        source = torch.cat((users, item_nodes), dim=0)
        target = torch.cat((item_nodes, users), dim=0)
        node_count = self.n_users + self.n_items
        degree = torch.zeros(node_count, dtype=torch.float32, device=self.device)
        degree.index_add_(
            0,
            source,
            torch.ones(source.numel(), dtype=torch.float32, device=self.device),
        )
        inverse_sqrt_degree = torch.where(
            degree > 0,
            degree.rsqrt(),
            torch.zeros_like(degree),
        )
        values = inverse_sqrt_degree[source] * inverse_sqrt_degree[target]
        adjacency = torch.sparse_coo_tensor(
            torch.stack((source, target), dim=0),
            values,
            (node_count, node_count),
            device=self.device,
            dtype=torch.float32,
        ).coalesce()
        # Bipartite offsetting makes diagonal entries impossible; retain this
        # assertion so a future graph-construction change cannot add loops.
        if adjacency._nnz() and torch.any(
            adjacency.indices()[0] == adjacency.indices()[1]
        ):
            raise RuntimeError("GGCF training adjacency must not contain self loops")
        return adjacency

    # ------------------------------------------------------------------
    # Curvature -1 Lorentz operations used by equations (5)--(7).
    # ------------------------------------------------------------------
    @staticmethod
    def _lorentz_inner(
        left: torch.Tensor,
        right: torch.Tensor,
        keepdim: bool = False,
    ) -> torch.Tensor:
        result = (
            (left[..., 1:] * right[..., 1:]).sum(dim=-1)
            - left[..., 0] * right[..., 0]
        )
        return result.unsqueeze(-1) if keepdim else result

    def _origin_like(self, reference: torch.Tensor) -> torch.Tensor:
        origin = torch.zeros_like(reference)
        origin[..., 0] = 1.0
        return origin

    def _lorentz_project(self, point: torch.Tensor) -> torch.Tensor:
        """Repair round-off while selecting the future hyperboloid sheet."""

        spatial = point[..., 1:]
        time = torch.sqrt(1.0 + spatial.square().sum(dim=-1, keepdim=True))
        return torch.cat((time, spatial), dim=-1)

    def _lorentz_exp0(self, spatial_tangent: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(
            spatial_tangent, dim=-1, keepdim=True
        )
        safe_norm = norm.clamp_min(self.lorentz_eps)
        sinhc = torch.where(
            norm > self.lorentz_eps,
            torch.sinh(norm) / safe_norm,
            1.0 + norm.square() / 6.0,
        )
        point = torch.cat(
            (torch.cosh(norm), sinhc * spatial_tangent), dim=-1
        )
        return self._lorentz_project(point)

    def _lorentz_log0(self, point: torch.Tensor) -> torch.Tensor:
        point = self._lorentz_project(point)
        spatial = point[..., 1:]
        spatial_norm = torch.linalg.vector_norm(
            spatial, dim=-1, keepdim=True
        )
        distance = torch.acosh(
            point[..., 0:1].clamp_min(1.0 + self.lorentz_eps)
        )
        scale = torch.where(
            spatial_norm > self.lorentz_eps,
            distance / spatial_norm.clamp_min(self.lorentz_eps),
            torch.ones_like(spatial_norm),
        )
        return scale * spatial

    def _lorentz_distance(
        self, left: torch.Tensor, right: torch.Tensor, keepdim: bool = False
    ) -> torch.Tensor:
        argument = -self._lorentz_inner(left, right, keepdim=keepdim)
        # A small lower guard avoids the infinite derivative of acosh at one.
        return torch.acosh(argument.clamp_min(1.0 + self.lorentz_eps))

    def _parallel_transport_origin_to(
        self, destination: torch.Tensor, spatial_tangent: torch.Tensor
    ) -> torch.Tensor:
        """Equation (6)'s exact ``P_(o->x)`` for curvature -1."""

        origin = self._origin_like(destination)
        tangent_at_origin = torch.cat(
            (torch.zeros_like(spatial_tangent[..., :1]), spatial_tangent),
            dim=-1,
        )
        coefficient = self._lorentz_inner(
            destination, tangent_at_origin, keepdim=True
        ) / (
            1.0
            - self._lorentz_inner(origin, destination, keepdim=True)
        ).clamp_min(self.lorentz_eps)
        transported = tangent_at_origin + coefficient * (origin + destination)
        # Analytically this is tangent already.  The projection removes only
        # floating-point residual in <destination, transported>_L.
        residual = self._lorentz_inner(
            destination, transported, keepdim=True
        )
        return transported + residual * destination

    def _lorentz_exp_at(
        self, base: torch.Tensor, tangent: torch.Tensor
    ) -> torch.Tensor:
        tangent_norm_squared = self._lorentz_inner(
            tangent, tangent, keepdim=True
        )
        guarded_norm = torch.sqrt(
            tangent_norm_squared.clamp_min(self.lorentz_eps**2)
        )
        tangent_norm = torch.where(
            tangent_norm_squared > self.lorentz_eps**2,
            guarded_norm,
            torch.zeros_like(guarded_norm),
        )
        safe_norm = tangent_norm.clamp_min(self.lorentz_eps)
        sinhc = torch.where(
            tangent_norm > self.lorentz_eps,
            torch.sinh(tangent_norm) / safe_norm,
            1.0 + tangent_norm.square() / 6.0,
        )
        return self._lorentz_project(
            torch.cosh(tangent_norm) * base + sinhc * tangent
        )

    def _lorentz_scalar_mul(
        self, scalar: torch.Tensor, point: torch.Tensor
    ) -> torch.Tensor:
        """``r (x) x = exp_o(r log_o(x))`` from equation (6)."""

        if scalar.ndim == point.ndim - 1:
            scalar = scalar.unsqueeze(-1)
        return self._lorentz_exp0(scalar * self._lorentz_log0(point))

    def _lorentz_gyro_add(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        """``x (+) y = exp_x(PT_(o->x)(log_o(y)))`` from equation (6)."""

        tangent = self._parallel_transport_origin_to(
            left, self._lorentz_log0(right)
        )
        return self._lorentz_exp_at(left, tangent)

    def _normalise_lorentz_ambient(
        self, ambient: torch.Tensor
    ) -> torch.Tensor:
        """Normalise a positive weighted ambient sum to the hyperboloid."""

        negative_norm_squared = -self._lorentz_inner(
            ambient, ambient, keepdim=True
        )
        valid = (
            torch.isfinite(ambient).all(dim=-1, keepdim=True)
            & (ambient[..., :1] > 0)
            & (negative_norm_squared > self.lorentz_eps)
        )
        normalized = ambient / torch.sqrt(
            negative_norm_squared.clamp_min(self.lorentz_eps)
        )
        normalized = torch.where(valid, normalized, self._origin_like(ambient))
        return self._lorentz_project(normalized)

    def _lorentz_centroid(
        self, points: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Closed-form Lorentz centroid over the leading point axis."""

        if points.ndim < 2:
            raise ValueError("Lorentz centroid requires a leading point axis")
        if weights is None:
            ambient = points.mean(dim=0)
        else:
            if weights.ndim != 1 or weights.numel() != points.shape[0]:
                raise ValueError("centroid weights must match the leading axis")
            shape = (weights.numel(),) + (1,) * (points.ndim - 1)
            normalized_weights = weights / weights.sum().clamp_min(
                self.lorentz_eps
            )
            ambient = (points * normalized_weights.reshape(shape)).sum(dim=0)
        return self._normalise_lorentz_ambient(ambient)

    @staticmethod
    def _graph_mm(adjacency: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if adjacency.layout == torch.strided:
            return adjacency @ values
        return torch.sparse.mm(adjacency, values)

    def _dual_geometry_interaction(
        self, euclidean: torch.Tensor, hyperbolic: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Equation (6), without tangent-space shortcuts."""

        hyperbolic_log = self._lorentz_log0(hyperbolic)
        euclidean_distance = torch.linalg.vector_norm(
            euclidean - hyperbolic_log, dim=-1, keepdim=True
        )
        fused_euclidean = (
            euclidean
            + self.gamma * euclidean_distance * hyperbolic_log
        )

        euclidean_on_hyperboloid = self._lorentz_exp0(euclidean)
        hyperbolic_distance = self._lorentz_distance(
            hyperbolic, euclidean_on_hyperboloid, keepdim=True
        )
        scaled_euclidean_point = self._lorentz_scalar_mul(
            self.gamma_prime * hyperbolic_distance,
            euclidean_on_hyperboloid,
        )
        fused_hyperbolic = self._lorentz_gyro_add(
            hyperbolic, scaled_euclidean_point
        )
        return fused_euclidean, fused_hyperbolic

    def _encode_all_layers(
        self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        euclidean = self.euclidean_embedding.weight
        hyperbolic = self._lorentz_exp0(
            self.hyperbolic_tangent_embedding.weight
        )
        euclidean_layers = [euclidean]
        hyperbolic_layers = [hyperbolic]

        for _ in range(self.n_layers):
            aggregated_euclidean = self._graph_mm(
                self.normalized_adjacency, euclidean
            )
            hyperbolic_ambient = self._graph_mm(
                self.normalized_adjacency, hyperbolic
            )
            aggregated_hyperbolic = self._normalise_lorentz_ambient(
                hyperbolic_ambient
            )
            euclidean, hyperbolic = self._dual_geometry_interaction(
                aggregated_euclidean, aggregated_hyperbolic
            )
            euclidean_layers.append(euclidean)
            hyperbolic_layers.append(hyperbolic)
        return euclidean_layers, hyperbolic_layers

    def _fuse_layers(
        self,
        euclidean_layers: Sequence[torch.Tensor],
        hyperbolic_layers: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(euclidean_layers) != self.n_layers + 1:
            raise ValueError("expected layer zero through K for equal fusion")
        euclidean = torch.stack(tuple(euclidean_layers), dim=0).mean(dim=0)
        hyperbolic = self._lorentz_centroid(
            torch.stack(tuple(hyperbolic_layers), dim=0)
        )
        return torch.cat((euclidean, hyperbolic), dim=-1)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        euclidean_layers, hyperbolic_layers = self._encode_all_layers()
        representation = self._fuse_layers(
            euclidean_layers, hyperbolic_layers
        )
        return torch.split(
            representation, [self.n_users, self.n_items], dim=0
        )

    def _split_representation(
        self, representation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            representation[..., : self.branch_size],
            representation[..., self.branch_size :],
        )

    def _pair_score(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        left_e, left_h = self._split_representation(left)
        right_e, right_h = self._split_representation(right)
        euclidean_score = (left_e * right_e).sum(dim=-1)
        hyperbolic_score = self._lorentz_inner(left_h, right_h)
        return euclidean_score + self.geometry_lambda * hyperbolic_score

    def _clear_full_sort_cache(self) -> None:
        self.restore_user_e = None
        self.restore_item_e = None

    def calculate_loss(self, interaction: Any) -> torch.Tensor:
        self._clear_full_sort_cache()
        users, items = self.forward()
        user_ids = interaction[self.USER_ID]
        positive_ids = interaction[self.ITEM_ID]
        negative_ids = interaction[self.NEG_ITEM_ID]

        user = users[user_ids]
        positive = items[positive_ids]
        negative = items[negative_ids]
        positive_score = self._pair_score(user, positive)
        negative_user = user
        if negative.ndim > user.ndim:
            negative_user = user.unsqueeze(-2)
            positive_score = positive_score.unsqueeze(-1)
        negative_score = self._pair_score(negative_user, negative)
        ranking_loss = self.bpr_loss(positive_score, negative_score)

        user_nodes = user_ids
        positive_nodes = positive_ids + self.n_users
        negative_nodes = negative_ids + self.n_users
        regularization = self.embedding_regularizer(
            self.euclidean_embedding(user_nodes),
            self.euclidean_embedding(positive_nodes),
            self.euclidean_embedding(negative_nodes),
            self.hyperbolic_tangent_embedding(user_nodes),
            self.hyperbolic_tangent_embedding(positive_nodes),
            self.hyperbolic_tangent_embedding(negative_nodes),
            require_pow=self.require_pow,
        )
        return ranking_loss + self.reg_weight * regularization.squeeze()

    def predict(self, interaction: Any) -> torch.Tensor:
        users, items = self.forward()
        return self._pair_score(
            users[interaction[self.USER_ID]],
            items[interaction[self.ITEM_ID]],
        )

    def _full_sort_scores(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        user_e, user_h = self._split_representation(users)
        chunks = []
        for start in range(0, items.shape[0], self.eval_item_chunk_size):
            item_chunk = items[start : start + self.eval_item_chunk_size]
            item_e, item_h = self._split_representation(item_chunk)
            euclidean_score = user_e @ item_e.transpose(0, 1)
            lorentz_score = (
                user_h[..., 1:] @ item_h[..., 1:].transpose(0, 1)
                - user_h[..., :1] @ item_h[..., :1].transpose(0, 1)
            )
            chunks.append(
                euclidean_score + self.geometry_lambda * lorentz_score
            )
        return torch.cat(chunks, dim=1)

    def full_sort_predict(self, interaction: Any) -> torch.Tensor:
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        requested = self.restore_user_e[interaction[self.USER_ID]]
        scores = self._full_sort_scores(requested, self.restore_item_e)
        if self.config["tail_analysis"] is True:
            return self.head_item, self.tail_item, scores.reshape(-1)
        if self.config["popularity_analysis"] is True:
            return (
                self.rank1item,
                self.rank2item,
                self.rank3item,
                self.rank4item,
                self.rank5item,
                scores.reshape(-1),
            )
        return scores.reshape(-1)

    @torch.no_grad()
    def geometry_diagnostics(self) -> Dict[str, Any]:
        """Check initial, propagated, and fused H points independently."""

        euclidean_layers, hyperbolic_layers = self._encode_all_layers()
        final = self._fuse_layers(euclidean_layers, hyperbolic_layers)
        _, final_hyperbolic = self._split_representation(final)

        def membership(points: torch.Tensor) -> Dict[str, float]:
            quadratic = self._lorentz_inner(points, points)
            return {
                "max_abs_quadratic_error": float(
                    (quadratic + 1.0).abs().max().cpu()
                ),
                "min_time_coordinate": float(points[..., 0].min().cpu()),
                "nonfinite_points": int(
                    (~torch.isfinite(points).all(dim=-1)).sum().cpu()
                ),
            }

        return {
            "implementation": "paper_faithful_clean_room_no_official_code",
            "intrinsic_coordinate_budget": self.embedding_size,
            "euclidean_branch_size": self.branch_size,
            "hyperbolic_branch_size": self.branch_size,
            "hyperbolic_ambient_size": self.branch_size + 1,
            "n_layers": self.n_layers,
            "layer_fusion": self.hyperbolic_layer_fusion,
            "layer_membership": [membership(x) for x in hyperbolic_layers],
            "final_membership": membership(final_hyperbolic),
        }

    def projection_diagnostics(self) -> Dict[str, Any]:
        return self.geometry_diagnostics()


__all__ = ["GGCF"]
