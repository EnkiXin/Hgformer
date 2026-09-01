"""Paper-derived clean-room implementation of AGCF (WWW 2026).

This is *not* the authors' official code.  It implements equations (6)--(22)
from "Learning on Adaptive Manifolds for Graph Collaborative Filtering":

* one learnable node-position table ``Q`` and ``P(0) = Pnet(Q)``;
* node-dependent inverse metrics ``A_v A_v^T + epsilon I``;
* ``M = (1 + delta) I + D^-1/2 A D^-1/2`` and a global SPD
  channel metric ``S``;
* damped Hamiltonian dynamics integrated with Symplectic Euler;
* the sum of the ``L + 1`` output positions;
* squared Mahalanobis margin ranking and negative-distance prediction.

The paper does not publish the MLP architecture, hidden widths, metric ranks,
initialisation, loss reduction, or dataset-specific dynamics settings.  Those
choices are therefore explicit, tunable clean-room defaults here rather than
claims about the unreleased reference implementation.

No ``Nd x Nd`` metric, ``M (x) S``, or dense graph matrix is materialised.
For the specified two-layer tanh metric MLP, the geometric force is evaluated
by an exact analytic vector-Jacobian product.  This is algebraically identical
to autograd of ``y^T G(x)^-1 y`` while avoiding nested autograd and continuing
to propagate training gradients into positions, momenta, and metric weights.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

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


class _AdaptiveInverseMetric(nn.Module):
    """Two-layer metric MLP with an exact differentiable geometric force."""

    def __init__(self, dimension: int, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.rank = rank
        self.input_layer = nn.Linear(dimension, hidden_size)
        self.factor_layer = nn.Linear(hidden_size, dimension * rank)

    def factor(self, position: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.input_layer(position))
        return self.factor_layer(hidden).reshape(
            *position.shape[:-1], self.dimension, self.rank
        )

    def velocity_and_geometric_force(
        self,
        position: torch.Tensor,
        momentum: torch.Tensor,
        epsilon: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``G^-1 y`` and ``grad_x(y^T G^-1 y)``.

        If ``A(x)`` is the MLP output and ``c = A(x)^T y``, then
        ``y^T G^-1 y = ||c||^2 + epsilon ||y||^2`` and its cotangent at
        ``A`` is ``2 y c^T``.  Back-propagating that cotangent through the
        explicit two-layer tanh MLP yields the force below using ordinary
        differentiable tensor operations.  In particular, no ``detach`` is
        used, so an outer ranking-loss backward pass still trains every input
        and parameter involved in the dynamics.
        """

        preactivation = self.input_layer(position)
        hidden = torch.tanh(preactivation)
        raw_factor = self.factor_layer(hidden)
        factor = raw_factor.reshape(
            *position.shape[:-1], self.dimension, self.rank
        )

        projected_momentum = torch.einsum("...dr,...d->...r", factor, momentum)
        velocity = (
            torch.einsum("...dr,...r->...d", factor, projected_momentum)
            + epsilon * momentum
        )

        factor_cotangent = (
            2.0
            * momentum.unsqueeze(-1)
            * projected_momentum.unsqueeze(-2)
        )
        raw_cotangent = factor_cotangent.reshape(
            *position.shape[:-1], self.dimension * self.rank
        )
        hidden_cotangent = raw_cotangent @ self.factor_layer.weight
        preactivation_cotangent = hidden_cotangent * (1.0 - hidden.square())
        geometric_force = preactivation_cotangent @ self.input_layer.weight
        return velocity, geometric_force

    def velocity(
        self,
        position: torch.Tensor,
        momentum: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        factor = self.factor(position)
        projected = torch.einsum("...dr,...d->...r", factor, momentum)
        return torch.einsum("...dr,...r->...d", factor, projected) + epsilon * momentum


class AGCF(GeneralGraphRecommender):
    """Adaptive Geometric Collaborative Filtering, paper-derived baseline."""

    input_type = InputType.PAIRWISE
    # Geometry adapters may fix the Hamiltonian chart dimension while reusing
    # the paper-derived dynamics.  ``None`` preserves AGCF's configured width.
    POSITION_DIMENSION = None

    def __init__(self, config: Any, dataset: Any) -> None:
        super().__init__(config, dataset)
        self.config = config
        configured_dimension = int(_config_get(config, "embedding_size", 64))
        self.embedding_size = (
            configured_dimension
            if self.POSITION_DIMENSION is None
            else int(self.POSITION_DIMENSION)
        )
        self.metric_rank = int(_config_get(config, "metric_rank", 4))
        self.metric_hidden_size = int(
            _config_get(config, "metric_hidden_size", self.embedding_size)
        )
        self.pnet_hidden_size = int(
            _config_get(config, "pnet_hidden_size", self.embedding_size)
        )
        self.channel_rank = int(
            _config_get(config, "channel_rank", self.metric_rank)
        )
        self.metric_epsilon = float(
            _config_get(config, "metric_epsilon", 1e-3)
        )
        self.delta = float(_config_get(config, "structural_delta", 1e-3))
        self.potential_strength = float(
            _config_get(config, "potential_strength", 0.1)
        )
        self.damping = float(_config_get(config, "damping", 0.01))
        self.evolution_time = float(
            _config_get(config, "evolution_time", 1.0)
        )
        self.output_steps = int(_config_get(config, "output_steps", 1))
        self.integration_steps = int(
            _config_get(config, "integration_steps", 1)
        )
        self.loss_margin = float(_config_get(config, "margin", 0.1))
        self.dynamics_chunk_size = int(
            _config_get(config, "dynamics_chunk_size", 4096)
        )
        self.checkpoint_dynamics = _as_bool(
            _config_get(config, "checkpoint_dynamics", True)
        )
        self.eval_item_chunk_size = int(
            _config_get(config, "eval_item_chunk_size", 4096)
        )

        positive_integers = {
            "embedding_size": self.embedding_size,
            "metric_rank": self.metric_rank,
            "metric_hidden_size": self.metric_hidden_size,
            "pnet_hidden_size": self.pnet_hidden_size,
            "channel_rank": self.channel_rank,
            "integration_steps": self.integration_steps,
            "dynamics_chunk_size": self.dynamics_chunk_size,
            "eval_item_chunk_size": self.eval_item_chunk_size,
        }
        for name, value in positive_integers.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.output_steps < 0:
            raise ValueError("output_steps must be non-negative")
        for name, value in {
            "metric_epsilon": self.metric_epsilon,
            "structural_delta": self.delta,
            "potential_strength": self.potential_strength,
            "damping": self.damping,
            "evolution_time": self.evolution_time,
            "margin": self.loss_margin,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.metric_epsilon == 0:
            raise ValueError("metric_epsilon must be strictly positive for SPD metrics")
        if self.evolution_time == 0 and self.output_steps > 0:
            raise ValueError("evolution_time must be positive when output_steps > 0")

        node_count = self.n_users + self.n_items
        self.node_embedding = nn.Embedding(node_count, self.embedding_size)
        self.pnet = nn.Sequential(
            nn.Linear(self.embedding_size, self.pnet_hidden_size),
            nn.Tanh(),
            nn.Linear(self.pnet_hidden_size, self.embedding_size),
        )
        self.inverse_metric = _AdaptiveInverseMetric(
            self.embedding_size, self.metric_hidden_size, self.metric_rank
        )
        # S = B B^T + epsilon I, implemented through its thin factor B.
        self.channel_factor = nn.Parameter(
            torch.empty(self.embedding_size, self.channel_rank)
        )

        self._reset_parameters()
        normalized_adjacency = self._native_sparse_adjacency(
            self.edge_index, self.edge_weight, node_count
        )
        self.register_buffer(
            "normalized_adjacency", normalized_adjacency, persistent=False
        )

        self.restore_user_e = None
        self.restore_item_e = None
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_embedding.weight)
        for module in (self.pnet, self.inverse_metric):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.channel_factor)

    @staticmethod
    def _native_sparse_adjacency(
        edge_index: Any,
        edge_weight: Any,
        node_count: int,
    ) -> torch.Tensor:
        """Convert PyG/torch_sparse graph storage to native sparse COO."""

        if isinstance(edge_index, torch.Tensor):
            if edge_index.layout != torch.strided:
                return edge_index.to_sparse_coo().coalesce()
            indices = edge_index.long()
            values = edge_weight
        elif hasattr(edge_index, "coo"):
            row, col, values = edge_index.coo()
            indices = torch.stack((row, col), dim=0).long()
        else:
            raise TypeError("unsupported normalized adjacency representation")
        if values is None:
            values = torch.ones(
                indices.shape[1], device=indices.device, dtype=torch.float32
            )
        return torch.sparse_coo_tensor(
            indices,
            values,
            (node_count, node_count),
            device=indices.device,
            dtype=values.dtype,
        ).coalesce()

    def _channel_metric_action(self, vectors: torch.Tensor) -> torch.Tensor:
        projected = vectors @ self.channel_factor
        return projected @ self.channel_factor.transpose(0, 1) + self.metric_epsilon * vectors

    def _structural_channel_force(self, position: torch.Tensor) -> torch.Tensor:
        # (M (x) S) vec(Q) == M Q S for row-wise node coordinates.
        structural = (1.0 + self.delta) * position + torch.sparse.mm(
            self.normalized_adjacency, position
        )
        return self._channel_metric_action(structural)

    def _velocity_and_geometric_force(
        self, position: torch.Tensor, momentum: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        velocities = []
        forces = []
        for start in range(0, position.shape[0], self.dynamics_chunk_size):
            stop = min(start + self.dynamics_chunk_size, position.shape[0])
            velocity, force = self.inverse_metric.velocity_and_geometric_force(
                position[start:stop], momentum[start:stop], self.metric_epsilon
            )
            velocities.append(velocity)
            forces.append(force)
        return torch.cat(velocities, dim=0), torch.cat(forces, dim=0)

    def _metric_velocity(
        self, position: torch.Tensor, momentum: torch.Tensor
    ) -> torch.Tensor:
        velocities = []
        for start in range(0, position.shape[0], self.dynamics_chunk_size):
            stop = min(start + self.dynamics_chunk_size, position.shape[0])
            velocities.append(
                self.inverse_metric.velocity(
                    position[start:stop], momentum[start:stop], self.metric_epsilon
                )
            )
        return torch.cat(velocities, dim=0)

    def _symplectic_step(
        self, position: torch.Tensor, momentum: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        step_size = self.evolution_time / self.integration_steps
        old_velocity, geometric_force = self._velocity_and_geometric_force(
            position, momentum
        )
        interaction_force = self._structural_channel_force(position)
        next_momentum = momentum - step_size * (
            0.5 * geometric_force
            + self.potential_strength * interaction_force
            + self.damping * old_velocity
        )
        # Equation (17) uses G(x_k)^-1 with the already-updated y_(k+1).
        next_position = position + step_size * self._metric_velocity(
            position, next_momentum
        )
        return next_position, next_momentum

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        position = self.node_embedding.weight
        momentum = self.pnet(position)
        output_positions = [position]
        for _ in range(self.output_steps):
            for _ in range(self.integration_steps):
                if (
                    self.checkpoint_dynamics
                    and self.training
                    and torch.is_grad_enabled()
                ):
                    position, momentum = checkpoint(
                        self._symplectic_step,
                        position,
                        momentum,
                        use_reentrant=False,
                    )
                else:
                    position, momentum = self._symplectic_step(position, momentum)
            output_positions.append(position)
        final = torch.stack(output_positions, dim=0).sum(dim=0)
        return torch.split(final, [self.n_users, self.n_items], dim=0)

    def _squared_distance(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        difference = left - right
        squared = (difference * self._channel_metric_action(difference)).sum(dim=-1)
        return squared.clamp_min(0.0)

    def _clear_full_sort_cache(self) -> None:
        self.restore_user_e = None
        self.restore_item_e = None

    def calculate_loss(self, interaction: Any) -> torch.Tensor:
        self._clear_full_sort_cache()
        users, items = self.forward()
        user = users[interaction[self.USER_ID]]
        positive = items[interaction[self.ITEM_ID]]
        negative = items[interaction[self.NEG_ITEM_ID]]
        positive_distance = self._squared_distance(user, positive)
        negative_user = user
        if negative.ndim > user.ndim:
            negative_user = user.unsqueeze(-2)
            positive_distance = positive_distance.unsqueeze(-1)
        negative_distance = self._squared_distance(negative_user, negative)
        return torch.relu(
            positive_distance - negative_distance + self.loss_margin
        ).mean()

    def predict(self, interaction: Any) -> torch.Tensor:
        users, items = self.forward()
        return -self._squared_distance(
            users[interaction[self.USER_ID]], items[interaction[self.ITEM_ID]]
        )

    def _full_sort_scores(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        user_metric = self._channel_metric_action(users)
        user_norm = (users * user_metric).sum(dim=-1, keepdim=True)
        chunks = []
        for start in range(0, items.shape[0], self.eval_item_chunk_size):
            item_chunk = items[start : start + self.eval_item_chunk_size]
            item_metric = self._channel_metric_action(item_chunk)
            item_norm = (item_chunk * item_metric).sum(dim=-1).unsqueeze(0)
            squared_distance = (
                user_norm
                + item_norm
                - 2.0 * (user_metric @ item_chunk.transpose(0, 1))
            ).clamp_min(0.0)
            chunks.append(-squared_distance)
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
    def geometry_diagnostics(self, sample_nodes: int = 32) -> Dict[str, float]:
        """Audit sampled local/global SPD matrices (diagnostics only)."""

        count = min(int(sample_nodes), self.node_embedding.num_embeddings)
        factors = self.inverse_metric.factor(self.node_embedding.weight[:count])
        identity = torch.eye(
            self.embedding_size,
            device=factors.device,
            dtype=factors.dtype,
        )
        local_inverse = factors @ factors.transpose(-1, -2) + self.metric_epsilon * identity
        channel_metric = (
            self.channel_factor @ self.channel_factor.transpose(0, 1)
            + self.metric_epsilon * identity
        )
        local_eigenvalues = torch.linalg.eigvalsh(local_inverse)
        channel_eigenvalues = torch.linalg.eigvalsh(channel_metric)
        return {
            "sample_nodes": count,
            "local_inverse_min_eigenvalue": float(local_eigenvalues.min().cpu()),
            "local_inverse_max_eigenvalue": float(local_eigenvalues.max().cpu()),
            "channel_min_eigenvalue": float(channel_eigenvalues.min().cpu()),
            "channel_max_eigenvalue": float(channel_eigenvalues.max().cpu()),
        }

    def projection_diagnostics(self) -> Dict[str, float]:
        # The shared runner already records this conventional hook name.
        return self.geometry_diagnostics()
