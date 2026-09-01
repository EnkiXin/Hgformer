"""Mixed-curvature recommendation models for controlled RecBole experiments.

This module is intentionally a *controlled adaptation*: it combines hyperbolic,
Euclidean, and spherical latent branches under one parameter budget, but it does
not claim to reproduce CGCF, HyDRA, or any other published system.  In
particular, it omits method-specific graph encoders, knowledge-graph objectives,
and data augmentation so that the effect of latent geometry can be isolated.

The implementation targets the public RecBole 1.2.1 ``GeneralRecommender`` API.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


def _project_tangent(x: Tensor, max_norm: Optional[float], eps: float) -> Tensor:
    """Project tangent vectors to a radius without changing their direction."""

    if max_norm is None:
        return x
    if max_norm <= 0:
        raise ValueError("tangent_clip must be positive or None")
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_norm) / norm.clamp_min(eps), max=1.0)
    return x * scale


def poincare_expmap0(
    tangent: Tensor,
    curvature: float = 1.0,
    tangent_clip: Optional[float] = None,
    eps: float = 1e-8,
) -> Tensor:
    r"""Map origin-tangent coordinates to a Poincare ball.

    ``curvature`` is the positive magnitude :math:`c` of curvature ``-c``.
    The convention is ``exp_0(v)=tanh(sqrt(c)||v||) v/(sqrt(c)||v||)``.
    """

    if curvature <= 0:
        raise ValueError("hyperbolic_curvature must be positive")
    tangent = _project_tangent(tangent, tangent_clip, eps)
    sqrt_c = math.sqrt(float(curvature))
    norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    scaled_norm = sqrt_c * norm
    # The clamped denominator keeps the unselected torch.where branch finite.
    ratio = torch.tanh(scaled_norm) / scaled_norm.clamp_min(eps)
    ratio = torch.where(
        scaled_norm > 1e-4,
        ratio,
        1.0 - scaled_norm.square() / 3.0,
    )
    point = ratio * tangent

    # Roundoff and low-precision kernels must not put a point on the boundary.
    max_ball_norm = (1.0 - 4.0 * eps) / sqrt_c
    point_norm = torch.linalg.vector_norm(point, dim=-1, keepdim=True)
    point_scale = torch.clamp(
        max_ball_norm / point_norm.clamp_min(eps), max=1.0
    )
    return point * point_scale


def poincare_distance_sq(
    x: Tensor,
    y: Tensor,
    curvature: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    r"""Squared geodesic distance in the Poincare ball of curvature ``-c``.

    A short series for ``acosh(1+q)^2`` removes the singular autograd product at
    coincident points while retaining an exact zero self-distance.
    """

    if curvature <= 0:
        raise ValueError("hyperbolic_curvature must be positive")
    c = float(curvature)
    x_sq = x.square().sum(dim=-1)
    y_sq = y.square().sum(dim=-1)
    diff_sq = (x - y).square().sum(dim=-1)
    denominator = ((1.0 - c * x_sq) * (1.0 - c * y_sq)).clamp_min(eps)
    q = (2.0 * c * diff_sq / denominator).clamp_min(0.0)

    threshold = 1e-4
    # acosh(1 + q)^2 = 2q - q^2/3 + 4q^3/45 + O(q^4).
    small = 2.0 * q - q.square() / 3.0 + 4.0 * q.pow(3) / 45.0
    safe_q = q.clamp_min(threshold)
    large = torch.acosh(1.0 + safe_q).square()
    return torch.where(q < threshold, small, large) / c


def poincare_distance(
    x: Tensor,
    y: Tensor,
    curvature: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Geodesic distance in a Poincare ball."""

    return poincare_distance_sq(x, y, curvature, eps).clamp_min(0.0).sqrt()


def sphere_expmap0(
    tangent: Tensor,
    tangent_clip: Optional[float] = None,
    eps: float = 1e-8,
) -> Tensor:
    r"""Map ``d`` tangent coordinates at the north pole to the unit ``S^d``.

    The returned point has ``d + 1`` ambient coordinates but only ``d`` learned
    degrees of freedom, so parameter-budget comparisons use the tangent size.
    """

    tangent = _project_tangent(tangent, tangent_clip, eps)
    norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    # torch.sinc(z) = sin(pi*z)/(pi*z), including its stable value at zero.
    tangential = torch.sinc(norm / math.pi) * tangent
    north = torch.cos(norm)
    return torch.cat((tangential, north), dim=-1)


def spherical_distance_sq(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """Squared great-circle distance between points on a unit sphere."""

    x = F.normalize(x, p=2, dim=-1, eps=eps)
    y = F.normalize(y, p=2, dim=-1, eps=eps)
    dot = (x * y).sum(dim=-1).clamp(-1.0, 1.0)
    q = (1.0 - dot).clamp(0.0, 2.0)

    threshold = 1e-4
    # acos(1 - q)^2 = 2q + q^2/3 + 4q^3/45 + O(q^4).
    small = 2.0 * q + q.square() / 3.0 + 4.0 * q.pow(3) / 45.0
    safe_q = q.clamp(min=threshold, max=2.0 - threshold)
    large = torch.acos(1.0 - safe_q).square()
    return torch.where(q < threshold, small, large)


def spherical_distance(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """Great-circle distance between points on a unit sphere."""

    return spherical_distance_sq(x, y, eps).clamp_min(0.0).sqrt()


def euclidean_distance_sq(x: Tensor, y: Tensor) -> Tensor:
    """Squared Euclidean distance along the last axis."""

    return (x - y).square().sum(dim=-1)


def _config_get(config, key: str, default):
    """Read RecBole Config objects and plain mappings with the same semantics."""

    try:
        value = config[key]
    except (KeyError, TypeError, AttributeError):
        return default
    return default if value is None else value


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _resolve_branch_dims(
    total: int,
    hyperbolic: Optional[int],
    euclidean: Optional[int],
    spherical: Optional[int],
) -> Tuple[int, int, int]:
    """Resolve omitted dimensions while preserving the exact total budget."""

    if total <= 0:
        raise ValueError("embedding_size must be positive")
    values = [hyperbolic, euclidean, spherical]
    values = [None if value is None else int(value) for value in values]
    if any(value is not None and value < 0 for value in values):
        raise ValueError("branch dimensions cannot be negative")

    known = sum(value for value in values if value is not None)
    missing = [index for index, value in enumerate(values) if value is None]
    remaining = total - known
    if remaining < 0:
        raise ValueError("branch dimensions exceed embedding_size")
    if missing:
        base, extra = divmod(remaining, len(missing))
        for offset, index in enumerate(missing):
            values[index] = base + int(offset < extra)
    elif remaining != 0:
        raise ValueError(
            "hyperbolic_dim + euclidean_dim + spherical_dim must equal "
            "embedding_size"
        )

    resolved = tuple(int(value) for value in values)
    if sum(resolved) != total or not any(resolved):
        raise ValueError("at least one branch must be active and dimensions must sum")
    return resolved  # type: ignore[return-value]


class MixedGeoRec(GeneralRecommender):
    """Equal-budget H/E/S pairwise recommender with optional entity-wise gates.

    The default 64 learned coordinates are split across Poincare, Euclidean, and
    unit-spherical branches.  Setting ``spherical_dim: 0`` and leaving the other
    two dimensions unspecified automatically produces a 32+32 H/E model.

    Gate modes:
        ``global`` (default): one learned branch-weight vector shared by all
        pairs; ``popularity``: add an MLP over normalized user/item frequency;
        ``entity`` or ``learned``: add learned user/item gate embeddings;
        ``uniform``: fixed equal weights.

    This class is a controlled mixed-geometry adaptation, not an official
    reproduction of CGCF, HyDRA, or their training objectives.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.embedding_size = int(_config_get(config, "embedding_size", 64))
        h_value = _config_get(config, "hyperbolic_dim", None)
        e_value = _config_get(config, "euclidean_dim", None)
        s_value = _config_get(config, "spherical_dim", None)
        self.hyperbolic_dim, self.euclidean_dim, self.spherical_dim = (
            _resolve_branch_dims(
                self.embedding_size,
                h_value,
                e_value,
                s_value,
            )
        )
        self.branch_dims: Dict[str, int] = {
            "hyperbolic": self.hyperbolic_dim,
            "euclidean": self.euclidean_dim,
            "spherical": self.spherical_dim,
        }
        self.active_branches = tuple(
            name for name, dim in self.branch_dims.items() if dim > 0
        )
        self.num_branches = len(self.active_branches)

        self.hyperbolic_curvature = float(
            _config_get(config, "hyperbolic_curvature", 1.0)
        )
        if self.hyperbolic_curvature <= 0:
            raise ValueError("hyperbolic_curvature must be positive")
        tangent_clip = _config_get(config, "tangent_clip", 2.0)
        self.tangent_clip = None if tangent_clip is None else float(tangent_clip)
        self.coordinate_scale = float(_config_get(config, "coordinate_scale", 1.0))
        self.distance_eps = float(_config_get(config, "distance_eps", 1e-8))
        self.reg_weight = float(_config_get(config, "reg_weight", 1e-6))
        self.eval_item_chunk_size = int(
            _config_get(config, "eval_item_chunk_size", 4096)
        )
        if self.eval_item_chunk_size <= 0:
            raise ValueError("eval_item_chunk_size must be positive")

        self.gate_mode = str(_config_get(config, "gate_mode", "global")).lower()
        if self.gate_mode == "learned":
            self.gate_mode = "entity"
        allowed_modes = {"global", "popularity", "entity", "uniform"}
        if self.gate_mode not in allowed_modes:
            raise ValueError(
                f"gate_mode must be one of {sorted(allowed_modes)}, got "
                f"{self.gate_mode!r}"
            )
        self.gate_temperature = float(_config_get(config, "gate_temperature", 1.0))
        if self.gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        # "global" is fixed across pairs but remains learnable during training.
        self.global_gate_logits = nn.Parameter(torch.zeros(self.num_branches))

        if self.gate_mode == "entity":
            self.user_gate_embedding = nn.Embedding(self.n_users, self.num_branches)
            self.item_gate_embedding = nn.Embedding(self.n_items, self.num_branches)
        else:
            self.user_gate_embedding = None
            self.item_gate_embedding = None

        if self.gate_mode == "popularity":
            hidden_size = int(_config_get(config, "popularity_gate_hidden", 8))
            if hidden_size <= 0:
                raise ValueError("popularity_gate_hidden must be positive")
            self.popularity_gate = nn.Sequential(
                nn.Linear(4, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, self.num_branches, bias=False),
            )
        else:
            self.popularity_gate = None

        user_popularity, item_popularity = self._dataset_popularity(dataset)
        self.register_buffer("user_popularity", user_popularity, persistent=True)
        self.register_buffer("item_popularity", item_popularity, persistent=True)

        self.score_scale = float(_config_get(config, "score_scale", 1.0))
        learnable_scale = _as_bool(
            _config_get(config, "learnable_score_scale", False)
        )
        initial_log_scale = math.log(max(self.score_scale, self.distance_eps))
        if learnable_scale:
            self.log_score_scale = nn.Parameter(torch.tensor(initial_log_scale))
        else:
            self.register_buffer(
                "log_score_scale", torch.tensor(initial_log_scale), persistent=True
            )

        self.bpr_loss = BPRLoss()
        self._reset_parameters(float(_config_get(config, "initializer_range", 0.05)))

    def _reset_parameters(self, std: float) -> None:
        if std <= 0:
            raise ValueError("initializer_range must be positive")
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=std)
        nn.init.zeros_(self.global_gate_logits)
        if self.user_gate_embedding is not None:
            nn.init.zeros_(self.user_gate_embedding.weight)
            nn.init.zeros_(self.item_gate_embedding.weight)
        if self.popularity_gate is not None:
            for module in self.popularity_gate.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _dataset_popularity(self, dataset) -> Tuple[Tensor, Tensor]:
        """Return normalized log-counts, falling back to zeros for toy datasets."""

        user_counts = torch.zeros(self.n_users, dtype=torch.float32)
        item_counts = torch.zeros(self.n_items, dtype=torch.float32)
        try:
            matrix = dataset.inter_matrix(form="coo").tocoo()
            rows = torch.as_tensor(matrix.row, dtype=torch.long)
            columns = torch.as_tensor(matrix.col, dtype=torch.long)
            user_counts = torch.bincount(rows, minlength=self.n_users).float()
            item_counts = torch.bincount(columns, minlength=self.n_items).float()
        except (AttributeError, TypeError, ValueError):
            # Geometry-only unit tests and custom datasets need not expose a
            # sparse interaction matrix when popularity gating is unused.
            pass

        def normalize(counts: Tensor) -> Tensor:
            logged = torch.log1p(counts)
            return logged / logged.max().clamp_min(1.0)

        return normalize(user_counts), normalize(item_counts)

    def _split_coordinates(self, coordinates: Tensor) -> Dict[str, Tensor]:
        chunks: Dict[str, Tensor] = {}
        start = 0
        for name, dim in self.branch_dims.items():
            if dim > 0:
                chunks[name] = coordinates[..., start : start + dim]
                start += dim
        return chunks

    def _gate_weights(self, user: Tensor, item: Tensor) -> Tensor:
        pair_shape = user.shape
        logits = self.global_gate_logits.expand(*pair_shape, self.num_branches)
        if self.gate_mode == "uniform":
            return torch.full_like(logits, 1.0 / self.num_branches)
        if self.gate_mode == "entity":
            assert self.user_gate_embedding is not None
            assert self.item_gate_embedding is not None
            logits = logits + 0.5 * (
                self.user_gate_embedding(user) + self.item_gate_embedding(item)
            )
        elif self.gate_mode == "popularity":
            assert self.popularity_gate is not None
            user_pop = self.user_popularity[user]
            item_pop = self.item_popularity[item]
            features = torch.stack(
                (
                    user_pop,
                    item_pop,
                    (user_pop - item_pop).abs(),
                    user_pop * item_pop,
                ),
                dim=-1,
            )
            logits = logits + self.popularity_gate(features)
        return torch.softmax(logits / self.gate_temperature, dim=-1)

    def branch_distances(self, user_coordinates: Tensor, item_coordinates: Tensor) -> Tensor:
        """Return dimension-normalized squared distances for active branches."""

        user_parts = self._split_coordinates(user_coordinates)
        item_parts = self._split_coordinates(item_coordinates)
        distances = []
        scale = self.coordinate_scale
        for name in self.active_branches:
            dim = self.branch_dims[name]
            user_part = user_parts[name] * scale
            item_part = item_parts[name] * scale
            if name == "hyperbolic":
                user_point = poincare_expmap0(
                    user_part,
                    self.hyperbolic_curvature,
                    self.tangent_clip,
                    self.distance_eps,
                )
                item_point = poincare_expmap0(
                    item_part,
                    self.hyperbolic_curvature,
                    self.tangent_clip,
                    self.distance_eps,
                )
                distance = poincare_distance_sq(
                    user_point,
                    item_point,
                    self.hyperbolic_curvature,
                    self.distance_eps,
                )
            elif name == "euclidean":
                distance = euclidean_distance_sq(user_part, item_part)
            else:
                user_point = sphere_expmap0(
                    user_part, self.tangent_clip, self.distance_eps
                )
                item_point = sphere_expmap0(
                    item_part, self.tangent_clip, self.distance_eps
                )
                distance = spherical_distance_sq(
                    user_point, item_point, self.distance_eps
                )
            distances.append(distance / float(dim))
        return torch.stack(distances, dim=-1)

    def score_pairs(self, user: Tensor, item: Tensor) -> Tensor:
        """Score aligned user-item id tensors; larger means more preferred."""

        if user.shape != item.shape:
            raise ValueError("user and item id tensors must have the same shape")
        original_shape = user.shape
        flat_user = user.reshape(-1)
        flat_item = item.reshape(-1)
        user_coordinates = self.user_embedding(flat_user)
        item_coordinates = self.item_embedding(flat_item)
        distances = self.branch_distances(user_coordinates, item_coordinates)
        weights = self._gate_weights(flat_user, flat_item)
        scale = self.log_score_scale.exp().clamp(max=100.0)
        scores = -scale * (weights * distances).sum(dim=-1)
        return scores.reshape(original_shape)

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        return self.score_pairs(user, item)

    def calculate_loss(self, interaction) -> Tensor:
        user = interaction[self.USER_ID]
        positive_item = interaction[self.ITEM_ID]
        negative_item = interaction[self.NEG_ITEM_ID]
        positive_score = self.score_pairs(user, positive_item)
        negative_score = self.score_pairs(user, negative_item)
        loss = self.bpr_loss(positive_score, negative_score)

        if self.reg_weight > 0:
            user_raw = self.user_embedding(user)
            positive_raw = self.item_embedding(positive_item)
            negative_raw = self.item_embedding(negative_item)
            regularizer = (
                user_raw.square().sum(dim=-1)
                + positive_raw.square().sum(dim=-1)
                + negative_raw.square().sum(dim=-1)
            ).mean()
            if self.gate_mode == "entity":
                assert self.user_gate_embedding is not None
                assert self.item_gate_embedding is not None
                regularizer = regularizer + (
                    self.user_gate_embedding(user).square().mean()
                    + self.item_gate_embedding(positive_item).square().mean()
                    + self.item_gate_embedding(negative_item).square().mean()
                )
            loss = loss + self.reg_weight * regularizer
        return loss

    def predict(self, interaction) -> Tensor:
        return self.score_pairs(
            interaction[self.USER_ID], interaction[self.ITEM_ID]
        )

    def full_sort_predict(self, interaction) -> Tensor:
        """Score every item in bounded chunks and return RecBole's flat layout."""

        users = interaction[self.USER_ID].reshape(-1)
        score_chunks = []
        for start in range(0, self.n_items, self.eval_item_chunk_size):
            stop = min(start + self.eval_item_chunk_size, self.n_items)
            items = torch.arange(start, stop, device=users.device)
            pair_users = users[:, None].expand(-1, stop - start)
            pair_items = items[None, :].expand(users.shape[0], -1)
            score_chunks.append(self.score_pairs(pair_users, pair_items))
        return torch.cat(score_chunks, dim=1).reshape(-1)


__all__ = [
    "MixedGeoRec",
    "euclidean_distance_sq",
    "poincare_distance",
    "poincare_distance_sq",
    "poincare_expmap0",
    "sphere_expmap0",
    "spherical_distance",
    "spherical_distance_sq",
]
