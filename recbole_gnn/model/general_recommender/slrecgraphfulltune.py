"""Loss-only ablation adapter for geometry-only SLRecGraph tuning.

The production :class:`SLRecGraph` uses the standard pairwise logistic/BPR
objective ``softplus(s_neg - s_pos)``.  This adapter leaves the embeddings,
SL(n) map, distance, score and full-sort evaluator unchanged, and adds a
margin-ranking alternative for a controlled loss ablation.  The tuning
driver uses the production model for BPR trials and this class only for hinge
trials, making the baseline trial literally identical to ``SLRecGraph``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from recbole_gnn.model.general_recommender.slrecgraph import SLRecGraph


def pairwise_ranking_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    *,
    variant: str,
    margin: float,
) -> torch.Tensor:
    """Return a mean pairwise ranking loss without changing score semantics."""

    if negative_scores.ndim == positive_scores.ndim + 1:
        positive_scores = positive_scores.unsqueeze(-1)
    score_difference = negative_scores - positive_scores
    if variant == "bpr":
        return F.softplus(score_difference).mean()
    if variant == "hinge":
        return F.relu(float(margin) + score_difference).mean()
    raise ValueError(f"unsupported pairwise_loss: {variant!r}")


class SLRecGraphFullTune(SLRecGraph):
    """SLRecGraph with an opt-in pairwise hinge-loss ablation."""

    def __init__(self, config: Any, dataset: Any) -> None:
        try:
            configured_loss = config["pairwise_loss"]
        except (KeyError, TypeError, AttributeError):
            configured_loss = None
        self.pairwise_loss = str(configured_loss or "bpr").strip().lower()
        if self.pairwise_loss not in {"bpr", "hinge"}:
            raise ValueError(
                "pairwise_loss must be one of {'bpr', 'hinge'}; "
                f"got {self.pairwise_loss!r}"
            )
        try:
            configured_margin = config["loss_margin"]
        except (KeyError, TypeError, AttributeError):
            configured_margin = None
        self.loss_margin = 1.0 if configured_margin is None else float(configured_margin)
        if self.loss_margin < 0:
            raise ValueError("loss_margin must be non-negative")
        super().__init__(config, dataset)

    def calculate_loss(self, interaction: Any) -> torch.Tensor:
        self._clear_full_sort_cache()
        user = interaction[self.USER_ID]
        positive_item = interaction[self.ITEM_ID]
        negative_item = interaction[self.NEG_ITEM_ID]

        all_user_coordinates, all_item_coordinates = self.forward()
        user_coordinates = all_user_coordinates[user]
        positive_coordinates = all_item_coordinates[positive_item]
        negative_coordinates = all_item_coordinates[negative_item]

        positive_scores = self._score_coordinates(
            user_coordinates, positive_coordinates
        )
        negative_scores = self._score_coordinates(
            user_coordinates, negative_coordinates
        )
        ranking_loss = pairwise_ranking_loss(
            positive_scores,
            negative_scores,
            variant=self.pairwise_loss,
            margin=self.loss_margin,
        )

        # Keep the production model's regularisation definition exactly: raw
        # embedding coordinates, averaged over user/positive/negative tables.
        raw_user = self.user_embedding(user)
        raw_positive = self.item_embedding(positive_item)
        raw_negative = self.item_embedding(negative_item)
        regularisation = (
            raw_user.square().sum(dim=-1).mean()
            + raw_positive.square().sum(dim=-1).mean()
            + raw_negative.square().sum(dim=-1).mean()
        ) / 3.0
        return ranking_loss + self.reg_weight * regularisation


__all__ = ["SLRecGraphFullTune", "pairwise_ranking_loss"]
