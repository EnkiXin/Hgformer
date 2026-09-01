"""SL(8) extension of Hgformer's standalone LHGCN graph framework.

This model is intentionally an independent ablation and does not modify the
historical ``HGCF`` or ``RecFormer`` implementations.  It keeps the parts of
the repository's LHGCN path that define the recommendation experiment:

* symmetric ``D^-1/2 A D^-1/2`` propagation on the user-item graph;
* no self loops by default (matching the released code, not the paper text);
* the final GCN layer only, rather than LightGCN's mean over layers;
* RecBole pairwise sampling and the same full-ranking evaluator.

Three SL graph operators isolate the new geometric hypothesis:

``ambient_retract`` (default)
    Map raw matrices to SL(8), aggregate the group matrices in the ambient
    matrix space, then determinant-normalise after every layer.  This is the
    closest scalable analogue of LHGCN's ambient-sum/closed-form-hyperboloid
    centroid.  It is an extrinsic retraction, not an exact Frechet mean for the
    Schatten semidistance.

``tangent_last``
    Propagate trace-free Lie-algebra coordinates and map the final layer to
    SL(8).  This stable control is analogous to the tangent-space aggregation
    that LHGCN was designed to contrast with.

``karcher1``
    One-step Cartan--Schouten exponential-barycenter aggregation on
    row-normalised weights: the tangent mean seeds one bi-invariant
    fixed-point step (Pennec & Arsigny 2012).  This is the intrinsic mean
    whose stationarity condition matches the model's Schatten log
    semidistance; the output is in SL(8) by construction, so no orientation
    repair or determinant retraction exists on this path.  The correction
    costs one 8x8 log per edge and layer (``sl_karcher_*`` keys control
    truncation, chunking, and checkpointing).

The released LHGCN code applies a shared ``LorentzBatchNorm`` after each graph
layer.  ``sl_layer_norm: liebn`` provides the operator-matched SL analogue
following the LieBN recipe (Chen et al., ICLR 2024) with the Cartan--Schouten
exponential barycenter in place of the Frechet mean (SL(n) admits no
bi-invariant Riemannian metric); see ``slrec_experiments/sl_liebn.py`` for the
exact correspondence and the documented deviations.  ``sl_layer_norm: none``
remains the historical control and the default.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from recbole_gnn.model.general_recommender.slrecgraph import SLRecGraph
from slrec_experiments.geometry import (
    matrix_log_gregory,
    relative_matrix,
    trace_free,
)
from slrec_experiments.sl_lhgcn import (
    ambient_sl_centroid_step,
    karcher_sl_centroid_step,
    row_normalise_sparse,
)
from slrec_experiments.sl_liebn import SLLieBatchNorm


def _config_get(config: Any, key: str, default: Any) -> Any:
    try:
        value = config[key]
    except (KeyError, TypeError, AttributeError):
        return default
    return default if value is None else value


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "yes", "y", "1", "on"}:
            return True
        if value in {"false", "no", "n", "0", "off"}:
            return False
    return bool(value)


class SL8LHGCN(SLRecGraph):
    """Final-layer LHGCN-style collaborative filtering in ``SL(8)``."""

    MODEL_NAME = "SL8LHGCN"
    REQUIRED_MATRIX_DIM = 8
    REQUIRED_NUM_FACTORS = 1
    _SUPPORTED_MODES = {"ambient_retract", "tangent_last", "karcher1"}
    _SUPPORTED_LOSSES = {"lhgcn_hinge_squared_sum", "bpr_mean"}
    _SUPPORTED_LAYER_NORMS = {"none", "liebn"}

    def __init__(self, config: Any, dataset: Any) -> None:
        super().__init__(config, dataset)

        if (
            self.matrix_dim != self.REQUIRED_MATRIX_DIM
            or self.num_factors != self.REQUIRED_NUM_FACTORS
        ):
            raise ValueError(
                f"{self.MODEL_NAME} is the controlled "
                f"SL({self.REQUIRED_MATRIX_DIM}) single-factor model; set "
                f"matrix_dim: {self.REQUIRED_MATRIX_DIM} and "
                f"num_factors: {self.REQUIRED_NUM_FACTORS}"
            )

        self.embedding_init = str(
            _config_get(config, "embedding_init", "normal")
        ).strip().lower()
        if self.embedding_init == "xavier_uniform_combined":
            # Historical LHGCN/HGCF owns one [num_users + num_items, 64]
            # embedding table and applies Xavier uniform when learner=Adam.
            # The SL adapter keeps separate RecBole tables, but using the bound
            # of that combined shape gives the same initial distribution.
            bound = math.sqrt(
                6.0
                / float(self.n_users + self.n_items + self.coordinate_dim)
            )
            with torch.no_grad():
                self.user_embedding.weight.uniform_(-bound, bound)
                self.item_embedding.weight.uniform_(-bound, bound)
        elif self.embedding_init != "normal":
            raise ValueError(
                "embedding_init must be one of "
                "{'normal', 'xavier_uniform_combined'}"
            )

        self.sl_gcn_mode = str(
            _config_get(config, "sl_gcn_mode", "ambient_retract")
        ).strip().lower()
        if self.sl_gcn_mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"sl_gcn_mode must be one of {sorted(self._SUPPORTED_MODES)}; "
                f"got {self.sl_gcn_mode!r}"
            )

        # ``gcn_layers`` is the name used by RecFormer/HGCF for LHGCN.  Fall
        # back to the existing SLRec ``n_layers`` key for compatibility.
        self.n_layers = int(
            _config_get(config, "gcn_layers", self.n_layers)
        )
        if self.n_layers < 0:
            raise ValueError("gcn_layers must be non-negative")
        layer_aggregation = str(
            _config_get(config, "lhgcn_layer_aggregation", "last")
        ).strip().lower()
        if layer_aggregation != "last":
            raise ValueError(
                f"{self.MODEL_NAME} only supports "
                "lhgcn_layer_aggregation: last, which "
                "matches HGCN.LGCN in the released Hgformer code"
            )
        self.lhgcn_layer_aggregation = layer_aggregation

        layer_norm = str(
            _config_get(config, "sl_layer_norm", "none")
        ).strip().lower()
        if layer_norm not in self._SUPPORTED_LAYER_NORMS:
            raise ValueError(
                "sl_layer_norm must be one of "
                f"{sorted(self._SUPPORTED_LAYER_NORMS)}; got {layer_norm!r}"
            )
        self.sl_layer_norm = layer_norm
        if layer_norm == "liebn":
            self.sl_liebn = SLLieBatchNorm(
                self.matrix_dim,
                self.num_factors,
                mean_mode=str(
                    _config_get(config, "liebn_mean", "karcher1")
                ).strip().lower(),
                dispersion=str(
                    _config_get(config, "liebn_dispersion", "mean_norm")
                ).strip().lower(),
                eps=float(_config_get(config, "liebn_eps", 1e-5)),
                log_terms=int(_config_get(config, "liebn_log_terms", 8)),
                jitter=self.log_jitter,
                learnable_bias=_as_bool(
                    _config_get(config, "liebn_learnable_bias", False)
                ),
                max_log_norm=float(
                    _config_get(config, "liebn_max_log_norm", 25.0)
                ),
                max_tangent_norm=float(
                    _config_get(config, "liebn_max_tangent_norm", 3.0)
                ),
            )
        else:
            self.sl_liebn = None

        self.sl_karcher_log_terms = int(
            _config_get(config, "sl_karcher_log_terms", 6)
        )
        self.sl_karcher_correction = _as_bool(
            _config_get(config, "sl_karcher_correction", True)
        )
        self.sl_karcher_edge_chunk = int(
            _config_get(config, "sl_karcher_edge_chunk", 262144)
        )
        self.sl_karcher_checkpoint = _as_bool(
            _config_get(config, "sl_karcher_checkpoint", True)
        )
        self.sl_karcher_max_log_norm = float(
            _config_get(config, "sl_karcher_max_log_norm", 25.0)
        )

        self.lhgcn_include_self = _as_bool(
            _config_get(config, "lhgcn_include_self", False)
        )
        self.lhgcn_self_loop_weight = float(
            _config_get(config, "lhgcn_self_loop_weight", 1.0)
        )
        if self.lhgcn_self_loop_weight <= 0:
            raise ValueError("lhgcn_self_loop_weight must be positive")
        self.sl_centroid_fallback_clip = float(
            _config_get(config, "sl_centroid_fallback_clip", 1.0)
        )
        if self.sl_centroid_fallback_clip <= 0:
            raise ValueError("sl_centroid_fallback_clip must be positive")
        self.sl_membership_check = _as_bool(
            _config_get(config, "sl_membership_check", True)
        )
        self.sl_membership_strict = _as_bool(
            _config_get(config, "sl_membership_strict", False)
        )
        self.sl_membership_tolerance = float(
            _config_get(config, "sl_membership_tolerance", 1e-4)
        )
        if self.sl_membership_tolerance <= 0:
            raise ValueError("sl_membership_tolerance must be positive")
        self.sl_distance_membership_check = _as_bool(
            _config_get(config, "sl_distance_membership_check", True)
        )
        self.sl_distance_check_samples = int(
            _config_get(config, "sl_distance_check_samples", 16)
        )
        self.sl_log_trace_tolerance = float(
            _config_get(config, "sl_log_trace_tolerance", 1e-3)
        )
        if self.sl_distance_check_samples < 1:
            raise ValueError("sl_distance_check_samples must be positive")
        if self.sl_log_trace_tolerance <= 0:
            raise ValueError("sl_log_trace_tolerance must be positive")

        self.pairwise_loss = str(
            _config_get(config, "pairwise_loss", "lhgcn_hinge_squared_sum")
        ).strip().lower()
        if self.pairwise_loss not in self._SUPPORTED_LOSSES:
            raise ValueError(
                f"pairwise_loss must be one of {sorted(self._SUPPORTED_LOSSES)}; "
                f"got {self.pairwise_loss!r}"
            )
        configured_margin = _config_get(
            config, "loss_margin", _config_get(config, "margin", 0.2)
        )
        self.loss_margin = float(configured_margin)
        if self.loss_margin < 0:
            raise ValueError("loss_margin/margin must be non-negative")
        if self.pairwise_loss == "lhgcn_hinge_squared_sum":
            if self.log_score_scale.requires_grad:
                raise ValueError(
                    "the faithful LHGCN hinge loss has no learned score scale; "
                    "set learnable_score_scale: false"
                )
            if abs(float(self.log_score_scale.detach().exp()) - 1.0) > 1e-7:
                raise ValueError(
                    "the faithful LHGCN hinge loss requires score_scale: 1.0"
                )

        # Rebuild only when the paper-style self-loop option is requested, or
        # when ``gcn_layers`` was positive but SLRec's ``n_layers`` was zero at
        # base-class construction time.
        if self.n_layers > 0:
            if self.lhgcn_include_self:
                adjacency = self._build_lhgcn_adjacency_with_self(dataset)
                self.norm_adj_matrix = adjacency.to(self.device)
            elif self.norm_adj_matrix is None:
                self.norm_adj_matrix = self._build_normalised_adjacency(dataset).to(
                    self.device
                )
        else:
            self.norm_adj_matrix = None

        if self.sl_gcn_mode == "karcher1" and self.norm_adj_matrix is not None:
            # The exponential barycenter needs convex per-row weights; the
            # symmetric normalisation's sub-unit row sums would otherwise act
            # as a per-layer contraction toward the identity.
            self.karcher_adj_matrix = row_normalise_sparse(
                self.norm_adj_matrix
            )
        else:
            self.karcher_adj_matrix = None

        if self.norm_adj_matrix is None:
            active_nodes = torch.ones(
                self.n_users + self.n_items, dtype=torch.bool, device=self.device
            )
        else:
            adjacency = self.norm_adj_matrix.coalesce()
            active_nodes = torch.zeros(
                adjacency.shape[0], dtype=torch.bool, device=adjacency.device
            )
            active_nodes[adjacency.indices()[0]] = True
        self.register_buffer(
            "sl_active_node_mask", active_nodes, persistent=False
        )

        self._warned_projection_event = False
        self._last_projection_diagnostics: Dict[str, Any] = {}
        self._last_distance_diagnostics: Dict[str, Any] = {}
        self._distance_diagnostics_pending = True

    def _clear_full_sort_cache(self) -> None:
        super()._clear_full_sort_cache()
        # A single bounded diagnostic sample is enough for one newly
        # materialised representation table.  In particular, do not rerun a
        # solve + matrix-log + matrix-exp audit for every item chunk during
        # full-sort validation; that audit never contributes to scores.
        self._distance_diagnostics_pending = True

    def _build_lhgcn_adjacency_with_self(self, dataset: Any) -> torch.Tensor:
        """Build symmetric-normalised bipartite adjacency plus self loops."""

        interaction = dataset.inter_matrix(form="coo")
        users = torch.as_tensor(interaction.row, dtype=torch.long)
        items = torch.as_tensor(interaction.col, dtype=torch.long) + self.n_users
        node_count = self.n_users + self.n_items
        nodes = torch.arange(node_count, dtype=torch.long)

        source = torch.cat((users, items, nodes), dim=0)
        target = torch.cat((items, users, nodes), dim=0)
        edge_values = torch.cat(
            (
                torch.ones(users.numel() + items.numel(), dtype=torch.float32),
                torch.full(
                    (node_count,),
                    self.lhgcn_self_loop_weight,
                    dtype=torch.float32,
                ),
            )
        )
        degree = torch.zeros(node_count, dtype=torch.float32)
        degree.index_add_(0, source, edge_values)
        inv_sqrt_degree = degree.clamp_min(1e-12).pow(-0.5)
        values = (
            edge_values * inv_sqrt_degree[source] * inv_sqrt_degree[target]
        )
        return torch.sparse_coo_tensor(
            torch.stack((source, target), dim=0),
            values,
            size=(node_count, node_count),
            dtype=torch.float32,
        ).coalesce()

    def _raw_coordinate_table(self) -> torch.Tensor:
        return torch.cat(
            (self.user_embedding.weight, self.item_embedding.weight), dim=0
        ).reshape(
            -1, self.num_factors, self.matrix_dim, self.matrix_dim
        )

    @staticmethod
    def _sparse_coordinate_step(
        adjacency: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        flat = coordinates.reshape(coordinates.shape[0], -1)
        propagated = torch.sparse.mm(adjacency, flat).reshape_as(coordinates)
        return trace_free(propagated)

    def _propagate_tangent_coordinates(self) -> Tuple[torch.Tensor, float]:
        """Propagate the full coordinate table without materialising groups."""

        coordinates = trace_free(self._raw_coordinate_table())
        max_trace_error = coordinates.diagonal(
            dim1=-2, dim2=-1
        ).sum(dim=-1).abs().max()
        for _ in range(self.n_layers):
            if self.norm_adj_matrix is None:
                raise RuntimeError("normalised adjacency was not initialised")
            coordinates = self._sparse_coordinate_step(
                self.norm_adj_matrix, coordinates
            )
            if self.sl_liebn is not None:
                # First-order (identity-anchored) form of the group LieBN;
                # keeps the tangent mode's no-materialisation efficiency.
                coordinates, _ = self.sl_liebn.normalise_tangent(
                    coordinates, mask=self.sl_active_node_mask
                )
            max_trace_error = torch.maximum(
                max_trace_error,
                coordinates.diagonal(dim1=-2, dim2=-1)
                .sum(dim=-1)
                .abs()
                .max(),
            )
        return coordinates, float(max_trace_error.detach().item())

    def _record_output_group_diagnostics(
        self, groups: torch.Tensor, diagnostics: Dict[str, Any]
    ) -> None:
        sign, log_abs_det = torch.linalg.slogdet(groups.detach().float())
        diagnostics["nonpositive_output_determinants"] = int(
            sign.le(0).sum().item()
        )
        finite = torch.isfinite(log_abs_det)
        diagnostics["nonfinite_output_log_determinants"] = int(
            (~finite).sum().item()
        )
        diagnostics["max_abs_output_log_determinant"] = (
            float(log_abs_det[finite].abs().max().item())
            if bool(finite.any())
            else float("inf")
        )
        violation = (
            sign.le(0)
            | ~finite
            | log_abs_det.abs().gt(self.sl_membership_tolerance)
        )
        diagnostics["output_membership_violations"] = int(
            violation.sum().item()
        )
        diagnostics["membership_tolerance"] = self.sl_membership_tolerance
        if self.sl_membership_strict and bool(violation.any()):
            raise RuntimeError(
                f"{self.MODEL_NAME} output left SL({self.matrix_dim}): "
                f"{int(violation.sum().item())}/{violation.numel()} matrices "
                "violate the determinant membership check"
            )

    def _group_membership_summary(self, groups: torch.Tensor) -> Dict[str, Any]:
        sign, log_abs_det = torch.linalg.slogdet(groups.detach().float())
        finite = torch.isfinite(groups.detach()).all(dim=(-2, -1)) & torch.isfinite(
            log_abs_det
        )
        violation = (
            sign.le(0)
            | ~finite
            | log_abs_det.abs().gt(self.sl_membership_tolerance)
        )
        summary = {
            "total": sign.numel(),
            "nonpositive_determinants": int(sign.le(0).sum().item()),
            "nonfinite_log_determinants": int((~finite).sum().item()),
            "membership_violations": int(violation.sum().item()),
            "max_abs_log_determinant": (
                float(log_abs_det[finite & sign.gt(0)].abs().max().item())
                if bool((finite & sign.gt(0)).any())
                else float("inf")
            ),
        }
        if self.sl_membership_strict and summary["membership_violations"]:
            raise RuntimeError(
                f"{self.MODEL_NAME} group construction left "
                f"SL({self.matrix_dim}): "
                f"{summary['membership_violations']}/{summary['total']} matrices"
            )
        return summary

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return final-layer user/item matrices lying in configured ``SL(n)``."""

        raw_coordinates = self._raw_coordinate_table()
        diagnostics: Dict[str, Any] = {
            "mode": self.sl_gcn_mode,
            "layers": self.n_layers,
            "projection_total": 0,
            "orientation_repairs": 0,
            "singular_fallbacks": 0,
            "active_singular_fallbacks": 0,
            "inactive_singular_fallbacks": 0,
            "layer_membership": [],
        }
        final_layer_diagnostics = None

        if self.sl_gcn_mode == "ambient_retract":
            all_groups = self._to_group(trace_free(raw_coordinates))
            if self.sl_membership_check:
                diagnostics["initial_group_membership"] = (
                    self._group_membership_summary(all_groups)
                )
            for layer_index in range(self.n_layers):
                if self.norm_adj_matrix is None:
                    raise RuntimeError("normalised adjacency was not initialised")
                all_groups, layer_diagnostics = ambient_sl_centroid_step(
                    all_groups,
                    self.norm_adj_matrix,
                    fallback_clip=self.sl_centroid_fallback_clip,
                    collect_diagnostics=True,
                    active_mask=self.sl_active_node_mask,
                    membership_tolerance=self.sl_membership_tolerance,
                    strict_membership=self.sl_membership_strict,
                )
                final_layer_diagnostics = layer_diagnostics
                layer_record = {"layer": layer_index + 1, **asdict(layer_diagnostics)}
                diagnostics["layer_membership"].append(layer_record)
                diagnostics["projection_total"] += layer_diagnostics.total
                diagnostics[
                    "orientation_repairs"
                ] += layer_diagnostics.orientation_repairs
                diagnostics[
                    "singular_fallbacks"
                ] += layer_diagnostics.singular_fallbacks
                diagnostics[
                    "active_singular_fallbacks"
                ] += layer_diagnostics.active_singular_fallbacks
                diagnostics[
                    "inactive_singular_fallbacks"
                ] += layer_diagnostics.inactive_singular_fallbacks
                if self.sl_liebn is not None:
                    all_groups, norm_diagnostics = self.sl_liebn(
                        all_groups, mask=self.sl_active_node_mask
                    )
                    layer_record["layer_norm"] = norm_diagnostics
        elif self.sl_gcn_mode == "karcher1":
            all_groups = self._to_group(trace_free(raw_coordinates))
            if self.sl_membership_check:
                diagnostics["initial_group_membership"] = (
                    self._group_membership_summary(all_groups)
                )
            for layer_index in range(self.n_layers):
                if self.karcher_adj_matrix is None:
                    raise RuntimeError(
                        "row-normalised adjacency was not initialised"
                    )
                all_groups, karcher_diagnostics = karcher_sl_centroid_step(
                    all_groups,
                    self.karcher_adj_matrix,
                    log_terms=self.sl_karcher_log_terms,
                    jitter=self.log_jitter,
                    correction=self.sl_karcher_correction,
                    edge_chunk=self.sl_karcher_edge_chunk,
                    use_checkpoint=self.sl_karcher_checkpoint,
                    max_log_norm=self.sl_karcher_max_log_norm,
                )
                layer_record = {
                    "layer": layer_index + 1,
                    **asdict(karcher_diagnostics),
                }
                if self.sl_liebn is not None:
                    all_groups, norm_diagnostics = self.sl_liebn(
                        all_groups, mask=self.sl_active_node_mask
                    )
                    layer_record["layer_norm"] = norm_diagnostics
                diagnostics["layer_membership"].append(layer_record)
        else:
            # Every coordinate tensor is trace-free, so exp(coordinates) is a
            # valid SL(8) representation at every conceptual graph layer.  We
            # materialise the exponential only for the final layer, which is
            # the intended efficient tangent-space control.
            coordinates, max_trace_error = self._propagate_tangent_coordinates()
            diagnostics["max_abs_layer_trace"] = max_trace_error
            all_groups = self._to_group(coordinates)

        projection_total = max(1, int(diagnostics["projection_total"]))
        diagnostics["orientation_repair_rate"] = (
            diagnostics["orientation_repairs"] / projection_total
        )
        diagnostics["singular_fallback_rate"] = (
            diagnostics["singular_fallbacks"] / projection_total
        )
        diagnostics["materialized_group_entities"] = all_groups.shape[0]
        diagnostics["materialized_full_entity_table"] = True
        if final_layer_diagnostics is None or self.sl_liebn is not None:
            # Tangent mode, the karcher1 mode, the zero-layer ambient control,
            # and any liebn-normalised output (the retraction audit precedes
            # the normalisation) have not already audited the materialised
            # output.
            self._record_output_group_diagnostics(all_groups, diagnostics)
        else:
            # The determinant retraction just performed a strict check of the
            # exact matrices in ``all_groups``.  Reuse those values instead of
            # running a redundant full-table slogdet after every forward.
            diagnostics["nonpositive_output_determinants"] = (
                final_layer_diagnostics.output_nonpositive_determinants
            )
            diagnostics["nonfinite_output_log_determinants"] = (
                final_layer_diagnostics.output_nonfinite_log_determinants
            )
            diagnostics["max_abs_output_log_determinant"] = (
                final_layer_diagnostics.max_abs_output_log_determinant
            )
            diagnostics["output_membership_violations"] = (
                final_layer_diagnostics.output_membership_violations
            )
            diagnostics["membership_tolerance"] = self.sl_membership_tolerance
        self._last_projection_diagnostics = diagnostics

        if (
            not self._warned_projection_event
            and (
                diagnostics["orientation_repairs"]
                or diagnostics["singular_fallbacks"]
            )
        ):
            self.logger.warning(
                "%s ambient projection diagnostics: %d orientation "
                "repairs and %d singular fallbacks among %d layer/entity "
                "projections (rates %.3e / %.3e).",
                self.MODEL_NAME,
                diagnostics["orientation_repairs"],
                diagnostics["singular_fallbacks"],
                diagnostics["projection_total"],
                diagnostics["orientation_repair_rate"],
                diagnostics["singular_fallback_rate"],
            )
            self._warned_projection_event = True

        return torch.split(all_groups, (self.n_users, self.n_items), dim=0)

    def projection_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics from the most recent forward pass."""

        diagnostics = dict(self._last_projection_diagnostics)
        if self._last_distance_diagnostics:
            diagnostics["distance_membership"] = dict(
                self._last_distance_diagnostics
            )
        return diagnostics

    def _record_effective_coordinate_diagnostics(
        self,
        coordinates: torch.Tensor,
        *,
        source: str,
        max_layer_trace: Optional[float] = None,
    ) -> None:
        """Record the bounded-cost invariants of the Euclidean chart control."""

        detached = coordinates.detach().float()
        trace = detached.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        norm = torch.linalg.matrix_norm(detached, ord="fro", dim=(-2, -1))
        normalised_trace = trace.abs() / (1.0 + norm)
        summary = {
            "source": source,
            "entities": int(coordinates.shape[0]),
            "nonfinite_coordinates": int(
                (~torch.isfinite(detached).all(dim=(-3, -2, -1))).sum().item()
            ),
            "max_normalized_trace": float(normalised_trace.max().item()),
            "trace_tolerance": self.sl_log_trace_tolerance,
        }

        if source == "final_group_log":
            # ``forward`` has just populated the full group-propagation audit;
            # retain it and attach the chart-specific facts.
            diagnostics = dict(self._last_projection_diagnostics)
        else:
            diagnostics = {
                "mode": self.sl_gcn_mode,
                "layers": self.n_layers,
                "projection_total": 0,
                "orientation_repairs": 0,
                "singular_fallbacks": 0,
                "orientation_repair_rate": 0.0,
                "singular_fallback_rate": 0.0,
                "materialized_group_entities": 0,
                "materialized_full_entity_table": False,
            }
            if max_layer_trace is not None:
                diagnostics["max_abs_layer_trace"] = max_layer_trace
        diagnostics["sl_score_mode"] = self.sl_score_mode
        diagnostics["effective_coordinates"] = summary
        self._last_projection_diagnostics = diagnostics

        if self.sl_membership_strict and (
            summary["nonfinite_coordinates"]
            or summary["max_normalized_trace"] > self.sl_log_trace_tolerance
        ):
            raise RuntimeError(
                f"{self.MODEL_NAME} effective chart coordinates violated "
                f"the sl({self.matrix_dim}) invariant: {summary}"
            )

    def _effective_coordinate_tables(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return actual post-propagation coordinates for the chart control.

        ``tangent_last`` (and every zero-layer model) already owns trace-free
        effective coordinates, so this path deliberately performs no
        exponential followed by a logarithm.  Group-propagating modes first
        execute their real ambient/Karcher forward, then logarithmise each
        final entity exactly once with the configured Gregory approximation.
        """

        if self.sl_gcn_mode == "tangent_last":
            all_coordinates, max_trace_error = (
                self._propagate_tangent_coordinates()
            )
            all_coordinates = self._to_effective_tangent_coordinates(
                all_coordinates
            )
            source = "propagated_tangent"
        elif self.n_layers == 0:
            all_coordinates = self._to_effective_tangent_coordinates(
                self._raw_coordinate_table()
            )
            max_trace_error = float(
                all_coordinates.diagonal(dim1=-2, dim2=-1)
                .sum(dim=-1)
                .abs()
                .max()
                .detach()
                .item()
            )
            source = "zero_layer_tangent"
        else:
            user_groups, item_groups = self.forward()
            all_groups = torch.cat((user_groups, item_groups), dim=0)
            all_coordinates = trace_free(
                matrix_log_gregory(
                    all_groups,
                    terms=self.log_terms,
                    jitter=self.log_jitter,
                )
            )
            max_trace_error = None
            source = "final_group_log"

        self._record_effective_coordinate_diagnostics(
            all_coordinates,
            source=source,
            max_layer_trace=max_trace_error,
        )
        return torch.split(
            all_coordinates, (self.n_users, self.n_items), dim=0
        )

    def _record_distance_membership_diagnostics(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> None:
        """Audit a bounded sample of the relative matrix and approximate log."""

        if not self.sl_distance_membership_check:
            return
        if not self._distance_diagnostics_pending:
            return
        self._distance_diagnostics_pending = False
        matrix_dim = self.matrix_dim
        left_flat = left.detach().reshape(-1, matrix_dim, matrix_dim)
        right_flat = right.detach().reshape(-1, matrix_dim, matrix_dim)
        sample_count = min(
            self.sl_distance_check_samples,
            max(left_flat.shape[0], right_flat.shape[0]),
        )
        sample_indices = torch.arange(sample_count, device=left.device)
        sampled_left = left_flat[sample_indices % left_flat.shape[0]].float()
        sampled_right = right_flat[sample_indices % right_flat.shape[0]].float()
        relative = relative_matrix(sampled_left, sampled_right)
        sign, log_abs_det = torch.linalg.slogdet(relative)
        finite_relative = torch.isfinite(relative).all(dim=(-2, -1)) & torch.isfinite(
            log_abs_det
        )
        relative_violation = (
            sign.le(0)
            | ~finite_relative
            | log_abs_det.abs().gt(self.sl_membership_tolerance)
        )

        solve_numerator = torch.linalg.matrix_norm(
            sampled_left @ relative - sampled_right,
            ord="fro",
            dim=(-2, -1),
        )
        solve_denominator = (
            torch.linalg.matrix_norm(sampled_left, ord="fro", dim=(-2, -1))
            * torch.linalg.matrix_norm(relative, ord="fro", dim=(-2, -1))
            + torch.linalg.matrix_norm(sampled_right, ord="fro", dim=(-2, -1))
        ).clamp_min(1e-12)
        solve_residual = solve_numerator / solve_denominator

        approximate_log = matrix_log_gregory(
            relative, terms=self.log_terms, jitter=self.log_jitter
        )
        log_norm = torch.linalg.matrix_norm(
            approximate_log, ord="fro", dim=(-2, -1)
        )
        log_trace = approximate_log.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        log_trace_residual = log_trace.abs() / (1.0 + log_norm)
        reconstruction_residual = torch.linalg.matrix_norm(
            torch.matrix_exp(approximate_log) - relative,
            ord="fro",
            dim=(-2, -1),
        ) / torch.linalg.matrix_norm(
            relative, ord="fro", dim=(-2, -1)
        ).clamp_min(1e-12)

        finite_log = torch.isfinite(approximate_log).all(dim=(-2, -1))
        diagnostics = {
            "samples": sample_count,
            "relative_membership_violations": int(
                relative_violation.sum().item()
            ),
            "relative_nonpositive_determinants": int(sign.le(0).sum().item()),
            "max_abs_relative_log_determinant": (
                float(log_abs_det[finite_relative].abs().max().item())
                if bool(finite_relative.any())
                else float("inf")
            ),
            "max_relative_solve_residual": float(solve_residual.max().item()),
            "nonfinite_approximate_logs": int((~finite_log).sum().item()),
            "max_normalized_approximate_log_trace": float(
                log_trace_residual.max().item()
            ),
            "max_approximate_log_reconstruction_residual": float(
                reconstruction_residual.max().item()
            ),
            "membership_tolerance": self.sl_membership_tolerance,
            "log_trace_tolerance": self.sl_log_trace_tolerance,
        }
        self._last_distance_diagnostics = diagnostics
        if self.sl_membership_strict and (
            diagnostics["relative_membership_violations"]
            or diagnostics["nonfinite_approximate_logs"]
            or diagnostics["max_normalized_approximate_log_trace"]
            > self.sl_log_trace_tolerance
        ):
            raise RuntimeError(
                f"{self.MODEL_NAME} distance path violated SL/sl membership: "
                f"{diagnostics}"
            )

    def _group_distance(
        self, user_group: torch.Tensor, item_group: torch.Tensor
    ) -> torch.Tensor:
        user_group, item_group = self._align_pair_shapes(user_group, item_group)
        self._record_distance_membership_diagnostics(user_group, item_group)
        factor_distances = self._factor_distances(user_group, item_group)
        return self._aggregate_factor_distances(factor_distances)

    def _score_groups(
        self, user_group: torch.Tensor, item_group: torch.Tensor
    ) -> torch.Tensor:
        aligned_user, aligned_item = self._align_pair_shapes(
            user_group, item_group
        )
        self._record_distance_membership_diagnostics(aligned_user, aligned_item)
        return super()._score_groups(user_group, item_group)

    def _full_sort_group_tables(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.restore_user_group is None or self.restore_item_group is None:
            self._distance_diagnostics_pending = True
            self.restore_user_group, self.restore_item_group = self.forward()
        return self.restore_user_group, self.restore_item_group

    def _record_tangent_selected_diagnostics(
        self,
        groups: Tuple[torch.Tensor, ...],
        max_trace_error: float,
        *,
        mode: str = "tangent_last",
    ) -> None:
        flattened = [
            group.reshape(-1, self.num_factors, self.matrix_dim, self.matrix_dim)
            for group in groups
        ]
        materialised = torch.cat(flattened, dim=0)
        diagnostics: Dict[str, Any] = {
            "mode": mode,
            "layers": self.n_layers,
            "projection_total": 0,
            "orientation_repairs": 0,
            "singular_fallbacks": 0,
            "orientation_repair_rate": 0.0,
            "singular_fallback_rate": 0.0,
            "max_abs_layer_trace": max_trace_error,
            "materialized_group_entities": materialised.shape[0],
            "materialized_full_entity_table": False,
        }
        self._record_output_group_diagnostics(materialised, diagnostics)
        self._last_projection_diagnostics = diagnostics

    def _decode_unique_selected_coordinates(
        self,
        all_user_coordinates: torch.Tensor,
        all_item_coordinates: torch.Tensor,
        user: torch.Tensor,
        positive_item: torch.Tensor,
        negative_item: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """Exponentiate each selected entity at most once per mini-batch."""

        unique_users, user_inverse = torch.unique(
            user.reshape(-1), sorted=False, return_inverse=True
        )
        flat_positive = positive_item.reshape(-1)
        flat_negative = negative_item.reshape(-1)
        requested_items = torch.cat((flat_positive, flat_negative), dim=0)
        unique_items, item_inverse = torch.unique(
            requested_items, sorted=False, return_inverse=True
        )

        unique_user_groups = self._to_group(
            all_user_coordinates[unique_users]
        )
        unique_item_groups = self._to_group(
            all_item_coordinates[unique_items]
        )
        user_groups = unique_user_groups[user_inverse].reshape(
            user.shape + unique_user_groups.shape[1:]
        )
        positive_inverse = item_inverse[: flat_positive.numel()]
        negative_inverse = item_inverse[flat_positive.numel() :]
        positive_groups = unique_item_groups[positive_inverse].reshape(
            positive_item.shape + unique_item_groups.shape[1:]
        )
        negative_groups = unique_item_groups[negative_inverse].reshape(
            negative_item.shape + unique_item_groups.shape[1:]
        )
        return (
            user_groups,
            positive_groups,
            negative_groups,
            (unique_user_groups, unique_item_groups),
        )

    @staticmethod
    def _stack_positive_and_negative_groups(
        positive_groups: torch.Tensor, negative_groups: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        """Pack positive/negative candidates into one decoder invocation."""

        has_negative_sample_axis = negative_groups.ndim == positive_groups.ndim + 1
        positive_groups = positive_groups.unsqueeze(-4)
        if not has_negative_sample_axis:
            negative_groups = negative_groups.unsqueeze(-4)
        return (
            torch.cat((positive_groups, negative_groups), dim=-4),
            has_negative_sample_axis,
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
            user_coordinates = all_user_coordinates[user]
            positive_coordinates = all_item_coordinates[positive_item]
            negative_coordinates = all_item_coordinates[negative_item]
            candidate_coordinates, has_negative_sample_axis = (
                self._stack_positive_and_negative_groups(
                    positive_coordinates, negative_coordinates
                )
            )
            decoder_users = user_coordinates.unsqueeze(-4)

            if self.pairwise_loss == "lhgcn_hinge_squared_sum":
                candidate_squared_distances = (
                    self._pairwise_squared_coordinate_distance(
                        decoder_users, candidate_coordinates
                    )
                )
                positive_squared_distance = candidate_squared_distances[..., 0]
                negative_squared_distance = candidate_squared_distances[..., 1:]
                if has_negative_sample_axis:
                    positive_squared_distance = positive_squared_distance.unsqueeze(
                        -1
                    )
                else:
                    negative_squared_distance = negative_squared_distance.squeeze(-1)
                return F.relu(
                    positive_squared_distance
                    - negative_squared_distance
                    + self.loss_margin
                ).sum()

            candidate_scores = self._score_effective_coordinates(
                decoder_users, candidate_coordinates
            )
            positive_scores = candidate_scores[..., 0]
            negative_scores = candidate_scores[..., 1:]
            if has_negative_sample_axis:
                positive_scores = positive_scores.unsqueeze(-1)
            else:
                negative_scores = negative_scores.squeeze(-1)
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

        if self.sl_gcn_mode == "tangent_last" or self.n_layers == 0:
            if self.sl_gcn_mode == "tangent_last":
                all_coordinates, max_trace_error = (
                    self._propagate_tangent_coordinates()
                )
                all_user_coordinates, all_item_coordinates = torch.split(
                    all_coordinates, (self.n_users, self.n_items), dim=0
                )
                diagnostic_mode = "tangent_last"
            else:
                # With zero graph layers, ambient mode is just the entity-wise
                # SL decoder.  Views of the two embedding tables avoid the
                # historical full-table concat/exp in every mini-batch.
                all_user_coordinates = trace_free(
                    self.user_embedding.weight.reshape(
                        self.n_users,
                        self.num_factors,
                        self.matrix_dim,
                        self.matrix_dim,
                    )
                )
                all_item_coordinates = trace_free(
                    self.item_embedding.weight.reshape(
                        self.n_items,
                        self.num_factors,
                        self.matrix_dim,
                        self.matrix_dim,
                    )
                )
                max_trace_error = 0.0
                # With zero layers every mode reduces to the entity-wise SL
                # decoder; label the diagnostics with the configured mode.
                diagnostic_mode = self.sl_gcn_mode
            (
                user_groups,
                positive_groups,
                negative_groups,
                materialised_unique_groups,
            ) = self._decode_unique_selected_coordinates(
                all_user_coordinates,
                all_item_coordinates,
                user,
                positive_item,
                negative_item,
            )
            self._record_tangent_selected_diagnostics(
                materialised_unique_groups,
                max_trace_error,
                mode=diagnostic_mode,
            )
        else:
            all_user_groups, all_item_groups = self.forward()
            user_groups = all_user_groups[user]
            positive_groups = all_item_groups[positive_item]
            negative_groups = all_item_groups[negative_item]

        candidate_groups, has_negative_sample_axis = (
            self._stack_positive_and_negative_groups(
                positive_groups, negative_groups
            )
        )
        decoder_users = user_groups.unsqueeze(-4)

        if self.pairwise_loss == "lhgcn_hinge_squared_sum":
            candidate_distances = self._group_distance(
                decoder_users, candidate_groups
            )
            positive_distance = candidate_distances[..., 0]
            negative_distance = candidate_distances[..., 1:]
            if has_negative_sample_axis:
                positive_distance = positive_distance.unsqueeze(-1)
            else:
                negative_distance = negative_distance.squeeze(-1)
            return F.relu(
                positive_distance.square()
                - negative_distance.square()
                + self.loss_margin
            ).sum()

        candidate_scores = self._score_groups(decoder_users, candidate_groups)
        positive_scores = candidate_scores[..., 0]
        negative_scores = candidate_scores[..., 1:]
        if has_negative_sample_axis:
            positive_scores = positive_scores.unsqueeze(-1)
        else:
            negative_scores = negative_scores.squeeze(-1)
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
        if self.sl_gcn_mode == "tangent_last":
            all_coordinates, max_trace_error = self._propagate_tangent_coordinates()
            all_user_coordinates, all_item_coordinates = torch.split(
                all_coordinates, (self.n_users, self.n_items), dim=0
            )
            user_groups = self._to_group(all_user_coordinates[user])
            item_groups = self._to_group(all_item_coordinates[item])
            self._record_tangent_selected_diagnostics(
                (user_groups, item_groups), max_trace_error
            )
            return self._score_groups(user_groups, item_groups)
        all_user_groups, all_item_groups = self.forward()
        return self._score_groups(all_user_groups[user], all_item_groups[item])


__all__ = ["SL8LHGCN"]
