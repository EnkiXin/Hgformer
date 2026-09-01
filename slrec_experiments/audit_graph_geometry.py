"""Reproducible, bounded-memory diagnostics for RecBole interaction graphs.

This module intentionally does not import RecBole or any training code.  It
reads a RecBole atomic ``.inter`` file, applies an inclusive rating threshold,
computes an interaction-count k-core (matching RecBole's row-count semantics),
and then forms a *simple* topology for shortest-path diagnostics.  Both
interaction multiplicities and unique user-item edges are reported.

The sampled four-point delta is the Gromov four-point statistic of the
unweighted shortest-path metric on the discrete interaction graph.  It is not
sectional curvature, and neither it nor the cycle/branching diagnostics imply
the curvature or even the appropriate family of a learned smooth manifold.

Let R be the number of input rows, V/E the post-filter graph size, L the number
of landmarks, and Q the number of sampled quadruples.  Parsing, k-core peeling,
and connected components are O(R + V + E).  Landmark distances use L sparse
BFS traversals plus predecessor walks between landmarks,
O(L(V_lcc + E_lcc) + L^2 D_lcc); quadruple evaluation is O(Q).  Peak graph
memory is O(R_pass + V + E), while distance memory is O(V_lcc + L^2).  The
default fixed L=32 avoids a V-by-V all-pairs allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from array import array
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph


_MISSING_PREDECESSOR = -9999


def _resolve_column(header: Sequence[str], requested: str) -> int:
    """Resolve either a full RecBole field or its name before ``:type``."""

    exact = [index for index, field in enumerate(header) if field == requested]
    if len(exact) == 1:
        return exact[0]
    base = [
        index
        for index, field in enumerate(header)
        if field.split(":", 1)[0] == requested
    ]
    if len(base) != 1:
        raise ValueError(
            f"Expected one column named {requested!r}; found {len(base)} in {header!r}"
        )
    return base[0]


def _read_atomic_inter(
    path: Path,
    delimiter: str,
    user_field: str,
    item_field: str,
    rating_field: str,
    rating_threshold: float,
) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Return the thresholded user-item matrix with event multiplicities."""

    user_ids: Dict[str, int] = {}
    item_ids: Dict[str, int] = {}
    retained_users = array("I")
    retained_items = array("I")
    digest = hashlib.sha256()
    input_rows = 0
    blank_rows = 0
    retained_rows = 0

    with path.open("rb") as source:
        raw_header = source.readline()
        if not raw_header:
            raise ValueError(f"Empty interaction file: {path}")
        digest.update(raw_header)
        header = raw_header.decode("utf-8-sig").rstrip("\r\n").split(delimiter)
        user_column = _resolve_column(header, user_field)
        item_column = _resolve_column(header, item_field)
        rating_column = _resolve_column(header, rating_field)
        required_column = max(user_column, item_column, rating_column)

        for line_number, raw_line in enumerate(source, start=2):
            digest.update(raw_line)
            text = raw_line.decode("utf-8").rstrip("\r\n")
            if not text:
                blank_rows += 1
                continue
            fields = text.split(delimiter)
            if len(fields) <= required_column:
                raise ValueError(
                    f"Line {line_number} has {len(fields)} columns; "
                    f"at least {required_column + 1} are required"
                )
            input_rows += 1
            user_token = fields[user_column]
            item_token = fields[item_column]
            user_index = user_ids.setdefault(user_token, len(user_ids))
            item_index = item_ids.setdefault(item_token, len(item_ids))
            try:
                rating = float(fields[rating_column])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid rating {fields[rating_column]!r} on line {line_number}"
                ) from exc
            if not math.isfinite(rating):
                raise ValueError(f"Non-finite rating on line {line_number}: {rating!r}")
            if rating >= rating_threshold:
                retained_users.append(user_index)
                retained_items.append(item_index)
                retained_rows += 1

    shape = (len(user_ids), len(item_ids))
    if retained_rows:
        rows = np.frombuffer(retained_users, dtype=np.uint32)
        columns = np.frombuffer(retained_items, dtype=np.uint32)
        values = np.ones(retained_rows, dtype=np.int32)
        interactions = sparse.coo_matrix(
            (values, (rows, columns)), shape=shape, dtype=np.int32
        ).tocsr()
        interactions.eliminate_zeros()
        interactions.sort_indices()
    else:
        interactions = sparse.csr_matrix(shape, dtype=np.int32)

    user_degree = np.asarray(interactions.sum(axis=1)).ravel()
    item_degree = np.asarray(interactions.sum(axis=0)).ravel()
    stats: Dict[str, Any] = {
        "sha256": digest.hexdigest(),
        "input_rows": input_rows,
        "blank_rows_ignored": blank_rows,
        "raw_distinct_users": len(user_ids),
        "raw_distinct_items": len(item_ids),
        "rating_pass_rows": retained_rows,
        "rating_pass_unique_edges": int(interactions.nnz),
        "duplicate_rating_pass_edges_collapsed": retained_rows - int(interactions.nnz),
        "rating_pass_users_with_edges": int(np.count_nonzero(user_degree)),
        "rating_pass_items_with_edges": int(np.count_nonzero(item_degree)),
    }
    return interactions, stats


def _iterative_bipartite_k_core(
    interactions: sparse.csr_matrix, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Peel a row-count k-core and return unique endpoints plus multiplicities."""

    if k < 0:
        raise ValueError("k_core must be non-negative")
    csr = interactions.tocsr(copy=False)
    csc = interactions.tocsc()
    user_degree = np.asarray(csr.sum(axis=1)).ravel().astype(np.int64, copy=False)
    item_degree = np.asarray(csc.sum(axis=0)).ravel().astype(np.int64, copy=False)

    # Vertices absent from the thresholded graph are not part of its 0-core.
    alive_user = user_degree > 0
    alive_item = item_degree > 0
    queue = deque()  # type: deque[int]
    if k > 0:
        queue.extend(
            int(index)
            for index in np.flatnonzero(alive_user & (user_degree < k))
        )
        user_count = csr.shape[0]
        queue.extend(
            user_count + int(index)
            for index in np.flatnonzero(alive_item & (item_degree < k))
        )
    else:
        user_count = csr.shape[0]

    while queue:
        encoded = queue.popleft()
        if encoded < user_count:
            user = encoded
            if not alive_user[user]:
                continue
            alive_user[user] = False
            start, end = csr.indptr[user], csr.indptr[user + 1]
            for position in range(start, end):
                item = int(csr.indices[position])
                if alive_item[item]:
                    previous_degree = item_degree[item]
                    item_degree[item] -= int(csr.data[position])
                    if k > 0 and previous_degree >= k and item_degree[item] < k:
                        queue.append(user_count + item)
        else:
            item = encoded - user_count
            if not alive_item[item]:
                continue
            alive_item[item] = False
            start, end = csc.indptr[item], csc.indptr[item + 1]
            for position in range(start, end):
                user = int(csc.indices[position])
                if alive_user[user]:
                    previous_degree = user_degree[user]
                    user_degree[user] -= int(csc.data[position])
                    if k > 0 and previous_degree >= k and user_degree[user] < k:
                        queue.append(user)

    edges = csr.tocoo(copy=False)
    edge_mask = alive_user[edges.row] & alive_item[edges.col]
    old_to_new_user = np.full(csr.shape[0], -1, dtype=np.int64)
    old_to_new_item = np.full(csr.shape[1], -1, dtype=np.int64)
    old_to_new_user[alive_user] = np.arange(np.count_nonzero(alive_user))
    old_to_new_item[alive_item] = np.arange(np.count_nonzero(alive_item))
    edge_users = old_to_new_user[edges.row[edge_mask]]
    edge_items = old_to_new_item[edges.col[edge_mask]]
    edge_multiplicities = edges.data[edge_mask].astype(np.int64, copy=False)
    return alive_user, alive_item, edge_users, edge_items, edge_multiplicities


def _combined_adjacency(
    user_count: int, item_count: int, edge_users: np.ndarray, edge_items: np.ndarray
) -> sparse.csr_matrix:
    node_count = user_count + item_count
    if not edge_users.size:
        return sparse.csr_matrix((node_count, node_count), dtype=np.uint8)
    item_nodes = edge_items + user_count
    rows = np.concatenate((edge_users, item_nodes))
    columns = np.concatenate((item_nodes, edge_users))
    adjacency = sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, columns)),
        shape=(node_count, node_count),
    ).tocsr()
    adjacency.sort_indices()
    return adjacency


def _gini(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    ordered = np.sort(values.astype(np.float64, copy=False))
    total = float(ordered.sum())
    if total == 0.0:
        return 0.0
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    result = 2.0 * float(np.dot(ranks, ordered)) / (ordered.size * total)
    result -= (ordered.size + 1.0) / ordered.size
    return float(min(1.0, max(0.0, result)))


def _log2_histogram(values: np.ndarray) -> Dict[str, int]:
    positive = values[values > 0].astype(np.int64, copy=False)
    if not positive.size:
        return {}
    buckets = np.floor(np.log2(positive)).astype(np.int64)
    bucket_ids, counts = np.unique(buckets, return_counts=True)
    result: Dict[str, int] = {}
    for bucket, count in zip(bucket_ids, counts):
        lower = 1 << int(bucket)
        upper = (1 << (int(bucket) + 1)) - 1
        label = str(lower) if lower == upper else f"{lower}-{upper}"
        result[label] = int(count)
    return result


def _degree_summary(values: np.ndarray) -> Dict[str, Any]:
    values = values.astype(np.int64, copy=False)
    if not values.size:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "gini": None,
            "log2_histogram": {},
        }
    percentiles = np.percentile(values, [25, 50, 75, 90, 95, 99])
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p25": float(percentiles[0]),
        "median": float(percentiles[1]),
        "p75": float(percentiles[2]),
        "p90": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "p99": float(percentiles[5]),
        "gini": _gini(values),
        "log2_histogram": _log2_histogram(values),
    }


def _mean_excess_degree(degrees: np.ndarray) -> Optional[float]:
    total = int(degrees.sum())
    if total == 0:
        return None
    degrees_float = degrees.astype(np.float64, copy=False)
    return float(np.dot(degrees_float, degrees_float - 1.0) / total)


def _landmark_distances(
    adjacency: sparse.csr_matrix, landmarks: np.ndarray
) -> np.ndarray:
    """Compute only an LxL distance matrix using one sparse BFS per landmark."""

    count = landmarks.size
    distances = np.zeros((count, count), dtype=np.int32)
    for source_offset in range(count - 1):
        source = int(landmarks[source_offset])
        _, predecessors = csgraph.breadth_first_order(
            adjacency,
            i_start=source,
            directed=False,
            return_predecessors=True,
        )
        for target_offset in range(source_offset + 1, count):
            cursor = int(landmarks[target_offset])
            distance = 0
            while cursor != source:
                cursor = int(predecessors[cursor])
                if cursor == _MISSING_PREDECESSOR:
                    raise RuntimeError("Landmarks must lie in one connected component")
                distance += 1
            distances[source_offset, target_offset] = distance
            distances[target_offset, source_offset] = distance
    return distances


def _sample_four_point_delta(
    adjacency: sparse.csr_matrix,
    seed: int,
    landmark_count: int,
    sample_count: int,
) -> Dict[str, Any]:
    node_count = adjacency.shape[0]
    used_landmarks = min(landmark_count, node_count)
    if used_landmarks < 4 or sample_count <= 0:
        return {
            "status": "insufficient_nodes_or_samples",
            "landmarks": used_landmarks,
            "samples": 0,
            "delta": None,
            "normalized_delta_over_landmark_pair_max": None,
            "landmark_pair_max_distance": None,
        }

    rng = np.random.default_rng(seed)
    landmarks = np.sort(
        rng.choice(node_count, size=used_landmarks, replace=False).astype(np.int64)
    )
    pair_distances = _landmark_distances(adjacency, landmarks)
    diameter_lower_bound = int(pair_distances.max())
    deltas = np.empty(sample_count, dtype=np.float64)
    for sample in range(sample_count):
        a, b, c, d = rng.choice(used_landmarks, size=4, replace=False)
        sums = (
            int(pair_distances[a, b]) + int(pair_distances[c, d]),
            int(pair_distances[a, c]) + int(pair_distances[b, d]),
            int(pair_distances[a, d]) + int(pair_distances[b, c]),
        )
        second_largest, largest = sorted(sums)[-2:]
        deltas[sample] = 0.5 * (largest - second_largest)

    quantiles = np.percentile(deltas, [50, 90, 95, 99])
    normalized = (
        deltas / diameter_lower_bound
        if diameter_lower_bound > 0
        else np.zeros_like(deltas)
    )
    return {
        "status": "ok",
        "method": (
            "Four-point delta=(largest-second-largest)/2 over three pair-sums; "
            "uniform node landmarks and repeated uniform distinct landmark quadruples"
        ),
        "landmarks": used_landmarks,
        "samples": sample_count,
        "delta": {
            "mean": float(deltas.mean()),
            "std": float(deltas.std()),
            "p50": float(quantiles[0]),
            "p90": float(quantiles[1]),
            "p95": float(quantiles[2]),
            "p99": float(quantiles[3]),
            "max": float(deltas.max()),
        },
        "normalized_delta_over_landmark_pair_max": {
            "mean": float(normalized.mean()),
            "p95": float(np.percentile(normalized, 95)),
            "max": float(normalized.max()),
        },
        "landmark_pair_max_distance": diameter_lower_bound,
        "normalization_note": (
            "The denominator is the maximum distance among sampled landmarks, "
            "a reproducible lower bound on graph diameter, not a certified diameter."
        ),
        "sampling_note": (
            "The sampled maximum is a lower bound on exact graph delta; draws share "
            "one landmark set and therefore are not independent confidence samples."
        ),
    }


def audit_interaction_graph(
    path: Path,
    *,
    rating_threshold: float = 3.0,
    k_core: int = 5,
    seed: int = 2024,
    landmarks: int = 32,
    four_point_samples: int = 4096,
    delimiter: str = "\t",
    user_field: str = "user_id",
    item_field: str = "item_id",
    rating_field: str = "rating",
) -> Dict[str, Any]:
    """Run the complete audit and return a JSON-serializable dictionary."""

    if landmarks < 0:
        raise ValueError("landmarks must be non-negative")
    if four_point_samples < 0:
        raise ValueError("four_point_samples must be non-negative")
    path = path.expanduser().resolve()
    interactions, input_stats = _read_atomic_inter(
        path,
        delimiter,
        user_field,
        item_field,
        rating_field,
        rating_threshold,
    )
    (
        alive_users,
        alive_items,
        edge_users,
        edge_items,
        edge_multiplicities,
    ) = _iterative_bipartite_k_core(interactions, k_core)
    user_count = int(np.count_nonzero(alive_users))
    item_count = int(np.count_nonzero(alive_items))
    unique_edge_count = int(edge_users.size)
    interaction_count = int(edge_multiplicities.sum())
    adjacency = _combined_adjacency(user_count, item_count, edge_users, edge_items)
    topology_degrees = np.diff(adjacency.indptr).astype(np.int64, copy=False)
    user_topology_degrees = topology_degrees[:user_count]
    item_topology_degrees = topology_degrees[user_count:]
    user_interaction_degrees = np.zeros(user_count, dtype=np.int64)
    item_interaction_degrees = np.zeros(item_count, dtype=np.int64)
    np.add.at(user_interaction_degrees, edge_users, edge_multiplicities)
    np.add.at(item_interaction_degrees, edge_items, edge_multiplicities)
    interaction_degrees = np.concatenate(
        (user_interaction_degrees, item_interaction_degrees)
    )

    report: Dict[str, Any] = {
        "schema_version": 1,
        "input": {
            "path": str(path),
            "delimiter": "\\t" if delimiter == "\t" else delimiter,
            "user_field": user_field,
            "item_field": item_field,
            "rating_field": rating_field,
            "sha256": input_stats.pop("sha256"),
        },
        "parameters": {
            "rating_threshold_inclusive": rating_threshold,
            "iterative_k_core": k_core,
            "seed": seed,
            "landmarks_requested": landmarks,
            "four_point_samples_requested": four_point_samples,
            "k_core_degree_semantics": "interaction row multiplicity (RecBole-compatible)",
            "shortest_path_topology": "simple unweighted user-item graph",
        },
        "counts": {
            **input_stats,
            "post_k_core_users": user_count,
            "post_k_core_items": item_count,
            "post_k_core_nodes": user_count + item_count,
            "post_k_core_interactions": interaction_count,
            "post_k_core_unique_user_item_edges": unique_edge_count,
        },
        "degree_distribution_post_k_core": {
            "interaction_multiplicity": {
                "users": _degree_summary(user_interaction_degrees),
                "items": _degree_summary(item_interaction_degrees),
                "all_nodes": _degree_summary(interaction_degrees),
            },
            "simple_topology": {
                "users": _degree_summary(user_topology_degrees),
                "items": _degree_summary(item_topology_degrees),
                "all_nodes": _degree_summary(topology_degrees),
            },
        },
    }

    if adjacency.shape[0] == 0:
        report.update(
            {
                "connected_components": {
                    "count": 0,
                    "largest": None,
                    "top_10_node_counts": [],
                },
                "four_point_hyperbolicity_lcc": {
                    "status": "empty_post_k_core_graph"
                },
                "cycle_and_branching_proxies": {
                    "cycle_rank": 0,
                    "cycle_rank_over_edges": None,
                    "largest_component_cycle_rank": None,
                    "user_mean_excess_degree": None,
                    "item_mean_excess_degree": None,
                    "bipartite_two_step_branching_product": None,
                    "per_step_branching_geometric_mean": None,
                },
            }
        )
    else:
        component_count, labels = csgraph.connected_components(
            adjacency, directed=False, return_labels=True
        )
        component_sizes = np.bincount(labels, minlength=component_count)
        largest_label = int(np.argmax(component_sizes))
        largest_nodes = np.flatnonzero(labels == largest_label)
        largest_user_count = int(np.count_nonzero(largest_nodes < user_count))
        largest_item_count = int(largest_nodes.size - largest_user_count)
        edge_in_largest = labels[edge_users] == largest_label
        largest_unique_edge_count = int(np.count_nonzero(edge_in_largest))
        largest_interaction_count = int(edge_multiplicities[edge_in_largest].sum())
        top_component_sizes = np.sort(component_sizes)[::-1][:10]
        lcc_adjacency = adjacency[largest_nodes][:, largest_nodes].tocsr()

        user_excess = _mean_excess_degree(user_topology_degrees)
        item_excess = _mean_excess_degree(item_topology_degrees)
        two_step_branching = (
            user_excess * item_excess
            if user_excess is not None and item_excess is not None
            else None
        )
        cycle_rank = unique_edge_count - adjacency.shape[0] + component_count
        largest_cycle_rank = largest_unique_edge_count - largest_nodes.size + 1
        report.update(
            {
                "connected_components": {
                    "count": int(component_count),
                    "top_10_node_counts": [int(value) for value in top_component_sizes],
                    "largest": {
                        "users": largest_user_count,
                        "items": largest_item_count,
                        "nodes": int(largest_nodes.size),
                        "interactions": largest_interaction_count,
                        "unique_user_item_edges": largest_unique_edge_count,
                        "node_fraction": float(largest_nodes.size / adjacency.shape[0]),
                        "interaction_fraction": float(
                            largest_interaction_count / interaction_count
                        ),
                        "unique_edge_fraction": float(
                            largest_unique_edge_count / unique_edge_count
                        ),
                    },
                },
                "four_point_hyperbolicity_lcc": _sample_four_point_delta(
                    lcc_adjacency,
                    seed=seed,
                    landmark_count=landmarks,
                    sample_count=four_point_samples,
                ),
                "cycle_and_branching_proxies": {
                    "cycle_rank": int(cycle_rank),
                    "cycle_rank_definition": (
                        "E_unique - V + connected_components on the simple topology"
                    ),
                    "cycle_rank_over_edges": (
                        float(cycle_rank / unique_edge_count)
                        if unique_edge_count
                        else None
                    ),
                    "largest_component_cycle_rank": int(largest_cycle_rank),
                    "user_mean_excess_degree": user_excess,
                    "item_mean_excess_degree": item_excess,
                    "bipartite_two_step_branching_product": two_step_branching,
                    "per_step_branching_geometric_mean": (
                        float(math.sqrt(two_step_branching))
                        if two_step_branching is not None
                        else None
                    ),
                    "branching_definition": (
                        "Using simple-topology degree, side mean excess "
                        "degree=sum d(d-1)/sum d; the product is the two-hop "
                        "configuration-model continuation proxy"
                    ),
                },
            }
        )

    report["interpretation_limits"] = {
        "diagnostic_only": True,
        "warning": (
            "These are structural diagnostics of an unweighted discrete bipartite "
            "graph. Discrete graph hyperbolicity, cycle rank, and branching proxies "
            "are not smooth-manifold curvature and cannot by themselves establish "
            "that Euclidean, hyperbolic, SL(n), or a mixed manifold will train better."
        ),
    }
    report["complexity"] = {
        "time": (
            "O(R + V + E + L*(V_lcc + E_lcc) + L^2*D_lcc + Q)"
        ),
        "peak_memory": "O(R_pass + V + E + V_lcc + L^2)",
        "bounded_sampling_note": (
            "No V-by-V all-pairs matrix is allocated; keep L fixed (default 32) "
            "rather than scaling it with V."
        ),
    }
    return report


def _decode_delimiter(value: str) -> str:
    aliases = {"tab": "\t", "\\t": "\t", "comma": ",", "space": " "}
    delimiter = aliases.get(value.lower(), value)
    if len(delimiter) != 1:
        raise argparse.ArgumentTypeError("delimiter must be one character or tab/comma")
    return delimiter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inter", type=Path, help="RecBole atomic .inter file")
    parser.add_argument("--rating-threshold", type=float, default=3.0)
    parser.add_argument("--k-core", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--landmarks", type=int, default=32)
    parser.add_argument("--four-point-samples", type=int, default=4096)
    parser.add_argument("--delimiter", type=_decode_delimiter, default="\t")
    parser.add_argument("--user-field", default="user_id")
    parser.add_argument("--item-field", default="item_id")
    parser.add_argument("--rating-field", default="rating")
    parser.add_argument(
        "--output", type=Path, default=None, help="JSON path (default: stdout)"
    )
    args = parser.parse_args()
    report = audit_interaction_graph(
        args.inter,
        rating_threshold=args.rating_threshold,
        k_core=args.k_core,
        seed=args.seed,
        landmarks=args.landmarks,
        four_point_samples=args.four_point_samples,
        delimiter=args.delimiter,
        user_field=args.user_field,
        item_field=args.item_field,
        rating_field=args.rating_field,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
