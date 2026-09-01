from pathlib import Path

import pytest

from slrec_experiments.audit_graph_geometry import audit_interaction_graph


def _write_inter(path: Path, rows: str) -> Path:
    path.write_text(
        "user_id:token\titem_id:token\trating:float\n" + rows,
        encoding="utf-8",
    )
    return path


def test_k22_cycle_counts_deduplication_and_exact_four_point_delta(tmp_path):
    path = _write_inter(
        tmp_path / "cycle.inter",
        "u1\ti1\t5\n"
        "u1\ti2\t4\n"
        "u2\ti1\t3\n"
        "u2\ti2\t5\n"
        "u2\ti2\t5\n"  # repeated event collapses in the simple graph
        "u3\ti3\t2\n",  # present in the raw file, removed by rating threshold
    )
    report = audit_interaction_graph(
        path, k_core=2, landmarks=10, four_point_samples=32, seed=7
    )

    counts = report["counts"]
    assert counts["raw_distinct_users"] == 3
    assert counts["raw_distinct_items"] == 3
    assert counts["rating_pass_rows"] == 5
    assert counts["rating_pass_unique_edges"] == 4
    assert counts["duplicate_rating_pass_edges_collapsed"] == 1
    assert counts["post_k_core_users"] == 2
    assert counts["post_k_core_items"] == 2
    assert counts["post_k_core_interactions"] == 5
    assert counts["post_k_core_unique_user_item_edges"] == 4

    assert report["connected_components"]["count"] == 1
    assert report["cycle_and_branching_proxies"]["cycle_rank"] == 1
    topology_degree = report["degree_distribution_post_k_core"]["simple_topology"]
    assert topology_degree["all_nodes"]["gini"] == 0.0
    hyperbolicity = report["four_point_hyperbolicity_lcc"]
    assert hyperbolicity["landmarks"] == 4
    assert hyperbolicity["delta"]["mean"] == 1.0
    assert hyperbolicity["delta"]["max"] == 1.0
    assert hyperbolicity["landmark_pair_max_distance"] == 2
    assert hyperbolicity["normalized_delta_over_landmark_pair_max"]["max"] == 0.5


def test_iterative_k_core_cascade_can_empty_graph(tmp_path):
    path = _write_inter(
        tmp_path / "path.inter",
        "u1\ti1\t5\n"
        "u2\ti1\t5\n"
        "u2\ti2\t5\n",
    )
    report = audit_interaction_graph(
        path, k_core=2, landmarks=4, four_point_samples=8
    )

    assert report["counts"]["post_k_core_nodes"] == 0
    assert report["counts"]["post_k_core_interactions"] == 0
    assert report["counts"]["post_k_core_unique_user_item_edges"] == 0
    assert report["connected_components"]["count"] == 0
    assert (
        report["four_point_hyperbolicity_lcc"]["status"]
        == "empty_post_k_core_graph"
    )


def test_audit_is_deterministic_for_fixed_seed(tmp_path):
    path = _write_inter(
        tmp_path / "graph.inter",
        "u1\ti1\t5\n"
        "u1\ti2\t5\n"
        "u2\ti1\t5\n"
        "u2\ti2\t5\n"
        "u2\ti3\t5\n"
        "u3\ti2\t5\n"
        "u3\ti3\t5\n",
    )
    first = audit_interaction_graph(
        path, k_core=1, landmarks=5, four_point_samples=50, seed=2024
    )
    second = audit_interaction_graph(
        path, k_core=1, landmarks=5, four_point_samples=50, seed=2024
    )
    assert first == second


def test_rejects_nonfinite_rating(tmp_path):
    path = _write_inter(tmp_path / "bad.inter", "u1\ti1\tnan\n")
    with pytest.raises(ValueError, match="Non-finite rating"):
        audit_interaction_graph(path)
