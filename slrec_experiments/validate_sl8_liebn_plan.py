#!/usr/bin/env python3
"""Validate the grouped SL8-LieBN Amazon-CD experiment plan."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PLAN_PATH = Path(__file__).parent / "queues" / "sl8_liebn_amazon_cd_plan.json"
PARAMETER_KEYS = (
    "layers",
    "batch_size",
    "learning_rate",
    "loss_margin",
    "coord_clip",
)
ANCHORS = {
    2: {
        "layers": 2,
        "batch_size": 16384,
        "learning_rate": 0.005,
        "loss_margin": 0.1,
        "coord_clip": 0.75,
    },
    4: {
        "layers": 4,
        "batch_size": 16384,
        "learning_rate": 0.005,
        "loss_margin": 0.1,
        "coord_clip": 0.75,
    },
}
GROUP_COUNTS = {"priority_l2_l4": 20, "planned_secondary": 10}
AXIS_LABELS = {
    "shared_anchor_l2": "Shared L2 anchor",
    "l2_learning_rate": "L2 learning rate",
    "l2_train_batch_size": "L2 train batch size",
    "shared_anchor_l4": "Shared L4 anchor",
    "l4_learning_rate": "L4 learning rate",
    "l4_train_batch_size": "L4 train batch size",
    "gcn_layers": "GCN layers",
    "l4_coord_clip": "L4 coordinate clip",
    "l4_loss_margin": "L4 hinge loss margin",
}
AXIS_PARAMETER = {
    "l2_learning_rate": "learning_rate",
    "l2_train_batch_size": "batch_size",
    "l4_learning_rate": "learning_rate",
    "l4_train_batch_size": "batch_size",
    "l4_coord_clip": "coord_clip",
    "l4_loss_margin": "loss_margin",
}
AXIS_VALUES = {
    "gcn_layers": {0, 2, 4, 6},
    "l2_train_batch_size": {16384, 32768, 65536, 131072},
    "l2_learning_rate": {0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01},
    "l4_train_batch_size": {16384, 32768, 65536, 131072},
    "l4_learning_rate": {0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01},
    "l4_coord_clip": {0.0, 0.5, 0.75, 1.0, 1.5, 2.0},
    "l4_loss_margin": {0.05, 0.1, 0.2, 0.3},
}
PROTOCOL = {
    "epochs": 500,
    "eval_step": 10,
    "stopping_step": 2,
    "validation_only": True,
    "seed": 2024,
    "reproducibility": True,
    "schatten_p": 2,
    "sl_score_mode": "group_log",
    "sl_gcn_mode": "karcher1",
    "sl_karcher_correction": False,
    "sl_layer_norm": "liebn",
    "liebn_mean": "karcher1",
    "liebn_dispersion": "mean_norm",
    "liebn_max_log_norm": 25.0,
    "liebn_max_tangent_norm": 3.0,
    "log_domain_sqrt_steps": 1,
    "eval_log_domain_sqrt_steps": 0,
    "log_domain_sqrt_iterations": 12,
    "log_domain_sqrt_residual_tolerance": 0.001,
    "log_domain_tail_tolerance": 0.001,
    "log_domain_guard_revision": "db_residual_spectral_tail_v1",
    "eval_prefilter": "frobenius",
    "eval_prefilter_candidates": 4096,
}


def _signature(trial: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(trial[key] for key in PARAMETER_KEYS)


def _assert_protocol(payload: Mapping[str, Any]) -> None:
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be an object")
    for key, expected in PROTOCOL.items():
        if protocol.get(key) != expected:
            raise ValueError(f"protocol mismatch for {key}: {protocol.get(key)!r}")


def _assert_group_metadata(payload: Mapping[str, Any]) -> None:
    metadata = payload.get("trial_groups")
    if not isinstance(metadata, Mapping) or set(metadata) != set(GROUP_COUNTS):
        raise ValueError("trial_groups must define exactly the two planned groups")
    for group, expected_count in GROUP_COUNTS.items():
        if metadata[group].get("required_count") != expected_count:
            raise ValueError(f"required_count mismatch for {group}")
    priority = metadata["priority_l2_l4"]
    if priority.get("design") != "one_factor_main_effects":
        raise ValueError("priority design must be one_factor_main_effects")
    if priority.get("cartesian_product") is not False:
        raise ValueError("priority grid must not be labelled Cartesian")

    expansions = payload.get("optional_expansions")
    if not isinstance(expansions, list) or len(expansions) != 1:
        raise ValueError("one optional batch-by-learning-rate expansion is required")
    expansion = expansions[0]
    expected_additional = 2 * 4 * 7 - GROUP_COUNTS["priority_l2_l4"]
    if expansion.get("enabled") is not False:
        raise ValueError("optional Cartesian expansion must be disabled")
    if expansion.get("additional_trial_count") != expected_additional:
        raise ValueError("optional Cartesian expansion must add 36 interactions")
    if expansion.get("additional_trial_count_per_layer") != expected_additional // 2:
        raise ValueError("optional Cartesian expansion must add 18 trials per layer")
    if expansion.get("resulting_priority_trial_count") != 56:
        raise ValueError("full L2/L4 Cartesian grid must contain 56 trials")
    if expansion.get("resulting_total_with_secondary") != 66:
        raise ValueError("full expanded plan plus secondary trials must total 66")


def _assert_axis_metadata(payload: Mapping[str, Any], trials_by_id: Mapping[str, Any]) -> None:
    definitions = payload.get("axis_definitions")
    if not isinstance(definitions, Mapping) or set(definitions) != set(AXIS_VALUES):
        raise ValueError("axis_definitions do not match the planned axes")
    for axis, expected_values in AXIS_VALUES.items():
        definition = definitions[axis]
        if definition.get("label") != AXIS_LABELS[axis]:
            raise ValueError(f"axis definition label mismatch for {axis}")
        values_key = (
            "required_values_including_anchors"
            if axis == "gcn_layers"
            else "required_values_including_anchor"
        )
        if set(definition.get(values_key, ())) != expected_values:
            raise ValueError(f"axis definition values mismatch for {axis}")
        anchor_ids = definition.get("anchor_trial_ids")
        if anchor_ids is None:
            anchor_ids = [definition.get("anchor_trial_id")]
        if any(anchor_id not in trials_by_id for anchor_id in anchor_ids):
            raise ValueError(f"axis definition references a missing anchor for {axis}")


def _assert_trial_axes(trials: list[Mapping[str, Any]]) -> None:
    anchor_by_axis = {"shared_anchor_l2": ANCHORS[2], "shared_anchor_l4": ANCHORS[4]}
    for trial in trials:
        axis = trial.get("axis")
        if trial.get("axis_label") != AXIS_LABELS.get(axis):
            raise ValueError(f"axis label mismatch in {trial.get('id')}")
        if axis in anchor_by_axis:
            if _signature(trial) != _signature(anchor_by_axis[axis]):
                raise ValueError(f"{axis} does not match its declared anchor")
            if trial.get("axis_value") != "anchor":
                raise ValueError(f"{axis} needs axis_value='anchor'")
            continue
        if axis == "gcn_layers":
            expected = dict(ANCHORS[4], layers=trial["layers"])
            if _signature(trial) != _signature(expected):
                raise ValueError(f"{trial['id']} is not a layer-only variation")
            if trial.get("axis_value") != trial["layers"]:
                raise ValueError(f"axis_value mismatch in {trial['id']}")
            continue
        parameter = AXIS_PARAMETER.get(axis)
        if parameter is None:
            raise ValueError(f"unknown trial axis: {axis!r}")
        layer = 2 if axis.startswith("l2_") else 4
        anchor = ANCHORS[layer]
        changed = {key for key in PARAMETER_KEYS if trial[key] != anchor[key]}
        if changed != {parameter}:
            raise ValueError(
                f"{trial['id']} must differ from its L{layer} anchor only on {parameter}"
            )
        expected_axis_value: Any = trial[parameter]
        if axis == "l4_coord_clip" and expected_axis_value == 0.0:
            expected_axis_value = "disabled"
        if trial.get("axis_value") != expected_axis_value:
            raise ValueError(f"axis_value mismatch in {trial['id']}")

    for axis, expected_values in AXIS_VALUES.items():
        if axis == "gcn_layers":
            actual = {2, 4}
            actual.update(trial["layers"] for trial in trials if trial["axis"] == axis)
        else:
            parameter = AXIS_PARAMETER[axis]
            layer = 2 if axis.startswith("l2_") else 4
            actual = {ANCHORS[layer][parameter]}
            actual.update(trial[parameter] for trial in trials if trial["axis"] == axis)
        if actual != expected_values:
            raise ValueError(f"unexpected allowed values for {axis}: {actual!r}")


def validate_plan(payload: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if *payload* is not the canonical grouped plan."""

    if payload.get("schema_version") != 2:
        raise ValueError("unsupported plan schema")
    if payload.get("model") != "SL8LHGCN" or payload.get("dataset") != "Amazon_cd":
        raise ValueError("plan must target SL8LHGCN on Amazon_cd")
    _assert_protocol(payload)
    _assert_group_metadata(payload)

    trials = payload.get("trials")
    if not isinstance(trials, list) or len(trials) != sum(GROUP_COUNTS.values()):
        raise ValueError("the grouped plan must contain exactly 30 trials")
    ids = [trial.get("id") for trial in trials]
    if any(not isinstance(trial_id, str) or not trial_id for trial_id in ids):
        raise ValueError("every trial needs a stable non-empty string id")
    if len(ids) != len(set(ids)):
        raise ValueError("trial ids must be unique")
    signatures = [_signature(trial) for trial in trials]
    if len(signatures) != len(set(signatures)):
        raise ValueError("parameter combinations must be unique")

    group_counts = Counter(trial.get("group") for trial in trials)
    if group_counts != Counter(GROUP_COUNTS):
        raise ValueError(f"unexpected trial group counts: {group_counts!r}")
    priority = [trial for trial in trials if trial["group"] == "priority_l2_l4"]
    if Counter(trial["layers"] for trial in priority) != Counter({2: 10, 4: 10}):
        raise ValueError("priority group must contain 10 L2 and 10 L4 trials")
    if any(
        trial["batch_size"] != 16384 and trial["learning_rate"] != 0.005
        for trial in priority
    ):
        raise ValueError("priority trials must be one-factor, not Cartesian interactions")

    trials_by_id = dict(zip(ids, trials))
    _assert_axis_metadata(payload, trials_by_id)
    _assert_trial_axes(trials)

    extensions = payload.get("conditional_extensions")
    if not isinstance(extensions, list) or len(extensions) != 1:
        raise ValueError("L8 must be documented as one conditional extension")
    l8 = extensions[0]
    if l8.get("included_in_planned_30") is not False or l8.get("layers") != 8:
        raise ValueError("conditional L8 must stay outside the planned 30 trials")
    if _signature(l8) in set(signatures):
        raise ValueError("conditional L8 must not duplicate a planned trial")


def load_and_validate(path: Path = PLAN_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("plan JSON root must be an object")
    validate_plan(payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", nargs="?", type=Path, default=PLAN_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = load_and_validate(args.plan)
    groups = Counter(trial["group"] for trial in payload["trials"])
    print(
        f"OK: {payload['plan_id']} has {groups['priority_l2_l4']} priority + "
        f"{groups['planned_secondary']} secondary unique trials; conditional L8 excluded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
