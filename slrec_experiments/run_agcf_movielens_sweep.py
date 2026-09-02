#!/usr/bin/env python3
"""Blocked, validation-only AGCF sweep for the paper MovieLens protocol.

This is a deliberately strict experiment runner:

* it composes ``AGCF_movielens_protocol.yaml`` before ``AGCF_cd.yaml``;
* every child is a validation-only, full-ranking RecBole run;
* the paper training contract is fixed at 500 epochs, validation every epoch,
  patience 30, seed 2020, and one negative per positive;
* one foreground child at a time may use one physical CUDA device;
* each search stage inherits the preceding stage's validation winner; and
* a completed result is resumed only after its metadata, split fingerprints,
  checkpoint configuration, source data, and campaign definition all match.

The AGCF paper says that learning rate, weight decay, margin, output count L,
integration count K, potential strength alpha, and damping gamma are tuned but
does not publish their grids or selected values.  It also omits the low-rank
metric parameterisation.  The grids below are therefore explicit clean-room
reproduction choices, not claimed author settings.

There is intentionally no held-out-test code path in this module.  Freeze one
validation winner first and evaluate it with a separate command exactly once.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import shlex
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SCHEMA_VERSION = 1
DATASET = "AGCF_MovieLens"
MODEL = "AGCF"
SEED = 2020
EPOCHS = 500
EVAL_STEP = 1
STOPPING_STEP = 30
SELECTION_METRIC = "Recall@10 on full-ranking validation"
CONFIG_NAMES = ("AGCF_movielens_protocol.yaml", "AGCF_cd.yaml")

# This is the official MovieLens-1M ratings file converted losslessly to
# RecBole atomic format.  The rating/5-core filters are applied by RecBole,
# not baked into this file.
MOVIELENS_SOURCE = {
    "relative_file": "AGCF_MovieLens/AGCF_MovieLens.inter",
    "bytes": 21_593_561,
    "lines_including_header": 1_000_210,
    "interactions_excluding_header": 1_000_209,
    "sha256": "e943abb91013a54c385828fdf5ab4ce49e957ca3a772adb30cde2a7d5539b389",
}
MOVIELENS_FILTERED = {
    # RecBole counters include reserved token id 0.  These are exactly the
    # MovieLens counts printed in Table 2 after rating>=3 and joint 5-core.
    "framework_users": 6_039,
    "framework_items": 3_308,
    "interactions": 835_789,
    "token_users": 6_038,
    "token_items": 3_307,
}

PROTECTED_PROTOCOL_KEYS = {
    "dataset",
    "seed",
    "reproducibility",
    "field_separator",
    "USER_ID_FIELD",
    "ITEM_ID_FIELD",
    "RATING_FIELD",
    "NEG_PREFIX",
    "load_col",
    "val_interval",
    "user_inter_num_interval",
    "item_inter_num_interval",
    "metrics",
    "topk",
    "valid_metric",
    "eval_args",
    "epochs",
    "eval_step",
    "stopping_step",
}


@dataclass(frozen=True)
class Parameters:
    """Complete AGCF search state for a single trial."""

    metric_rank: int = 16
    channel_rank: int = 64
    train_batch_size: int = 4096
    learning_rate: float = 1e-3
    margin: float = 0.1
    output_steps: int = 1
    integration_steps: int = 1
    potential_strength: float = 0.1
    damping: float = 0.01
    weight_decay: float = 0.0
    metric_epsilon: float = 1e-3
    structural_delta: float = 1e-3

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Parameters":
        return cls(
            metric_rank=int(payload["metric_rank"]),
            channel_rank=int(payload["channel_rank"]),
            train_batch_size=int(payload["train_batch_size"]),
            learning_rate=float(payload["learning_rate"]),
            margin=float(payload["margin"]),
            output_steps=int(payload["output_steps"]),
            integration_steps=int(payload["integration_steps"]),
            potential_strength=float(payload["potential_strength"]),
            damping=float(payload["damping"]),
            weight_decay=float(payload["weight_decay"]),
            metric_epsilon=float(payload["metric_epsilon"]),
            structural_delta=float(payload["structural_delta"]),
        )

    def validate(self) -> None:
        for name in (
            "metric_rank",
            "channel_rank",
            "train_batch_size",
            "integration_steps",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.output_steps < 0:
            raise ValueError("output_steps must be non-negative")
        if self.metric_rank > 64 or self.channel_rank > 64:
            raise ValueError("metric/channel rank cannot exceed embedding_size=64")
        for name in (
            "learning_rate",
            "margin",
            "potential_strength",
            "damping",
            "weight_decay",
            "metric_epsilon",
            "structural_delta",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate == 0 or self.metric_epsilon == 0:
            raise ValueError("learning_rate and metric_epsilon must be positive")

    def recbole_values(self) -> dict[str, Any]:
        return asdict(self)


ANCHOR = Parameters()

# Blocked one-parameter stages.  The winner of each row becomes the fixed
# anchor for the next row; these values are never expanded into one enormous
# Cartesian product.  The user's "2028" batch request is treated as the
# conventional power-of-two value 2048.
STAGE_GRIDS: tuple[tuple[str, str, tuple[Any, ...]], ...] = (
    # Optimizer exposure is the first ambiguity to resolve.  Include a genuinely
    # small batch for MovieLens, but stop at B=256: B=128 roughly doubles an
    # already expensive B=256 trial without adding a meaningfully distinct
    # optimization regime.  Large-batch B=8192 is deliberately excluded.
    ("batch-size", "train_batch_size", (256, 512, 1024, 2048, 4096)),
    (
        "learning-rate",
        "learning_rate",
        (1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3),
    ),
    ("margin", "margin", (0.02, 0.05, 0.1, 0.2, 0.3)),
    ("metric-rank", "metric_rank", (4, 8, 16, 32, 64)),
    ("channel-rank", "channel_rank", (16, 32, 64)),
    ("output-steps", "output_steps", (1, 2, 3, 4, 6)),
    ("integration-steps", "integration_steps", (1, 2, 4, 8)),
    (
        "potential-strength",
        "potential_strength",
        (0.01, 0.05, 0.1, 0.2, 0.5, 1.0),
    ),
    ("damping", "damping", (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5)),
    ("weight-decay", "weight_decay", (0.0, 1e-6, 1e-5, 1e-4, 1e-3)),
)
STAGE_ORDER = ("anchor", *(item[0] for item in STAGE_GRIDS))
STAGE_DEFINITIONS = {
    name: {"field": field, "values": values}
    for name, field, values in STAGE_GRIDS
}


@dataclass(frozen=True)
class Trial:
    stage: str
    label: str
    parameters: Parameters

    @property
    def name(self) -> str:
        state = json.dumps(
            asdict(self.parameters), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        suffix = hashlib.sha256(state).hexdigest()[:12]
        return f"{self.stage}__{self.label}__{suffix}"


@dataclass(frozen=True)
class Candidate:
    name: str
    source: str
    checkpoint_file: str
    parameters: Parameters
    best_valid_score: float
    best_valid_result: Mapping[str, Any]

    def json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "checkpoint_file": self.checkpoint_file,
            "parameters": asdict(self.parameters),
            "best_valid_score": self.best_valid_score,
            "best_valid_result": dict(self.best_valid_result),
        }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return payload


def config_paths(repo: Path) -> tuple[Path, ...]:
    root = repo / "baseline_config_fixed"
    return tuple(root / name for name in CONFIG_NAMES)


def audit_movielens_source(data_root: Path) -> dict[str, Any]:
    """Require the exact official-ML1M-derived atomic source."""

    resolved_root = data_root.expanduser().resolve()
    path = resolved_root / MOVIELENS_SOURCE["relative_file"]
    if not path.is_file():
        raise FileNotFoundError(
            f"missing pinned MovieLens source: {path}; --data-path must contain "
            "AGCF_MovieLens/AGCF_MovieLens.inter"
        )
    byte_count = path.stat().st_size
    digest = hashlib.sha256()
    newline_count = 0
    final_byte = b""
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
            newline_count += block.count(b"\n")
            final_byte = block[-1:]
    line_count = newline_count + (1 if final_byte and final_byte != b"\n" else 0)
    actual = {
        "bytes": byte_count,
        "lines_including_header": line_count,
        "sha256": digest.hexdigest(),
    }
    expected = {
        key: MOVIELENS_SOURCE[key]
        for key in ("bytes", "lines_including_header", "sha256")
    }
    differences = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value
    }
    if differences:
        raise ValueError(
            f"MovieLens atomic source is not the pinned release: {differences}"
        )
    return {
        "data_root": str(resolved_root),
        "file": str(path),
        **actual,
        "interactions_excluding_header": MOVIELENS_SOURCE[
            "interactions_excluding_header"
        ],
        "verified": True,
    }


def validate_model_registration(repo: Path) -> Path:
    """Check import discovery without importing PyTorch for plan-only mode."""

    module = (
        repo
        / "recbole_gnn"
        / "model"
        / "general_recommender"
        / "agcf.py"
    )
    if not module.is_file():
        raise RuntimeError(f"AGCF module is missing: {module}")
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    if not any(
        isinstance(node, ast.ClassDef) and node.name == MODEL for node in tree.body
    ):
        raise RuntimeError(f"{module} does not define class {MODEL}")
    return module


def validate_protocol(
    repo: Path, *, source_audit: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    paths = config_paths(repo)
    protocol = _yaml_mapping(paths[0])
    model = _yaml_mapping(paths[1])
    overlap = PROTECTED_PROTOCOL_KEYS.intersection(model)
    if overlap:
        raise RuntimeError(
            f"model overlay {paths[1].name} overrides protocol fields: "
            f"{sorted(overlap)}"
        )

    expected = {
        "dataset": DATASET,
        "seed": SEED,
        "reproducibility": True,
        "learner": "adam",
        "embedding_size": 64,
        "val_interval": {"rating": "[3,inf)"},
        "user_inter_num_interval": "[5,inf)",
        "item_inter_num_interval": "[5,inf)",
        "metrics": ["Recall", "NDCG"],
        "topk": [10, 20],
        "valid_metric": "Recall@10",
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        },
        "epochs": EPOCHS,
        "eval_step": EVAL_STEP,
        "stopping_step": STOPPING_STEP,
    }
    differences = {
        key: {"expected": value, "actual": protocol.get(key)}
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if differences:
        raise RuntimeError(f"MovieLens paper protocol changed: {differences}")

    expected_model = {
        "model": MODEL,
        "embedding_size": 64,
        "evolution_time": 1.0,
        "metric_hidden_size": 64,
        "pnet_hidden_size": 64,
    }
    model_differences = {
        key: {"expected": value, "actual": model.get(key)}
        for key, value in expected_model.items()
        if model.get(key) != value
    }
    if model_differences:
        raise RuntimeError(f"AGCF model overlay changed: {model_differences}")

    ANCHOR.validate()
    return {
        "dataset": DATASET,
        "model": MODEL,
        "seed": SEED,
        "config_files": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in paths
        ],
        "filters": {"rating": "[3,inf)", "users": "[5,inf)", "items": "[5,inf)"},
        "split": expected["eval_args"],
        "evaluation": {
            "metrics": expected["metrics"],
            "topk": expected["topk"],
            "selection_metric": expected["valid_metric"],
            "mode": "full",
            "validation_only": True,
            "held_out_test_evaluated": False,
        },
        "raw_source": (
            dict(source_audit)
            if source_audit is not None
            else {
                "data_root": str((repo / "dataset").resolve()),
                "file": str(
                    (repo / "dataset" / MOVIELENS_SOURCE["relative_file"]).resolve()
                ),
                **MOVIELENS_SOURCE,
                "verified": False,
            }
        ),
        "expected_filtered_dataset": dict(MOVIELENS_FILTERED),
        "anchor_parameters": asdict(ANCHOR),
    }


def campaign_contract(
    protocol: Mapping[str, Any], data_root: Path
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "training": {
            "epochs": EPOCHS,
            "eval_step": EVAL_STEP,
            "stopping_step": STOPPING_STEP,
            "seed": SEED,
            "negative_sampling": {"uniform": 1},
            "data_root": str(data_root.expanduser().resolve()),
            "serial_workers": 1,
            "logical_gpu": 0,
            "dynamics_chunk_size": 4096,
            "checkpoint_dynamics": False,
            "eval_item_chunk_size": 4096,
        },
        "search": {
            "method": "blocked_validation_winner_continuation",
            "anchor": asdict(ANCHOR),
            "stages": [
                {"name": "anchor", "field": None, "values": None},
                *[
                    {"name": name, "field": field, "values": list(values)}
                    for name, field, values in STAGE_GRIDS
                ],
            ],
        },
        "selection": {
            "metric": SELECTION_METRIC,
            "validation_only": True,
            "test_evaluated": False,
        },
    }


def _value_label(field: str, value: Any) -> str:
    if isinstance(value, float):
        token = f"{value:.8g}".replace("-", "m").replace(".", "p")
    else:
        token = str(value)
    return f"{field}-{token}"


def build_stage_trials(stage: str, anchor: Parameters) -> tuple[Trial, ...]:
    """Materialise one blocked stage around its inherited winner."""

    anchor.validate()
    if stage == "anchor":
        return (Trial("anchor", "mr16-cr64-B4096-lr1em3", ANCHOR),)
    if stage not in STAGE_DEFINITIONS:
        raise ValueError(f"unknown stage {stage!r}; choices={STAGE_ORDER}")
    definition = STAGE_DEFINITIONS[stage]
    field = str(definition["field"])
    trials = []
    seen = {anchor}
    for value in definition["values"]:
        parameters = replace(anchor, **{field: value})
        parameters.validate()
        if parameters in seen:
            continue
        seen.add(parameters)
        trials.append(Trial(stage, _value_label(field, value), parameters))
    return tuple(trials)


def trial_paths(output_root: Path, trial: Trial) -> dict[str, Path]:
    root = output_root / MODEL.lower() / trial.stage
    return {
        "root": root,
        "result": root / "results" / f"{trial.name}.json",
        "log": root / "logs" / f"{trial.name}.log",
        "checkpoint_dir": root / "checkpoints" / trial.name,
        "summary": root / "summary.json",
    }


def stage_summary_path(output_root: Path, stage: str) -> Path:
    return output_root / MODEL.lower() / stage / "summary.json"


def _recbole_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def trial_command(
    args: argparse.Namespace,
    trial: Trial,
    result_path: Path,
    checkpoint_dir: Path,
) -> list[str]:
    configs = " ".join(str(path) for path in config_paths(args.repo))
    command = [
        args.python,
        "-u",
        str(args.repo / "run_recbole_gnn.py"),
        "--model",
        MODEL,
        "--dataset",
        DATASET,
        "--config-files",
        configs,
        "--validation-only",
        "--result-file",
        str(result_path),
        f"--checkpoint_dir={checkpoint_dir}",
        # The vendored Config rewrites CUDA_VISIBLE_DEVICES from gpu_id.  Pass
        # the same physical id in both places; PyTorch then sees logical cuda:0.
        f"--gpu_id={args.gpu_id}",
        "--use_gpu=True",
        "--show_progress=False",
        f"--epochs={EPOCHS}",
        f"--eval_step={EVAL_STEP}",
        f"--stopping_step={STOPPING_STEP}",
        f"--data_path={args.data_path}",
        f"--seed={SEED}",
        "--reproducibility=True",
        "--reg_weight=0",
        "--neg_sampling={'uniform': 1}",
        "--dynamics_chunk_size=4096",
        "--checkpoint_dynamics=False",
        "--eval_item_chunk_size=4096",
    ]
    for key, value in trial.parameters.recbole_values().items():
        command.append(f"--{key}={_recbole_scalar(value)}")
    return command


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    try:
        return config[key]
    except (KeyError, TypeError, AttributeError):
        dictionary = getattr(config, "final_config_dict", None)
        return dictionary.get(key) if isinstance(dictionary, Mapping) else None


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _load_checkpoint_config(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")
    import torch

    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # Older PyTorch.
        checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, Mapping) or "config" not in checkpoint:
        raise ValueError(f"not a RecBole checkpoint: {path}")
    return checkpoint["config"]


def validate_checkpoint_contract(
    checkpoint: Path,
    trial: Trial,
    contract: Mapping[str, Any],
) -> None:
    config = _load_checkpoint_config(checkpoint)
    expected: dict[str, Any] = {
        "model": MODEL,
        "dataset": DATASET,
        "seed": SEED,
        "embedding_size": 64,
        "epochs": EPOCHS,
        "eval_step": EVAL_STEP,
        "stopping_step": STOPPING_STEP,
        "valid_metric": "Recall@10",
        "data_path": str(Path(contract["training"]["data_root"]) / DATASET),
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        },
        "metric_hidden_size": 64,
        "pnet_hidden_size": 64,
        "evolution_time": 1.0,
        "dynamics_chunk_size": 4096,
        "checkpoint_dynamics": False,
        "eval_item_chunk_size": 4096,
        **trial.parameters.recbole_values(),
    }
    mismatches = {
        key: {"expected": value, "actual": _config_value(config, key)}
        for key, value in expected.items()
        if not _same_value(_config_value(config, key), value)
    }
    if mismatches:
        raise ValueError(f"checkpoint trial contract mismatch: {mismatches}")


def _validate_raw_result(
    payload: Mapping[str, Any], path: Path, trial: Trial, contract: Mapping[str, Any]
) -> Path:
    if payload.get("model") != MODEL or payload.get("dataset") != DATASET:
        raise ValueError(f"result model/dataset mismatch: {path}")
    if int(payload.get("seed", -1)) != SEED:
        raise ValueError(f"result seed mismatch: {path}")
    if "test_result" not in payload or payload["test_result"] is not None:
        raise RuntimeError(f"tuning result touched or omitted held-out test state: {path}")
    try:
        score = float(payload["best_valid_score"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"result has no numeric validation score: {path}") from error
    if not math.isfinite(score) or not isinstance(payload.get("best_valid_result"), Mapping):
        raise ValueError(f"result has invalid validation selection: {path}")
    splits = payload.get("split_fingerprints")
    if not isinstance(splits, Mapping):
        raise ValueError(f"result has no split fingerprints: {path}")
    for name in ("train", "valid", "test"):
        item = splits.get(name)
        if (
            not isinstance(item, Mapping)
            or int(item.get("interactions", 0)) <= 0
            or not item.get("sha256")
        ):
            raise ValueError(f"invalid {name} split fingerprint: {path}")
    checkpoint_token = payload.get("checkpoint_file")
    if not isinstance(checkpoint_token, str) or not checkpoint_token:
        raise ValueError(f"result has no checkpoint: {path}")
    checkpoint = Path(checkpoint_token).expanduser().resolve()
    validate_checkpoint_contract(checkpoint, trial, contract)
    return checkpoint


def _expected_metadata(trial: Trial, contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL,
        "stage": trial.stage,
        "trial_name": trial.name,
        "parameters": asdict(trial.parameters),
        "campaign_contract": contract,
        "campaign_contract_sha256": _canonical_hash(contract),
        "validation_only": True,
        "test_evaluated": False,
    }


def annotate_or_load_result(
    path: Path, trial: Trial, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Load an exact result, or recover one written just before a crash.

    A raw child result can be annotated only after its checkpoint and split
    fingerprints pass the full contract.  Existing runner metadata is never
    repaired or overwritten when it differs.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result JSON is not an object: {path}")
    _validate_raw_result(payload, path, trial, contract)
    metadata = payload.get("agcf_movielens_runner")
    if metadata is None:
        payload["agcf_movielens_runner"] = {
            **_expected_metadata(trial, contract),
            "annotated_at": _utc_now(),
        }
        _atomic_json(path, payload)
    return load_complete_result(path, trial, contract)


def load_complete_result(
    path: Path, trial: Trial, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Read an already annotated result without changing any file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result JSON is not an object: {path}")
    _validate_raw_result(payload, path, trial, contract)
    expected = _expected_metadata(trial, contract)
    metadata = payload.get("agcf_movielens_runner")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"result has no exact runner metadata: {path}")
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"result resume contract mismatch: {mismatches}")
    return payload


def candidate_from_plain_result(
    path: Path,
    parameters: Parameters,
    contract: Mapping[str, Any],
    *,
    name: str = "imported-external-anchor",
) -> Candidate:
    """Import a validation-only ``run_recbole_gnn.py`` result read-only.

    This is intended for a manually started exact anchor that predates this
    runner.  No runner metadata is required or written; the checkpoint config,
    split fingerprints, validation result, and untouched test state are still
    verified against the complete campaign contract.
    """

    parameters.validate()
    trial = Trial("anchor", "external-contract-check", parameters)
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"external result JSON is not an object: {resolved}")
    checkpoint = _validate_raw_result(payload, resolved, trial, contract)
    return Candidate(
        name=name,
        source=str(resolved),
        checkpoint_file=str(checkpoint),
        parameters=parameters,
        best_valid_score=float(payload["best_valid_score"]),
        best_valid_result=dict(payload["best_valid_result"]),
    )


def candidate_from_result(
    path: Path, trial: Trial, contract: Mapping[str, Any]
) -> Candidate:
    payload = annotate_or_load_result(path, trial, contract)
    return Candidate(
        name=trial.name,
        source=str(path.resolve()),
        checkpoint_file=str(Path(payload["checkpoint_file"]).expanduser().resolve()),
        parameters=trial.parameters,
        best_valid_score=float(payload["best_valid_score"]),
        best_valid_result=dict(payload["best_valid_result"]),
    )


def load_continuation_candidate(
    path: Path,
    contract: Mapping[str, Any],
    *,
    required_stage: str,
) -> Candidate:
    summary_path = path.expanduser().resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or "best" not in payload:
        raise ValueError(f"continuation must be a completed stage summary: {path}")
    if payload.get("complete") is not True:
        raise ValueError(f"continuation stage is incomplete: {path}")
    if payload.get("stage") != required_stage:
        raise ValueError(
            f"expected {required_stage!r} continuation, got {payload.get('stage')!r}"
        )
    if payload.get("campaign_contract_sha256") != _canonical_hash(contract):
        raise ValueError(f"continuation campaign contract changed: {path}")
    best = payload.get("best")
    if not isinstance(best, Mapping) or not best.get("source"):
        raise ValueError(f"continuation has no validation winner: {path}")
    source = Path(str(best["source"])).expanduser().resolve()
    parameters = Parameters.from_mapping(best.get("parameters", {}))
    result = json.loads(source.read_text(encoding="utf-8"))
    metadata = result.get("agcf_movielens_runner") if isinstance(result, Mapping) else None
    if metadata is None:
        # A stage may retain a read-only, externally imported anchor as its
        # winner.  Recheck its raw result and checkpoint, then cross-check all
        # values persisted in the summary rather than mutating the source.
        candidate = candidate_from_plain_result(
            source,
            parameters,
            contract,
            name=str(best.get("name", "imported-external-anchor")),
        )
    else:
        if not isinstance(metadata, Mapping):
            raise ValueError(f"continuation result metadata is invalid: {source}")
        name = str(metadata.get("trial_name", ""))
        parts = name.split("__", 2)
        if len(parts) != 3:
            raise ValueError(f"invalid continuation trial name: {name!r}")
        trial = Trial(
            stage=str(metadata.get("stage")),
            label=parts[1],
            parameters=Parameters.from_mapping(metadata.get("parameters", {})),
        )
        if trial.name != name:
            raise ValueError(f"continuation name/parameter hash mismatch: {source}")
        candidate = candidate_from_result(source, trial, contract)
    persisted = {
        "name": candidate.name,
        "source": candidate.source,
        "checkpoint_file": candidate.checkpoint_file,
        "parameters": asdict(candidate.parameters),
        "best_valid_score": candidate.best_valid_score,
        "best_valid_result": dict(candidate.best_valid_result),
    }
    mismatches = {
        key: {"expected": persisted[key], "actual": best.get(key)}
        for key in persisted
        if best.get(key) != persisted[key]
    }
    if mismatches:
        raise ValueError(f"continuation summary winner mismatch: {mismatches}")
    return candidate


def write_summary(
    path: Path,
    *,
    stage: str,
    contract: Mapping[str, Any],
    candidates: Sequence[Candidate],
    planned_new_trial_count: int,
    completed_new_trial_count: int,
    inherited_anchor: Candidate | None,
    complete: bool,
) -> dict[str, Any]:
    ranked = sorted(candidates, key=lambda item: item.best_valid_score, reverse=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL,
        "dataset": DATASET,
        "stage": stage,
        "selection_metric": SELECTION_METRIC,
        "validation_only": True,
        "test_evaluated": False,
        "campaign_contract": contract,
        "campaign_contract_sha256": _canonical_hash(contract),
        "planned_new_trial_count": planned_new_trial_count,
        "completed_new_trial_count": completed_new_trial_count,
        "complete": complete,
        "inherited_anchor": inherited_anchor.json() if inherited_anchor else None,
        "best": ranked[0].json() if ranked else None,
        "ranking": [candidate.json() for candidate in ranked],
        "updated_at": _utc_now(),
    }
    _atomic_json(path, payload)
    return payload


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def refuse_live_orphan(output_root: Path) -> None:
    state_path = output_root / ".agcf_movielens_active.json"
    if not state_path.is_file():
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot verify active-trial state {state_path}") from error
    same_host = state.get("hostname") == socket.gethostname()
    pid = int(state.get("child_pid", -1))
    if same_host and _pid_is_alive(pid):
        raise RuntimeError(
            f"trial {state.get('trial_name')} is still running as PID {pid}; "
            "refusing a duplicate launch"
        )
    if not same_host:
        raise RuntimeError(
            f"active-trial state belongs to host {state.get('hostname')!r}; "
            "verify that host before removing the state file or launching here"
        )
    stale = output_root / (
        f".agcf_movielens_active.stale.{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    )
    os.replace(state_path, stale)


def run_and_tee(
    command: list[str],
    log_path: Path,
    cwd: Path,
    env: Mapping[str, str],
    *,
    active_state: Path,
    trial: Trial,
    contract: Mapping[str, Any],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND=" + shlex.join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _atomic_json(
            active_state,
            {
                "schema_version": SCHEMA_VERSION,
                "hostname": socket.gethostname(),
                "runner_pid": os.getpid(),
                "child_pid": process.pid,
                "trial_name": trial.name,
                "campaign_contract_sha256": _canonical_hash(contract),
                "command": command,
                "started_at": _utc_now(),
            },
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return_code = process.wait()
        except BaseException:
            # Do not leave an untracked GPU child when the foreground runner
            # is interrupted normally.  SIGKILL cannot execute this block, in
            # which case the persisted active-state PID protects the restart.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            if active_state.is_file():
                active_state.unlink()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


@contextlib.contextmanager
def single_runner_lock(output_root: Path):
    """Hold a non-blocking process lock on POSIX and Windows."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".agcf_movielens_single_gpu.lock"
    with lock_path.open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":  # pragma: no cover - exercised on Yanglab/Windows.
            import msvcrt

            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(
                    f"another runner owns {lock_path}; refusing GPU concurrency"
                ) from error
            unlock = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"another runner owns {lock_path}; refusing GPU concurrency"
                ) from error
            unlock = lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        try:
            yield
        finally:
            lock.seek(0)
            unlock()


def execute_stage(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    stage: str,
    anchor: Candidate | None,
    remaining_new_trials: list[int | None],
    env: Mapping[str, str],
) -> tuple[Candidate, bool]:
    if stage == "anchor":
        if anchor is not None:
            raise ValueError("anchor stage cannot inherit a candidate")
        stage_parameters = ANCHOR
    else:
        if anchor is None:
            raise ValueError(f"stage {stage!r} requires its predecessor winner")
        stage_parameters = anchor.parameters
    trials = build_stage_trials(stage, stage_parameters)
    candidates: list[Candidate] = [anchor] if anchor is not None else []
    complete_count = 0
    summary_path = stage_summary_path(args.output_root, stage)
    active_state = args.output_root / ".agcf_movielens_active.json"

    for index, trial in enumerate(trials, 1):
        paths = trial_paths(args.output_root, trial)
        if paths["result"].is_file():
            candidate = candidate_from_result(paths["result"], trial, contract)
            print(f"[{stage} {index}/{len(trials)}] resume {trial.name}")
        else:
            budget = remaining_new_trials[0]
            if budget is not None and budget <= 0:
                write_summary(
                    summary_path,
                    stage=stage,
                    contract=contract,
                    candidates=candidates,
                    planned_new_trial_count=len(trials),
                    completed_new_trial_count=complete_count,
                    inherited_anchor=anchor,
                    complete=False,
                )
                if not candidates:
                    raise RuntimeError("new-trial budget exhausted before anchor ran")
                return max(candidates, key=lambda item: item.best_valid_score), False
            paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
            command = trial_command(
                args, trial, paths["result"], paths["checkpoint_dir"]
            )
            print(f"[{stage} {index}/{len(trials)}] start {trial.name}")
            run_and_tee(
                command,
                paths["log"],
                args.repo,
                env,
                active_state=active_state,
                trial=trial,
                contract=contract,
            )
            candidate = candidate_from_result(paths["result"], trial, contract)
            if budget is not None:
                remaining_new_trials[0] = budget - 1
        candidates.append(candidate)
        complete_count += 1
        summary = write_summary(
            summary_path,
            stage=stage,
            contract=contract,
            candidates=candidates,
            planned_new_trial_count=len(trials),
            completed_new_trial_count=complete_count,
            inherited_anchor=anchor,
            complete=complete_count == len(trials),
        )
        print(
            f"current {stage} best={summary['best']['name']} "
            f"valid Recall@10={summary['best']['best_valid_score']:.6f}"
        )
    if not candidates:
        raise RuntimeError(f"stage {stage!r} produced no candidate")
    return max(candidates, key=lambda item: item.best_valid_score), True


def _predecessor(stage: str) -> str:
    index = STAGE_ORDER.index(stage)
    if index == 0:
        raise ValueError("anchor has no predecessor")
    return STAGE_ORDER[index - 1]


def resolve_stage_anchor(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    *,
    plan_only: bool,
) -> tuple[Candidate | None, str | None]:
    if args.stage in {"anchor", "all"}:
        return None, None
    predecessor = _predecessor(args.stage)
    continuation = (
        args.resume_from.expanduser().resolve()
        if args.resume_from is not None
        else stage_summary_path(args.output_root, predecessor)
    )
    if continuation.is_file():
        return (
            load_continuation_candidate(
                continuation, contract, required_stage=predecessor
            ),
            str(continuation),
        )
    if plan_only:
        return None, (
            f"nominal anchor used for display; run/finish {predecessor!r} first "
            f"or pass --resume-from"
        )
    raise FileNotFoundError(
        f"stage {args.stage!r} requires completed predecessor summary {continuation}"
    )


def resolve_external_anchor(
    args: argparse.Namespace, contract: Mapping[str, Any]
) -> Candidate | None:
    if args.continuation_result is None:
        return None
    return candidate_from_plain_result(
        args.continuation_result, ANCHOR, contract, name="imported-external-anchor"
    )


def dry_run_plan(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    initial_anchor: Candidate | None = None,
    continuation_note: str | None = None,
) -> dict[str, Any]:
    stages = (
        STAGE_ORDER[1:]
        if args.stage == "all" and initial_anchor is not None
        else (STAGE_ORDER if args.stage == "all" else (args.stage,))
    )
    nominal_anchor = initial_anchor.parameters if initial_anchor is not None else ANCHOR
    rendered = []
    for stage in stages:
        parameters = ANCHOR if stage == "anchor" else nominal_anchor
        trials = build_stage_trials(stage, parameters)
        jobs = []
        for trial in trials:
            paths = trial_paths(args.output_root, trial)
            status = "run"
            if paths["result"].is_file():
                # Plan-only mode still refuses stale/inexact results.
                load_complete_result(paths["result"], trial, contract)
                status = "resume"
            jobs.append(
                {
                    "name": trial.name,
                    "parameters": asdict(trial.parameters),
                    "status": status,
                    "result": str(paths["result"]),
                    "log": str(paths["log"]),
                    "checkpoint_dir": str(paths["checkpoint_dir"]),
                    "command": trial_command(
                        args, trial, paths["result"], paths["checkpoint_dir"]
                    ),
                }
            )
        rendered.append(
            {
                "stage": stage,
                "field": STAGE_DEFINITIONS.get(stage, {}).get("field"),
                "grid": (
                    list(STAGE_DEFINITIONS[stage]["values"])
                    if stage in STAGE_DEFINITIONS
                    else None
                ),
                "jobs": jobs,
                "depends_on_preceding_validation_winner": stage != "anchor",
                "note": (
                    "parameters are nominal until the preceding stage selects its "
                    "actual validation winner"
                    if args.stage == "all" and stage != "anchor"
                    else None
                ),
            }
        )
    return {
        "plan_only": True,
        "model": MODEL,
        "dataset": DATASET,
        "physical_gpu": args.gpu_id,
        "serial_workers": 1,
        "validation_only": True,
        "test_evaluated": False,
        "imported_initial_anchor": (
            initial_anchor.json() if initial_anchor is not None else None
        ),
        "continuation": continuation_note,
        "campaign_contract": contract,
        "campaign_contract_sha256": _canonical_hash(contract),
        "stages": rendered,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="anchor",
        choices=(*STAGE_ORDER, "all"),
        help="one blocked stage, or all stages in validation-winner order",
    )
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--data-path",
        type=Path,
        help="root containing AGCF_MovieLens/AGCF_MovieLens.inter",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--gpu-id",
        default="0",
        help="exactly one non-negative physical CUDA device index",
    )
    parser.add_argument(
        "--resume-from",
        "--continue-from",
        dest="resume_from",
        type=Path,
        help=(
            "completed predecessor summary; omitted means auto-discover it "
            "under --output-root"
        ),
    )
    parser.add_argument(
        "--continuation-result",
        type=Path,
        help=(
            "read-only import of an exact manually-run anchor result; valid "
            "with --stage batch-size or all and skips duplicate anchor training"
        ),
    )
    parser.add_argument(
        "--max-new-trials",
        type=int,
        help="run at most N new trials, then exit cleanly for exact continuation",
    )
    parser.add_argument(
        "--plan-only",
        "--dry-run",
        dest="plan_only",
        action="store_true",
        help="validate inputs and print the serial plan without training",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    gpu = str(args.gpu_id).strip()
    if not gpu.isdigit() or "," in gpu:
        raise ValueError("--gpu-id must name exactly one non-negative physical device")
    args.gpu_id = str(int(gpu))
    if args.max_new_trials is not None and args.max_new_trials < 0:
        raise ValueError("--max-new-trials must be non-negative")
    if args.stage in {"anchor", "all"} and args.resume_from is not None:
        raise ValueError(f"--stage {args.stage} manages continuation automatically")
    if args.resume_from is not None and args.continuation_result is not None:
        raise ValueError("use only one of --resume-from and --continuation-result")
    if args.continuation_result is not None and args.stage not in {
        "batch-size",
        "all",
    }:
        raise ValueError(
            "--continuation-result imports the anchor and requires "
            "--stage batch-size or all"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.repo = args.repo.expanduser().resolve()
    # Checkpoints pickle RecBole/RecBole-GNN Config classes.  When this file is
    # invoked by path, Python otherwise places only ``slrec_experiments`` on
    # sys.path and cannot unpickle those project-local modules.
    repo_token = str(args.repo)
    if repo_token not in sys.path:
        sys.path.insert(0, repo_token)
    args.data_path = (
        args.data_path.expanduser().resolve()
        if args.data_path is not None
        else (args.repo / "dataset").resolve()
    )
    args.output_root = args.output_root.expanduser().resolve()
    _validate_args(args)
    validate_model_registration(args.repo)
    source = audit_movielens_source(args.data_path)
    protocol = validate_protocol(args.repo, source_audit=source)
    contract = campaign_contract(protocol, args.data_path)
    external_anchor = resolve_external_anchor(args, contract)
    if external_anchor is not None:
        initial_anchor = external_anchor
        continuation_note = (
            f"read-only imported exact anchor: {external_anchor.source}"
        )
    else:
        initial_anchor, continuation_note = resolve_stage_anchor(
            args, contract, plan_only=args.plan_only
        )

    if args.plan_only:
        print(
            json.dumps(
                dry_run_plan(args, contract, initial_anchor, continuation_note),
                indent=2,
            )
        )
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    remaining: list[int | None] = [args.max_new_trials]
    with single_runner_lock(args.output_root):
        refuse_live_orphan(args.output_root)
        if args.stage == "all":
            anchor = initial_anchor
            stages = STAGE_ORDER[1:] if anchor is not None else STAGE_ORDER
            for stage in stages:
                anchor, complete = execute_stage(
                    args, contract, stage, anchor, remaining, env
                )
                if not complete:
                    print(
                        "New-trial budget reached; rerun the exact command to "
                        "continue. The held-out test was not evaluated."
                    )
                    return 0
        else:
            _, complete = execute_stage(
                args, contract, args.stage, initial_anchor, remaining, env
            )
            if not complete:
                print("New-trial budget reached; rerun this command to continue.")

    print("Validation search complete; the held-out test was not evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
