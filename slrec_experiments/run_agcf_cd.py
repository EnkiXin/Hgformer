#!/usr/bin/env python3
"""Validation-only, resumable, single-GPU Amazon-CD AGCF experiments.

This runner has intentionally narrow scope:

* ``RecFormer_cd.yaml`` remains the sole authority for the rating/5-core
  filters, seed, per-user 8:1:1 split, metrics, and full-ranking evaluator;
* every child process contains ``--validation-only`` and saves its
  validation-selected checkpoint, result JSON, and stdout/stderr log;
* exactly one child process runs at a time on one physical CUDA device;
* completed trials resume only after their runner metadata, split
  fingerprints, and checkpoint configuration match the exact contract; and
* tuning is a small staged search over values omitted by the AGCF paper, not a
  large Cartesian product.

There is deliberately no held-out-test mode in this file.  Test evaluation is
an explicit later action for one frozen validation winner.
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
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SCHEMA_VERSION = 1
DATASET = "Amazon_cd"
SEED = 2024
SELECTION_METRIC = "Recall@10 on full-ranking validation"
BASE_CONFIG = "RecFormer_cd.yaml"
AGCF_CONFIG = "AGCF_cd.yaml"
SL8_CONFIG = "AGCFSL8Coord_cd.yaml"

AMAZON_CD_SOURCE = {
    "relative_file": "Amazon_cd/Amazon_cd.inter",
    "bytes": 152_336_079,
    "lines_including_header": 3_749_005,
    "interactions_excluding_header": 3_749_004,
    "sha256": "7061471c288df93ba65bfede355aeb013e10dbdfc249db8f20a02bbf8ae031c4",
}
AMAZON_CD_FILTERED = {
    # RecBole cardinalities include the reserved zero id for users/items.
    "framework_users": 66_317,
    "framework_items": 58_869,
    "interactions": 952_547,
    "token_users": 66_316,
    "token_items": 58_868,
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
}


@dataclass(frozen=True)
class FamilySpec:
    slug: str
    model: str
    config_names: tuple[str, ...]
    embedding_size: int
    is_sl8_chart: bool = False


FAMILIES = {
    "agcf": FamilySpec(
        slug="agcf",
        model="AGCF",
        config_names=(BASE_CONFIG, AGCF_CONFIG),
        embedding_size=64,
    ),
    "agcf-sl8coord": FamilySpec(
        slug="agcf-sl8coord",
        model="AGCFSL8Coord",
        config_names=(BASE_CONFIG, AGCF_CONFIG, SL8_CONFIG),
        embedding_size=63,
        is_sl8_chart=True,
    ),
}
FAMILY_ALIASES = {
    "agcf": "agcf",
    "sl8": "agcf-sl8coord",
    "sl8coord": "agcf-sl8coord",
    "agcfsl8coord": "agcf-sl8coord",
    "agcf-sl8coord": "agcf-sl8coord",
}

COMMON_STAGE_ORDER = ("pilot", "dynamics", "metric", "forces", "optimizer")
SL8_STAGE_ORDER = ("pilot", "dynamics", "metric", "sl8-chart", "forces", "optimizer")


@dataclass(frozen=True)
class Parameters:
    """Complete tunable state for one trial.

    The fields are limited to AGCF values that the WWW 2026 paper leaves
    undisclosed (plus ``coord_clip`` for our explicitly non-paper SL chart
    extension).  MLP widths and resource chunks stay fixed in the overlay for
    this first pilot so search size remains useful and auditable.
    """

    metric_rank: int
    channel_rank: int
    metric_epsilon: float
    structural_delta: float
    output_steps: int
    integration_steps: int
    potential_strength: float
    damping: float
    margin: float
    learning_rate: float
    weight_decay: float
    train_batch_size: int
    coord_clip: float | None = None

    @classmethod
    def from_config(cls, merged: Mapping[str, Any], family: FamilySpec) -> "Parameters":
        margin_key = "loss_margin" if family.is_sl8_chart else "margin"
        parameters = cls(
            metric_rank=int(merged["metric_rank"]),
            channel_rank=int(merged["channel_rank"]),
            metric_epsilon=float(merged["metric_epsilon"]),
            structural_delta=float(merged["structural_delta"]),
            output_steps=int(merged["output_steps"]),
            integration_steps=int(merged["integration_steps"]),
            potential_strength=float(merged["potential_strength"]),
            damping=float(merged["damping"]),
            margin=float(merged[margin_key]),
            learning_rate=float(merged["learning_rate"]),
            weight_decay=float(merged["weight_decay"]),
            train_batch_size=int(merged["train_batch_size"]),
            coord_clip=(
                float(merged["coord_clip"]) if family.is_sl8_chart else None
            ),
        )
        parameters.validate(family)
        return parameters

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Parameters":
        values = dict(payload)
        if values.get("coord_clip") is not None:
            values["coord_clip"] = float(values["coord_clip"])
        return cls(
            metric_rank=int(values["metric_rank"]),
            channel_rank=int(values["channel_rank"]),
            metric_epsilon=float(values["metric_epsilon"]),
            structural_delta=float(values["structural_delta"]),
            output_steps=int(values["output_steps"]),
            integration_steps=int(values["integration_steps"]),
            potential_strength=float(values["potential_strength"]),
            damping=float(values["damping"]),
            margin=float(values["margin"]),
            learning_rate=float(values["learning_rate"]),
            weight_decay=float(values["weight_decay"]),
            train_batch_size=int(values["train_batch_size"]),
            coord_clip=values.get("coord_clip"),
        )

    def validate(self, family: FamilySpec) -> None:
        for name in (
            "metric_rank",
            "channel_rank",
            "output_steps",
            "integration_steps",
            "train_batch_size",
        ):
            value = int(getattr(self, name))
            if name == "output_steps":
                if value < 0:
                    raise ValueError("output_steps must be non-negative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.metric_rank > family.embedding_size:
            raise ValueError("metric_rank cannot exceed the position dimension")
        if self.channel_rank > family.embedding_size:
            raise ValueError("channel_rank cannot exceed the position dimension")
        for name in (
            "metric_epsilon",
            "structural_delta",
            "potential_strength",
            "damping",
            "margin",
            "learning_rate",
            "weight_decay",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.metric_epsilon == 0 or self.learning_rate == 0:
            raise ValueError("metric_epsilon and learning_rate must be positive")
        if family.is_sl8_chart:
            if self.coord_clip is None or not math.isfinite(float(self.coord_clip)):
                raise ValueError("SL8Coord requires a finite coord_clip (<=0 disables it)")
        elif self.coord_clip is not None:
            raise ValueError("coord_clip is only valid for AGCF-SL8Coord")

    def recbole_values(self, family: FamilySpec) -> dict[str, Any]:
        values: dict[str, Any] = {
            "metric_rank": self.metric_rank,
            "channel_rank": self.channel_rank,
            "metric_epsilon": self.metric_epsilon,
            "structural_delta": self.structural_delta,
            "output_steps": self.output_steps,
            "integration_steps": self.integration_steps,
            "potential_strength": self.potential_strength,
            "damping": self.damping,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "train_batch_size": self.train_batch_size,
        }
        if family.is_sl8_chart:
            values["loss_margin"] = self.margin
            values["coord_clip"] = self.coord_clip
        else:
            values["margin"] = self.margin
        return values


@dataclass(frozen=True)
class Trial:
    stage: str
    label: str
    parameters: Parameters

    @property
    def name(self) -> str:
        payload = json.dumps(
            asdict(self.parameters), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        suffix = hashlib.sha256(payload).hexdigest()[:10]
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
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_amazon_cd_source(data_root: Path) -> dict[str, Any]:
    """Pin the exact raw McAuley Amazon-CD file before planning or training."""

    resolved_root = data_root.expanduser().resolve()
    path = resolved_root / AMAZON_CD_SOURCE["relative_file"]
    if not path.is_file():
        raise FileNotFoundError(
            f"missing pinned Amazon-CD source: {path}; pass --data-path to the "
            "directory containing Amazon_cd/Amazon_cd.inter"
        )
    actual_bytes = path.stat().st_size
    digest = hashlib.sha256()
    newline_count = 0
    final_byte = b""
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
            newline_count += block.count(b"\n")
            final_byte = block[-1:]
    actual_lines = newline_count + (
        1 if final_byte and final_byte != b"\n" else 0
    )
    actual_sha256 = digest.hexdigest()
    expected = {
        "bytes": AMAZON_CD_SOURCE["bytes"],
        "lines_including_header": AMAZON_CD_SOURCE["lines_including_header"],
        "sha256": AMAZON_CD_SOURCE["sha256"],
    }
    actual = {
        "bytes": actual_bytes,
        "lines_including_header": actual_lines,
        "sha256": actual_sha256,
    }
    differences = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value
    }
    if differences:
        raise ValueError(
            f"Amazon-CD raw source does not match the pinned release: {differences}"
        )
    return {
        "data_root": str(resolved_root),
        "file": str(path),
        **actual,
        "interactions_excluding_header": AMAZON_CD_SOURCE[
            "interactions_excluding_header"
        ],
        "verified": True,
    }


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


def family_from_token(token: str) -> FamilySpec:
    try:
        return FAMILIES[FAMILY_ALIASES[token.strip().lower()]]
    except KeyError as error:
        raise ValueError(
            f"unknown family {token!r}; choices={sorted(FAMILIES)}"
        ) from error


def stage_order(family: FamilySpec) -> tuple[str, ...]:
    return SL8_STAGE_ORDER if family.is_sl8_chart else COMMON_STAGE_ORDER


def config_paths(repo: Path, family: FamilySpec) -> tuple[Path, ...]:
    root = repo / "baseline_config_fixed"
    return tuple(root / name for name in family.config_names)


def validate_model_registration(repo: Path, family: FamilySpec) -> Path:
    """Fail early if the expected clean-room class is not import-discoverable.

    AST inspection avoids importing torch/RecBole just to print a dry-run
    plan, while still catching a class/module naming mismatch before a long
    campaign is scheduled.
    """

    module = (
        repo
        / "recbole_gnn"
        / "model"
        / "general_recommender"
        / f"{family.model.lower()}.py"
    )
    if not module.is_file():
        raise RuntimeError(
            f"{family.model} is not available: expected module {module}. "
            "Install/finish the model before starting this family."
        )
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    if family.model not in classes:
        raise RuntimeError(
            f"module {module} does not define the required class {family.model}"
        )
    return module


def validate_protocol(
    repo: Path,
    family: FamilySpec,
    *,
    source_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = config_paths(repo, family)
    documents = [_yaml_mapping(path) for path in paths]
    base = documents[0]
    overlays = documents[1:]

    for path, overlay in zip(paths[1:], overlays):
        overlap = PROTECTED_PROTOCOL_KEYS.intersection(overlay)
        if overlap:
            raise RuntimeError(
                f"model overlay {path.name} overrides the CD protocol: "
                f"{sorted(overlap)}"
            )

    expected_base = {
        "dataset": DATASET,
        "seed": SEED,
        "reproducibility": True,
        "learner": "adam",
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": "rating",
        "val_interval": {"rating": "[3,inf)"},
        "user_inter_num_interval": "[5,inf)",
        "item_inter_num_interval": "[5,inf)",
        "metrics": ["Recall", "NDCG"],
        "topk": [5, 10, 20, 50],
        "valid_metric": "Recall@10",
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        },
    }
    differences = {
        key: {"expected": value, "actual": base.get(key)}
        for key, value in expected_base.items()
        if base.get(key) != value
    }
    if differences:
        raise RuntimeError(f"RecFormer Amazon-CD protocol changed: {differences}")

    merged: dict[str, Any] = {}
    for document in documents:
        merged.update(document)
    expected_model = {
        "model": family.model,
        "embedding_size": family.embedding_size,
        "evolution_time": 1.0,
    }
    if family.is_sl8_chart:
        expected_model.update(
            {
                "matrix_dim": 8,
                "pairwise_loss": "hinge",
                "schatten_p": 2,
                "symmetric_distance": False,
            }
        )
    model_differences = {
        key: {"expected": value, "actual": merged.get(key)}
        for key, value in expected_model.items()
        if merged.get(key) != value
    }
    if model_differences:
        raise RuntimeError(f"{family.model} overlay contract changed: {model_differences}")

    canonical_agcf_keys = {
        "metric_rank",
        "metric_hidden_size",
        "pnet_hidden_size",
        "channel_rank",
        "metric_epsilon",
        "structural_delta",
        "output_steps",
        "integration_steps",
        "evolution_time",
        "potential_strength",
        "damping",
        "margin",
        "dynamics_chunk_size",
        "checkpoint_dynamics",
    }
    missing = canonical_agcf_keys.difference(merged)
    if missing:
        raise RuntimeError(f"AGCF overlay lacks canonical model keys: {sorted(missing)}")

    parameters = Parameters.from_config(merged, family)
    return {
        "dataset": DATASET,
        "seed": SEED,
        "family": family.slug,
        "model": family.model,
        "config_files": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in paths
        ],
        "filters": {"rating": "[3,inf)", "users": "[5,inf)", "items": "[5,inf)"},
        "split": expected_base["eval_args"],
        "evaluation": {
            "metrics": expected_base["metrics"],
            "topk": expected_base["topk"],
            "selection_metric": expected_base["valid_metric"],
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
                    (repo / "dataset" / AMAZON_CD_SOURCE["relative_file"]).resolve()
                ),
                "relative_file": AMAZON_CD_SOURCE["relative_file"],
                "bytes": AMAZON_CD_SOURCE["bytes"],
                "lines_including_header": AMAZON_CD_SOURCE[
                    "lines_including_header"
                ],
                "interactions_excluding_header": AMAZON_CD_SOURCE[
                    "interactions_excluding_header"
                ],
                "sha256": AMAZON_CD_SOURCE["sha256"],
                "verified": False,
            }
        ),
        "expected_filtered_dataset": dict(AMAZON_CD_FILTERED),
        "base_parameters": asdict(parameters),
    }


def campaign_contract(
    protocol: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "training": {
            "epochs": int(args.epochs),
            "eval_step": int(args.eval_step),
            "stopping_step": int(args.stopping_step),
            "negative_sampling": {"uniform": 1},
            "serial_workers": 1,
            "logical_gpu": 0,
        },
        "selection": {
            "metric": SELECTION_METRIC,
            "validation_only": True,
            "test_evaluated": False,
        },
    }


def _unique_variants(
    stage: str,
    anchor: Parameters,
    family: FamilySpec,
    variants: Iterable[tuple[str, Mapping[str, Any]]],
) -> tuple[Trial, ...]:
    trials: list[Trial] = []
    seen = {anchor}
    for label, updates in variants:
        parameters = replace(anchor, **updates)
        parameters.validate(family)
        if parameters in seen:
            continue
        seen.add(parameters)
        trials.append(Trial(stage, label, parameters))
    return tuple(trials)


def build_stage_trials(
    stage: str, anchor: Parameters, family: FamilySpec
) -> tuple[Trial, ...]:
    """Return a blocked, one/few-factor search rather than a Cartesian grid."""

    if stage == "pilot":
        return (Trial("pilot", "paper-guided-anchor", anchor),)
    if stage == "dynamics":
        variants = (
            ("L1-K2", {"output_steps": 1, "integration_steps": 2}),
            ("L2-K1", {"output_steps": 2, "integration_steps": 1}),
            ("L2-K2", {"output_steps": 2, "integration_steps": 2}),
        )
    elif stage == "metric":
        variants = (
            ("rank2", {"metric_rank": 2, "channel_rank": 2}),
            ("rank8", {"metric_rank": 8, "channel_rank": 8}),
            ("epsilon1e-2", {"metric_epsilon": 1e-2}),
        )
    elif stage == "sl8-chart":
        if not family.is_sl8_chart:
            raise ValueError("sl8-chart stage is available only for AGCF-SL8Coord")
        # The 1.0 anchor is evidence-based from the tiny E2E saturation audit.
        # This small ablation checks the old 0.75 choice and disabling the clip.
        variants = (
            ("clip0p75", {"coord_clip": 0.75}),
            ("clip-disabled", {"coord_clip": 0.0}),
        )
    elif stage == "forces":
        # Deliberately one-factor perturbations.  alpha/gamma/m/delta winners
        # are not published for Amazon-CD; no factorial interaction claim is
        # made in this first pilot.
        variants = (
            ("alpha0p05", {"potential_strength": 0.05}),
            ("alpha0p2", {"potential_strength": 0.2}),
            ("gamma0p05", {"damping": 0.05}),
            ("margin0p2", {"margin": 0.2}),
            ("delta0p01", {"structural_delta": 0.01}),
        )
    elif stage == "optimizer":
        variants = (
            ("lr3em4", {"learning_rate": 3e-4}),
            ("lr1em3", {"learning_rate": 1e-3}),
            ("wd1em6", {"weight_decay": 1e-6}),
        )
    else:
        raise ValueError(f"unknown stage {stage!r}; choices={stage_order(family)}")
    return _unique_variants(stage, anchor, family, variants)


def trial_paths(output_root: Path, family: FamilySpec, trial: Trial) -> dict[str, Path]:
    root = output_root / family.slug / trial.stage
    return {
        "root": root,
        "result": root / "results" / f"{trial.name}.json",
        "log": root / "logs" / f"{trial.name}.log",
        "checkpoint_dir": root / "checkpoints" / trial.name,
        "summary": root / "summary.json",
    }


def _recbole_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def trial_command(
    args: argparse.Namespace,
    family: FamilySpec,
    trial: Trial,
    result_path: Path,
    checkpoint_dir: Path,
) -> list[str]:
    configs = " ".join(str(path) for path in config_paths(args.repo, family))
    command = [
        args.python,
        "-u",
        str(args.repo / "run_recbole_gnn.py"),
        "--model",
        family.model,
        "--dataset",
        DATASET,
        "--config-files",
        configs,
        "--validation-only",
        "--result-file",
        str(result_path),
        f"--checkpoint_dir={checkpoint_dir}",
        # Vendored RecBole rewrites CUDA_VISIBLE_DEVICES from gpu_id during
        # Config startup, so pass the same physical id used by the parent.
        f"--gpu_id={args.gpu_id}",
        "--use_gpu=True",
        "--show_progress=False",
        f"--epochs={args.epochs}",
        f"--eval_step={args.eval_step}",
        f"--stopping_step={args.stopping_step}",
        f"--data_path={args.data_path}",
        f"--seed={SEED}",
        "--reg_weight=0",
        "--neg_sampling={'uniform': 1}",
    ]
    for key, value in trial.parameters.recbole_values(family).items():
        command.append(f"--{key}={_recbole_scalar(value)}")
    return command


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    try:
        return config[key]
    except (KeyError, TypeError, AttributeError):
        dictionary = getattr(config, "final_config_dict", None)
        if isinstance(dictionary, Mapping):
            return dictionary.get(key)
        return None


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
    except TypeError:  # Older torch without weights_only.
        checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, Mapping) or "config" not in checkpoint:
        raise ValueError(f"not a RecBole checkpoint: {path}")
    return checkpoint["config"]


def validate_checkpoint_contract(
    checkpoint: Path,
    family: FamilySpec,
    trial: Trial,
    contract: Mapping[str, Any],
) -> None:
    config = _load_checkpoint_config(checkpoint)
    training = contract["training"]
    expected: dict[str, Any] = {
        "model": family.model,
        "dataset": DATASET,
        "seed": SEED,
        "embedding_size": family.embedding_size,
        "epochs": training["epochs"],
        "eval_step": training["eval_step"],
        "stopping_step": training["stopping_step"],
        "valid_metric": "Recall@10",
        # Vendored RecBole appends the dataset name to the CLI data root when
        # materialising its final checkpoint Config.
        "data_path": str(
            Path(contract["protocol"]["raw_source"]["data_root"]) / DATASET
        ),
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        },
        **trial.parameters.recbole_values(family),
    }
    if family.is_sl8_chart:
        expected.update(
            {
                "matrix_dim": 8,
                "num_factors": 1,
                "schatten_p": 2,
                "symmetric_distance": False,
                "pairwise_loss": "hinge",
            }
        )
    mismatches = {
        key: {"expected": value, "actual": _config_value(config, key)}
        for key, value in expected.items()
        if not _same_value(_config_value(config, key), value)
    }
    if mismatches:
        raise ValueError(f"checkpoint trial contract mismatch: {mismatches}")


def _require_split_fingerprints(payload: Mapping[str, Any]) -> None:
    splits = payload.get("split_fingerprints")
    if not isinstance(splits, Mapping):
        raise ValueError("result has no split fingerprints")
    for name in ("train", "valid", "test"):
        item = splits.get(name)
        if not isinstance(item, Mapping):
            raise ValueError(f"result lacks {name} split fingerprint")
        if int(item.get("interactions", 0)) <= 0 or not item.get("sha256"):
            raise ValueError(f"invalid {name} split fingerprint")


def load_complete_result(
    path: Path,
    family: FamilySpec,
    trial: Trial,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result JSON is not an object: {path}")
    if payload.get("model") != family.model or payload.get("dataset") != DATASET:
        raise ValueError(f"result model/dataset mismatch: {path}")
    if int(payload.get("seed", -1)) != SEED:
        raise ValueError(f"result seed mismatch: {path}")
    if "test_result" not in payload or payload["test_result"] is not None:
        raise RuntimeError(f"tuning result touched or omitted held-out test state: {path}")
    if payload.get("best_valid_score") is None or not isinstance(
        payload.get("best_valid_result"), Mapping
    ):
        raise ValueError(f"result has no validation selection: {path}")
    _require_split_fingerprints(payload)

    metadata = payload.get("agcf_runner")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"result has no AGCF runner metadata: {path}")
    expected_hash = _canonical_hash(contract)
    expected_metadata = {
        "schema_version": SCHEMA_VERSION,
        "family": family.slug,
        "stage": trial.stage,
        "trial_name": trial.name,
        "parameters": asdict(trial.parameters),
        "campaign_contract": contract,
        "campaign_contract_sha256": expected_hash,
        "test_evaluated": False,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"result resume contract mismatch: {mismatches}")

    checkpoint_token = payload.get("checkpoint_file")
    if not isinstance(checkpoint_token, str) or not checkpoint_token:
        raise ValueError(f"result has no checkpoint: {path}")
    checkpoint = Path(checkpoint_token).expanduser().resolve()
    validate_checkpoint_contract(checkpoint, family, trial, contract)
    return payload


def annotate_result(
    path: Path,
    family: FamilySpec,
    trial: Trial,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result JSON is not an object: {path}")
    if payload.get("test_result") is not None:
        raise RuntimeError(f"child unexpectedly evaluated held-out test: {path}")
    checkpoint_token = payload.get("checkpoint_file")
    if not isinstance(checkpoint_token, str) or not checkpoint_token:
        raise ValueError(f"validation run saved no checkpoint: {path}")
    validate_checkpoint_contract(
        Path(checkpoint_token).expanduser().resolve(), family, trial, contract
    )
    payload["agcf_runner"] = {
        "schema_version": SCHEMA_VERSION,
        "family": family.slug,
        "stage": trial.stage,
        "trial_name": trial.name,
        "parameters": asdict(trial.parameters),
        "campaign_contract": contract,
        "campaign_contract_sha256": _canonical_hash(contract),
        "test_evaluated": False,
        "checkpoint_contract_verified": True,
        "annotated_at": _utc_now(),
    }
    _atomic_json(path, payload)
    return load_complete_result(path, family, trial, contract)


def candidate_from_result(
    path: Path,
    family: FamilySpec,
    trial: Trial,
    contract: Mapping[str, Any],
) -> Candidate:
    payload = load_complete_result(path, family, trial, contract)
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
    family: FamilySpec,
    contract: Mapping[str, Any],
    *,
    required_stage: str | None = None,
) -> Candidate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"continuation JSON is not an object: {path}")
    from_summary = "best" in payload
    if from_summary:
        if payload.get("complete") is not True:
            raise ValueError(f"continuation stage is incomplete: {path}")
        if payload.get("campaign_contract_sha256") != _canonical_hash(contract):
            raise ValueError(f"continuation campaign contract changed: {path}")
        if payload.get("family") != family.slug:
            raise ValueError(f"continuation family changed: {path}")
        if required_stage is not None and payload.get("stage") != required_stage:
            raise ValueError(
                f"stage requires completed {required_stage!r} summary; got "
                f"{payload.get('stage')!r}"
            )
        best = payload.get("best")
        if not isinstance(best, Mapping) or not best.get("source"):
            raise ValueError(f"continuation summary has no selected result: {path}")
        source = Path(str(best["source"])).expanduser().resolve()
    else:
        source = path.expanduser().resolve()

    result = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(result, Mapping):
        raise ValueError(f"continuation result is not an object: {source}")
    metadata = result.get("agcf_runner")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"continuation result lacks runner metadata: {source}")
    persisted_name = str(metadata["trial_name"])
    name_parts = persisted_name.split("__", 2)
    if len(name_parts) != 3:
        raise ValueError(f"invalid persisted trial name: {persisted_name!r}")
    trial = Trial(
        stage=str(metadata["stage"]),
        label=name_parts[1],
        parameters=Parameters.from_mapping(metadata["parameters"]),
    )
    if trial.name != persisted_name:
        raise ValueError(f"continuation trial name/parameters mismatch: {source}")
    # A completed stage may legitimately select its inherited anchor.  In
    # that case the summary proves the predecessor stage even though the
    # selected source result was produced earlier.  A bare result continuation
    # has no such stage-level proof and must itself come from the predecessor.
    if required_stage is not None and not from_summary and trial.stage != required_stage:
        raise ValueError(
            f"stage requires {required_stage!r} continuation; got {trial.stage!r}"
        )
    return candidate_from_result(source, family, trial, contract)


def write_summary(
    path: Path,
    *,
    family: FamilySpec,
    stage: str,
    contract: Mapping[str, Any],
    candidates: Sequence[Candidate],
    planned_trial_count: int,
    completed_trial_count: int,
    complete: bool,
    inherited_anchor: Candidate | None,
) -> dict[str, Any]:
    ranked = sorted(candidates, key=lambda item: item.best_valid_score, reverse=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "family": family.slug,
        "model": family.model,
        "dataset": DATASET,
        "stage": stage,
        "selection_metric": SELECTION_METRIC,
        "validation_only": True,
        "test_evaluated": False,
        "campaign_contract": contract,
        "campaign_contract_sha256": _canonical_hash(contract),
        "planned_new_trial_count": planned_trial_count,
        "completed_new_trial_count": completed_trial_count,
        "complete": complete,
        "inherited_anchor": inherited_anchor.json() if inherited_anchor else None,
        "best": ranked[0].json() if ranked else None,
        "ranking": [candidate.json() for candidate in ranked],
        "updated_at": _utc_now(),
    }
    _atomic_json(path, payload)
    return payload


def run_and_tee(
    command: list[str], log_path: Path, cwd: Path, env: Mapping[str, str]
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
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


@contextlib.contextmanager
def single_runner_lock(output_root: Path):
    """Hold a non-blocking process lock on POSIX and Windows."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".agcf_single_gpu.lock"
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
                    f"another AGCF runner owns {lock_path}; refusing GPU concurrency"
                ) from error
            unlock = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"another AGCF runner owns {lock_path}; refusing GPU concurrency"
                ) from error
            unlock = lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.truncate()
        lock.write(f"pid={os.getpid()} acquired={_utc_now()}\n".encode("utf-8"))
        lock.flush()
        try:
            yield
        finally:
            lock.seek(0)
            unlock()


def execute_stage(
    args: argparse.Namespace,
    family: FamilySpec,
    contract: Mapping[str, Any],
    anchor: Candidate | None,
    stage: str,
    env: Mapping[str, str],
    remaining_new_trials: list[int | None],
) -> tuple[Candidate, bool]:
    base = Parameters.from_mapping(contract["protocol"]["base_parameters"])
    stage_anchor = anchor.parameters if anchor is not None else base
    trials = build_stage_trials(stage, stage_anchor, family)
    if stage == "pilot" and anchor is not None:
        raise ValueError("pilot stage cannot inherit an anchor")
    if stage != "pilot" and anchor is None:
        raise ValueError(f"stage {stage!r} requires a prior validation winner")

    candidates: list[Candidate] = [anchor] if anchor is not None else []
    complete_count = 0
    summary_path = args.output_root / family.slug / stage / "summary.json"
    for index, trial in enumerate(trials, 1):
        paths = trial_paths(args.output_root, family, trial)
        if paths["result"].is_file():
            candidate = candidate_from_result(
                paths["result"], family, trial, contract
            )
            print(f"[{stage} {index}/{len(trials)}] resume {trial.name}")
        else:
            budget = remaining_new_trials[0]
            if budget is not None and budget <= 0:
                write_summary(
                    summary_path,
                    family=family,
                    stage=stage,
                    contract=contract,
                    candidates=candidates,
                    planned_trial_count=len(trials),
                    completed_trial_count=complete_count,
                    complete=False,
                    inherited_anchor=anchor,
                )
                if not candidates:
                    raise RuntimeError("new-trial budget exhausted before pilot ran")
                return max(candidates, key=lambda item: item.best_valid_score), False

            paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
            command = trial_command(
                args, family, trial, paths["result"], paths["checkpoint_dir"]
            )
            print(f"[{stage} {index}/{len(trials)}] start {trial.name}")
            run_and_tee(command, paths["log"], args.repo, env)
            annotate_result(paths["result"], family, trial, contract)
            candidate = candidate_from_result(
                paths["result"], family, trial, contract
            )
            if budget is not None:
                remaining_new_trials[0] = budget - 1
        candidates.append(candidate)
        complete_count += 1
        summary = write_summary(
            summary_path,
            family=family,
            stage=stage,
            contract=contract,
            candidates=candidates,
            planned_trial_count=len(trials),
            completed_trial_count=complete_count,
            complete=complete_count == len(trials),
            inherited_anchor=anchor,
        )
        print(
            f"current {stage} best={summary['best']['name']} "
            f"valid Recall@10={summary['best']['best_valid_score']:.6f}"
        )
    return max(candidates, key=lambda item: item.best_valid_score), True


def dry_run_plan(
    args: argparse.Namespace,
    family: FamilySpec,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    base = Parameters.from_mapping(contract["protocol"]["base_parameters"])
    selected_stages = stage_order(family) if args.stage == "all" else (args.stage,)
    anchor = base
    stages = []
    for stage in selected_stages:
        trials = build_stage_trials(stage, anchor, family)
        jobs = []
        for trial in trials:
            paths = trial_paths(args.output_root, family, trial)
            jobs.append(
                {
                    "name": trial.name,
                    "parameters": asdict(trial.parameters),
                    "result": str(paths["result"]),
                    "log": str(paths["log"]),
                    "checkpoint_dir": str(paths["checkpoint_dir"]),
                    "status": "resume" if paths["result"].is_file() else "run",
                    "command": trial_command(
                        args,
                        family,
                        trial,
                        paths["result"],
                        paths["checkpoint_dir"],
                    ),
                }
            )
        stages.append(
            {
                "stage": stage,
                "jobs": jobs,
                "note": (
                    "nominal anchor only; actual --stage all variants are built "
                    "from the preceding validation winner"
                    if args.stage == "all" and stage != "pilot"
                    else None
                ),
            }
        )
    return {
        "dry_run": True,
        "family": family.slug,
        "model": family.model,
        "physical_gpu": args.gpu_id,
        "serial_workers": 1,
        "validation_only": True,
        "test_evaluated": False,
        "campaign_contract": contract,
        "campaign_contract_sha256": _canonical_hash(contract),
        "stages": stages,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        default="agcf",
        help="agcf or agcf-sl8coord (aliases sl8/sl8coord accepted)",
    )
    parser.add_argument(
        "--stage",
        default="pilot",
        help="pilot, dynamics, metric, sl8-chart, forces, optimizer, or all",
    )
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--data-path",
        type=Path,
        help=(
            "dataset root containing Amazon_cd/Amazon_cd.inter; defaults to "
            "<repo>/dataset"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--eval-step",
        type=int,
        default=10,
        help="full-ranking validation cadence; user-requested default is 10 epochs",
    )
    parser.add_argument("--stopping-step", type=int, default=30)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="completed preceding-stage summary/result for a non-pilot stage",
    )
    parser.add_argument(
        "--max-new-trials",
        type=int,
        help="stop cleanly after this many newly trained trials; completed trials resume",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and print the serial plan only"
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace, family: FamilySpec) -> None:
    if args.epochs <= 0 or args.eval_step <= 0:
        raise ValueError("--epochs and --eval-step must be positive")
    if args.stopping_step < 0:
        raise ValueError("--stopping-step must be non-negative")
    if args.max_new_trials is not None and args.max_new_trials < 0:
        raise ValueError("--max-new-trials must be non-negative")
    gpu = str(args.gpu_id).strip()
    if not gpu.isdigit() or "," in gpu:
        raise ValueError("--gpu-id must name exactly one non-negative physical CUDA device")
    args.gpu_id = str(int(gpu))
    allowed_stages = {*stage_order(family), "all"}
    if args.stage not in allowed_stages:
        raise ValueError(
            f"stage {args.stage!r} is invalid for {family.slug}; "
            f"choices={sorted(allowed_stages)}"
        )
    if args.stage == "pilot" and args.resume_from is not None:
        raise ValueError("pilot resumes automatically from its own result; omit --resume-from")
    if args.stage == "all" and args.resume_from is not None:
        raise ValueError("--stage all manages stage continuation; omit --resume-from")
    if (
        args.stage not in {"pilot", "all"}
        and args.resume_from is None
        and not args.dry_run
    ):
        raise ValueError("a non-pilot stage requires --resume-from")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.repo = args.repo.expanduser().resolve()
    args.data_path = (
        args.data_path.expanduser().resolve()
        if args.data_path is not None
        else (args.repo / "dataset").resolve()
    )
    args.output_root = args.output_root.expanduser().resolve()
    family = family_from_token(args.family)
    _validate_args(args, family)
    validate_model_registration(args.repo, family)
    source_audit = audit_amazon_cd_source(args.data_path)
    protocol = validate_protocol(args.repo, family, source_audit=source_audit)
    contract = campaign_contract(protocol, args)

    if args.dry_run:
        print(json.dumps(dry_run_plan(args, family, contract), indent=2))
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    remaining: list[int | None] = [args.max_new_trials]

    with single_runner_lock(args.output_root):
        if args.stage == "all":
            anchor: Candidate | None = None
            for stage in stage_order(family):
                anchor, complete = execute_stage(
                    args, family, contract, anchor, stage, env, remaining
                )
                if not complete:
                    print(
                        "New-trial budget reached; resume the same command to continue. "
                        "The held-out test split was not evaluated."
                    )
                    return 0
        else:
            anchor = None
            if args.stage != "pilot":
                sequence = stage_order(family)
                predecessor = sequence[sequence.index(args.stage) - 1]
                anchor = load_continuation_candidate(
                    args.resume_from.expanduser().resolve(),
                    family,
                    contract,
                    required_stage=predecessor,
                )
            _, complete = execute_stage(
                args, family, contract, anchor, args.stage, env, remaining
            )
            if not complete:
                print("New-trial budget reached; rerun the same command to resume.")

    print("Validation selection complete; the held-out test split was not evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
