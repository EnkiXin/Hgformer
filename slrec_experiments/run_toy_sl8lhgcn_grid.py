#!/usr/bin/env python3
"""Strict, resumable single-GPU grid for the Amazon-Toy SL8-LHGCN ablation.

The finite grid is deliberately fixed to the comparison requested for the
current experiment::

    gcn_layers in {2, 4, 6, 8, 10}
    train_batch_size in {32768, 65536, 131072}

Everything else is pinned to the current Amazon-Toy protocol and the proposed
``ambient_retract`` + faithful LHGCN hinge configuration.  Every subprocess is
validation-only.  This module contains no held-out-test execution path.

Resume is evidence based: a result is skipped only when its embedded runner
metadata and checkpoint configuration match the complete trial contract.  An
older L4/B65536 result can be adopted explicitly, but only after the same deep
checkpoint and manifold-diagnostic checks pass.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import itertools
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

try:
    from slrec_experiments.run_multidataset_sl8 import (
        DATASET_BY_SLUG,
        audit_filtered_dataset,
        audit_source_file,
    )
except ModuleNotFoundError:  # Allows direct execution from slrec_experiments/.
    from run_multidataset_sl8 import (  # type: ignore
        DATASET_BY_SLUG,
        audit_filtered_dataset,
        audit_source_file,
    )


SCHEMA_VERSION = 1
DATASET_SLUG = "amazon-toy"
DATASET = "Amazon_toy"
SEED = 2024
EPOCHS = 500
EVAL_STEP = 50
STOPPING_STEP = 1000
LAYERS = (2, 4, 6, 8, 10)
BATCH_SIZES = (32_768, 65_536, 131_072)
# The runner reports every scalar in ``model.parameters()``, including the
# fixed (requires_grad=False) log-score-scale parameter. Trainable parameters
# are 1,614,464; the serialized/reported total is therefore 1,614,465.
EXPECTED_PARAMETER_COUNT = 1_614_465
EXPECTED_SPLIT_INTERACTIONS = {
    "train": 99_623,
    "valid": 17_107,
    "test": 17_107,
}
BASE_CONFIG_NAME = "RecFormer_toy.yaml"
MODEL_OVERLAY_NAME = "SL8LHGCN_reproduction.yaml"
ANCHOR = (4, 65_536)

# This is the four-decimal result already reported in the current comparison.
# Supplying --lhgcn-result replaces it with an auditable, full-precision
# reference and enables split-fingerprint verification.
REPORTED_LHGCN_REFERENCE = {
    "recall@10": 0.0659,
    "ndcg@10": 0.0375,
}


@dataclass(frozen=True, order=True)
class Trial:
    gcn_layers: int
    train_batch_size: int

    def validate(self) -> None:
        if self.gcn_layers not in LAYERS:
            raise ValueError(f"gcn_layers must be one of {LAYERS}")
        if self.train_batch_size not in BATCH_SIZES:
            raise ValueError(f"train_batch_size must be one of {BATCH_SIZES}")

    @property
    def name(self) -> str:
        self.validate()
        return f"L{self.gcn_layers:02d}_B{self.train_batch_size:06d}"


def grid_trials() -> tuple[Trial, ...]:
    return tuple(Trial(layer, batch) for layer, batch in itertools.product(LAYERS, BATCH_SIZES))


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
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


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def config_paths(repo: Path) -> tuple[Path, Path]:
    root = repo / "baseline_config_fixed"
    return root / BASE_CONFIG_NAME, root / MODEL_OVERLAY_NAME


def validate_protocol(repo: Path) -> dict[str, Any]:
    """Validate the immutable data/evaluation/model contract before planning."""

    base_path, overlay_path = config_paths(repo)
    merged: dict[str, Any] = {}
    documents: list[dict[str, Any]] = []
    for path in (base_path, overlay_path):
        if not path.is_file():
            raise FileNotFoundError(f"required config does not exist: {path}")
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"config is not a mapping: {path}")
        documents.append(document)
        merged.update(document)

    # Dataset protocol keys must come only from RecFormer_toy.yaml.  This makes
    # a future accidental split/filter override in the model overlay fatal.
    protected = {
        "dataset",
        "seed",
        "reproducibility",
        "USER_ID_FIELD",
        "ITEM_ID_FIELD",
        "RATING_FIELD",
        "load_col",
        "val_interval",
        "user_inter_num_interval",
        "item_inter_num_interval",
        "metrics",
        "topk",
        "valid_metric",
        "eval_args",
    }
    overlap = protected.intersection(documents[1])
    if overlap:
        raise RuntimeError(
            "SL8LHGCN overlay may not override the Toy data/evaluation protocol: "
            f"{sorted(overlap)}"
        )

    expected = {
        "model": "SL8LHGCN",
        "dataset": DATASET,
        "seed": SEED,
        "reproducibility": True,
        "embedding_size": 64,
        "matrix_dim": 8,
        "num_factors": 1,
        "factor_aggregation": "l2",
        "embedding_init": "xavier_uniform_combined",
        "gcn_layers": 4,
        "n_layers": 4,
        "sl_gcn_mode": "ambient_retract",
        "lhgcn_include_self": False,
        "lhgcn_layer_aggregation": "last",
        "sl_layer_norm": "none",
        "sl_membership_check": True,
        "sl_membership_strict": True,
        "pairwise_loss": "lhgcn_hinge_squared_sum",
        "loss_margin": 0.1,
        "score_scale": 1.0,
        "learnable_score_scale": False,
        "learner": "adam",
        "learning_rate": 0.0005,
        "weight_decay": 0.0,
        "reg_weight": 0.0,
        "neg_sampling": {"uniform": 1},
        "epochs": EPOCHS,
        "eval_step": EVAL_STEP,
        "stopping_step": STOPPING_STEP,
        "metrics": ["Recall", "NDCG"],
        "topk": [5, 10, 20, 50],
        "valid_metric": "Recall@10",
        "val_interval": {"rating": "[3,inf)"},
        "user_inter_num_interval": "[5,inf)",
        "item_inter_num_interval": "[5,inf)",
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        },
    }
    actual = {key: merged.get(key) for key in expected}
    if actual != expected:
        differences = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if actual[key] != expected[key]
        }
        raise RuntimeError(f"current Toy SL8-LHGCN protocol changed: {differences}")

    return {
        "dataset": DATASET,
        "seed": SEED,
        "base_config": BASE_CONFIG_NAME,
        "base_config_sha256": _sha256(base_path),
        "model_overlay": MODEL_OVERLAY_NAME,
        "model_overlay_sha256": _sha256(overlay_path),
        "filters": {
            "rating": "[3,inf)",
            "users": "[5,inf)",
            "items": "[5,inf)",
        },
        "split": {
            "args": expected["eval_args"],
            "expected_interactions": EXPECTED_SPLIT_INTERACTIONS,
        },
        "evaluation": {
            "metrics": expected["metrics"],
            "topk": expected["topk"],
            "selection_metric": expected["valid_metric"],
            "validation_only": True,
            "held_out_test_evaluated": False,
        },
        "fixed_model_training": {
            "model": "SL8LHGCN",
            "matrix_dim": 8,
            "embedding_size": 64,
            "num_factors": 1,
            "sl_gcn_mode": "ambient_retract",
            "pairwise_loss": "lhgcn_hinge_squared_sum",
            "loss_margin": 0.1,
            "negative_sampling": {"uniform": 1},
            "learning_rate": 0.0005,
            "epochs": EPOCHS,
            "eval_step": EVAL_STEP,
            "stopping_step": STOPPING_STEP,
        },
        "grid": {
            "gcn_layers": list(LAYERS),
            "train_batch_size": list(BATCH_SIZES),
        },
    }


def trial_metadata(trial: Trial, protocol: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        "schema_version": SCHEMA_VERSION,
        "kind": "amazon-toy-sl8lhgcn-layer-batch-grid",
        "trial": asdict(trial),
        "protocol": protocol,
        "test_evaluated": False,
    }
    return {**core, "signature_sha256": _canonical_hash(core)}


def result_paths(output_root: Path, trial: Trial) -> dict[str, Path]:
    root = output_root.expanduser()
    return {
        "result": root / "results" / f"{trial.name}.json",
        "raw": root / "work" / f"{trial.name}.raw.json",
        "log": root / "logs" / f"{trial.name}.log",
        "checkpoint_dir": root / "checkpoints" / trial.name,
        "failure": root / "failures" / f"{trial.name}.json",
    }


def trial_command(
    args: argparse.Namespace,
    trial: Trial,
    raw_result: Path,
    checkpoint_dir: Path,
) -> list[str]:
    trial.validate()
    base_path, overlay_path = config_paths(args.repo)
    return [
        args.python,
        "-u",
        str(args.repo / "run_recbole_gnn.py"),
        "--model",
        "SL8LHGCN",
        "--dataset",
        DATASET,
        "--config-files",
        f"{base_path} {overlay_path}",
        "--validation-only",
        "--result-file",
        str(raw_result),
        f"--checkpoint_dir={checkpoint_dir}",
        f"--data_path={args.data_root}",
        # This vendored RecBole rewrites CUDA_VISIBLE_DEVICES from gpu_id
        # inside Config._init_device. Passing logical 0 would therefore undo
        # the parent mask and select physical GPU 0. Keep both values equal to
        # the requested physical index; PyTorch still exposes it as cuda:0.
        f"--gpu_id={args.gpu_id}",
        "--use_gpu=true",
        "--show_progress=false",
        f"--seed={SEED}",
        f"--epochs={EPOCHS}",
        f"--eval_step={EVAL_STEP}",
        f"--stopping_step={STOPPING_STEP}",
        f"--gcn_layers={trial.gcn_layers}",
        f"--n_layers={trial.gcn_layers}",
        f"--train_batch_size={trial.train_batch_size}",
        "--eval_batch_size=1048576",
        "--eval_user_chunk_size=64",
        "--eval_item_chunk_size=1024",
        "--embedding_size=64",
        "--matrix_dim=8",
        "--num_factors=1",
        "--factor_aggregation=l2",
        "--embedding_init=xavier_uniform_combined",
        "--init_std=0.01",
        "--coord_clip=0.75",
        "--sl_scale=1.0",
        "--sl_gcn_mode=ambient_retract",
        "--lhgcn_include_self=false",
        "--lhgcn_layer_aggregation=last",
        "--sl_layer_norm=none",
        "--sl_centroid_fallback_clip=1.0",
        "--sl_membership_check=true",
        "--sl_membership_strict=true",
        "--sl_membership_tolerance=0.0001",
        "--sl_distance_membership_check=true",
        "--sl_distance_check_samples=16",
        "--sl_log_trace_tolerance=0.001",
        "--schatten_p=2",
        "--log_terms=12",
        "--log_jitter=0.0",
        "--symmetric_distance=false",
        "--score_scale=1.0",
        "--learnable_score_scale=false",
        "--pairwise_loss=lhgcn_hinge_squared_sum",
        "--loss_margin=0.1",
        "--learner=adam",
        "--learning_rate=0.0005",
        "--weight_decay=0.0",
        "--reg_weight=0.0",
        "--neg_sampling={'uniform': 1}",
        "--tail_analysis=false",
        "--popularity_analysis=false",
    ]


def _config_dictionary(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    values = getattr(config, "final_config_dict", None)
    if isinstance(values, Mapping):
        return values
    raise ValueError("checkpoint config is neither a mapping nor a RecBole Config")


def _load_checkpoint_config(path: Path, repo: Path) -> tuple[Mapping[str, Any], Any]:
    """Load an explicitly trusted local RecBole checkpoint for provenance checks."""

    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")
    repo_token = str(repo.resolve())
    inserted = repo_token not in sys.path
    if inserted:
        sys.path.insert(0, repo_token)
    try:
        import torch

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    finally:
        if inserted:
            sys.path.remove(repo_token)
    if not isinstance(checkpoint, Mapping) or "config" not in checkpoint:
        raise ValueError(f"not a RecBole checkpoint: {path}")
    return _config_dictionary(checkpoint["config"]), checkpoint.get("epoch")


def expected_checkpoint_values(trial: Trial) -> dict[str, Any]:
    return {
        "model": "SL8LHGCN",
        "dataset": DATASET,
        "seed": SEED,
        "reproducibility": True,
        "epochs": EPOCHS,
        "eval_step": EVAL_STEP,
        "stopping_step": STOPPING_STEP,
        "gcn_layers": trial.gcn_layers,
        "n_layers": trial.gcn_layers,
        "train_batch_size": trial.train_batch_size,
        "embedding_size": 64,
        "matrix_dim": 8,
        "num_factors": 1,
        "factor_aggregation": "l2",
        "embedding_init": "xavier_uniform_combined",
        "sl_gcn_mode": "ambient_retract",
        "lhgcn_include_self": False,
        "lhgcn_layer_aggregation": "last",
        "sl_layer_norm": "none",
        "sl_membership_check": True,
        "sl_membership_strict": True,
        "pairwise_loss": "lhgcn_hinge_squared_sum",
        "loss_margin": 0.1,
        "score_scale": 1.0,
        "learnable_score_scale": False,
        "schatten_p": 2,
        "log_terms": 12,
        "log_jitter": 0.0,
        "symmetric_distance": False,
        "learner": "adam",
        "learning_rate": 0.0005,
        "weight_decay": 0.0,
        "reg_weight": 0.0,
        "neg_sampling": {"uniform": 1},
        "eval_batch_size": 1_048_576,
        "eval_user_chunk_size": 64,
        "eval_item_chunk_size": 1024,
        "metrics": ["Recall", "NDCG"],
        "topk": [5, 10, 20, 50],
        "valid_metric": "Recall@10",
        "val_interval": {"rating": "[3,inf)"},
        "user_inter_num_interval": "[5,inf)",
        "item_inter_num_interval": "[5,inf)",
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        },
    }


def validate_checkpoint_contract(path: Path, repo: Path, trial: Trial) -> Any:
    config, checkpoint_epoch = _load_checkpoint_config(path, repo)
    expected = expected_checkpoint_values(trial)
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint trial contract mismatch: {mismatches}")
    return checkpoint_epoch


def _finite_metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"missing finite validation metric {name!r}")
    return float(value)


def validate_split_fingerprints(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fingerprints = payload.get("split_fingerprints")
    if not isinstance(fingerprints, Mapping) or set(fingerprints) != set(
        EXPECTED_SPLIT_INTERACTIONS
    ):
        raise ValueError("result lacks train/valid/test split fingerprints")
    for split, expected_count in EXPECTED_SPLIT_INTERACTIONS.items():
        record = fingerprints[split]
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid {split} split fingerprint")
        if int(record.get("interactions", -1)) != expected_count:
            raise ValueError(
                f"{split} interactions {record.get('interactions')} != {expected_count}"
            )
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid {split} SHA256 fingerprint")
    return fingerprints


def manifold_summary(diagnostics: Mapping[str, Any], trial: Trial) -> dict[str, Any]:
    """Validate and compact the every-layer determinant/matrix-log audit."""

    if diagnostics.get("mode") != "ambient_retract":
        raise ValueError("model diagnostics are not for ambient_retract")
    if int(diagnostics.get("layers", -1)) != trial.gcn_layers:
        raise ValueError("model diagnostic layer count does not match the trial")

    zero_fields = (
        "active_singular_fallbacks",
        "output_membership_violations",
        "nonpositive_output_determinants",
        "nonfinite_output_log_determinants",
    )
    for field in zero_fields:
        if int(diagnostics.get(field, -1)) != 0:
            raise ValueError(f"manifold diagnostic {field} is not zero")

    tolerance = float(diagnostics.get("membership_tolerance", math.nan))
    output_error = float(diagnostics.get("max_abs_output_log_determinant", math.inf))
    if not math.isfinite(tolerance) or tolerance <= 0 or output_error > tolerance:
        raise ValueError("final determinant error exceeds the membership tolerance")

    initial = diagnostics.get("initial_group_membership")
    if not isinstance(initial, Mapping) or int(initial.get("membership_violations", -1)) != 0:
        raise ValueError("initial group table did not pass the SL(8) membership check")

    layers = diagnostics.get("layer_membership")
    if not isinstance(layers, list) or len(layers) != trial.gcn_layers:
        raise ValueError("missing every-layer retraction diagnostics")
    compact_layers: list[dict[str, Any]] = []
    for index, layer in enumerate(layers, start=1):
        if not isinstance(layer, Mapping) or int(layer.get("layer", -1)) != index:
            raise ValueError(f"invalid diagnostics for layer {index}")
        if int(layer.get("output_membership_violations", -1)) != 0:
            raise ValueError(f"layer {index} left SL(8) after retraction")
        layer_error = float(layer.get("max_abs_output_log_determinant", math.inf))
        if not math.isfinite(layer_error) or layer_error > tolerance:
            raise ValueError(f"layer {index} determinant error exceeds tolerance")
        compact_layers.append(
            {
                "layer": index,
                "ambient_input_membership_violations": int(
                    layer.get("input_membership_violations", 0)
                ),
                "orientation_repairs": int(layer.get("orientation_repairs", 0)),
                "active_singular_fallbacks": int(
                    layer.get("active_singular_fallbacks", 0)
                ),
                "inactive_singular_fallbacks": int(
                    layer.get("inactive_singular_fallbacks", 0)
                ),
                "output_membership_violations": 0,
                "max_abs_output_log_determinant": layer_error,
            }
        )

    distance = diagnostics.get("distance_membership")
    if not isinstance(distance, Mapping):
        raise ValueError("missing SL(8) distance-path diagnostics")
    if int(distance.get("relative_membership_violations", -1)) != 0:
        raise ValueError("relative matrices left SL(8)")
    if int(distance.get("nonfinite_approximate_logs", -1)) != 0:
        raise ValueError("matrix-log approximation produced non-finite values")
    trace_error = float(
        distance.get("max_normalized_approximate_log_trace", math.inf)
    )
    trace_tolerance = float(distance.get("log_trace_tolerance", math.nan))
    if (
        not math.isfinite(trace_error)
        or not math.isfinite(trace_tolerance)
        or trace_error > trace_tolerance
    ):
        raise ValueError("matrix-log trace error exceeds tolerance")

    return {
        "passed": True,
        "mode": "ambient_retract",
        "layers": trial.gcn_layers,
        "membership_tolerance": tolerance,
        "initial_membership_violations": 0,
        "per_layer": compact_layers,
        "orientation_repairs": int(diagnostics.get("orientation_repairs", 0)),
        "active_singular_fallbacks": 0,
        "inactive_singular_fallbacks": int(
            diagnostics.get("inactive_singular_fallbacks", 0)
        ),
        "final_membership_violations": 0,
        "max_abs_final_log_determinant": output_error,
        "distance": {
            "samples": int(distance.get("samples", 0)),
            "relative_membership_violations": 0,
            "nonfinite_approximate_logs": 0,
            "max_normalized_approximate_log_trace": trace_error,
            "log_trace_tolerance": trace_tolerance,
            "max_approximate_log_reconstruction_residual": float(
                distance.get(
                    "max_approximate_log_reconstruction_residual", math.nan
                )
            ),
        },
    }


def validate_result(
    payload: Mapping[str, Any],
    *,
    repo: Path,
    trial: Trial,
    protocol: Mapping[str, Any],
    require_metadata: bool,
) -> dict[str, Any]:
    if payload.get("model") != "SL8LHGCN" or payload.get("dataset") != DATASET:
        raise ValueError("wrong model or dataset in result")
    if int(payload.get("seed", -1)) != SEED:
        raise ValueError("wrong seed in result")
    if int(payload.get("epochs", -1)) != EPOCHS:
        raise ValueError("wrong epoch budget in result")
    if int(payload.get("stopping_step", -1)) != STOPPING_STEP:
        raise ValueError("wrong stopping_step in result")
    if payload.get("test_result") is not None:
        raise RuntimeError("grid result touched the held-out test split")
    if int(payload.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("unexpected SL8-LHGCN parameter count")

    config_files = payload.get("config_files")
    if not isinstance(config_files, list) or [Path(str(item)).name for item in config_files] != [
        BASE_CONFIG_NAME,
        MODEL_OVERLAY_NAME,
    ]:
        raise ValueError("result was not trained with the exact Toy + SL8LHGCN configs")

    metrics = payload.get("best_valid_result")
    if not isinstance(metrics, Mapping):
        raise ValueError("missing validation metrics")
    recall = _finite_metric(metrics, "recall@10")
    ndcg = _finite_metric(metrics, "ndcg@10")
    score = payload.get("best_valid_score")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError("missing finite best validation score")
    if not math.isclose(float(score), recall, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("best_valid_score is not full-ranking Recall@10")

    fingerprints = validate_split_fingerprints(payload)
    diagnostics = payload.get("model_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("missing model manifold diagnostics")
    compact_diagnostics = manifold_summary(diagnostics, trial)

    checkpoint_token = payload.get("checkpoint_file")
    if not isinstance(checkpoint_token, str) or not checkpoint_token:
        raise ValueError("missing checkpoint path")
    checkpoint = Path(checkpoint_token).expanduser()
    checkpoint_epoch = validate_checkpoint_contract(checkpoint, repo, trial)

    expected_metadata = trial_metadata(trial, protocol)
    if require_metadata and payload.get("toy_sl8_grid") != expected_metadata:
        raise ValueError("resume metadata does not match the exact trial contract")

    runtime = payload.get("grid_runtime")
    if runtime is not None:
        if not isinstance(runtime, Mapping):
            raise ValueError("invalid grid runtime metadata")
        duration = runtime.get("duration_seconds")
        if duration is not None and (
            not isinstance(duration, (int, float)) or float(duration) < 0
        ):
            raise ValueError("invalid trial duration")

    return {
        "recall@10": recall,
        "ndcg@10": ndcg,
        "split_fingerprints": fingerprints,
        "checkpoint_epoch": checkpoint_epoch,
        "manifold": compact_diagnostics,
    }


def completed_result(
    path: Path,
    *,
    repo: Path,
    trial: Trial,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        payload = _load_mapping(path)
        validate_result(
            payload,
            repo=repo,
            trial=trial,
            protocol=protocol,
            require_metadata=True,
        )
        return payload, None
    except RuntimeError:
        # Never silently overwrite an artifact which accessed the test split.
        raise
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return None, str(error)


def annotate_result(
    raw_path: Path,
    final_path: Path,
    *,
    repo: Path,
    trial: Trial,
    protocol: Mapping[str, Any],
    runtime: Mapping[str, Any],
    reused_from: Path | None = None,
) -> dict[str, Any]:
    payload = _load_mapping(raw_path)
    # Deep validation before annotation prevents adopting a merely
    # filename-matching legacy result.
    validate_result(
        payload,
        repo=repo,
        trial=trial,
        protocol=protocol,
        require_metadata=False,
    )
    payload["toy_sl8_grid"] = trial_metadata(trial, protocol)
    payload["grid_runtime"] = dict(runtime)
    if reused_from is not None:
        payload["grid_reuse"] = {
            "source_result": str(reused_from.expanduser().resolve()),
            "adopted_at": _utc_now(),
            "checkpoint_contract_verified": True,
        }
    _atomic_json(final_path, payload)
    validated, reason = completed_result(
        final_path,
        repo=repo,
        trial=trial,
        protocol=protocol,
    )
    if validated is None:
        raise RuntimeError(f"annotated result failed its own resume check: {reason}")
    return validated


def _trial_candidate(path: Path, trial: Trial, payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["best_valid_result"]
    compact = manifold_summary(payload["model_diagnostics"], trial)
    runtime = payload.get("grid_runtime") or {}
    return {
        "trial": trial.name,
        "gcn_layers": trial.gcn_layers,
        "train_batch_size": trial.train_batch_size,
        "status": "complete",
        "result_file": str(path.expanduser().resolve()),
        "checkpoint_file": payload["checkpoint_file"],
        "recall@10": float(metrics["recall@10"]),
        "ndcg@10": float(metrics["ndcg@10"]),
        "runtime": {
            "started_at": runtime.get("started_at"),
            "finished_at": runtime.get("finished_at"),
            "duration_seconds": runtime.get("duration_seconds"),
            "source": runtime.get("source", "unknown"),
        },
        "split_fingerprints": payload["split_fingerprints"],
        "manifold": compact,
        "test_evaluated": False,
    }


def load_lhgcn_reference(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "source": "reported-four-decimal-current-Toy-LHGCN-L4",
            "result_file": None,
            "model": "LHGCN",
            "gcn_layers": 4,
            "recall@10": REPORTED_LHGCN_REFERENCE["recall@10"],
            "ndcg@10": REPORTED_LHGCN_REFERENCE["ndcg@10"],
            "precision_note": "rounded to four decimals; split SHA not independently rechecked",
            "split_fingerprints": None,
            "test_evaluated": False,
        }

    payload = _load_mapping(path.expanduser())
    if payload.get("model") not in {"LHGCN", "HGCF"} or payload.get("dataset") != DATASET:
        raise ValueError("LHGCN reference has the wrong model or dataset")
    if int(payload.get("seed", -1)) != SEED:
        raise ValueError("LHGCN reference has the wrong seed")
    if payload.get("test_result") is not None:
        raise RuntimeError("LHGCN reference touched the held-out test split")
    metrics = payload.get("best_valid_result")
    if not isinstance(metrics, Mapping):
        raise ValueError("LHGCN reference lacks validation metrics")
    fingerprints = validate_split_fingerprints(payload)
    return {
        "source": "validation-result-json",
        "result_file": str(path.expanduser().resolve()),
        "model": payload["model"],
        "gcn_layers": 4,
        "recall@10": _finite_metric(metrics, "recall@10"),
        "ndcg@10": _finite_metric(metrics, "ndcg@10"),
        "precision_note": "full precision from supplied result",
        "split_fingerprints": fingerprints,
        "test_evaluated": False,
    }


def _delta(value: float, reference: float) -> float:
    return value - reference


def write_summary(
    path: Path,
    *,
    repo: Path,
    output_root: Path,
    protocol: Mapping[str, Any],
    reference: Mapping[str, Any],
    failures: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    failures = failures or {}
    candidates: list[dict[str, Any]] = []
    pending: list[str] = []
    invalid: dict[str, str] = {}
    canonical_split: Mapping[str, Any] | None = None
    reference_split = reference.get("split_fingerprints")

    for trial in grid_trials():
        result_path = result_paths(output_root, trial)["result"]
        result, reason = completed_result(
            result_path, repo=repo, trial=trial, protocol=protocol
        )
        if result is None:
            if reason:
                invalid[trial.name] = reason
            else:
                pending.append(trial.name)
            continue
        candidate = _trial_candidate(result_path, trial, result)
        split = candidate["split_fingerprints"]
        if canonical_split is None:
            canonical_split = split
        elif split != canonical_split:
            raise RuntimeError("SL8 grid trials use different data splits")
        if reference_split is not None and split != reference_split:
            raise RuntimeError("SL8 grid split differs from the supplied LHGCN split")
        candidate["delta_vs_lhgcn"] = {
            "recall@10": _delta(candidate["recall@10"], float(reference["recall@10"])),
            "ndcg@10": _delta(candidate["ndcg@10"], float(reference["ndcg@10"])),
        }
        candidates.append(candidate)

    anchor = next(
        (
            item
            for item in candidates
            if (item["gcn_layers"], item["train_batch_size"]) == ANCHOR
        ),
        None,
    )
    for candidate in candidates:
        candidate["delta_vs_l4_b65536"] = (
            {
                "recall@10": _delta(candidate["recall@10"], anchor["recall@10"]),
                "ndcg@10": _delta(candidate["ndcg@10"], anchor["ndcg@10"]),
            }
            if anchor is not None
            else None
        )

    ranking = sorted(
        candidates,
        key=lambda item: (
            -item["recall@10"],
            -item["ndcg@10"],
            item["gcn_layers"],
            item["train_batch_size"],
        ),
    )
    durations = [
        float(item["runtime"]["duration_seconds"])
        for item in candidates
        if isinstance(item["runtime"]["duration_seconds"], (int, float))
    ]
    complete = len(candidates) == len(grid_trials())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "amazon-toy-sl8lhgcn-layer-batch-grid-summary",
        "state": "complete" if complete else "incomplete",
        "dataset": DATASET,
        "protocol": protocol,
        "grid": {
            "gcn_layers": list(LAYERS),
            "train_batch_size": list(BATCH_SIZES),
            "expected_trials": len(grid_trials()),
            "completed_trials": len(candidates),
            "failed_trials": sorted(failures),
            "pending_trials": pending,
            "invalid_results": invalid,
        },
        "lhgcn_reference": reference,
        "anchor": anchor,
        "winner": ranking[0] if complete and ranking else None,
        "provisional_winner": ranking[0] if ranking else None,
        "ranking": ranking,
        "runtime": {
            "known_trial_seconds": sum(durations),
            "known_trial_hours": sum(durations) / 3600.0,
            "trials_with_known_duration": len(durations),
        },
        "split_fingerprints": canonical_split,
        "manifold_acceptance": {
            "every_completed_trial_passed": all(
                item["manifold"]["passed"] for item in candidates
            ),
            "criterion": (
                "initial/final and every post-retraction matrix table in SL(8); "
                "zero active singular fallbacks; relative matrices in SL(8); "
                "finite approximately trace-free matrix logs"
            ),
        },
        "failures": dict(failures),
        "test_evaluated": False,
    }
    _atomic_json(path, payload)
    return payload


def _gpu_token(value: str) -> str:
    token = value.strip()
    if not token.isdigit():
        raise ValueError("--gpu-id must be one non-negative physical CUDA index")
    return str(int(token))


def default_lock_path(gpu_id: str) -> Path:
    digest = hashlib.sha256(gpu_id.encode("utf-8")).hexdigest()[:16]
    return (
        Path(tempfile.gettempdir())
        / f"hgformer-toy-sl8lhgcn-uid{os.getuid()}-gpu-{digest}.lock"
    )


@contextlib.contextmanager
def exclusive_gpu_lock(path: Path, gpu_id: str) -> Iterable[int]:
    """Hold one non-blocking physical-GPU lock for the entire serial grid."""

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock.seek(0)
            owner = lock.read().strip() or "unknown owner"
            raise RuntimeError(
                f"physical GPU {gpu_id} is already reserved: {resolved} ({owner})"
            ) from error
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()} gpu={gpu_id} acquired_at={_utc_now()}\n")
        lock.flush()
        try:
            yield lock.fileno()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _run_and_tee(
    command: list[str],
    *,
    log_path: Path,
    cwd: Path,
    env: Mapping[str, str],
    lock_fd: int,
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
            pass_fds=(lock_fd,),
            start_new_session=True,
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
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _audit_data(args: argparse.Namespace) -> dict[str, Any]:
    spec = DATASET_BY_SLUG[DATASET_SLUG]
    if args.skip_data_audit:
        return {
            "status": "skipped-explicit-dry-run-only",
            "filtered_reference": spec.filtered.json(),
        }
    source = audit_source_file(args.data_root, spec)
    filtered = (
        audit_filtered_dataset(args.repo, args.data_root, spec)
        if args.deep_data_audit
        else None
    )
    return {"source": source, "filtered": filtered}


def _dry_run_plan(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    reference: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for trial in grid_trials():
        paths = result_paths(args.output_root, trial)
        complete, invalid_reason = completed_result(
            paths["result"], repo=args.repo, trial=trial, protocol=protocol
        )
        status = "skip-exact-complete" if complete is not None else "run"
        if (
            complete is None
            and (trial.gcn_layers, trial.train_batch_size) == ANCHOR
            and args.reuse_l4_b65536_result is not None
        ):
            legacy = _load_mapping(args.reuse_l4_b65536_result.expanduser())
            validate_result(
                legacy,
                repo=args.repo,
                trial=trial,
                protocol=protocol,
                require_metadata=False,
            )
            status = "adopt-verified-existing-result"
        jobs.append(
            {
                "trial": trial.name,
                "parameters": asdict(trial),
                "status": status,
                "invalid_existing_reason": invalid_reason,
                "result_file": str(paths["result"]),
                "command": trial_command(
                    args, trial, paths["raw"], paths["checkpoint_dir"]
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "dry_run": True,
        "protocol": protocol,
        "data_audit": audit,
        "lhgcn_reference": reference,
        "single_physical_gpu": args.gpu_id,
        "child_cuda_visible_devices": args.gpu_id,
        "child_config_gpu_id": args.gpu_id,
        "child_torch_device_after_mask": "cuda:0",
        "lock_file": str(args.lock_file),
        "strict_serial": True,
        "jobs": jobs,
        "test_evaluated": False,
    }


def _load_failures(output_root: Path) -> dict[str, Mapping[str, Any]]:
    failures: dict[str, Mapping[str, Any]] = {}
    for trial in grid_trials():
        path = result_paths(output_root, trial)["failure"]
        if path.is_file():
            try:
                failures[trial.name] = _load_mapping(path)
            except (OSError, ValueError, json.JSONDecodeError):
                failures[trial.name] = {"status": "invalid-failure-record"}
    return failures


def execute(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    attempts = 0
    summary_path = args.output_root / "summary.json"

    with exclusive_gpu_lock(args.lock_file, args.gpu_id) as lock_fd:
        for trial in grid_trials():
            paths = result_paths(args.output_root, trial)
            complete, invalid_reason = completed_result(
                paths["result"], repo=args.repo, trial=trial, protocol=protocol
            )
            if complete is not None:
                print(f"SKIP {trial.name}: exact completed result")
                continue

            # A training subprocess writes its raw result only after the full
            # validation-only run succeeds. If orchestration/annotation then
            # fails (for example, an overly strict metadata assertion), recover
            # that expensive artifact after the same checkpoint, split, test-
            # isolation, and manifold checks instead of retraining it.
            if paths["raw"].is_file():
                try:
                    failure_payload = (
                        _load_mapping(paths["failure"])
                        if paths["failure"].is_file()
                        else {}
                    )
                    recovered = annotate_result(
                        paths["raw"],
                        paths["result"],
                        repo=args.repo,
                        trial=trial,
                        protocol=protocol,
                        runtime={
                            "started_at": failure_payload.get("started_at"),
                            "finished_at": failure_payload.get("failed_at", _utc_now()),
                            "duration_seconds": failure_payload.get("duration_seconds"),
                            "source": "recovered-complete-raw-result",
                        },
                    )
                except RuntimeError:
                    # A raw artifact that touched test is never ignored or
                    # overwritten by a fresh training run.
                    raise
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    print(f"IGNORE RAW {trial.name}: {error}")
                else:
                    if paths["failure"].is_file():
                        paths["failure"].unlink()
                    metrics = recovered["best_valid_result"]
                    print(
                        f"RECOVER {trial.name}: Recall@10="
                        f"{float(metrics['recall@10']):.6f}"
                    )
                    write_summary(
                        summary_path,
                        repo=args.repo,
                        output_root=args.output_root,
                        protocol=protocol,
                        reference=reference,
                        failures=_load_failures(args.output_root),
                    )
                    continue

            if (
                (trial.gcn_layers, trial.train_batch_size) == ANCHOR
                and args.reuse_l4_b65536_result is not None
            ):
                source = args.reuse_l4_b65536_result.expanduser()
                legacy = _load_mapping(source)
                validate_result(
                    legacy,
                    repo=args.repo,
                    trial=trial,
                    protocol=protocol,
                    require_metadata=False,
                )
                annotate_result(
                    source,
                    paths["result"],
                    repo=args.repo,
                    trial=trial,
                    protocol=protocol,
                    runtime={
                        "started_at": None,
                        "finished_at": _utc_now(),
                        "duration_seconds": None,
                        "source": "verified-existing-result",
                    },
                    reused_from=source,
                )
                print(f"ADOPT {trial.name}: verified exact existing result")
                if paths["failure"].is_file():
                    paths["failure"].unlink()
                write_summary(
                    summary_path,
                    repo=args.repo,
                    output_root=args.output_root,
                    protocol=protocol,
                    reference=reference,
                    failures=_load_failures(args.output_root),
                )
                continue

            if args.max_new_trials is not None and attempts >= args.max_new_trials:
                break
            attempts += 1
            paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
            paths["raw"].parent.mkdir(parents=True, exist_ok=True)
            started_at = _utc_now()
            started_clock = time.monotonic()
            command = trial_command(
                args, trial, paths["raw"], paths["checkpoint_dir"]
            )
            print(
                f"START {trial.name}: layers={trial.gcn_layers} "
                f"batch={trial.train_batch_size}"
            )
            try:
                _run_and_tee(
                    command,
                    log_path=paths["log"],
                    cwd=args.repo,
                    env=environment,
                    lock_fd=lock_fd,
                )
                duration = time.monotonic() - started_clock
                annotate_result(
                    paths["raw"],
                    paths["result"],
                    repo=args.repo,
                    trial=trial,
                    protocol=protocol,
                    runtime={
                        "started_at": started_at,
                        "finished_at": _utc_now(),
                        "duration_seconds": duration,
                        "source": "runner-measured",
                    },
                )
                if paths["failure"].is_file():
                    paths["failure"].unlink()
                print(f"DONE {trial.name}: {duration:.1f}s")
            except (subprocess.CalledProcessError, OSError, ValueError) as error:
                failure = {
                    "trial": trial.name,
                    "parameters": asdict(trial),
                    "status": "failed",
                    "started_at": started_at,
                    "failed_at": _utc_now(),
                    "duration_seconds": time.monotonic() - started_clock,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "invalid_existing_reason": invalid_reason,
                    "log_file": str(paths["log"].expanduser().resolve()),
                    "test_evaluated": False,
                }
                _atomic_json(paths["failure"], failure)
                print(f"FAILED {trial.name}: {error}", file=sys.stderr)
                write_summary(
                    summary_path,
                    repo=args.repo,
                    output_root=args.output_root,
                    protocol=protocol,
                    reference=reference,
                    failures=_load_failures(args.output_root),
                )
                if not args.continue_on_error:
                    raise
                continue

            write_summary(
                summary_path,
                repo=args.repo,
                output_root=args.output_root,
                protocol=protocol,
                reference=reference,
                failures=_load_failures(args.output_root),
            )

    write_summary(
        summary_path,
        repo=args.repo,
        output_root=args.output_root,
        protocol=protocol,
        reference=reference,
        failures=_load_failures(args.output_root),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--data-root", type=Path, default=repo / "dataset")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--max-new-trials", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--deep-data-audit", action="store_true")
    parser.add_argument(
        "--skip-data-audit",
        action="store_true",
        help="only valid with --dry-run on a planning machine without Toy data",
    )
    parser.add_argument(
        "--lhgcn-result",
        type=Path,
        help="exact validation-only LHGCN L4 result used for full-precision deltas",
    )
    parser.add_argument(
        "--reuse-l4-b65536-result",
        type=Path,
        help=(
            "adopt an older L4/B65536 validation result only after its checkpoint "
            "and manifold diagnostics pass the exact current contract"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_trials is not None and args.max_new_trials < 0:
        parser.error("--max-new-trials must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.repo = args.repo.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.gpu_id = _gpu_token(args.gpu_id)
    args.lock_file = (
        args.lock_file.expanduser().resolve()
        if args.lock_file is not None
        else default_lock_path(args.gpu_id)
    )
    if args.skip_data_audit and not args.dry_run:
        raise ValueError("--skip-data-audit is permitted only with --dry-run")
    protocol = validate_protocol(args.repo)
    reference = load_lhgcn_reference(args.lhgcn_result)
    audit = _audit_data(args)
    if args.dry_run:
        print(json.dumps(_dry_run_plan(args, protocol, reference, audit), indent=2))
        return 0
    execute(args, protocol, reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
