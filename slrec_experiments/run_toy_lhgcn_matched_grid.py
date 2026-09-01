#!/usr/bin/env python3
"""Strict matched LHGCN layer/batch grid for the Amazon-Toy SL(8) study.

This runner is the hyperbolic control for ``run_toy_sl8lhgcn_grid.py``.  It
uses the explicit :class:`LHGCN` adapter, which is exactly the released
``HGCF + conv=lGCN`` path, and varies only::

    gcn_layers in {2, 4, 6, 8, 10}
    train_batch_size in {32768, 65536, 131072}

The Toy filtering, random split, seed, pairwise sampler, 500-epoch budget,
50-epoch full-ranking validation cadence, and all remaining LHGCN settings
are fixed.  The held-out test split is fingerprinted but never evaluated.

Resume checks are intentionally deep.  Besides result metadata and the exact
split fingerprints, the checkpoint configuration and state dictionary must
prove that the model contains one ``[25226, 64]`` hyperboloid embedding table
plus the single shared ``LorentzBatchNorm.gamma`` scalar used by every lGCN
layer.  Thus all layer settings have 1,614,465 parameters; depth does not add
parameters in the released implementation.

The physical-GPU lock is shared with the SL(8) grid runner.  Consequently the
two grids cannot overlap accidentally when both target physical GPU 7.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

try:
    from slrec_experiments.run_multidataset_sl8 import (
        DATASET_BY_SLUG,
        audit_filtered_dataset,
        audit_source_file,
    )
    from slrec_experiments.run_toy_sl8lhgcn_grid import (
        _atomic_json,
        _canonical_hash,
        _gpu_token,
        _load_mapping,
        _run_and_tee,
        default_lock_path,
        exclusive_gpu_lock,
    )
except ModuleNotFoundError:  # Allows direct execution from slrec_experiments/.
    from run_multidataset_sl8 import (  # type: ignore
        DATASET_BY_SLUG,
        audit_filtered_dataset,
        audit_source_file,
    )
    from run_toy_sl8lhgcn_grid import (  # type: ignore
        _atomic_json,
        _canonical_hash,
        _gpu_token,
        _load_mapping,
        _run_and_tee,
        default_lock_path,
        exclusive_gpu_lock,
    )


SCHEMA_VERSION = 1
DATASET_SLUG = "amazon-toy"
DATASET = "Amazon_toy"
SEED = 2024
EPOCHS = 500
EVAL_STEP = 50
# With ten validation events this makes early stopping impossible, while also
# matching the fixed-budget contract of the companion SL(8) grid.
STOPPING_STEP = 1000
LAYERS = (2, 4, 6, 8, 10)
BATCH_SIZES = (32_768, 65_536, 131_072)
ANCHOR = (4, 65_536)

N_USERS = 15_529
N_ITEMS = 9_697
N_NODES = N_USERS + N_ITEMS
EMBEDDING_SIZE = 64
EMBEDDING_PARAMETER_COUNT = N_NODES * EMBEDDING_SIZE
LORENTZ_BN_PARAMETER_COUNT = 1
EXPECTED_PARAMETER_COUNT = EMBEDDING_PARAMETER_COUNT + LORENTZ_BN_PARAMETER_COUNT

EXPECTED_SPLIT_FINGERPRINTS: dict[str, dict[str, Any]] = {
    "train": {
        "interactions": 99_623,
        "sha256": "f849f957e38ad14a49d6a230859d9f6485c8d176dc099358991a79794a48c10c",
    },
    "valid": {
        "interactions": 17_107,
        "sha256": "987284366cd6526fa94fc80f1ae95259e74f4df1ad823aeafc3df6a5fcd90af0",
    },
    "test": {
        "interactions": 17_107,
        "sha256": "2a4b7f533c250c3125857beb0c1a6f9f6ad3934a3466f0432139383c20750b4a",
    },
}

BASE_CONFIG_NAME = "RecFormer_toy.yaml"
MODEL_OVERLAY_NAME = "LHGCN_reproduction.yaml"


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


def config_paths(repo: Path) -> tuple[Path, Path]:
    root = repo / "baseline_config_fixed"
    return root / BASE_CONFIG_NAME, root / MODEL_OVERLAY_NAME


def validate_protocol(repo: Path) -> dict[str, Any]:
    """Validate the immutable Toy data/evaluation and LHGCN contracts."""

    base_path, overlay_path = config_paths(repo)
    documents: list[dict[str, Any]] = []
    for path in (base_path, overlay_path):
        if not path.is_file():
            raise FileNotFoundError(f"required config does not exist: {path}")
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"config is not a mapping: {path}")
        documents.append(document)
    base, overlay = documents

    protected = {
        "dataset",
        "seed",
        "reproducibility",
        "field_separator",
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
    overlap = protected.intersection(overlay)
    if overlap:
        raise RuntimeError(
            "LHGCN overlay may not override the Toy data/evaluation protocol: "
            f"{sorted(overlap)}"
        )

    expected_base = {
        "dataset": DATASET,
        "seed": SEED,
        "reproducibility": True,
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": "rating",
        "load_col": {"inter": ["user_id", "item_id", "rating"]},
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
    base_actual = {key: base.get(key) for key in expected_base}
    if base_actual != expected_base:
        differences = {
            key: {"expected": expected_base[key], "actual": base_actual[key]}
            for key in expected_base
            if base_actual[key] != expected_base[key]
        }
        raise RuntimeError(f"current Toy protocol changed: {differences}")

    expected_overlay = {
        "model": "LHGCN",
        "embedding_size": EMBEDDING_SIZE,
        "conv": "lGCN",
        "gcn_layers": 4,
        "curve": 0.5,
        "scale": 0.1,
        "margin": 0.1,
        "learner": "adam",
        "learning_rate": 0.0005,
        "reg_weight": 0.0,
        "weight_decay": 0.0,
        "neg_sampling": {"uniform": 1},
        "train_batch_size": 65_536,
        "epochs": EPOCHS,
        "eval_step": EVAL_STEP,
        "tail_analysis": False,
        "popularity_analysis": False,
    }
    overlay_actual = {key: overlay.get(key) for key in expected_overlay}
    if overlay_actual != expected_overlay:
        differences = {
            key: {"expected": expected_overlay[key], "actual": overlay_actual[key]}
            for key in expected_overlay
            if overlay_actual[key] != expected_overlay[key]
        }
        raise RuntimeError(f"current explicit LHGCN overlay changed: {differences}")

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
            "args": expected_base["eval_args"],
            "exact_fingerprints": EXPECTED_SPLIT_FINGERPRINTS,
        },
        "evaluation": {
            "metrics": expected_base["metrics"],
            "topk": expected_base["topk"],
            "selection_metric": expected_base["valid_metric"],
            "validation_only": True,
            "held_out_test_evaluated": False,
        },
        "fixed_model_training": {
            "model": "LHGCN",
            "released_equivalent": "HGCF + conv=lGCN",
            "embedding_size": EMBEDDING_SIZE,
            "curve": 0.5,
            "scale": 0.1,
            "margin": 0.1,
            "loss": "batch-summed squared hyperbolic-distance hinge",
            "negative_sampling": {"uniform": 1},
            "learning_rate": 0.0005,
            "weight_decay": 0.0,
            "epochs": EPOCHS,
            "eval_step": EVAL_STEP,
            "stopping_step": STOPPING_STEP,
        },
        "parameter_budget": {
            "combined_embedding": EMBEDDING_PARAMETER_COUNT,
            "shared_lorentz_batch_norm_gamma": LORENTZ_BN_PARAMETER_COUNT,
            "reported_and_trainable_total": EXPECTED_PARAMETER_COUNT,
            "depth_dependent_parameters": 0,
        },
        "grid": {
            "gcn_layers": list(LAYERS),
            "train_batch_size": list(BATCH_SIZES),
        },
    }


def trial_metadata(trial: Trial, protocol: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        "schema_version": SCHEMA_VERSION,
        "kind": "amazon-toy-lhgcn-matched-layer-batch-grid",
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
        "LHGCN",
        "--dataset",
        DATASET,
        "--config-files",
        f"{base_path} {overlay_path}",
        "--validation-only",
        "--result-file",
        str(raw_result),
        f"--checkpoint_dir={checkpoint_dir}",
        f"--data_path={args.data_root}",
        # RecBole rewrites CUDA_VISIBLE_DEVICES from gpu_id.  Supplying the
        # same physical id keeps the child on that card, exposed as cuda:0.
        f"--gpu_id={args.gpu_id}",
        "--use_gpu=true",
        "--show_progress=false",
        f"--seed={SEED}",
        f"--epochs={EPOCHS}",
        f"--eval_step={EVAL_STEP}",
        f"--stopping_step={STOPPING_STEP}",
        f"--gcn_layers={trial.gcn_layers}",
        f"--train_batch_size={trial.train_batch_size}",
        "--eval_batch_size=1048576",
        "--eval_user_chunk_size=64",
        "--eval_item_chunk_size=1024",
        f"--embedding_size={EMBEDDING_SIZE}",
        "--conv=lGCN",
        "--curve=0.5",
        "--scale=0.1",
        "--margin=0.1",
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


def _load_checkpoint(path: Path, repo: Path) -> Mapping[str, Any]:
    """Load an explicitly trusted local checkpoint for structural auditing."""

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
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint root is not a mapping: {path}")
    return checkpoint


def expected_checkpoint_values(trial: Trial) -> dict[str, Any]:
    return {
        "model": "LHGCN",
        "dataset": DATASET,
        "seed": SEED,
        "reproducibility": True,
        "epochs": EPOCHS,
        "eval_step": EVAL_STEP,
        "stopping_step": STOPPING_STEP,
        "gcn_layers": trial.gcn_layers,
        "train_batch_size": trial.train_batch_size,
        "embedding_size": EMBEDDING_SIZE,
        "conv": "lGCN",
        "curve": 0.5,
        "scale": 0.1,
        "margin": 0.1,
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
        "tail_analysis": False,
        "popularity_analysis": False,
    }


def validate_checkpoint_contract(path: Path, repo: Path, trial: Trial) -> dict[str, Any]:
    """Check config plus the exact released LHGCN parameter/buffer layout."""

    checkpoint = _load_checkpoint(path, repo)
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("checkpoint lacks config or state_dict")
    config = _config_dictionary(checkpoint["config"])
    expected = expected_checkpoint_values(trial)
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint trial contract mismatch: {mismatches}")

    epoch = checkpoint.get("epoch")
    if (
        not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch < EVAL_STEP - 1
        or epoch >= EPOCHS
        or (epoch + 1) % EVAL_STEP != 0
    ):
        raise ValueError(f"checkpoint epoch is not a validation epoch: {epoch!r}")

    state = checkpoint["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint state_dict is not a mapping")
    expected_keys = {
        "embedding.weight",
        "gcn_conv.layer_norm.gamma",
        "gcn_conv.layer_norm.curve",
        "gcn_conv.layer_norm.beta",
    }
    if set(state) != expected_keys:
        raise ValueError(
            "checkpoint is not the parameter-free released lGCN structure; "
            f"expected state keys {sorted(expected_keys)}, got {sorted(state)}"
        )

    import torch

    embedding = state["embedding.weight"]
    gamma = state["gcn_conv.layer_norm.gamma"]
    curve = state["gcn_conv.layer_norm.curve"]
    beta = state["gcn_conv.layer_norm.beta"]
    tensor_fields = {
        "embedding.weight": embedding,
        "gcn_conv.layer_norm.gamma": gamma,
        "gcn_conv.layer_norm.curve": curve,
        "gcn_conv.layer_norm.beta": beta,
    }
    for name, value in tensor_fields.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"checkpoint tensor {name} is missing or non-finite")
    if tuple(embedding.shape) != (N_NODES, EMBEDDING_SIZE):
        raise ValueError(
            "checkpoint embedding shape does not match the exact Toy graph: "
            f"{tuple(embedding.shape)}"
        )
    if tuple(gamma.shape) != (1,):
        raise ValueError("checkpoint lacks the one shared LorentzBatchNorm gamma")
    if curve.ndim != 0 or not math.isclose(float(curve), 0.5, abs_tol=1e-7):
        raise ValueError("checkpoint LorentzBatchNorm curvature buffer is not 0.5")
    if tuple(beta.shape) != (EMBEDDING_SIZE,):
        raise ValueError("checkpoint LorentzBatchNorm beta has the wrong shape")
    expected_beta = torch.zeros_like(beta)
    expected_beta[0] = math.sqrt(2.0)
    if not torch.allclose(beta, expected_beta, rtol=1e-6, atol=1e-6):
        raise ValueError("checkpoint LorentzBatchNorm beta is not the north pole")

    parameter_count = int(embedding.numel() + gamma.numel())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(f"checkpoint parameter count is {parameter_count}")
    return {
        "passed": True,
        "checkpoint_epoch": epoch,
        "state_keys": sorted(expected_keys),
        "combined_embedding_shape": list(embedding.shape),
        "combined_embedding_parameters": int(embedding.numel()),
        "shared_lorentz_batch_norm_gamma_shape": list(gamma.shape),
        "shared_lorentz_batch_norm_gamma_parameters": int(gamma.numel()),
        "parameter_count": parameter_count,
        "depth_dependent_parameters": 0,
    }


def _finite_metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"missing finite validation metric {name!r}")
    return float(value)


def validate_split_fingerprints(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fingerprints = payload.get("split_fingerprints")
    if fingerprints != EXPECTED_SPLIT_FINGERPRINTS:
        raise ValueError(
            "result split fingerprints differ from the exact seed-2024 Toy split"
        )
    return fingerprints


def validate_result(
    payload: Mapping[str, Any],
    *,
    repo: Path,
    trial: Trial,
    protocol: Mapping[str, Any],
    require_metadata: bool,
) -> dict[str, Any]:
    if payload.get("model") != "LHGCN" or payload.get("dataset") != DATASET:
        raise ValueError("wrong model or dataset in result")
    if int(payload.get("seed", -1)) != SEED:
        raise ValueError("wrong seed in result")
    if int(payload.get("epochs", -1)) != EPOCHS:
        raise ValueError("wrong epoch budget in result")
    if int(payload.get("stopping_step", -1)) != STOPPING_STEP:
        raise ValueError("wrong stopping_step in result")
    if payload.get("test_result") is not None:
        raise RuntimeError("LHGCN grid result touched the held-out test split")
    if int(payload.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("unexpected LHGCN parameter count")

    config_files = payload.get("config_files")
    expected_names = [BASE_CONFIG_NAME, MODEL_OVERLAY_NAME]
    if (
        not isinstance(config_files, list)
        or [Path(str(item)).name for item in config_files] != expected_names
    ):
        raise ValueError("result was not trained with the exact Toy + LHGCN configs")

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
    checkpoint_token = payload.get("checkpoint_file")
    if not isinstance(checkpoint_token, str) or not checkpoint_token:
        raise ValueError("missing checkpoint path")
    checkpoint_audit = validate_checkpoint_contract(
        Path(checkpoint_token).expanduser(), repo, trial
    )

    expected_metadata = trial_metadata(trial, protocol)
    if require_metadata and payload.get("toy_lhgcn_grid") != expected_metadata:
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
        "checkpoint": checkpoint_audit,
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
    validate_result(
        payload,
        repo=repo,
        trial=trial,
        protocol=protocol,
        require_metadata=False,
    )
    payload["toy_lhgcn_grid"] = trial_metadata(trial, protocol)
    payload["grid_runtime"] = dict(runtime)
    if reused_from is not None:
        payload["grid_reuse"] = {
            "source_result": str(reused_from.expanduser().resolve()),
            "adopted_at": _utc_now(),
            "checkpoint_and_split_contract_verified": True,
        }
    _atomic_json(final_path, payload)
    validated, reason = completed_result(
        final_path, repo=repo, trial=trial, protocol=protocol
    )
    if validated is None:
        raise RuntimeError(f"annotated result failed its own resume check: {reason}")
    return validated


def _load_sl8_pairs(
    path: Path | None,
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any] | None]:
    """Load completed same-cell SL(8) metrics from its validated grid summary."""

    if path is None:
        return {}, None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {}, {
            "summary_file": str(resolved),
            "state": "pending-not-found",
            "completed_trials": 0,
        }
    payload = _load_mapping(resolved)
    if (
        payload.get("kind") != "amazon-toy-sl8lhgcn-layer-batch-grid-summary"
        or payload.get("dataset") != DATASET
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError("--sl8-summary is not the companion Toy SL8 grid summary")
    split = payload.get("split_fingerprints")
    if split is not None and split != EXPECTED_SPLIT_FINGERPRINTS:
        raise ValueError("SL8 summary uses a different Toy split")
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError("SL8 summary lacks a ranking list")
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for row in ranking:
        if not isinstance(row, Mapping) or row.get("test_evaluated") is not False:
            raise ValueError("invalid/test-touched row in SL8 summary")
        key = (int(row["gcn_layers"]), int(row["train_batch_size"]))
        if key in pairs or key[0] not in LAYERS or key[1] not in BATCH_SIZES:
            raise ValueError("duplicate or out-of-grid row in SL8 summary")
        pairs[key] = {
            "result_file": row.get("result_file"),
            "recall@10": _finite_metric(row, "recall@10"),
            "ndcg@10": _finite_metric(row, "ndcg@10"),
        }
    return pairs, {
        "summary_file": str(resolved),
        "state": payload.get("state"),
        "completed_trials": len(pairs),
    }


def write_summary(
    path: Path,
    *,
    repo: Path,
    output_root: Path,
    protocol: Mapping[str, Any],
    sl8_summary_path: Path | None = None,
    failures: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    failures = failures or {}
    sl8_pairs, sl8_source = _load_sl8_pairs(sl8_summary_path)
    candidates: list[dict[str, Any]] = []
    pending: list[str] = []
    invalid: dict[str, str] = {}

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
        # Avoid a second checkpoint load here: completed_result just performed
        # the deep audit, and its compact evidence is deterministic.
        metrics = result["best_valid_result"]
        runtime = result.get("grid_runtime") or {}
        sl8 = sl8_pairs.get((trial.gcn_layers, trial.train_batch_size))
        candidate = {
            "trial": trial.name,
            "gcn_layers": trial.gcn_layers,
            "train_batch_size": trial.train_batch_size,
            "status": "complete",
            "result_file": str(result_path.expanduser().resolve()),
            "checkpoint_file": result["checkpoint_file"],
            "recall@10": float(metrics["recall@10"]),
            "ndcg@10": float(metrics["ndcg@10"]),
            "runtime": {
                "started_at": runtime.get("started_at"),
                "finished_at": runtime.get("finished_at"),
                "duration_seconds": runtime.get("duration_seconds"),
                "source": runtime.get("source", "unknown"),
            },
            "split_fingerprints": result["split_fingerprints"],
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "test_evaluated": False,
        }
        candidate["paired_sl8"] = (
            {
                **dict(sl8),
                "delta_sl8_minus_lhgcn": {
                    "recall@10": float(sl8["recall@10"]) - candidate["recall@10"],
                    "ndcg@10": float(sl8["ndcg@10"]) - candidate["ndcg@10"],
                },
            }
            if sl8 is not None
            else None
        )
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
                "recall@10": candidate["recall@10"] - anchor["recall@10"],
                "ndcg@10": candidate["ndcg@10"] - anchor["ndcg@10"],
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
    paired = [item for item in candidates if item["paired_sl8"] is not None]
    durations = [
        float(item["runtime"]["duration_seconds"])
        for item in candidates
        if isinstance(item["runtime"]["duration_seconds"], (int, float))
    ]
    complete = len(candidates) == len(grid_trials())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "amazon-toy-lhgcn-matched-layer-batch-grid-summary",
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
        "architecture_evidence": {
            "entrypoint": "LHGCN",
            "released_equivalent": "HGCF + conv=lGCN",
            "combined_embedding_shape": [N_NODES, EMBEDDING_SIZE],
            "combined_embedding_parameters": EMBEDDING_PARAMETER_COUNT,
            "shared_lorentz_batch_norm_gamma_parameters": 1,
            "reported_and_trainable_parameters": EXPECTED_PARAMETER_COUNT,
            "same_parameter_count_at_every_depth": True,
        },
        "paired_parameter_budget": {
            "lhgcn_reported_total": EXPECTED_PARAMETER_COUNT,
            "lhgcn_trainable_total": EXPECTED_PARAMETER_COUNT,
            "lhgcn_extra_scalar": "shared LorentzBatchNorm.gamma",
            "companion_sl8_reported_total": EXPECTED_PARAMETER_COUNT,
            "companion_sl8_trainable_total": EMBEDDING_PARAMETER_COUNT,
            "companion_sl8_nontrainable_scalar": "fixed log_score_scale",
            "lhgcn_minus_sl8_trainable_parameters": 1,
        },
        "anchor": anchor,
        "winner": ranking[0] if complete and ranking else None,
        "provisional_winner": ranking[0] if ranking else None,
        "ranking": ranking,
        "paired_sl8_comparison": {
            "source": sl8_source,
            "completed_same_cell_pairs": len(paired),
            "delta_convention": "positive delta_sl8_minus_lhgcn favors SL8",
        },
        "runtime": {
            "known_trial_seconds": sum(durations),
            "known_trial_hours": sum(durations) / 3600.0,
            "trials_with_known_duration": len(durations),
        },
        "split_fingerprints": EXPECTED_SPLIT_FINGERPRINTS,
        "checkpoint_acceptance": {
            "every_completed_trial_passed": True,
            "criterion": (
                "exact config; one 25226x64 combined embedding; one shared "
                "LorentzBatchNorm gamma; exact lGCN buffers; no extra state keys"
            ),
        },
        "failures": dict(failures),
        "test_evaluated": False,
    }
    _atomic_json(path, payload)
    return payload


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
        "single_physical_gpu": args.gpu_id,
        "child_cuda_visible_devices": args.gpu_id,
        "child_config_gpu_id": args.gpu_id,
        "child_torch_device_after_mask": "cuda:0",
        "shared_sl8_lhgcn_lock_file": str(args.lock_file),
        "strict_serial": True,
        "sl8_summary": str(args.sl8_summary) if args.sl8_summary else None,
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


def execute(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
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
                    raise
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    print(f"IGNORE RAW {trial.name}: {error}")
                else:
                    if paths["failure"].is_file():
                        paths["failure"].unlink()
                    print(
                        f"RECOVER {trial.name}: Recall@10="
                        f"{float(recovered['best_valid_result']['recall@10']):.6f}"
                    )
                    write_summary(
                        summary_path,
                        repo=args.repo,
                        output_root=args.output_root,
                        protocol=protocol,
                        sl8_summary_path=args.sl8_summary,
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
                    sl8_summary_path=args.sl8_summary,
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
            command = trial_command(args, trial, paths["raw"], paths["checkpoint_dir"])
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
                    sl8_summary_path=args.sl8_summary,
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
                sl8_summary_path=args.sl8_summary,
                failures=_load_failures(args.output_root),
            )

    write_summary(
        summary_path,
        repo=args.repo,
        output_root=args.output_root,
        protocol=protocol,
        sl8_summary_path=args.sl8_summary,
        failures=_load_failures(args.output_root),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--data-root", type=Path, default=repo / "dataset")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--gpu-id",
        default="7",
        help="one physical CUDA index; defaults to the authorised GPU 7",
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        help="override the lock shared with the companion SL8 grid",
    )
    parser.add_argument("--max-new-trials", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--deep-data-audit", action="store_true")
    parser.add_argument(
        "--skip-data-audit",
        action="store_true",
        help="only valid with --dry-run on a planning machine without Toy data",
    )
    parser.add_argument(
        "--reuse-l4-b65536-result",
        type=Path,
        help=(
            "adopt an exact legacy anchor only after its checkpoint and exact "
            "split fingerprints pass the current contract"
        ),
    )
    parser.add_argument(
        "--sl8-summary",
        type=Path,
        help="companion SL8 grid summary used for same-layer/same-batch deltas",
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
    args.sl8_summary = (
        args.sl8_summary.expanduser().resolve()
        if args.sl8_summary is not None
        else None
    )
    if args.skip_data_audit and not args.dry_run:
        raise ValueError("--skip-data-audit is permitted only with --dry-run")
    protocol = validate_protocol(args.repo)
    audit = _audit_data(args)
    if args.dry_run:
        print(json.dumps(_dry_run_plan(args, protocol, audit), indent=2))
        return 0
    execute(args, protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
