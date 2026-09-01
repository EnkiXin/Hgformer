#!/usr/bin/env python3
"""Resumable single-GPU multi-fidelity search for Amazon-CD SL8-LHGCN.

This driver deliberately has no test-evaluation path.  It first trains the
complete requested 96-point grid for 50 epochs and performs exactly one
full-ranking validation at epoch 50.  The best four configurations from each
GCN depth advance (16 total), so the cheap screen cannot discard an entire
depth merely because one depth is temporarily favoured at low fidelity.

The 16 configurations are trained *from scratch* for 100 epochs and validated
once.  Their 50/100-epoch rank Spearman correlation controls the next budget:
eight advance when rho < 0.5, otherwise four.  Survivors are trained from
scratch for 200 epochs with one validation, the best two are then trained from
scratch for 500 epochs and validated every 10 epochs.

Resume is evidence-based.  A result is skipped only after its JSON metadata,
split fingerprints, metrics, model diagnostics, checkpoint hash, checkpoint
configuration, and checkpoint epoch all satisfy the exact current contract.
The parent holds an exclusive physical-GPU lock for the complete serial run.
The same physical index is passed in CUDA_VISIBLE_DEVICES and RecBole's
``gpu_id`` because this vendored RecBole rewrites the CUDA mask at startup.
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


SCHEMA_VERSION = 1
DATASET = "Amazon_cd"
SEED = 2024
BASE_CONFIG_NAME = "RecFormer_cd.yaml"
MODEL_OVERLAY_NAME = "SL8LHGCN_reproduction.yaml"

LAYERS = (0, 2, 4, 6)
BATCH_SIZES = (16_384, 32_768, 65_536, 131_072)
LEARNING_RATES = (0.0005, 0.001, 0.005)
LOSS_MARGINS = (0.1, 0.3)
SCHATTEN_P = 2

# End-to-end full-ranking measurements favour RecBole's 17-user outer batch
# and 1,024-item scorer chunks.  ``eval_user_chunk_size=64`` is deliberately
# above the effective outer batch, so it adds no further user subdivision.
# Microbenchmarks of isolated scorer shapes did not predict evaluator time.
EVAL_BATCH_SIZE = 1_048_576  # floor(1_048_576 / 58_869) == 17 users
EVAL_USER_CHUNK_SIZE = 64
EVAL_ITEM_CHUNK_SIZE = 1024
STOPPING_STEP = 1000

EXPECTED_SPLIT_INTERACTIONS = {
    "train": 746_199,
    "valid": 103_174,
    "test": 103_174,
}


@dataclass(frozen=True, order=True)
class Parameters:
    gcn_layers: int
    train_batch_size: int
    learning_rate: float
    loss_margin: float
    schatten_p: int = SCHATTEN_P

    def validate(self) -> None:
        if self.gcn_layers not in LAYERS:
            raise ValueError(f"gcn_layers must be one of {LAYERS}")
        if self.train_batch_size not in BATCH_SIZES:
            raise ValueError(f"train_batch_size must be one of {BATCH_SIZES}")
        if self.learning_rate not in LEARNING_RATES:
            raise ValueError(f"learning_rate must be one of {LEARNING_RATES}")
        if self.loss_margin not in LOSS_MARGINS:
            raise ValueError(f"loss_margin must be one of {LOSS_MARGINS}")
        if self.schatten_p != SCHATTEN_P:
            raise ValueError(f"schatten_p is fixed to {SCHATTEN_P}")

    @property
    def name(self) -> str:
        self.validate()
        lr = f"{self.learning_rate:.4g}".replace(".", "p")
        margin = f"{self.loss_margin:.3g}".replace(".", "p")
        return (
            f"L{self.gcn_layers:02d}_B{self.train_batch_size:06d}"
            f"_LR{lr}_M{margin}_P{self.schatten_p}"
        )


@dataclass(frozen=True)
class Stage:
    name: str
    epochs: int
    eval_step: int

    @property
    def validation_events(self) -> int:
        return self.epochs // self.eval_step


SCREEN_50 = Stage("screen50", 50, 50)
RERANK_100 = Stage("rerank100", 100, 100)
RERANK_200 = Stage("rerank200", 200, 200)
FINAL_500 = Stage("final500", 500, 10)
STAGES = (SCREEN_50, RERANK_100, RERANK_200, FINAL_500)


def full_grid() -> tuple[Parameters, ...]:
    return tuple(
        Parameters(layer, batch, lr, margin)
        for layer, batch, lr, margin in itertools.product(
            LAYERS, BATCH_SIZES, LEARNING_RATES, LOSS_MARGINS
        )
    )


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
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
    """Fail closed if the Hgformer split/evaluation or SL8 contract changed."""

    base_path, overlay_path = config_paths(repo)
    documents: list[dict[str, Any]] = []
    merged: dict[str, Any] = {}
    for path in (base_path, overlay_path):
        if not path.is_file():
            raise FileNotFoundError(f"required config does not exist: {path}")
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"config is not a mapping: {path}")
        documents.append(document)
        merged.update(document)

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
            "SL8 overlay may not change the Amazon-CD data/evaluation protocol: "
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
        "sl_gcn_mode": "ambient_retract",
        "lhgcn_include_self": False,
        "lhgcn_layer_aggregation": "last",
        "sl_layer_norm": "none",
        "sl_membership_check": True,
        "sl_membership_strict": True,
        "pairwise_loss": "lhgcn_hinge_squared_sum",
        "learnable_score_scale": False,
        "neg_sampling": {"uniform": 1},
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
    differences = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value
    }
    if differences:
        raise RuntimeError(f"Amazon-CD SL8-LHGCN protocol changed: {differences}")

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "seed": SEED,
        "base_config": BASE_CONFIG_NAME,
        "base_config_sha256": _sha256(base_path),
        "model_overlay": MODEL_OVERLAY_NAME,
        "model_overlay_sha256": _sha256(overlay_path),
        "filters": {"rating": "[3,inf)", "users": "[5,inf)", "items": "[5,inf)"},
        "validation": {
            "split": expected["eval_args"],
            "expected_interactions": EXPECTED_SPLIT_INTERACTIONS,
            "metrics": expected["metrics"],
            "topk": expected["topk"],
            "selection": ["Recall@10 descending", "NDCG@10 descending"],
            "mode": "full",
            "test_evaluated": False,
        },
        "model": {
            "name": "SL8LHGCN",
            "matrix_dim": 8,
            "gcn_mode": "ambient_retract",
            "loss": "lhgcn_hinge_squared_sum",
            "schatten_p": SCHATTEN_P,
            "fast_one_sided_frobenius": True,
        },
        "exact_full_sort_chunks": {
            "eval_batch_size": EVAL_BATCH_SIZE,
            "eval_user_chunk_size": EVAL_USER_CHUNK_SIZE,
            "eval_item_chunk_size": EVAL_ITEM_CHUNK_SIZE,
            "sampling": None,
        },
    }


def manifest(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "amazon-cd-sl8lhgcn-multifidelity-grid",
        "protocol": protocol,
        "grid": {
            "gcn_layers": list(LAYERS),
            "train_batch_size": list(BATCH_SIZES),
            "learning_rate": list(LEARNING_RATES),
            "loss_margin": list(LOSS_MARGINS),
            "schatten_p": [SCHATTEN_P],
            "cartesian_trials": len(full_grid()),
        },
        "stages": [
            {
                "name": SCREEN_50.name,
                "epochs": 50,
                "eval_step": 50,
                "validation_events_per_trial": 1,
                "trials": 96,
                "advance": "best four within each of L0/L2/L4/L6 (16 total)",
            },
            {
                "name": RERANK_100.name,
                "epochs": 100,
                "eval_step": 100,
                "validation_events_per_trial": 1,
                "trials": 16,
                "advance": "top 8 if 50/100 rank Spearman < 0.5, otherwise top 4",
            },
            {
                "name": RERANK_200.name,
                "epochs": 200,
                "eval_step": 200,
                "validation_events_per_trial": 1,
                "trials": "4 or 8",
                "advance": "top 2",
            },
            {
                "name": FINAL_500.name,
                "epochs": 500,
                "eval_step": 10,
                "validation_events_per_trial": 50,
                "trials": 2,
                "advance": "winner",
            },
        ],
        "totals": {
            "fresh_training_runs": {"minimum": 118, "maximum": 122},
            "trial_epochs": {"minimum": 8200, "maximum": 9000},
            "exact_full_ranking_validations": {"minimum": 216, "maximum": 220},
        },
        "selection_split": "validation only",
        "held_out_test_evaluated": False,
    }


def rank_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            -float(item["recall@10"]),
            -float(item["ndcg@10"]),
            str(item["trial"]),
        ),
    )


def select_diverse_top16(candidates: Sequence[Mapping[str, Any]]) -> tuple[Parameters, ...]:
    """Take exactly four best trials per GCN depth, then globally order them."""

    selected: list[Mapping[str, Any]] = []
    for layer in LAYERS:
        layer_candidates = [item for item in candidates if int(item["gcn_layers"]) == layer]
        if len(layer_candidates) < 4:
            raise ValueError(f"need at least four completed candidates for layer {layer}")
        selected.extend(rank_candidates(layer_candidates)[:4])
    return tuple(_parameters_from_candidate(item) for item in rank_candidates(selected))


def spearman_rank(
    earlier: Sequence[Mapping[str, Any]], later: Sequence[Mapping[str, Any]]
) -> float:
    """Spearman correlation of deterministic total rankings over the same trials."""

    first = [str(item["trial"]) for item in rank_candidates(earlier)]
    second = [str(item["trial"]) for item in rank_candidates(later)]
    if len(first) != len(second) or set(first) != set(second):
        raise ValueError("Spearman rankings must contain the same unique trials")
    if len(first) < 2 or len(set(first)) != len(first):
        raise ValueError("Spearman correlation requires at least two unique trials")
    second_rank = {name: index for index, name in enumerate(second, start=1)}
    squared = sum(
        (index - second_rank[name]) ** 2
        for index, name in enumerate(first, start=1)
    )
    count = len(first)
    return 1.0 - 6.0 * squared / (count * (count * count - 1))


def adaptive_survivor_count(rank_spearman: float) -> int:
    if not math.isfinite(rank_spearman) or not -1.0 <= rank_spearman <= 1.0:
        raise ValueError("rank Spearman must be finite and lie in [-1, 1]")
    return 8 if rank_spearman < 0.5 else 4


def _parameters_from_candidate(candidate: Mapping[str, Any]) -> Parameters:
    values = candidate["parameters"]
    if not isinstance(values, Mapping):
        raise ValueError("candidate parameters are missing")
    result = Parameters(
        gcn_layers=int(values["gcn_layers"]),
        train_batch_size=int(values["train_batch_size"]),
        learning_rate=float(values["learning_rate"]),
        loss_margin=float(values["loss_margin"]),
        schatten_p=int(values["schatten_p"]),
    )
    result.validate()
    return result


def result_paths(output_root: Path, stage: Stage, parameters: Parameters) -> dict[str, Path]:
    root = output_root / "stages" / stage.name
    return {
        "result": root / "results" / f"{parameters.name}.json",
        "raw": root / "work" / f"{parameters.name}.raw.json",
        "log": root / "logs" / f"{parameters.name}.log",
        "checkpoint_dir": root / "checkpoints" / parameters.name,
        "failure": root / "failures" / f"{parameters.name}.json",
        "summary": root / "summary.json",
    }


def _selection_signature(parameters: Sequence[Parameters]) -> str:
    return _canonical_hash([asdict(item) for item in parameters])


def trial_metadata(
    stage: Stage,
    parameters: Parameters,
    protocol: Mapping[str, Any],
    gpu_id: str,
    parent_parameters: Sequence[Parameters],
) -> dict[str, Any]:
    core = {
        "schema_version": SCHEMA_VERSION,
        "kind": "amazon-cd-sl8lhgcn-multifidelity-trial",
        "stage": asdict(stage),
        "parameters": asdict(parameters),
        "parent_selection_sha256": _selection_signature(parent_parameters),
        "protocol": protocol,
        "physical_gpu_id": gpu_id,
        "fresh_training": True,
        "test_evaluated": False,
    }
    return {**core, "signature_sha256": _canonical_hash(core)}


def trial_command(
    args: argparse.Namespace,
    stage: Stage,
    parameters: Parameters,
    raw_result: Path,
    checkpoint_dir: Path,
) -> list[str]:
    parameters.validate()
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
        f"--gpu_id={args.gpu_id}",
        "--use_gpu=true",
        "--show_progress=false",
        f"--seed={SEED}",
        f"--epochs={stage.epochs}",
        f"--eval_step={stage.eval_step}",
        f"--stopping_step={STOPPING_STEP}",
        f"--gcn_layers={parameters.gcn_layers}",
        f"--n_layers={parameters.gcn_layers}",
        f"--train_batch_size={parameters.train_batch_size}",
        f"--learning_rate={parameters.learning_rate:.12g}",
        f"--loss_margin={parameters.loss_margin:.12g}",
        f"--schatten_p={parameters.schatten_p}",
        f"--eval_batch_size={EVAL_BATCH_SIZE}",
        f"--eval_user_chunk_size={EVAL_USER_CHUNK_SIZE}",
        f"--eval_item_chunk_size={EVAL_ITEM_CHUNK_SIZE}",
        f"--full_sort_user_batch_size={EVAL_USER_CHUNK_SIZE}",
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
        "--sl_score_mode=group_log",
        "--log_terms=12",
        "--log_jitter=0.0",
        "--symmetric_distance=false",
        "--fast_one_sided_frobenius=true",
        "--score_scale=1.0",
        "--learnable_score_scale=false",
        "--pairwise_loss=lhgcn_hinge_squared_sum",
        "--learner=adam",
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


def _load_checkpoint(path: Path, repo: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
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
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"checkpoint has no model state: {path}")
    return _config_dictionary(checkpoint["config"]), checkpoint


def expected_checkpoint_values(stage: Stage, parameters: Parameters) -> dict[str, Any]:
    return {
        "model": "SL8LHGCN",
        "dataset": DATASET,
        "seed": SEED,
        "reproducibility": True,
        "epochs": stage.epochs,
        "eval_step": stage.eval_step,
        "stopping_step": STOPPING_STEP,
        "gcn_layers": parameters.gcn_layers,
        "n_layers": parameters.gcn_layers,
        "train_batch_size": parameters.train_batch_size,
        "learning_rate": parameters.learning_rate,
        "loss_margin": parameters.loss_margin,
        "schatten_p": parameters.schatten_p,
        "embedding_size": 64,
        "matrix_dim": 8,
        "num_factors": 1,
        "factor_aggregation": "l2",
        "embedding_init": "xavier_uniform_combined",
        "coord_clip": 0.75,
        "sl_gcn_mode": "ambient_retract",
        "lhgcn_include_self": False,
        "lhgcn_layer_aggregation": "last",
        "sl_layer_norm": "none",
        "sl_membership_check": True,
        "sl_membership_strict": True,
        "sl_score_mode": "group_log",
        "pairwise_loss": "lhgcn_hinge_squared_sum",
        "fast_one_sided_frobenius": True,
        "symmetric_distance": False,
        "learnable_score_scale": False,
        "neg_sampling": {"uniform": 1},
        "eval_batch_size": EVAL_BATCH_SIZE,
        "eval_user_chunk_size": EVAL_USER_CHUNK_SIZE,
        "eval_item_chunk_size": EVAL_ITEM_CHUNK_SIZE,
        "full_sort_user_batch_size": EVAL_USER_CHUNK_SIZE,
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


def validate_checkpoint_contract(
    path: Path,
    *,
    repo: Path,
    stage: Stage,
    parameters: Parameters,
    gpu_id: str,
) -> tuple[int, str]:
    config, checkpoint = _load_checkpoint(path, repo)
    expected = expected_checkpoint_values(stage, parameters)
    mismatches = {
        key: {"expected": expected_value, "actual": config.get(key)}
        for key, expected_value in expected.items()
        if config.get(key) != expected_value
    }
    if str(config.get("gpu_id")) != gpu_id:
        mismatches["gpu_id"] = {"expected": gpu_id, "actual": config.get("gpu_id")}
    if mismatches:
        raise ValueError(f"checkpoint trial contract mismatch: {mismatches}")

    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int):
        raise ValueError("checkpoint epoch is missing")
    if stage.validation_events == 1:
        if epoch != stage.epochs - 1:
            raise ValueError(
                f"single-validation checkpoint epoch {epoch} != {stage.epochs - 1}"
            )
    elif not (0 <= epoch < stage.epochs and (epoch + 1) % stage.eval_step == 0):
        raise ValueError(f"checkpoint epoch {epoch} is not an evaluation epoch")
    return epoch, _sha256(path)


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


def _validate_manifold_diagnostics(payload: Mapping[str, Any], parameters: Parameters) -> None:
    diagnostics = payload.get("model_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("missing model manifold diagnostics")
    if diagnostics.get("mode") != "ambient_retract":
        raise ValueError("wrong manifold propagation mode in diagnostics")
    if int(diagnostics.get("layers", -1)) != parameters.gcn_layers:
        raise ValueError("manifold diagnostic layer count mismatch")
    for field in (
        "active_singular_fallbacks",
        "output_membership_violations",
        "nonpositive_output_determinants",
        "nonfinite_output_log_determinants",
    ):
        if int(diagnostics.get(field, -1)) != 0:
            raise ValueError(f"manifold diagnostic {field} is not zero")
    layers = diagnostics.get("layer_membership")
    if not isinstance(layers, list) or len(layers) != parameters.gcn_layers:
        raise ValueError("missing every-layer SL(8) membership diagnostics")
    distance = diagnostics.get("distance_membership")
    if not isinstance(distance, Mapping):
        raise ValueError("missing SL(8) distance diagnostics")
    if int(distance.get("relative_membership_violations", -1)) != 0:
        raise ValueError("relative score matrices left SL(8)")
    if int(distance.get("nonfinite_approximate_logs", -1)) != 0:
        raise ValueError("non-finite matrix-log approximation")


def validate_result(
    payload: Mapping[str, Any],
    *,
    repo: Path,
    stage: Stage,
    parameters: Parameters,
    protocol: Mapping[str, Any],
    gpu_id: str,
    parent_parameters: Sequence[Parameters],
    checkpoint_dir: Path,
    require_metadata: bool,
) -> dict[str, Any]:
    if payload.get("model") != "SL8LHGCN" or payload.get("dataset") != DATASET:
        raise ValueError("wrong model or dataset in result")
    if int(payload.get("seed", -1)) != SEED:
        raise ValueError("wrong seed in result")
    if int(payload.get("epochs", -1)) != stage.epochs:
        raise ValueError("wrong epoch budget in result")
    if int(payload.get("stopping_step", -1)) != STOPPING_STEP:
        raise ValueError("wrong stopping_step in result")
    if payload.get("test_result") is not None:
        raise RuntimeError("multi-fidelity result touched the held-out test split")
    names = payload.get("config_files")
    if not isinstance(names, list) or [Path(str(item)).name for item in names] != [
        BASE_CONFIG_NAME,
        MODEL_OVERLAY_NAME,
    ]:
        raise ValueError("result used different config overlays")

    metrics = payload.get("best_valid_result")
    if not isinstance(metrics, Mapping):
        raise ValueError("missing validation metrics")
    recall = _finite_metric(metrics, "recall@10")
    ndcg = _finite_metric(metrics, "ndcg@10")
    score = payload.get("best_valid_score")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError("missing finite best validation score")
    if not math.isclose(float(score), recall, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("best_valid_score is not Recall@10")
    split = validate_split_fingerprints(payload)
    _validate_manifold_diagnostics(payload, parameters)

    checkpoint_token = payload.get("checkpoint_file")
    if not isinstance(checkpoint_token, str) or not checkpoint_token:
        raise ValueError("missing checkpoint path")
    checkpoint = Path(checkpoint_token).expanduser().resolve()
    expected_root = checkpoint_dir.expanduser().resolve()
    if checkpoint.parent != expected_root:
        raise ValueError(
            f"checkpoint is outside this trial directory: {checkpoint} != {expected_root}"
        )
    checkpoint_epoch, checkpoint_sha256 = validate_checkpoint_contract(
        checkpoint,
        repo=repo,
        stage=stage,
        parameters=parameters,
        gpu_id=gpu_id,
    )

    expected_metadata = trial_metadata(
        stage, parameters, protocol, gpu_id, parent_parameters
    )
    if require_metadata:
        metadata = payload.get("sl8_multifidelity")
        if metadata != expected_metadata:
            raise ValueError("resume metadata does not match the trial contract")
        artifact = payload.get("checkpoint_artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError("missing checkpoint artifact metadata")
        if artifact.get("sha256") != checkpoint_sha256:
            raise ValueError("checkpoint hash changed since result annotation")
        if int(artifact.get("size_bytes", -1)) != checkpoint.stat().st_size:
            raise ValueError("checkpoint size changed since result annotation")

    return {
        "recall@10": recall,
        "ndcg@10": ndcg,
        "split_fingerprints": split,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_sha256,
    }


def completed_result(
    path: Path,
    *,
    repo: Path,
    stage: Stage,
    parameters: Parameters,
    protocol: Mapping[str, Any],
    gpu_id: str,
    parent_parameters: Sequence[Parameters],
    checkpoint_dir: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        payload = _load_mapping(path)
        validate_result(
            payload,
            repo=repo,
            stage=stage,
            parameters=parameters,
            protocol=protocol,
            gpu_id=gpu_id,
            parent_parameters=parent_parameters,
            checkpoint_dir=checkpoint_dir,
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
    stage: Stage,
    parameters: Parameters,
    protocol: Mapping[str, Any],
    gpu_id: str,
    parent_parameters: Sequence[Parameters],
    checkpoint_dir: Path,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _load_mapping(raw_path)
    validated = validate_result(
        payload,
        repo=repo,
        stage=stage,
        parameters=parameters,
        protocol=protocol,
        gpu_id=gpu_id,
        parent_parameters=parent_parameters,
        checkpoint_dir=checkpoint_dir,
        require_metadata=False,
    )
    checkpoint = Path(str(payload["checkpoint_file"])).expanduser().resolve()
    payload["sl8_multifidelity"] = trial_metadata(
        stage, parameters, protocol, gpu_id, parent_parameters
    )
    payload["checkpoint_artifact"] = {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": validated["checkpoint_sha256"],
    }
    payload["multifidelity_runtime"] = dict(runtime)
    _atomic_json(final_path, payload)
    complete, reason = completed_result(
        final_path,
        repo=repo,
        stage=stage,
        parameters=parameters,
        protocol=protocol,
        gpu_id=gpu_id,
        parent_parameters=parent_parameters,
        checkpoint_dir=checkpoint_dir,
    )
    if complete is None:
        raise RuntimeError(f"annotated result failed its own resume audit: {reason}")
    return complete


def candidate(path: Path, parameters: Parameters, payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["best_valid_result"]
    runtime = payload.get("multifidelity_runtime") or {}
    return {
        "trial": parameters.name,
        "parameters": asdict(parameters),
        "gcn_layers": parameters.gcn_layers,
        "recall@10": float(metrics["recall@10"]),
        "ndcg@10": float(metrics["ndcg@10"]),
        "result_file": str(path.expanduser().resolve()),
        "checkpoint_file": payload["checkpoint_file"],
        "duration_seconds": runtime.get("duration_seconds"),
        "split_fingerprints": payload["split_fingerprints"],
        "test_evaluated": False,
    }


def collect_stage(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    stage: Stage,
    parameters: Sequence[Parameters],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    complete: list[dict[str, Any]] = []
    invalid: dict[str, str] = {}
    canonical_split: Mapping[str, Any] | None = None
    for item in parameters:
        paths = result_paths(args.output_root, stage, item)
        payload, reason = completed_result(
            paths["result"],
            repo=args.repo,
            stage=stage,
            parameters=item,
            protocol=protocol,
            gpu_id=args.gpu_id,
            parent_parameters=parameters,
            checkpoint_dir=paths["checkpoint_dir"],
        )
        if payload is None:
            if reason:
                invalid[item.name] = reason
            continue
        item_candidate = candidate(paths["result"], item, payload)
        split = item_candidate["split_fingerprints"]
        if canonical_split is None:
            canonical_split = split
        elif split != canonical_split:
            raise RuntimeError("completed trials use different data splits")
        complete.append(item_candidate)
    return complete, invalid


def write_stage_summary(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    stage: Stage,
    parameters: Sequence[Parameters],
    *,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    complete, invalid = collect_stage(args, protocol, stage, parameters)
    ranking = rank_candidates(complete)
    finished = len(complete) == len(parameters)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": asdict(stage),
        "state": "complete" if finished else "incomplete",
        "expected_trials": len(parameters),
        "completed_trials": len(complete),
        "pending_trials": sorted(
            set(item.name for item in parameters) - set(item["trial"] for item in complete)
        ),
        "invalid_results": invalid,
        "ranking": ranking,
        "selection": selection,
        "split_fingerprints": complete[0]["split_fingerprints"] if complete else None,
        "protocol": protocol,
        "test_evaluated": False,
    }
    path = args.output_root / "stages" / stage.name / "summary.json"
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
        / f"hgformer-sl8-cd-multifidelity-uid{os.getuid()}-gpu-{digest}.lock"
    )


@contextlib.contextmanager
def exclusive_gpu_lock(path: Path, gpu_id: str) -> Iterable[int]:
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


def run_stage(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    stage: Stage,
    parameters: Sequence[Parameters],
    *,
    lock_fd: int,
    remaining_launches: list[int | None],
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    for item in parameters:
        paths = result_paths(args.output_root, stage, item)
        complete, invalid_reason = completed_result(
            paths["result"],
            repo=args.repo,
            stage=stage,
            parameters=item,
            protocol=protocol,
            gpu_id=args.gpu_id,
            parent_parameters=parameters,
            checkpoint_dir=paths["checkpoint_dir"],
        )
        if complete is not None:
            print(f"SKIP {stage.name}/{item.name}: deeply validated result")
            continue

        # Recover a subprocess result written before annotation/summary work.
        if paths["raw"].is_file():
            try:
                recovered = annotate_result(
                    paths["raw"],
                    paths["result"],
                    repo=args.repo,
                    stage=stage,
                    parameters=item,
                    protocol=protocol,
                    gpu_id=args.gpu_id,
                    parent_parameters=parameters,
                    checkpoint_dir=paths["checkpoint_dir"],
                    runtime={
                        "source": "recovered-complete-raw-result",
                        "finished_at": _utc_now(),
                        "duration_seconds": None,
                    },
                )
            except RuntimeError:
                raise
            except (OSError, ValueError, json.JSONDecodeError) as error:
                print(f"IGNORE RAW {stage.name}/{item.name}: {error}")
            else:
                print(
                    f"RECOVER {stage.name}/{item.name}: "
                    f"Recall@10={float(recovered['best_valid_result']['recall@10']):.6f}"
                )
                continue

        if remaining_launches[0] is not None and remaining_launches[0] <= 0:
            break
        if remaining_launches[0] is not None:
            remaining_launches[0] -= 1
        paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
        paths["raw"].parent.mkdir(parents=True, exist_ok=True)
        command = trial_command(args, stage, item, paths["raw"], paths["checkpoint_dir"])
        started_at = _utc_now()
        started_clock = time.monotonic()
        print(f"START {stage.name}/{item.name}")
        try:
            _run_and_tee(
                command,
                log_path=paths["log"],
                cwd=args.repo,
                env=environment,
                lock_fd=lock_fd,
            )
            result = annotate_result(
                paths["raw"],
                paths["result"],
                repo=args.repo,
                stage=stage,
                parameters=item,
                protocol=protocol,
                gpu_id=args.gpu_id,
                parent_parameters=parameters,
                checkpoint_dir=paths["checkpoint_dir"],
                runtime={
                    "source": "runner-measured",
                    "started_at": started_at,
                    "finished_at": _utc_now(),
                    "duration_seconds": time.monotonic() - started_clock,
                    "invalid_existing_reason": invalid_reason,
                },
            )
            if paths["failure"].is_file():
                paths["failure"].unlink()
            print(
                f"DONE {stage.name}/{item.name}: "
                f"Recall@10={float(result['best_valid_result']['recall@10']):.6f}"
            )
        except (subprocess.CalledProcessError, OSError, ValueError) as error:
            _atomic_json(
                paths["failure"],
                {
                    "stage": stage.name,
                    "trial": item.name,
                    "parameters": asdict(item),
                    "started_at": started_at,
                    "failed_at": _utc_now(),
                    "duration_seconds": time.monotonic() - started_clock,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "test_evaluated": False,
                },
            )
            print(f"FAILED {stage.name}/{item.name}: {error}", file=sys.stderr)
            if not args.continue_on_error:
                # A fail-fast exit still leaves an auditable stage snapshot.
                write_stage_summary(args, protocol, stage, parameters)
                raise
    return write_stage_summary(args, protocol, stage, parameters)


def _attach_selection(
    args: argparse.Namespace,
    summary: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist a deterministic advancement decision without reloading checkpoints."""

    updated = dict(summary)
    updated["selection"] = dict(selection)
    stage_name = str(updated["stage"]["name"])
    _atomic_json(args.output_root / "stages" / stage_name / "summary.json", updated)
    return updated


def _write_global_summary(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    *,
    state: str,
    stage_summaries: Sequence[Mapping[str, Any]],
    spearman: float | None = None,
    adaptive_survivors: int | None = None,
) -> None:
    final_ranking = (
        stage_summaries[-1].get("ranking", []) if stage_summaries else []
    )
    _atomic_json(
        args.output_root / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "amazon-cd-sl8lhgcn-multifidelity-summary",
            "state": state,
            "protocol": protocol,
            "rank_spearman_50_vs_100": spearman,
            "adaptive_200_epoch_survivors": adaptive_survivors,
            "stages": [
                {
                    "stage": item["stage"]["name"],
                    "state": item["state"],
                    "expected_trials": item["expected_trials"],
                    "completed_trials": item["completed_trials"],
                    "summary_file": str(
                        (args.output_root / "stages" / item["stage"]["name"] / "summary.json").resolve()
                    ),
                }
                for item in stage_summaries
            ],
            "winner": final_ranking[0] if state == "complete" and final_ranking else None,
            "provisional_winner": final_ranking[0] if final_ranking else None,
            "held_out_test_evaluated": False,
        },
    )


def execute(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    _atomic_json(args.output_root / "manifest.json", manifest(protocol))
    remaining_launches: list[int | None] = [args.max_new_trials]
    summaries: list[Mapping[str, Any]] = []
    rho: float | None = None
    survivor_count: int | None = None

    with exclusive_gpu_lock(args.lock_file, args.gpu_id) as lock_fd:
        screen_parameters = full_grid()
        screen = run_stage(
            args,
            protocol,
            SCREEN_50,
            screen_parameters,
            lock_fd=lock_fd,
            remaining_launches=remaining_launches,
        )
        summaries.append(screen)
        if screen["state"] != "complete":
            _write_global_summary(args, protocol, state="screen50-incomplete", stage_summaries=summaries)
            return

        top16 = select_diverse_top16(screen["ranking"])
        screen_selection = {
            "policy": "top four per gcn_layers value, then globally ranked",
            "advanced": [item.name for item in top16],
            "advanced_count": 16,
        }
        screen = _attach_selection(args, screen, screen_selection)
        summaries[-1] = screen

        rerank100 = run_stage(
            args,
            protocol,
            RERANK_100,
            top16,
            lock_fd=lock_fd,
            remaining_launches=remaining_launches,
        )
        summaries.append(rerank100)
        if rerank100["state"] != "complete":
            _write_global_summary(args, protocol, state="rerank100-incomplete", stage_summaries=summaries)
            return

        selected_names = {item.name for item in top16}
        screen_selected_ranking = [
            item for item in screen["ranking"] if item["trial"] in selected_names
        ]
        rho = spearman_rank(screen_selected_ranking, rerank100["ranking"])
        survivor_count = adaptive_survivor_count(rho)
        survivors = tuple(
            _parameters_from_candidate(item)
            for item in rank_candidates(rerank100["ranking"])[:survivor_count]
        )
        rerank100_selection = {
            "rank_spearman_50_vs_100": rho,
            "threshold": 0.5,
            "rule": "rho < 0.5 advances 8; rho >= 0.5 advances 4",
            "advanced_count": survivor_count,
            "advanced": [item.name for item in survivors],
        }
        rerank100 = _attach_selection(args, rerank100, rerank100_selection)
        summaries[-1] = rerank100

        rerank200 = run_stage(
            args,
            protocol,
            RERANK_200,
            survivors,
            lock_fd=lock_fd,
            remaining_launches=remaining_launches,
        )
        summaries.append(rerank200)
        if rerank200["state"] != "complete":
            _write_global_summary(
                args,
                protocol,
                state="rerank200-incomplete",
                stage_summaries=summaries,
                spearman=rho,
                adaptive_survivors=survivor_count,
            )
            return

        top2 = tuple(
            _parameters_from_candidate(item)
            for item in rank_candidates(rerank200["ranking"])[:2]
        )
        rerank200_selection = {
            "policy": "top two by Recall@10, NDCG@10 tie-break",
            "advanced_count": 2,
            "advanced": [item.name for item in top2],
        }
        rerank200 = _attach_selection(args, rerank200, rerank200_selection)
        summaries[-1] = rerank200

        final500 = run_stage(
            args,
            protocol,
            FINAL_500,
            top2,
            lock_fd=lock_fd,
            remaining_launches=remaining_launches,
        )
        summaries.append(final500)
        _write_global_summary(
            args,
            protocol,
            state="complete" if final500["state"] == "complete" else "final500-incomplete",
            stage_summaries=summaries,
            spearman=rho,
            adaptive_survivors=survivor_count,
        )


def _dry_run_plan(args: argparse.Namespace, protocol: Mapping[str, Any]) -> dict[str, Any]:
    examples = []
    for parameters in full_grid():
        paths = result_paths(args.output_root, SCREEN_50, parameters)
        examples.append(
            {
                "stage": SCREEN_50.name,
                "trial": parameters.name,
                "parameters": asdict(parameters),
                "command": trial_command(
                    args, SCREEN_50, parameters, paths["raw"], paths["checkpoint_dir"]
                ),
            }
        )
    return {
        "dry_run": True,
        "manifest": manifest(protocol),
        "physical_gpu_id": args.gpu_id,
        "CUDA_VISIBLE_DEVICES": args.gpu_id,
        "RecBole_gpu_id": args.gpu_id,
        "strict_serial": True,
        "lock_file": str(args.lock_file),
        "initial_96_jobs": examples,
        "later_jobs": "data-dependent; deterministic selection policies are in manifest",
        "held_out_test_evaluated": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--data-root", type=Path, default=repo / "dataset")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument(
        "--max-new-trials",
        type=int,
        help="launch at most this many new subprocesses, then exit resumably",
    )
    parser.add_argument("--continue-on-error", action="store_true")
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
    protocol = validate_protocol(args.repo)
    if args.dry_run:
        print(json.dumps(_dry_run_plan(args, protocol), indent=2))
        return 0
    execute(args, protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
