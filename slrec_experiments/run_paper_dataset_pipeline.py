#!/usr/bin/env python3
"""Single-physical-GPU pipeline for the six Hgformer paper datasets.

The pipeline is deliberately stricter than a collection of shell loops:

* every dataset inherits filtering, seed, user-wise 8:1:1 split, metrics and
  full-ranking evaluation from its released ``RecFormer_*.yaml``; Amazon Book
  applies the explicit 8-core correction required to reproduce the paper's
  reported cardinalities (the released YAML accidentally says 5-core);
* physical GPU 7 is written both to ``CUDA_VISIBLE_DEVICES`` and to every
  RecBole training child's ``--gpu_id`` argument (the vendored RecBole resets
  the former from the latter);
* LightGCN, matched LHGCN and fixed RecFormer are validation-only controls;
* SL8/SL16-LHGCN are searched independently, either with a practical blocked
  factorial (layer x batch, then geometry x margin) or a full Cartesian grid;
* resumability is based on exact protocol/job metadata, checkpoints and split
  fingerprints, rather than merely on a filename existing; and
* held-out test evaluation is available only through the separate
  ``--phase final-test`` checkpoint-evaluation path.

Default dataset order is Amazon CD, Amazon Movies, Amazon Book, then Douban
Book/Movie/Music.  ``--phase all`` never touches test.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import itertools
import math
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

try:
    from slrec_experiments.run_multidataset_sl8 import (
        DATASET_BY_SLUG,
        DatasetSpec,
        audit_filtered_dataset,
        audit_source_file,
        config_path,
        validate_fixed_protocol,
    )
except ModuleNotFoundError:  # Direct ``python slrec_experiments/<file>.py``.
    from run_multidataset_sl8 import (  # type: ignore
        DATASET_BY_SLUG,
        DatasetSpec,
        audit_filtered_dataset,
        audit_source_file,
        config_path,
        validate_fixed_protocol,
    )


SCHEMA_VERSION = 1
PHYSICAL_GPU = "7"
SL16_MODEL_NAME = "SL16LHGCN"
SL_MODEL_NAMES = {8: "SL8LHGCN", 16: SL16_MODEL_NAME}
SL_OVERLAYS = {
    8: "SL8LHGCN_reproduction.yaml",
    16: "SL16LHGCN_reproduction.yaml",
}
SEED = 2024
SELECTION_METRIC = "Recall@10 on full-ranking validation"
EPOCHS = 500
EVAL_STEP = 50
STOPPING_STEP = 1000

PAPER_DATASET_SLUGS = (
    "amazon-cd",
    "amazon-movies",
    "amazon-book",
    "douban-book",
    "douban-movie",
    "douban-music",
)
PAPER_DATASETS: tuple[DatasetSpec, ...] = tuple(
    DATASET_BY_SLUG[slug] for slug in PAPER_DATASET_SLUGS
)
PAPER_DATASET_BY_SLUG = {spec.slug: spec for spec in PAPER_DATASETS}
PAPER_DATASET_BY_NAME = {spec.dataset.lower(): spec for spec in PAPER_DATASETS}

LIGHTGCN_OVERLAY = "LightGCN_matched.yaml"
LHGCN_OVERLAY = "LHGCN_reproduction.yaml"
SL8_OVERLAY = "SL8LHGCN_reproduction.yaml"
AMAZON_BOOK_PROTOCOL_OVERLAY = "PaperProtocol_amazon_book_8core.yaml"

PAPER_PROTOCOL_OVERLAYS = {
    "amazon-book": AMAZON_BOOK_PROTOCOL_OVERLAY,
}

AMAZON_BOOK_PAPER_COUNTS = {
    "framework_users": 211_170,
    "framework_items": 163_789,
    "token_users": 211_169,
    "token_items": 163_788,
    "interactions": 5_069_747,
}

HISTORICAL_LIGHTGCN_CONFIGS = {
    "amazon-cd": "Baseline_cd.yaml",
    "amazon-movies": "Baseline_movie.yaml",
    "amazon-book": "Baseline_book.yaml",
    "douban-book": "Baseline_doubanbook.yaml",
    "douban-movie": "Baseline_doubanmovie.yaml",
    "douban-music": "Baseline_doubanmusic.yaml",
}

PROTECTED_DATA_KEYS = {
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

LAYERS = (2, 4, 6, 8, 10)
BATCH_SIZES = (32_768, 65_536, 131_072)
SL_BATCH_SIZES = {
    8: BATCH_SIZES,
    16: (4_096, 8_192, 16_384),
}
SL_EVAL_CHUNKS = {
    8: {"eval_batch_size": 1_048_576, "users": 64, "items": 1024},
    16: {"eval_batch_size": 262_144, "users": 16, "items": 256},
}


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


def _yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return payload


def overlay_path(repo: Path, name: str) -> Path:
    return repo / "baseline_config_fixed" / name


def historical_lightgcn_path(repo: Path, spec: DatasetSpec) -> Path:
    return repo / "baseline_config_fixed" / HISTORICAL_LIGHTGCN_CONFIGS[spec.slug]


def protocol_config_files(repo: Path, spec: DatasetSpec) -> tuple[Path, ...]:
    paths = [config_path(repo, spec)]
    overlay = PAPER_PROTOCOL_OVERLAYS.get(spec.slug)
    if overlay is not None:
        paths.append(overlay_path(repo, overlay))
    return tuple(paths)


def sl_model_config_files(
    repo: Path, spec: DatasetSpec, matrix_dim: int
) -> tuple[Path, ...]:
    if matrix_dim == 8:
        overlays = (SL_OVERLAYS[8],)
    elif matrix_dim == 16:
        # SL16 is a dimension-only subclass/overlay; shared SL-LHGCN semantics
        # remain owned by the SL8 reproduction overlay immediately before it.
        overlays = (SL_OVERLAYS[8], SL_OVERLAYS[16])
    else:
        raise ValueError(f"unsupported special-linear dimension: {matrix_dim}")
    return (
        *protocol_config_files(repo, spec),
        *(overlay_path(repo, name) for name in overlays),
    )


def _normalise_gpu(value: str) -> str:
    token = str(value).strip()
    if not token.isdigit():
        raise ValueError("--gpu-id must be one non-negative physical CUDA index")
    canonical = str(int(token))
    if canonical != PHYSICAL_GPU:
        raise ValueError(
            f"this controlled pipeline is authorised only on physical GPU {PHYSICAL_GPU}; "
            f"got {canonical}"
        )
    return canonical


def select_paper_datasets(tokens: Sequence[str]) -> tuple[DatasetSpec, ...]:
    flattened = [part.strip() for token in tokens for part in token.split(",") if part.strip()]
    if not flattened or "all" in {part.lower() for part in flattened}:
        return PAPER_DATASETS
    selected: list[DatasetSpec] = []
    seen: set[str] = set()
    for token in flattened:
        key = token.lower()
        spec = PAPER_DATASET_BY_SLUG.get(key) or PAPER_DATASET_BY_NAME.get(key)
        if spec is None:
            raise ValueError(
                f"unknown/non-paper dataset {token!r}; choices={list(PAPER_DATASET_SLUGS)}"
            )
        if spec.slug not in seen:
            selected.append(spec)
            seen.add(spec.slug)
    return tuple(selected)


@dataclass(frozen=True)
class LightGCNParameters:
    embedding_size: int
    n_layers: int
    reg_weight: float
    require_pow: bool
    learning_rate: float
    train_batch_size: int
    learner: str
    weight_decay: float


@dataclass(frozen=True)
class LHGCNParameters:
    embedding_size: int
    gcn_layers: int
    curve: float
    scale: float
    margin: float
    learning_rate: float
    train_batch_size: int
    learner: str
    weight_decay: float = 0.0


@dataclass(frozen=True)
class SL8Parameters:
    gcn_layers: int
    train_batch_size: int
    matrix_dim: int = 8
    learning_rate: float = 5e-4
    loss_margin: float = 0.1
    coord_clip: float = 0.75
    schatten_p: int | str = 2
    weight_decay: float = 0.0
    lhgcn_include_self: bool = False
    lhgcn_self_loop_weight: float = 1.0
    sl_scale: float = 1.0
    negative_samples: int = 1

    def validate(self) -> None:
        if self.gcn_layers not in LAYERS:
            raise ValueError(f"gcn_layers must be one of {LAYERS}")
        if self.matrix_dim not in SL_MODEL_NAMES:
            raise ValueError(f"matrix_dim must be one of {tuple(SL_MODEL_NAMES)}")
        allowed_batches = SL_BATCH_SIZES[self.matrix_dim]
        if self.train_batch_size not in allowed_batches:
            raise ValueError(
                f"train_batch_size must be one of {allowed_batches} for SL({self.matrix_dim})"
            )
        positive = {
            "learning_rate": self.learning_rate,
            "coord_clip": self.coord_clip,
            "sl_scale": self.sl_scale,
        }
        for key, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not math.isfinite(self.loss_margin) or self.loss_margin < 0:
            raise ValueError("loss_margin must be finite and non-negative")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.schatten_p not in {1, 2, 4, 8, "inf"}:
            raise ValueError(f"unsupported Schatten order: {self.schatten_p!r}")
        if self.lhgcn_self_loop_weight <= 0:
            raise ValueError("self-loop weight must be positive")
        if not self.lhgcn_include_self and self.lhgcn_self_loop_weight != 1.0:
            raise ValueError("self-loop weight is canonicalised to 1.0 while loops are off")
        if self.negative_samples not in {1, 2, 4}:
            raise ValueError("negative_samples must be one of 1, 2, 4")


@dataclass(frozen=True)
class TuningStage:
    key: str
    values: tuple[Any, ...]


CURRENT_TUNING_STAGES = (
    TuningStage("learning_rate", (1e-4, 3e-4, 5e-4, 1e-3)),
    TuningStage("loss_margin", (0.05, 0.1, 0.2, 0.3)),
    TuningStage("coord_clip", (0.25, 0.5, 0.75, 1.0, 1.5)),
    TuningStage("schatten_p", (1, 2, 4, 8, "inf")),
    TuningStage("weight_decay", (0.0, 1e-5, 1e-4, 1e-3, 5e-3)),
    TuningStage(
        "self_loop",
        (
            (False, 1.0),
            (True, 0.1),
            (True, 0.5),
            (True, 1.0),
        ),
    ),
)

EXPANDED_TUNING_STAGES = (
    TuningStage(
        "learning_rate",
        (5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3),
    ),
    TuningStage("loss_margin", (0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)),
    TuningStage("coord_clip", (0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)),
    TuningStage("schatten_p", (1, 2, 4, 8, "inf")),
    TuningStage(
        "weight_decay", (0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2)
    ),
    TuningStage(
        "self_loop",
        (
            (False, 1.0),
            (True, 0.1),
            (True, 0.5),
            (True, 1.0),
        ),
    ),
    TuningStage("sl_scale", (0.5, 0.75, 1.0, 1.5, 2.0)),
    TuningStage("negative_samples", (1, 2, 4)),
)

TUNING_PROFILES = {
    "current": CURRENT_TUNING_STAGES,
    "expanded": EXPANDED_TUNING_STAGES,
}

SL8_GEOMETRY_VALUES = (1, 2, 4, 8, "inf")
SL8_MARGIN_VALUES = (0.05, 0.1, 0.2, 0.3)
LHGCN_CURVE_VALUES = (0.05, 0.1, 0.2, 0.5, 1.0)
LHGCN_MARGIN_VALUES = (0.05, 0.1, 0.2, 0.3, 0.5)


def _config_reads(path: Path) -> set[str]:
    """Statically collect literal ``config[key]``/``_config_get`` reads."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    reads: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id != "config":
                continue
            slice_node = node.slice
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                reads.add(slice_node.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id != "_config_get" or len(node.args) < 2:
                continue
            first, second = node.args[:2]
            if (
                isinstance(first, ast.Name)
                and first.id == "config"
                and isinstance(second, ast.Constant)
                and isinstance(second.value, str)
            ):
                reads.add(second.value)
    return reads


def validate_parameter_activity(repo: Path) -> dict[str, Any]:
    """Prove that ``curve`` is dead in SL8LHGCN but active in LHGCN/HGCF."""

    sl8_path = repo / "recbole_gnn/model/general_recommender/sl8lhgcn.py"
    sl_parent_path = repo / "slrec_experiments/slrec.py"
    hgcf_path = repo / "recbole_gnn/model/general_recommender/hgcf.py"
    sl16_path = repo / "recbole_gnn/model/general_recommender/sl16lhgcn.py"
    sl8_reads = _config_reads(sl8_path) | _config_reads(sl_parent_path)
    hgcf_reads = _config_reads(hgcf_path)
    if "curve" in sl8_reads:
        raise RuntimeError(
            "SL8LHGCN started reading curve; revisit the effective-parameter grids"
        )
    if "curve" not in hgcf_reads:
        raise RuntimeError("LHGCN/HGCF no longer reads curve; its control grid is stale")
    required_sl8 = {
        "gcn_layers",
        "coord_clip",
        "schatten_p",
        "sl_scale",
        "loss_margin",
    }
    missing = required_sl8 - sl8_reads
    if missing:
        raise RuntimeError(f"SL8 effective parameter audit lost reads: {sorted(missing)}")
    result = {
        "sl8lhgcn": {
            "curve": "dead-not-read",
            "searched_effective_geometry_key": "schatten_p",
            "active_primary_search_keys": [
                "gcn_layers",
                "train_batch_size",
                "schatten_p",
                "loss_margin",
            ],
            "active_but_fixed_in_primary_design": [
                "coord_clip",
                "sl_scale",
                "learning_rate",
                "weight_decay",
                "lhgcn_include_self",
                "negative_samples",
            ],
            "source_files": {
                str(sl8_path.relative_to(repo)): _sha256(sl8_path),
                str(sl_parent_path.relative_to(repo)): _sha256(sl_parent_path),
            },
        },
        "lhgcn": {
            "curve": "active-read-by-HGCF-and-used-by-hyperboloid-ops",
            "source_file": str(hgcf_path.relative_to(repo)),
            "source_sha256": _sha256(hgcf_path),
        },
    }
    if sl16_path.is_file():
        sl16_reads = _config_reads(sl16_path)
        if "curve" in sl16_reads:
            raise RuntimeError("SL16LHGCN unexpectedly reads curve")
        result["sl16lhgcn"] = {
            "curve": "dead-not-read-in-subclass-or-shared-parent",
            "inherits_effective_geometry": "schatten_p",
            "source_file": str(sl16_path.relative_to(repo)),
            "source_sha256": _sha256(sl16_path),
        }
    return result


def _stage_update(parameters: SL8Parameters, stage: TuningStage, value: Any) -> SL8Parameters:
    if stage.key == "self_loop":
        include, weight = value
        updated = replace(
            parameters,
            lhgcn_include_self=bool(include),
            lhgcn_self_loop_weight=float(weight),
        )
    else:
        updated = replace(parameters, **{stage.key: value})
    updated.validate()
    return updated


def staged_new_trial_count(profile: str) -> int:
    """Maximum new jobs after the grid, assuming the default-value anchor."""

    anchor = SL8Parameters(gcn_layers=4, train_batch_size=65_536)
    count = 0
    for stage in TUNING_PROFILES[profile]:
        candidates = {_canonical_hash(asdict(_stage_update(anchor, stage, value))) for value in stage.values}
        candidates.discard(_canonical_hash(asdict(anchor)))
        count += len(candidates)
    return count


def _protected_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in sorted(PROTECTED_DATA_KEYS)}


def _validate_overlay(repo: Path, name: str, expected_model: str) -> dict[str, Any]:
    path = overlay_path(repo, name)
    payload = _yaml_mapping(path)
    overlap = PROTECTED_DATA_KEYS.intersection(payload)
    if overlap:
        raise RuntimeError(f"{name} may not override data/evaluation keys: {sorted(overlap)}")
    if payload.get("model") != expected_model:
        raise RuntimeError(f"{name} must select {expected_model}, got {payload.get('model')!r}")
    return {
        "path": str(path.relative_to(repo)),
        "sha256": _sha256(path),
        "payload": payload,
    }


def _lightgcn_parameters(repo: Path, spec: DatasetSpec, base: Mapping[str, Any]) -> tuple[LightGCNParameters, dict[str, Any]]:
    path = historical_lightgcn_path(repo, spec)
    historical = _yaml_mapping(path)
    if _protected_projection(historical) != _protected_projection(base):
        raise RuntimeError(
            f"historical LightGCN data protocol differs from {spec.recformer_config}: {path}"
        )
    parameters = LightGCNParameters(
        embedding_size=int(historical.get("embedding_size", 64)),
        n_layers=int(historical.get("n_layers", 2)),
        reg_weight=float(historical.get("reg_weight", 1e-5)),
        require_pow=bool(historical.get("require_pow", True)),
        learning_rate=float(historical["learning_rate"]),
        train_batch_size=int(historical["train_batch_size"]),
        learner=str(historical.get("learner", "adam")),
        weight_decay=float(historical.get("weight_decay", 0.0)),
    )
    return parameters, {
        "historical_config": str(path.relative_to(repo)),
        "historical_config_sha256": _sha256(path),
        "historical_model_training_projection": asdict(parameters),
        "data_protocol_replaced_by": spec.recformer_config,
    }


def _lhgcn_parameters(base: Mapping[str, Any]) -> LHGCNParameters:
    return LHGCNParameters(
        embedding_size=int(base["embedding_size"]),
        gcn_layers=int(base["gcn_layers"]),
        curve=float(base["curve"]),
        scale=float(base["scale"]),
        margin=float(base["margin"]),
        learning_rate=float(base["learning_rate"]),
        train_batch_size=int(base["train_batch_size"]),
        learner=str(base.get("learner", "adam")),
        weight_decay=0.0,
    )


def validate_pipeline_protocol(repo: Path, spec: DatasetSpec) -> dict[str, Any]:
    """Pin the RecFormer data protocol and every model overlay/source config."""

    fixed = validate_fixed_protocol(repo, spec)
    base_path = config_path(repo, spec)
    base = _yaml_mapping(base_path)
    if base.get("reproducibility") is not True:
        raise RuntimeError(f"{base_path} must keep reproducibility=true")
    light_overlay = _validate_overlay(repo, LIGHTGCN_OVERLAY, "LightGCN")
    lhgcn_overlay = _validate_overlay(repo, LHGCN_OVERLAY, "LHGCN")
    sl8_overlay = _validate_overlay(repo, SL8_OVERLAY, "SL8LHGCN")
    sl16_overlay = _validate_overlay(repo, SL_OVERLAYS[16], "SL16LHGCN")
    light_parameters, light_source = _lightgcn_parameters(repo, spec, base)
    lhgcn_parameters = _lhgcn_parameters(base)
    parameter_activity = validate_parameter_activity(repo)
    effective_filters = dict(fixed["filters"])
    paper_overlay_record: dict[str, Any] | None = None
    if spec.slug == "amazon-book":
        correction_path = overlay_path(repo, AMAZON_BOOK_PROTOCOL_OVERLAY)
        correction = _yaml_mapping(correction_path)
        expected_correction = {
            "user_inter_num_interval": "[8,inf)",
            "item_inter_num_interval": "[8,inf)",
        }
        if correction != expected_correction:
            raise RuntimeError(
                f"Amazon Book paper correction changed: expected={expected_correction}, "
                f"actual={correction}"
            )
        effective_filters["users"] = correction["user_inter_num_interval"]
        effective_filters["items"] = correction["item_inter_num_interval"]
        paper_overlay_record = {
            "path": str(correction_path.relative_to(repo)),
            "sha256": _sha256(correction_path),
            "reason": (
                "released RecFormer_book.yaml says 5-core, but the paper table is "
                "reproduced exactly by iterative 8-core after rating>=3"
            ),
            "expected_filtered_counts": AMAZON_BOOK_PAPER_COUNTS,
        }
    fixed_effective = {
        **fixed,
        "config_files": [
            str(path.relative_to(repo)) for path in protocol_config_files(repo, spec)
        ],
        "filters": effective_filters,
        "paper_protocol_overlay": paper_overlay_record,
    }
    published_count_note = None
    if spec.slug == "douban-movie":
        published_count_note = {
            "accepted_interactions": 2_552_305,
            "paper_table_interactions": 2_553_305,
            "assessment": "paper table has a +1000 typographical error",
        }
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "dataset_slug": spec.slug,
        "dataset": spec.dataset,
        "protocol_authority": fixed_effective,
        "protocol_config_sha256": _sha256(base_path),
        "published_count_note": published_count_note,
        "parameter_activity": parameter_activity,
        "reproducibility": True,
        "lightgcn": {
            "overlay": light_overlay,
            "parameters": asdict(light_parameters),
            "source": light_source,
        },
        "lhgcn": {
            "overlay": lhgcn_overlay,
            "parameters_matched_to_recformer": asdict(lhgcn_parameters),
        },
        "recformer": {
            "config": str(base_path.relative_to(repo)),
            "config_sha256": _sha256(base_path),
            "fixed_reproduction": True,
        },
        "sl8lhgcn": {
            "overlay": sl8_overlay,
            "grid": {"gcn_layers": list(LAYERS), "train_batch_size": list(BATCH_SIZES)},
            "fixed_training": {
                "epochs": EPOCHS,
                "eval_step": EVAL_STEP,
                "stopping_step": STOPPING_STEP,
                "selection": SELECTION_METRIC,
            },
        },
        "sl16lhgcn": {
            "base_overlay": sl8_overlay,
            "dimension_overlay": sl16_overlay,
            "raw_dimension": 256,
            "intrinsic_dimension": 255,
            "raw_entity_parameter_ratio_vs_sl8": 4.0,
            "dense_cubic_compute_proxy_vs_sl8": 8.0,
            "conservative_train_batch_sizes": list(SL_BATCH_SIZES[16]),
            "evaluation_chunks": SL_EVAL_CHUNKS[16],
        },
        "test_evaluated": False,
    }
    return {**protocol, "signature_sha256": _canonical_hash(protocol)}


def _value_token(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.12g}".replace("-", "m").replace("+", "").replace(".", "p")
    return str(value).replace("-", "m").replace(".", "p")


def sl8_trial_name(parameters: SL8Parameters) -> str:
    parameters.validate()
    core = (
        f"SL{parameters.matrix_dim}_L{parameters.gcn_layers:02d}"
        f"_B{parameters.train_batch_size:06d}"
        f"_lr{_value_token(parameters.learning_rate)}"
        f"_m{_value_token(parameters.loss_margin)}"
        f"_c{_value_token(parameters.coord_clip)}"
        f"_p{_value_token(parameters.schatten_p)}"
        f"_wd{_value_token(parameters.weight_decay)}"
        f"_self{_value_token(parameters.lhgcn_include_self)}"
        f"w{_value_token(parameters.lhgcn_self_loop_weight)}"
        f"_s{_value_token(parameters.sl_scale)}"
        f"_neg{parameters.negative_samples}"
    )
    return f"{core}_{_canonical_hash(asdict(parameters))[:10]}"


@dataclass(frozen=True)
class SelectionJob:
    kind: str
    model: str
    label: str
    parameters: Mapping[str, Any]
    config_files: tuple[Path, ...]
    result_path: Path
    checkpoint_dir: Path
    extra_args: tuple[str, ...]


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"


def _lightgcn_args(parameters: LightGCNParameters) -> tuple[str, ...]:
    return (
        f"--embedding_size={parameters.embedding_size}",
        f"--n_layers={parameters.n_layers}",
        f"--reg_weight={parameters.reg_weight:.12g}",
        f"--require_pow={_bool_arg(parameters.require_pow)}",
        f"--learning_rate={parameters.learning_rate:.12g}",
        f"--train_batch_size={parameters.train_batch_size}",
        f"--learner={parameters.learner}",
        f"--weight_decay={parameters.weight_decay:.12g}",
        "--neg_sampling={'uniform': 1}",
        "--tail_analysis=false",
        "--popularity_analysis=false",
        # Reproduce the historical control's checkpoint-selection cadence.
        "--epochs=500",
        "--eval_step=1",
        "--stopping_step=30",
    )


def _lhgcn_args(parameters: LHGCNParameters) -> tuple[str, ...]:
    return (
        f"--embedding_size={parameters.embedding_size}",
        "--conv=lGCN",
        f"--gcn_layers={parameters.gcn_layers}",
        f"--curve={parameters.curve:.12g}",
        f"--scale={parameters.scale:.12g}",
        f"--margin={parameters.margin:.12g}",
        f"--learning_rate={parameters.learning_rate:.12g}",
        f"--train_batch_size={parameters.train_batch_size}",
        f"--learner={parameters.learner}",
        f"--weight_decay={parameters.weight_decay:.12g}",
        "--reg_weight=0.0",
        "--neg_sampling={'uniform': 1}",
        "--tail_analysis=false",
        "--popularity_analysis=false",
        f"--epochs={EPOCHS}",
        f"--eval_step={EVAL_STEP}",
        f"--stopping_step={STOPPING_STEP}",
    )


def _sl8_args(parameters: SL8Parameters) -> tuple[str, ...]:
    parameters.validate()
    chunks = SL_EVAL_CHUNKS[parameters.matrix_dim]
    raw_dimension = parameters.matrix_dim**2
    return (
        f"--gcn_layers={parameters.gcn_layers}",
        f"--n_layers={parameters.gcn_layers}",
        f"--train_batch_size={parameters.train_batch_size}",
        f"--eval_batch_size={chunks['eval_batch_size']}",
        f"--eval_user_chunk_size={chunks['users']}",
        f"--eval_item_chunk_size={chunks['items']}",
        f"--full_sort_user_batch_size={chunks['users']}",
        f"--embedding_size={raw_dimension}",
        f"--matrix_dim={parameters.matrix_dim}",
        "--num_factors=1",
        "--factor_aggregation=l2",
        "--embedding_init=xavier_uniform_combined",
        "--init_std=0.01",
        f"--coord_clip={parameters.coord_clip:.12g}",
        f"--sl_scale={parameters.sl_scale:.12g}",
        "--sl_gcn_mode=ambient_retract",
        f"--lhgcn_include_self={_bool_arg(parameters.lhgcn_include_self)}",
        f"--lhgcn_self_loop_weight={parameters.lhgcn_self_loop_weight:.12g}",
        "--lhgcn_layer_aggregation=last",
        "--sl_layer_norm=none",
        "--sl_centroid_fallback_clip=1.0",
        "--sl_membership_check=true",
        "--sl_membership_strict=true",
        "--sl_membership_tolerance=0.0001",
        "--sl_distance_membership_check=true",
        f"--sl_distance_check_samples={4 if parameters.matrix_dim == 16 else 16}",
        "--sl_log_trace_tolerance=0.001",
        "--sl_score_mode=group_log",
        f"--schatten_p={parameters.schatten_p}",
        "--log_terms=12",
        "--log_jitter=0.0",
        "--symmetric_distance=false",
        "--fast_one_sided_frobenius=true",
        "--score_scale=1.0",
        "--learnable_score_scale=false",
        "--pairwise_loss=lhgcn_hinge_squared_sum",
        f"--loss_margin={parameters.loss_margin:.12g}",
        "--learner=adam",
        f"--learning_rate={parameters.learning_rate:.12g}",
        f"--weight_decay={parameters.weight_decay:.12g}",
        "--reg_weight=0.0",
        f"--neg_sampling={{'uniform': {parameters.negative_samples}}}",
        "--tail_analysis=false",
        "--popularity_analysis=false",
        f"--epochs={EPOCHS}",
        f"--eval_step={EVAL_STEP}",
        f"--stopping_step={STOPPING_STEP}",
    )


def _dataset_root(args: argparse.Namespace, spec: DatasetSpec) -> Path:
    return args.output_root / spec.slug


def control_jobs(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
) -> tuple[SelectionJob, ...]:
    root = _dataset_root(args, spec)
    data_configs = protocol_config_files(args.repo, spec)
    light = LightGCNParameters(**protocol["lightgcn"]["parameters"])
    lhgcn = LHGCNParameters(**protocol["lhgcn"]["parameters_matched_to_recformer"])
    jobs = (
        SelectionJob(
            kind="lightgcn-control",
            model="LightGCN",
            label="lightgcn",
            parameters=asdict(light),
            config_files=(*data_configs, overlay_path(args.repo, LIGHTGCN_OVERLAY)),
            result_path=root / "controls" / "lightgcn" / "selection.json",
            checkpoint_dir=root / "controls" / "lightgcn" / "checkpoints",
            extra_args=_lightgcn_args(light),
        ),
        SelectionJob(
            kind="lhgcn-matched-control",
            model="LHGCN",
            label="lhgcn",
            parameters=asdict(lhgcn),
            config_files=(*data_configs, overlay_path(args.repo, LHGCN_OVERLAY)),
            result_path=root / "controls" / "lhgcn" / "selection.json",
            checkpoint_dir=root / "controls" / "lhgcn" / "checkpoints",
            extra_args=_lhgcn_args(lhgcn),
        ),
        SelectionJob(
            kind="recformer-fixed-reproduction",
            model="RecFormer",
            label="recformer",
            parameters={"source": "fixed RecFormer YAML; no model/training override"},
            config_files=data_configs,
            result_path=root / "controls" / "recformer" / "selection.json",
            checkpoint_dir=root / "controls" / "recformer" / "checkpoints",
            extra_args=(),
        ),
    )
    return jobs


def lhgcn_capacity_job(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
) -> SelectionJob:
    """256-wide LHGCN control matched to SL(16)'s raw entity capacity."""

    base = LHGCNParameters(
        **protocol["lhgcn"]["parameters_matched_to_recformer"]
    )
    parameters = replace(base, embedding_size=256, train_batch_size=4096)
    root = _dataset_root(args, spec) / "controls" / "lhgcn-capacity-256"
    return SelectionJob(
        kind="lhgcn-matched-capacity-256-control",
        model="LHGCN",
        label="lhgcn-capacity-256",
        parameters={
            **asdict(parameters),
            "capacity_match_target": "SL16LHGCN",
            "raw_entity_dimension": 256,
            "ratio_vs_64d_lhgcn": 4.0,
        },
        config_files=(
            *protocol_config_files(args.repo, spec),
            overlay_path(args.repo, LHGCN_OVERLAY),
        ),
        result_path=root / "selection.json",
        checkpoint_dir=root / "checkpoints",
        extra_args=_lhgcn_args(parameters),
    )


def grid_parameters(matrix_dim: int = 8) -> tuple[SL8Parameters, ...]:
    return tuple(
        SL8Parameters(
            gcn_layers=layer, train_batch_size=batch, matrix_dim=matrix_dim
        )
        for layer in LAYERS
        for batch in SL_BATCH_SIZES[matrix_dim]
    )


def practical_joint_parameters(anchor: SL8Parameters) -> tuple[SL8Parameters, ...]:
    """Joint effective-geometry × margin block at the selected L/B setting."""

    anchor.validate()
    return tuple(
        replace(anchor, schatten_p=geometry, loss_margin=margin)
        for geometry, margin in itertools.product(
            SL8_GEOMETRY_VALUES, SL8_MARGIN_VALUES
        )
    )


def full_cartesian_sl8_parameters(matrix_dim: int = 8) -> tuple[SL8Parameters, ...]:
    """Full primary Cartesian: effective L × B × geometry × margin.

    Numerical/optimisation controls stay fixed at the audited defaults. Curve
    is deliberately absent because SL8LHGCN and its SLRec parent never read it.
    """

    return tuple(
        SL8Parameters(
            gcn_layers=layer,
            train_batch_size=batch,
            matrix_dim=matrix_dim,
            schatten_p=geometry,
            loss_margin=margin,
        )
        for layer, batch, geometry, margin in itertools.product(
            LAYERS,
            SL_BATCH_SIZES[matrix_dim],
            SL8_GEOMETRY_VALUES,
            SL8_MARGIN_VALUES,
        )
    )


def lhgcn_full_cartesian_parameters(anchor: LHGCNParameters) -> tuple[LHGCNParameters, ...]:
    """Independent LHGCN L × B × active curvature × margin control grid."""

    return tuple(
        replace(
            anchor,
            gcn_layers=layer,
            train_batch_size=batch,
            curve=curve,
            margin=margin,
        )
        for layer, batch, curve, margin in itertools.product(
            LAYERS, BATCH_SIZES, LHGCN_CURVE_VALUES, LHGCN_MARGIN_VALUES
        )
    )


def lhgcn_trial_name(parameters: LHGCNParameters) -> str:
    core = (
        f"L{parameters.gcn_layers:02d}_B{parameters.train_batch_size:06d}"
        f"_curve{_value_token(parameters.curve)}"
        f"_margin{_value_token(parameters.margin)}"
    )
    return f"{core}_{_canonical_hash(asdict(parameters))[:10]}"


def grid_jobs(
    args: argparse.Namespace, spec: DatasetSpec, matrix_dim: int = 8
) -> tuple[SelectionJob, ...]:
    root = _dataset_root(args, spec) / f"sl{matrix_dim}" / "grid"
    configs = sl_model_config_files(args.repo, spec, matrix_dim)
    jobs = []
    for parameters in grid_parameters(matrix_dim):
        name = sl8_trial_name(parameters)
        jobs.append(
            SelectionJob(
                kind=f"sl{matrix_dim}-layer-batch-grid",
                model=SL_MODEL_NAMES[matrix_dim],
                label=name,
                parameters=asdict(parameters),
                config_files=configs,
                result_path=root / "results" / f"{name}.json",
                checkpoint_dir=root / "checkpoints" / name,
                extra_args=_sl8_args(parameters),
            )
        )
    return tuple(jobs)


def practical_joint_jobs(
    args: argparse.Namespace,
    spec: DatasetSpec,
    anchor: SL8Parameters,
) -> tuple[SelectionJob, ...]:
    matrix_dim = anchor.matrix_dim
    root = (
        _dataset_root(args, spec)
        / f"sl{matrix_dim}"
        / "practical-geometry-margin"
    )
    configs = sl_model_config_files(args.repo, spec, matrix_dim)
    jobs: list[SelectionJob] = []
    for parameters in practical_joint_parameters(anchor):
        if parameters == anchor:
            continue
        name = sl8_trial_name(parameters)
        jobs.append(
            SelectionJob(
                kind=f"sl{matrix_dim}-practical-geometry-margin",
                model=SL_MODEL_NAMES[matrix_dim],
                label=name,
                parameters=asdict(parameters),
                config_files=configs,
                result_path=root / "results" / f"{name}.json",
                checkpoint_dir=root / "checkpoints" / name,
                extra_args=_sl8_args(parameters),
            )
        )
    return tuple(jobs)


def full_cartesian_sl8_jobs(
    args: argparse.Namespace, spec: DatasetSpec, matrix_dim: int = 8
) -> tuple[SelectionJob, ...]:
    root = _dataset_root(args, spec) / f"sl{matrix_dim}" / "full-cartesian"
    configs = sl_model_config_files(args.repo, spec, matrix_dim)
    return tuple(
        SelectionJob(
            kind=f"sl{matrix_dim}-primary-full-cartesian",
            model=SL_MODEL_NAMES[matrix_dim],
            label=sl8_trial_name(parameters),
            parameters=asdict(parameters),
            config_files=configs,
            result_path=root / "results" / f"{sl8_trial_name(parameters)}.json",
            checkpoint_dir=root / "checkpoints" / sl8_trial_name(parameters),
            extra_args=_sl8_args(parameters),
        )
        for parameters in full_cartesian_sl8_parameters(matrix_dim)
    )


def lhgcn_full_cartesian_jobs(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
) -> tuple[SelectionJob, ...]:
    root = _dataset_root(args, spec) / "lhgcn-control" / "full-cartesian"
    data_configs = protocol_config_files(args.repo, spec)
    overlay = overlay_path(args.repo, LHGCN_OVERLAY)
    anchor = LHGCNParameters(**protocol["lhgcn"]["parameters_matched_to_recformer"])
    jobs: list[SelectionJob] = []
    for parameters in lhgcn_full_cartesian_parameters(anchor):
        name = lhgcn_trial_name(parameters)
        jobs.append(
            SelectionJob(
                kind="lhgcn-layer-batch-curve-margin-full-cartesian",
                model="LHGCN",
                label=name,
                parameters=asdict(parameters),
                config_files=(*data_configs, overlay),
                result_path=root / "results" / f"{name}.json",
                checkpoint_dir=root / "checkpoints" / name,
                extra_args=_lhgcn_args(parameters),
            )
        )
    return tuple(jobs)


def tuning_jobs(
    args: argparse.Namespace,
    spec: DatasetSpec,
    stage_index: int,
    stage: TuningStage,
    anchor_parameters: SL8Parameters,
) -> tuple[SelectionJob, ...]:
    root = _dataset_root(args, spec) / "sl8" / "tuning" / args.tuning_profile
    stage_root = root / "stages" / f"{stage_index:02d}-{stage.key.replace('_', '-')}"
    data_configs = protocol_config_files(args.repo, spec)
    overlay = overlay_path(args.repo, SL8_OVERLAY)
    jobs: list[SelectionJob] = []
    seen: set[SL8Parameters] = set()
    for value in stage.values:
        parameters = _stage_update(anchor_parameters, stage, value)
        if parameters == anchor_parameters or parameters in seen:
            continue
        seen.add(parameters)
        name = sl8_trial_name(parameters)
        jobs.append(
            SelectionJob(
                kind=f"sl8-staged-{stage.key}",
                model="SL8LHGCN",
                label=name,
                parameters=asdict(parameters),
                config_files=(*data_configs, overlay),
                result_path=stage_root / "results" / f"{name}.json",
                checkpoint_dir=stage_root / "checkpoints" / name,
                extra_args=_sl8_args(parameters),
            )
        )
    return tuple(jobs)


def selection_command(
    args: argparse.Namespace, spec: DatasetSpec, job: SelectionJob
) -> list[str]:
    return [
        args.python,
        "-u",
        str(args.repo / "run_recbole_gnn.py"),
        "--model",
        job.model,
        "--dataset",
        spec.dataset,
        "--config-files",
        " ".join(str(path) for path in job.config_files),
        "--validation-only",
        "--result-file",
        str(job.result_path),
        f"--checkpoint_dir={job.checkpoint_dir}",
        f"--data_path={args.data_root}",
        # Do not pass logical zero: vendored RecBole rewrites the environment
        # from this value and would otherwise unmask/select physical GPU 0.
        f"--gpu_id={PHYSICAL_GPU}",
        "--use_gpu=true",
        "--show_progress=false",
        f"--seed={SEED}",
        *job.extra_args,
    ]


def final_test_command(
    args: argparse.Namespace,
    selection_path: Path,
    checkpoint_path: Path,
    result_path: Path,
    matrix_dim: int = 8,
) -> list[str]:
    chunks = SL_EVAL_CHUNKS[matrix_dim]
    return [
        args.python,
        "-u",
        str(args.repo / "evaluate_recbole_gnn_checkpoint.py"),
        "--checkpoint-file",
        str(checkpoint_path),
        "--selection-result-file",
        str(selection_path),
        "--result-file",
        str(result_path),
        "--skip-valid",
        "--eval-batch-size",
        str(chunks["eval_batch_size"]),
        "--eval-user-chunk-size",
        str(chunks["users"]),
        "--eval-item-chunk-size",
        str(chunks["items"]),
        "--full-sort-user-batch-size",
        str(chunks["users"]),
        "--device",
        "cuda",
    ]


def _relative_or_absolute(path: Path, repo: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(repo))
    except ValueError:
        return str(resolved)


def job_metadata(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    job: SelectionJob,
) -> dict[str, Any]:
    configs = [
        {
            "path": _relative_or_absolute(path, args.repo),
            "sha256": _sha256(path),
        }
        for path in job.config_files
    ]
    geometry_capacity = None
    if job.model in SL_MODEL_NAMES.values():
        matrix_dim = int(job.parameters["matrix_dim"])
        geometry_capacity = {
            "matrix_dim": matrix_dim,
            "raw_dimension": matrix_dim**2,
            "intrinsic_dimension": matrix_dim**2 - 1,
            "raw_entity_parameter_ratio_vs_sl8": (matrix_dim / 8) ** 2,
            "dense_cubic_compute_proxy_vs_sl8": (matrix_dim / 8) ** 3,
            "parameter_budget_compared_separately": True,
        }
    core = {
        "schema_version": SCHEMA_VERSION,
        "kind": job.kind,
        "label": job.label,
        "model": job.model,
        "dataset_slug": spec.slug,
        "dataset": spec.dataset,
        "parameters": dict(job.parameters),
        "geometry_capacity": geometry_capacity,
        "config_files": configs,
        "protocol_signature_sha256": protocol["signature_sha256"],
        "physical_gpu": PHYSICAL_GPU,
        "selection_metric": SELECTION_METRIC,
        "validation_only": True,
        "test_evaluated": False,
    }
    return {**core, "signature_sha256": _canonical_hash(core)}


def _validate_selection_common(
    payload: Mapping[str, Any],
    *,
    path: Path,
    model: str,
    spec: DatasetSpec,
) -> None:
    if payload.get("model") != model or payload.get("dataset") != spec.dataset:
        raise ValueError(f"wrong model/dataset in selection artifact: {path}")
    if int(payload.get("seed", -1)) != SEED:
        raise ValueError(f"wrong seed in selection artifact: {path}")
    if payload.get("test_result") is not None:
        raise RuntimeError(f"selection artifact touched held-out test: {path}")
    score = payload.get("best_valid_score")
    metrics = payload.get("best_valid_result")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError(f"missing finite validation score: {path}")
    if not isinstance(metrics, Mapping) or not isinstance(metrics.get("recall@10"), (int, float)):
        raise ValueError(f"missing validation Recall@10: {path}")
    checkpoint = payload.get("checkpoint_file")
    if not checkpoint or not Path(str(checkpoint)).expanduser().is_file():
        raise ValueError(f"missing selected checkpoint: {path}")
    fingerprints = payload.get("split_fingerprints")
    if not isinstance(fingerprints, Mapping) or set(fingerprints) != {
        "train",
        "valid",
        "test",
    }:
        raise ValueError(f"missing exact split fingerprints: {path}")
    for split, fingerprint in fingerprints.items():
        if not isinstance(fingerprint, Mapping):
            raise ValueError(f"invalid {split} split fingerprint: {path}")
        if not isinstance(fingerprint.get("interactions"), int) or not isinstance(
            fingerprint.get("sha256"), str
        ):
            raise ValueError(f"invalid {split} split fingerprint: {path}")


def load_selection(
    path: Path,
    *,
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    job: SelectionJob,
) -> dict[str, Any]:
    payload = _load_mapping(path)
    _validate_selection_common(payload, path=path, model=job.model, spec=spec)
    expected = job_metadata(args, spec, protocol, job)
    if payload.get("paper_dataset_pipeline") != expected:
        raise ValueError(f"selection resume contract mismatch: {path}")
    return payload


def completed_selection(
    path: Path,
    *,
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    job: SelectionJob,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    # Existing but incompatible artifacts are never overwritten implicitly.
    # A changed protocol or parameter grid deserves a new output root.
    return load_selection(path, args=args, spec=spec, protocol=protocol, job=job)


def annotate_selection(
    path: Path,
    *,
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    job: SelectionJob,
) -> dict[str, Any]:
    payload = _load_mapping(path)
    _validate_selection_common(payload, path=path, model=job.model, spec=spec)
    payload["paper_dataset_pipeline"] = job_metadata(args, spec, protocol, job)
    _atomic_json(path, payload)
    return load_selection(path, args=args, spec=spec, protocol=protocol, job=job)


def _candidate(job: SelectionJob, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": job.label,
        "kind": job.kind,
        "parameters": dict(job.parameters),
        "selection_result_file": str(job.result_path.expanduser().resolve()),
        "checkpoint_file": str(Path(str(result["checkpoint_file"])).expanduser().resolve()),
        "best_valid_score": float(result["best_valid_score"]),
        "best_valid_result": dict(result["best_valid_result"]),
        "split_fingerprints": dict(result["split_fingerprints"]),
        "selection_signature_sha256": result["paper_dataset_pipeline"]["signature_sha256"],
        "test_evaluated": False,
    }


def _rank(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(candidate)
        for candidate in sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate["best_valid_score"]),
                str(candidate["label"]),
            ),
        )
    ]


def _ensure_same_split(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], spec: DatasetSpec
) -> None:
    if reference["split_fingerprints"] != candidate["split_fingerprints"]:
        raise RuntimeError(f"model/trial split fingerprints differ for {spec.dataset}")


def _signed_payload(core: Mapping[str, Any]) -> dict[str, Any]:
    return {**core, "signature_sha256": _canonical_hash(core)}


def _load_signed_payload(path: Path) -> dict[str, Any]:
    payload = _load_mapping(path)
    signature = payload.pop("signature_sha256", None)
    expected = _canonical_hash(payload)
    payload["signature_sha256"] = signature
    if signature != expected:
        raise ValueError(f"summary signature mismatch: {path}")
    return payload


def validate_candidate_record(
    candidate: Mapping[str, Any],
    *,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(str(candidate.get("selection_result_file", ""))).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"candidate selection result is missing: {path}")
    payload = _load_mapping(path)
    metadata = payload.get("paper_dataset_pipeline")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"candidate lacks pipeline metadata: {path}")
    metadata_core = dict(metadata)
    metadata_signature = metadata_core.pop("signature_sha256", None)
    if metadata_signature != _canonical_hash(metadata_core):
        raise ValueError(f"candidate pipeline metadata signature mismatch: {path}")
    _validate_selection_common(
        payload,
        path=path,
        model=str(metadata.get("model")),
        spec=spec,
    )
    if metadata.get("protocol_signature_sha256") != protocol["signature_sha256"]:
        raise ValueError(f"candidate protocol changed: {path}")
    expected = {
        "label": metadata.get("label"),
        "kind": metadata.get("kind"),
        "parameters": metadata.get("parameters"),
        "selection_result_file": str(path),
        "checkpoint_file": str(Path(str(payload["checkpoint_file"])).expanduser().resolve()),
        "best_valid_score": float(payload["best_valid_score"]),
        "best_valid_result": dict(payload["best_valid_result"]),
        "split_fingerprints": dict(payload["split_fingerprints"]),
        "selection_signature_sha256": metadata.get("signature_sha256"),
        "test_evaluated": False,
    }
    if dict(candidate) != expected:
        raise ValueError(f"candidate summary differs from selection artifact: {path}")
    return expected


def grid_summary_path(
    args: argparse.Namespace, spec: DatasetSpec, matrix_dim: int = 8
) -> Path:
    return _dataset_root(args, spec) / f"sl{matrix_dim}" / "grid" / "summary.json"


def write_grid_summary(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    matrix_dim: int = 8,
) -> dict[str, Any]:
    if candidates:
        split = candidates[0]["split_fingerprints"]
        if any(candidate["split_fingerprints"] != split for candidate in candidates[1:]):
            raise RuntimeError(f"SL8 grid uses different splits for {spec.dataset}")
    ranking = _rank(candidates)
    complete = len(ranking) == len(grid_parameters(matrix_dim))
    core = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"sl{matrix_dim}-layer-batch-grid-summary",
        "matrix_dim": matrix_dim,
        "raw_dimension": matrix_dim**2,
        "intrinsic_dimension": matrix_dim**2 - 1,
        "raw_entity_parameter_ratio_vs_sl8": (matrix_dim / 8) ** 2,
        "dataset_slug": spec.slug,
        "dataset": spec.dataset,
        "protocol_signature_sha256": protocol["signature_sha256"],
        "selection_metric": SELECTION_METRIC,
        "state": "complete" if complete else "incomplete",
        "expected_trials": len(grid_parameters(matrix_dim)),
        "completed_trials": len(ranking),
        "winner": ranking[0] if complete else None,
        "provisional_winner": ranking[0] if ranking else None,
        "ranking": ranking,
        "test_evaluated": False,
    }
    payload = _signed_payload(core)
    _atomic_json(grid_summary_path(args, spec, matrix_dim), payload)
    return payload


def load_grid_winner(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    matrix_dim: int = 8,
) -> dict[str, Any]:
    path = grid_summary_path(args, spec, matrix_dim)
    payload = _load_signed_payload(path)
    if (
        payload.get("kind") != f"sl{matrix_dim}-layer-batch-grid-summary"
        or payload.get("dataset_slug") != spec.slug
        or payload.get("dataset") != spec.dataset
        or payload.get("protocol_signature_sha256") != protocol["signature_sha256"]
        or payload.get("state") != "complete"
        or payload.get("matrix_dim") != matrix_dim
        or payload.get("expected_trials") != len(grid_parameters(matrix_dim))
        or payload.get("completed_trials") != len(grid_parameters(matrix_dim))
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"stale or incomplete grid summary: {path}")
    ranking = payload.get("ranking")
    winner = payload.get("winner")
    if not isinstance(ranking, list) or len(ranking) != len(grid_parameters(matrix_dim)):
        raise ValueError(f"invalid complete grid ranking: {path}")
    validated = [
        validate_candidate_record(candidate, spec=spec, protocol=protocol)
        for candidate in ranking
    ]
    expected_parameter_signatures = {
        _canonical_hash(asdict(parameters)) for parameters in grid_parameters(matrix_dim)
    }
    actual_parameter_signatures = {
        _canonical_hash(candidate["parameters"]) for candidate in validated
    }
    if actual_parameter_signatures != expected_parameter_signatures or any(
        candidate["kind"] != f"sl{matrix_dim}-layer-batch-grid"
        for candidate in validated
    ):
        raise ValueError(f"complete grid does not contain the exact layer/batch grid: {path}")
    if not validated or winner != validated[0] or validated != _rank(validated):
        raise ValueError(f"grid winner/ranking is inconsistent: {path}")
    return dict(winner)


def factorial_summary_path(
    args: argparse.Namespace, spec: DatasetSpec, search: str
) -> Path:
    if search == "lhgcn-full-cartesian":
        relative = Path("lhgcn-control/full-cartesian/summary.json")
    elif search.startswith("sl") and search.endswith("-practical"):
        dimension = int(search[2:].split("-", 1)[0])
        relative = Path(f"sl{dimension}/practical-geometry-margin/summary.json")
    elif search.startswith("sl") and search.endswith("-full-cartesian"):
        dimension = int(search[2:].split("-", 1)[0])
        relative = Path(f"sl{dimension}/full-cartesian/summary.json")
    else:
        raise ValueError(f"unknown factorial search {search!r}")
    return _dataset_root(args, spec) / relative


def write_factorial_summary(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    *,
    search: str,
    candidates: Sequence[Mapping[str, Any]],
    expected_parameters: Sequence[Mapping[str, Any]],
    axes: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    if candidates:
        split = candidates[0]["split_fingerprints"]
        if any(candidate["split_fingerprints"] != split for candidate in candidates[1:]):
            raise RuntimeError(f"{search} candidates use different splits for {spec.dataset}")
    ranking = _rank(candidates)
    expected_hashes = {_canonical_hash(parameters) for parameters in expected_parameters}
    actual_hashes = {_canonical_hash(candidate["parameters"]) for candidate in ranking}
    no_duplicates = len(actual_hashes) == len(ranking)
    complete = no_duplicates and actual_hashes == expected_hashes
    matrix_dim = None
    if search.startswith("sl"):
        matrix_dim = int(search[2:].split("-", 1)[0])
    core = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{search}-summary",
        "dataset_slug": spec.slug,
        "dataset": spec.dataset,
        "protocol_signature_sha256": protocol["signature_sha256"],
        "selection_metric": SELECTION_METRIC,
        "design": search,
        "matrix_dim": matrix_dim,
        "raw_dimension": matrix_dim**2 if matrix_dim else None,
        "intrinsic_dimension": matrix_dim**2 - 1 if matrix_dim else None,
        "raw_entity_parameter_ratio_vs_sl8": (
            (matrix_dim / 8) ** 2 if matrix_dim else None
        ),
        "parameter_budget_compared_separately": matrix_dim is not None,
        "axes": {key: list(values) for key, values in axes.items()},
        "state": "complete" if complete else "incomplete",
        "expected_trials": len(expected_hashes),
        "completed_unique_trials": len(actual_hashes),
        "winner": ranking[0] if complete else None,
        "provisional_winner": ranking[0] if ranking else None,
        "ranking": ranking,
        "test_evaluated": False,
    }
    payload = _signed_payload(core)
    _atomic_json(factorial_summary_path(args, spec, search), payload)
    return payload


def load_factorial_winner(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    *,
    search: str,
    expected_parameters: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = factorial_summary_path(args, spec, search)
    payload = _load_signed_payload(path)
    expected_hashes = {_canonical_hash(parameters) for parameters in expected_parameters}
    if (
        payload.get("kind") != f"{search}-summary"
        or payload.get("dataset_slug") != spec.slug
        or payload.get("dataset") != spec.dataset
        or payload.get("protocol_signature_sha256") != protocol["signature_sha256"]
        or payload.get("state") != "complete"
        or payload.get("expected_trials") != len(expected_hashes)
        or payload.get("completed_unique_trials") != len(expected_hashes)
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"stale or incomplete factorial summary: {path}")
    ranking = payload.get("ranking")
    winner = payload.get("winner")
    if not isinstance(ranking, list) or len(ranking) != len(expected_hashes):
        raise ValueError(f"invalid complete factorial ranking: {path}")
    validated = [
        validate_candidate_record(candidate, spec=spec, protocol=protocol)
        for candidate in ranking
    ]
    actual_hashes = {_canonical_hash(candidate["parameters"]) for candidate in validated}
    if actual_hashes != expected_hashes or winner != validated[0] or validated != _rank(validated):
        raise ValueError(f"factorial winner/ranking is inconsistent: {path}")
    return dict(winner)


def tuning_stage_root(
    args: argparse.Namespace, spec: DatasetSpec, stage_index: int, stage: TuningStage
) -> Path:
    return (
        _dataset_root(args, spec)
        / "sl8"
        / "tuning"
        / args.tuning_profile
        / "stages"
        / f"{stage_index:02d}-{stage.key.replace('_', '-')}"
    )


def tuning_stage_summary_path(
    args: argparse.Namespace, spec: DatasetSpec, stage_index: int, stage: TuningStage
) -> Path:
    return tuning_stage_root(args, spec, stage_index, stage) / "summary.json"


def write_tuning_stage_summary(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    stage_index: int,
    stage: TuningStage,
    anchor: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    expected_jobs: int,
) -> dict[str, Any]:
    all_candidates = [dict(anchor), *[dict(candidate) for candidate in candidates]]
    for candidate in all_candidates[1:]:
        _ensure_same_split(anchor, candidate, spec)
    ranking = _rank(all_candidates)
    complete = len(candidates) == expected_jobs
    core = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sl8-staged-tuning-summary",
        "dataset_slug": spec.slug,
        "dataset": spec.dataset,
        "profile": args.tuning_profile,
        "stage_index": stage_index,
        "stage_key": stage.key,
        "search_values": list(stage.values),
        "protocol_signature_sha256": protocol["signature_sha256"],
        "selection_metric": SELECTION_METRIC,
        "state": "complete" if complete else "incomplete",
        "anchor": dict(anchor),
        "expected_new_trials": expected_jobs,
        "completed_new_trials": len(candidates),
        "winner": ranking[0] if complete else None,
        "provisional_winner": ranking[0] if ranking else None,
        "ranking": ranking,
        "test_evaluated": False,
    }
    payload = _signed_payload(core)
    _atomic_json(tuning_stage_summary_path(args, spec, stage_index, stage), payload)
    return payload


def load_tuning_stage_winner(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    stage_index: int,
    stage: TuningStage,
    expected_anchor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = tuning_stage_summary_path(args, spec, stage_index, stage)
    payload = _load_signed_payload(path)
    if (
        payload.get("kind") != "sl8-staged-tuning-summary"
        or payload.get("dataset_slug") != spec.slug
        or payload.get("dataset") != spec.dataset
        or payload.get("profile") != args.tuning_profile
        or payload.get("stage_index") != stage_index
        or payload.get("stage_key") != stage.key
        or payload.get("search_values")
        != json.loads(json.dumps(list(stage.values)))
        or payload.get("protocol_signature_sha256") != protocol["signature_sha256"]
        or payload.get("state") != "complete"
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"stale or incomplete tuning stage: {path}")
    ranking = payload.get("ranking")
    winner = payload.get("winner")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError(f"completed stage has no ranking: {path}")
    anchor = payload.get("anchor")
    if not isinstance(anchor, Mapping):
        raise ValueError(f"completed stage has no anchor: {path}")
    if expected_anchor is not None and dict(anchor) != dict(expected_anchor):
        raise ValueError(f"tuning stage anchor differs from previous winner: {path}")
    validate_candidate_record(anchor, spec=spec, protocol=protocol)
    anchor_parameters = _sl8_parameters_from_candidate(anchor)
    expected_jobs = tuning_jobs(args, spec, stage_index, stage, anchor_parameters)
    if (
        payload.get("expected_new_trials") != len(expected_jobs)
        or payload.get("completed_new_trials") != len(expected_jobs)
        or len(ranking) != len(expected_jobs) + 1
    ):
        raise ValueError(f"completed stage has the wrong trial count: {path}")
    validated = [
        validate_candidate_record(candidate, spec=spec, protocol=protocol)
        for candidate in ranking
    ]
    expected_parameters = {
        _canonical_hash(anchor["parameters"]),
        *(_canonical_hash(job.parameters) for job in expected_jobs),
    }
    actual_parameters = {
        _canonical_hash(candidate["parameters"]) for candidate in validated
    }
    if actual_parameters != expected_parameters:
        raise ValueError(f"stage ranking does not contain the exact search candidates: {path}")
    if winner != validated[0] or validated != _rank(validated):
        raise ValueError(f"stage winner/ranking is inconsistent: {path}")
    return dict(winner)


def tuning_summary_path(args: argparse.Namespace, spec: DatasetSpec) -> Path:
    return (
        _dataset_root(args, spec)
        / "sl8"
        / "tuning"
        / args.tuning_profile
        / "summary.json"
    )


def write_tuning_summary(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    completed_stages: Sequence[Mapping[str, Any]],
    final_winner: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected = len(TUNING_PROFILES[args.tuning_profile])
    complete = len(completed_stages) == expected and final_winner is not None
    core = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sl8-staged-tuning-final-summary",
        "dataset_slug": spec.slug,
        "dataset": spec.dataset,
        "profile": args.tuning_profile,
        "protocol_signature_sha256": protocol["signature_sha256"],
        "selection_metric": SELECTION_METRIC,
        "state": "complete" if complete else "incomplete",
        "expected_stages": expected,
        "completed_stages": list(completed_stages),
        "final_validation_winner": dict(final_winner) if complete else None,
        "latest_complete_stage_winner": dict(final_winner) if final_winner else None,
        "test_evaluated": False,
    }
    payload = _signed_payload(core)
    _atomic_json(tuning_summary_path(args, spec), payload)
    return payload


def load_tuning_winner(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    path = tuning_summary_path(args, spec)
    payload = _load_signed_payload(path)
    expected = len(TUNING_PROFILES[args.tuning_profile])
    if (
        payload.get("kind") != "sl8-staged-tuning-final-summary"
        or payload.get("dataset_slug") != spec.slug
        or payload.get("dataset") != spec.dataset
        or payload.get("profile") != args.tuning_profile
        or payload.get("protocol_signature_sha256") != protocol["signature_sha256"]
        or payload.get("state") != "complete"
        or payload.get("expected_stages") != expected
        or len(payload.get("completed_stages", [])) != expected
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"stale or incomplete final tuning summary: {path}")
    winner = payload.get("final_validation_winner")
    if not isinstance(winner, Mapping):
        raise ValueError(f"complete tuning summary has no winner: {path}")
    validate_candidate_record(winner, spec=spec, protocol=protocol)
    stages = TUNING_PROFILES[args.tuning_profile]
    anchor = load_grid_winner(args, spec, protocol)
    expected_stage_records: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(stages, 1):
        anchor = load_tuning_stage_winner(
            args,
            spec,
            protocol,
            stage_index,
            stage,
            expected_anchor=anchor,
        )
        stage_path = tuning_stage_summary_path(args, spec, stage_index, stage)
        stage_payload = _load_signed_payload(stage_path)
        expected_stage_records.append(
            {
                "stage_index": stage_index,
                "stage_key": stage.key,
                "summary_file": str(stage_path.resolve()),
                "summary_signature_sha256": stage_payload["signature_sha256"],
                "winner": anchor,
            }
        )
    if payload.get("completed_stages") != expected_stage_records:
        raise ValueError(f"final summary stage provenance is inconsistent: {path}")
    if dict(winner) != anchor:
        raise ValueError(f"final summary winner differs from last staged winner: {path}")
    return dict(winner)


def load_selected_sl8_winner(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    matrix_dim: int = 8,
) -> dict[str, Any]:
    if args.sl8_search == "staged":
        if matrix_dim != 8:
            raise ValueError("legacy staged search is implemented only for SL(8)")
        return load_tuning_winner(args, spec, protocol)
    if args.sl8_search == "full-cartesian":
        return load_factorial_winner(
            args,
            spec,
            protocol,
            search=f"sl{matrix_dim}-full-cartesian",
            expected_parameters=[
                asdict(parameters)
                for parameters in full_cartesian_sl8_parameters(matrix_dim)
            ],
        )
    grid_winner = load_grid_winner(args, spec, protocol, matrix_dim)
    anchor = _sl8_parameters_from_candidate(grid_winner)
    return load_factorial_winner(
        args,
        spec,
        protocol,
        search=f"sl{matrix_dim}-practical",
        expected_parameters=[
            asdict(parameters) for parameters in practical_joint_parameters(anchor)
        ],
    )


def load_selected_lhgcn(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if args.lhgcn_search == "full-cartesian":
        jobs = lhgcn_full_cartesian_jobs(args, spec, protocol)
        winner = load_factorial_winner(
            args,
            spec,
            protocol,
            search="lhgcn-full-cartesian",
            expected_parameters=[job.parameters for job in jobs],
        )
        path = Path(winner["selection_result_file"]).resolve()
        return path, _load_mapping(path)
    job = next(job for job in control_jobs(args, spec, protocol) if job.label == "lhgcn")
    result = load_selection(
        job.result_path,
        args=args,
        spec=spec,
        protocol=protocol,
        job=job,
    )
    return job.result_path, result


def default_lock_path() -> Path:
    """Use the same lock filename as the current Toy SL8/LHGCN runners."""

    digest = hashlib.sha256(PHYSICAL_GPU.encode("utf-8")).hexdigest()[:16]
    return (
        Path(tempfile.gettempdir())
        / f"hgformer-toy-sl8lhgcn-uid{os.getuid()}-gpu-{digest}.lock"
    )


@contextlib.contextmanager
def exclusive_gpu_lock(path: Path) -> Iterable[int]:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock.seek(0)
            owner = lock.read().strip() or "unknown owner"
            raise RuntimeError(
                f"physical GPU {PHYSICAL_GPU} is already reserved: {resolved} ({owner})"
            ) from error
        lock.seek(0)
        lock.truncate()
        lock.write(
            f"pid={os.getpid()} gpu={PHYSICAL_GPU} paper_pipeline=true "
            f"acquired_at={_utc_now()}\n"
        )
        lock.flush()
        try:
            yield lock.fileno()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _run_and_tee(
    command: Sequence[str],
    *,
    log_path: Path,
    cwd: Path,
    environment: Mapping[str, str],
    lock_fd: int,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nSTARTED_AT={_utc_now()}\nCOMMAND={shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
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
        finally:
            log.write(f"FINISHED_AT={_utc_now()}\n")
            log.flush()
    if return_code:
        raise subprocess.CalledProcessError(return_code, list(command))


def _job_log_path(job: SelectionJob) -> Path:
    return job.result_path.parent.parent / "logs" / f"{job.label}.log"


def _job_failure_path(job: SelectionJob) -> Path:
    return job.result_path.parent.parent / "failures" / f"{job.label}.json"


def execute_selection_job(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    job: SelectionJob,
    *,
    environment: Mapping[str, str],
    lock_fd: int,
    budget: list[int],
) -> dict[str, Any] | None:
    complete = completed_selection(
        job.result_path,
        args=args,
        spec=spec,
        protocol=protocol,
        job=job,
    )
    if complete is not None:
        print(f"SKIP {spec.slug} {job.kind} {job.label}", flush=True)
        return complete
    if budget[0] == 0:
        return None

    job.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    command = selection_command(args, spec, job)
    started = _utc_now()
    print(f"START {spec.slug} {job.kind} {job.label}", flush=True)
    try:
        _run_and_tee(
            command,
            log_path=_job_log_path(job),
            cwd=args.repo,
            environment=environment,
            lock_fd=lock_fd,
        )
        result = annotate_selection(
            job.result_path,
            args=args,
            spec=spec,
            protocol=protocol,
            job=job,
        )
    except BaseException as error:
        _atomic_json(
            _job_failure_path(job),
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_slug": spec.slug,
                "kind": job.kind,
                "label": job.label,
                "started_at": started,
                "failed_at": _utc_now(),
                "command": list(command),
                "error_type": type(error).__name__,
                "error": str(error),
                "test_evaluated": False,
            },
        )
        raise
    failure = _job_failure_path(job)
    if failure.is_file():
        failure.unlink()
    budget[0] -= 1
    return result


def _first_split_reference(candidates: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next(iter(candidates), None)


def execute_controls(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    lock_fd: int,
    budget: list[int],
    labels: set[str] | None = None,
) -> bool:
    reference: Mapping[str, Any] | None = None
    for job in control_jobs(args, spec, protocol):
        if labels is not None and job.label not in labels:
            continue
        result = execute_selection_job(
            args,
            spec,
            protocol,
            job,
            environment=environment,
            lock_fd=lock_fd,
            budget=budget,
        )
        if result is None:
            return False
        candidate = _candidate(job, result)
        if reference is not None:
            _ensure_same_split(reference, candidate, spec)
        reference = candidate
    return True


def _load_existing_control_candidates(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for job in control_jobs(args, spec, protocol):
        if not job.result_path.is_file():
            continue
        result = completed_selection(
            job.result_path,
            args=args,
            spec=spec,
            protocol=protocol,
            job=job,
        )
        assert result is not None
        candidates.append(_candidate(job, result))
    for candidate in candidates[1:]:
        _ensure_same_split(candidates[0], candidate, spec)
    return candidates


def execute_grid(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    lock_fd: int,
    budget: list[int],
    matrix_dim: int = 8,
) -> bool:
    control_candidates = _load_existing_control_candidates(args, spec, protocol)
    reference = _first_split_reference(control_candidates)
    candidates: list[dict[str, Any]] = []
    for job in grid_jobs(args, spec, matrix_dim):
        result = execute_selection_job(
            args,
            spec,
            protocol,
            job,
            environment=environment,
            lock_fd=lock_fd,
            budget=budget,
        )
        if result is None:
            write_grid_summary(args, spec, protocol, candidates, matrix_dim)
            return False
        candidate = _candidate(job, result)
        if reference is not None:
            _ensure_same_split(reference, candidate, spec)
        else:
            reference = candidate
        candidates.append(candidate)
        write_grid_summary(args, spec, protocol, candidates, matrix_dim)
    write_grid_summary(args, spec, protocol, candidates, matrix_dim)
    return True


def execute_lhgcn_full_cartesian(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    lock_fd: int,
    budget: list[int],
) -> bool:
    jobs = lhgcn_full_cartesian_jobs(args, spec, protocol)
    expected = [job.parameters for job in jobs]
    control_candidates = _load_existing_control_candidates(args, spec, protocol)
    reference = _first_split_reference(control_candidates)
    candidates: list[dict[str, Any]] = []
    axes = {
        "gcn_layers": LAYERS,
        "train_batch_size": BATCH_SIZES,
        "curve": LHGCN_CURVE_VALUES,
        "margin": LHGCN_MARGIN_VALUES,
    }
    for job in jobs:
        result = execute_selection_job(
            args,
            spec,
            protocol,
            job,
            environment=environment,
            lock_fd=lock_fd,
            budget=budget,
        )
        if result is None:
            write_factorial_summary(
                args,
                spec,
                protocol,
                search="lhgcn-full-cartesian",
                candidates=candidates,
                expected_parameters=expected,
                axes=axes,
            )
            return False
        candidate = _candidate(job, result)
        if reference is not None:
            _ensure_same_split(reference, candidate, spec)
        else:
            reference = candidate
        candidates.append(candidate)
        write_factorial_summary(
            args,
            spec,
            protocol,
            search="lhgcn-full-cartesian",
            candidates=candidates,
            expected_parameters=expected,
            axes=axes,
        )
    write_factorial_summary(
        args,
        spec,
        protocol,
        search="lhgcn-full-cartesian",
        candidates=candidates,
        expected_parameters=expected,
        axes=axes,
    )
    return True


def execute_practical_joint(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    lock_fd: int,
    budget: list[int],
    matrix_dim: int = 8,
) -> bool:
    anchor = load_grid_winner(args, spec, protocol, matrix_dim)
    anchor_parameters = _sl8_parameters_from_candidate(anchor)
    jobs = practical_joint_jobs(args, spec, anchor_parameters)
    expected_parameters = [
        asdict(parameters) for parameters in practical_joint_parameters(anchor_parameters)
    ]
    candidates: list[dict[str, Any]] = [dict(anchor)]
    axes = {
        "fixed_grid_winner_gcn_layers": (anchor_parameters.gcn_layers,),
        "fixed_grid_winner_train_batch_size": (anchor_parameters.train_batch_size,),
        "schatten_p": SL8_GEOMETRY_VALUES,
        "loss_margin": SL8_MARGIN_VALUES,
    }
    for job in jobs:
        result = execute_selection_job(
            args,
            spec,
            protocol,
            job,
            environment=environment,
            lock_fd=lock_fd,
            budget=budget,
        )
        if result is None:
            write_factorial_summary(
                args,
                spec,
                protocol,
                search=f"sl{matrix_dim}-practical",
                candidates=candidates,
                expected_parameters=expected_parameters,
                axes=axes,
            )
            return False
        candidate = _candidate(job, result)
        _ensure_same_split(anchor, candidate, spec)
        candidates.append(candidate)
        write_factorial_summary(
            args,
            spec,
            protocol,
            search=f"sl{matrix_dim}-practical",
            candidates=candidates,
            expected_parameters=expected_parameters,
            axes=axes,
        )
    write_factorial_summary(
        args,
        spec,
        protocol,
        search=f"sl{matrix_dim}-practical",
        candidates=candidates,
        expected_parameters=expected_parameters,
        axes=axes,
    )
    return True


def execute_sl8_full_cartesian(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    lock_fd: int,
    budget: list[int],
    matrix_dim: int = 8,
) -> bool:
    jobs = full_cartesian_sl8_jobs(args, spec, matrix_dim)
    expected = [job.parameters for job in jobs]
    control_candidates = _load_existing_control_candidates(args, spec, protocol)
    reference = _first_split_reference(control_candidates)
    candidates: list[dict[str, Any]] = []
    axes = {
        "gcn_layers": LAYERS,
        "train_batch_size": SL_BATCH_SIZES[matrix_dim],
        "schatten_p": SL8_GEOMETRY_VALUES,
        "loss_margin": SL8_MARGIN_VALUES,
    }
    for job in jobs:
        result = execute_selection_job(
            args,
            spec,
            protocol,
            job,
            environment=environment,
            lock_fd=lock_fd,
            budget=budget,
        )
        if result is None:
            write_factorial_summary(
                args,
                spec,
                protocol,
                search=f"sl{matrix_dim}-full-cartesian",
                candidates=candidates,
                expected_parameters=expected,
                axes=axes,
            )
            return False
        candidate = _candidate(job, result)
        if reference is not None:
            _ensure_same_split(reference, candidate, spec)
        else:
            reference = candidate
        candidates.append(candidate)
        write_factorial_summary(
            args,
            spec,
            protocol,
            search=f"sl{matrix_dim}-full-cartesian",
            candidates=candidates,
            expected_parameters=expected,
            axes=axes,
        )
    write_factorial_summary(
        args,
        spec,
        protocol,
        search=f"sl{matrix_dim}-full-cartesian",
        candidates=candidates,
        expected_parameters=expected,
        axes=axes,
    )
    return True


def _sl8_parameters_from_candidate(candidate: Mapping[str, Any]) -> SL8Parameters:
    values = candidate.get("parameters")
    if not isinstance(values, Mapping):
        raise ValueError("SL8 candidate has no parameter mapping")
    try:
        parameters = SL8Parameters(
            gcn_layers=int(values["gcn_layers"]),
            train_batch_size=int(values["train_batch_size"]),
            matrix_dim=int(values.get("matrix_dim", 8)),
            learning_rate=float(values["learning_rate"]),
            loss_margin=float(values["loss_margin"]),
            coord_clip=float(values["coord_clip"]),
            schatten_p=values["schatten_p"],
            weight_decay=float(values["weight_decay"]),
            lhgcn_include_self=bool(values["lhgcn_include_self"]),
            lhgcn_self_loop_weight=float(values["lhgcn_self_loop_weight"]),
            sl_scale=float(values["sl_scale"]),
            negative_samples=int(values["negative_samples"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid SL8 candidate parameters") from error
    parameters.validate()
    return parameters


def execute_tuning(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    lock_fd: int,
    budget: list[int],
) -> bool:
    anchor = load_grid_winner(args, spec, protocol)
    completed_stages: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(TUNING_PROFILES[args.tuning_profile], 1):
        anchor_parameters = _sl8_parameters_from_candidate(anchor)
        jobs = tuning_jobs(args, spec, stage_index, stage, anchor_parameters)
        candidates: list[dict[str, Any]] = []
        for job in jobs:
            result = execute_selection_job(
                args,
                spec,
                protocol,
                job,
                environment=environment,
                lock_fd=lock_fd,
                budget=budget,
            )
            if result is None:
                write_tuning_stage_summary(
                    args,
                    spec,
                    protocol,
                    stage_index,
                    stage,
                    anchor,
                    candidates,
                    len(jobs),
                )
                write_tuning_summary(
                    args, spec, protocol, completed_stages, anchor
                )
                return False
            candidate = _candidate(job, result)
            _ensure_same_split(anchor, candidate, spec)
            candidates.append(candidate)
            write_tuning_stage_summary(
                args,
                spec,
                protocol,
                stage_index,
                stage,
                anchor,
                candidates,
                len(jobs),
            )
        stage_summary = write_tuning_stage_summary(
            args,
            spec,
            protocol,
            stage_index,
            stage,
            anchor,
            candidates,
            len(jobs),
        )
        if stage_summary["state"] != "complete":
            raise RuntimeError("internal error: fully iterated tuning stage is incomplete")
        anchor = dict(stage_summary["winner"])
        completed_stages.append(
            {
                "stage_index": stage_index,
                "stage_key": stage.key,
                "summary_file": str(
                    tuning_stage_summary_path(args, spec, stage_index, stage).resolve()
                ),
                "summary_signature_sha256": stage_summary["signature_sha256"],
                "winner": anchor,
            }
        )
    write_tuning_summary(args, spec, protocol, completed_stages, anchor)
    return True


def load_final_test(
    path: Path,
    *,
    spec: DatasetSpec,
    selection: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _load_mapping(path)
    if payload.get("dataset") != spec.dataset or payload.get("evaluation_eval_mode") != "full":
        raise ValueError(f"wrong dataset/evaluation mode in final test: {path}")
    if payload.get("valid_result") is not None or not isinstance(payload.get("test_result"), Mapping):
        raise ValueError(f"final-test artifact must contain test only: {path}")
    if payload.get("split_fingerprints") != selection.get("split_fingerprints"):
        raise RuntimeError(f"final-test split differs from selection: {path}")
    if Path(str(payload.get("checkpoint_file", ""))).resolve() != Path(
        str(selection["checkpoint_file"])
    ).resolve():
        raise ValueError(f"final-test checkpoint differs from selection: {path}")
    if payload.get("split_fingerprints_match") is not True:
        raise ValueError(f"final-test evaluator did not confirm the split: {path}")
    return payload


def execute_final_tests(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    lock_fd: int,
    budget: list[int],
    special_linear_only: bool = False,
) -> bool:
    selections: list[tuple[str, Path, dict[str, Any]]] = []
    if not special_linear_only:
        for job in control_jobs(args, spec, protocol):
            if job.label == "lhgcn":
                continue
            result = load_selection(
                job.result_path,
                args=args,
                spec=spec,
                protocol=protocol,
                job=job,
            )
            selections.append((job.label, job.result_path, result))
        lhgcn_path, lhgcn_selection = load_selected_lhgcn(args, spec, protocol)
        selections.insert(1, ("lhgcn", lhgcn_path, lhgcn_selection))
        if 16 in args.sl_dims:
            capacity_job = lhgcn_capacity_job(args, spec, protocol)
            capacity_selection = load_selection(
                capacity_job.result_path,
                args=args,
                spec=spec,
                protocol=protocol,
                job=capacity_job,
            )
            selections.insert(
                2,
                (
                    "lhgcn-capacity-256",
                    capacity_job.result_path,
                    capacity_selection,
                ),
            )
    for matrix_dim in args.sl_dims:
        winner = load_selected_sl8_winner(args, spec, protocol, matrix_dim)
        selection_path = Path(winner["selection_result_file"]).resolve()
        selection = _load_mapping(selection_path)
        validate_candidate_record(winner, spec=spec, protocol=protocol)
        selections.append(
            (f"sl{matrix_dim}lhgcn", selection_path, selection)
        )
    reference = selections[0][2]
    for _, _, selection in selections[1:]:
        _ensure_same_split(reference, selection, spec)

    root = _dataset_root(args, spec) / "final-test"
    for label, selection_path, selection in selections:
        result_path = root / f"{label}.json"
        if load_final_test(result_path, spec=spec, selection=selection) is not None:
            print(f"SKIP {spec.slug} final-test {label}", flush=True)
            continue
        if budget[0] == 0:
            return False
        command = final_test_command(
            args,
            selection_path,
            Path(str(selection["checkpoint_file"])),
            result_path,
            matrix_dim=(16 if selection.get("model") == "SL16LHGCN" else 8),
        )
        print(f"START {spec.slug} final-test {label}", flush=True)
        _run_and_tee(
            command,
            log_path=root / "logs" / f"{label}.log",
            cwd=args.repo,
            environment=environment,
            lock_fd=lock_fd,
        )
        if load_final_test(result_path, spec=spec, selection=selection) is None:
            raise RuntimeError(f"final test did not produce a valid artifact: {result_path}")
        budget[0] -= 1
    return True


def _audit_data(args: argparse.Namespace, spec: DatasetSpec) -> dict[str, Any]:
    if args.skip_data_audit:
        return {
            "status": "skipped-explicit-dry-run-only",
            "filtered_reference": spec.filtered.json(),
        }
    source = audit_source_file(args.data_root, spec)
    # Book's released YAML is known to encode the wrong 5-core protocol, so its
    # exact post-filter cardinalities are a mandatory pre-training gate.  The
    # other five (already aligned) datasets keep the opt-in deep audit.
    filtered = (
        audit_paper_filtered_dataset(args, spec)
        if args.deep_data_audit or spec.slug == "amazon-book"
        else None
    )
    return {"source": source, "filtered": filtered}


@contextlib.contextmanager
def _isolated_recbole_argv() -> Iterable[None]:
    previous = sys.argv
    sys.argv = [previous[0]]
    try:
        yield
    finally:
        sys.argv = previous


def audit_paper_filtered_dataset(
    args: argparse.Namespace, spec: DatasetSpec
) -> dict[str, Any]:
    """Instantiate the effective protocol, including Book's 8-core correction."""

    if spec.slug != "amazon-book":
        return audit_filtered_dataset(args.repo, args.data_root, spec)

    from recbole_gnn.config import Config
    from recbole_gnn.utils import create_dataset

    with _isolated_recbole_argv():
        config = Config(
            model="RecFormer",
            dataset=spec.dataset,
            config_file_list=[str(path) for path in protocol_config_files(args.repo, spec)],
            config_dict={"data_path": str(args.data_root), "use_gpu": False},
        )
        dataset = create_dataset(config)
    actual = {
        "framework_users": int(dataset.user_num),
        "framework_items": int(dataset.item_num),
        "token_users": int(dataset.user_num) - 1,
        "token_items": int(dataset.item_num) - 1,
        "interactions": int(len(dataset)),
    }
    if actual != AMAZON_BOOK_PAPER_COUNTS:
        raise ValueError(
            "Amazon Book filtered cardinalities do not match the paper's iterative "
            f"8-core protocol: expected={AMAZON_BOOK_PAPER_COUNTS}, actual={actual}"
        )
    return {
        "status": "accepted-exact-paper-8-core",
        **actual,
        "protocol_configs": [
            str(path.relative_to(args.repo))
            for path in protocol_config_files(args.repo, spec)
        ],
    }


def _job_plan(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    job: SelectionJob,
) -> dict[str, Any]:
    complete = completed_selection(
        job.result_path,
        args=args,
        spec=spec,
        protocol=protocol,
        job=job,
    )
    command = selection_command(args, spec, job)
    return {
        "kind": job.kind,
        "label": job.label,
        "model": job.model,
        "parameters": dict(job.parameters),
        "status": "skip-complete" if complete is not None else "run",
        "result_file": str(job.result_path),
        "command": command,
        "child_cuda_visible_devices": PHYSICAL_GPU,
        "child_config_gpu_id": PHYSICAL_GPU,
        "test_evaluated": False,
    }


def _phase_control_labels(phase: str) -> set[str] | None:
    mapping = {
        "lightgcn": {"lightgcn"},
        "lhgcn": {"lhgcn"},
        "recformer": {"recformer"},
        "controls": None,
        "all": None,
    }
    return mapping.get(phase)


def _dry_sl_search_plan(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    matrix_dim: int,
) -> dict[str, Any]:
    dimension_metadata = {
        "matrix_dim": matrix_dim,
        "model": SL_MODEL_NAMES[matrix_dim],
        "raw_dimension": matrix_dim**2,
        "intrinsic_dimension": matrix_dim**2 - 1,
        "raw_entity_parameter_ratio_vs_sl8": (matrix_dim / 8) ** 2,
        "dense_cubic_compute_proxy_vs_sl8": (matrix_dim / 8) ** 3,
        "train_batch_sizes": list(SL_BATCH_SIZES[matrix_dim]),
        "evaluation_chunks": SL_EVAL_CHUNKS[matrix_dim],
        "curve": "excluded-dead-key",
    }
    if args.sl8_search == "full-cartesian":
        full_jobs = full_cartesian_sl8_jobs(args, spec, matrix_dim)
        return {
            **dimension_metadata,
            "design": "full primary effective-parameter Cartesian",
            "axes": {
                "gcn_layers": list(LAYERS),
                "train_batch_size": list(SL_BATCH_SIZES[matrix_dim]),
                "schatten_p": list(SL8_GEOMETRY_VALUES),
                "loss_margin": list(SL8_MARGIN_VALUES),
            },
            "job_count": len(full_jobs),
            "first_job": _job_plan(args, spec, protocol, full_jobs[0]),
            "last_job": _job_plan(args, spec, protocol, full_jobs[-1]),
            "all_jobs_omitted_from_plan": True,
            "status": "ready",
        }
    if args.sl8_search == "practical":
        plan: dict[str, Any] = {
            **dimension_metadata,
            "design": "blocked-factorial: LxB then joint effective-geometry x margin",
            "block_1": {
                "axes": {
                    "gcn_layers": list(LAYERS),
                    "train_batch_size": list(SL_BATCH_SIZES[matrix_dim]),
                },
                "job_count": len(grid_parameters(matrix_dim)),
            },
            "block_2": {
                "axes": {
                    "schatten_p": list(SL8_GEOMETRY_VALUES),
                    "loss_margin": list(SL8_MARGIN_VALUES),
                },
                "candidate_count_including_carried_anchor": len(
                    SL8_GEOMETRY_VALUES
                )
                * len(SL8_MARGIN_VALUES),
                "maximum_new_jobs": len(SL8_GEOMETRY_VALUES)
                * len(SL8_MARGIN_VALUES)
                - 1,
            },
            "total_maximum_jobs": len(grid_parameters(matrix_dim))
            + len(SL8_GEOMETRY_VALUES) * len(SL8_MARGIN_VALUES)
            - 1,
            "status": "blocked-until-complete-layer-batch-winner",
        }
        if matrix_dim == 8:
            # Kept for consumers of the original single-dimension dry plan.
            plan["total_maximum_sl8_jobs"] = plan["total_maximum_jobs"]
        try:
            winner = load_grid_winner(args, spec, protocol, matrix_dim)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return plan
        anchor = _sl8_parameters_from_candidate(winner)
        practical_jobs = practical_joint_jobs(args, spec, anchor)
        plan["status"] = "layer-batch-complete-joint-block-ready"
        plan["grid_winner"] = winner
        plan["first_joint_job"] = _job_plan(
            args, spec, protocol, practical_jobs[0]
        )
        return plan

    stages = TUNING_PROFILES[args.tuning_profile]
    plan = {
        **dimension_metadata,
        "profile": args.tuning_profile,
        "design": "greedy stage-carried validation search",
        "stages": [
            {"index": index, "key": stage.key, "values": list(stage.values)}
            for index, stage in enumerate(stages, 1)
        ],
        "maximum_new_trials_after_grid": staged_new_trial_count(
            args.tuning_profile
        ),
        "status": "blocked-until-complete-grid-winner",
    }
    try:
        winner = load_grid_winner(args, spec, protocol, matrix_dim)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return plan
    anchor = _sl8_parameters_from_candidate(winner)
    first = stages[0]
    plan["status"] = "grid-complete-first-stage-ready"
    plan["grid_winner"] = winner
    plan["first_stage_jobs"] = [
        _job_plan(args, spec, protocol, job)
        for job in tuning_jobs(args, spec, 1, first, anchor)
    ]
    return plan


def dry_run_plan(
    args: argparse.Namespace,
    selected: Sequence[DatasetSpec],
    protocols: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    for spec in selected:
        protocol = protocols[spec.slug]
        jobs: list[dict[str, Any]] = []
        if args.phase in {"lightgcn", "lhgcn", "recformer", "controls", "all"}:
            labels = _phase_control_labels(args.phase)
            jobs.extend(
                _job_plan(args, spec, protocol, job)
                for job in control_jobs(args, spec, protocol)
                if labels is None or job.label in labels
            )
        if args.phase == "lhgcn-capacity" or (
            args.phase == "all" and 16 in args.sl_dims
        ):
            jobs.append(
                _job_plan(
                    args,
                    spec,
                    protocol,
                    lhgcn_capacity_job(args, spec, protocol),
                )
            )
        if args.phase == "grid" or (
            args.phase in {"all", "sl-all"}
            and args.sl8_search in {"practical", "staged"}
        ):
            for matrix_dim in args.sl_dims:
                jobs.extend(
                    _job_plan(args, spec, protocol, job)
                    for job in grid_jobs(args, spec, matrix_dim)
                )

        lhgcn_factorial: dict[str, Any] | None = None
        if args.phase == "lhgcn-grid" or (
            args.phase == "all" and args.lhgcn_search == "full-cartesian"
        ):
            lhgcn_jobs = lhgcn_full_cartesian_jobs(args, spec, protocol)
            lhgcn_factorial = {
                "design": "full L x B x active-curve x margin Cartesian",
                "axes": {
                    "gcn_layers": list(LAYERS),
                    "train_batch_size": list(BATCH_SIZES),
                    "curve": list(LHGCN_CURVE_VALUES),
                    "margin": list(LHGCN_MARGIN_VALUES),
                },
                "job_count": len(lhgcn_jobs),
                "curve_activity": protocol["parameter_activity"]["lhgcn"]["curve"],
                "first_job": _job_plan(args, spec, protocol, lhgcn_jobs[0]),
                "last_job": _job_plan(args, spec, protocol, lhgcn_jobs[-1]),
                "all_jobs_omitted_from_plan": True,
                "test_evaluated": False,
            }

        tuning: dict[str, Any] | None = None
        sl_searches: dict[str, Any] | None = None
        if args.phase in {"tune", "all", "sl-all"}:
            sl_searches = {
                str(matrix_dim): _dry_sl_search_plan(
                    args, spec, protocol, matrix_dim
                )
                for matrix_dim in args.sl_dims
            }
            # Backward-compatible convenience for the default one-dimension plan.
            if len(args.sl_dims) == 1:
                tuning = sl_searches[str(args.sl_dims[0])]

        final_test: dict[str, Any] | None = None
        if args.phase in {"final-test", "sl-final-test"}:
            final_test = {
                "status": "requires complete validation-selected checkpoints",
                "scope": (
                    "special-linear-only"
                    if args.phase == "sl-final-test"
                    else "all-selected-models"
                ),
                "test_is_separate_from_selection": True,
            }
            try:
                selections = []
                if args.phase != "sl-final-test":
                    for job in control_jobs(args, spec, protocol):
                        if job.label == "lhgcn":
                            continue
                        result = load_selection(
                            job.result_path,
                            args=args,
                            spec=spec,
                            protocol=protocol,
                            job=job,
                        )
                        selections.append((job.label, job.result_path, result))
                    lhgcn_path, lhgcn_selection = load_selected_lhgcn(
                        args, spec, protocol
                    )
                    selections.insert(1, ("lhgcn", lhgcn_path, lhgcn_selection))
                    if 16 in args.sl_dims:
                        capacity_job = lhgcn_capacity_job(args, spec, protocol)
                        capacity_selection = load_selection(
                            capacity_job.result_path,
                            args=args,
                            spec=spec,
                            protocol=protocol,
                            job=capacity_job,
                        )
                        selections.insert(
                            2,
                            (
                                "lhgcn-capacity-256",
                                capacity_job.result_path,
                                capacity_selection,
                            ),
                        )
                for matrix_dim in args.sl_dims:
                    winner = load_selected_sl8_winner(
                        args, spec, protocol, matrix_dim
                    )
                    selection_path = Path(
                        winner["selection_result_file"]
                    ).resolve()
                    selections.append(
                        (
                            f"sl{matrix_dim}lhgcn",
                            selection_path,
                            _load_mapping(selection_path),
                        )
                    )
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                pass
            else:
                final_jobs = []
                for label, selection_path, selection in selections:
                    result_path = _dataset_root(args, spec) / "final-test" / f"{label}.json"
                    final_jobs.append(
                        {
                            "label": label,
                            "status": (
                                "skip-complete"
                                if load_final_test(
                                    result_path, spec=spec, selection=selection
                                )
                                else "run"
                            ),
                            "command": final_test_command(
                                args,
                                selection_path,
                                Path(str(selection["checkpoint_file"])),
                                result_path,
                                matrix_dim=(
                                    16
                                    if selection.get("model") == "SL16LHGCN"
                                    else 8
                                ),
                            ),
                        }
                    )
                final_test = {"status": "ready", "jobs": final_jobs}

        datasets.append(
            {
                "slug": spec.slug,
                "dataset": spec.dataset,
                "priority_index": PAPER_DATASET_SLUGS.index(spec.slug),
                "protocol": protocol,
                "data_audit": _audit_data(args, spec),
                "selection_jobs": jobs,
                "lhgcn_factorial": lhgcn_factorial,
                "sl_searches": sl_searches,
                "tuning": tuning,
                "final_test": final_test,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "dry_run": True,
        "phase": args.phase,
        "sl8_search": args.sl8_search,
        "lhgcn_search": args.lhgcn_search,
        "sl_dims": list(args.sl_dims),
        "parameter_activity": validate_parameter_activity(args.repo),
        "single_physical_gpu": PHYSICAL_GPU,
        "child_cuda_visible_devices": PHYSICAL_GPU,
        "child_config_gpu_id": PHYSICAL_GPU,
        "child_torch_device_after_mask": "cuda:0",
        "strict_serial": True,
        "lock_file": str(args.lock_file),
        "dataset_order": [spec.slug for spec in selected],
        "selection_metric": SELECTION_METRIC,
        "test_evaluated": args.phase in {"final-test", "sl-final-test"},
        "datasets": datasets,
    }


def _safe_completed_control_records(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job in control_jobs(args, spec, protocol):
        if not job.result_path.is_file():
            records.append({"label": job.label, "state": "pending"})
            continue
        try:
            result = load_selection(
                job.result_path,
                args=args,
                spec=spec,
                protocol=protocol,
                job=job,
            )
        except Exception as error:  # Summary reports; execution still fails loudly.
            records.append(
                {"label": job.label, "state": "invalid", "error": str(error)}
            )
        else:
            records.append(
                {
                    "label": job.label,
                    "state": "complete",
                    "selection": _candidate(job, result),
                }
            )
    return records


def write_dataset_summary(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    controls = _safe_completed_control_records(args, spec, protocol)

    capacity_requested = 16 in args.sl_dims and args.phase not in {
        "sl-all",
        "sl-final-test",
    }
    capacity_control: dict[str, Any] = {
        "state": "not-requested" if not capacity_requested else "pending",
        "capacity_match_target": "SL16LHGCN",
    }
    if capacity_requested:
        capacity_job = lhgcn_capacity_job(args, spec, protocol)
        if capacity_job.result_path.is_file():
            try:
                capacity_result = load_selection(
                    capacity_job.result_path,
                    args=args,
                    spec=spec,
                    protocol=protocol,
                    job=capacity_job,
                )
                capacity_control = {
                    "state": "complete",
                    "capacity_match_target": "SL16LHGCN",
                    "selection": _candidate(capacity_job, capacity_result),
                }
            except Exception as error:
                capacity_control = {
                    "state": "invalid",
                    "capacity_match_target": "SL16LHGCN",
                    "error": str(error),
                }

    lhgcn_factorial: dict[str, Any] = {
        "state": "not-requested" if args.lhgcn_search == "matched" else "pending",
        "design": args.lhgcn_search,
    }
    lhgcn_path = factorial_summary_path(args, spec, "lhgcn-full-cartesian")
    if args.lhgcn_search == "full-cartesian" and lhgcn_path.is_file():
        try:
            payload = _load_signed_payload(lhgcn_path)
            lhgcn_factorial = {
                "state": payload.get("state", "invalid"),
                "design": args.lhgcn_search,
                "completed_trials": payload.get("completed_unique_trials", 0),
                "expected_trials": payload.get("expected_trials", 375),
                "winner": payload.get("winner"),
                "summary_file": str(lhgcn_path.resolve()),
            }
        except Exception as error:
            lhgcn_factorial = {
                "state": "invalid",
                "design": args.lhgcn_search,
                "error": str(error),
            }

    special_linear: dict[str, Any] = {}
    for matrix_dim in args.sl_dims:
        grid_path = grid_summary_path(args, spec, matrix_dim)
        grid: dict[str, Any] = {
            "state": (
                "not-requested"
                if args.sl8_search == "full-cartesian"
                else "pending"
            ),
            "expected_trials": len(grid_parameters(matrix_dim)),
        }
        if grid_path.is_file():
            try:
                grid_payload = _load_signed_payload(grid_path)
                grid = {
                    "state": grid_payload.get("state", "invalid"),
                    "completed_trials": grid_payload.get("completed_trials", 0),
                    "expected_trials": grid_payload.get(
                        "expected_trials", len(grid_parameters(matrix_dim))
                    ),
                    "winner": grid_payload.get("winner"),
                    "summary_file": str(grid_path.resolve()),
                }
            except Exception as error:
                grid = {"state": "invalid", "error": str(error)}

        tuning: dict[str, Any] = {
            "state": "pending",
            "design": args.sl8_search,
        }
        if args.sl8_search == "staged":
            selected_summary_path = tuning_summary_path(args, spec)
        elif args.sl8_search == "full-cartesian":
            selected_summary_path = factorial_summary_path(
                args, spec, f"sl{matrix_dim}-full-cartesian"
            )
        else:
            selected_summary_path = factorial_summary_path(
                args, spec, f"sl{matrix_dim}-practical"
            )
        if selected_summary_path.is_file():
            try:
                tune_payload = _load_signed_payload(selected_summary_path)
                tuning = {
                    "state": tune_payload.get("state", "invalid"),
                    "design": args.sl8_search,
                    "profile": (
                        args.tuning_profile if args.sl8_search == "staged" else None
                    ),
                    "completed_stage_count": len(
                        tune_payload.get("completed_stages", [])
                    ),
                    "completed_trials": tune_payload.get(
                        "completed_unique_trials"
                    ),
                    "expected_trials": tune_payload.get("expected_trials"),
                    "final_validation_winner": tune_payload.get(
                        "final_validation_winner", tune_payload.get("winner")
                    ),
                    "latest_complete_stage_winner": tune_payload.get(
                        "latest_complete_stage_winner",
                        tune_payload.get("provisional_winner"),
                    ),
                    "summary_file": str(selected_summary_path.resolve()),
                }
            except Exception as error:
                tuning = {
                    "state": "invalid",
                    "design": args.sl8_search,
                    "error": str(error),
                }
        special_linear[str(matrix_dim)] = {
            "model": SL_MODEL_NAMES[matrix_dim],
            "matrix_dim": matrix_dim,
            "raw_dimension": matrix_dim**2,
            "intrinsic_dimension": matrix_dim**2 - 1,
            "raw_entity_parameter_ratio_vs_sl8": (matrix_dim / 8) ** 2,
            "dense_cubic_compute_proxy_vs_sl8": (matrix_dim / 8) ** 3,
            "parameter_budget_compared_separately": True,
            "grid": grid,
            "search": tuning,
        }

    final_tests = []
    final_labels = (
        []
        if args.phase == "sl-final-test"
        else ["lightgcn", "lhgcn", "recformer"]
    )
    if 16 in args.sl_dims and args.phase != "sl-final-test":
        final_labels.append("lhgcn-capacity-256")
    final_labels.extend(f"sl{matrix_dim}lhgcn" for matrix_dim in args.sl_dims)
    for label in final_labels:
        path = _dataset_root(args, spec) / "final-test" / f"{label}.json"
        final_tests.append(
            {"label": label, "state": "complete" if path.is_file() else "pending", "file": str(path)}
        )
    special_linear_complete = all(
            (
                args.sl8_search == "full-cartesian"
                or dimension["grid"].get("state") == "complete"
            )
            and dimension["search"].get("state") == "complete"
            for dimension in special_linear.values()
    )
    selection_complete = special_linear_complete and (
        args.phase in {"sl-all", "sl-final-test"}
        or (
            all(record["state"] == "complete" for record in controls)
            and (
                args.lhgcn_search == "matched"
                or lhgcn_factorial.get("state") == "complete"
            )
            and (
                16 not in args.sl_dims
                or capacity_control.get("state") == "complete"
            )
        )
    )
    core = {
        "schema_version": SCHEMA_VERSION,
        "dataset_slug": spec.slug,
        "dataset": spec.dataset,
        "protocol_signature_sha256": protocol["signature_sha256"],
        "selection_state": "complete" if selection_complete else "incomplete",
        "selection_metric": SELECTION_METRIC,
        "controls": controls,
        "lhgcn_capacity_control": capacity_control,
        "lhgcn_factorial": lhgcn_factorial,
        "sl_dims": list(args.sl_dims),
        "special_linear": special_linear,
        # Compatibility aliases for the original SL8-only summary consumers.
        "sl8_grid": special_linear.get("8", {}).get("grid"),
        "sl8_tuning": special_linear.get("8", {}).get("search"),
        "final_tests": final_tests,
        "test_evaluated_during_selection": False,
    }
    payload = _signed_payload(core)
    _atomic_json(_dataset_root(args, spec) / "summary.json", payload)
    return payload


def write_global_summary(
    args: argparse.Namespace,
    selected: Sequence[DatasetSpec],
    protocols: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    summaries = [
        write_dataset_summary(args, spec, protocols[spec.slug]) for spec in selected
    ]
    core = {
        "schema_version": SCHEMA_VERSION,
        "phase": args.phase,
        "tuning_profile": args.tuning_profile,
        "sl8_search": args.sl8_search,
        "lhgcn_search": args.lhgcn_search,
        "sl_dims": list(args.sl_dims),
        "single_physical_gpu": PHYSICAL_GPU,
        "dataset_order": [spec.slug for spec in selected],
        "selection_state": (
            "complete"
            if all(summary["selection_state"] == "complete" for summary in summaries)
            else "incomplete"
        ),
        "selection_metric": SELECTION_METRIC,
        "datasets": summaries,
        "test_evaluated_during_selection": False,
    }
    payload = _signed_payload(core)
    _atomic_json(args.output_root / "summary.json", payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--data-root", type=Path, default=repo / "dataset")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="paper dataset slugs/names; default keeps the paper-priority order",
    )
    parser.add_argument(
        "--phase",
        choices=(
            "lightgcn",
            "lhgcn",
            "recformer",
            "controls",
            "lhgcn-capacity",
            "lhgcn-grid",
            "grid",
            "tune",
            "sl-all",
            "all",
            "sl-final-test",
            "final-test",
        ),
        default="all",
        help=(
            "sl-all runs only requested SL8/SL16 searches (recommended); all also "
            "runs legacy controls; sl-final-test tests only selected SL models"
        ),
    )
    parser.add_argument(
        "--sl-dims",
        type=int,
        nargs="+",
        choices=(8, 16),
        default=[8],
        help="special-linear dimensions to search independently (default: 8)",
    )
    parser.add_argument(
        "--sl-search",
        "--sl8-search",
        dest="sl8_search",
        choices=("practical", "staged", "full-cartesian"),
        default="practical",
        help=(
            "practical: LxB then joint Schatten-p x margin; staged: legacy "
            "one-axis search; full-cartesian: LxBxSchatten-pxmargin"
        ),
    )
    parser.add_argument(
        "--lhgcn-search",
        choices=("matched", "full-cartesian"),
        default="matched",
        help="optionally search LHGCN LxBxcurvexmargin as an independent control",
    )
    parser.add_argument(
        "--tuning-profile",
        choices=tuple(TUNING_PROFILES),
        default="expanded",
        help="current mirrors the existing 21-job staged search; expanded searches more",
    )
    parser.add_argument("--gpu-id", default=PHYSICAL_GPU)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-new-jobs", type=int)
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--deep-data-audit", action="store_true")
    parser.add_argument(
        "--skip-data-audit",
        action="store_true",
        help="allowed only with --dry-run on a planning machine without all datasets",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.repo = args.repo.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.gpu_id = _normalise_gpu(args.gpu_id)
    args.lock_file = (
        args.lock_file.expanduser().resolve()
        if args.lock_file is not None
        else default_lock_path()
    )
    if args.max_new_jobs is not None and args.max_new_jobs < 0:
        raise ValueError("--max-new-jobs must be non-negative")
    if args.skip_data_audit and not args.dry_run:
        raise ValueError("--skip-data-audit is permitted only with --dry-run")
    selected = select_paper_datasets(args.datasets)
    protocols = {
        spec.slug: validate_pipeline_protocol(args.repo, spec) for spec in selected
    }

    args.sl_dims = tuple(dict.fromkeys(args.sl_dims))
    if args.sl8_search == "staged" and any(dim != 8 for dim in args.sl_dims):
        raise ValueError("legacy staged search supports only --sl-dims 8")

    if args.dry_run:
        print(json.dumps(dry_run_plan(args, selected, protocols), indent=2))
        return 0

    # Set the mask before a deep audit can import torch.  Training children
    # additionally receive --gpu_id=7 so old RecBole cannot undo the mask.
    os.environ["CUDA_VISIBLE_DEVICES"] = PHYSICAL_GPU
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = PHYSICAL_GPU
    budget = [args.max_new_jobs if args.max_new_jobs is not None else 2**63 - 1]

    with exclusive_gpu_lock(args.lock_file) as lock_fd:
        audits = {spec.slug: _audit_data(args, spec) for spec in selected}
        manifest_core = {
            "schema_version": SCHEMA_VERSION,
            "phase": args.phase,
            "tuning_profile": args.tuning_profile,
            "sl8_search": args.sl8_search,
            "lhgcn_search": args.lhgcn_search,
            "sl_dims": list(args.sl_dims),
            "job_counts_per_dataset": {
                "controls": (
                    3
                    if args.phase in {"controls", "all"}
                    else 1
                    if args.phase in {"lightgcn", "lhgcn", "recformer"}
                    else 0
                ),
                "lhgcn_capacity_256": (
                    1
                    if 16 in args.sl_dims
                    and args.phase in {"lhgcn-capacity", "all"}
                    else 0
                ),
                "lhgcn_full_cartesian": len(
                    lhgcn_full_cartesian_parameters(
                        LHGCNParameters(
                            **protocols[selected[0].slug]["lhgcn"][
                                "parameters_matched_to_recformer"
                            ]
                        )
                    )
                ),
                "special_linear_by_dimension": {
                    str(matrix_dim): {
                        "layer_batch": len(grid_parameters(matrix_dim)),
                        "practical_new_after_grid": len(SL8_GEOMETRY_VALUES)
                        * len(SL8_MARGIN_VALUES)
                        - 1,
                        "practical_total": len(grid_parameters(matrix_dim))
                        + len(SL8_GEOMETRY_VALUES) * len(SL8_MARGIN_VALUES)
                        - 1,
                        "primary_full_cartesian": len(
                            full_cartesian_sl8_parameters(matrix_dim)
                        ),
                        "raw_dimension": matrix_dim**2,
                        "intrinsic_dimension": matrix_dim**2 - 1,
                    }
                    for matrix_dim in args.sl_dims
                },
            },
            "single_physical_gpu": PHYSICAL_GPU,
            "child_cuda_visible_devices": PHYSICAL_GPU,
            "child_config_gpu_id": PHYSICAL_GPU,
            "strict_serial": True,
            "lock_file": str(args.lock_file),
            "dataset_order": [spec.slug for spec in selected],
            "protocols": protocols,
            "data_audits": audits,
            "selection_metric": SELECTION_METRIC,
            "test_evaluated": args.phase in {"final-test", "sl-final-test"},
        }
        _atomic_json(args.output_root / "manifest.json", _signed_payload(manifest_core))

        paused = False
        for spec in selected:
            protocol = protocols[spec.slug]
            complete = True
            if args.phase in {"lightgcn", "lhgcn", "recformer", "controls", "all"}:
                complete = execute_controls(
                    args,
                    spec,
                    protocol,
                    environment=environment,
                    lock_fd=lock_fd,
                    budget=budget,
                    labels=_phase_control_labels(args.phase),
                )
            if complete and (
                args.phase == "lhgcn-capacity"
                or (args.phase == "all" and 16 in args.sl_dims)
            ):
                capacity_result = execute_selection_job(
                    args,
                    spec,
                    protocol,
                    lhgcn_capacity_job(args, spec, protocol),
                    environment=environment,
                    lock_fd=lock_fd,
                    budget=budget,
                )
                complete = capacity_result is not None
            if complete and (
                args.phase == "lhgcn-grid"
                or (args.phase == "all" and args.lhgcn_search == "full-cartesian")
            ):
                complete = execute_lhgcn_full_cartesian(
                    args,
                    spec,
                    protocol,
                    environment=environment,
                    lock_fd=lock_fd,
                    budget=budget,
                )
            if complete and (
                args.phase == "grid"
                or (
                    args.phase in {"all", "sl-all"}
                    and args.sl8_search in {"practical", "staged"}
                )
            ):
                for matrix_dim in args.sl_dims:
                    complete = execute_grid(
                        args,
                        spec,
                        protocol,
                        environment=environment,
                        lock_fd=lock_fd,
                        budget=budget,
                        matrix_dim=matrix_dim,
                    )
                    if not complete:
                        break
            if complete and args.phase in {"tune", "all", "sl-all"}:
                if args.sl8_search == "practical":
                    for matrix_dim in args.sl_dims:
                        complete = execute_practical_joint(
                            args,
                            spec,
                            protocol,
                            environment=environment,
                            lock_fd=lock_fd,
                            budget=budget,
                            matrix_dim=matrix_dim,
                        )
                        if not complete:
                            break
                elif args.sl8_search == "full-cartesian":
                    for matrix_dim in args.sl_dims:
                        complete = execute_sl8_full_cartesian(
                            args,
                            spec,
                            protocol,
                            environment=environment,
                            lock_fd=lock_fd,
                            budget=budget,
                            matrix_dim=matrix_dim,
                        )
                        if not complete:
                            break
                else:
                    complete = execute_tuning(
                        args,
                        spec,
                        protocol,
                        environment=environment,
                        lock_fd=lock_fd,
                        budget=budget,
                    )
            if complete and args.phase in {"final-test", "sl-final-test"}:
                complete = execute_final_tests(
                    args,
                    spec,
                    protocol,
                    environment=environment,
                    lock_fd=lock_fd,
                    budget=budget,
                    special_linear_only=args.phase == "sl-final-test",
                )
            write_dataset_summary(args, spec, protocol)
            if not complete:
                paused = True
                print("PAUSED_BY_MAX_NEW_JOBS", flush=True)
                break
        summary = write_global_summary(args, selected, protocols)

    print(
        f"PAPER_PIPELINE_SUMMARY={(args.output_root / 'summary.json').resolve()} "
        f"selection_state={summary['selection_state']} paused={str(paused).lower()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
