#!/usr/bin/env python3
"""Shardable Hgformer reproduction and geometry-only SL(8) experiments.

This driver deliberately treats each existing ``RecFormer_*.yaml`` file as
the dataset protocol authority.  SL(8) changes only the model, optimisation,
and memory-related settings; rating/k-core filters, seed, user-wise split,
metrics, top-k values, and the full-ranking candidate set are inherited from
the corresponding Hgformer configuration.

``--phase all`` trains validation-only RecFormer and SL(8) selections.  It
never evaluates test.  ``--phase final-test`` is a separate, explicit action
which evaluates the already selected checkpoints without retraining them.
Datasets, rather than individual trials, are sharded so prerequisites and
split fingerprints remain local to one server.

``--gpu-id`` names one physical CUDA card.  Every training child receives the
same value through both ``CUDA_VISIBLE_DEVICES`` and RecBole's ``--gpu_id``;
the vendored RecBole configuration otherwise rewrites the former.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

try:
    from slrec_experiments.dataset_registry import DATASET_BY_SLUG as SOURCE_BY_SLUG
except ModuleNotFoundError:  # Allows ``python slrec_experiments/<file>.py``.
    from dataset_registry import DATASET_BY_SLUG as SOURCE_BY_SLUG  # type: ignore

try:
    from slrec_experiments.tune_sl8_full_cd import (
        PAPER_PARAMETERS,
        Trial,
        build_stage_trials,
        run_and_tee,
    )
except ModuleNotFoundError:  # Allows ``python slrec_experiments/<file>.py``.
    from tune_sl8_full_cd import (  # type: ignore
        PAPER_PARAMETERS,
        Trial,
        build_stage_trials,
        run_and_tee,
    )


SCHEMA_VERSION = 1
SEED = 2024
FULL_METRICS = ("Recall", "NDCG")
FULL_TOPK = (5, 10, 20, 50)
FULL_SPLIT = {"RS": [0.8, 0.1, 0.1]}
SL8_CONFIG_NAME = "SLRecGraph_ablation_sl8.yaml"


@dataclass(frozen=True)
class CountRange:
    minimum: int
    maximum: int

    @classmethod
    def exact(cls, value: int) -> "CountRange":
        return cls(value, value)

    def accepts(self, value: int) -> bool:
        return self.minimum <= value <= self.maximum

    def json(self) -> int | list[int]:
        if self.minimum == self.maximum:
            return self.minimum
        return [self.minimum, self.maximum]


@dataclass(frozen=True)
class FilteredReference:
    users: CountRange
    items: CountRange
    interactions: CountRange
    count_basis: str = "token"

    def json(self) -> dict[str, Any]:
        return {
            "users": self.users.json(),
            "items": self.items.json(),
            "interactions": self.interactions.json(),
            "count_basis": self.count_basis,
        }


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    dataset: str
    recformer_config: str
    source_rows: int | None
    source_bytes: int | None
    source_sha256: str | None
    filtered: FilteredReference
    source_release: str
    douban_official: bool = False


def _registered_spec(
    slug: str,
    filtered: FilteredReference,
    *,
    douban_official: bool = False,
) -> DatasetSpec:
    """Adapt the unified source registry to this runner's legacy shape."""

    source = SOURCE_BY_SLUG[slug]
    return DatasetSpec(
        slug=source.slug,
        dataset=source.dataset,
        recformer_config=source.recformer_config,
        source_rows=source.raw_rows,
        # ``audit_source_file`` receives the prepared RecBole atomic file.
        source_bytes=source.atomic_bytes,
        source_sha256=source.atomic_sha256,
        filtered=filtered,
        source_release=source.release,
        douban_official=douban_official,
    )


def _exact(value: int) -> CountRange:
    return CountRange.exact(value)


# Source rows/bytes/hashes come from ``dataset_registry.py``.  This legacy
# seven-dataset runner retains the released per-dataset YAML filters (including
# the erroneous Book 5-core); ``run_paper_dataset_pipeline.py`` refuses that
# Book path and applies the exact 8-core paper overlay.  The ranges below are
# legacy-runner sanity references, not a second provenance registry.
DATASETS: tuple[DatasetSpec, ...] = (
    _registered_spec(
        "amazon-cd",
        FilteredReference(CountRange(66_316, 66_317), CountRange(58_868, 58_869), _exact(952_547), "token-or-framework"),
    ),
    _registered_spec(
        "amazon-movies",
        FilteredReference(CountRange(26_968, 26_969), CountRange(18_563, 18_564), _exact(762_957), "token-or-framework"),
    ),
    _registered_spec(
        "amazon-toy",
        FilteredReference(CountRange(15_528, 15_529), CountRange(9_696, 9_697), _exact(133_837), "token-or-framework"),
    ),
    _registered_spec(
        "amazon-book",
        FilteredReference(CountRange(210_000, 212_000), CountRange(163_000, 165_000), CountRange(5_060_000, 5_080_000), "rounded-reference"),
    ),
    _registered_spec(
        "douban-book",
        FilteredReference(_exact(18_085), _exact(33_067), _exact(809_248)),
        douban_official=True,
    ),
    _registered_spec(
        "douban-movie",
        FilteredReference(_exact(22_040), _exact(25_801), _exact(2_552_305)),
        douban_official=True,
    ),
    _registered_spec(
        "douban-music",
        FilteredReference(_exact(15_995), _exact(39_748), _exact(1_116_984)),
        douban_official=True,
    ),
)
DATASET_BY_SLUG = {item.slug: item for item in DATASETS}
DATASET_BY_NAME = {item.dataset.lower(): item for item in DATASETS}


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
        help="slugs or RecBole dataset names; comma-separated values are accepted",
    )
    parser.add_argument(
        "--phase",
        choices=("reproduce", "tune", "all", "final-test"),
        default="all",
        help="all means validation-only reproduction plus tuning; test is always separate",
    )
    parser.add_argument("--tuning-profile", choices=("paper", "core"), default="core")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval-step", type=int, default=50)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-new-jobs", type=int)
    parser.add_argument(
        "--deep-data-audit",
        action="store_true",
        help="also instantiate the filtered RecBole dataset and check U/I/R counts",
    )
    parser.add_argument(
        "--skip-data-audit",
        action="store_true",
        help="allowed only with --dry-run; emits a plan for machines without data",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def select_datasets(tokens: Sequence[str], shard_index: int, shard_count: int) -> tuple[DatasetSpec, ...]:
    if shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < count")

    flattened = [piece for token in tokens for piece in token.split(",") if piece]
    if not flattened or "all" in {piece.lower() for piece in flattened}:
        chosen = list(DATASETS)
    else:
        chosen = []
        seen: set[str] = set()
        for token in flattened:
            normalized = token.strip().lower()
            spec = DATASET_BY_SLUG.get(normalized) or DATASET_BY_NAME.get(normalized)
            if spec is None:
                raise ValueError(f"unknown dataset {token!r}; choices={list(DATASET_BY_SLUG)}")
            if spec.slug not in seen:
                chosen.append(spec)
                seen.add(spec.slug)
    return tuple(chosen[shard_index::shard_count])


def config_path(repo: Path, spec: DatasetSpec) -> Path:
    return repo / "baseline_config_fixed" / spec.recformer_config


def sl8_config_path(repo: Path) -> Path:
    return repo / "baseline_config_fixed" / SL8_CONFIG_NAME


def atomic_path(data_root: Path, spec: DatasetSpec) -> Path:
    return data_root / spec.dataset / f"{spec.dataset}.inter"


def _file_sha256_and_lines(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    newline_count = 0
    final_byte = b""
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
            newline_count += block.count(b"\n")
            final_byte = block[-1:]
    line_count = newline_count + (1 if final_byte and final_byte != b"\n" else 0)
    return digest.hexdigest(), line_count


def audit_source_file(data_root: Path, spec: DatasetSpec) -> dict[str, Any]:
    path = atomic_path(data_root, spec).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {spec.dataset} atomic file: {path}")
    size = path.stat().st_size
    if spec.source_bytes is not None and size != spec.source_bytes:
        suffix = (
            " Refusing a possible CoPD/small Douban substitute; prepare the pinned "
            "full RecBole-CDR release with prepare_douban.py."
            if spec.douban_official
            else ""
        )
        raise ValueError(
            f"{spec.dataset} source byte count {size:,} != expected "
            f"{spec.source_bytes:,}.{suffix}"
        )

    with path.open("rb") as source:
        header = source.readline().decode("utf-8").rstrip("\r\n")
    fields = {column.split(":", 1)[0] for column in header.split("\t")}
    required = {"user_id", "item_id", "rating"}
    if not required.issubset(fields):
        raise ValueError(f"{path} header lacks {sorted(required)}: {header!r}")

    digest, line_count = _file_sha256_and_lines(path)
    rows = max(0, line_count - 1)
    if spec.source_rows is not None and rows != spec.source_rows:
        raise ValueError(
            f"{spec.dataset} source rows {rows:,} != expected {spec.source_rows:,}; "
            f"expected {spec.source_release}"
        )
    if spec.source_sha256 is not None and digest != spec.source_sha256:
        suffix = (
            " This is not the pinned full RecBole-CDR file; CoPD is forbidden."
            if spec.douban_official
            else ""
        )
        raise ValueError(
            f"{spec.dataset} SHA256 {digest} != expected {spec.source_sha256}.{suffix}"
        )
    return {
        "status": "accepted",
        "path": str(path),
        "bytes": size,
        "source_rows": rows,
        "sha256": digest,
        "source_release": spec.source_release,
        "filtered_reference": spec.filtered.json(),
    }


@contextlib.contextmanager
def _isolated_argv() -> Iterable[None]:
    previous = sys.argv
    sys.argv = [previous[0]]
    try:
        yield
    finally:
        sys.argv = previous


def audit_filtered_dataset(repo: Path, data_root: Path, spec: DatasetSpec) -> dict[str, Any]:
    """Instantiate exactly the fixed RecBole filter and check reference counts."""

    from recbole_gnn.config import Config
    from recbole_gnn.utils import create_dataset

    with _isolated_argv():
        config = Config(
            model="RecFormer",
            dataset=spec.dataset,
            config_file_list=[str(config_path(repo, spec))],
            config_dict={"data_path": str(data_root), "use_gpu": False},
        )
        dataset = create_dataset(config)

    framework_users = int(dataset.user_num)
    framework_items = int(dataset.item_num)
    token_users = framework_users - 1
    token_items = framework_items - 1
    interactions = int(len(dataset))
    if spec.filtered.count_basis == "token":
        user_values = (token_users,)
        item_values = (token_items,)
    elif spec.filtered.count_basis == "token-or-framework":
        user_values = (token_users, framework_users)
        item_values = (token_items, framework_items)
    else:  # Rounded historical Amazon Book reference.
        user_values = (token_users, framework_users)
        item_values = (token_items, framework_items)
    if not any(spec.filtered.users.accepts(value) for value in user_values):
        raise ValueError(f"{spec.dataset} filtered user count mismatch: {user_values}")
    if not any(spec.filtered.items.accepts(value) for value in item_values):
        raise ValueError(f"{spec.dataset} filtered item count mismatch: {item_values}")
    if not spec.filtered.interactions.accepts(interactions):
        raise ValueError(f"{spec.dataset} filtered interactions mismatch: {interactions:,}")
    return {
        "status": "accepted",
        "framework_users": framework_users,
        "framework_items": framework_items,
        "token_users": token_users,
        "token_items": token_items,
        "interactions": interactions,
        "reference": spec.filtered.json(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_fixed_protocol(repo: Path, spec: DatasetSpec) -> dict[str, Any]:
    path = config_path(repo, spec)
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed Hgformer config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"fixed config is not a mapping: {path}")
    expected = {
        "model": "RecFormer",
        "dataset": spec.dataset,
        "seed": SEED,
        "metrics": list(FULL_METRICS),
        "topk": list(FULL_TOPK),
        "valid_metric": "Recall@10",
        "val_interval": {"rating": "[3,inf)"},
        "eval_args": {
            "split": FULL_SPLIT,
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        },
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"fixed Hgformer protocol changed for {spec.dataset}; "
            f"expected={expected}, actual={actual}"
        )
    for key in ("user_inter_num_interval", "item_inter_num_interval"):
        if key not in payload:
            raise RuntimeError(f"fixed Hgformer config lacks {key}: {path}")
    sl_path = sl8_config_path(repo)
    sl_payload = yaml.safe_load(sl_path.read_text(encoding="utf-8"))
    allowed_sl_keys = {"embedding_size", "matrix_dim", "num_factors", "factor_aggregation"}
    if not isinstance(sl_payload, dict) or set(sl_payload) - allowed_sl_keys:
        raise RuntimeError(f"SL8 overlay may not alter the dataset protocol: {sl_path}")
    if sl_payload != {
        "embedding_size": 64,
        "matrix_dim": 8,
        "num_factors": 1,
        "factor_aggregation": "l2",
    }:
        raise RuntimeError(f"SL8 overlay changed: {sl_path} -> {sl_payload}")
    return {
        "config_file": f"baseline_config_fixed/{spec.recformer_config}",
        "config_sha256": _sha256(path),
        "dataset": spec.dataset,
        "seed": SEED,
        "filters": {
            "rating": payload["val_interval"]["rating"],
            "users": payload["user_inter_num_interval"],
            "items": payload["item_inter_num_interval"],
        },
        "validation": {
            "eval_args": payload["eval_args"],
            "metrics": payload["metrics"],
            "topk": payload["topk"],
            "selection_metric": payload["valid_metric"],
        },
    }


def tuning_protocol(base_protocol: Mapping[str, Any], epochs: int, eval_step: int) -> dict[str, Any]:
    return {
        "base_hgformer_protocol": base_protocol,
        "architecture": {
            "model": "SLRecGraph",
            "matrix_dim": 8,
            "num_factors": 1,
            "embedding_size": 64,
            "intrinsic_dimension": 63,
            "n_layers": 0,
            "message_passing": False,
            "log_terms": 12,
            "log_jitter": 0.0,
            "symmetric_distance": False,
        },
        "training": {
            "epochs": epochs,
            "eval_step": eval_step,
            "full_validation_events": epochs // eval_step,
            "test_evaluated": False,
        },
    }


def profile_trials(profile: str) -> tuple[Trial, ...]:
    if profile == "paper":
        return (Trial("paper", PAPER_PARAMETERS),)
    if profile == "core":
        return build_stage_trials("core_lr_clip")
    raise ValueError(f"unknown tuning profile: {profile}")


def _common_runner_prefix(args: argparse.Namespace, spec: DatasetSpec) -> list[str]:
    return [args.python, "-u", str(args.repo / "run_recbole_gnn.py"), "--dataset", spec.dataset]


def recformer_command(args: argparse.Namespace, spec: DatasetSpec, result: Path, checkpoints: Path) -> list[str]:
    return [
        *_common_runner_prefix(args, spec),
        "--model",
        "RecFormer",
        "--config-files",
        str(config_path(args.repo, spec)),
        "--validation-only",
        "--result-file",
        str(result),
        f"--checkpoint_dir={checkpoints}",
        f"--data_path={args.data_root}",
        # Vendored RecBole resets CUDA_VISIBLE_DEVICES from this argument.
        f"--gpu_id={args.gpu_id}",
        "--use_gpu=true",
        "--show_progress=false",
    ]


def sl8_command(
    args: argparse.Namespace,
    spec: DatasetSpec,
    trial: Trial,
    result: Path,
    checkpoints: Path,
) -> list[str]:
    configs = f"{config_path(args.repo, spec)} {sl8_config_path(args.repo)}"
    stopping_step = args.epochs // args.eval_step + 1
    return [
        *_common_runner_prefix(args, spec),
        "--model",
        trial.parameters.model_name,
        "--config-files",
        configs,
        "--validation-only",
        "--result-file",
        str(result),
        f"--checkpoint_dir={checkpoints}",
        f"--data_path={args.data_root}",
        # Keep the RecBole config and the parent physical-card mask identical.
        f"--gpu_id={args.gpu_id}",
        "--use_gpu=true",
        "--show_progress=false",
        f"--epochs={args.epochs}",
        f"--eval_step={args.eval_step}",
        f"--stopping_step={stopping_step}",
        "--train_batch_size=8192",
        "--eval_batch_size=40960000",
        "--eval_user_chunk_size=64",
        "--eval_item_chunk_size=4096",
        f"--seed={SEED}",
        "--embedding_size=64",
        "--matrix_dim=8",
        "--num_factors=1",
        "--factor_aggregation=l2",
        "--n_layers=0",
        "--sl_scale=1.0",
        "--log_terms=12",
        "--log_jitter=0.0",
        "--symmetric_distance=false",
        *trial.parameters.recbole_args(),
    ]


def final_test_command(
    args: argparse.Namespace,
    selection_result: Path,
    checkpoint: Path,
    result: Path,
) -> list[str]:
    return [
        args.python,
        "-u",
        str(args.repo / "evaluate_recbole_gnn_checkpoint.py"),
        "--checkpoint-file",
        str(checkpoint),
        "--selection-result-file",
        str(selection_result),
        "--result-file",
        str(result),
        "--skip-valid",
        "--eval-batch-size",
        "40960000",
        "--eval-user-chunk-size",
        "64",
        "--eval-item-chunk-size",
        "4096",
        "--full-sort-user-batch-size",
        "64",
        "--device",
        "cuda",
    ]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _selection_metadata(kind: str, spec: DatasetSpec, protocol: Mapping[str, Any], trial: Trial | None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "dataset_slug": spec.slug,
        "dataset": spec.dataset,
        "trial_name": trial.name if trial else None,
        "parameters": asdict(trial.parameters) if trial else None,
        "protocol": protocol,
        "test_evaluated": False,
    }


def annotate_selection(
    path: Path,
    *,
    kind: str,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    trial: Trial | None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["multidataset"] = _selection_metadata(kind, spec, protocol, trial)
    _atomic_json(path, payload)
    return load_selection(path, kind=kind, spec=spec, protocol=protocol, trial=trial)


def load_selection(
    path: Path,
    *,
    kind: str,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    trial: Trial | None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_model = "RecFormer" if kind == "recformer" else trial.parameters.model_name  # type: ignore[union-attr]
    if payload.get("model") != expected_model or payload.get("dataset") != spec.dataset:
        raise ValueError(f"wrong model/dataset in {path}")
    if int(payload.get("seed", -1)) != SEED:
        raise ValueError(f"wrong seed in {path}")
    if payload.get("test_result") is not None:
        raise RuntimeError(f"selection result touched held-out test: {path}")
    score = payload.get("best_valid_score")
    metrics = payload.get("best_valid_result")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError(f"missing finite validation score: {path}")
    if not isinstance(metrics, Mapping) or "recall@10" not in metrics:
        raise ValueError(f"missing Recall@10: {path}")
    checkpoint = payload.get("checkpoint_file")
    if not checkpoint or not Path(checkpoint).expanduser().is_file():
        raise ValueError(f"missing checkpoint: {path}")
    fingerprints = payload.get("split_fingerprints")
    if not isinstance(fingerprints, Mapping) or set(fingerprints) != {"train", "valid", "test"}:
        raise ValueError(f"missing split fingerprints: {path}")
    expected_metadata = _selection_metadata(kind, spec, protocol, trial)
    if payload.get("multidataset") != expected_metadata:
        raise ValueError(f"resume metadata mismatch: {path}")
    return payload


def completed_selection(
    path: Path,
    *,
    kind: str,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
    trial: Trial | None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return load_selection(path, kind=kind, spec=spec, protocol=protocol, trial=trial)
    except RuntimeError:
        raise
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _candidate(path: Path, trial: Trial, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trial_name": trial.name,
        "parameters": asdict(trial.parameters),
        "selection_result_file": str(path.resolve()),
        "checkpoint_file": result["checkpoint_file"],
        "best_valid_score": float(result["best_valid_score"]),
        "best_valid_result": result["best_valid_result"],
        "split_fingerprints": result["split_fingerprints"],
        "test_evaluated": False,
    }


def write_tuning_summary(
    path: Path,
    spec: DatasetSpec,
    profile: str,
    protocol: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    expected: int,
) -> dict[str, Any]:
    if candidates:
        split = candidates[0]["split_fingerprints"]
        if any(item["split_fingerprints"] != split for item in candidates[1:]):
            raise RuntimeError(f"SL8 trials use different splits for {spec.dataset}")
    ranking = sorted(candidates, key=lambda item: (-float(item["best_valid_score"]), item["trial_name"]))
    complete = len(ranking) == expected
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_slug": spec.slug,
        "dataset": spec.dataset,
        "profile": profile,
        "protocol": protocol,
        "selection_metric": "Recall@10 on full-ranking validation",
        "test_evaluated": False,
        "state": "complete" if complete else "incomplete",
        "expected_trials": expected,
        "completed_trials": len(ranking),
        "best": ranking[0] if complete else None,
        "provisional_best": ranking[0] if ranking else None,
        "ranking": ranking,
    }
    _atomic_json(path, payload)
    return payload


def load_tuning_best(
    path: Path,
    *,
    spec: DatasetSpec,
    profile: str,
    protocol: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Validate a completed summary and its selected validation artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("dataset_slug") != spec.slug
        or payload.get("dataset") != spec.dataset
        or payload.get("profile") != profile
        or payload.get("protocol") != protocol
        or payload.get("state") != "complete"
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"stale or incompatible tuning summary: {path}")
    best = payload.get("best")
    if not isinstance(best, Mapping):
        raise ValueError(f"completed tuning summary has no best candidate: {path}")
    matching = [trial for trial in profile_trials(profile) if trial.name == best.get("trial_name")]
    if len(matching) != 1:
        raise ValueError(f"summary selected an unknown trial: {path}")
    trial = matching[0]
    selection_path = Path(str(best.get("selection_result_file", ""))).expanduser().resolve()
    selection = load_selection(
        selection_path,
        kind="sl8",
        spec=spec,
        protocol=protocol,
        trial=trial,
    )
    expected = _candidate(selection_path, trial, selection)
    if dict(best) != expected:
        raise ValueError(f"summary best candidate differs from its selection JSON: {path}")
    return selection_path, selection


def load_final_test(path: Path, spec: DatasetSpec, selection: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset") != spec.dataset or payload.get("evaluation_eval_mode") != "full":
            return None
        if payload.get("valid_result") is not None or not isinstance(payload.get("test_result"), Mapping):
            return None
        if payload.get("split_fingerprints") != selection.get("split_fingerprints"):
            raise RuntimeError(f"final-test split differs from selection split: {path}")
        if Path(payload.get("checkpoint_file", "")).resolve() != Path(selection["checkpoint_file"]).resolve():
            return None
        return payload
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _paths(output_root: Path, spec: DatasetSpec) -> dict[str, Path]:
    root = output_root / spec.slug
    return {
        "root": root,
        "rec_result": root / "recformer" / "selection.json",
        "rec_log": root / "recformer" / "train.log",
        "rec_checkpoints": root / "recformer" / "checkpoints",
        "tune_root": root / "sl8-tuning",
        "tune_summary": root / "sl8-tuning" / "summary.json",
        "final_root": root / "final-test",
    }


def _protocols(args: argparse.Namespace, spec: DatasetSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    base = validate_fixed_protocol(args.repo, spec)
    recformer = {"base_hgformer_protocol": base, "model": "RecFormer", "test_evaluated": False}
    return recformer, tuning_protocol(base, args.epochs, args.eval_step)


def _audit(args: argparse.Namespace, spec: DatasetSpec) -> dict[str, Any]:
    if args.skip_data_audit:
        return {"status": "skipped-explicit-dry-run-only", "filtered_reference": spec.filtered.json()}
    source = audit_source_file(args.data_root, spec)
    filtered = audit_filtered_dataset(args.repo, args.data_root, spec) if args.deep_data_audit else None
    return {"source": source, "filtered": filtered}


def _dry_run_dataset(args: argparse.Namespace, spec: DatasetSpec, audit: Mapping[str, Any]) -> dict[str, Any]:
    paths = _paths(args.output_root, spec)
    rec_protocol, sl_protocol = _protocols(args, spec)
    rec_complete = completed_selection(
        paths["rec_result"], kind="recformer", spec=spec, protocol=rec_protocol, trial=None
    )
    jobs: list[dict[str, Any]] = []
    if args.phase in {"reproduce", "all"}:
        jobs.append(
            {
                "kind": "recformer-validation-selection",
                "status": "skip-complete" if rec_complete else "run",
                "result": str(paths["rec_result"]),
                "command": recformer_command(args, spec, paths["rec_result"], paths["rec_checkpoints"]),
            }
        )
    if args.phase in {"tune", "all"}:
        dependency = rec_complete is not None or args.phase == "all"
        for trial in profile_trials(args.tuning_profile):
            result = paths["tune_root"] / "results" / f"{trial.name}.json"
            complete = completed_selection(
                result, kind="sl8", spec=spec, protocol=sl_protocol, trial=trial
            )
            jobs.append(
                {
                    "kind": "sl8-validation-trial",
                    "trial": trial.name,
                    "status": (
                        "skip-complete"
                        if complete
                        else "run" if dependency else "blocked-missing-recformer-selection"
                    ),
                    "result": str(result),
                    "command": sl8_command(
                        args,
                        spec,
                        trial,
                        result,
                        paths["tune_root"] / "checkpoints" / trial.name,
                    ),
                }
            )
    if args.phase == "final-test":
        selections: list[tuple[str, Path, Mapping[str, Any]]] = []
        if rec_complete is not None:
            selections.append(("recformer", paths["rec_result"], rec_complete))
        if paths["tune_summary"].is_file():
            selection_path, selection = load_tuning_best(
                paths["tune_summary"],
                spec=spec,
                profile=args.tuning_profile,
                protocol=sl_protocol,
            )
            selections.append(("sl8", selection_path, selection))
        for label, selection_path, selection in selections:
            result = paths["final_root"] / f"{label}.json"
            jobs.append(
                {
                    "kind": f"{label}-final-test",
                    "status": "skip-complete" if load_final_test(result, spec, selection) else "run",
                    "result": str(result),
                    "command": final_test_command(
                        args, selection_path, Path(selection["checkpoint_file"]), result
                    ),
                }
            )
        if not selections:
            jobs.append({"kind": "final-test", "status": "blocked-no-complete-selection"})
    return {
        "slug": spec.slug,
        "dataset": spec.dataset,
        "fixed_config": str(config_path(args.repo, spec)),
        "audit": audit,
        "filtered_reference": spec.filtered.json(),
        "jobs": jobs,
    }


def dry_run_plan(args: argparse.Namespace, selected: Sequence[DatasetSpec]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dry_run": True,
        "phase": args.phase,
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "test_evaluated": args.phase == "final-test",
        "datasets": [_dry_run_dataset(args, spec, _audit(args, spec)) for spec in selected],
    }


def _run_command(args: argparse.Namespace, command: list[str], log: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    run_and_tee(command, log, args.repo, env)


def _ensure_same_split(reference: Mapping[str, Any], candidate: Mapping[str, Any], spec: DatasetSpec) -> None:
    if reference["split_fingerprints"] != candidate["split_fingerprints"]:
        raise RuntimeError(f"RecFormer and SL8 split fingerprints differ for {spec.dataset}")


def execute_dataset(args: argparse.Namespace, spec: DatasetSpec, budget: list[int]) -> None:
    paths = _paths(args.output_root, spec)
    rec_protocol, sl_protocol = _protocols(args, spec)
    rec = completed_selection(
        paths["rec_result"], kind="recformer", spec=spec, protocol=rec_protocol, trial=None
    )
    if args.phase in {"reproduce", "all"} and rec is None:
        if budget[0] == 0:
            return
        paths["rec_checkpoints"].mkdir(parents=True, exist_ok=True)
        command = recformer_command(args, spec, paths["rec_result"], paths["rec_checkpoints"])
        print(f"START {spec.slug} RecFormer validation selection")
        _run_command(args, command, paths["rec_log"])
        rec = annotate_selection(
            paths["rec_result"], kind="recformer", spec=spec, protocol=rec_protocol, trial=None
        )
        budget[0] -= 1
    if args.phase == "reproduce":
        return

    if args.phase in {"tune", "all"}:
        if rec is None:
            raise RuntimeError(
                f"{spec.dataset} has no completed RecFormer validation selection; "
                "run --phase reproduce first"
            )
        candidates: list[dict[str, Any]] = []
        trials = profile_trials(args.tuning_profile)
        for index, trial in enumerate(trials, 1):
            result_path = paths["tune_root"] / "results" / f"{trial.name}.json"
            result = completed_selection(
                result_path, kind="sl8", spec=spec, protocol=sl_protocol, trial=trial
            )
            if result is None:
                if budget[0] == 0:
                    write_tuning_summary(
                        paths["tune_summary"], spec, args.tuning_profile, sl_protocol, candidates, len(trials)
                    )
                    return
                checkpoint_dir = paths["tune_root"] / "checkpoints" / trial.name
                command = sl8_command(args, spec, trial, result_path, checkpoint_dir)
                print(f"START {spec.slug} SL8 {index}/{len(trials)} {trial.name}")
                _run_command(
                    args,
                    command,
                    paths["tune_root"] / "logs" / f"{trial.name}.log",
                )
                result = annotate_selection(
                    result_path, kind="sl8", spec=spec, protocol=sl_protocol, trial=trial
                )
                budget[0] -= 1
            else:
                print(f"SKIP {spec.slug} complete {trial.name}")
            _ensure_same_split(rec, result, spec)
            candidates.append(_candidate(result_path, trial, result))
            write_tuning_summary(
                paths["tune_summary"], spec, args.tuning_profile, sl_protocol, candidates, len(trials)
            )
        return

    if args.phase == "final-test":
        if rec is None:
            raise RuntimeError(f"missing RecFormer selection for {spec.dataset}")
        summary_path = paths["tune_summary"]
        if not summary_path.is_file():
            raise RuntimeError(f"missing completed SL8 tuning summary for {spec.dataset}")
        best_path, sl_selection = load_tuning_best(
            summary_path,
            spec=spec,
            profile=args.tuning_profile,
            protocol=sl_protocol,
        )
        _ensure_same_split(rec, sl_selection, spec)
        for label, selection_path, selection in (
            ("recformer", paths["rec_result"], rec),
            ("sl8", best_path, sl_selection),
        ):
            result_path = paths["final_root"] / f"{label}.json"
            if load_final_test(result_path, spec, selection) is not None:
                print(f"SKIP {spec.slug} complete {label} final test")
                continue
            if budget[0] == 0:
                return
            command = final_test_command(
                args, selection_path, Path(selection["checkpoint_file"]), result_path
            )
            print(f"START {spec.slug} {label} final test")
            _run_command(args, command, paths["final_root"] / f"{label}.log")
            if load_final_test(result_path, spec, selection) is None:
                raise RuntimeError(f"invalid final-test artifact: {result_path}")
            budget[0] -= 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.repo = args.repo.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.epochs <= 0 or args.eval_step <= 0:
        raise ValueError("--epochs and --eval-step must be positive")
    if args.epochs % args.eval_step:
        raise ValueError("--epochs must be divisible by --eval-step so epoch N is validated")
    if args.max_new_jobs is not None and args.max_new_jobs <= 0:
        raise ValueError("--max-new-jobs must be positive")
    if args.skip_data_audit and not args.dry_run:
        raise ValueError("--skip-data-audit is permitted only with --dry-run")
    selected = select_datasets(args.datasets, args.shard_index, args.shard_count)
    if any(spec.slug == "amazon-book" for spec in selected):
        raise RuntimeError(
            "run_multidataset_sl8.py is disabled for Amazon Book: the released "
            "RecFormer_book.yaml says 5-core but the paper cardinalities require "
            "iterative 8-core. Use run_paper_dataset_pipeline.py, which applies "
            "PaperProtocol_amazon_book_8core.yaml to every compared model."
        )
    for spec in selected:
        validate_fixed_protocol(args.repo, spec)

    if args.dry_run:
        print(json.dumps(dry_run_plan(args, selected), indent=2))
        return 0

    audits = {spec.slug: _audit(args, spec) for spec in selected}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": args.phase,
        "tuning_profile": args.tuning_profile,
        "epochs": args.epochs,
        "eval_step": args.eval_step,
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "datasets": [spec.slug for spec in selected],
        "audits": audits,
        "test_evaluated": args.phase == "final-test",
    }
    manifest_path = (
        args.output_root
        / "manifests"
        / f"{args.phase}-shard-{args.shard_index}-of-{args.shard_count}.json"
    )
    _atomic_json(manifest_path, manifest)
    budget = [args.max_new_jobs if args.max_new_jobs is not None else 2**63 - 1]
    for spec in selected:
        execute_dataset(args, spec, budget)
        if budget[0] == 0:
            print("PAUSED_BY_MAX_NEW_JOBS")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
