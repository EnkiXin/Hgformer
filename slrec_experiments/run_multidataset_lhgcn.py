#!/usr/bin/env python3
"""Single-GPU, resumable LHGCN reproduction and staged tuning.

The released Hgformer repository calls the standalone Light Hyperbolic GCN
through ``HGCF`` with ``conv: lGCN``.  This driver keeps that implementation
and treats each dataset-specific ``RecFormer_*.yaml`` as the data/evaluation
protocol authority.  Only LHGCN model and optimisation keys are overlaid.

Search is deliberately greedy and staged rather than a Cartesian product:

``baseline -> gcn_layers -> curve -> learning_rate -> margin``

The extended profile then considers ``scale``, ``weight_decay``, training
batch size, and optimiser.  The validation winner of each stage is carried
into the next stage.  Trials use full-ranking validation only; this file has
no test-evaluation mode.

All subprocesses are synchronous, see exactly one physical GPU through
``CUDA_VISIBLE_DEVICES``, and share a non-blocking per-user/per-GPU lock so
two invocations of this runner cannot accidentally overlap.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import signal
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

try:
    from slrec_experiments.run_multidataset_sl8 import (
        DATASETS,
        FULL_METRICS,
        FULL_SPLIT,
        FULL_TOPK,
        SEED,
        DatasetSpec,
        audit_filtered_dataset,
        audit_source_file,
        config_path,
        select_datasets,
    )
except ModuleNotFoundError:  # Allows ``python slrec_experiments/<file>.py``.
    from run_multidataset_sl8 import (  # type: ignore
        DATASETS,
        FULL_METRICS,
        FULL_SPLIT,
        FULL_TOPK,
        SEED,
        DatasetSpec,
        audit_filtered_dataset,
        audit_source_file,
        config_path,
        select_datasets,
    )


SCHEMA_VERSION = 1
OVERLAY_NAME = "LHGCN_reproduction.yaml"
DEFAULT_MODEL_ENTRYPOINT = "HGCF"

BASELINE_STAGE = "baseline"
CORE_STAGES = (BASELINE_STAGE, "gcn_layers", "curve", "learning_rate", "margin")
EXTENDED_STAGES = ("scale", "weight_decay", "train_batch_size", "learner")
PROFILE_STAGES = {
    "baseline": (BASELINE_STAGE,),
    "core": CORE_STAGES,
    "extended": (*CORE_STAGES, *EXTENDED_STAGES),
}

SEARCH_VALUES: dict[str, tuple[Any, ...]] = {
    "gcn_layers": tuple(range(1, 9)),
    "curve": (0.05, 0.1, 0.2, 0.5, 1.0),
    "learning_rate": (1e-4, 3e-4, 5e-4, 1e-3, 3e-3),
    "margin": (0.05, 0.1, 0.2, 0.3, 0.5),
    "scale": (0.01, 0.05, 0.1, 0.2),
    "weight_decay": (0.0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2),
    "train_batch_size": (8192, 32768, 65536, 131072),
    "learner": ("adam", "rsgd", "adagrad", "rmsprop", "sgd"),
}


@dataclass(frozen=True)
class LHGCNParameters:
    """The standalone implementation's tunable model/training parameters."""

    gcn_layers: int = 4
    curve: float = 0.5
    learning_rate: float = 5e-4
    margin: float = 0.1
    scale: float = 0.1
    weight_decay: float = 0.0
    train_batch_size: int = 65536
    learner: str = "adam"

    def validate(self) -> None:
        positive = {
            "gcn_layers": self.gcn_layers,
            "curve": self.curve,
            "learning_rate": self.learning_rate,
            "margin": self.margin,
            "scale": self.scale,
            "train_batch_size": self.train_batch_size,
        }
        if self.gcn_layers < 1 or self.train_batch_size < 1:
            raise ValueError(f"positive integer parameters required: {positive}")
        for key in ("curve", "learning_rate", "margin", "scale"):
            value = float(positive[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.learner not in {"adam", "rsgd", "adagrad", "rmsprop", "sgd"}:
            raise ValueError(f"unsupported learner: {self.learner!r}")

    def recbole_args(self) -> list[str]:
        self.validate()
        return [
            f"--gcn_layers={self.gcn_layers}",
            f"--curve={self.curve:.12g}",
            f"--learning_rate={self.learning_rate:.12g}",
            f"--margin={self.margin:.12g}",
            f"--scale={self.scale:.12g}",
            f"--weight_decay={self.weight_decay:.12g}",
            f"--train_batch_size={self.train_batch_size}",
            f"--learner={self.learner}",
        ]


BASELINE_PARAMETERS = LHGCNParameters()


def _value_token(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}".replace("-", "m").replace("+", "").replace(".", "p")
    return str(value).replace("-", "m")


@dataclass(frozen=True)
class Trial:
    stage: str
    parameters: LHGCNParameters

    @property
    def name(self) -> str:
        p = self.parameters
        return (
            f"{self.stage}"
            f"__g-{p.gcn_layers}"
            f"__c-{_value_token(p.curve)}"
            f"__lr-{_value_token(p.learning_rate)}"
            f"__m-{_value_token(p.margin)}"
            f"__s-{_value_token(p.scale)}"
            f"__wd-{_value_token(p.weight_decay)}"
            f"__b-{p.train_batch_size}"
            f"__opt-{p.learner}"
        )


def build_stage_trials(
    stage: str, anchor: LHGCNParameters | None = None
) -> tuple[Trial, ...]:
    """Build one stage; an identical carried anchor is never retrained."""

    if stage == BASELINE_STAGE:
        if anchor is not None:
            raise ValueError("baseline does not accept an anchor")
        return (Trial(stage, BASELINE_PARAMETERS),)
    if stage not in SEARCH_VALUES:
        raise ValueError(f"unknown LHGCN stage: {stage!r}")
    if anchor is None:
        raise ValueError(f"stage {stage!r} requires the previous validation winner")

    trials: list[Trial] = []
    seen: set[LHGCNParameters] = set()
    for value in SEARCH_VALUES[stage]:
        candidate = replace(anchor, **{stage: value})
        candidate.validate()
        if candidate != anchor and candidate not in seen:
            trials.append(Trial(stage, candidate))
            seen.add(candidate)
    return tuple(trials)


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
        help="seven-dataset slugs/names; comma-separated values are accepted",
    )
    parser.add_argument("--profile", choices=tuple(PROFILE_STAGES), default="core")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval-step", type=int, default=50)
    parser.add_argument(
        "--gpu-id",
        default="0",
        help="one non-negative physical CUDA index; lists and UUID aliases are rejected",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--model-entrypoint",
        choices=("HGCF", "LHGCN"),
        default=DEFAULT_MODEL_ENTRYPOINT,
        help="HGCF is the released path; LHGCN is its tested naming-only adapter",
    )
    parser.add_argument("--max-new-jobs", type=int)
    parser.add_argument("--deep-data-audit", action="store_true")
    parser.add_argument(
        "--skip-data-audit",
        action="store_true",
        help="allowed only with --dry-run on machines that do not hold data",
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        help="override the default per-user/per-physical-GPU lock in the temp directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def overlay_path(repo: Path) -> Path:
    return repo / "baseline_config_fixed" / OVERLAY_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_lhgcn_protocol(repo: Path, spec: DatasetSpec) -> dict[str, Any]:
    """Pin Hgformer's dataset protocol and the standalone LHGCN overlay."""

    base_path = config_path(repo, spec)
    model_path = overlay_path(repo)
    if not base_path.is_file():
        raise FileNotFoundError(f"missing dataset config: {base_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"missing LHGCN overlay: {model_path}")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    overlay = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        raise ValueError("dataset config and LHGCN overlay must be YAML mappings")

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
    actual = {key: base.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"fixed Hgformer protocol changed for {spec.dataset}; "
            f"expected={expected}, actual={actual}"
        )
    for key in ("user_inter_num_interval", "item_inter_num_interval"):
        if key not in base:
            raise RuntimeError(f"fixed dataset config lacks {key}: {base_path}")

    forbidden_overlay_keys = {
        "dataset",
        "seed",
        "field_separator",
        "USER_ID_FIELD",
        "ITEM_ID_FIELD",
        "RATING_FIELD",
        "load_col",
        "user_inter_num_interval",
        "item_inter_num_interval",
        "val_interval",
        "metrics",
        "topk",
        "valid_metric",
        "eval_args",
    }
    overlap = forbidden_overlay_keys.intersection(overlay)
    if overlap:
        raise RuntimeError(
            f"LHGCN overlay may not override data/split/evaluation keys: {sorted(overlap)}"
        )
    essential = {
        "embedding_size": 64,
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
        "train_batch_size": 65536,
    }
    overlay_actual = {key: overlay.get(key) for key in essential}
    overlay_model = overlay.get("model")
    if overlay_model not in {"HGCF", "LHGCN"} or overlay_actual != essential:
        raise RuntimeError(
            "LHGCN reproduction overlay changed; expected model HGCF or its "
            f"naming-only LHGCN adapter plus {essential}, actual model="
            f"{overlay_model!r}, values={overlay_actual}"
        )

    return {
        "dataset": spec.dataset,
        "seed": SEED,
        "base_config": str(base_path.relative_to(repo)),
        "base_config_sha256": _sha256(base_path),
        "overlay": str(model_path.relative_to(repo)),
        "overlay_sha256": _sha256(model_path),
        "filters": {
            "rating": base["val_interval"]["rating"],
            "users": base["user_inter_num_interval"],
            "items": base["item_inter_num_interval"],
        },
        "validation": {
            "eval_args": base["eval_args"],
            "metrics": base["metrics"],
            "topk": base["topk"],
            "selection_metric": base["valid_metric"],
        },
    }


def runtime_protocol(
    base: Mapping[str, Any],
    *,
    epochs: int,
    eval_step: int,
    model_entrypoint: str,
) -> dict[str, Any]:
    return {
        "base_hgformer_protocol": base,
        "model": {
            "label": "LHGCN",
            "entrypoint": model_entrypoint,
            "released_equivalent": "HGCF + conv=lGCN",
            "embedding_size": 64,
            "conv": "lGCN",
            "negative_sampling": {"uniform": 1},
            "loss": "HGCF batch-summed squared-distance margin ranking",
        },
        "training": {
            "epochs": epochs,
            "eval_step": eval_step,
            "full_validation_events": epochs // eval_step,
            "early_stopping_disabled_for_fixed_budget": True,
        },
        "selection": "Recall@10 on full-ranking validation; stage winner carried",
        "test_evaluated": False,
    }


def trial_command(
    args: argparse.Namespace,
    spec: DatasetSpec,
    trial: Trial,
    result_path: Path,
    checkpoint_dir: Path,
) -> list[str]:
    configs = f"{config_path(args.repo, spec)} {overlay_path(args.repo)}"
    stopping_step = args.epochs // args.eval_step + 1
    return [
        args.python,
        "-u",
        str(args.repo / "run_recbole_gnn.py"),
        "--model",
        args.model_entrypoint,
        "--dataset",
        spec.dataset,
        "--config-files",
        configs,
        "--validation-only",
        "--result-file",
        str(result_path),
        f"--checkpoint_dir={checkpoint_dir}",
        f"--data_path={args.data_root}",
        "--gpu_id=0",
        "--use_gpu=true",
        "--show_progress=false",
        f"--epochs={args.epochs}",
        f"--eval_step={args.eval_step}",
        f"--stopping_step={stopping_step}",
        f"--seed={SEED}",
        "--embedding_size=64",
        "--conv=lGCN",
        "--neg_sampling={'uniform': 1}",
        "--tail_analysis=false",
        "--popularity_analysis=false",
        *trial.parameters.recbole_args(),
    ]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _result_metadata(
    spec: DatasetSpec,
    trial: Trial,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_slug": spec.slug,
        "dataset": spec.dataset,
        "stage": trial.stage,
        "trial_name": trial.name,
        "parameters": asdict(trial.parameters),
        "protocol": protocol,
        "test_evaluated": False,
    }


def load_complete_result(
    path: Path,
    *,
    spec: DatasetSpec,
    trial: Trial,
    protocol: Mapping[str, Any],
    model_entrypoint: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result is not a JSON object: {path}")
    if payload.get("model") != model_entrypoint or payload.get("dataset") != spec.dataset:
        raise ValueError(f"wrong model/dataset result: {path}")
    if int(payload.get("seed", -1)) != SEED:
        raise ValueError(f"wrong seed result: {path}")
    if payload.get("test_result") is not None:
        raise RuntimeError(f"LHGCN tuning result touched held-out test: {path}")
    score = payload.get("best_valid_score")
    metrics = payload.get("best_valid_result")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError(f"missing finite validation score: {path}")
    if not isinstance(metrics, Mapping) or "recall@10" not in metrics:
        raise ValueError(f"missing full-validation Recall@10: {path}")
    checkpoint = payload.get("checkpoint_file")
    if not checkpoint or not Path(str(checkpoint)).expanduser().is_file():
        raise ValueError(f"missing checkpoint: {path}")
    fingerprints = payload.get("split_fingerprints")
    if not isinstance(fingerprints, Mapping) or set(fingerprints) != {
        "train",
        "valid",
        "test",
    }:
        raise ValueError(f"missing split fingerprints: {path}")
    if payload.get("lhgcn_tuning") != _result_metadata(spec, trial, protocol):
        raise ValueError(f"resume metadata mismatch: {path}")
    return payload


def completed_result(
    path: Path,
    *,
    spec: DatasetSpec,
    trial: Trial,
    protocol: Mapping[str, Any],
    model_entrypoint: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return load_complete_result(
            path,
            spec=spec,
            trial=trial,
            protocol=protocol,
            model_entrypoint=model_entrypoint,
        )
    except RuntimeError:
        # Never silently replace an artifact which already inspected test.
        raise
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def annotate_result(
    path: Path,
    *,
    spec: DatasetSpec,
    trial: Trial,
    protocol: Mapping[str, Any],
    model_entrypoint: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["lhgcn_tuning"] = _result_metadata(spec, trial, protocol)
    _atomic_json(path, payload)
    return load_complete_result(
        path,
        spec=spec,
        trial=trial,
        protocol=protocol,
        model_entrypoint=model_entrypoint,
    )


def candidate_from_result(
    result_path: Path, trial: Trial, result: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "trial_name": trial.name,
        "stage": trial.stage,
        "parameters": asdict(trial.parameters),
        "selection_result_file": str(result_path.expanduser().resolve()),
        "checkpoint_file": result["checkpoint_file"],
        "best_valid_score": float(result["best_valid_score"]),
        "best_valid_result": result["best_valid_result"],
        "split_fingerprints": result["split_fingerprints"],
        "test_evaluated": False,
    }


def _rank(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in sorted(
            candidates,
            key=lambda item: (-float(item["best_valid_score"]), str(item["trial_name"])),
        )
    ]


def write_stage_summary(
    path: Path,
    *,
    spec: DatasetSpec,
    stage: str,
    protocol: Mapping[str, Any],
    anchor: Mapping[str, Any] | None,
    new_candidates: Sequence[Mapping[str, Any]],
    expected_trial_names: Sequence[str],
) -> dict[str, Any]:
    candidates = ([anchor] if anchor is not None else []) + list(new_candidates)
    if candidates:
        split = candidates[0]["split_fingerprints"]
        if any(candidate["split_fingerprints"] != split for candidate in candidates[1:]):
            raise RuntimeError(f"LHGCN stage candidates use different splits: {spec.dataset}")
    ranking = _rank(candidates)
    completed_names = {item["trial_name"] for item in new_candidates}
    complete = completed_names == set(expected_trial_names)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_slug": spec.slug,
        "dataset": spec.dataset,
        "stage": stage,
        "protocol": protocol,
        "selection_metric": "Recall@10 on full-ranking validation",
        "test_evaluated": False,
        "state": "complete" if complete else "incomplete",
        "anchor_carried": anchor,
        "expected_new_trials": list(expected_trial_names),
        "completed_new_trials": len(completed_names),
        "winner": ranking[0] if complete and ranking else None,
        "provisional_winner": ranking[0] if ranking else None,
        "ranking": ranking,
    }
    _atomic_json(path, payload)
    return payload


def load_stage_winner(
    path: Path,
    *,
    spec: DatasetSpec,
    stage: str,
    protocol: Mapping[str, Any],
    model_entrypoint: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("dataset_slug") != spec.slug
        or payload.get("dataset") != spec.dataset
        or payload.get("stage") != stage
        or payload.get("protocol") != protocol
        or payload.get("state") != "complete"
        or payload.get("test_evaluated") is not False
        or not isinstance(payload.get("winner"), Mapping)
    ):
        raise ValueError(f"incomplete/stale LHGCN stage summary: {path}")
    winner = dict(payload["winner"])
    winner_stage = winner.get("stage")
    winner_path = winner.get("selection_result_file")
    if not isinstance(winner_stage, str) or not isinstance(winner_path, str):
        raise ValueError(f"stage winner lacks its source identity: {path}")
    winner_trial = Trial(winner_stage, _parameters_from_candidate(winner))
    if winner_trial.name != winner.get("trial_name"):
        raise ValueError(f"stage winner trial identity changed: {path}")
    resolved_result = Path(winner_path).expanduser().resolve()
    result = load_complete_result(
        resolved_result,
        spec=spec,
        trial=winner_trial,
        protocol=protocol,
        model_entrypoint=model_entrypoint,
    )
    if candidate_from_result(resolved_result, winner_trial, result) != winner:
        raise ValueError(f"stage winner differs from its selection result: {path}")
    return winner


def _stage_root(output_root: Path, spec: DatasetSpec, stage: str) -> Path:
    return output_root / spec.slug / "lhgcn-tuning" / "stages" / stage


def _result_path(output_root: Path, spec: DatasetSpec, trial: Trial) -> Path:
    return _stage_root(output_root, spec, trial.stage) / "results" / f"{trial.name}.json"


def _summary_path(output_root: Path, spec: DatasetSpec, stage: str) -> Path:
    return _stage_root(output_root, spec, stage) / "summary.json"


def _parameters_from_candidate(candidate: Mapping[str, Any]) -> LHGCNParameters:
    values = candidate.get("parameters")
    if not isinstance(values, Mapping):
        raise ValueError("stage winner has no parameter mapping")
    try:
        parameters = LHGCNParameters(
            gcn_layers=int(values["gcn_layers"]),
            curve=float(values["curve"]),
            learning_rate=float(values["learning_rate"]),
            margin=float(values["margin"]),
            scale=float(values["scale"]),
            weight_decay=float(values["weight_decay"]),
            train_batch_size=int(values["train_batch_size"]),
            learner=str(values["learner"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid parameter mapping in stage winner") from error
    parameters.validate()
    return parameters


def _run_and_tee(
    command: list[str],
    log_path: Path,
    cwd: Path,
    env: Mapping[str, str],
    *,
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
            # The child keeps the advisory lock if this orchestrator is killed
            # abruptly.  A replacement runner therefore cannot overlap a
            # still-live training process.
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
            # Do not release the run lock while a child survives an interrupt
            # or a logging failure.  The child owns a process group so any
            # dataloader descendants receive the same termination signal.
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


def _gpu_token(value: str) -> str:
    token = value.strip()
    if not token.isdigit():
        raise ValueError(
            "--gpu-id must be one non-negative physical CUDA index (for example 0)"
        )
    # Canonicalisation prevents aliases such as 0 and 00 from acquiring
    # different lock files for the same physical card.
    return str(int(token))


def default_lock_path(gpu_id: str) -> Path:
    digest = hashlib.sha256(gpu_id.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"hgformer-lhgcn-uid{os.getuid()}-gpu-{digest}.lock"


@contextlib.contextmanager
def exclusive_gpu_lock(path: Path, gpu_id: str) -> Iterable[int]:
    """Hold a non-blocking advisory lock for this runner's entire real run."""

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock.seek(0)
            owner = lock.read().strip() or "unknown owner"
            raise RuntimeError(
                f"physical GPU {gpu_id!r} is already reserved by this runner: "
                f"{resolved} ({owner})"
            ) from error
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()} gpu={gpu_id} output_lock={resolved}\n")
        lock.flush()
        try:
            yield lock.fileno()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _audit(args: argparse.Namespace, spec: DatasetSpec) -> dict[str, Any]:
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


def search_manifest(args: argparse.Namespace) -> dict[str, Any]:
    maximum = 1
    per_stage: dict[str, Any] = {
        BASELINE_STAGE: {"values": [asdict(BASELINE_PARAMETERS)], "maximum_new_trials": 1}
    }
    for stage in (*CORE_STAGES[1:], *EXTENDED_STAGES):
        values = list(SEARCH_VALUES[stage])
        # Earlier stages never alter this stage's field, so its value is still
        # the baseline value when reached and is carried without retraining.
        new_trial_count = len(values) - int(getattr(BASELINE_PARAMETERS, stage) in values)
        per_stage[stage] = {
            "parameter": stage,
            "values": values,
            "new_trials": new_trial_count,
            "depends_on_previous_validation_winner": True,
        }
        if stage in PROFILE_STAGES[args.profile]:
            maximum += new_trial_count
    return {
        "design": "greedy staged one-parameter search; no full Cartesian product",
        "profile": args.profile,
        "stage_order": list(PROFILE_STAGES[args.profile]),
        "stages": per_stage,
        "new_trials_per_dataset": maximum,
    }


def _existing_stage_anchor(
    args: argparse.Namespace,
    spec: DatasetSpec,
    stages: Sequence[str],
    stage_index: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any] | None:
    if stage_index == 0:
        return None
    previous = stages[stage_index - 1]
    return load_stage_winner(
        _summary_path(args.output_root, spec, previous),
        spec=spec,
        stage=previous,
        protocol=protocol,
        model_entrypoint=args.model_entrypoint,
    )


def dry_run_plan(
    args: argparse.Namespace,
    selected: Sequence[DatasetSpec],
    protocols: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    stages = PROFILE_STAGES[args.profile]
    for spec in selected:
        stage_plans: list[dict[str, Any]] = []
        anchor: dict[str, Any] | None = None
        for index, stage in enumerate(stages):
            if index:
                try:
                    anchor = _existing_stage_anchor(
                        args, spec, stages, index, protocols[spec.slug]
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    stage_plans.append(
                        {
                            "stage": stage,
                            "status": f"blocked-awaiting-{stages[index - 1]}-winner",
                            "search_values": list(SEARCH_VALUES[stage]),
                        }
                    )
                    # Every still-later stage is data-dependent too.
                    continue
            anchor_parameters = _parameters_from_candidate(anchor) if anchor else None
            trials = build_stage_trials(stage, anchor_parameters)
            jobs: list[dict[str, Any]] = []
            for trial in trials:
                result_path = _result_path(args.output_root, spec, trial)
                complete = completed_result(
                    result_path,
                    spec=spec,
                    trial=trial,
                    protocol=protocols[spec.slug],
                    model_entrypoint=args.model_entrypoint,
                )
                jobs.append(
                    {
                        "trial": trial.name,
                        "parameters": asdict(trial.parameters),
                        "status": "skip-complete" if complete else "run",
                        "command": trial_command(
                            args,
                            spec,
                            trial,
                            result_path,
                            _stage_root(args.output_root, spec, stage)
                            / "checkpoints"
                            / trial.name,
                        ),
                    }
                )
            stage_plans.append(
                {
                    "stage": stage,
                    "status": "ready",
                    "anchor": anchor,
                    "jobs": jobs,
                }
            )
        datasets.append(
            {
                "slug": spec.slug,
                "dataset": spec.dataset,
                "audit": _audit(args, spec),
                "protocol": protocols[spec.slug],
                "stages": stage_plans,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "dry_run": True,
        "single_physical_gpu": args.gpu_id,
        "child_visible_gpu": 0,
        "lock_file": str(args.lock_file),
        "search": search_manifest(args),
        "test_evaluated": False,
        "datasets": datasets,
    }


def _write_dataset_summary(
    args: argparse.Namespace,
    spec: DatasetSpec,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    final_winner: dict[str, Any] | None = None
    for stage in PROFILE_STAGES[args.profile]:
        path = _summary_path(args.output_root, spec, stage)
        if not path.is_file():
            stages.append({"stage": stage, "state": "pending"})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stages.append({"stage": stage, "state": "invalid"})
            continue
        state = payload.get("state", "invalid")
        winner = payload.get("winner") if state == "complete" else None
        stages.append(
            {
                "stage": stage,
                "state": state,
                "completed_new_trials": payload.get("completed_new_trials", 0),
                "winner": winner,
                "provisional_winner": payload.get("provisional_winner"),
                "summary_file": str(path.resolve()),
            }
        )
        if winner is not None:
            final_winner = dict(winner)
    complete = len(stages) == len(PROFILE_STAGES[args.profile]) and all(
        item["state"] == "complete" for item in stages
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_slug": spec.slug,
        "dataset": spec.dataset,
        "profile": args.profile,
        "protocol": protocol,
        "state": "complete" if complete else "incomplete",
        "selection_metric": "Recall@10 on full-ranking validation",
        "test_evaluated": False,
        "final_validation_winner": final_winner if complete else None,
        "latest_complete_stage_winner": final_winner,
        "stages": stages,
    }
    path = args.output_root / spec.slug / "lhgcn-tuning" / "summary.json"
    _atomic_json(path, summary)
    return summary


def _write_global_summary(
    args: argparse.Namespace,
    selected: Sequence[DatasetSpec],
    protocols: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    dataset_summaries = [
        _write_dataset_summary(args, spec, protocols[spec.slug]) for spec in selected
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "profile": args.profile,
        "state": (
            "complete"
            if all(item["state"] == "complete" for item in dataset_summaries)
            else "incomplete"
        ),
        "single_physical_gpu": args.gpu_id,
        "selection_metric": "Recall@10 on full-ranking validation",
        "test_evaluated": False,
        "datasets": dataset_summaries,
    }
    _atomic_json(args.output_root / "lhgcn-summary.json", payload)
    return payload


def execute_stage_for_dataset(
    args: argparse.Namespace,
    spec: DatasetSpec,
    stage: str,
    anchor: Mapping[str, Any] | None,
    protocol: Mapping[str, Any],
    budget: list[int],
    env: Mapping[str, str],
    lock_fd: int,
) -> bool:
    anchor_parameters = _parameters_from_candidate(anchor) if anchor else None
    trials = build_stage_trials(stage, anchor_parameters)
    new_candidates: list[dict[str, Any]] = []
    expected_names = [trial.name for trial in trials]

    for index, trial in enumerate(trials, 1):
        result_path = _result_path(args.output_root, spec, trial)
        result = completed_result(
            result_path,
            spec=spec,
            trial=trial,
            protocol=protocol,
            model_entrypoint=args.model_entrypoint,
        )
        if result is None:
            if budget[0] == 0:
                write_stage_summary(
                    _summary_path(args.output_root, spec, stage),
                    spec=spec,
                    stage=stage,
                    protocol=protocol,
                    anchor=anchor,
                    new_candidates=new_candidates,
                    expected_trial_names=expected_names,
                )
                return False
            stage_root = _stage_root(args.output_root, spec, stage)
            checkpoint_dir = stage_root / "checkpoints" / trial.name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            command = trial_command(
                args, spec, trial, result_path, checkpoint_dir
            )
            print(
                f"START {spec.slug} {stage} {index}/{len(trials)} {trial.name}",
                flush=True,
            )
            _run_and_tee(
                command,
                stage_root / "logs" / f"{trial.name}.log",
                args.repo,
                env,
                lock_fd=lock_fd,
            )
            result = annotate_result(
                result_path,
                spec=spec,
                trial=trial,
                protocol=protocol,
                model_entrypoint=args.model_entrypoint,
            )
            budget[0] -= 1
        else:
            print(f"SKIP {spec.slug} complete {trial.name}", flush=True)
        candidate = candidate_from_result(result_path, trial, result)
        if anchor and candidate["split_fingerprints"] != anchor["split_fingerprints"]:
            raise RuntimeError(
                f"stage split differs from carried winner for {spec.dataset}"
            )
        new_candidates.append(candidate)
        write_stage_summary(
            _summary_path(args.output_root, spec, stage),
            spec=spec,
            stage=stage,
            protocol=protocol,
            anchor=anchor,
            new_candidates=new_candidates,
            expected_trial_names=expected_names,
        )

    # This also handles the theoretically empty stage whose grid contained
    # only the carried anchor.
    write_stage_summary(
        _summary_path(args.output_root, spec, stage),
        spec=spec,
        stage=stage,
        protocol=protocol,
        anchor=anchor,
        new_candidates=new_candidates,
        expected_trial_names=expected_names,
    )
    return True


def execute(
    args: argparse.Namespace,
    selected: Sequence[DatasetSpec],
    protocols: Mapping[str, Mapping[str, Any]],
    lock_fd: int,
) -> None:
    budget = [args.max_new_jobs if args.max_new_jobs is not None else 2**63 - 1]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    stages = PROFILE_STAGES[args.profile]

    # Stage-major order gives the user a seven-dataset baseline before the
    # longer tuning sweep proceeds.
    for stage_index, stage in enumerate(stages):
        for spec in selected:
            anchor = _existing_stage_anchor(
                args, spec, stages, stage_index, protocols[spec.slug]
            )
            complete = execute_stage_for_dataset(
                args,
                spec,
                stage,
                anchor,
                protocols[spec.slug],
                budget,
                env,
                lock_fd,
            )
            _write_global_summary(args, selected, protocols)
            if not complete or budget[0] == 0:
                print("PAUSED_BY_MAX_NEW_JOBS", flush=True)
                return


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.repo = args.repo.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.gpu_id = _gpu_token(str(args.gpu_id))
    args.lock_file = (
        args.lock_file.expanduser().resolve()
        if args.lock_file is not None
        else default_lock_path(args.gpu_id)
    )
    if args.epochs <= 0 or args.eval_step <= 0:
        raise ValueError("--epochs and --eval-step must be positive")
    if args.epochs % args.eval_step:
        raise ValueError("--epochs must be divisible by --eval-step")
    if args.max_new_jobs is not None and args.max_new_jobs <= 0:
        raise ValueError("--max-new-jobs must be positive")
    if args.skip_data_audit and not args.dry_run:
        raise ValueError("--skip-data-audit is permitted only with --dry-run")

    selected = select_datasets(args.datasets, 0, 1)
    if any(spec.slug == "amazon-book" for spec in selected):
        raise RuntimeError(
            "run_multidataset_lhgcn.py is disabled for Amazon Book: the released "
            "RecFormer_book.yaml says 5-core but the paper cardinalities require "
            "iterative 8-core. Use run_paper_dataset_pipeline.py, which applies "
            "PaperProtocol_amazon_book_8core.yaml to every compared model."
        )
    base_protocols = {
        spec.slug: validate_lhgcn_protocol(args.repo, spec) for spec in selected
    }
    protocols = {
        spec.slug: runtime_protocol(
            base_protocols[spec.slug],
            epochs=args.epochs,
            eval_step=args.eval_step,
            model_entrypoint=args.model_entrypoint,
        )
        for spec in selected
    }

    if args.dry_run:
        print(json.dumps(dry_run_plan(args, selected, protocols), indent=2))
        return 0

    # Set this before the optional deep RecBole audit imports torch.  Children
    # inherit the same one-device view and always address it as logical cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    with exclusive_gpu_lock(args.lock_file, args.gpu_id) as lock_fd:
        audits = {spec.slug: _audit(args, spec) for spec in selected}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "single_physical_gpu": args.gpu_id,
            "child_visible_gpu": 0,
            "lock_file": str(args.lock_file),
            "model_entrypoint": args.model_entrypoint,
            "epochs": args.epochs,
            "eval_step": args.eval_step,
            "datasets": [spec.slug for spec in selected],
            "search": search_manifest(args),
            "protocols": protocols,
            "audits": audits,
            "test_evaluated": False,
        }
        _atomic_json(args.output_root / "lhgcn-manifest.json", manifest)
        execute(args, selected, protocols, lock_fd)
        summary = _write_global_summary(args, selected, protocols)
    print(
        f"LHGCN_SUMMARY={(args.output_root / 'lhgcn-summary.json').resolve()} "
        f"state={summary['state']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
