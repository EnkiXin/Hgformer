#!/usr/bin/env python3
"""Resumable, single-GPU full-ranking tuning for geometry-only SL(8).

The default ``core`` profile exhaustively evaluates a deliberately finite
``learning_rate x coord_clip`` grid.  The ``extended`` profile first completes
that same core grid and then greedily ablates initialisation, regularisation,
Schatten order / score scale, and finally negative sampling / ranking loss.
Each later stage carries the previous winner, so a harmful ablation cannot
silently replace a better configuration.

Every training trial is validation-only and uses the Hgformer Amazon-CD
random user-grouped 8:1:1 split, Recall/NDCG metrics and *full-ranking*
candidate set.  By default a trial trains for 500 epochs and evaluates only
at epoch 500.  This makes the large SL(8) search practical and gives every
configuration an equal optimisation budget, but it is intentionally a
fixed-epoch comparison rather than Hgformer's per-epoch early-stop protocol.
The held-out test split is never evaluated by this driver.

``--gpu-id`` is a physical CUDA index.  The runner writes that same index to
both ``CUDA_VISIBLE_DEVICES`` and the vendored RecBole child's ``--gpu_id``;
the latter is required because this RecBole copy rewrites the environment
from its configuration during startup.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = 1
PRODUCTION_MODEL = "SLRecGraph"
LOSS_ABLATION_MODEL = "SLRecGraphFullTune"
DATASET = "Amazon_cd"
SEED = 2024

CONFIG_NAMES = (
    "SLRecGraph_cd.yaml",
    "SLRecGraph_ablation_sl8.yaml",
    "SLRecGraph_eval_full.yaml",
)

CORE_LEARNING_RATES = (1e-3, 3e-3, 6e-3)
CORE_COORD_CLIPS = (0.5, 0.75, 1.0, 1.5)
EXTENDED_STAGES = ("initialisation", "regularisation", "metric_scale", "loss_neg")
PROFILE_STAGES = {
    "core": ("core_lr_clip",),
    "extended": ("core_lr_clip", *EXTENDED_STAGES),
}


@dataclass(frozen=True)
class Parameters:
    """All train/geometry/loss scalars that may vary in this search."""

    learning_rate: float = 3e-3
    coord_clip: float = 0.75
    init_std: float = 0.012
    reg_weight: float = 0.0
    schatten_p: str = "2"
    score_scale: float = 1.0
    learnable_score_scale: bool = True
    negative_count: int = 1
    pairwise_loss: str = "bpr"
    loss_margin: float = 1.0

    @property
    def model_name(self) -> str:
        # The BPR baseline remains literally the production implementation.
        return PRODUCTION_MODEL if self.pairwise_loss == "bpr" else LOSS_ABLATION_MODEL

    def validate(self) -> None:
        positive = {
            "learning_rate": self.learning_rate,
            "coord_clip": self.coord_clip,
            "init_std": self.init_std,
            "score_scale": self.score_scale,
        }
        if any(not math.isfinite(value) or value <= 0 for value in positive.values()):
            raise ValueError(f"positive finite parameters required: {positive}")
        if not math.isfinite(self.reg_weight) or self.reg_weight < 0:
            raise ValueError("reg_weight must be finite and non-negative")
        if self.schatten_p not in {"1", "2", "inf"}:
            raise ValueError("schatten_p must be one of {'1', '2', 'inf'}")
        if self.negative_count < 1:
            raise ValueError("negative_count must be positive")
        if self.pairwise_loss not in {"bpr", "hinge"}:
            raise ValueError("pairwise_loss must be one of {'bpr', 'hinge'}")
        if not math.isfinite(self.loss_margin) or self.loss_margin < 0:
            raise ValueError("loss_margin must be finite and non-negative")

    def recbole_args(self) -> list[str]:
        self.validate()
        learnable = "true" if self.learnable_score_scale else "false"
        return [
            f"--learning_rate={self.learning_rate:.12g}",
            f"--coord_clip={self.coord_clip:.12g}",
            f"--init_std={self.init_std:.12g}",
            f"--reg_weight={self.reg_weight:.12g}",
            f"--schatten_p={self.schatten_p}",
            f"--score_scale={self.score_scale:.12g}",
            f"--learnable_score_scale={learnable}",
            f"--neg_sampling={{'uniform': {self.negative_count}}}",
            f"--pairwise_loss={self.pairwise_loss}",
            f"--loss_margin={self.loss_margin:.12g}",
        ]


PAPER_PARAMETERS = Parameters()


@dataclass(frozen=True)
class Trial:
    stage: str
    parameters: Parameters

    @property
    def name(self) -> str:
        p = self.parameters
        return (
            f"{self.stage}"
            f"__lr-{_float_token(p.learning_rate)}"
            f"__c-{_float_token(p.coord_clip)}"
            f"__i-{_float_token(p.init_std)}"
            f"__r-{_float_token(p.reg_weight)}"
            f"__p-{p.schatten_p}"
            f"__s-{_float_token(p.score_scale)}"
            f"__learn-{int(p.learnable_score_scale)}"
            f"__neg-{p.negative_count}"
            f"__loss-{p.pairwise_loss}"
            f"__m-{_float_token(p.loss_margin)}"
        )


def _float_token(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace("+", "").replace(".", "p")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Tune geometry-only SL(8) with full-ranking validation only."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILE_STAGES), default="core")
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--max-new-trials",
        type=int,
        help="stop cleanly after this many newly trained trials; rerun to continue",
    )
    parser.add_argument(
        "--existing-paper-result",
        type=Path,
        help=(
            "reuse a completed validation-only production SLRecGraph result for "
            "the exact paper-parameter core trial"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the protocol and print the finite plan without writing/training",
    )
    return parser.parse_args(argv)


def config_paths(repo: Path) -> tuple[Path, ...]:
    root = repo / "baseline_config_fixed"
    return tuple(root / name for name in CONFIG_NAMES)


def validate_protocol(repo: Path) -> None:
    merged: dict[str, Any] = {}
    for path in config_paths(repo):
        if not path.is_file():
            raise FileNotFoundError(f"required config does not exist: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"config is not a mapping: {path}")
        merged.update(payload)

    expected = {
        "dataset": DATASET,
        "seed": SEED,
        "embedding_size": 64,
        "matrix_dim": 8,
        "num_factors": 1,
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
    actual = {key: merged.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            "Hgformer-compatible SL(8) full-ranking protocol changed; refusing "
            f"to tune: expected={expected}, actual={actual}"
        )
    if any("sample" in name.lower() for name in CONFIG_NAMES):
        raise RuntimeError("sampled-evaluation overlays are forbidden")


def runtime_protocol(epochs: int) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "seed": SEED,
        "architecture": {
            "matrix_dim": 8,
            "num_factors": 1,
            "embedding_size": 64,
            "intrinsic_dimension": 63,
            "n_layers": 0,
        },
        "geometry_fixed": {
            "sl_scale": 1.0,
            "factor_aggregation": "l2",
            "log_terms": 12,
            "log_jitter": 0.0,
            "symmetric_distance": False,
        },
        "validation": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
            "metrics": ["Recall", "NDCG"],
            "topk": [5, 10, 20, 50],
            "selection_metric": "Recall@10",
        },
        "training_budget": {
            "epochs": epochs,
            "eval_step": epochs,
            "validation_events_per_trial": 1,
            "selection": "fixed final epoch",
            "early_stopping": "effectively disabled: the only validation is after the final epoch",
        },
        "loss_contract": {
            "bpr": "exact production softplus(s_neg - s_pos) mean",
            "hinge": "loss-only adapter: relu(margin + s_neg - s_pos) mean",
            "negative_sampling": (
                "RecBole uniform pairwise sampling; counts above one repeat each "
                "positive into independent positive-negative pairs, not InfoNCE"
            ),
            "negative_count_cost": (
                "train_batch_size remains 8192 scored pairs, so larger counts increase "
                "batches/optimizer steps and wall time per epoch"
            ),
        },
        "test_evaluated": False,
    }


def _unique_trials(stage: str, parameters: Sequence[Parameters]) -> tuple[Trial, ...]:
    trials: list[Trial] = []
    seen: set[Parameters] = set()
    for item in parameters:
        item.validate()
        if item not in seen:
            trials.append(Trial(stage, item))
            seen.add(item)
    return tuple(trials)


def build_stage_trials(stage: str, anchor: Parameters | None = None) -> tuple[Trial, ...]:
    """Instantiate one finite stage; carried anchors are not retrained."""

    if stage == "core_lr_clip":
        if anchor is not None:
            raise ValueError("the core stage does not take an anchor")
        parameters = [
            replace(PAPER_PARAMETERS, learning_rate=lr, coord_clip=clip)
            for lr, clip in itertools.product(CORE_LEARNING_RATES, CORE_COORD_CLIPS)
        ]
        return _unique_trials(stage, parameters)
    if anchor is None:
        raise ValueError(f"stage {stage!r} requires the preceding winner")

    if stage == "initialisation":
        parameters = [replace(anchor, init_std=value) for value in (0.005, 0.01, 0.02)]
    elif stage == "regularisation":
        parameters = [
            replace(anchor, reg_weight=value) for value in (1e-7, 1e-6, 1e-5)
        ]
    elif stage == "metric_scale":
        parameters = [
            replace(anchor, schatten_p="1", score_scale=1.0, learnable_score_scale=True),
            replace(anchor, schatten_p="inf", score_scale=1.0, learnable_score_scale=True),
            replace(anchor, schatten_p="2", score_scale=1.0, learnable_score_scale=False),
            replace(anchor, schatten_p="2", score_scale=5.0, learnable_score_scale=False),
            replace(anchor, schatten_p="2", score_scale=5.0, learnable_score_scale=True),
        ]
    elif stage == "loss_neg":
        parameters = [
            replace(anchor, pairwise_loss="bpr", negative_count=4),
            replace(anchor, pairwise_loss="hinge", loss_margin=0.5, negative_count=1),
            replace(anchor, pairwise_loss="hinge", loss_margin=1.0, negative_count=1),
            replace(anchor, pairwise_loss="hinge", loss_margin=1.0, negative_count=4),
        ]
    else:
        raise ValueError(f"unknown stage: {stage!r}")

    # The preceding winner is carried into the stage ranking without spending
    # another 500-epoch run.  Exclude an accidental identical variant here.
    return _unique_trials(stage, [item for item in parameters if item != anchor])


def finite_search_manifest(profile: str, epochs: int) -> dict[str, Any]:
    core_count = len(build_stage_trials("core_lr_clip"))
    extended_new_counts = {
        "initialisation": 3,
        "regularisation": 3,
        "metric_scale": 5,
        "loss_neg": 4,
    }
    maximum = core_count
    if profile == "extended":
        maximum += sum(extended_new_counts.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "protocol": runtime_protocol(epochs),
        "finite_search": {
            "core": {
                "design": "complete Cartesian product",
                "learning_rate": list(CORE_LEARNING_RATES),
                "coord_clip": list(CORE_COORD_CLIPS),
                "fixed": asdict(PAPER_PARAMETERS),
                "training_trials": core_count,
            },
            "extended": {
                "design": "greedy staged ablations; previous winner is carried",
                "stage_order": list(EXTENDED_STAGES),
                "initialisation": {"init_std": [0.005, 0.01, 0.02]},
                "regularisation": {"reg_weight": [1e-7, 1e-6, 1e-5]},
                "metric_scale": {
                    "variants": [
                        {"schatten_p": "1", "score_scale": 1.0, "learnable": True},
                        {"schatten_p": "inf", "score_scale": 1.0, "learnable": True},
                        {"schatten_p": "2", "score_scale": 1.0, "learnable": False},
                        {"schatten_p": "2", "score_scale": 5.0, "learnable": False},
                        {"schatten_p": "2", "score_scale": 5.0, "learnable": True},
                    ]
                },
                "loss_neg": {
                    "variants": [
                        {"loss": "bpr", "negative_count": 4},
                        {"loss": "hinge", "margin": 0.5, "negative_count": 1},
                        {"loss": "hinge", "margin": 1.0, "negative_count": 1},
                        {"loss": "hinge", "margin": 1.0, "negative_count": 4},
                    ]
                },
                "additional_training_trials_at_most": sum(extended_new_counts.values()),
            },
            "maximum_training_trials_for_profile": maximum,
            "not_a_full_cartesian_product": (
                "Only lr x clip is exhaustively crossed. Crossing every continuous and "
                "categorical parameter would be unbounded and scientifically wasteful."
            ),
        },
        "artifacts": {
            "manifest": "manifest.json",
            "summary": "summary.json",
            "per_stage": "stages/<stage>/{results,logs,checkpoints,summary.json}",
        },
        "stages": {},
    }


def trial_command(
    args: argparse.Namespace,
    trial: Trial,
    result_path: Path,
    checkpoint_dir: Path,
) -> list[str]:
    configs = " ".join(str(path) for path in config_paths(args.repo))
    return [
        args.python,
        "-u",
        str(args.repo / "run_recbole_gnn.py"),
        "--model",
        trial.parameters.model_name,
        "--dataset",
        DATASET,
        "--config-files",
        configs,
        "--validation-only",
        "--result-file",
        str(result_path),
        f"--checkpoint_dir={checkpoint_dir}",
        # The vendored RecBole Config overwrites CUDA_VISIBLE_DEVICES from
        # gpu_id, so this must remain the physical id rather than logical 0.
        f"--gpu_id={args.gpu_id}",
        "--use_gpu=true",
        "--show_progress=false",
        f"--epochs={args.epochs}",
        f"--eval_step={args.epochs}",
        "--stopping_step=1",
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


def run_and_tee(command: list[str], log_path: Path, cwd: Path, env: Mapping[str, str]) -> None:
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


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parameters_from_result(payload: Mapping[str, Any]) -> Parameters | None:
    tuning = payload.get("tuning")
    if not isinstance(tuning, Mapping) or not isinstance(tuning.get("parameters"), Mapping):
        return None
    values = tuning["parameters"]
    try:
        return Parameters(
            learning_rate=float(values["learning_rate"]),
            coord_clip=float(values["coord_clip"]),
            init_std=float(values["init_std"]),
            reg_weight=float(values["reg_weight"]),
            schatten_p=str(values["schatten_p"]),
            score_scale=float(values["score_scale"]),
            learnable_score_scale=bool(values["learnable_score_scale"]),
            negative_count=int(values["negative_count"]),
            pairwise_loss=str(values["pairwise_loss"]),
            loss_margin=float(values["loss_margin"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _validate_common_result(payload: Mapping[str, Any], path: Path) -> None:
    if payload.get("dataset") not in (None, DATASET):
        raise ValueError(f"unexpected dataset in result: {path}")
    if int(payload.get("seed", -1)) != SEED:
        raise ValueError(f"unexpected seed in result: {path}")
    if payload.get("test_result") is not None:
        raise RuntimeError(f"tuning result touched the held-out test split: {path}")
    score = payload.get("best_valid_score")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError(f"result has no finite validation score: {path}")
    metrics = payload.get("best_valid_result")
    if not isinstance(metrics, Mapping) or "recall@10" not in metrics:
        raise ValueError(f"result has no full validation Recall@10: {path}")
    checkpoint = payload.get("checkpoint_file")
    if not checkpoint or not Path(checkpoint).expanduser().is_file():
        raise ValueError(f"result has no existing checkpoint: {path}")
    fingerprints = payload.get("split_fingerprints")
    if not isinstance(fingerprints, Mapping) or set(fingerprints) != {"train", "valid", "test"}:
        raise ValueError(f"result has no auditable split fingerprints: {path}")


def load_complete_result(
    path: Path,
    trial: Trial,
    *,
    expected_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result JSON is not an object: {resolved}")
    _validate_common_result(payload, resolved)
    if payload.get("model") != trial.parameters.model_name:
        raise ValueError(f"unexpected model in result: {resolved}")
    actual = _parameters_from_result(payload)
    if actual != trial.parameters:
        raise ValueError(
            f"result parameters do not match trial {trial.name}: "
            f"expected={trial.parameters}, actual={actual}"
        )
    tuning = payload["tuning"]
    if tuning.get("trial_name") != trial.name or tuning.get("stage") != trial.stage:
        raise ValueError(f"result tuning identity does not match: {resolved}")
    protocol = tuning.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError(f"result lacks protocol metadata: {resolved}")
    validation = protocol.get("validation")
    if not isinstance(validation, Mapping) or validation.get("mode") != "full":
        raise RuntimeError(f"result was not selected by full-ranking validation: {resolved}")
    if bool(protocol.get("test_evaluated", True)):
        raise RuntimeError(f"result metadata says test was evaluated: {resolved}")
    if protocol != expected_protocol:
        raise ValueError(
            f"result protocol does not match this invocation: {resolved}; "
            f"expected={expected_protocol}, actual={protocol}"
        )
    return payload


def _completed_result(
    path: Path,
    trial: Trial,
    *,
    expected_protocol: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return load_complete_result(path, trial, expected_protocol=expected_protocol)
    except RuntimeError:
        # A test-touched or sampled result must never be silently overwritten.
        raise
    except (OSError, ValueError, json.JSONDecodeError):
        # Interrupted output is retrained in-place; valid checkpoints are never
        # skipped without complete protocol and parameter metadata.
        return None


def add_trial_metadata(
    path: Path,
    trial: Trial,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tuning"] = {
        "schema_version": SCHEMA_VERSION,
        "stage": trial.stage,
        "trial_name": trial.name,
        "parameters": asdict(trial.parameters),
        "protocol": protocol,
        "test_evaluated": False,
    }
    _atomic_json_write(path, payload)
    return load_complete_result(path, trial, expected_protocol=protocol)


def load_existing_paper_result(
    path: Path,
    trial: Trial,
    *,
    expected_epochs: int,
) -> dict[str, Any]:
    """Validate a legacy direct-run result before reusing the exact baseline."""

    if trial.parameters != PAPER_PARAMETERS or trial.parameters.model_name != PRODUCTION_MODEL:
        raise ValueError("an existing paper result can only replace the exact paper trial")
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"paper result JSON is not an object: {resolved}")
    _validate_common_result(payload, resolved)
    if payload.get("model") != PRODUCTION_MODEL:
        raise ValueError(f"paper result is not the production SLRecGraph: {resolved}")
    if int(payload.get("epochs", -1)) != expected_epochs:
        raise ValueError(
            f"paper result did not use {expected_epochs} epochs: {resolved}"
        )
    names = payload.get("config_files")
    if not isinstance(names, list) or [Path(item).name for item in names] != list(CONFIG_NAMES):
        raise ValueError(f"paper result used different config overlays: {resolved}")

    # Direct run JSON does not contain CLI overrides.  They are persisted in
    # the checkpoint Config, so validate them before treating it as a trial.
    import torch

    checkpoint = torch.load(payload["checkpoint_file"], map_location="cpu")
    config = checkpoint.get("config") if isinstance(checkpoint, Mapping) else None
    if config is None:
        raise ValueError(f"paper checkpoint has no saved Config: {resolved}")
    expected = {
        "n_layers": 0,
        "matrix_dim": 8,
        "num_factors": 1,
        "learning_rate": PAPER_PARAMETERS.learning_rate,
        "coord_clip": PAPER_PARAMETERS.coord_clip,
        "init_std": PAPER_PARAMETERS.init_std,
        "reg_weight": PAPER_PARAMETERS.reg_weight,
        "schatten_p": 2,
        "score_scale": PAPER_PARAMETERS.score_scale,
        "learnable_score_scale": True,
        "neg_sampling": {"uniform": 1},
        "symmetric_distance": False,
        "log_terms": 12,
        "log_jitter": 0.0,
        "epochs": expected_epochs,
        "eval_step": expected_epochs,
    }
    actual = {key: config[key] for key in expected}
    if actual != expected or config["eval_args"]["mode"] != "full":
        raise ValueError(
            f"paper checkpoint parameters/protocol do not match: expected={expected}, "
            f"actual={actual}, eval_args={config['eval_args']}"
        )
    return payload


def _candidate(
    trial: Trial,
    source: Path,
    result: Mapping[str, Any],
    *,
    origin: str,
) -> dict[str, Any]:
    return {
        "name": trial.name,
        "stage": trial.stage,
        "origin": origin,
        "source": str(source.expanduser().resolve()),
        "checkpoint_file": result["checkpoint_file"],
        "model": result["model"],
        "parameters": asdict(trial.parameters),
        "best_valid_score": float(result["best_valid_score"]),
        "best_valid_result": result["best_valid_result"],
        "split_fingerprints": result["split_fingerprints"],
        "test_evaluated": False,
    }


def _rank(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(candidates, key=lambda item: (-float(item["best_valid_score"]), item["name"]))


def _assert_same_split(candidates: Sequence[Mapping[str, Any]]) -> None:
    if not candidates:
        return
    expected = candidates[0]["split_fingerprints"]
    for candidate in candidates[1:]:
        if candidate["split_fingerprints"] != expected:
            raise RuntimeError(
                "candidate split fingerprints differ; refusing to compare "
                f"{candidates[0]['name']} and {candidate['name']}"
            )


def write_stage_summary(
    path: Path,
    *,
    stage: str,
    anchor: Mapping[str, Any] | None,
    candidates: Sequence[Mapping[str, Any]],
    expected_new_trials: int,
    complete: bool,
) -> dict[str, Any]:
    _assert_same_split(candidates)
    ranking = _rank(candidates)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "selection_metric": "Recall@10 on full-ranking validation at the final epoch",
        "test_evaluated": False,
        "complete": complete,
        "expected_new_trials": expected_new_trials,
        "completed_candidates_including_carried": len(ranking),
        "carried_from_previous_stage": anchor,
        "best": ranking[0] if complete and ranking else None,
        "provisional_best": ranking[0] if ranking else None,
        "ranking": ranking,
    }
    _atomic_json_write(path, summary)
    return summary


def write_global_summary(
    path: Path,
    *,
    profile: str,
    protocol: Mapping[str, Any],
    completed_trials: Sequence[Mapping[str, Any]],
    stage_summaries: Sequence[Mapping[str, Any]],
    state: str,
    final_best: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _assert_same_split(completed_trials)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "state": state,
        "selection_metric": "Recall@10 on full-ranking validation at the final epoch",
        "protocol": protocol,
        "test_evaluated": False,
        "completed_training_trials": len(completed_trials),
        "best": final_best if state == "complete" else None,
        "provisional_best": _rank(completed_trials)[0] if completed_trials else None,
        "stage_summaries": [
            {
                "stage": item["stage"],
                "complete": item["complete"],
                "best": item["best"],
            }
            for item in stage_summaries
        ],
        "all_completed_trials_ranking": _rank(completed_trials),
    }
    _atomic_json_write(path, summary)
    return summary


def _dry_run_plan(args: argparse.Namespace, manifest: Mapping[str, Any]) -> dict[str, Any]:
    trials = build_stage_trials("core_lr_clip")
    paper_path = args.existing_paper_result.expanduser().resolve() if args.existing_paper_result else None
    tuning_root = args.output_root / "sl8-full-tuning"
    protocol = runtime_protocol(args.epochs)
    plan: list[dict[str, Any]] = []
    for trial in trials:
        stage_root = tuning_root / "stages" / trial.stage
        result_path = stage_root / "results" / f"{trial.name}.json"
        if paper_path is not None and trial.parameters == PAPER_PARAMETERS:
            load_existing_paper_result(
                paper_path,
                trial,
                expected_epochs=args.epochs,
            )
            status = "reuse-existing-paper-result"
            source = paper_path
        else:
            complete = _completed_result(
                result_path,
                trial,
                expected_protocol=protocol,
            )
            status = "skip-complete" if complete is not None else "run"
            source = result_path
        plan.append(
            {
                "name": trial.name,
                "status": status,
                "source": str(source),
                "parameters": asdict(trial.parameters),
                "command": trial_command(
                    args,
                    trial,
                    result_path,
                    stage_root / "checkpoints" / trial.name,
                ),
            }
        )
    return {
        "dry_run": True,
        "manifest": manifest,
        "instantiated_core_trials": plan,
        "extended_note": (
            "Extended trials are instantiated only after the preceding full-validation "
            "winner is known; their complete finite templates are in the manifest."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.max_new_trials is not None and args.max_new_trials <= 0:
        raise ValueError("--max-new-trials must be positive")
    args.repo = args.repo.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.existing_paper_result:
        args.existing_paper_result = args.existing_paper_result.expanduser().resolve()
    validate_protocol(args.repo)
    protocol = runtime_protocol(args.epochs)
    manifest = finite_search_manifest(args.profile, args.epochs)

    if args.dry_run:
        print(json.dumps(_dry_run_plan(args, manifest), indent=2))
        return 0

    tuning_root = args.output_root / "sl8-full-tuning"
    manifest_path = tuning_root / "manifest.json"
    summary_path = tuning_root / "summary.json"
    _atomic_json_write(manifest_path, manifest)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    completed_trials: list[dict[str, Any]] = []
    stage_summaries: list[dict[str, Any]] = []
    anchor: dict[str, Any] | None = None
    new_trials_run = 0

    for stage in PROFILE_STAGES[args.profile]:
        anchor_parameters = Parameters(**anchor["parameters"]) if anchor else None
        trials = build_stage_trials(stage, anchor_parameters)
        stage_root = tuning_root / "stages" / stage
        results_dir = stage_root / "results"
        logs_dir = stage_root / "logs"
        checkpoints_dir = stage_root / "checkpoints"
        stage_summary_path = stage_root / "summary.json"
        for directory in (results_dir, logs_dir, checkpoints_dir):
            directory.mkdir(parents=True, exist_ok=True)

        manifest["stages"][stage] = {
            "anchor": anchor,
            "new_trial_count": len(trials),
            "trials": [
                {"name": trial.name, "parameters": asdict(trial.parameters)}
                for trial in trials
            ],
        }
        _atomic_json_write(manifest_path, manifest)

        candidates: list[dict[str, Any]] = [anchor] if anchor else []
        completed_in_stage = 0
        for index, trial in enumerate(trials, 1):
            result_path = results_dir / f"{trial.name}.json"
            use_paper = (
                stage == "core_lr_clip"
                and args.existing_paper_result is not None
                and trial.parameters == PAPER_PARAMETERS
            )
            if use_paper:
                result_path = args.existing_paper_result
                result = load_existing_paper_result(
                    result_path,
                    trial,
                    expected_epochs=args.epochs,
                )
                origin = "existing-paper-result"
                print(f"[{stage} {index}/{len(trials)}] reuse paper result {trial.name}")
            else:
                result = _completed_result(
                    result_path,
                    trial,
                    expected_protocol=protocol,
                )
                if result is not None:
                    origin = "resumed-complete"
                    print(f"[{stage} {index}/{len(trials)}] skip complete {trial.name}")
                else:
                    if (
                        args.max_new_trials is not None
                        and new_trials_run >= args.max_new_trials
                    ):
                        stage_summary = write_stage_summary(
                            stage_summary_path,
                            stage=stage,
                            anchor=anchor,
                            candidates=candidates,
                            expected_new_trials=len(trials),
                            complete=False,
                        )
                        stage_summaries.append(stage_summary)
                        write_global_summary(
                            summary_path,
                            profile=args.profile,
                            protocol=protocol,
                            completed_trials=completed_trials,
                            stage_summaries=stage_summaries,
                            state="paused-by-max-new-trials",
                            final_best=None,
                        )
                        print(f"PAUSED_AFTER_NEW_TRIALS={new_trials_run}")
                        print(f"SUMMARY_JSON={summary_path}")
                        return 0
                    command = trial_command(
                        args,
                        trial,
                        result_path,
                        checkpoints_dir / trial.name,
                    )
                    print(f"[{stage} {index}/{len(trials)}] start {trial.name}")
                    run_and_tee(command, logs_dir / f"{trial.name}.log", args.repo, env)
                    result = add_trial_metadata(result_path, trial, protocol=protocol)
                    origin = "trained"
                    new_trials_run += 1

            candidate = _candidate(trial, result_path, result, origin=origin)
            candidates.append(candidate)
            if origin != "existing-paper-result":
                completed_trials.append(candidate)
            else:
                # The imported paper trial is still a completed comparable
                # training trial; append it once to the global ranking.
                completed_trials.append(candidate)
            completed_in_stage += 1
            write_stage_summary(
                stage_summary_path,
                stage=stage,
                anchor=anchor,
                candidates=candidates,
                expected_new_trials=len(trials),
                complete=False,
            )
            write_global_summary(
                summary_path,
                profile=args.profile,
                protocol=protocol,
                completed_trials=completed_trials,
                stage_summaries=stage_summaries,
                state="running",
                final_best=None,
            )

        if completed_in_stage != len(trials):
            raise RuntimeError(f"internal error: stage {stage} ended incompletely")
        stage_summary = write_stage_summary(
            stage_summary_path,
            stage=stage,
            anchor=anchor,
            candidates=candidates,
            expected_new_trials=len(trials),
            complete=True,
        )
        stage_summaries.append(stage_summary)
        anchor = dict(stage_summary["best"])
        print(
            f"stage {stage} best={anchor['name']} "
            f"full Recall@10={anchor['best_valid_score']:.6f}"
        )

    final = write_global_summary(
        summary_path,
        profile=args.profile,
        protocol=protocol,
        completed_trials=completed_trials,
        stage_summaries=stage_summaries,
        state="complete",
        final_best=anchor,
    )
    print(f"MANIFEST_JSON={manifest_path}")
    print(f"SUMMARY_JSON={summary_path}")
    print(
        f"complete best={final['best']['name']} "
        f"full Recall@10={final['best']['best_valid_score']:.6f}; test untouched"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
