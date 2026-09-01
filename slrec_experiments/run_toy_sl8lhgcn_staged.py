#!/usr/bin/env python3
"""Adaptive, validation-only tuning for Amazon-Toy SL8-LHGCN.

This runner starts *only* after the strict layer-by-batch grid has a complete
summary.  It reads and deeply validates that grid's winner, then tunes one
parameter family at a time.  The winner of each completed stage is the parent
of the next stage, so this is 21 new trials rather than a 4*4*5*5*5*4 Cartesian
product.  The unchanged parent is reused as one candidate in every stage.

The stage order is deliberately fixed:

1. learning rate: 1e-4, 3e-4, 5e-4, 1e-3
2. faithful-hinge margin: .05, .1, .2, .3
3. Lie-coordinate Frobenius cap: .25, .5, .75, 1.0, 1.5
4. Schatten order: 1, 2, 4, 8, infinity
5. Adam weight decay: 0, 1e-5, 1e-4, 1e-3, 5e-3
6. self loop: off, or on with weight .1, .5, 1.0

Every model-selection subprocess uses exactly physical GPU 7, 500 epochs,
validation every 50 epochs, and full-ranking validation.  Only after the final
500-epoch winner is frozen, two separate 750/1000-epoch budget-sensitivity
runs are allowed; they are reported separately and can never replace the fair
500-epoch winner.  There is intentionally no held-out-test execution path.
Results are resumable only after their complete checkpoint, split-fingerprint,
protocol, and SL(8) manifold contracts pass.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from slrec_experiments import run_toy_sl8lhgcn_grid as layer_grid
except ModuleNotFoundError:  # Allows direct execution from slrec_experiments/.
    import run_toy_sl8lhgcn_grid as layer_grid  # type: ignore


SCHEMA_VERSION = 1
PHYSICAL_GPU = "7"
EXPECTED_NEW_TRIALS = 21
EPOCH_EXTENSION_BUDGETS = (750, 1000)


@dataclass(frozen=True)
class Parameters:
    """The active, stage-carried model/training configuration."""

    gcn_layers: int
    train_batch_size: int
    learning_rate: float = 5e-4
    loss_margin: float = 0.1
    coord_clip: float = 0.75
    schatten_p: int | str = 2
    weight_decay: float = 0.0
    lhgcn_include_self: bool = False
    lhgcn_self_loop_weight: float = 1.0
    epochs: int = layer_grid.EPOCHS

    def validate(self) -> None:
        if self.gcn_layers not in layer_grid.LAYERS:
            raise ValueError(f"unexpected prerequisite layer count: {self.gcn_layers}")
        if self.train_batch_size not in layer_grid.BATCH_SIZES:
            raise ValueError(
                f"unexpected prerequisite batch size: {self.train_batch_size}"
            )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.loss_margin < 0:
            raise ValueError("loss_margin must be non-negative")
        if self.coord_clip <= 0:
            raise ValueError("coord_clip must be positive in this controlled sweep")
        if self.schatten_p not in {1, 2, 4, 8, "inf"}:
            raise ValueError(f"unsupported staged Schatten order: {self.schatten_p!r}")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.lhgcn_self_loop_weight <= 0:
            raise ValueError("lhgcn_self_loop_weight must be positive")
        if not self.lhgcn_include_self and self.lhgcn_self_loop_weight != 1.0:
            raise ValueError(
                "self-loop weight is canonicalised to 1.0 while self loops are off"
            )
        if self.epochs not in {layer_grid.EPOCHS, *EPOCH_EXTENSION_BUDGETS}:
            raise ValueError(f"unsupported epoch budget: {self.epochs}")

    @property
    def signature(self) -> str:
        self.validate()
        return layer_grid._canonical_hash(asdict(self))


@dataclass(frozen=True)
class Candidate:
    label: str
    updates: Mapping[str, Any]


@dataclass(frozen=True)
class Stage:
    index: int
    key: str
    candidates: tuple[Candidate, ...]


STAGES = (
    Stage(
        1,
        "learning_rate",
        tuple(
            Candidate(label, {"learning_rate": value})
            for label, value in (
                ("lr_1e-4", 1e-4),
                ("lr_3e-4", 3e-4),
                ("lr_5e-4", 5e-4),
                ("lr_1e-3", 1e-3),
            )
        ),
    ),
    Stage(
        2,
        "loss_margin",
        tuple(
            Candidate(label, {"loss_margin": value})
            for label, value in (
                ("margin_0p05", 0.05),
                ("margin_0p10", 0.1),
                ("margin_0p20", 0.2),
                ("margin_0p30", 0.3),
            )
        ),
    ),
    Stage(
        3,
        "coord_clip",
        tuple(
            Candidate(label, {"coord_clip": value})
            for label, value in (
                ("clip_0p25", 0.25),
                ("clip_0p50", 0.5),
                ("clip_0p75", 0.75),
                ("clip_1p00", 1.0),
                ("clip_1p50", 1.5),
            )
        ),
    ),
    Stage(
        4,
        "schatten_p",
        tuple(
            Candidate(label, {"schatten_p": value})
            for label, value in (
                ("p_1", 1),
                ("p_2", 2),
                ("p_4", 4),
                ("p_8", 8),
                ("p_inf", "inf"),
            )
        ),
    ),
    Stage(
        5,
        "weight_decay",
        tuple(
            Candidate(label, {"weight_decay": value})
            for label, value in (
                ("wd_0", 0.0),
                ("wd_1e-5", 1e-5),
                ("wd_1e-4", 1e-4),
                ("wd_1e-3", 1e-3),
                ("wd_5e-3", 5e-3),
            )
        ),
    ),
    Stage(
        6,
        "self_loop",
        (
            Candidate(
                "self_off",
                {
                    "lhgcn_include_self": False,
                    "lhgcn_self_loop_weight": 1.0,
                },
            ),
            Candidate(
                "self_on_w0p1",
                {
                    "lhgcn_include_self": True,
                    "lhgcn_self_loop_weight": 0.1,
                },
            ),
            Candidate(
                "self_on_w0p5",
                {
                    "lhgcn_include_self": True,
                    "lhgcn_self_loop_weight": 0.5,
                },
            ),
            Candidate(
                "self_on_w1p0",
                {
                    "lhgcn_include_self": True,
                    "lhgcn_self_loop_weight": 1.0,
                },
            ),
        ),
    ),
)


PARAMETER_STATUS: Mapping[str, Mapping[str, str]] = {
    "staged_active": {
        "learning_rate": "Adam step size; directly changes optimisation.",
        "loss_margin": "Appears in the faithful squared-distance hinge loss.",
        "coord_clip": "Caps trace-free coordinates before matrix_exp.",
        "schatten_p": "Changes the singular-value norm in the SL distance.",
        "weight_decay": "Adam weight decay; active even though reg_weight is not.",
        "lhgcn_include_self": "Changes the normalised graph operator.",
        "lhgcn_self_loop_weight": "Active only when lhgcn_include_self=true.",
    },
    "already_selected": {
        "gcn_layers": "Read from the complete layer-by-batch grid winner.",
        "train_batch_size": "Read from the complete layer-by-batch grid winner.",
    },
    "separate_budget_sensitivity": {
        "epochs": (
            "The fair search is fixed at 500. Only its frozen final winner is "
            "rerun at 750 and 1000, outside the model-selection ranking."
        ),
    },
    "inactive_or_dead_in_current_model": {
        "init_std": (
            "Dead while embedding_init=xavier_uniform_combined: the subclass "
            "overwrites the base normal initialisation."
        ),
        "reg_weight": (
            "Dead in pairwise_loss=lhgcn_hinge_squared_sum; it is used only in "
            "the BPR branch."
        ),
        "score_scale/max_score_scale": (
            "Not used by the faithful training loss; a fixed positive scale also "
            "does not change full-ranking order."
        ),
        "learnable_score_scale": (
            "Must be false for faithful hinge and its parameter has no gradient."
        ),
        "factor_aggregation": "Dead with num_factors=1.",
        "n_layers": "An alias shadowed by gcn_layers in SL8LHGCN; kept equal only.",
        "margin": "Alias shadowed because loss_margin is explicitly present.",
        "lhgcn_self_loop_weight_when_off": (
            "Conditional dead parameter; canonicalised to 1.0 when loops are off."
        ),
    },
    "active_but_deferred_or_not_for_model_selection": {
        "sl_scale": (
            "Active but strongly confounded with coord_clip; tune only in a "
            "separate scale/clip study after this sequence."
        ),
        "embedding_init": "Active architecture/initialisation ablation; not init_std.",
        "sl_gcn_mode": "Architecture ablation (ambient vs tangent), not mixed into tuning.",
        "negative_sample_count": "Active but changes training cost and protocol.",
        "log_terms": (
            "Numerical approximation accuracy/cost; choose by convergence audit, "
            "not validation cherry-picking."
        ),
        "log_jitter": "Numerical stabiliser fixed at zero under the strict audit.",
        "sl_centroid_fallback_clip": (
            "Dormant when active singular fallbacks are zero; any active fallback "
            "makes a staged trial invalid."
        ),
        "eval_user_chunk_size/eval_item_chunk_size": (
            "Memory/runtime knobs only; they do not change full-ranking metrics."
        ),
        "matrix_dim/num_factors": "Change the manifold and parameter budget.",
        "lhgcn_layer_aggregation/sl_layer_norm": "Only last/none are implemented here.",
    },
}


@dataclass(frozen=True)
class Prerequisite:
    summary_path: Path
    summary_sha256: str
    winner_result_path: Path
    winner_result_sha256: str
    split_fingerprints: Mapping[str, Any]
    parameters: Parameters
    recall_at_10: float
    ndcg_at_10: float
    checkpoint_file: str
    artifact_signature: str


@dataclass(frozen=True)
class Artifact:
    result_path: Path
    result_sha256: str
    parameters: Parameters
    recall_at_10: float
    ndcg_at_10: float
    checkpoint_file: str
    split_fingerprints: Mapping[str, Any]
    artifact_signature: str
    source: str


@dataclass(frozen=True)
class StagedTrial:
    stage_index: int
    stage_key: str
    candidate_index: int
    candidate_label: str
    parameters: Parameters
    parent_artifact_signature: str

    @property
    def contract_signature(self) -> str:
        core = {
            "stage_index": self.stage_index,
            "stage_key": self.stage_key,
            "candidate_index": self.candidate_index,
            "candidate_label": self.candidate_label,
            "parameters": asdict(self.parameters),
            "parent_artifact_signature": self.parent_artifact_signature,
        }
        return layer_grid._canonical_hash(core)

    @property
    def name(self) -> str:
        return f"{self.candidate_label}__{self.contract_signature[:12]}"


@dataclass(frozen=True)
class CandidateState:
    trial: StagedTrial
    status: str
    artifact: Artifact | None
    invalid_reason: str | None = None


@dataclass(frozen=True)
class StageState:
    stage: Stage
    parent: Artifact
    candidates: tuple[CandidateState, ...]
    complete: bool
    winner: CandidateState | None


@dataclass(frozen=True)
class EpochExtensionState:
    parent_500: Artifact
    candidates: tuple[CandidateState, ...]
    complete: bool


@dataclass(frozen=True)
class TuningState:
    prerequisite: Prerequisite
    stages: tuple[StageState, ...]
    current_stage: StageState | None
    final_artifact: Artifact
    selection_complete: bool
    epoch_extension: EpochExtensionState | None
    complete: bool


class PrerequisiteNotReady(RuntimeError):
    """The layer-by-batch grid is absent or still legitimately incomplete."""


def apply_candidate(parent: Parameters, candidate: Candidate) -> Parameters:
    fields = {field.name for field in dataclasses.fields(Parameters)}
    unknown = set(candidate.updates).difference(fields)
    if unknown:
        raise ValueError(f"candidate updates unknown parameters: {sorted(unknown)}")
    updated = replace(parent, **dict(candidate.updates))
    updated.validate()
    return updated


def stage_trials(stage: Stage, parent: Artifact) -> tuple[StagedTrial, ...]:
    trials = tuple(
        StagedTrial(
            stage_index=stage.index,
            stage_key=stage.key,
            candidate_index=index,
            candidate_label=candidate.label,
            parameters=apply_candidate(parent.parameters, candidate),
            parent_artifact_signature=parent.artifact_signature,
        )
        for index, candidate in enumerate(stage.candidates)
    )
    if sum(trial.parameters == parent.parameters for trial in trials) != 1:
        raise RuntimeError(
            f"stage {stage.key} must contain exactly one carry-forward candidate"
        )
    return trials


def epoch_extension_trials(parent: Artifact) -> tuple[StagedTrial, ...]:
    """Create budget-sensitivity runs without a 500-vs-longer winner choice."""

    if parent.parameters.epochs != layer_grid.EPOCHS:
        raise ValueError("epoch extensions must descend from the fair 500-epoch winner")
    return tuple(
        StagedTrial(
            stage_index=7,
            stage_key="epoch_extension",
            candidate_index=index,
            candidate_label=f"epochs_{epochs}",
            parameters=replace(parent.parameters, epochs=epochs),
            parent_artifact_signature=parent.artifact_signature,
        )
        for index, epochs in enumerate(EPOCH_EXTENSION_BUDGETS)
    )


def _equivalent(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-12)
    return expected == actual


def expected_checkpoint_values(parameters: Parameters) -> dict[str, Any]:
    parameters.validate()
    expected = layer_grid.expected_checkpoint_values(
        layer_grid.Trial(parameters.gcn_layers, parameters.train_batch_size)
    )
    expected.update(
        {
            "epochs": parameters.epochs,
            "init_std": 0.01,
            "sl_scale": 1.0,
            "coord_clip": parameters.coord_clip,
            "lhgcn_include_self": parameters.lhgcn_include_self,
            "lhgcn_self_loop_weight": parameters.lhgcn_self_loop_weight,
            "sl_centroid_fallback_clip": 1.0,
            "sl_membership_tolerance": 1e-4,
            "sl_distance_membership_check": True,
            "sl_distance_check_samples": 16,
            "sl_log_trace_tolerance": 1e-3,
            "schatten_p": parameters.schatten_p,
            "max_score_scale": 100.0,
            "loss_margin": parameters.loss_margin,
            "learning_rate": parameters.learning_rate,
            "weight_decay": parameters.weight_decay,
            "tail_analysis": False,
            "popularity_analysis": False,
        }
    )
    return expected


def validate_checkpoint_contract(
    path: Path, repo: Path, parameters: Parameters
) -> Any:
    config, checkpoint_epoch = layer_grid._load_checkpoint_config(path, repo)
    expected = expected_checkpoint_values(parameters)
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if not _equivalent(value, config.get(key))
    }
    if mismatches:
        raise ValueError(f"checkpoint staged contract mismatch: {mismatches}")
    return checkpoint_epoch


def _artifact_signature(
    *,
    result_path: Path,
    result_sha256: str,
    parameters: Parameters,
    recall: float,
    ndcg: float,
    split_fingerprints: Mapping[str, Any],
) -> str:
    return layer_grid._canonical_hash(
        {
            "result_path": str(result_path.expanduser().resolve()),
            "result_sha256": result_sha256,
            "parameters": asdict(parameters),
            "recall@10": recall,
            "ndcg@10": ndcg,
            "split_fingerprints": split_fingerprints,
            "test_evaluated": False,
        }
    )


def _artifact_from_payload(
    path: Path,
    payload: Mapping[str, Any],
    parameters: Parameters,
    *,
    source: str,
) -> Artifact:
    result_path = path.expanduser().resolve()
    metrics = payload["best_valid_result"]
    recall = float(metrics["recall@10"])
    ndcg = float(metrics["ndcg@10"])
    digest = layer_grid._sha256(result_path)
    split = payload["split_fingerprints"]
    return Artifact(
        result_path=result_path,
        result_sha256=digest,
        parameters=parameters,
        recall_at_10=recall,
        ndcg_at_10=ndcg,
        checkpoint_file=str(payload["checkpoint_file"]),
        split_fingerprints=split,
        artifact_signature=_artifact_signature(
            result_path=result_path,
            result_sha256=digest,
            parameters=parameters,
            recall=recall,
            ndcg=ndcg,
            split_fingerprints=split,
        ),
        source=source,
    )


def load_complete_prerequisite(summary_path: Path, repo: Path) -> Prerequisite:
    """Read, deeply validate, and freeze the completed layer/batch winner."""

    path = summary_path.expanduser().resolve()
    if not path.is_file():
        raise PrerequisiteNotReady(f"layer-by-batch summary does not exist: {path}")
    try:
        summary = layer_grid._load_mapping(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise PrerequisiteNotReady(
            f"layer-by-batch summary is not readable: {error}"
        ) from error

    if summary.get("state") != "complete":
        completed = (summary.get("grid") or {}).get("completed_trials")
        raise PrerequisiteNotReady(
            f"layer-by-batch grid is not complete ({completed}/15)"
        )
    if summary.get("kind") != "amazon-toy-sl8lhgcn-layer-batch-grid-summary":
        raise ValueError("wrong prerequisite summary kind")
    if summary.get("test_evaluated") is not False:
        raise RuntimeError("prerequisite summary touched or ambiguously reports test use")

    protocol = layer_grid.validate_protocol(repo)
    if summary.get("protocol") != protocol:
        raise ValueError("prerequisite protocol/config hashes differ from this checkout")
    grid = summary.get("grid")
    if not isinstance(grid, Mapping):
        raise ValueError("prerequisite summary has no grid contract")
    expected_grid = {
        "gcn_layers": list(layer_grid.LAYERS),
        "train_batch_size": list(layer_grid.BATCH_SIZES),
        "expected_trials": len(layer_grid.grid_trials()),
    }
    if any(grid.get(key) != value for key, value in expected_grid.items()):
        raise ValueError("prerequisite grid domain is not the requested 5x3 grid")
    if int(grid.get("completed_trials", -1)) != len(layer_grid.grid_trials()):
        raise ValueError("prerequisite complete state has the wrong completed count")
    if grid.get("pending_trials") or grid.get("invalid_results") or grid.get("failed_trials"):
        raise ValueError("prerequisite complete state still contains pending/invalid/failed trials")
    ranking = summary.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != len(layer_grid.grid_trials()):
        raise ValueError("prerequisite summary does not rank all 15 trials")
    if any(
        not isinstance(row, Mapping) or row.get("test_evaluated") is not False
        for row in ranking
    ):
        raise RuntimeError("a prerequisite candidate touched or ambiguously reports test use")
    acceptance = summary.get("manifold_acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get(
        "every_completed_trial_passed"
    ) is not True:
        raise ValueError("not every prerequisite trial passed the manifold audit")

    split = layer_grid.validate_split_fingerprints(summary)
    winner = summary.get("winner")
    if not isinstance(winner, Mapping) or winner != ranking[0]:
        raise ValueError("prerequisite winner is absent or differs from ranking[0]")
    layers = int(winner.get("gcn_layers", -1))
    batch = int(winner.get("train_batch_size", -1))
    parameters = Parameters(gcn_layers=layers, train_batch_size=batch)
    parameters.validate()
    result_token = winner.get("result_file")
    if not isinstance(result_token, str) or not result_token:
        raise ValueError("prerequisite winner has no result file")
    result_path = Path(result_token).expanduser().resolve()
    result, reason = layer_grid.completed_result(
        result_path,
        repo=repo,
        trial=layer_grid.Trial(layers, batch),
        protocol=protocol,
    )
    if result is None:
        raise ValueError(f"prerequisite winner failed deep result validation: {reason}")
    if result["split_fingerprints"] != split:
        raise ValueError("prerequisite winner split differs from summary split")
    checkpoint = Path(str(result["checkpoint_file"])).expanduser()
    validate_checkpoint_contract(checkpoint, repo, parameters)
    metrics = result["best_valid_result"]
    recall = float(metrics["recall@10"])
    ndcg = float(metrics["ndcg@10"])
    if not _equivalent(recall, winner.get("recall@10")) or not _equivalent(
        ndcg, winner.get("ndcg@10")
    ):
        raise ValueError("prerequisite winner metrics differ from its result artifact")
    result_digest = layer_grid._sha256(result_path)
    artifact_signature = _artifact_signature(
        result_path=result_path,
        result_sha256=result_digest,
        parameters=parameters,
        recall=recall,
        ndcg=ndcg,
        split_fingerprints=split,
    )
    return Prerequisite(
        summary_path=path,
        summary_sha256=layer_grid._sha256(path),
        winner_result_path=result_path,
        winner_result_sha256=result_digest,
        split_fingerprints=split,
        parameters=parameters,
        recall_at_10=recall,
        ndcg_at_10=ndcg,
        checkpoint_file=str(result["checkpoint_file"]),
        artifact_signature=artifact_signature,
    )


def wait_for_prerequisite(
    summary_path: Path,
    repo: Path,
    *,
    poll_seconds: float,
    timeout_seconds: float | None,
) -> Prerequisite:
    if poll_seconds <= 0:
        raise ValueError("prerequisite poll interval must be positive")
    started = time.monotonic()
    last_message: str | None = None
    while True:
        try:
            return load_complete_prerequisite(summary_path, repo)
        except PrerequisiteNotReady as error:
            message = str(error)
            if message != last_message:
                print(f"WAIT prerequisite: {message}", flush=True)
                last_message = message
            elapsed = time.monotonic() - started
            if timeout_seconds is not None and elapsed >= timeout_seconds:
                raise TimeoutError(
                    f"timed out waiting for complete layer-by-batch summary: {message}"
                ) from error
            remaining = None if timeout_seconds is None else timeout_seconds - elapsed
            time.sleep(poll_seconds if remaining is None else min(poll_seconds, max(0.0, remaining)))


def prerequisite_artifact(prerequisite: Prerequisite) -> Artifact:
    return Artifact(
        result_path=prerequisite.winner_result_path,
        result_sha256=prerequisite.winner_result_sha256,
        parameters=prerequisite.parameters,
        recall_at_10=prerequisite.recall_at_10,
        ndcg_at_10=prerequisite.ndcg_at_10,
        checkpoint_file=prerequisite.checkpoint_file,
        split_fingerprints=prerequisite.split_fingerprints,
        artifact_signature=prerequisite.artifact_signature,
        source="layer-by-batch-grid-winner",
    )


def trial_metadata(
    trial: StagedTrial,
    prerequisite: Prerequisite,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": SCHEMA_VERSION,
        "kind": "amazon-toy-sl8lhgcn-staged-tuning-trial",
        "stage": {
            "index": trial.stage_index,
            "key": trial.stage_key,
            "candidate_index": trial.candidate_index,
            "candidate_label": trial.candidate_label,
        },
        "parameters": asdict(trial.parameters),
        "parent_artifact_signature": trial.parent_artifact_signature,
        "prerequisite": {
            "summary_file": str(prerequisite.summary_path),
            "summary_sha256": prerequisite.summary_sha256,
            "winner_result_file": str(prerequisite.winner_result_path),
            "winner_result_sha256": prerequisite.winner_result_sha256,
            "winner_artifact_signature": prerequisite.artifact_signature,
        },
        "protocol": protocol,
        "test_evaluated": False,
    }
    return {**core, "signature_sha256": layer_grid._canonical_hash(core)}


def result_paths(output_root: Path, trial: StagedTrial) -> dict[str, Path]:
    stage_dir = output_root.expanduser() / (
        f"stage_{trial.stage_index:02d}_{trial.stage_key}"
    )
    return {
        "result": stage_dir / "results" / f"{trial.name}.json",
        "raw": stage_dir / "work" / f"{trial.name}.raw.json",
        "log": stage_dir / "logs" / f"{trial.name}.log",
        "checkpoint_dir": stage_dir / "checkpoints" / trial.name,
        "failure": stage_dir / "failures" / f"{trial.name}.json",
    }


def trial_command(
    args: argparse.Namespace,
    trial: StagedTrial,
    raw_result: Path,
    checkpoint_dir: Path,
) -> list[str]:
    parameters = trial.parameters
    parameters.validate()
    base_path, overlay_path = layer_grid.config_paths(args.repo)
    return [
        args.python,
        "-u",
        str(args.repo / "run_recbole_gnn.py"),
        "--model",
        "SL8LHGCN",
        "--dataset",
        layer_grid.DATASET,
        "--config-files",
        f"{base_path} {overlay_path}",
        "--validation-only",
        "--result-file",
        str(raw_result),
        f"--checkpoint_dir={checkpoint_dir}",
        f"--data_path={args.data_root}",
        f"--gpu_id={PHYSICAL_GPU}",
        "--use_gpu=true",
        "--show_progress=false",
        f"--seed={layer_grid.SEED}",
        f"--epochs={parameters.epochs}",
        f"--eval_step={layer_grid.EVAL_STEP}",
        f"--stopping_step={layer_grid.STOPPING_STEP}",
        f"--gcn_layers={parameters.gcn_layers}",
        f"--n_layers={parameters.gcn_layers}",
        f"--train_batch_size={parameters.train_batch_size}",
        "--eval_batch_size=1048576",
        "--eval_user_chunk_size=64",
        "--eval_item_chunk_size=1024",
        "--embedding_size=64",
        "--matrix_dim=8",
        "--num_factors=1",
        "--factor_aggregation=l2",
        "--embedding_init=xavier_uniform_combined",
        "--init_std=0.01",
        f"--coord_clip={parameters.coord_clip}",
        "--sl_scale=1.0",
        "--sl_gcn_mode=ambient_retract",
        f"--lhgcn_include_self={str(parameters.lhgcn_include_self).lower()}",
        f"--lhgcn_self_loop_weight={parameters.lhgcn_self_loop_weight}",
        "--lhgcn_layer_aggregation=last",
        "--sl_layer_norm=none",
        "--sl_centroid_fallback_clip=1.0",
        "--sl_membership_check=true",
        "--sl_membership_strict=true",
        "--sl_membership_tolerance=0.0001",
        "--sl_distance_membership_check=true",
        "--sl_distance_check_samples=16",
        "--sl_log_trace_tolerance=0.001",
        f"--schatten_p={parameters.schatten_p}",
        "--log_terms=12",
        "--log_jitter=0.0",
        "--symmetric_distance=false",
        "--score_scale=1.0",
        "--learnable_score_scale=false",
        "--max_score_scale=100.0",
        "--pairwise_loss=lhgcn_hinge_squared_sum",
        f"--loss_margin={parameters.loss_margin}",
        "--learner=adam",
        f"--learning_rate={parameters.learning_rate}",
        f"--weight_decay={parameters.weight_decay}",
        "--reg_weight=0.0",
        "--neg_sampling={'uniform': 1}",
        "--tail_analysis=false",
        "--popularity_analysis=false",
    ]


def validate_result(
    payload: Mapping[str, Any],
    *,
    repo: Path,
    trial: StagedTrial,
    prerequisite: Prerequisite,
    protocol: Mapping[str, Any],
    require_metadata: bool,
) -> dict[str, Any]:
    if payload.get("model") != "SL8LHGCN" or payload.get("dataset") != layer_grid.DATASET:
        raise ValueError("wrong model or dataset in staged result")
    if int(payload.get("seed", -1)) != layer_grid.SEED:
        raise ValueError("wrong seed in staged result")
    if int(payload.get("epochs", -1)) != trial.parameters.epochs:
        raise ValueError("wrong epoch budget in staged result")
    if int(payload.get("stopping_step", -1)) != layer_grid.STOPPING_STEP:
        raise ValueError("wrong stopping_step in staged result")
    if payload.get("test_result") is not None:
        raise RuntimeError("staged tuning result touched the held-out test split")
    if int(payload.get("parameter_count", -1)) != layer_grid.EXPECTED_PARAMETER_COUNT:
        raise ValueError("unexpected staged SL8-LHGCN parameter count")
    config_files = payload.get("config_files")
    expected_names = [layer_grid.BASE_CONFIG_NAME, layer_grid.MODEL_OVERLAY_NAME]
    if not isinstance(config_files, list) or [
        Path(str(item)).name for item in config_files
    ] != expected_names:
        raise ValueError("staged result did not use the exact Toy + SL8 configs")

    metrics = payload.get("best_valid_result")
    if not isinstance(metrics, Mapping):
        raise ValueError("staged result lacks validation metrics")
    recall = layer_grid._finite_metric(metrics, "recall@10")
    ndcg = layer_grid._finite_metric(metrics, "ndcg@10")
    score = payload.get("best_valid_score")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError("staged result lacks a finite best validation score")
    if not math.isclose(float(score), recall, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("best_valid_score is not full-ranking Recall@10")

    split = layer_grid.validate_split_fingerprints(payload)
    if split != prerequisite.split_fingerprints:
        raise ValueError("staged trial split differs from the prerequisite grid")
    diagnostics = payload.get("model_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("staged result lacks manifold diagnostics")
    manifold = layer_grid.manifold_summary(
        diagnostics,
        layer_grid.Trial(
            trial.parameters.gcn_layers, trial.parameters.train_batch_size
        ),
    )
    checkpoint_token = payload.get("checkpoint_file")
    if not isinstance(checkpoint_token, str) or not checkpoint_token:
        raise ValueError("staged result lacks a checkpoint path")
    checkpoint_epoch = validate_checkpoint_contract(
        Path(checkpoint_token).expanduser(), repo, trial.parameters
    )
    expected_metadata = trial_metadata(trial, prerequisite, protocol)
    if require_metadata and payload.get("toy_sl8_staged_tuning") != expected_metadata:
        raise ValueError("resume metadata differs from the exact staged trial contract")
    runtime = payload.get("staged_runtime")
    if runtime is not None:
        if not isinstance(runtime, Mapping):
            raise ValueError("invalid staged runtime metadata")
        duration = runtime.get("duration_seconds")
        if duration is not None and (
            not isinstance(duration, (int, float)) or float(duration) < 0
        ):
            raise ValueError("invalid staged trial duration")
    return {
        "recall@10": recall,
        "ndcg@10": ndcg,
        "split_fingerprints": split,
        "checkpoint_epoch": checkpoint_epoch,
        "manifold": manifold,
    }


def completed_result(
    path: Path,
    *,
    repo: Path,
    trial: StagedTrial,
    prerequisite: Prerequisite,
    protocol: Mapping[str, Any],
) -> tuple[Artifact | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        payload = layer_grid._load_mapping(path)
        validate_result(
            payload,
            repo=repo,
            trial=trial,
            prerequisite=prerequisite,
            protocol=protocol,
            require_metadata=True,
        )
        return (
            _artifact_from_payload(
                path, payload, trial.parameters, source="staged-trial"
            ),
            None,
        )
    except RuntimeError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return None, str(error)


def annotate_result(
    raw_path: Path,
    final_path: Path,
    *,
    repo: Path,
    trial: StagedTrial,
    prerequisite: Prerequisite,
    protocol: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> Artifact:
    payload = layer_grid._load_mapping(raw_path)
    validate_result(
        payload,
        repo=repo,
        trial=trial,
        prerequisite=prerequisite,
        protocol=protocol,
        require_metadata=False,
    )
    payload["toy_sl8_staged_tuning"] = trial_metadata(
        trial, prerequisite, protocol
    )
    payload["staged_runtime"] = dict(runtime)
    layer_grid._atomic_json(final_path, payload)
    artifact, reason = completed_result(
        final_path,
        repo=repo,
        trial=trial,
        prerequisite=prerequisite,
        protocol=protocol,
    )
    if artifact is None:
        raise RuntimeError(f"annotated staged result failed resume check: {reason}")
    return artifact


def select_winner(
    candidates: Iterable[CandidateState], parent: Artifact
) -> CandidateState:
    complete = [candidate for candidate in candidates if candidate.artifact is not None]
    if not complete:
        raise ValueError("cannot select a winner without completed candidates")
    return min(
        complete,
        key=lambda candidate: (
            -candidate.artifact.recall_at_10,
            -candidate.artifact.ndcg_at_10,
            0
            if candidate.artifact.artifact_signature == parent.artifact_signature
            else 1,
            candidate.trial.candidate_index,
        ),
    )


def build_state(
    *,
    repo: Path,
    output_root: Path,
    prerequisite: Prerequisite,
    protocol: Mapping[str, Any],
) -> TuningState:
    parent = prerequisite_artifact(prerequisite)
    stage_states: list[StageState] = []
    current: StageState | None = None
    for stage in STAGES:
        candidates: list[CandidateState] = []
        for trial in stage_trials(stage, parent):
            if trial.parameters == parent.parameters:
                candidates.append(
                    CandidateState(
                        trial=trial,
                        status="carry-forward",
                        artifact=parent,
                    )
                )
                continue
            paths = result_paths(output_root, trial)
            artifact, invalid_reason = completed_result(
                paths["result"],
                repo=repo,
                trial=trial,
                prerequisite=prerequisite,
                protocol=protocol,
            )
            candidates.append(
                CandidateState(
                    trial=trial,
                    status="complete" if artifact is not None else "pending",
                    artifact=artifact,
                    invalid_reason=invalid_reason,
                )
            )
        complete = all(candidate.artifact is not None for candidate in candidates)
        winner = select_winner(candidates, parent) if complete else None
        stage_state = StageState(
            stage=stage,
            parent=parent,
            candidates=tuple(candidates),
            complete=complete,
            winner=winner,
        )
        stage_states.append(stage_state)
        if not complete:
            current = stage_state
            break
        assert winner is not None and winner.artifact is not None
        parent = winner.artifact
    selection_complete = len(stage_states) == len(STAGES) and current is None
    epoch_extension: EpochExtensionState | None = None
    extensions_complete = False
    if selection_complete:
        extension_candidates: list[CandidateState] = []
        for trial in epoch_extension_trials(parent):
            paths = result_paths(output_root, trial)
            artifact, invalid_reason = completed_result(
                paths["result"],
                repo=repo,
                trial=trial,
                prerequisite=prerequisite,
                protocol=protocol,
            )
            extension_candidates.append(
                CandidateState(
                    trial=trial,
                    status="complete" if artifact is not None else "pending",
                    artifact=artifact,
                    invalid_reason=invalid_reason,
                )
            )
        extensions_complete = all(
            candidate.artifact is not None for candidate in extension_candidates
        )
        epoch_extension = EpochExtensionState(
            parent_500=parent,
            candidates=tuple(extension_candidates),
            complete=extensions_complete,
        )
    return TuningState(
        prerequisite=prerequisite,
        stages=tuple(stage_states),
        current_stage=current,
        final_artifact=parent,
        selection_complete=selection_complete,
        epoch_extension=epoch_extension,
        complete=selection_complete and extensions_complete,
    )


def _artifact_record(artifact: Artifact) -> dict[str, Any]:
    return {
        "result_file": str(artifact.result_path),
        "checkpoint_file": artifact.checkpoint_file,
        "parameters": asdict(artifact.parameters),
        "recall@10": artifact.recall_at_10,
        "ndcg@10": artifact.ndcg_at_10,
        "artifact_signature": artifact.artifact_signature,
        "source": artifact.source,
        "test_evaluated": False,
    }


def stage_definitions() -> list[dict[str, Any]]:
    return [
        {
            "index": stage.index,
            "key": stage.key,
            "candidates": [
                {"label": candidate.label, "updates": dict(candidate.updates)}
                for candidate in stage.candidates
            ],
        }
        for stage in STAGES
    ]


def _failure_records(output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not output_root.is_dir():
        return records
    for path in sorted(output_root.glob("stage_*/failures/*.json")):
        try:
            records.append(layer_grid._load_mapping(path))
        except (OSError, ValueError, json.JSONDecodeError):
            records.append({"failure_file": str(path), "status": "invalid-record"})
    return records


def summary_payload(
    state: TuningState,
    *,
    protocol: Mapping[str, Any],
    output_root: Path,
    data_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    completed_new_trials = 0
    known_durations: list[float] = []
    for stage_state in state.stages:
        candidates: list[dict[str, Any]] = []
        for candidate in stage_state.candidates:
            paths = result_paths(output_root, candidate.trial)
            runtime: Mapping[str, Any] = {}
            if candidate.artifact is not None and candidate.status != "carry-forward":
                completed_new_trials += 1
                try:
                    payload = layer_grid._load_mapping(candidate.artifact.result_path)
                    value = payload.get("staged_runtime") or {}
                    if isinstance(value, Mapping):
                        runtime = value
                        duration = runtime.get("duration_seconds")
                        if isinstance(duration, (int, float)):
                            known_durations.append(float(duration))
                except (OSError, ValueError, json.JSONDecodeError):
                    runtime = {}
            record = {
                "candidate_index": candidate.trial.candidate_index,
                "candidate_label": candidate.trial.candidate_label,
                "status": candidate.status,
                "parameters": asdict(candidate.trial.parameters),
                "contract_signature": candidate.trial.contract_signature,
                "result_file": (
                    str(candidate.artifact.result_path)
                    if candidate.artifact is not None
                    else str(paths["result"])
                ),
                "invalid_existing_reason": candidate.invalid_reason,
                "runtime": dict(runtime),
                "test_evaluated": False,
            }
            if candidate.artifact is not None:
                record.update(
                    {
                        "recall@10": candidate.artifact.recall_at_10,
                        "ndcg@10": candidate.artifact.ndcg_at_10,
                        "artifact_signature": candidate.artifact.artifact_signature,
                    }
                )
            candidates.append(record)
        stages.append(
            {
                "index": stage_state.stage.index,
                "key": stage_state.stage.key,
                "state": "complete" if stage_state.complete else "incomplete",
                "parent": _artifact_record(stage_state.parent),
                "candidates": candidates,
                "winner": (
                    _artifact_record(stage_state.winner.artifact)
                    if stage_state.winner is not None
                    and stage_state.winner.artifact is not None
                    else None
                ),
            }
        )
    extension_record: dict[str, Any] | None = None
    if state.epoch_extension is not None:
        extension_candidates: list[dict[str, Any]] = []
        for candidate in state.epoch_extension.candidates:
            paths = result_paths(output_root, candidate.trial)
            runtime: Mapping[str, Any] = {}
            if candidate.artifact is not None:
                try:
                    payload = layer_grid._load_mapping(candidate.artifact.result_path)
                    value = payload.get("staged_runtime") or {}
                    if isinstance(value, Mapping):
                        runtime = value
                        duration = runtime.get("duration_seconds")
                        if isinstance(duration, (int, float)):
                            known_durations.append(float(duration))
                except (OSError, ValueError, json.JSONDecodeError):
                    runtime = {}
            record = {
                "candidate_label": candidate.trial.candidate_label,
                "epochs": candidate.trial.parameters.epochs,
                "status": candidate.status,
                "parameters": asdict(candidate.trial.parameters),
                "contract_signature": candidate.trial.contract_signature,
                "result_file": (
                    str(candidate.artifact.result_path)
                    if candidate.artifact is not None
                    else str(paths["result"])
                ),
                "invalid_existing_reason": candidate.invalid_reason,
                "runtime": dict(runtime),
                "test_evaluated": False,
            }
            if candidate.artifact is not None:
                record.update(
                    {
                        "recall@10": candidate.artifact.recall_at_10,
                        "ndcg@10": candidate.artifact.ndcg_at_10,
                        "delta_vs_frozen_500_winner": {
                            "recall@10": (
                                candidate.artifact.recall_at_10
                                - state.epoch_extension.parent_500.recall_at_10
                            ),
                            "ndcg@10": (
                                candidate.artifact.ndcg_at_10
                                - state.epoch_extension.parent_500.ndcg_at_10
                            ),
                        },
                        "artifact_signature": candidate.artifact.artifact_signature,
                    }
                )
            extension_candidates.append(record)
        extension_record = {
            "state": (
                "complete" if state.epoch_extension.complete else "incomplete"
            ),
            "purpose": "post-selection epoch-budget sensitivity only",
            "excluded_from_500_epoch_model_selection": True,
            "frozen_500_epoch_reference": _artifact_record(
                state.epoch_extension.parent_500
            ),
            "candidates": extension_candidates,
            "winner": None,
            "note": (
                "750/1000-epoch metrics are not comparable as equal-budget "
                "hyperparameter candidates and never replace the 500-epoch winner."
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "amazon-toy-sl8lhgcn-staged-tuning-summary",
        "state": "complete" if state.complete else "incomplete",
        "dataset": layer_grid.DATASET,
        "protocol": protocol,
        "prerequisite": {
            "summary_file": str(state.prerequisite.summary_path),
            "summary_sha256": state.prerequisite.summary_sha256,
            "winner": _artifact_record(prerequisite_artifact(state.prerequisite)),
        },
        "strategy": {
            "type": "sequential-stage-winner-carry-forward",
            "selection": (
                "max Recall@10, then max NDCG@10, then unchanged parent on an "
                "exact tie, then declared candidate order"
            ),
            "stage_definitions": stage_definitions(),
            "cartesian_product_avoided": True,
            "new_trials_expected": EXPECTED_NEW_TRIALS,
            "new_trials_completed": completed_new_trials,
            "post_selection_epoch_extension_budgets": list(
                EPOCH_EXTENSION_BUDGETS
            ),
            "post_selection_extension_trials_expected": len(
                EPOCH_EXTENSION_BUDGETS
            ),
        },
        "parameter_status": PARAMETER_STATUS,
        "single_physical_gpu": PHYSICAL_GPU,
        "child_cuda_visible_devices": PHYSICAL_GPU,
        "child_torch_device_after_mask": "cuda:0",
        "strict_serial": True,
        "data_audit": data_audit,
        "stages": stages,
        "selection_state": "complete" if state.selection_complete else "incomplete",
        "current_stage": (
            {
                "index": state.current_stage.stage.index,
                "key": state.current_stage.stage.key,
            }
            if state.current_stage is not None
            else None
        ),
        "winner": (
            _artifact_record(state.final_artifact)
            if state.selection_complete
            else None
        ),
        "provisional_parent": _artifact_record(state.final_artifact),
        "epoch_extension": extension_record,
        "runtime": {
            "known_trial_seconds": sum(known_durations),
            "known_trial_hours": sum(known_durations) / 3600.0,
            "trials_with_known_duration": len(known_durations),
        },
        "failures": _failure_records(output_root),
        "split_fingerprints": state.prerequisite.split_fingerprints,
        "manifold_acceptance": (
            "Every new result must pass initial, every-layer post-retraction, "
            "final, relative-matrix, and approximate-log checks with zero active "
            "singular fallbacks."
        ),
        "test_evaluated": False,
    }


def write_summary(
    path: Path,
    state: TuningState,
    *,
    protocol: Mapping[str, Any],
    output_root: Path,
    data_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = summary_payload(
        state,
        protocol=protocol,
        output_root=output_root,
        data_audit=data_audit,
    )
    layer_grid._atomic_json(path, payload)
    return payload


@contextlib.contextmanager
def blocking_gpu_lock(path: Path, gpu_id: str) -> Iterable[int]:
    """Share the layer-grid lock and wait instead of racing its final write."""

    if gpu_id != PHYSICAL_GPU:
        raise ValueError(f"this controlled runner permits only physical GPU {PHYSICAL_GPU}")
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        lock.seek(0)
        lock.truncate()
        lock.write(
            f"pid={os.getpid()} gpu={gpu_id} staged=true "
            f"acquired_at={layer_grid._utc_now()}\n"
        )
        lock.flush()
        try:
            yield lock.fileno()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _verify_prerequisite_unchanged(prerequisite: Prerequisite, repo: Path) -> None:
    current = load_complete_prerequisite(prerequisite.summary_path, repo)
    if current.summary_sha256 != prerequisite.summary_sha256 or (
        current.artifact_signature != prerequisite.artifact_signature
    ):
        raise RuntimeError("layer-by-batch prerequisite changed after tuning began")


def _pending_jobs(state: TuningState) -> list[CandidateState]:
    if state.current_stage is not None:
        return [
            candidate
            for candidate in state.current_stage.candidates
            if candidate.artifact is None
        ]
    if state.epoch_extension is not None and not state.epoch_extension.complete:
        return [
            candidate
            for candidate in state.epoch_extension.candidates
            if candidate.artifact is None
        ]
    return []


def dry_run_plan(
    args: argparse.Namespace,
    *,
    prerequisite: Prerequisite,
    protocol: Mapping[str, Any],
    data_audit: Mapping[str, Any],
) -> dict[str, Any]:
    state = build_state(
        repo=args.repo,
        output_root=args.output_root,
        prerequisite=prerequisite,
        protocol=protocol,
    )
    jobs = []
    for candidate in _pending_jobs(state):
        paths = result_paths(args.output_root, candidate.trial)
        jobs.append(
            {
                "stage": candidate.trial.stage_key,
                "candidate": candidate.trial.candidate_label,
                "parameters": asdict(candidate.trial.parameters),
                "status": "run",
                "invalid_existing_reason": candidate.invalid_reason,
                "command": trial_command(
                    args, candidate.trial, paths["raw"], paths["checkpoint_dir"]
                ),
            }
        )
    payload = summary_payload(
        state,
        protocol=protocol,
        output_root=args.output_root,
        data_audit=data_audit,
    )
    payload.update(
        {
            "dry_run": True,
            "lock_file": str(args.lock_file),
            "jobs_for_current_stage": jobs,
            "future_stages_are_adaptive": True,
        }
    )
    return payload


def execute(
    args: argparse.Namespace,
    *,
    prerequisite: Prerequisite,
    protocol: Mapping[str, Any],
    data_audit: Mapping[str, Any],
) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = PHYSICAL_GPU
    summary_path = args.output_root / "summary.json"
    attempted: set[str] = set()
    new_attempts = 0

    with blocking_gpu_lock(args.lock_file, PHYSICAL_GPU) as lock_fd:
        while True:
            _verify_prerequisite_unchanged(prerequisite, args.repo)
            state = build_state(
                repo=args.repo,
                output_root=args.output_root,
                prerequisite=prerequisite,
                protocol=protocol,
            )
            write_summary(
                summary_path,
                state,
                protocol=protocol,
                output_root=args.output_root,
                data_audit=data_audit,
            )
            if state.complete:
                print("DONE all six staged sweeps", flush=True)
                return
            pending = [
                candidate
                for candidate in _pending_jobs(state)
                if candidate.trial.contract_signature not in attempted
            ]
            if not pending:
                print(
                    "STOP current selection/extension phase remains incomplete; "
                    "resume to retry failures",
                    file=sys.stderr,
                )
                return
            for candidate in pending:
                if args.max_new_trials is not None and new_attempts >= args.max_new_trials:
                    return
                trial = candidate.trial
                attempted.add(trial.contract_signature)
                new_attempts += 1
                paths = result_paths(args.output_root, trial)
                paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
                paths["raw"].parent.mkdir(parents=True, exist_ok=True)
                # A successful child must produce the current attempt's raw
                # artifact; never allow a stale pre-crash file to be adopted.
                if paths["raw"].is_file():
                    paths["raw"].unlink()
                started_at = layer_grid._utc_now()
                started_clock = time.monotonic()
                command = trial_command(
                    args, trial, paths["raw"], paths["checkpoint_dir"]
                )
                print(
                    f"START S{trial.stage_index:02d} {trial.candidate_label}: "
                    f"{asdict(trial.parameters)}",
                    flush=True,
                )
                try:
                    layer_grid._run_and_tee(
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
                        prerequisite=prerequisite,
                        protocol=protocol,
                        runtime={
                            "started_at": started_at,
                            "finished_at": layer_grid._utc_now(),
                            "duration_seconds": duration,
                            "source": "runner-measured",
                        },
                    )
                    if paths["failure"].is_file():
                        paths["failure"].unlink()
                    print(f"DONE {trial.candidate_label}: {duration:.1f}s", flush=True)
                except (subprocess.CalledProcessError, OSError, ValueError) as error:
                    failure = {
                        "stage_index": trial.stage_index,
                        "stage_key": trial.stage_key,
                        "candidate_label": trial.candidate_label,
                        "contract_signature": trial.contract_signature,
                        "parameters": asdict(trial.parameters),
                        "status": "failed",
                        "started_at": started_at,
                        "failed_at": layer_grid._utc_now(),
                        "duration_seconds": time.monotonic() - started_clock,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "invalid_existing_reason": candidate.invalid_reason,
                        "log_file": str(paths["log"].expanduser().resolve()),
                        "test_evaluated": False,
                    }
                    layer_grid._atomic_json(paths["failure"], failure)
                    print(f"FAILED {trial.candidate_label}: {error}", file=sys.stderr)
                    if not args.continue_on_error:
                        raise
                refreshed = build_state(
                    repo=args.repo,
                    output_root=args.output_root,
                    prerequisite=prerequisite,
                    protocol=protocol,
                )
                write_summary(
                    summary_path,
                    refreshed,
                    protocol=protocol,
                    output_root=args.output_root,
                    data_audit=data_audit,
                )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--data-root", type=Path, default=repo / "dataset")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layer-batch-summary", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-id", default=PHYSICAL_GPU)
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--max-new-trials", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--deep-data-audit", action="store_true")
    parser.add_argument(
        "--skip-data-audit",
        action="store_true",
        help="only valid with --dry-run on a planning machine without Toy data",
    )
    parser.add_argument("--prerequisite-poll-seconds", type=float, default=30.0)
    parser.add_argument("--prerequisite-timeout-seconds", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_trials is not None and args.max_new_trials < 0:
        parser.error("--max-new-trials must be non-negative")
    if args.prerequisite_poll_seconds <= 0:
        parser.error("--prerequisite-poll-seconds must be positive")
    if (
        args.prerequisite_timeout_seconds is not None
        and args.prerequisite_timeout_seconds < 0
    ):
        parser.error("--prerequisite-timeout-seconds must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.repo = args.repo.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.layer_batch_summary = args.layer_batch_summary.expanduser().resolve()
    args.gpu_id = layer_grid._gpu_token(args.gpu_id)
    if args.gpu_id != PHYSICAL_GPU:
        raise ValueError(
            f"this experiment is authorised only on physical GPU {PHYSICAL_GPU}; "
            f"got {args.gpu_id}"
        )
    args.lock_file = (
        args.lock_file.expanduser().resolve()
        if args.lock_file is not None
        else layer_grid.default_lock_path(PHYSICAL_GPU)
    )
    if args.skip_data_audit and not args.dry_run:
        raise ValueError("--skip-data-audit is permitted only with --dry-run")
    protocol = layer_grid.validate_protocol(args.repo)
    if args.dry_run:
        # Planning is intentionally non-blocking; it still refuses to plan
        # adaptive trials from a provisional layer/batch winner.
        prerequisite = load_complete_prerequisite(
            args.layer_batch_summary, args.repo
        )
    else:
        prerequisite = wait_for_prerequisite(
            args.layer_batch_summary,
            args.repo,
            poll_seconds=args.prerequisite_poll_seconds,
            timeout_seconds=args.prerequisite_timeout_seconds,
        )
    data_audit = layer_grid._audit_data(args)
    if args.dry_run:
        print(
            json.dumps(
                dry_run_plan(
                    args,
                    prerequisite=prerequisite,
                    protocol=protocol,
                    data_audit=data_audit,
                ),
                indent=2,
            )
        )
        return 0
    execute(
        args,
        prerequisite=prerequisite,
        protocol=protocol,
        data_audit=data_audit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
