#!/usr/bin/env python3
"""Safe, resumable sampled-validation tuning for geometry-only SL(4).

This script deliberately has no final-test mode.  Every spawned RecBole run
uses the fixed Amazon-CD split and ``uni100`` validation protocol, saves its
best checkpoint, and passes ``--validation-only``.  Exact full-ranking test
evaluation is a separate, explicit operation after hyperparameter selection.

The search is staged so a later stage inherits the setting selected by the
previous one instead of silently starting from the original defaults.  Supply
that setting either with ``--resume-from`` (a result or stage-summary JSON) or
with the relevant scalar command-line options.
"""

from __future__ import annotations

import argparse
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


STAGE_VALUES: dict[str, tuple[float, ...]] = {
    "lr": (5e-4, 1e-3, 3e-3),
    "coord_clip": (0.5, 0.75, 1.0),
    "init_std": (0.008, 0.012, 0.02),
    "reg_weight": (0.0, 1e-6, 1e-5),
}

CONFIG_NAMES = (
    "SLRecGraph_cd.yaml",
    "SLRecGraph_geometry_sl4.yaml",
    "SLRecGraph_tune_sampled.yaml",
)

# These are the values established by the geometry overlay.  Only stage 1 may
# use them implicitly.  Later stages require an explicit continuation source.
GEOMETRY_DEFAULTS = {
    "learning_rate": 3e-3,
    "coord_clip": 0.75,
    "init_std": 0.012,
    "reg_weight": 0.0,
}

PROTOCOL = {
    "model": "SLRecGraph",
    "dataset": "Amazon_cd",
    "seed": 2024,
    "n_layers": 0,
    "matrix_dim": 4,
    "num_factors": 1,
    "eval_args": {
        "split": {"RS": [0.8, 0.1, 0.1]},
        "group_by": "user",
        "order": "RO",
        "mode": "uni100",
    },
}


@dataclass(frozen=True)
class Parameters:
    learning_rate: float
    coord_clip: float
    init_std: float
    reg_weight: float

    def varied(self, stage: str, value: float) -> "Parameters":
        field = "learning_rate" if stage == "lr" else stage
        return replace(self, **{field: float(value)})

    def recbole_args(self) -> list[str]:
        return [
            f"--learning_rate={self.learning_rate:.12g}",
            f"--coord_clip={self.coord_clip:.12g}",
            f"--init_std={self.init_std:.12g}",
            f"--reg_weight={self.reg_weight:.12g}",
        ]


@dataclass(frozen=True)
class Trial:
    stage: str
    parameters: Parameters

    @property
    def name(self) -> str:
        p = self.parameters
        return (
            f"{self.stage}__lr-{_float_token(p.learning_rate)}"
            f"__clip-{_float_token(p.coord_clip)}"
            f"__init-{_float_token(p.init_std)}"
            f"__reg-{_float_token(p.reg_weight)}"
        )


def _float_token(value: float) -> str:
    return (
        f"{value:.12g}"
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Tune geometry-only SL(4) using validation only."
    )
    parser.add_argument("--stage", choices=tuple(STAGE_VALUES), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--eval-step",
        type=int,
        default=5,
        help="validate every N epochs (default: 5 from the sampled overlay)",
    )
    parser.add_argument("--stopping-step", type=int, default=12)
    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        help="override the grid for the selected stage",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--coord-clip", type=float)
    parser.add_argument("--init-std", type=float)
    parser.add_argument("--reg-weight", type=float)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="prior result or stage-summary JSON whose best parameters are inherited",
    )
    parser.add_argument(
        "--existing-base-result",
        type=Path,
        help=(
            "map an existing geometry-default (lr=3e-3) validation JSON to "
            "the matching stage-1 trial"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan without creating files or training",
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
        "model": PROTOCOL["model"],
        "dataset": PROTOCOL["dataset"],
        "seed": PROTOCOL["seed"],
        "n_layers": PROTOCOL["n_layers"],
        "matrix_dim": PROTOCOL["matrix_dim"],
        "num_factors": PROTOCOL["num_factors"],
        "eval_args": PROTOCOL["eval_args"],
    }
    actual = {key: merged.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            "geometry-only sampled-validation protocol changed; refusing to tune: "
            f"expected={expected}, actual={actual}"
        )


def _parameters_from_mapping(payload: Mapping[str, Any]) -> Parameters | None:
    aliases = (
        payload.get("parameters"),
        payload.get("tuning_parameters"),
        payload.get("tuning", {}).get("parameters")
        if isinstance(payload.get("tuning"), Mapping)
        else None,
    )
    for candidate in aliases:
        if not isinstance(candidate, Mapping):
            continue
        try:
            return Parameters(
                learning_rate=float(candidate["learning_rate"]),
                coord_clip=float(candidate["coord_clip"]),
                init_std=float(candidate["init_std"]),
                reg_weight=float(candidate["reg_weight"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return None


def load_continuation(path: Path) -> Parameters:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"continuation JSON is not an object: {resolved}")

    # A stage summary stores the selected candidate under ``best``; a trial
    # result stores its metadata at the top level.
    candidates: Iterable[Mapping[str, Any]] = (
        item
        for item in (payload.get("best"), payload)
        if isinstance(item, Mapping)
    )
    for candidate in candidates:
        parameters = _parameters_from_mapping(candidate)
        if parameters is not None:
            return parameters
    raise ValueError(f"no complete tuning parameters in continuation JSON: {resolved}")


def resolve_base_parameters(args: argparse.Namespace) -> Parameters:
    if args.resume_from:
        base = load_continuation(args.resume_from)
    else:
        base = Parameters(**GEOMETRY_DEFAULTS)

    explicit = {
        "learning_rate": args.learning_rate,
        "coord_clip": args.coord_clip,
        "init_std": args.init_std,
        "reg_weight": args.reg_weight,
    }
    base = replace(
        base,
        **{key: float(value) for key, value in explicit.items() if value is not None},
    )

    if args.stage != "lr" and args.resume_from is None:
        prerequisites = {
            "coord_clip": ("learning_rate",),
            "init_std": ("learning_rate", "coord_clip"),
            "reg_weight": ("learning_rate", "coord_clip", "init_std"),
        }[args.stage]
        missing = [name for name in prerequisites if explicit[name] is None]
        if missing:
            flags = ", ".join("--" + name.replace("_", "-") for name in missing)
            raise ValueError(
                f"stage {args.stage!r} must continue the preceding selection; "
                f"provide --resume-from or explicitly set {flags}"
            )
    return base


def build_trials(
    stage: str, base: Parameters, values: Sequence[float] | None = None
) -> tuple[Trial, ...]:
    selected_values = tuple(STAGE_VALUES[stage] if values is None else values)
    if not selected_values:
        raise ValueError("the stage grid must contain at least one value")
    if any(not math.isfinite(value) for value in selected_values):
        raise ValueError("all stage values must be finite")
    trials = tuple(Trial(stage, base.varied(stage, value)) for value in selected_values)
    if len({trial.name for trial in trials}) != len(trials):
        raise ValueError("stage grid contains duplicate values")
    return trials


def trial_command(
    args: argparse.Namespace, trial: Trial, result_path: Path, checkpoint_dir: Path
) -> list[str]:
    configs = " ".join(str(path) for path in config_paths(args.repo))
    return [
        args.python,
        "-u",
        str(args.repo / "run_recbole_gnn.py"),
        "--model",
        PROTOCOL["model"],
        "--dataset",
        PROTOCOL["dataset"],
        "--config-files",
        configs,
        "--validation-only",
        "--result-file",
        str(result_path),
        f"--checkpoint_dir={checkpoint_dir}",
        # CUDA_VISIBLE_DEVICES selects the physical card.  Inside the child it
        # is renumbered to logical device zero.
        "--gpu_id=0",
        "--use_gpu=True",
        "--show_progress=False",
        f"--epochs={args.epochs}",
        f"--eval_step={args.eval_step}",
        f"--stopping_step={args.stopping_step}",
        # Repeat invariants on the CLI so a later, unrelated config edit cannot
        # accidentally turn a tuning trial into graph propagation.
        f"--seed={PROTOCOL['seed']}",
        f"--n_layers={PROTOCOL['n_layers']}",
        *trial.parameters.recbole_args(),
    ]


def run_and_tee(command: list[str], log_path: Path, cwd: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND=" + shlex.join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
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
        # Deliberately uncaught by main: one failed trial stops the entire
        # stage, leaving completed JSON/checkpoints available for the next run.
        raise subprocess.CalledProcessError(return_code, command)


def load_complete_result(
    path: Path,
    *,
    expected_parameters: Parameters | None,
    allow_legacy_metadata: bool = False,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result JSON is not an object: {resolved}")
    if payload.get("model") != PROTOCOL["model"]:
        raise ValueError(f"unexpected model in result: {resolved}")
    if payload.get("dataset") not in (None, PROTOCOL["dataset"]):
        raise ValueError(f"unexpected dataset in result: {resolved}")
    if int(payload.get("seed", -1)) != PROTOCOL["seed"]:
        raise ValueError(f"unexpected seed in result: {resolved}")
    if payload.get("test_result") is not None:
        raise RuntimeError(f"tuning result touched the test split: {resolved}")
    if payload.get("best_valid_score") is None or not isinstance(
        payload.get("best_valid_result"), Mapping
    ):
        raise ValueError(f"result has no completed validation metrics: {resolved}")

    checkpoint = payload.get("checkpoint_file")
    if not checkpoint or not Path(checkpoint).expanduser().is_file():
        raise ValueError(f"result has no existing saved checkpoint: {resolved}")

    actual_parameters = _parameters_from_mapping(payload)
    if expected_parameters is not None:
        if actual_parameters is None and not allow_legacy_metadata:
            raise ValueError(f"result has no tuning parameter metadata: {resolved}")
        if actual_parameters is None and allow_legacy_metadata:
            config_files = payload.get("config_files")
            config_names = (
                [Path(item).name for item in config_files]
                if isinstance(config_files, list)
                else None
            )
            if config_names != list(CONFIG_NAMES):
                raise ValueError(
                    "legacy base result was not produced with the three fixed "
                    f"geometry-only overlays: {resolved}"
                )
        if actual_parameters is not None and actual_parameters != expected_parameters:
            raise ValueError(
                f"result parameters do not match trial: {resolved}; "
                f"expected={expected_parameters}, actual={actual_parameters}"
            )
    return payload


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def add_trial_metadata(path: Path, trial: Trial) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tuning"] = {
        "stage": trial.stage,
        "trial_name": trial.name,
        "parameters": asdict(trial.parameters),
        "protocol": PROTOCOL,
        "test_evaluated": False,
    }
    _atomic_json_write(path, payload)
    return load_complete_result(path, expected_parameters=trial.parameters)


def _legacy_base_path(args: argparse.Namespace) -> Path | None:
    if args.existing_base_result:
        return args.existing_base_result.expanduser().resolve()
    conventional = (
        args.output_root / "results" / "sl4-nognn-start.json",
        args.output_root / "sl4-nognn-start.json",
    )
    return next((path.resolve() for path in conventional if path.is_file()), None)


def _same_parameters(left: Parameters, right: Parameters) -> bool:
    return all(
        math.isclose(getattr(left, field), getattr(right, field), rel_tol=0, abs_tol=1e-15)
        for field in asdict(left)
    )


def write_summary(
    path: Path,
    *,
    stage: str,
    base: Parameters,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ranked = sorted(candidates, key=lambda item: item["best_valid_score"], reverse=True)
    summary = {
        "selection_metric": "Recall@10 on uni100 validation only",
        "test_evaluated": False,
        "stage": stage,
        "base_parameters": asdict(base),
        "candidate_count": len(ranked),
        "best": ranked[0] if ranked else None,
        "ranking": ranked,
    }
    _atomic_json_write(path, summary)
    return summary


def _candidate(trial: Trial, source: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": trial.name,
        "source": str(source.resolve()),
        "checkpoint_file": result["checkpoint_file"],
        "parameters": asdict(trial.parameters),
        "best_valid_score": float(result["best_valid_score"]),
        "best_valid_result": result["best_valid_result"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.eval_step <= 0:
        raise ValueError("--eval-step must be positive")
    if args.stopping_step < 0:
        raise ValueError("--stopping-step must be non-negative")
    args.repo = args.repo.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    validate_protocol(args.repo)
    base = resolve_base_parameters(args)
    trials = build_trials(args.stage, base, args.values)

    legacy_path = _legacy_base_path(args) if args.stage == "lr" else None
    legacy_trial = next(
        (
            trial
            for trial in trials
            if _same_parameters(trial.parameters, Parameters(**GEOMETRY_DEFAULTS))
        ),
        None,
    )
    if legacy_path is not None and legacy_trial is None:
        raise ValueError(
            "an existing base result was supplied, but the stage grid has no "
            "geometry-default lr=3e-3 trial"
        )

    stage_root = args.output_root / "sl4-geometry-tuning" / args.stage
    results_dir = stage_root / "results"
    logs_dir = stage_root / "logs"
    checkpoints_dir = stage_root / "checkpoints"
    summary_path = stage_root / "summary.json"

    plan: list[dict[str, Any]] = []
    for trial in trials:
        result_path = results_dir / f"{trial.name}.json"
        source = legacy_path if trial == legacy_trial and legacy_path else result_path
        status = "skip" if source.is_file() else "run"
        command = trial_command(
            args,
            trial,
            result_path,
            checkpoints_dir / trial.name,
        )
        plan.append(
            {
                "name": trial.name,
                "status": status,
                "source": str(source),
                "parameters": asdict(trial.parameters),
                "command": command,
            }
        )

    if args.dry_run:
        # Validate any result that would be skipped, but do not create output
        # directories or mutate result JSON during a dry run.
        for item, trial in zip(plan, trials):
            if item["status"] == "skip":
                is_legacy = legacy_path is not None and Path(item["source"]) == legacy_path
                load_complete_result(
                    Path(item["source"]),
                    expected_parameters=trial.parameters,
                    allow_legacy_metadata=is_legacy,
                )
        print(json.dumps({"stage": args.stage, "dry_run": True, "trials": plan}, indent=2))
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    candidates: list[dict[str, Any]] = []
    for index, (item, trial) in enumerate(zip(plan, trials), 1):
        result_path = Path(item["source"])
        is_legacy = legacy_path is not None and result_path == legacy_path
        if item["status"] == "skip":
            print(f"[{index}/{len(trials)}] skip complete {trial.name}")
            result = load_complete_result(
                result_path,
                expected_parameters=trial.parameters,
                allow_legacy_metadata=is_legacy,
            )
        else:
            print(f"[{index}/{len(trials)}] start {trial.name}")
            run_and_tee(
                item["command"],
                logs_dir / f"{trial.name}.log",
                args.repo,
                env,
            )
            result_path = results_dir / f"{trial.name}.json"
            result = add_trial_metadata(result_path, trial)

        candidates.append(_candidate(trial, result_path, result))
        summary = write_summary(
            summary_path,
            stage=args.stage,
            base=base,
            candidates=candidates,
        )
        print(
            f"current best={summary['best']['name']} "
            f"valid Recall@10={summary['best']['best_valid_score']:.6f}"
        )

    print(f"SUMMARY_JSON={summary_path}")
    print("Stage complete; the held-out test split was not evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
