#!/usr/bin/env python3
"""Run an explicit SL8-LHGCN validation queue on exactly one GPU.

The queue is intentionally a JSON file rather than a Cartesian-product
generator.  This keeps temporary-server shards disjoint and auditable.  Every
child is validation-only, uses the accelerated mask-aware PF4096 evaluator,
and retains the guarded sqrt-extended scorer for training.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL = "SL8LHGCN"
DATASET = "Amazon_cd"
BASE_CONFIG = "baseline_config_fixed/SL8LHGCN_cd.yaml"
METHOD_CONFIG = "baseline_config_fixed/SL8LHGCN_liebn_rowmean_4070ti.yaml"
DATA_SHA256 = "7061471c288df93ba65bfede355aeb013e10dbdfc249db8f20a02bbf8ae031c4"
EPOCHS = 500
EVAL_STEP = 10
STOPPING_STEP = 2
PREFILTER_CANDIDATES = 4096


@dataclass(frozen=True)
class Trial:
    trial_id: str
    layers: int
    batch_size: int
    learning_rate: float
    loss_margin: float
    coord_clip: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Trial":
        trial = cls(
            trial_id=str(raw["id"]),
            layers=int(raw["layers"]),
            batch_size=int(raw["batch_size"]),
            learning_rate=float(raw["learning_rate"]),
            loss_margin=float(raw["loss_margin"]),
            coord_clip=float(raw["coord_clip"]),
        )
        if not trial.trial_id or any(char.isspace() for char in trial.trial_id):
            raise ValueError(f"invalid trial id: {trial.trial_id!r}")
        if trial.layers < 0 or trial.batch_size < 1:
            raise ValueError(f"invalid trial: {trial}")
        if trial.learning_rate <= 0 or trial.loss_margin <= 0:
            raise ValueError(f"invalid trial: {trial}")
        if trial.coord_clip < 0:
            raise ValueError(f"invalid trial: {trial}")
        return trial


def _load_queue(path: Path) -> list[Trial]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("queue JSON must be a non-empty list")
    trials = [Trial.from_mapping(item) for item in raw]
    ids = [trial.trial_id for trial in trials]
    signatures = [
        (
            trial.layers,
            trial.batch_size,
            trial.learning_rate,
            trial.loss_margin,
            trial.coord_clip,
        )
        for trial in trials
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("queue contains duplicate ids")
    if len(signatures) != len(set(signatures)):
        raise ValueError("queue contains duplicate parameter combinations")
    return trials


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _result_is_complete(path: Path, trial: Trial) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    expected = {
        "model": MODEL,
        "dataset": DATASET,
        "epochs": EPOCHS,
        "eval_step": EVAL_STEP,
        "stopping_step": STOPPING_STEP,
        "gcn_layers": trial.layers,
        "n_layers": trial.layers,
        "train_batch_size": trial.batch_size,
        "log_domain_sqrt_steps": 1,
        "eval_log_domain_sqrt_steps": 0,
        "log_domain_sqrt_iterations": 12,
        "log_domain_guard_revision": "db_residual_spectral_tail_v1",
        "eval_prefilter": "frobenius",
        "eval_prefilter_candidates": PREFILTER_CANDIDATES,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return False
    for key, value in (
        ("learning_rate", trial.learning_rate),
        ("loss_margin", trial.loss_margin),
        ("coord_clip", trial.coord_clip),
        ("log_domain_sqrt_residual_tolerance", 1e-3),
        ("log_domain_tail_tolerance", 1e-3),
    ):
        try:
            if abs(float(payload[key]) - value) > 1e-12:
                return False
        except (KeyError, TypeError, ValueError):
            return False
    if payload.get("test_result") is not None or not payload.get("best_valid_result"):
        return False
    checkpoint = payload.get("checkpoint_file")
    return isinstance(checkpoint, str) and Path(checkpoint).is_file()


def _command(
    args: argparse.Namespace,
    trial: Trial,
    result_path: Path,
    checkpoint_dir: Path,
) -> list[str]:
    return [
        str(args.python),
        "-u",
        str(args.repo / "run_recbole_gnn.py"),
        "--model",
        MODEL,
        "--dataset",
        DATASET,
        "--config-files",
        f"{BASE_CONFIG} {METHOD_CONFIG}",
        "--validation-only",
        "--result-file",
        str(result_path),
        f"--checkpoint_dir={checkpoint_dir}",
        f"--gcn_layers={trial.layers}",
        f"--n_layers={trial.layers}",
        f"--epochs={EPOCHS}",
        f"--eval_step={EVAL_STEP}",
        f"--stopping_step={STOPPING_STEP}",
        f"--train_batch_size={trial.batch_size}",
        f"--learning_rate={trial.learning_rate}",
        f"--loss_margin={trial.loss_margin}",
        f"--coord_clip={trial.coord_clip}",
        "--log_domain_sqrt_steps=1",
        "--eval_log_domain_sqrt_steps=0",
        "--log_domain_sqrt_iterations=12",
        "--log_domain_sqrt_residual_tolerance=0.001",
        "--log_domain_tail_tolerance=0.001",
        "--log_domain_guard_revision=db_residual_spectral_tail_v1",
        f"--gpu_id={args.gpu_id}",
        f"--data_path={args.data_root}",
        "--full_sort_user_batch_size=64",
        "--eval_user_chunk_size=64",
        "--eval_item_chunk_size=1024",
        "--sl_score_mode=group_log",
        "--eval_prefilter=frobenius",
        f"--eval_prefilter_candidates={PREFILTER_CANDIDATES}",
        "--show_progress=False",
        f"--run_marker={trial.trial_id}",
    ]


def _parse_not_after(value: str | None) -> dt.datetime | None:
    if value is None:
        return None
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--not-after must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    """Return an absolute path while preserving virtualenv interpreter links."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--data-root", type=Path, default=repo / "dataset")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--not-after",
        help="UTC/offset timestamp after which no new trial may start",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.repo = args.repo.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.queue_file = args.queue_file.expanduser().resolve()
    # Do not call Path.resolve() here.  Virtualenv ``python`` executables are
    # commonly symlinks; resolving one launches the base interpreter directly
    # and silently drops the virtualenv's site-packages.
    args.python = _absolute_without_resolving_symlinks(args.python)
    not_after = _parse_not_after(args.not_after)
    trials = _load_queue(args.queue_file)

    data_file = args.data_root / DATASET / f"{DATASET}.inter"
    if _sha256(data_file) != DATA_SHA256:
        raise RuntimeError(f"Amazon-CD fingerprint mismatch: {data_file}")
    if not args.python.is_file():
        raise FileNotFoundError(args.python)

    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "queue_state.json"
    lock_path = args.output_root / "queue.lock"
    state: dict[str, Any] = {
        "started_at": _now(),
        "not_after": not_after.isoformat() if not_after else None,
        "validation_only": True,
        "gpu_id": args.gpu_id,
        "queue_file": str(args.queue_file),
        "trials": [],
    }
    _atomic_json(state_path, state)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"queue already owns GPU {args.gpu_id}") from error

        for trial in trials:
            trial_dir = args.output_root / trial.trial_id
            result_path = trial_dir / "result.json"
            checkpoint_dir = trial_dir / "checkpoints"
            log_path = trial_dir / "train.log"
            if _result_is_complete(result_path, trial):
                state["trials"].append(
                    {"id": trial.trial_id, "status": "resume", "at": _now()}
                )
                _atomic_json(state_path, state)
                continue
            if not_after and dt.datetime.now(dt.timezone.utc) >= not_after:
                state["stopped_before"] = trial.trial_id
                state["stop_reason"] = "not_after"
                break

            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            command = _command(args, trial, result_path, checkpoint_dir)
            record: dict[str, Any] = {
                "id": trial.trial_id,
                "parameters": trial.__dict__,
                "started_at": _now(),
                "status": "running",
                "log": str(log_path),
            }
            state["trials"].append(record)
            _atomic_json(state_path, state)
            with log_path.open("a", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=args.repo,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record["finished_at"] = _now()
            record["returncode"] = completed.returncode
            record["status"] = (
                "complete"
                if completed.returncode == 0 and _result_is_complete(result_path, trial)
                else "failed"
            )
            _atomic_json(state_path, state)

    state["finished_at"] = _now()
    _atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
