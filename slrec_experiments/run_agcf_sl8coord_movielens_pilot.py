#!/usr/bin/env python3
"""Run one strict, validation-only AGCF-SL8Coord MovieLens-1M pilot.

The command has no held-out-test mode.  It pins the exact AGCF MovieLens
source/filter/split contract, runs one process on one physical GPU, saves the
checkpoint and combined log, and records a hash-rich manifest.  Re-running a
completed exact pilot verifies and resumes it rather than training twice.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

# Direct ``python slrec_experiments/...py`` execution otherwise exposes only
# the script directory, while checkpoint validation also needs project-local
# RecBole classes importable by their package names.
_DEFAULT_REPO = Path(__file__).resolve().parents[1]
if str(_DEFAULT_REPO) not in sys.path:
    sys.path.insert(0, str(_DEFAULT_REPO))

from slrec_experiments.run_agcf_movielens_sweep import (
    DATASET,
    MOVIELENS_FILTERED,
    SEED,
    audit_movielens_source,
)


MODEL = "AGCFSL8Coord"
EPOCHS = 500
EVAL_STEP = 1
STOPPING_STEP = 30
SCHEMA_VERSION = 1
CONFIG_NAMES = (
    "AGCF_movielens_protocol.yaml",
    "AGCF_cd.yaml",
    "AGCFSL8Coord_cd.yaml",
    "AGCFSL8Coord_movielens_pilot.yaml",
)
PILOT_PARAMETERS: dict[str, Any] = {
    "embedding_size": 63,
    "matrix_dim": 8,
    "num_factors": 1,
    "metric_rank": 16,
    "channel_rank": 63,
    "train_batch_size": 2048,
    "learning_rate": 1e-3,
    "evolution_time": 1.0,
    "metric_epsilon": 1e-3,
    "structural_delta": 1e-3,
    "margin": 0.1,
    "pairwise_loss": "hinge",
    "loss_margin": 0.1,
    "output_steps": 1,
    "integration_steps": 1,
    "potential_strength": 0.1,
    "damping": 0.01,
    "weight_decay": 0.0,
    "reg_weight": 0.0,
    "sl_scale": 1.0,
    "coord_clip": 1.0,
    "log_terms": 12,
    "log_jitter": 0.0,
    "schatten_p": 2,
    "symmetric_distance": False,
    "sl_membership_tolerance": 1e-4,
    "checkpoint_dynamics": False,
    "dynamics_chunk_size": 4096,
    "eval_user_chunk_size": 64,
    "eval_item_chunk_size": 512,
}
PROTOCOL_KEYS = {
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
    "epochs",
    "eval_step",
    "stopping_step",
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return payload


def config_paths(repo: Path) -> tuple[Path, ...]:
    return tuple(repo / "baseline_config_fixed" / name for name in CONFIG_NAMES)


def validate_protocol(
    repo: Path, source_audit: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    paths = config_paths(repo)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing required config: {path}")
    protocol, agcf, sl8, pilot = (_yaml(path) for path in paths)

    for path, overlay in zip(paths[1:], (agcf, sl8, pilot)):
        overlap = PROTOCOL_KEYS.intersection(overlay)
        if overlap:
            raise RuntimeError(
                f"model overlay {path.name} overrides protocol keys: {sorted(overlap)}"
            )

    expected_protocol = {
        "dataset": DATASET,
        "seed": SEED,
        "reproducibility": True,
        "val_interval": {"rating": "[3,inf)"},
        "user_inter_num_interval": "[5,inf)",
        "item_inter_num_interval": "[5,inf)",
        "metrics": ["Recall", "NDCG"],
        "topk": [10, 20],
        "valid_metric": "Recall@10",
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        },
        "epochs": EPOCHS,
        "eval_step": EVAL_STEP,
        "stopping_step": STOPPING_STEP,
    }
    differences = {
        key: {"expected": value, "actual": protocol.get(key)}
        for key, value in expected_protocol.items()
        if protocol.get(key) != value
    }
    if differences:
        raise RuntimeError(f"MovieLens exact protocol changed: {differences}")

    effective = dict(agcf)
    effective.update(sl8)
    effective.update(pilot)
    model_differences = {
        key: {"expected": value, "actual": effective.get(key)}
        for key, value in PILOT_PARAMETERS.items()
        if effective.get(key) != value
    }
    if model_differences:
        raise RuntimeError(f"SL8 pilot parameters changed: {model_differences}")
    if effective.get("model") != MODEL or effective.get("pairwise_loss") != "hinge":
        raise RuntimeError("SL8 pilot must use AGCFSL8Coord with faithful hinge")

    module = (
        repo / "recbole_gnn" / "model" / "general_recommender" / "agcfsl8coord.py"
    )
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    if not any(
        isinstance(node, ast.ClassDef) and node.name == MODEL for node in tree.body
    ):
        raise RuntimeError(f"{module} does not define {MODEL}")

    return {
        "dataset": DATASET,
        "model": MODEL,
        "seed": SEED,
        "config_files": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in paths
        ],
        "filters": {"rating": "[3,inf)", "users": "[5,inf)", "items": "[5,inf)"},
        "split": expected_protocol["eval_args"],
        "evaluation": {
            "metrics": ["Recall", "NDCG"],
            "topk": [10, 20],
            "selection_metric": "Recall@10",
            "mode": "full",
            "validation_only": True,
            "held_out_test_evaluated": False,
        },
        "raw_source": dict(source_audit) if source_audit is not None else None,
        "expected_filtered_dataset": dict(MOVIELENS_FILTERED),
        "pilot_parameters": dict(PILOT_PARAMETERS),
    }


def _source_hashes(repo: Path) -> list[dict[str, str]]:
    relative = (
        "run_recbole_gnn.py",
        "recbole_gnn/quick_start.py",
        "recbole_gnn/model/general_recommender/agcf.py",
        "recbole_gnn/model/general_recommender/agcfsl8coord.py",
        "slrec_experiments/geometry.py",
        "slrec_experiments/run_agcf_sl8coord_movielens_pilot.py",
    )
    return [
        {"path": str((repo / name).resolve()), "sha256": _sha256(repo / name)}
        for name in relative
    ]


def campaign_contract(
    repo: Path, data_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": dict(protocol),
        "training": {
            "epochs": EPOCHS,
            "eval_step": EVAL_STEP,
            "stopping_step": STOPPING_STEP,
            "seed": SEED,
            "negative_sampling": {"uniform": 1},
            "serial_workers": 1,
            "logical_gpu": 0,
            "data_root": str(data_root.resolve()),
            "validation_only": True,
            "held_out_test_evaluated": False,
        },
        "parameters": dict(PILOT_PARAMETERS),
        "source_files": _source_hashes(repo),
    }


def artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "result": output_root / "result.json",
        "checkpoint_dir": output_root / "checkpoints",
        "log": output_root / "pilot.log",
        "manifest": output_root / "manifest.json",
        "active": output_root / ".active.json",
    }


def trial_command(args: argparse.Namespace, paths: Mapping[str, Path]) -> list[str]:
    configs = " ".join(str(path) for path in config_paths(args.repo))
    return [
        args.python,
        "-u",
        str(args.repo / "run_recbole_gnn.py"),
        "--model",
        MODEL,
        "--dataset",
        DATASET,
        "--config-files",
        configs,
        "--validation-only",
        "--result-file",
        str(paths["result"]),
        f"--checkpoint_dir={paths['checkpoint_dir']}",
        f"--gpu_id={args.gpu_id}",
        "--use_gpu=True",
        "--show_progress=False",
        f"--epochs={EPOCHS}",
        f"--eval_step={EVAL_STEP}",
        f"--stopping_step={STOPPING_STEP}",
        f"--seed={SEED}",
        "--reproducibility=True",
        f"--data_path={args.data_path}",
        "--neg_sampling={'uniform': 1}",
    ]


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    try:
        return config[key]
    except (KeyError, TypeError, AttributeError):
        final = getattr(config, "final_config_dict", None)
        return final.get(key) if isinstance(final, Mapping) else None


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_result(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result is not a JSON object: {path}")
    if payload.get("model") != MODEL or payload.get("dataset") != DATASET:
        raise ValueError("result model/dataset does not match pilot")
    if int(payload.get("seed", -1)) != SEED:
        raise ValueError("result seed does not match pilot")
    if "test_result" not in payload or payload["test_result"] is not None:
        raise RuntimeError("pilot result touched or omitted the held-out test")
    score = float(payload.get("best_valid_score", float("nan")))
    if not math.isfinite(score) or not isinstance(payload.get("best_valid_result"), Mapping):
        raise ValueError("result lacks a finite validation selection")
    splits = payload.get("split_fingerprints")
    if not isinstance(splits, Mapping):
        raise ValueError("result lacks split fingerprints")
    for name in ("train", "valid", "test"):
        split = splits.get(name)
        if not isinstance(split, Mapping) or not split.get("sha256"):
            raise ValueError(f"result has invalid {name} split fingerprint")

    checkpoint_token = payload.get("checkpoint_file")
    if not checkpoint_token:
        raise ValueError("result lacks a saved checkpoint")
    checkpoint = Path(str(checkpoint_token)).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    import torch

    try:
        saved = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch
        saved = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(saved, Mapping) or "config" not in saved:
        raise ValueError("checkpoint is not a RecBole checkpoint")
    expected = {
        "model": MODEL,
        "dataset": DATASET,
        "seed": SEED,
        "epochs": EPOCHS,
        "eval_step": EVAL_STEP,
        "stopping_step": STOPPING_STEP,
        "valid_metric": "Recall@10",
        "data_path": str(Path(contract["training"]["data_root"]) / DATASET),
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        },
        **PILOT_PARAMETERS,
    }
    mismatches = {
        key: {"expected": value, "actual": _config_value(saved["config"], key)}
        for key, value in expected.items()
        if not _same(_config_value(saved["config"], key), value)
    }
    if mismatches:
        raise ValueError(f"checkpoint pilot contract mismatch: {mismatches}")
    return payload


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def single_runner_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".single_gpu.lock"
    with lock_path.open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":  # pragma: no cover - exercised on Yanglab.
            import msvcrt

            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("another pilot runner owns this output") from error
            unlock = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another pilot runner owns this output") from error
            unlock = lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        try:
            yield
        finally:
            lock.seek(0)
            unlock()


def _refuse_live_duplicate(active: Path) -> None:
    if not active.is_file():
        return
    state = json.loads(active.read_text(encoding="utf-8"))
    if state.get("hostname") != socket.gethostname():
        raise RuntimeError("active pilot belongs to another host; verify it first")
    pid = int(state.get("child_pid", -1))
    if _pid_alive(pid):
        raise RuntimeError(f"pilot child PID {pid} is already running")
    stale = active.with_name(f".active.stale.{dt.datetime.now():%Y%m%dT%H%M%S}.json")
    os.replace(active, stale)


def _run(command: list[str], paths: Mapping[str, Path], contract: Mapping[str, Any], env: Mapping[str, str]) -> None:
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    with paths["log"].open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND=" + shlex.join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=Path(contract["source_files"][0]["path"]).parent,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _atomic_json(
            paths["active"],
            {
                "schema_version": SCHEMA_VERSION,
                "hostname": socket.gethostname(),
                "runner_pid": os.getpid(),
                "child_pid": process.pid,
                "command": command,
                "started_at": _utc_now(),
            },
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            if paths["active"].is_file():
                paths["active"].unlink()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = _DEFAULT_REPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo / "experiment_runs" / "agcf_sl8coord_movielens_pilot",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--plan-only", "--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    token = str(args.gpu_id).strip()
    if not token.isdigit() or "," in token:
        raise ValueError("--gpu-id must name exactly one non-negative physical GPU")
    args.gpu_id = str(int(token))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.repo = args.repo.expanduser().resolve()
    args.data_path = (
        args.data_path.expanduser().resolve()
        if args.data_path is not None
        else (args.repo / "dataset").resolve()
    )
    args.output_root = args.output_root.expanduser().resolve()
    _validate_args(args)
    source = audit_movielens_source(args.data_path)
    protocol = validate_protocol(args.repo, source)
    contract = campaign_contract(args.repo, args.data_path, protocol)
    paths = artifact_paths(args.output_root)
    command = trial_command(args, paths)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL,
        "dataset": DATASET,
        "status": "planned" if args.plan_only else "running",
        "physical_gpu": args.gpu_id,
        "serial_workers": 1,
        "validation_only": True,
        "held_out_test_evaluated": False,
        "command": command,
        "artifacts": {key: str(value) for key, value in paths.items() if key != "active"},
        "campaign_contract": contract,
        "created_at": _utc_now(),
    }
    if args.plan_only:
        print(json.dumps(manifest, indent=2))
        return 0

    with single_runner_lock(args.output_root):
        _refuse_live_duplicate(paths["active"])
        if paths["result"].is_file():
            result = validate_result(paths["result"], contract)
            manifest.update(
                status="complete",
                resumed=True,
                best_valid_score=float(result["best_valid_score"]),
                best_valid_result=dict(result["best_valid_result"]),
                checkpoint_file=str(Path(result["checkpoint_file"]).resolve()),
                completed_at=_utc_now(),
            )
            _atomic_json(paths["manifest"], manifest)
            print(json.dumps(manifest, indent=2))
            return 0

        _atomic_json(paths["manifest"], manifest)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        _run(command, paths, contract, env)
        result = validate_result(paths["result"], contract)
        manifest.update(
            status="complete",
            resumed=False,
            best_valid_score=float(result["best_valid_score"]),
            best_valid_result=dict(result["best_valid_result"]),
            checkpoint_file=str(Path(result["checkpoint_file"]).resolve()),
            completed_at=_utc_now(),
        )
        _atomic_json(paths["manifest"], manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
