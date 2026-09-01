#!/usr/bin/env python3
"""Validation-only, resumable Hgformer tuning on Amazon-CD.

The search deliberately never evaluates the held-out test split. After all
trials finish, ``--final-test`` retrains only the validation-selected setting
and evaluates test once. The already-running fixed-configuration reproduction
can optionally be included as a validation candidate via ``--baseline-result``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Trial:
    name: str
    learning_rate: float
    gcn_layers: int
    alpha: float
    curve: float
    temp: float
    margin: float

    def recbole_args(self) -> list[str]:
        return [
            f"--learning_rate={self.learning_rate}",
            f"--gcn_layers={self.gcn_layers}",
            f"--alpha={self.alpha}",
            f"--curve={self.curve}",
            f"--temp={self.temp}",
            f"--margin={self.margin}",
        ]


# Fractional grid centred on the paper/repository sensitivity region. Existing
# analysis already rules out high curvature, shallow LHGCN, and alpha >= 0.4.
TRIALS = (
    Trial("m015-a020-t005-lr5e4-l7", 5e-4, 7, 0.20, 0.1, 0.05, 0.15),
    Trial("m015-a025-t005-lr5e4-l7", 5e-4, 7, 0.25, 0.1, 0.05, 0.15),
    Trial("m030-a025-t005-lr5e4-l7", 5e-4, 7, 0.25, 0.1, 0.05, 0.30),
    Trial("m015-a030-t005-lr5e4-l7", 5e-4, 7, 0.30, 0.1, 0.05, 0.15),
    Trial("m030-a030-t005-lr5e4-l7", 5e-4, 7, 0.30, 0.1, 0.05, 0.30),
    Trial("m015-a020-t001-lr5e4-l7", 5e-4, 7, 0.20, 0.1, 0.01, 0.15),
    Trial("m030-a025-t001-lr5e4-l7", 5e-4, 7, 0.25, 0.1, 0.01, 0.30),
    Trial("m015-a030-t001-lr5e4-l7", 5e-4, 7, 0.30, 0.1, 0.01, 0.15),
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--stopping-step", type=int, default=30)
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("best_valid_score") is None:
        raise ValueError(f"missing best_valid_score in {path}")
    return payload


def run_and_tee(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=env["HGFORMER_REPO"],
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
        raise subprocess.CalledProcessError(return_code, command)


def trial_command(
    args: argparse.Namespace,
    trial: Trial,
    result_path: Path,
    *,
    final_test: bool,
) -> list[str]:
    command = [
        args.python,
        "-u",
        str(args.repo / "run_recbole_gnn.py"),
        "--model",
        "RecFormer",
        "--config-files",
        str(args.repo / "baseline_config_fixed/RecFormer_cd.yaml"),
        "--result-file",
        str(result_path),
        "--gpu_id=0",
        "--use_gpu=True",
        "--show_progress=False",
        f"--checkpoint_dir={args.output_root / 'checkpoints'}",
        f"--epochs={args.epochs}",
        f"--stopping_step={args.stopping_step}",
        *trial.recbole_args(),
    ]
    if final_test:
        return command
    return [*command, "--validation-only", "--no-save"]


def write_summary(output_root: Path, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(candidates, key=lambda item: item["best_valid_score"], reverse=True)
    summary = {
        "selection_metric": "Recall@10 on validation only",
        "candidate_count": len(ranked),
        "best": ranked[0] if ranked else None,
        "ranking": ranked,
    }
    path = output_root / "tuning-summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    args.repo = args.repo.resolve()
    args.output_root = args.output_root.resolve()
    if args.list:
        print(json.dumps([asdict(trial) for trial in TRIALS], indent=2))
        return 0

    results_dir = args.output_root / "tuning-results"
    logs_dir = args.output_root / "tuning-logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (args.output_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["HGFORMER_REPO"] = str(args.repo)

    candidates: list[dict[str, Any]] = []
    if args.baseline_result and args.baseline_result.exists():
        baseline = load_result(args.baseline_result)
        candidates.append(
            {
                "name": "fixed-config-reproduction",
                "source": str(args.baseline_result.resolve()),
                "parameters": {
                    "learning_rate": 5e-4,
                    "gcn_layers": 7,
                    "alpha": 0.20,
                    "curve": 0.1,
                    "temp": 0.05,
                    "margin": 0.30,
                },
                "best_valid_score": float(baseline["best_valid_score"]),
                "best_valid_result": baseline["best_valid_result"],
            }
        )

    for index, trial in enumerate(TRIALS, 1):
        result_path = results_dir / f"{trial.name}.json"
        log_path = logs_dir / f"{trial.name}.log"
        if result_path.exists():
            print(f"[{index}/{len(TRIALS)}] resume {trial.name}")
        else:
            print(f"[{index}/{len(TRIALS)}] start {trial.name}")
            run_and_tee(
                trial_command(args, trial, result_path, final_test=False),
                log_path,
                env,
            )
        result = load_result(result_path)
        if result.get("test_result") is not None:
            raise RuntimeError(f"tuning trial touched test split: {result_path}")
        candidates.append(
            {
                "name": trial.name,
                "source": str(result_path),
                "parameters": asdict(trial) | {},
                "best_valid_score": float(result["best_valid_score"]),
                "best_valid_result": result["best_valid_result"],
            }
        )
        summary = write_summary(args.output_root, candidates)
        print(
            f"current best={summary['best']['name']} "
            f"valid Recall@10={summary['best']['best_valid_score']:.6f}"
        )

    summary = write_summary(args.output_root, candidates)
    best = summary["best"]
    if not args.final_test:
        print("Search complete; test split was not evaluated.")
        return 0

    if best["name"] == "fixed-config-reproduction":
        print("Fixed reproduction remains validation-best; its existing test result is final.")
        return 0

    selected = next(trial for trial in TRIALS if trial.name == best["name"])
    final_result = args.output_root / "results" / f"hgformer-cd-tuned-{selected.name}.json"
    final_log = args.output_root / "logs" / f"hgformer-cd-tuned-{selected.name}.log"
    if not final_result.exists():
        print(f"Retraining validation-selected setting for one final test: {selected.name}")
        run_and_tee(
            trial_command(args, selected, final_result, final_test=True),
            final_log,
            env,
        )
    print(f"FINAL_RESULT={final_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
