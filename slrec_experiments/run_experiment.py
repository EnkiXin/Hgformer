#!/usr/bin/env python3
"""Run controlled recommendation experiments with upstream RecBole 1.2.1.

This entry point deliberately avoids the vendored ``recbole`` directory in this
repository.  Invoke it as a file from the repository root, for example::

    .venv-slrec/bin/python slrec_experiments/run_experiment.py \
        --model SLRec --config slrec_experiments/configs/ml-100k-smoke.yaml

Dataset/profile YAML files are merged in this order: base, model profile, then
each ``--config`` file. Values supplied by ``--set`` and the explicit CLI flags
have the highest priority.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = SCRIPT_DIR / "configs"
EXPECTED_RECBOLE_VERSION = "1.2.1"
CUSTOM_MODELS = {"slrec": "SLRec", "mixedgeorec": "MixedGeoRec"}
OFFICIAL_MODELS = {"bpr": "BPR", "lightgcn": "LightGCN"}
MODEL_PROFILES = {
    "BPR": "bpr.yaml",
    "LightGCN": "lightgcn.yaml",
    "SLRec": "slrec.yaml",
    "MixedGeoRec": "mixedgeo-product.yaml",
}


def _path_key(raw_path: str) -> Optional[Path]:
    try:
        return (Path.cwd() if raw_path == "" else Path(raw_path)).resolve()
    except (OSError, RuntimeError):
        return None


def _prepare_import_path() -> Tuple[str, str]:
    """Import the installed RecBole while keeping local experiment imports usable."""

    local_recbole = (REPO_ROOT / "recbole").resolve()
    cleaned: List[str] = []
    for entry in sys.path:
        resolved = _path_key(entry)
        if resolved == REPO_ROOT or resolved == local_recbole:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned

    try:
        import recbole  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "Upstream RecBole is not installed. Install the isolated requirements with "
            "`.venv-slrec/bin/python -m pip install -r "
            "slrec_experiments/requirements.txt`."
        ) from exc

    origin = Path(recbole.__file__).resolve()
    try:
        origin.relative_to(local_recbole)
    except ValueError:
        pass
    else:
        raise RuntimeError(
            f"Refusing to use the incomplete vendored RecBole at {origin}. "
            "Run this file directly with an environment containing recbole==1.2.1."
        )

    try:
        installed_version = importlib.metadata.version("recbole")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Imported RecBole from {origin}, but no installed distribution was found. "
            f"Expected recbole=={EXPECTED_RECBOLE_VERSION}."
        ) from exc
    module_version = getattr(recbole, "__version__", installed_version)
    if installed_version != EXPECTED_RECBOLE_VERSION or module_version != EXPECTED_RECBOLE_VERSION:
        raise RuntimeError(
            "Wrong RecBole version: imported "
            f"{module_version} from {origin}; installed distribution is {installed_version}. "
            f"This experiment requires exactly recbole=={EXPECTED_RECBOLE_VERSION}."
        )

    # Append rather than prepend: experiment modules remain importable, while the
    # installed RecBole keeps precedence over the repository's directory.
    sys.path.append(str(REPO_ROOT))
    return installed_version, str(origin)


def _canonical_model_name(raw_name: str) -> str:
    normalized = raw_name.replace("-", "").replace("_", "").lower()
    aliases = {
        **OFFICIAL_MODELS,
        **CUSTOM_MODELS,
        "mixedgeo": "MixedGeoRec",
        "productrec": "MixedGeoRec",
    }
    if normalized not in aliases:
        supported = ", ".join(["BPR", "LightGCN", "SLRec", "MixedGeoRec"])
        raise argparse.ArgumentTypeError(f"unknown model {raw_name!r}; choose one of {supported}")
    return aliases[normalized]


def _single_gpu(raw_value: str) -> str:
    if not raw_value.isdigit() or int(raw_value) < 0:
        raise argparse.ArgumentTypeError("--gpu must be one non-negative physical GPU index")
    return str(int(raw_value))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SL/mixed-geometry recommendation experiments with official RecBole 1.2.1."
    )
    parser.add_argument("--model", type=_canonical_model_name, default="SLRec")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override the dataset in YAML (for example Amazon_cd or DoubanBook).",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="YAML",
        help="Additional YAML file; repeat to merge several files in order.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Highest-priority RecBole setting; YAML scalar/list/dict syntax is accepted.",
    )
    parser.add_argument("--seed", type=int, default=2024)
    device = parser.add_mutually_exclusive_group()
    device.add_argument(
        "--gpu",
        type=_single_gpu,
        default="0",
        metavar="INDEX",
        help="Expose exactly this physical GPU (default: 0).",
    )
    device.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from YAML.")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--save-model", action="store_true", help="Keep the best checkpoint.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse configs, prepare data, and instantiate the model without training.",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Parse and print the resolved configuration without loading data.",
    )
    parser.add_argument(
        "--no-default-profile",
        action="store_true",
        help="Do not automatically load the model profile YAML.",
    )
    parser.add_argument(
        "--result-file",
        default=None,
        metavar="JSON",
        help="Result JSON path; otherwise a timestamped file is written under results/.",
    )
    return parser


def _configure_device(args: argparse.Namespace) -> None:
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # A physical index is mapped to RecBole's single logical device zero.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def _resolve_config_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {candidate}")
    return candidate


def _config_files(args: argparse.Namespace) -> List[str]:
    files: List[Path] = [CONFIG_DIR / "base.yaml"]
    if not args.no_default_profile:
        files.append(CONFIG_DIR / MODEL_PROFILES[args.model])
    if args.config:
        files.extend(_resolve_config_path(path) for path in args.config)
    else:
        files.append(CONFIG_DIR / "ml-100k-smoke.yaml")

    deduplicated: List[str] = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"configuration file does not exist: {resolved}")
        if resolved not in seen:
            deduplicated.append(str(resolved))
            seen.add(resolved)
    return deduplicated


def _parse_overrides(raw_overrides: Iterable[str]) -> Dict[str, Any]:
    import yaml  # Imported after the environment is selected.

    parsed: Dict[str, Any] = {}
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set has an empty key: {item!r}")
        parsed[key] = yaml.safe_load(raw_value)
    return parsed


def _load_model(model_name: str) -> Union[str, type]:
    if model_name in OFFICIAL_MODELS.values():
        return model_name
    if model_name == "SLRec":
        from slrec_experiments.slrec import SLRec

        return SLRec
    if model_name == "MixedGeoRec":
        from slrec_experiments.mixedgeo import MixedGeoRec

        return MixedGeoRec
    raise ValueError(f"unsupported model: {model_name}")


@contextmanager
def _without_recbole_cli_args():
    """Prevent RecBole from interpreting this runner's argparse flags."""

    original = sys.argv
    sys.argv = [original[0]]
    try:
        yield
    finally:
        sys.argv = original


def _apply_experiment_data_path(config: Any) -> None:
    """Resolve repository-local dataset paths after RecBole's own defaults."""

    exact_path = config["experiment_data_path"]
    data_root = config["experiment_data_root"]
    if exact_path and data_root:
        raise ValueError("set only one of experiment_data_path and experiment_data_root")
    if exact_path:
        path = Path(str(exact_path)).expanduser()
    elif data_root:
        path = Path(str(data_root)).expanduser() / str(config["dataset"])
    else:
        return
    if not path.is_absolute():
        path = REPO_ROOT / path
    config["data_path"] = str(path.resolve())


def _apply_eval_batch_cap(config: Any) -> None:
    """Bound full-sort geometry batches without penalizing light baselines."""

    cap = config["experiment_eval_batch_cap"]
    if cap is None:
        return
    cap = int(cap)
    if cap <= 0:
        return
    config["eval_batch_size"] = min(int(config["eval_batch_size"]), cap)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _result_path(args: argparse.Namespace, dataset: str) -> Path:
    if args.result_file:
        path = Path(args.result_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{dataset}-{args.model}-seed{args.seed}-{timestamp}.json"
    return SCRIPT_DIR / "results" / filename


def _run(args: argparse.Namespace, recbole_version: str, recbole_origin: str) -> int:
    import torch
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_trainer, init_logger, init_seed

    model_spec = _load_model(args.model)
    config_files = _config_files(args)
    overrides = _parse_overrides(args.overrides)
    overrides.update(
        {
            "seed": args.seed,
            "reproducibility": True,
            # RecBole 1.2.1 selects CPU from an empty gpu_id; use logical zero
            # after CUDA_VISIBLE_DEVICES maps the chosen physical card.
            "gpu_id": "" if args.cpu else 0,
            "use_gpu": not args.cpu,
            "show_progress": args.show_progress,
        }
    )
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        overrides["epochs"] = args.epochs

    with _without_recbole_cli_args():
        config = Config(
            model=model_spec,
            dataset=args.dataset,
            config_file_list=config_files,
            config_dict=overrides,
        )
    _apply_experiment_data_path(config)
    _apply_eval_batch_cap(config)

    import_report = {
        "recbole_version": recbole_version,
        "recbole_origin": recbole_origin,
        "torch_version": torch.__version__,
        "device": str(config["device"]),
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "model": config["model"],
        "dataset": config["dataset"],
        "seed": config["seed"],
        "config_files": config_files,
        "data_path": config["data_path"],
        "eval_batch_size": config["eval_batch_size"],
    }
    print(json.dumps(import_report, indent=2, ensure_ascii=False))
    if args.config_only:
        return 0

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model_class = config.model_class
    model = model_class(config, train_data._dataset).to(config["device"])

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": "ok",
                    "model_class": model_class.__name__,
                    "parameters": parameter_count,
                    "users": dataset.user_num,
                    "items": dataset.item_num,
                    "interactions": len(dataset),
                },
                indent=2,
            )
        )
        return 0

    trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer = trainer_class(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        # Always evaluate the selected best epoch. The temporary checkpoint is
        # removed below unless the caller explicitly asks to keep it.
        saved=True,
        show_progress=config["show_progress"],
    )
    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config["show_progress"],
    )
    result = {
        **import_report,
        "parameter_count": parameter_count,
        "best_valid_score": best_valid_score,
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }
    checkpoint_path = Path(trainer.saved_model_file)
    if args.save_model:
        result["checkpoint"] = str(checkpoint_path.resolve())
    elif checkpoint_path.is_file():
        checkpoint_path.unlink()
    output_path = _result_path(args, str(config["dataset"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"RESULT_JSON={output_path}")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_device(args)
    try:
        recbole_version, recbole_origin = _prepare_import_path()
        return _run(args, recbole_version, recbole_origin)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
