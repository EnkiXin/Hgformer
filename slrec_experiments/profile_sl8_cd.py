#!/usr/bin/env python3
"""Profile the exact SL8-LHGCN Amazon-CD execution path.

This is a read-only performance diagnostic: it builds the same filtered
dataset/split and model used by the experiment, but it never takes an
optimizer step, saves a checkpoint, or evaluates the held-out test split.

The report separates:

* dataset construction and splitting;
* initial all-entity matrix exponential;
* each sparse graph aggregation and determinant retraction;
* one real-size training-batch forward and backward;
* one full-sort scorer chunk, with and without audit-only diagnostics;
* optionally, one complete full-ranking validation pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from recbole.utils import init_seed
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer
from slrec_experiments.geometry import (
    one_sided_gregory_frobenius_distance_k12,
    trace_free,
)
from slrec_experiments.sl_lhgcn import project_ambient_to_sl


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


class Timer:
    def __init__(self, device: torch.device, report: Dict[str, Any]) -> None:
        self.device = device
        self.report = report

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def call(
        self,
        name: str,
        function: Callable[[], Any],
        *,
        section: str = "timings",
    ) -> Any:
        self.sync()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        result = function()
        self.sync()
        seconds = time.perf_counter() - started
        record = {"seconds": seconds}
        if self.device.type == "cuda":
            record["peak_allocated_mib"] = (
                torch.cuda.max_memory_allocated(self.device) / 2**20
            )
            record["peak_reserved_mib"] = (
                torch.cuda.max_memory_reserved(self.device) / 2**20
            )
        self.report.setdefault(section, {})[name] = record
        print(
            f"PROFILE {section}.{name}: {seconds:.6f}s"
            + (
                f" peak={record['peak_allocated_mib']:.1f}MiB"
                if "peak_allocated_mib" in record
                else ""
            ),
            flush=True,
        )
        return result


class ProfilerCompiledFullSortScorer:
    """Profiler-only, shape-cached compiler for the production fast scorer.

    The base model and formal experiment configuration remain untouched.  The
    wrapper is installed only around one profiler validation call.  It keeps
    SL8LHGCN's one-time distance-membership diagnostic in eager mode, then
    dispatches the pure K12/Frobenius geometry kernel through one
    ``fullgraph=True`` compiled callable per concrete user/item tail shape.

    A shape automatically falls back to the original eager model method if
    compilation, execution, or the first-call score agreement check fails.
    """

    def __init__(
        self,
        model: Any,
        *,
        compile_mode: str = "reduce-overhead",
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> None:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile-scorer requires torch.compile")
        if int(model.matrix_dim) != 8:
            raise ValueError("the profiler compiled scorer is specialised to SL(8)")
        if int(model.num_factors) != 1:
            raise ValueError("the profiler compiled scorer requires num_factors=1")
        if not bool(model._uses_fast_one_sided_frobenius()):
            raise ValueError(
                "the profiler compiled scorer requires the one-sided p=2 K12 fast path"
            )
        if compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError(f"unsupported torch.compile mode: {compile_mode!r}")
        if atol < 0 or rtol < 0:
            raise ValueError("compiled scorer tolerances must be non-negative")

        self.model = model
        self.compile_mode = compile_mode
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.log_jitter = float(model.log_jitter)
        self._original_score_groups = model._score_groups
        self._compiled_by_shape: Dict[Tuple[Any, ...], Optional[Callable[..., Any]]] = {}
        self._shape_records: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._installed = False

        # Keep this function free of model state and Python-side diagnostics so
        # Dynamo sees only the production geometry graph and scalar score.
        log_jitter = self.log_jitter

        def pure_score(
            left: torch.Tensor,
            right: torch.Tensor,
            score_scale: torch.Tensor,
        ) -> torch.Tensor:
            factor_distance = one_sided_gregory_frobenius_distance_k12(
                left, right, jitter=log_jitter
            )
            return -score_scale * factor_distance.squeeze(-1)

        self._pure_score = pure_score

    @staticmethod
    def _sync(tensor: torch.Tensor) -> None:
        if tensor.device.type == "cuda":
            torch.cuda.synchronize(tensor.device)

    @staticmethod
    def _shape_key(
        left: torch.Tensor, right: torch.Tensor, scale: torch.Tensor
    ) -> Tuple[Any, ...]:
        return (
            tuple(left.shape),
            tuple(right.shape),
            str(left.dtype),
            str(right.dtype),
            str(scale.dtype),
            str(left.device),
        )

    @staticmethod
    def _shape_label(key: Tuple[Any, ...]) -> str:
        left_shape, right_shape = key[0], key[1]
        return f"u{left_shape[0]}_i{right_shape[1]}"

    def _new_record(self, key: Tuple[Any, ...]) -> Dict[str, Any]:
        return {
            "label": self._shape_label(key),
            "left_shape": list(key[0]),
            "right_shape": list(key[1]),
            "left_dtype": key[2],
            "right_dtype": key[3],
            "scale_dtype": key[4],
            "device": key[5],
            "calls": 0,
            "compiled_calls": 0,
            "eager_fallback_calls": 0,
            "compile_callable_creation_seconds": 0.0,
            "first_compiled_call_seconds": 0.0,
            "first_eager_reference_seconds": 0.0,
            "first_shape_setup_seconds": 0.0,
            "agreement": None,
            "fallback_error": None,
            "runtime_fallback_error": None,
        }

    def _compile_shape(
        self,
        key: Tuple[Any, ...],
        left: torch.Tensor,
        right: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        record = self._shape_records[key]
        shape_started = time.perf_counter()
        try:
            creation_started = time.perf_counter()
            compiled = torch.compile(
                self._pure_score,
                fullgraph=True,
                dynamic=False,
                mode=self.compile_mode,
            )
            record["compile_callable_creation_seconds"] = (
                time.perf_counter() - creation_started
            )

            eager_started = time.perf_counter()
            eager_reference = self._original_score_groups(left, right)
            self._sync(eager_reference)
            record["first_eager_reference_seconds"] = (
                time.perf_counter() - eager_started
            )

            compiled_started = time.perf_counter()
            compiled_result = compiled(left, right, scale)
            self._sync(compiled_result)
            record["first_compiled_call_seconds"] = (
                time.perf_counter() - compiled_started
            )

            absolute_difference = (compiled_result - eager_reference).abs()
            max_abs_difference = float(absolute_difference.max().item())
            max_relative_difference = float(
                (
                    absolute_difference
                    / eager_reference.abs().clamp_min(1e-12)
                )
                .max()
                .item()
            )
            scores_close = bool(
                torch.allclose(
                    compiled_result,
                    eager_reference,
                    atol=self.atol,
                    rtol=self.rtol,
                )
            )
            record["agreement"] = {
                "scores_bitwise_equal": bool(
                    torch.equal(compiled_result, eager_reference)
                ),
                "scores_close": scores_close,
                "max_abs_difference": max_abs_difference,
                "max_relative_difference": max_relative_difference,
                "atol": self.atol,
                "rtol": self.rtol,
            }
            if not scores_close:
                record["fallback_error"] = (
                    "compiled score agreement check failed: "
                    f"max_abs={max_abs_difference:.6g}, "
                    f"max_relative={max_relative_difference:.6g}"
                )
                self._compiled_by_shape[key] = None
                record["eager_fallback_calls"] += 1
                return eager_reference

            self._compiled_by_shape[key] = compiled
            record["compiled_calls"] += 1
            return compiled_result
        except Exception as error:
            record["fallback_error"] = f"{type(error).__name__}: {error}"
            self._compiled_by_shape[key] = None
            record["eager_fallback_calls"] += 1
            return self._original_score_groups(left, right)
        finally:
            record["first_shape_setup_seconds"] = (
                time.perf_counter() - shape_started
            )

    def __call__(
        self, user_group: torch.Tensor, item_group: torch.Tensor
    ) -> torch.Tensor:
        left, right = self.model._align_pair_shapes(user_group, item_group)
        # Preserve the production one-time audit before entering the pure
        # compiled graph.  It becomes a no-op on all later chunks for this
        # freshly materialised representation cache.
        self.model._record_distance_membership_diagnostics(left, right)
        scale = self.model._score_scale()
        key = self._shape_key(left, right, scale)
        record = self._shape_records.setdefault(key, self._new_record(key))
        record["calls"] += 1

        if key not in self._compiled_by_shape:
            return self._compile_shape(key, left, right, scale)
        compiled = self._compiled_by_shape[key]
        if compiled is None:
            record["eager_fallback_calls"] += 1
            return self._original_score_groups(left, right)
        try:
            result = compiled(left, right, scale)
            record["compiled_calls"] += 1
            return result
        except Exception as error:
            record["runtime_fallback_error"] = f"{type(error).__name__}: {error}"
            self._compiled_by_shape[key] = None
            record["eager_fallback_calls"] += 1
            return self._original_score_groups(left, right)

    @contextmanager
    def installed(self) -> Iterator["ProfilerCompiledFullSortScorer"]:
        if self._installed:
            raise RuntimeError("compiled scorer wrapper is already installed")

        def override(
            _model: Any,
            user_group: torch.Tensor,
            item_group: torch.Tensor,
        ) -> torch.Tensor:
            return self(user_group, item_group)

        self._installed = True
        self.model._score_groups = types.MethodType(override, self.model)
        try:
            yield self
        finally:
            # The method originates on the class, so remove the temporary
            # instance override and reveal the original descriptor again.
            del self.model._score_groups
            self._installed = False

    def report(self) -> Dict[str, Any]:
        records = sorted(
            self._shape_records.values(), key=lambda item: item["label"]
        )
        return {
            "enabled": True,
            "scope": "profiler full-sort validation only",
            "compile_mode": self.compile_mode,
            "fullgraph": True,
            "dynamic": False,
            "shape_cache_entries": len(records),
            "shape_records": records,
            "total_calls": sum(record["calls"] for record in records),
            "total_compiled_calls": sum(
                record["compiled_calls"] for record in records
            ),
            "total_eager_fallback_calls": sum(
                record["eager_fallback_calls"] for record in records
            ),
            "total_first_shape_setup_seconds": sum(
                record["first_shape_setup_seconds"] for record in records
            ),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=REPO_ROOT,
        help="Hgformer repository root",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "dataset",
        help="directory containing Amazon_cd/Amazon_cd.inter",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiment_runs" / "sl8_cd_profile.json",
    )
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=7,
        help="physical GPU index passed through RecBole's gpu_id setting",
    )
    parser.add_argument("--train-batch-size", type=int, default=131072)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--loss-margin", type=float, default=0.3)
    parser.add_argument("--score-user-chunk", type=int, default=64)
    parser.add_argument("--score-item-chunk", type=int, default=1024)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1048576,
        help="RecBole full-sort score budget; does not change candidates",
    )
    parser.add_argument("--score-repeats", type=int, default=3)
    parser.add_argument(
        "--skip-training-batch-profile",
        action="store_true",
        help="skip the diagnostic loss/backward batch (useful for validation-only runs)",
    )
    parser.add_argument(
        "--compile-scorer",
        action="store_true",
        help=(
            "profiler-only: compile/cache the pure production fast scorer "
            "during full validation; base model/config remain unchanged"
        ),
    )
    parser.add_argument(
        "--compile-scorer-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="also time one complete full-ranking validation pass",
    )
    return parser.parse_args()


def _mean_measurement(
    timer: Timer,
    name: str,
    function: Callable[[], Any],
    repeats: int,
) -> Tuple[Any, Dict[str, float]]:
    values = []
    result = None
    for repeat in range(repeats):
        result = timer.call(
            f"{name}.repeat_{repeat + 1}", function, section="scorer_microbench"
        )
        values.append(
            timer.report["scorer_microbench"][f"{name}.repeat_{repeat + 1}"][
                "seconds"
            ]
        )
    summary = {
        "mean_seconds": sum(values) / len(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }
    timer.report.setdefault("scorer_microbench_summary", {})[name] = summary
    return result, summary


def main() -> None:
    args = _parse_args()
    # RecBole's legacy Config also inspects process argv. The profiler options
    # are not model overrides and must not leak into that parser.
    sys.argv = [sys.argv[0]]
    args.repo = args.repo.expanduser().resolve()
    args.data_path = args.data_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if str(args.repo) not in sys.path:
        sys.path.insert(0, str(args.repo))

    config_files = [
        args.repo / "baseline_config_fixed" / "RecFormer_cd.yaml",
        args.repo / "baseline_config_fixed" / "SL8LHGCN_reproduction.yaml",
    ]
    for path in config_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    report: Dict[str, Any] = {
        "schema_version": 1,
        "purpose": "SL8-LHGCN exact-path performance profile; no optimizer step/test",
        "config_files": [str(path) for path in config_files],
        "requested": {
            "layers": args.layers,
            "gpu_id": args.gpu_id,
            "train_batch_size": args.train_batch_size,
            "learning_rate": args.learning_rate,
            "loss_margin": args.loss_margin,
            "score_user_chunk": args.score_user_chunk,
            "score_item_chunk": args.score_item_chunk,
            "eval_batch_size": args.eval_batch_size,
            "full_validation": args.full_validation,
            "compile_scorer": args.compile_scorer,
            "compile_scorer_mode": args.compile_scorer_mode,
            "skip_training_batch_profile": args.skip_training_batch_profile,
        },
    }

    cpu_started = time.perf_counter()
    config = Config(
        model="SL8LHGCN",
        dataset="Amazon_cd",
        config_file_list=[str(path) for path in config_files],
        config_dict={
            "data_path": str(args.data_path),
            # RecBole rewrites CUDA_VISIBLE_DEVICES from this value. Merely
            # setting CUDA_VISIBLE_DEVICES in the parent shell is insufficient
            # when a dataset YAML still says gpu_id: 0.
            "gpu_id": args.gpu_id,
            "gcn_layers": args.layers,
            "n_layers": args.layers,
            "train_batch_size": args.train_batch_size,
            "learning_rate": args.learning_rate,
            "loss_margin": args.loss_margin,
            "schatten_p": 2,
            "eval_user_chunk_size": args.score_user_chunk,
            "eval_item_chunk_size": args.score_item_chunk,
            "eval_batch_size": args.eval_batch_size,
            "save_dataset": False,
            "save_dataloaders": False,
            "show_progress": False,
        },
    )
    report.setdefault("setup", {})["config_seconds"] = time.perf_counter() - cpu_started
    device = torch.device(config["device"])
    if device.type != "cuda":
        raise RuntimeError(f"this profiler requires CUDA; resolved device={device}")
    report["environment"] = {
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
    }
    timer = Timer(device, report)

    init_seed(config["seed"], config["reproducibility"])
    started = time.perf_counter()
    dataset = create_dataset(config)
    report["setup"]["create_dataset_seconds"] = time.perf_counter() - started
    print(
        f"PROFILE setup.create_dataset: {report['setup']['create_dataset_seconds']:.6f}s",
        flush=True,
    )
    started = time.perf_counter()
    train_data, valid_data, _test_data = data_preparation(config, dataset)
    report["setup"]["data_preparation_seconds"] = time.perf_counter() - started
    print(
        f"PROFILE setup.data_preparation: {report['setup']['data_preparation_seconds']:.6f}s",
        flush=True,
    )
    configured_outer_users = config["full_sort_user_batch_size"]
    outer_user_batch = (
        int(configured_outer_users)
        if configured_outer_users is not None
        else max(
            int(config["eval_batch_size"])
            // int(train_data.dataset.item_num),
            1,
        )
    )
    report["dataset"] = {
        "users_including_padding": int(train_data.dataset.user_num),
        "items_including_padding": int(train_data.dataset.item_num),
        "train_interactions": int(len(train_data.dataset.inter_feat)),
        "valid_interactions": int(len(valid_data.dataset.inter_feat)),
        "valid_loader_batches": int(len(valid_data)),
        "eval_batch_size": int(config["eval_batch_size"]),
        "configured_full_sort_user_batch_size": (
            int(configured_outer_users)
            if configured_outer_users is not None
            else None
        ),
        "full_sort_users_per_outer_batch": outer_user_batch,
        "train_batches": int(len(train_data)),
    }

    init_seed(config["seed"], config["reproducibility"])
    started = time.perf_counter()
    model = get_model(config["model"])(config, train_data.dataset).to(device)
    timer.sync()
    report["setup"]["model_init_and_to_cuda_seconds"] = time.perf_counter() - started
    report["model"] = {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "layers": int(model.n_layers),
        "matrix_dim": int(model.matrix_dim),
        "log_terms": int(model.log_terms),
        "symmetric_distance": bool(model.symmetric_distance),
        "membership_check": bool(model.sl_membership_check),
        "distance_membership_check": bool(model.sl_distance_membership_check),
        "adjacency_nnz": int(model.norm_adj_matrix._nnz()),
    }
    print(
        "PROFILE setup.model_init_and_to_cuda: "
        f"{report['setup']['model_init_and_to_cuda_seconds']:.6f}s",
        flush=True,
    )

    # Warm CUDA libraries once so the following stage timings do not charge
    # one-time context initialisation to the first matrix operation.
    _ = torch.eye(8, device=device) @ torch.eye(8, device=device)
    timer.sync()

    model.eval()
    with torch.no_grad():
        raw = timer.call("raw_coordinate_table", model._raw_coordinate_table)
        coordinates = timer.call("trace_free", lambda: trace_free(raw))
        groups = timer.call("initial_all_entity_matrix_exp", lambda: model._to_group(coordinates))

        layer_records = []
        for layer_index in range(model.n_layers):
            flat = groups.reshape(groups.shape[0], -1)
            ambient_flat = timer.call(
                f"layer_{layer_index + 1}.sparse_mm",
                lambda flat=flat: torch.sparse.mm(model.norm_adj_matrix, flat),
            )
            ambient = ambient_flat.reshape_as(groups)
            _ = timer.call(
                f"layer_{layer_index + 1}.input_slogdet_only",
                lambda ambient=ambient: torch.linalg.slogdet(ambient),
            )
            projected, diagnostics = timer.call(
                f"layer_{layer_index + 1}.current_retraction_total",
                lambda ambient=ambient: project_ambient_to_sl(
                    ambient,
                    fallback_clip=model.sl_centroid_fallback_clip,
                    collect_diagnostics=True,
                    active_mask=model.sl_active_node_mask,
                    membership_tolerance=model.sl_membership_tolerance,
                    strict_membership=model.sl_membership_strict,
                ),
            )
            groups = projected
            _ = timer.call(
                f"layer_{layer_index + 1}.output_slogdet_only",
                lambda groups=groups: torch.linalg.slogdet(groups),
            )
            layer_records.append(
                {
                    "layer": layer_index + 1,
                    "projection_total": diagnostics.total,
                    "orientation_repairs": diagnostics.orientation_repairs,
                    "singular_fallbacks": diagnostics.singular_fallbacks,
                }
            )
        report["manual_layer_diagnostics"] = layer_records
        final_groups = groups

        # Current forward includes all audit-only membership checks and their
        # host synchronisations. Time it separately from the decomposed path.
        _ = timer.call("current_model_forward_with_diagnostics", model.forward)

    # Score a chunk of the exact shape used by full_sort_predict.  Compare the
    # current audit-on path against the numerically identical audit-off path.
    user_groups, item_groups = torch.split(
        final_groups, (model.n_users, model.n_items), dim=0
    )
    # The model's user chunk cannot exceed the number of users supplied by
    # RecBole's outer full-sort loader. Benchmark the shape that really runs.
    user_count = min(
        args.score_user_chunk, outer_user_batch, model.n_users - 1
    )
    item_count = min(args.score_item_chunk, model.n_items - 1)
    scorer_users = user_groups[1 : 1 + user_count, None, ...]
    scorer_items = item_groups[None, 1 : 1 + item_count, ...]
    score_pairs = user_count * item_count
    report["scorer_microbench_shape"] = {
        "users": user_count,
        "items": item_count,
        "pairs": score_pairs,
    }
    original_distance_check = model.sl_distance_membership_check
    model.sl_distance_membership_check = True
    _, scorer_with_audit = _mean_measurement(
        timer,
        "full_sort_chunk_audit_on",
        lambda: model._score_groups(scorer_users, scorer_items),
        args.score_repeats,
    )
    model.sl_distance_membership_check = False
    _, scorer_without_audit = _mean_measurement(
        timer,
        "full_sort_chunk_audit_off",
        lambda: model._score_groups(scorer_users, scorer_items),
        args.score_repeats,
    )
    model.sl_distance_membership_check = original_distance_check

    total_pairs = (model.n_users - 1) * model.n_items
    full_outer_batches, remainder_users = divmod(
        model.n_users - 1, outer_user_batch
    )
    user_chunks = full_outer_batches * math.ceil(outer_user_batch / user_count)
    if remainder_users:
        user_chunks += math.ceil(remainder_users / user_count)
    approximate_chunks = user_chunks * math.ceil(model.n_items / item_count)
    report["full_sort_extrapolation"] = {
        "total_user_item_pairs": total_pairs,
        "real_candidate_pairs_excluding_padding": (
            (model.n_users - 1) * (model.n_items - 1)
        ),
        "approximate_chunks": approximate_chunks,
        "audit_on_scorer_seconds": approximate_chunks
        * scorer_with_audit["mean_seconds"],
        "audit_off_scorer_seconds": approximate_chunks
        * scorer_without_audit["mean_seconds"],
        "note": "linear scorer-only estimate; evaluator/data-collection overhead excluded",
    }
    print(
        "PROFILE estimated full scorer: "
        f"audit_on={report['full_sort_extrapolation']['audit_on_scorer_seconds']:.1f}s "
        f"audit_off={report['full_sort_extrapolation']['audit_off_scorer_seconds']:.1f}s",
        flush=True,
    )

    # A real dataloader batch exercises the exact current loss, including one
    # all-node graph rollout and positive/negative matrix-log distances.
    if args.skip_training_batch_profile:
        report["training_batch"] = {"skipped": True}
    else:
        model.train()
        interaction = next(iter(train_data)).to(device)
        report["training_batch"] = {
            "rows": int(len(interaction)),
            "configured_batch_size": int(config["train_batch_size"]),
        }
        model.zero_grad(set_to_none=True)
        loss = timer.call(
            "one_training_batch_forward_loss",
            lambda: model.calculate_loss(interaction),
        )
        report["training_batch"]["loss"] = float(loss.detach().cpu())
        _ = timer.call("one_training_batch_backward", loss.backward)
        model.zero_grad(set_to_none=True)

    _write_report(args.output, report)
    print(f"PROFILE_PARTIAL_JSON={args.output}", flush=True)

    if args.full_validation:
        model.eval()
        model._clear_full_sort_cache()
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        compiled_scorer = None
        if args.compile_scorer:
            compiled_scorer = ProfilerCompiledFullSortScorer(
                model, compile_mode=args.compile_scorer_mode
            )
        try:
            if compiled_scorer is None:
                validation_result = timer.call(
                    "one_complete_full_ranking_validation",
                    lambda: trainer.evaluate(
                        valid_data, load_best_model=False, show_progress=False
                    ),
                    section="validation",
                )
            else:
                with compiled_scorer.installed():
                    validation_result = timer.call(
                        "one_complete_full_ranking_validation",
                        lambda: trainer.evaluate(
                            valid_data,
                            load_best_model=False,
                            show_progress=False,
                        ),
                        section="validation",
                    )
        finally:
            if compiled_scorer is not None:
                report["compiled_full_sort_scorer"] = compiled_scorer.report()
                _write_report(args.output, report)
        report["validation_result_untrained_timing_only"] = dict(validation_result)
        _write_report(args.output, report)

    print(f"PROFILE_JSON={args.output}", flush=True)


if __name__ == "__main__":
    main()
