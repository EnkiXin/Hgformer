#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SLREC_PYTHON:-${REPO_DIR}/.venv-slrec/bin/python}"
RUNNER="${REPO_DIR}/slrec_experiments/run_experiment.py"
CONFIG_DIR="${REPO_DIR}/slrec_experiments/configs"
RESULT_DIR="${REPO_DIR}/slrec_experiments/results"
LOG_DIR="${REPO_DIR}/slrec_experiments/run_logs"

STAGE="${1:-ml100k}"
SEED="${SLREC_SEED:-2024}"
GPU="${SLREC_GPU:-0}"
EPOCHS="${SLREC_EPOCHS:-50}"
STOPPING_STEP="${SLREC_STOPPING_STEP:-10}"
EVAL_STEP="${SLREC_EVAL_STEP:-1}"
TRAIN_BATCH_SIZE="${SLREC_TRAIN_BATCH_SIZE:-}"
SL_EVAL_BATCH_CAP="${SLREC_SL_EVAL_BATCH_CAP:-512}"
MIXED_EVAL_BATCH_CAP="${SLREC_MIXED_EVAL_BATCH_CAP:-1024}"
RUN_TAG="${SLREC_RUN_TAG:-pilot}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

run_one() {
  local label="$1"
  local model="$2"
  local dataset="$3"
  local dataset_config="$4"
  shift 4

  local result_file="${RESULT_DIR}/${dataset}-${label}-seed${SEED}-${RUN_TAG}.json"
  local log_file="${LOG_DIR}/${dataset}-${label}-seed${SEED}-${RUN_TAG}.log"
  local extra_overrides=(--set "eval_step=${EVAL_STEP}")
  if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
    extra_overrides+=(--set "train_batch_size=${TRAIN_BATCH_SIZE}")
  fi
  if [[ "${model}" == "SLRec" ]]; then
    extra_overrides+=(--set "experiment_eval_batch_cap=${SL_EVAL_BATCH_CAP}")
  elif [[ "${model}" == "MixedGeoRec" ]]; then
    extra_overrides+=(--set "experiment_eval_batch_cap=${MIXED_EVAL_BATCH_CAP}")
  fi
  echo "START ${dataset} ${label}"
  if ! "${PYTHON_BIN}" -u "${RUNNER}" \
      --model "${model}" \
      --dataset "${dataset}" \
      --config "${dataset_config}" \
      --seed "${SEED}" \
      --gpu "${GPU}" \
      --epochs "${EPOCHS}" \
      --set "stopping_step=${STOPPING_STEP}" \
      "${extra_overrides[@]}" \
      --result-file "${result_file}" \
      "$@" >"${log_file}" 2>&1; then
    tail -80 "${log_file}" >&2
    return 1
  fi

  "${PYTHON_BIN}" - "${result_file}" "${label}" <<'PY'
import json
import sys

path, label = sys.argv[1:]
result = json.load(open(path, encoding="utf-8"))
test = result["test_result"]
fields = ("recall@10", "ndcg@10", "recall@20", "recall@50", "ndcg@50")
summary = " ".join(f"{field}={test[field]:.4f}" for field in fields)
print(f"DONE {label} {summary}")
PY
}

run_matrix() {
  local dataset="$1"
  local dataset_config="$2"

  run_one bpr BPR "${dataset}" "${dataset_config}"
  run_one lightgcn LightGCN "${dataset}" "${dataset_config}"
  run_one slrec SLRec "${dataset}" "${dataset_config}"
  run_one slrec-graph SLRec "${dataset}" "${dataset_config}" \
    --config "${CONFIG_DIR}/slrec-graph.yaml"
  run_one mixed-hes MixedGeoRec "${dataset}" "${dataset_config}"
  run_one mixed-hes-gated MixedGeoRec "${dataset}" "${dataset_config}" \
    --config "${CONFIG_DIR}/mixedgeo-gated.yaml"
  run_one mixed-he MixedGeoRec "${dataset}" "${dataset_config}" \
    --config "${CONFIG_DIR}/mixedgeo-dual.yaml"
}

case "${STAGE}" in
  ml100k)
    run_matrix ml-100k "${CONFIG_DIR}/ml-100k-smoke.yaml"
    ;;
  doubanbook)
    run_matrix DoubanBook "${CONFIG_DIR}/douban-pilot.yaml"
    ;;
  doubanmusic)
    run_matrix DoubanMusic "${CONFIG_DIR}/douban-pilot.yaml"
    ;;
  *)
    echo "usage: $0 {ml100k|doubanbook|doubanmusic}" >&2
    exit 2
    ;;
esac
