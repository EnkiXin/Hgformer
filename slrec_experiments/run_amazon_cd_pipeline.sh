#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/experiment_runs/amazon_cd}"
STAGE="${1:-help}"

RUNNER="${REPO_DIR}/run_recbole_gnn.py"
DATASET_FILE="${REPO_DIR}/dataset/Amazon_cd/Amazon_cd.inter"
HGFORMER_CONFIG="${REPO_DIR}/baseline_config_fixed/RecFormer_cd.yaml"
SL_BASE_CONFIG="${REPO_DIR}/baseline_config_fixed/SLRecGraph_cd.yaml"
SL4_CONFIG="${REPO_DIR}/baseline_config_fixed/SLRecGraph_ablation_sl4.yaml"
SL8_CONFIG="${REPO_DIR}/baseline_config_fixed/SLRecGraph_ablation_sl8.yaml"
SL4X4_CONFIG="${REPO_DIR}/baseline_config_fixed/SLRecGraph_ablation_sl4x4.yaml"

EXPECTED_BYTES=152336079
EXPECTED_LINES=3749005
EXPECTED_SHA256=7061471c288df93ba65bfede355aeb013e10dbdfc249db8f20a02bbf8ae031c4
EXPECTED_FILTERED_BYTES=38694878
EXPECTED_FILTERED_LINES=952548
EXPECTED_FILTERED_SHA256=949f7c9443e4548afe17a28159a7407a9f08828119b0967feeb2db826156a146

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/results" "${OUTPUT_ROOT}/checkpoints"

validate_runtime() {
  [[ -f "${RUNNER}" ]] || { echo "Missing runner: ${RUNNER}" >&2; exit 2; }
  [[ -f "${DATASET_FILE}" ]] || { echo "Missing dataset: ${DATASET_FILE}" >&2; exit 2; }

  REPO_DIR="${REPO_DIR}" DATASET_FILE="${DATASET_FILE}" \
    EXPECTED_BYTES="${EXPECTED_BYTES}" EXPECTED_LINES="${EXPECTED_LINES}" \
    EXPECTED_SHA256="${EXPECTED_SHA256}" \
    EXPECTED_FILTERED_BYTES="${EXPECTED_FILTERED_BYTES}" \
    EXPECTED_FILTERED_LINES="${EXPECTED_FILTERED_LINES}" \
    EXPECTED_FILTERED_SHA256="${EXPECTED_FILTERED_SHA256}" \
    "${PYTHON_BIN}" - <<'PY'
import hashlib
import os
from pathlib import Path

path = Path(os.environ["DATASET_FILE"])
expected_size = int(os.environ["EXPECTED_BYTES"])
expected_lines = int(os.environ["EXPECTED_LINES"])
expected_sha = os.environ["EXPECTED_SHA256"]
filtered_size = int(os.environ["EXPECTED_FILTERED_BYTES"])
filtered_lines = int(os.environ["EXPECTED_FILTERED_LINES"])
filtered_sha = os.environ["EXPECTED_FILTERED_SHA256"]

size = path.stat().st_size
with path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
with path.open("rb") as stream:
    lines = sum(1 for _ in stream)

print(f"dataset={path}")
print(f"bytes={size} lines={lines} sha256={digest}")
fingerprint = (size, lines, digest)
if fingerprint == (expected_size, expected_lines, expected_sha):
    print("dataset_variant=full-2014-atomic")
elif fingerprint == (filtered_size, filtered_lines, filtered_sha):
    print("dataset_variant=rating>=3-and-iterative-5-core-preserving-row-order")
else:
    raise SystemExit("Amazon CD input does not match either pinned fingerprint")

import torch
print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()} gpu_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY

  CUDA_VISIBLE_DEVICES="${GPU_ID}" nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version \
    --format=csv,noheader
}

run_case() {
  local run_name="$1"
  local model="$2"
  local config_files="$3"
  shift 3

  local log_file="${OUTPUT_ROOT}/logs/${run_name}.log"
  local result_file="${OUTPUT_ROOT}/results/${run_name}.json"
  echo "[$(date -Is)] starting ${run_name} on physical GPU ${GPU_ID}"

  (
    cd "${REPO_DIR}"
    # Vendored RecBole rewrites CUDA_VISIBLE_DEVICES from gpu_id, so both
    # settings deliberately carry the same physical device index.
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u "${RUNNER}" \
      --model "${model}" \
      --config-files "${config_files}" \
      --gpu_id="${GPU_ID}" \
      --use_gpu=True \
      --show_progress=False \
      --checkpoint_dir="${OUTPUT_ROOT}/checkpoints" \
      --result-file "${result_file}" \
      "$@"
  ) 2>&1 | tee "${log_file}"

  echo "[$(date -Is)] completed ${run_name}"
}

run_hgformer_smoke() {
  run_case hgformer-cd-smoke RecFormer "${HGFORMER_CONFIG}" \
    --epochs=1 --stopping_step=1 --no-save
}

run_hgformer() {
  run_case hgformer-cd-seed2024 RecFormer "${HGFORMER_CONFIG}"
}

run_sl_smoke() {
  run_case sl4-cd-smoke SLRecGraph "${SL_BASE_CONFIG} ${SL4_CONFIG}" \
    --epochs=1 --stopping_step=1 --no-save
  run_case sl8-cd-smoke SLRecGraph "${SL_BASE_CONFIG} ${SL8_CONFIG}" \
    --epochs=1 --stopping_step=1 --no-save
  run_case sl4x4-cd-smoke SLRecGraph "${SL_BASE_CONFIG} ${SL4X4_CONFIG}" \
    --epochs=1 --stopping_step=1 --no-save
}

run_sl4() {
  run_case sl4-cd-seed2024 SLRecGraph "${SL_BASE_CONFIG} ${SL4_CONFIG}"
}

run_sl8() {
  run_case sl8-cd-seed2024 SLRecGraph "${SL_BASE_CONFIG} ${SL8_CONFIG}"
}

run_sl4x4() {
  run_case sl4x4-cd-seed2024 SLRecGraph "${SL_BASE_CONFIG} ${SL4X4_CONFIG}"
}

usage() {
  cat <<'EOF'
Usage: run_amazon_cd_pipeline.sh STAGE

Stages:
  validate          Verify the pinned Amazon CD data and CUDA runtime.
  hgformer-smoke    Run one Hgformer epoch on the full CD protocol.
  hgformer          Run Hgformer (max 500 epochs, patience 30).
  sl-smoke          Run one epoch for SL(4), SL(8), and SL(4)^4.
  sl4               Run SL(4) (max 500 epochs, patience 30).
  sl8               Run SL(8) (max 500 epochs, patience 30).
  sl4x4             Run SL(4)^4 (max 500 epochs, patience 30).

Environment:
  PYTHON_BIN=/path/to/python
  GPU_ID=0  # physical index; passed identically to CVD and RecBole gpu_id
  OUTPUT_ROOT=/persistent/output/directory
EOF
}

case "${STAGE}" in
  validate) validate_runtime ;;
  hgformer-smoke) validate_runtime; run_hgformer_smoke ;;
  hgformer) validate_runtime; run_hgformer ;;
  sl-smoke) validate_runtime; run_sl_smoke ;;
  sl4) validate_runtime; run_sl4 ;;
  sl8) validate_runtime; run_sl8 ;;
  sl4x4) validate_runtime; run_sl4x4 ;;
  help|-h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
