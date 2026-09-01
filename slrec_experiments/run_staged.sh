#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SLREC_PYTHON:-${REPO_DIR}/.venv-slrec/bin/python}"
RUNNER="${REPO_DIR}/slrec_experiments/run_experiment.py"
CONFIG_DIR="${REPO_DIR}/slrec_experiments/configs"
STAGE="${1:-smoke}"
SEEDS="${SLREC_SEEDS:-2024}"
GPU="${SLREC_GPU:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ "${GPU}" == "cpu" ]]; then
  DEVICE_ARGS=(--cpu)
else
  DEVICE_ARGS=(--gpu "${GPU}")
fi

run_one() {
  local model="$1"
  local dataset_config="$2"
  local dataset="$3"
  local seed="$4"
  shift 4
  "${PYTHON_BIN}" "${RUNNER}" \
    --model "${model}" \
    --dataset "${dataset}" \
    --config "${dataset_config}" \
    --seed "${seed}" \
    "${DEVICE_ARGS[@]}" \
    "$@"
}

run_smoke() {
  local seed="${SEEDS%% *}"
  local config="${CONFIG_DIR}/ml-100k-smoke.yaml"
  run_one BPR "${config}" ml-100k "${seed}"
  run_one LightGCN "${config}" ml-100k "${seed}"
  run_one SLRec "${config}" ml-100k "${seed}"
  run_one MixedGeoRec "${config}" ml-100k "${seed}"
  run_one SLRec "${config}" ml-100k "${seed}" \
    --config "${CONFIG_DIR}/slrec-graph.yaml"
  run_one MixedGeoRec "${config}" ml-100k "${seed}" \
    --config "${CONFIG_DIR}/mixedgeo-gated.yaml"
  run_one MixedGeoRec "${config}" ml-100k "${seed}" \
    --config "${CONFIG_DIR}/mixedgeo-dual.yaml"
}

run_pilot() {
  local seed dataset model
  for seed in ${SEEDS}; do
    for dataset in Amazon_toy Amazon_cd; do
      for model in BPR LightGCN SLRec MixedGeoRec; do
        run_one "${model}" "${CONFIG_DIR}/amazon-pilot.yaml" "${dataset}" "${seed}"
      done
      run_one SLRec "${CONFIG_DIR}/amazon-pilot.yaml" "${dataset}" "${seed}" \
        --config "${CONFIG_DIR}/slrec-graph.yaml"
      run_one MixedGeoRec "${CONFIG_DIR}/amazon-pilot.yaml" "${dataset}" "${seed}" \
        --config "${CONFIG_DIR}/mixedgeo-gated.yaml"
      run_one MixedGeoRec "${CONFIG_DIR}/amazon-pilot.yaml" "${dataset}" "${seed}" \
        --config "${CONFIG_DIR}/mixedgeo-dual.yaml"
    done
  done
}

run_main() {
  local seed dataset model config
  for seed in ${SEEDS}; do
    for dataset in Amazon_toy Amazon_cd Amazon_movies DoubanBook DoubanMovie DoubanMusic; do
      if [[ "${dataset}" == Douban* ]]; then
        config="${CONFIG_DIR}/douban-pilot.yaml"
      else
        config="${CONFIG_DIR}/amazon-pilot.yaml"
      fi
      for model in BPR LightGCN SLRec MixedGeoRec; do
        run_one "${model}" "${config}" "${dataset}" "${seed}"
      done
    done
  done
}

case "${STAGE}" in
  smoke) run_smoke ;;
  pilot) run_pilot ;;
  main) run_main ;;
  *) echo "usage: $0 {smoke|pilot|main}" >&2; exit 2 ;;
esac
