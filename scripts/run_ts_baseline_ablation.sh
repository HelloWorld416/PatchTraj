#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-./configs/nba.yml}
shift || true

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODELS=(patchformer patchtst timesnet fedformer)

for MODEL in "${MODELS[@]}"; do
  if [[ "${MODEL}" == "patchformer" ]]; then
    FREQ_RECON_LAMBDA=1.0
    NUM_SAMPLE="config"
  else
    FREQ_RECON_LAMBDA=0
    NUM_SAMPLE=10
  fi

  echo "Running time-series baseline: ${MODEL} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, freq_recon_lambda=${FREQ_RECON_LAMBDA}, num_sample=${NUM_SAMPLE}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python main.py \
    --cfg "${CFG}" \
    --mode train \
    --model_name "${MODEL}" \
    --freq_recon_lambda "${FREQ_RECON_LAMBDA}" \
    --exp_name "ts_${MODEL}" \
    "$@"
done
