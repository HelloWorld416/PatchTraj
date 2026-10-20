#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-./configs/sdd.yml}
shift || true

LAMBDAS=(0 0.1 0.5 1.0)

for LAMBDA in "${LAMBDAS[@]}"; do
  echo "Running frequency reconstruction ablation: lambda=${LAMBDA}"
  python main.py \
    --cfg "${CFG}" \
    --mode train \
    --freq_recon_lambda "${LAMBDA}" \
    --exp_name "lambda_${LAMBDA}" \
    "$@"
done
