#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-./configs/sdd.yml}
shift || true

MODES=(concat freq_to_time time_to_freq bidirectional)

for MODE in "${MODES[@]}"; do
  echo "Running fusion ablation: ${MODE}"
  python main.py \
    --cfg "${CFG}" \
    --mode train \
    --fusion_mode "${MODE}" \
    --exp_name "fusion_${MODE}" \
    "$@"
done
