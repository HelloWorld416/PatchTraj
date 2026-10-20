#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-./configs/nba.yml}
shift || true

PATCH_CSV=$(python - "${CFG}" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
match = re.search(r"^\s*patch_list\s*:\s*\[([^\]]+)\]", text, re.M)
if not match:
    raise SystemExit(f"Cannot find patch_list in {sys.argv[1]}")
values = [v.strip() for v in match.group(1).split(",") if v.strip()]
print(",".join(values))
PY
)

IFS=',' read -r -a PATCHES <<< "${PATCH_CSV}"
N=${#PATCHES[@]}

LABELS=()
SETS=()
MODES=()

add_variant() {
  LABELS+=("$1")
  SETS+=("$2")
  MODES+=("$3")
}

for PATCH in "${PATCHES[@]}"; do
  add_variant "single_${PATCH}_nonoverlap" "${PATCH}" "nonoverlap"
done

if (( N >= 2 )); then
  FIRST=${PATCHES[0]}
  LAST=${PATCHES[$((N - 1))]}

  if (( N >= 3 )); then
    MID=${PATCHES[$((N / 2))]}
    add_variant "multi_${FIRST}_${MID}_nonoverlap" "${FIRST},${MID}" "nonoverlap"
    add_variant "multi_${MID}_${LAST}_nonoverlap" "${MID},${LAST}" "nonoverlap"
    add_variant "multi_${FIRST}_${LAST}_nonoverlap" "${FIRST},${LAST}" "nonoverlap"
  fi

  add_variant "multi_${PATCH_CSV//,/_}_nonoverlap" "${PATCH_CSV}" "nonoverlap"
  add_variant "multi_${PATCH_CSV//,/_}_overlap" "${PATCH_CSV}" "overlap"
fi

for IDX in "${!LABELS[@]}"; do
  echo "Running patch-scale ablation: ${LABELS[$IDX]} (${MODES[$IDX]}, patch_list=${SETS[$IDX]})"
  python main.py \
    --cfg "${CFG}" \
    --mode train \
    --patch_list "${SETS[$IDX]}" \
    --patch_stride_mode "${MODES[$IDX]}" \
    --exp_name "scale_${LABELS[$IDX]}" \
    "$@"
done
