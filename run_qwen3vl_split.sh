#!/usr/bin/env bash
# Run Qwen3-VL-32B on ONE IU X-Ray split (val or test).
# Detached, incremental per-sample save, auto-resume if interrupted.
#
# NOTE: Qwen3-VL-32B takes ~25 min/image on a V100.
#   Val (891 images): ~15 days   |   Test (~1,400 images): ~24 days
#
# Usage:
#   bash run_qwen3vl_split.sh val       # run val split
#   bash run_qwen3vl_split.sh test      # run test split (after val completes)

set -euo pipefail

SPLIT="${1:-val}"
if [[ "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then
  echo "Usage: $0 {val|test}"
  exit 1
fi

cd "/home/doli/Chest Xpert"

ENV_PY="/home/doli/anaconda3/envs/hct-lsr/bin/python"
OUT="./checkpoints/iu_xray_${SPLIT}_qwen3vl_metrics.json"
LOG="./checkpoints/logs/iu_xray_${SPLIT}_qwen3vl_run.log"
mkdir -p ./checkpoints/logs

if [[ -f "$OUT" ]]; then
  echo "Existing output found: $OUT (will resume from where it left off)."
fi

echo "[$(date)] Launching Qwen3-VL-32B on '$SPLIT' split -> $OUT"
setsid bash -c "
  cd '/home/doli/Chest Xpert'
  exec env CUDA_VISIBLE_DEVICES=0 '$ENV_PY' -u chest_xpert/evaluate_iu_xray.py \
    --data-root ./data/iu_xray \
    --split '$SPLIT' \
    --device cuda \
    --llm-model Qwen/Qwen3-VL-32B-Instruct \
    --output '$OUT'
" > "$LOG" 2>&1 < /dev/null &

echo "[$(date)] Launched detached. Log: $LOG"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG"
echo ""
echo "Check completed count:"
echo "  grep -c 'STUDY ID' ./checkpoints/iu_xray_${SPLIT}_qwen3vl_metrics.txt"
echo ""
echo "To resume if interrupted, re-run: bash $0 $SPLIT"
