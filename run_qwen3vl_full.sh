#!/usr/bin/env bash
# Run Qwen3-VL-32B full evaluation on IU X-Ray val + test splits.
# Runs detached (survives terminal close), saves incrementally per sample,
# and resumes automatically if interrupted.
#
# Usage: bash run_qwen3vl_full.sh
#
# NOTE: Qwen3-VL-32B takes ~25 min/image on a V100.
#   Val (891 images): ~15 days
#   Test (~1,400 images): ~24 days
# Use --max-samples for small demo batches.

set -euo pipefail

cd "/home/doli/Chest Xpert"

ENV_PY="/home/doli/anaconda3/envs/hct-lsr/bin/python"
LOG_DIR="./checkpoints/logs"
mkdir -p "$LOG_DIR"

run_split() {
  local split="$1"
  local out="./checkpoints/iu_xray_${split}_qwen3vl_metrics.json"
  local log="$LOG_DIR/iu_xray_${split}_qwen3vl_run.log"

  echo "[$(date)] Starting Qwen3-VL-32B on $split split -> $out"
  setsid bash -c "
    cd '/home/doli/Chest Xpert'
    exec env CUDA_VISIBLE_DEVICES=0 '$ENV_PY' -u chest_xpert/evaluate_iu_xray.py \
      --data-root ./data/iu_xray \
      --split '$split' \
      --device cuda \
      --llm-model Qwen/Qwen3-VL-32B-Instruct \
      --output '$out'
  " > "$log" 2>&1 < /dev/null &
  echo "[$(date)] Launched $split detached (PID=$!), log: $log"
}

run_split val
run_split test

echo ""
echo "Both splits launched detached. Monitor progress with:"
echo "  tail -f ./checkpoints/logs/iu_xray_val_qwen3vl_run.log"
echo "  tail -f ./checkpoints/logs/iu_xray_test_qwen3vl_run.log"
echo ""
echo "Check completed count:"
echo "  grep -c 'STUDY ID' ./checkpoints/iu_xray_val_qwen3vl_metrics.txt"
echo "  grep -c 'STUDY ID' ./checkpoints/iu_xray_test_qwen3vl_metrics.txt"
echo ""
echo "To resume if interrupted, re-run this same script."
