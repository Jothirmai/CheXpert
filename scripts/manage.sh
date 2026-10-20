#!/usr/bin/env bash
#
# Chest Xpert — Backend management scripts
#
# Source the hct-lsr environment before running:
#   source /home/doli/anaconda3/bin/activate hct-lsr
#
# Usage: ./scripts/manage.sh <command> [args]
#
set -euo pipefail

PROJECT_DIR="/home/doli/Chest Xpert"
cd "$PROJECT_DIR"

PORT="${CHESTXPERT_PORT:-8000}"
PID_FILE="/tmp/chestxpert_backend.pid"
LOG_FILE="/tmp/chestxpert_backend.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
is_running() {
    ss -tlnp | grep -q ":${PORT} "
}

get_pid() {
    ss -tlnp | grep ":${PORT} " | awk '{print $NF}' | cut -d'=' -f2 | cut -d'/' -f1 | head -1
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_gpu() {
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv 2>/dev/null || \
        echo "nvidia-smi not available"
    echo ""
    echo "=== GPU Processes ==="
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=',' read -r pid name mem; do
            echo "PID: ${pid} | ${name} | ${mem} MiB"
        done || echo "No GPU processes or nvidia-smi unavailable."
}

cmd_ram() {
    echo "=== System RAM ==="
    free -h
    echo ""
    echo "=== Top memory processes ==="
    ps aux --sort=-%mem | head -10
}

cmd_disk() {
    echo "=== Disk Usage ==="
    df -h / /home 2>/dev/null
    echo ""
    echo "=== Chest Xpert directories ==="
    du -sh "$PROJECT_DIR/temp" "$PROJECT_DIR/reports" "$HOME/.torchxrayvision/models_data" 2>/dev/null || true
}

cmd_cache() {
    echo "=== Model cache sizes ==="
    python -c "
from backend.services.cleanup_service import cleanup_service
from backend.services.resource_manager import MONITOR
sizes = cleanup_service.cache_sizes()
for k, v in sizes.items():
    print(f'{k}: {v} MB')
disk = MONITOR.disk_usage()
for k, v in disk.items():
    if isinstance(v, dict) and 'error' not in v:
        print(f'{k} disk: {v[\"free_gb\"]} GB free / {v[\"total_gb\"]} GB total')
"
}

cmd_cleanup() {
    local dry_run="${1:-true}"
    local dry_run_py="True"
    if [ "$dry_run" = "false" ]; then dry_run_py="False"; fi
    echo "=== Cleanup (dry_run=$dry_run) ==="
    python -c "
from backend.services.cleanup_service import cleanup_service
import json
result = cleanup_service.run(ttl_s=None, dry_run=$dry_run_py)
print(json.dumps(result, indent=2, default=str))
"
}

cmd_test_classifier() {
    echo "=== Testing classifier alone ==="
    python -c "
import time
import torch
import torchxrayvision as xrv
from chest_xpert.config import DEVICE_CONFIG, MODEL_CONFIG
from chest_xpert.inference import InferenceEngine
from chest_xpert.preprocessing import preprocess
from pathlib import Path

device = DEVICE_CONFIG.resolve()
print(f'Device: {device}')
engine = InferenceEngine(model_name=MODEL_CONFIG.name, device=device)
t0 = time.perf_counter()
_ = engine.model
load_ms = (time.perf_counter() - t0) * 1000
print(f'Model load time: {load_ms:.1f} ms')

sample = Path('samples/sample_frontal.png')
t0 = time.perf_counter()
pp = preprocess(sample, device)
result = engine.predict(pp.tensor)
inf_ms = (time.perf_counter() - t0) * 1000
print(f'Inference time: {inf_ms:.1f} ms')
print(f'Labels supported: {len(result.supported_labels)}/18')
print(f'Sample scores: Atelectasis={result.scores.get(\"Atelectasis\", 0):.4f}, Effusion={result.scores.get(\"Effusion\", 0):.4f}')
print('Classifier test: PASSED')
"
}

cmd_test_gradcam() {
    echo "=== Testing Grad-CAM ==="
    python -c "
import time
import numpy as np
from PIL import Image as PILImage
from chest_xpert.config import DEVICE_CONFIG, MODEL_CONFIG
from chest_xpert.inference import InferenceEngine
from chest_xpert.preprocessing import preprocess
from chest_xpert.explainability import GradCAM
from pathlib import Path

device = DEVICE_CONFIG.resolve()
engine = InferenceEngine(model_name=MODEL_CONFIG.name, device=device)
_ = engine.model

sample = Path('samples/sample_frontal.png')
pp = preprocess(sample, device)
result = engine.predict(pp.tensor)

disp = pp.tensor.detach().cpu().numpy()[0, 0]
disp = np.clip((disp + 1024) / 2048, 0, 1)

mappings_support = [m for m in __import__('chest_xpert.label_mapping', fromlist=['map_outputs_to_labels']).map_outputs_to_labels(result.raw, result.model_name) if m.supported]
target = mappings_support[0]

t0 = time.perf_counter()
cam = GradCAM(engine.model, target_layer_name='features.norm5')
grad = cam.generate(
    input_tensor=pp.tensor,
    target_label=target.requested_label,
    model_outputs=result.model_outputs,
    original_image=disp,
)
cam_ms = (time.perf_counter() - t0) * 1000
cam.close()

out = Path('temp/test_gradcam.png')
out.parent.mkdir(parents=True, exist_ok=True)
PILImage.fromarray(grad.overlay).save(out)
print(f'Grad-CAM time: {cam_ms:.1f} ms')
print(f'Target: {target.requested_label}')
print(f'Saved: {out}')
print('Grad-CAM test: PASSED')
"
}

cmd_test_report() {
    echo "=== Testing report generator ==="
    python -c "
from chest_xpert.report import ReportData, ExamDetails, generate_html, generate_pdf
from chest_xpert.label_mapping import map_outputs_to_labels
from chest_xpert.config import MODEL_CONFIG
from pathlib import Path
import tempfile

rd = ReportData()
rd.exam = ExamDetails(patient_id='TEST-001', modality='Chest X-ray')
rd.model_name = MODEL_CONFIG.name
rd.mappings = map_outputs_to_labels({}, MODEL_CONFIG.name)
rd.unsupported_labels = []
rd.impression = 'Test impression.'
rd.reviewer_name = 'Test Reviewer'

html = generate_html(rd)
assert 'AI-generated research draft' in html
print('HTML report: PASSED')

out = Path(tempfile.mkdtemp()) / 'test_report.pdf'
generate_pdf(rd, out)
assert out.exists() and out.stat().st_size > 0
print(f'PDF report: PASSED ({out.stat().st_size} bytes)')
"
}

cmd_test_e2e() {
    echo "=== End-to-end pipeline test ==="
    python -c "
import asyncio
from backend.services.inference_service import inference_service
from backend.config import STORAGE_CONFIG
from pathlib import Path

async def main():
    ok, err = await inference_service.ensure_model_loaded()
    if not ok:
        print(f'Model load failed: {err}')
        return
    print('Model loaded.')

    result = inference_service._run_pipeline_sync(
        analysis_id='e2e-test',
        image_path='samples/sample_frontal.png',
        request_data={'patient_id': 'E2E-001'},
        generate_grad_cam=True,
    )
    print(f'Status: {result[\"status\"]}')
    print(f'Modality: {result[\"modality\"][\"decision\"]}')
    print(f'Labels: {len(result[\"labels\"])}/18')
    print(f'Inference: {result[\"inference_latency_ms\"]} ms')
    print(f'Total: {result[\"total_latency_ms\"]} ms')
    print(f'Report available: {result[\"report_available\"]}')
    print(f'Grad-CAM: {result[\"grad_cam_available\"]}')
    print('E2E test: PASSED')

asyncio.run(main())
"
}

cmd_start() {
    if is_running; then
        echo "Backend already running on port $PORT"
        return 0
    fi
    echo "Starting backend on port $PORT..."
    nohup python -m uvicorn backend.main:app --host 0.0.0.0 --port "$PORT" \
        >"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    disown
    sleep 5
    if is_running; then
        echo "Backend started (PID: $(cat "$PID_FILE"))"
        echo "Health: http://localhost:$PORT/api/v1/health"
        echo "Docs: http://localhost:$PORT/docs"
    else
        echo "Failed to start. Check $LOG_FILE"
        return 1
    fi
}

cmd_stop() {
    if ! is_running; then
        echo "Backend not running on port $PORT"
        return 0
    fi
    local pid
    pid=$(get_pid)
    echo "Stopping backend (PID: $pid)..."
    kill "$pid" 2>/dev/null || true
    sleep 2
    if is_running; then
        kill -9 "$pid" 2>/dev/null || true
    fi
    if ! is_running; then
        echo "Backend stopped."
    else
        echo "Failed to stop backend."
        return 1
    fi
}

cmd_restart() {
    cmd_stop
    sleep 2
    cmd_start
}

cmd_status() {
    if is_running; then
        echo "Backend is running on port $PORT (PID: $(get_pid))"
        curl -s --max-time 5 "http://localhost:$PORT/api/v1/health" 2>/dev/null || echo "Health check failed"
    else
        echo "Backend is not running on port $PORT"
    fi
}

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
cmd="${1:-help}"
shift || true

case "$cmd" in
    gpu)       cmd_gpu ;;
    ram)       cmd_ram ;;
    disk)      cmd_disk ;;
    cache)     cmd_cache ;;
    cleanup)   cmd_cleanup "${1:-true}" ;;
    test-classifier) cmd_test_classifier ;;
    test-gradcam)    cmd_test_gradcam ;;
    test-report)     cmd_test_report ;;
    test-e2e)        cmd_test_e2e ;;
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    restart)   cmd_restart ;;
    status)    cmd_status ;;
    help|*)
        echo "Chest Xpert Backend Management"
        echo ""
        echo "Usage: $0 <command> [args]"
        echo ""
        echo "Resource commands:"
        echo "  gpu          Show GPU VRAM and processes"
        echo "  ram          Show system RAM usage"
        echo "  disk         Show disk usage"
        echo "  cache        Show model cache and directory sizes"
        echo "  cleanup [dry_run]  Clean expired artifacts (default: dry-run)"
        echo ""
        echo "Test commands:"
        echo "  test-classifier  Test DenseNet-121 inference alone"
        echo "  test-gradcam     Test Grad-CAM generation"
        echo "  test-report      Test HTML/PDF report generation"
        echo "  test-e2e         Run full end-to-end pipeline test"
        echo ""
        echo "Server commands:"
        echo "  start          Start the FastAPI backend"
        echo "  stop           Stop the FastAPI backend"
        echo "  restart        Restart the FastAPI backend"
        echo "  status         Check backend status"
        ;;
esac
