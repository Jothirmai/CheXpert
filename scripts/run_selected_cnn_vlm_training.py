"""Run the selected-CNN VLM fine-tune and restore the local backend afterward."""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
SOURCE = ROOT / 'checkpoints/report_lora_full_study_v2'
OUTPUT = ROOT / 'checkpoints/report_lora_selected_cnn_v1'
OUTPUT.mkdir(parents=True, exist_ok=True)
lock = (OUTPUT / 'controller.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
STATUS = OUTPUT / 'completion_status.json'


def update(phase, **kwargs):
    payload = {'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(), **kwargs}
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(STATUS)
    print('STATUS', json.dumps(payload), flush=True)


def run(command, env):
    child = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT, env=env)
    (OUTPUT / 'child_process.json').write_text(json.dumps({'pid': child.pid, 'command': command}, indent=2))
    code = child.wait()
    if code:
        raise RuntimeError(f'{command[0]} exited {code}')


def main():
    active = json.loads((ROOT / 'checkpoints/active_models.json').read_text())['cnn']
    if active.get('architecture') != 'densenet201_v6_v12_tta':
        raise RuntimeError('Selected CNN ensemble changed before VLM training')
    for name in ('train.jsonl', 'validation.jsonl', 'cohort_audit.json', 'reserved_test_ids.json'):
        (OUTPUT / name).write_bytes((SOURCE / name).read_bytes())
    backend_env = None
    backend_pid = None
    runtime = ROOT / 'reports/backend_runtime.json'
    if runtime.exists():
        backend_pid = json.loads(runtime.read_text()).get('pid')
        proc = Path(f'/proc/{backend_pid}')
        if proc.exists() and b'backend.main:app' in (proc / 'cmdline').read_bytes():
            backend_env = dict(item.split('=', 1) for item in (proc / 'environ').read_text().split('\0') if '=' in item)
            request = urllib.request.Request('http://127.0.0.1:8000/api/v1/admin/diagnostics')
            if backend_env.get('CHESTXPERT_API_KEY'):
                request.add_header('X-API-Key', backend_env['CHESTXPERT_API_KEY'])
            with urllib.request.urlopen(request, timeout=10) as response:
                diagnostics = json.load(response)
            if diagnostics['active_jobs'] or diagnostics['queued_jobs']:
                raise RuntimeError('Backend has active analyses; retry training after they finish')
            os.kill(backend_pid, signal.SIGTERM)
            for _ in range(200):
                if not proc.exists() or (proc / 'stat').read_text().split()[2] == 'Z':
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('Backend did not stop cleanly')
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONPATH': str(ROOT),
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
    try:
        update('training', backend_paused=backend_env is not None,
               cnn_architecture=active['architecture'], controller_pid=os.getpid())
        run(['scripts/train_report_adapter.py', '--offline',
             '--cnn-evidence', 'selected_ensemble',
             '--output', str(OUTPUT), '--train', str(OUTPUT / 'train.jsonl'),
             '--validation', str(OUTPUT / 'validation.jsonl'),
             '--cache', 'data/report_training_v3/evidence', '--train-limit', '512',
             '--validation-limit', '24', '--sampling', 'first',
             '--validation-sampling', 'first', '--steps', '128', '--epochs', '2',
             '--grad-accum', '8', '--rank', '8', '--lr', '0.00003',
             '--max-pixels', str(256*28*28), '--generate-examples', '12',
             '--precision', 'fp16', '--checkpoint-every', '16'], env)
        update('assessing_validation', backend_paused=backend_env is not None)
        run(['scripts/assess_full_study_adapter.py', '--experiment', str(OUTPUT)], env)
        assessment = json.loads((OUTPUT / 'assessment.json').read_text())
        update('complete', backend_paused=backend_env is not None,
               eligible_for_experimental_selection=assessment['eligible_for_experimental_selection'],
               adapter_auto_activated=False,
               comparison=str(OUTPUT / 'COMPARISON.md'),
               assessment=str(OUTPUT / 'assessment.json'))
    except BaseException as error:
        update('failed', error=repr(error), backend_paused=backend_env is not None)
        raise
    finally:
        if backend_env is not None:
            backend_env.pop('CHESTXPERT_LORA_ENABLED', None)
            backend_env.pop('CHESTXPERT_LORA_ADAPTER_PATH', None)
            with (ROOT / 'reports/backend_required_evidence.log').open('a') as log:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'uvicorn',
                    'backend.main:app', '--host', '0.0.0.0', '--port', '8000'],
                    cwd=ROOT, env=backend_env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            runtime.write_text(json.dumps({'pid': child.pid,
                'started_at': datetime.now(timezone.utc).isoformat(),
                'log': str(ROOT / 'reports/backend_required_evidence.log'),
                'required_report_inputs': ['cnn_scores', 'original_xray', 'gradcam_contact_sheet']}, indent=2))
            payload = json.loads(STATUS.read_text())
            payload.update(backend_paused=False, backend_pid=child.pid)
            STATUS.write_text(json.dumps(payload, indent=2) + '\n')
            print('BACKEND_RESTORED', child.pid, flush=True)


if __name__ == '__main__':
    main()
