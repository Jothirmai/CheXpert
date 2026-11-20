"""Queue CheXpert Plus VLM fine-tuning after the active IU experiment."""
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
DATA = ROOT / 'data/chexpert_plus_vlm_v1'
IU = ROOT / 'checkpoints/report_lora_selected_cnn_v1'
OUTPUT = ROOT / 'checkpoints/report_lora_chexpert_plus_v1'
OUTPUT.mkdir(parents=True, exist_ok=True)
lock = (OUTPUT / 'controller.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
STATUS = OUTPUT / 'completion_status.json'


def update(phase, **kwargs):
    state = {'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(), **kwargs}
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(STATUS)
    print('STATUS', json.dumps(state), flush=True)


def wait_for_iu():
    launcher = json.loads((IU / 'launcher.json').read_text())
    pid = launcher['pid']
    while True:
        path = IU / 'completion_status.json'
        state = json.loads(path.read_text()) if path.is_file() else {}
        if state.get('phase') in ('complete', 'failed') and state.get('backend_paused') is False:
            return state['phase']
        if not Path(f'/proc/{pid}').exists() and state.get('phase') not in ('complete', 'failed'):
            raise RuntimeError('IU controller exited without a terminal status')
        time.sleep(30)


def stop_backend():
    runtime = ROOT / 'reports/backend_runtime.json'
    if not runtime.is_file():
        return None
    pid = json.loads(runtime.read_text()).get('pid')
    proc = Path(f'/proc/{pid}')
    if not proc.exists() or b'backend.main:app' not in (proc / 'cmdline').read_bytes():
        return None
    env = dict(item.split('=', 1) for item in (proc / 'environ').read_text().split('\0') if '=' in item)
    request = urllib.request.Request('http://127.0.0.1:8000/api/v1/admin/diagnostics')
    if env.get('CHESTXPERT_API_KEY'):
        request.add_header('X-API-Key', env['CHESTXPERT_API_KEY'])
    for _ in range(30):
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = json.load(response)
            break
        except Exception:
            time.sleep(5)
    else:
        raise RuntimeError('Backend diagnostics unavailable after IU run')
    if status['active_jobs'] or status['queued_jobs']:
        raise RuntimeError('Backend has active analyses')
    os.kill(pid, signal.SIGTERM)
    for _ in range(200):
        if not proc.exists() or (proc / 'stat').read_text().split()[2] == 'Z':
            return env
        time.sleep(.1)
    raise RuntimeError('Backend did not stop cleanly')


def run(command, env):
    child = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT, env=env)
    (OUTPUT / 'child_process.json').write_text(json.dumps({'pid': child.pid, 'command': command}, indent=2))
    if child.wait():
        raise RuntimeError(f'{command[0]} failed')


def main():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        import hashlib
        if hashlib.sha256((DATA / f'{name}.jsonl').read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen CheXpert Plus {name} manifest changed')
    active = json.loads((ROOT / 'checkpoints/active_models.json').read_text())['cnn']
    if active.get('architecture') != 'densenet201_v6_v12_tta':
        raise RuntimeError('Selected CNN changed')
    update('queued_after_iu', train_studies=manifest['counts']['train'],
           validation_studies=manifest['counts']['validation'],
           reserved_test_studies=manifest['counts']['test'])
    wait_for_iu()
    backend_env = stop_backend()
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONPATH': str(ROOT),
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
    try:
        update('training', backend_paused=backend_env is not None,
               cnn_architecture=active['architecture'])
        run(['scripts/train_report_adapter.py', '--offline',
             '--cnn-evidence', 'selected_ensemble', '--train', str(DATA / 'train.jsonl'),
             '--validation', str(DATA / 'validation.jsonl'), '--output', str(OUTPUT),
             '--cache', 'data/report_training_v3/evidence', '--train-limit', '2048',
             '--validation-limit', '96', '--sampling', 'first',
             '--validation-sampling', 'first', '--steps', '256', '--epochs', '1',
             '--grad-accum', '8', '--rank', '8', '--lr', '0.00003',
             '--max-pixels', str(256*28*28), '--generate-examples', '24',
             '--precision', 'fp16', '--checkpoint-every', '32'], env)
        update('scoring_validation', backend_paused=backend_env is not None)
        run(['scripts/score_vlm_pair_standard.py', '--experiment', str(OUTPUT), '--bert-score'], env)
        if (IU / 'evaluation.json').is_file():
            run(['scripts/score_vlm_pair_standard.py', '--experiment', str(IU), '--bert-score'], env)
        update('scoring_selected_cnn_all14', backend_paused=backend_env is not None)
        if not (ROOT / 'reports/selected_ensemble_all14_development.json').exists():
            run(['scripts/evaluate_selected_ensemble_all14.py'], env)
        run(['scripts/build_current_comparison_tables.py'], env)
        update('complete', backend_paused=backend_env is not None,
               adapter_auto_activated=False,
               metrics=str(OUTPUT / 'standard_validation_metrics.json'),
               reserved_test_evaluated=False)
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
            (ROOT / 'reports/backend_runtime.json').write_text(json.dumps({
                'pid': child.pid, 'started_at': datetime.now(timezone.utc).isoformat(),
                'log': str(ROOT / 'reports/backend_required_evidence.log'),
                'required_report_inputs': ['cnn_scores','original_xray','gradcam_contact_sheet']}, indent=2))
            state = json.loads(STATUS.read_text())
            state.update(backend_paused=False, backend_pid=child.pid)
            STATUS.write_text(json.dumps(state, indent=2) + '\n')
            print('BACKEND_RESTORED', child.pid, flush=True)


if __name__ == '__main__':
    main()
