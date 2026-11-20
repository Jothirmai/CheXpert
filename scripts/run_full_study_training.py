"""Durable training controller: conditional selection, reserved test, backend restore."""
import fcntl,hashlib,json,os,signal,subprocess,sys,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
OUT=ROOT/'checkpoints/report_lora_full_study'
lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=OUT/'completion_status.json';runtime=ROOT/'reports/backend_runtime.json'
def save(phase,**kwargs):
 data={'phase':phase,'updated_at':datetime.now(timezone.utc).isoformat(),**kwargs};tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(status)
def run(command,env):
 child=subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,env=env)
 (OUT/'child_process.json').write_text(json.dumps({'pid':child.pid,'command':command},indent=2))
 code=child.wait()
 if code:raise RuntimeError(f'{command[0]} exited {code}')
pid=json.loads(runtime.read_text())['pid'];proc=Path(f'/proc/{pid}')
assert (proc/'cwd').resolve()==ROOT and 'backend.main:app' in (proc/'cmdline').read_bytes().decode().split('\0')
backend_env=dict(x.split('=',1) for x in (proc/'environ').read_text().split('\0') if '=' in x)
request=urllib.request.Request('http://127.0.0.1:8000/api/v1/admin/diagnostics')
if backend_env.get('CHESTXPERT_API_KEY'):request.add_header('X-API-Key',backend_env['CHESTXPERT_API_KEY'])
with urllib.request.urlopen(request,timeout=10) as r:d=json.load(r)
assert d['active_jobs']==0 and d['queued_jobs']==0,'Active analyses prevent training startup'
previous=json.loads((ROOT/'checkpoints/active_models.json').read_text());selected=False;completed=False
os.kill(pid,signal.SIGTERM)
for _ in range(200):
 try:
  if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(.1)
else:raise RuntimeError('Backend failed graceful shutdown')
env={**os.environ,'HF_HUB_OFFLINE':'1','PYTHONPATH':str(ROOT),'CHESTXPERT_LORA_ENABLED':'false','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
try:
 save('training',backend_paused=True,controller_pid=os.getpid())
 run(['scripts/train_report_adapter.py','--offline','--output',str(OUT),'--train',str(OUT/'train.jsonl'),'--validation',str(OUT/'validation.jsonl'),'--cache','data/report_training_v3/evidence','--train-limit','512','--validation-limit','24','--sampling','first','--validation-sampling','first','--steps','64','--epochs','1','--grad-accum','8','--rank','8','--lr','0.00005','--max-pixels',str(256*28*28),'--generate-examples','12','--precision','fp16','--checkpoint-every','8'],env)
 for name,digest in json.loads((OUT/'protocol.json').read_text())['source_sha256'].items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,f'Protocol source changed during training: {name}'
 save('assessing_validation',backend_paused=True)
 run(['scripts/assess_full_study_adapter.py','--experiment',str(OUT)],env)
 assessment=json.loads((OUT/'assessment.json').read_text());selection_path=ROOT/'checkpoints/active_models.json'
 if assessment['eligible_for_experimental_selection']:
  active=json.loads(selection_path.read_text());evidence=json.loads((OUT/'input_evidence.json').read_text())
  assert active==previous,'Active models changed during training; refusing automatic replacement'
  assert active['cnn']['sha256']==evidence['cnn_sha256']
  # Verify candidate through the actual application before persisting selection.
  candidate_env={**env,'CHESTXPERT_LORA_ENABLED':'true','CHESTXPERT_LORA_ADAPTER_PATH':str(OUT)}
  save('verifying_candidate_application',backend_paused=True)
  run(['scripts/verify_application.py','--output',str(OUT)],candidate_env)
  verification=json.loads((OUT/'application_verification.json').read_text())
  assert verification['passed'] and verification['adapter_loaded'] and verification['within_two_minute_target'],verification
  active['report_adapter']={'path':str(OUT.relative_to(ROOT)),'status':'selected_experimental','base_model':'Qwen/Qwen2.5-VL-7B-Instruct','selection_basis':'Fixed full-study validation: lower loss, fewer missed and unsupported facts (or maintained zero unsupported facts), no category regression or increase in contradictions; application check passed','limitations':'Weak lexical assertion proxies on 12 validation studies, not clinical validation.'}
  tmp=selection_path.with_suffix('.tmp');tmp.write_text(json.dumps(active,indent=2));tmp.replace(selection_path);selected=True
  save('reserved_test_evaluation',backend_paused=True,adapter_selected_experimentally=True)
  run(['scripts/evaluate_full_study_adapter_test.py','--experiment',str(OUT)],candidate_env)
 run(['scripts/assess_full_study_adapter.py','--experiment',str(OUT)],env)
 completed=True
 save('complete',adapter_selected_experimentally=selected,assessment=str(OUT/'assessment.json'),comparisons=str(OUT/'COMPARISON.md'),reserved_test_evaluated=selected,backend_paused=True)
except BaseException as exc:
 # A failed controller must leave the previously functioning model selected.
 if selected:
  path=ROOT/'checkpoints/active_models.json';active=json.loads(path.read_text())
  if (active.get('report_adapter') or {}).get('path')==str(OUT.relative_to(ROOT)):
   active['report_adapter']=previous.get('report_adapter');tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(active,indent=2));tmp.replace(path)
  selected=False
 save('failed',error=str(exc),adapter_selected_experimentally=False,backend_paused=True)
 raise
finally:
 if selected:
  backend_env.pop('CHESTXPERT_LORA_ENABLED',None);backend_env.pop('CHESTXPERT_LORA_ADAPTER_PATH',None)
 with (ROOT/'reports/backend_required_evidence.log').open('a') as log:
  child=subprocess.Popen([sys.executable,'-u','-m','uvicorn','backend.main:app','--host','0.0.0.0','--port','8000'],cwd=ROOT,env=backend_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 runtime.write_text(json.dumps({'pid':child.pid,'started_at':datetime.now(timezone.utc).isoformat(),'log':str(ROOT/'reports/backend_required_evidence.log'),'required_report_inputs':['cnn_scores','all_original_xrays','gradcam_contact_sheet']},indent=2))
 data=json.loads(status.read_text());data.update(backend_paused=False,backend_pid=child.pid);status.write_text(json.dumps(data,indent=2));print('BACKEND_RESTORED',child.pid,flush=True)

 ready=False
 for _ in range(90):
  try:
   request=urllib.request.Request('http://127.0.0.1:8000/api/v1/ready')
   if backend_env.get('CHESTXPERT_API_KEY'):request.add_header('X-API-Key',backend_env['CHESTXPERT_API_KEY'])
   with urllib.request.urlopen(request,timeout=3) as response:ready=json.load(response).get('ready',False)
   if ready:break
  except Exception:pass
  time.sleep(1)
 data=json.loads(status.read_text());data['backend_ready']=ready
 if not ready:data.update(phase='backend_restore_failed',backend_error='Restored process did not pass readiness')
 status.write_text(json.dumps(data,indent=2))
