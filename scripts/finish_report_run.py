"""Complete a long report experiment after its training process exits.

Writes real comparisons and conservative assessment. Selection is experimental
and conditional; no promise of clinical accuracy or perfect reports.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.assess_report_adapter import assess

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--training-pid',type=int,required=True);p.add_argument('--select-if-passes',action='store_true');args=p.parse_args()
experiment=Path(args.experiment).resolve();status=experiment/'completion_status.json';last=None
while not (experiment/'evaluation.json').exists():
    try:os.kill(args.training_pid,0)
    except ProcessLookupError:
        status.write_text(json.dumps({'phase':'training_failed','reason':'Training exited without evaluation.json'},indent=2));raise SystemExit('Training did not complete; no adapter selected')
    progress_path=experiment/'progress.json'
    try:progress=json.loads(progress_path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):progress={'phase':'starting'}
    if progress!=last:
        last=progress;status.write_text(json.dumps({'phase':'waiting_for_training','progress':progress},indent=2));print('PROGRESS',progress.get('phase'),progress.get('step'),flush=True)
    time.sleep(30)
assessment=assess(experiment/'evaluation.json')
evaluation=json.loads((experiment/'evaluation.json').read_text())
lines=['# Report adapter comparison','',f"Base model: {evaluation['base_model']}",f"Base validation token loss: {evaluation['base_validation_loss']:.4f}",f"Adapter validation token loss: {evaluation['adapter_validation_loss']:.4f}",'','This is experimental, not expert clinical validation.','']
for case in evaluation['comparisons']:
    lines.extend([f"## {case['study_id']}",'','### Source clinical reference','',json.dumps(case['reference'],indent=2),'','### Base output','',case['base_output'],'','### Adapter output','',case['adapter_output'],''])
(experiment/'COMPARISON.md').write_text('\n'.join(lines))
selected=False
previous_adapter=None
if args.select_if_passes and assessment['eligible_for_experimental_selection']:
    active_path=ROOT/'checkpoints/active_models.json';active=json.loads(active_path.read_text())
    previous_adapter=active.get('report_adapter')
    evidence=json.loads((experiment/'input_evidence.json').read_text())
    if active['cnn']['sha256']!=evidence['cnn_sha256']:
        raise RuntimeError('Active CNN differs from the model used for training evidence; refusing adapter selection')
    active['report_adapter']={'path':str(experiment.relative_to(ROOT)),'status':'selected_experimental','base_model':evaluation['base_model'],'selection_basis':'Lower held-out token loss plus improved generated reference-assertion proxies without observed regression','limitations':assessment['limitations']}
    temporary=active_path.with_suffix('.tmp');temporary.write_text(json.dumps(active,indent=2));temporary.replace(active_path);selected=True
# Let the training process release its GPU allocations before loading the app.
for _ in range(12):
    try:os.kill(args.training_pid,0)
    except ProcessLookupError:break
    time.sleep(5)
import subprocess
with (experiment/'application_verification.log').open('w') as log:
    smoke=subprocess.run([sys.executable,'scripts/verify_application.py','--output',str(experiment)],cwd=ROOT,env={**os.environ,'HF_HUB_OFFLINE':'1'},stdout=log,stderr=subprocess.STDOUT)
verification_path=experiment/'application_verification.json'
verification=json.loads(verification_path.read_text()) if verification_path.exists() else {'passed':False,'error':'Verification exited without a result'}
if selected and (smoke.returncode or not verification.get('within_two_minute_target')):
    active=json.loads(active_path.read_text());active['report_adapter']=previous_adapter
    temporary=active_path.with_suffix('.tmp');temporary.write_text(json.dumps(active,indent=2));temporary.replace(active_path);selected=False
result={'phase':'complete','adapter_selected_experimentally':selected,'application_verification':verification,'eligible_for_experimental_selection':assessment['eligible_for_experimental_selection'],'assessment':str(experiment/'assessment.json'),'comparisons':str(experiment/'COMPARISON.md'),'limitations':assessment['limitations']}
status.write_text(json.dumps(result,indent=2));print('FINISHED',json.dumps(result),flush=True)
