"""Print local model selection and the running report-training status."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
import argparse
p=argparse.ArgumentParser();p.add_argument('--experiment',default='checkpoints/report_lora_full_study');args=p.parse_args()
experiment=root/args.experiment
active_path=root/'checkpoints/active_models.json'
if active_path.exists():
    active=json.loads(active_path.read_text())
    print('CNN:',active.get('cnn',{}).get('path','original'))
    print('Report adapter:',(active.get('report_adapter') or {}).get('path','base Qwen; no replacement selected'))
progress_path=experiment/'progress.json'
if progress_path.exists():
    progress=json.loads(progress_path.read_text())
    phase=progress.get('phase')
    log_path=experiment/'controller.log'
    if phase=='preparing_evidence' and log_path.exists():
        tail=log_path.read_text()[-16000:]
        if 'BASE_VALIDATION_LOSS' in tail:phase='training (first optimizer update pending)'
        elif 'LOADING_QWEN' in tail:phase='model loading / initial validation'
    print('Training phase:',phase)
    step=progress.get('step',0);total=progress.get('target_steps',64)
    print(f'Updates: {step}/{total}')
    history=progress.get('history',[])
    if history and step<total:
        seconds=history[-1]['elapsed_s']/max(step,1)
        print(f'Estimated remaining training: {(total-step)*seconds/60:.0f} minutes, plus evaluation')
completion_path=experiment/'completion_status.json'
if completion_path.exists():
    completion=json.loads(completion_path.read_text())
    print('Completion checks:',completion.get('phase'))
    if completion.get('phase')=='complete':
        print('Replacement adapter selected:',completion.get('adapter_selected_experimentally'))
        print('Application verification:',completion.get('application_verification',{}).get('passed'))
        print('Comparisons:',completion.get('comparisons'))
