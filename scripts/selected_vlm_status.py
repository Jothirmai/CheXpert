"""Show live IU and CheXpert Plus VLM training status and percentages."""
import json
from pathlib import Path

for name, label in [('report_lora_selected_cnn_v1', 'IU X-Ray'),
                    ('report_lora_chexpert_plus_v1', 'CheXpert Plus')]:
    folder = Path('checkpoints') / name
    status = json.loads((folder / 'completion_status.json').read_text()) if (folder / 'completion_status.json').exists() else {}
    progress = json.loads((folder / 'progress.json').read_text()) if (folder / 'progress.json').exists() else {}
    control = json.loads((folder / 'control.json').read_text()) if (folder / 'control.json').exists() else {}
    target = min(progress.get('target_steps', 0), control.get('stop_after_step', progress.get('target_steps', 0)))
    step = progress.get('step', 0)
    percent = round(100 * step / target, 2) if target else None
    print(json.dumps({'dataset':label,'phase':progress.get('phase') or status.get('phase'),
                      'optimizer_steps':step,'target_steps':target or None,
                      'training_percent':percent,'controller_phase':status.get('phase')}))

test_progress = Path('checkpoints/report_lora_chexpert_plus_v1/reserved_test_progress.json')
if test_progress.exists():
    print(json.dumps({'dataset': 'CheXpert Plus reserved test',
                      **json.loads(test_progress.read_text())}))
fp16_progress = Path('checkpoints/report_lora_chexpert_plus_v1/reserved_test_fp16_progress.json')
if fp16_progress.exists():
    print(json.dumps({'dataset': 'CheXpert Plus FP16 exploratory follow-up',
                      **json.loads(fp16_progress.read_text())}))
