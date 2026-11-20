"""Wait for bounded CNN training, then run frozen original-vs-candidate evaluation."""
import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'checkpoints/chexpert_plus_imagenet_full_v1';started=time.monotonic()
while time.monotonic()-started<12*3600:
 try:
  status=json.loads((out/'progress.json').read_text())
  if status.get('phase')=='complete' and (out/'best_model.pth').exists():break
 except (OSError,ValueError):pass
 time.sleep(20)
else:raise RuntimeError('Training did not finish in twelve hours; inspect progress.json and training process')
result=subprocess.run([sys.executable,'-u','scripts/evaluate_chexpert_plus_head.py'],cwd=root)
if result.returncode:raise SystemExit(result.returncode)
print('CHEXPERT_PLUS_CNN_COMPARISON_COMPLETE',flush=True)
