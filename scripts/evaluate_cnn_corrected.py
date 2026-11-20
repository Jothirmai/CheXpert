"""One-time held-out CNN check after validation-based model selection."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.train_cnn_corrected import load_rows,ImageRows,metrics
from chest_xpert.checkpoint_engine import CheckpointEngine
from chest_xpert.disease_targets import disease_targets

p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/cnn_corrected/cnn_model.pth');p.add_argument('--manifest',default='data/report_training_v3/test.jsonl');args=p.parse_args()
output=Path(args.checkpoint).parent/'test_metrics.json'
if output.exists():
    print(output.read_text())
else:
    rows=load_rows(args.manifest);engine=CheckpointEngine(device='cuda:0',checkpoint=args.checkpoint)
    outcomes=[disease_targets(row) for row in rows]
    scores=[]
    with torch.no_grad():
        for images in DataLoader(ImageRows(rows,engine.pixel_scale,engine.crop),batch_size=32,num_workers=4):
            scores.append(engine.model(images.to('cuda:0'))['disease_probs'].cpu())
    result=metrics(torch.cat(scores).numpy(),np.array([x[0] for x in outcomes]),np.array([x[1] for x in outcomes]),rows)
    result['limitations']='Weak annotation/report-derived labels, not independently adjudicated clinical outcomes. Test set used once after validation selection.'
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
