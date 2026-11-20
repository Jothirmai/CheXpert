"""Paired study bootstrap of validation deltas; conditional on selected models."""
import json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from sklearn.metrics import roc_auc_score,average_precision_score
from chest_xpert.checkpoint_engine import CheckpointEngine
from chest_xpert.disease_targets import disease_targets
from scripts.train_cnn_corrected import load_rows,predict_features
folder=Path('checkpoints/cnn_validation_blend')
rows=load_rows('data/report_training_v3/validation.jsonl')
features=torch.load('checkpoints/cnn_corrected/features.pt',map_location='cpu',weights_only=True)['validation']
predictions=[]
for path in ['checkpoints/cnn_corrected/cnn_model.pth',str(folder/'cnn_model.pth')]:
 engine=CheckpointEngine(checkpoint=path);predictions.append(predict_features(engine.model,features,'cuda:0'));del engine;torch.cuda.empty_cache()
groups=defaultdict(list)
for i,r in enumerate(rows):groups[r['study_id']].append(i)
groups=list(groups.values());pairs=[disease_targets(rows[ix[0]]) for ix in groups]
y=np.array([p[0] for p in pairs]);mask=np.array([p[1] for p in pairs],dtype=bool)
preds=[np.stack([s[ix].mean(0) for ix in groups]) for s in predictions]
np.savez(folder/'validation_predictions.npz',baseline=preds[0],candidate=preds[1],targets=y,masks=mask)
rng=np.random.default_rng(491);deltas=[];attempts=0
while len(deltas)<500 and attempts<2000:
 attempts+=1;ix=rng.integers(0,len(groups),len(groups));by=y[ix];bm=mask[ix]
 scores=[[],[]];valid=True
 for j in range(y.shape[1]):
  truth=by[bm[:,j],j]
  if len(np.unique(truth))<2:valid=False;break
  for mode,p in enumerate(preds):
   values=p[ix][bm[:,j],j];scores[mode].append([roc_auc_score(truth,values),average_precision_score(truth,values)])
 if valid:deltas.append(np.mean(scores[1],axis=0)-np.mean(scores[0],axis=0))
deltas=np.array(deltas)
result={'replicates':len(deltas),'attempts':attempts,'paired_studies':len(groups),'auroc_delta_95_percentile_interval':np.quantile(deltas[:,0],[.025,.975]).tolist(),'average_precision_delta_95_percentile_interval':np.quantile(deltas[:,1],[.025,.975]).tolist(),'limitations':'Validation bootstrap after adaptive model selection; does not account for search optimism or weak-label error. Resamples lacking either class for any of the fixed 14 labels are rejected. Not a clinical confidence interval.'}
(folder/'bootstrap.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
