"""Validation-only search over cached frozen-encoder features; no test access."""
import copy,hashlib,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from scripts.train_cnn_corrected import load_rows,metrics,predict_features
from chest_xpert.checkpoint_engine import CheckpointEngine
from chest_xpert.disease_targets import disease_targets
out=Path('checkpoints/cnn_validation_search');out.mkdir(exist_ok=True)
source=Path('checkpoints/cnn_corrected');config=json.loads((source/'training_config.json').read_text())
rows={split:load_rows(f'data/report_training_v3/{split}.jsonl') for split in ['train','validation']}
for split in rows:
 assert hashlib.sha256(Path(f'data/report_training_v3/{split}.jsonl').read_bytes()).hexdigest()==config['manifest_sha256'][split]
assert not {r['study_id'] for r in rows['train']}&{r['study_id'] for r in rows['validation']}
features=torch.load(source/'features.pt',map_location='cpu',weights_only=True)
for split in rows:assert len(features[split])==len(rows[split])
labels={split:tuple(torch.tensor(a,dtype=torch.float32) for a in zip(*(disease_targets(r) for r in rows[split]))) for split in rows}
engine=CheckpointEngine(device='cuda:0',checkpoint=source/'cnn_model.pth');model=engine.model
model.alignment.requires_grad_(True)
initial=copy.deepcopy(model.alignment.state_dict())
y,m=labels['train'];vy,vm=labels['validation']
baseline=metrics(predict_features(model,features['validation'],'cuda:0'),vy.numpy(),vm.numpy(),rows['validation'])
# Fixed objective set before inspecting candidate validation results.
objective=lambda d:(d['macro_auroc']+d['macro_average_precision'])/2
best=objective(baseline);best_metrics=baseline;history=[];selected=None
counts={}
for r in rows['train']:counts[r['study_id']]=counts.get(r['study_id'],0)+1
study_weight=torch.tensor([1/counts[r['study_id']] for r in rows['train']])
positive=(y*m).sum(0);negative=((1-y)*m).sum(0)
start=time.perf_counter()
for lr,cap in [(1e-5,1),(3e-5,1),(1e-5,5),(3e-5,5)]:
 torch.manual_seed(73);random.seed(73);np.random.seed(73)
 model.alignment.load_state_dict(initial)
 optimizer=torch.optim.AdamW(model.alignment.parameters(),lr=lr,weight_decay=1e-3)
 loader=DataLoader(TensorDataset(features['train'],y,m,study_weight),batch_size=64,shuffle=True)
 weights=(negative/positive.clamp_min(1)).clamp(1,cap).cuda()
 stale=0;local_best=-1
 for epoch in range(1,13):
  model.alignment.train()
  for f,t,mask,sw in loader:
   f=f.cuda().float();t=t.cuda();mask=mask.cuda();sw=sw.cuda()[:,None]
   loss=(torch.nn.functional.binary_cross_entropy_with_logits(model.alignment(f)['disease_logits'],t,pos_weight=weights,reduction='none')*mask*sw).sum()/(mask*sw).sum().clamp_min(1)
   optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.alignment.parameters(),1);optimizer.step()
  score=metrics(predict_features(model,features['validation'],'cuda:0'),vy.numpy(),vm.numpy(),rows['validation'])
  value=objective(score);entry={'lr':lr,'positive_weight_cap':cap,'epoch':epoch,'auroc':score['macro_auroc'],'ap':score['macro_average_precision'],'objective':value};history.append(entry)
  print(json.dumps(entry),flush=True)
  if value>best:
   best=value;best_metrics=score;selected=entry;torch.save(model.state_dict(),out/'cnn_model.pth')
  if value>local_best:local_best=value;stale=0
  else:stale+=1
  if stale>=4:break
  (out/'progress.json').write_text(json.dumps({'phase':'training','latest':entry,'best':selected}))
report={'baseline':baseline,'candidate':best_metrics,'selected':selected,'objective':'Mean validation macro AUROC and macro average precision; both higher is better','history':history,'test_used':False,'auto_activated':False,'elapsed_seconds':time.perf_counter()-start,'limitations':'Adaptive validation search on weak labels; no independent clinical claim. Existing test results have already been inspected and are not used for selection.'}
(out/'comparison.json').write_text(json.dumps(report,indent=2));(out/'validation_metrics.json').write_text(json.dumps(best_metrics,indent=2))
if selected:
 config.update({'model_name':'Chest Xpert validation-selected Stage1CNN candidate','selection':report['objective'],'parent_checkpoint_sha256':hashlib.sha256((source/'cnn_model.pth').read_bytes()).hexdigest(),'fine_tuning':selected})
 (out/'training_config.json').write_text(json.dumps(config,indent=2))
(out/'progress.json').write_text(json.dumps({'phase':'complete','selected':selected,'auto_activated':False}))
print('COMPLETE',json.dumps({'baseline_auroc':baseline['macro_auroc'],'baseline_ap':baseline['macro_average_precision'],'candidate_auroc':best_metrics['macro_auroc'],'candidate_ap':best_metrics['macro_average_precision'],'selected':selected}),flush=True)
