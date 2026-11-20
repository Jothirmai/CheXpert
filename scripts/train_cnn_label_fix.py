"""Retrain only the head using corrected negation labels and cached encoder features.

Selection compares parent and candidate on the SAME corrected validation labels.
No test examples are loaded. Previous label-based metrics are not comparable.
"""
import copy,hashlib,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from scripts.train_cnn_corrected import load_rows,metrics,predict_features
from chest_xpert.checkpoint_engine import CheckpointEngine
from chest_xpert.disease_targets import disease_targets,DISEASES
out=Path('checkpoints/cnn_label_fixed');out.mkdir(exist_ok=True)
source=Path('checkpoints/cnn_corrected');config=json.loads((source/'training_config.json').read_text())
rows={split:load_rows(f'data/report_training_v3/{split}.jsonl') for split in ['train','validation']}
for split in rows:assert hashlib.sha256(Path(f'data/report_training_v3/{split}.jsonl').read_bytes()).hexdigest()==config['manifest_sha256'][split]
assert not {r['study_id'] for r in rows['train']}&{r['study_id'] for r in rows['validation']}
features=torch.load(source/'features.pt',map_location='cpu',weights_only=True)
labels={split:tuple(torch.tensor(a,dtype=torch.float32) for a in zip(*(disease_targets(r) for r in rows[split]))) for split in rows}
for split in rows:assert len(features[split])==len(rows[split])
y,m=labels['train'];vy,vm=labels['validation'];positive=(y*m).sum(0);negative=((1-y)*m).sum(0)
engine=CheckpointEngine(checkpoint=source/'cnn_model.pth');model=engine.model;model.alignment.requires_grad_(True)
baseline=metrics(predict_features(model,features['validation'],'cuda:0'),vy.numpy(),vm.numpy(),rows['validation'])
initial=copy.deepcopy(model.alignment.state_dict());best=baseline;best_selection=None;history=[];start=time.perf_counter()
counts={}
for r in rows['train']:counts[r['study_id']]=counts.get(r['study_id'],0)+1
study_weights=torch.tensor([1/counts[r['study_id']] for r in rows['train']])
objective=lambda score:(score['macro_auroc']+score['macro_average_precision'])/2
for lr,cap in [(3e-5,1),(1e-4,1),(3e-5,5),(1e-4,5)]:
 torch.manual_seed(193);random.seed(193);np.random.seed(193)
 model.alignment.load_state_dict(initial);model.alignment.disease_specific_queries=True
 optimizer=torch.optim.AdamW(model.alignment.parameters(),lr=lr,weight_decay=1e-3)
 loader=DataLoader(TensorDataset(features['train'],y,m,study_weights),batch_size=64,shuffle=True)
 pos_weight=(negative/positive.clamp_min(1)).clamp(1,cap).cuda();stale=0;local_best=-1
 for epoch in range(1,16):
  model.alignment.train()
  for f,t,mask,sw in loader:
   f=f.cuda().float();t=t.cuda();mask=mask.cuda();sw=sw.cuda()[:,None]
   logits=model.alignment(f)['disease_logits']
   loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,t,pos_weight=pos_weight,reduction='none')*mask*sw).sum()/(mask*sw).sum().clamp_min(1)
   assert torch.isfinite(loss)
   optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.alignment.parameters(),1);optimizer.step()
  score=metrics(predict_features(model,features['validation'],'cuda:0'),vy.numpy(),vm.numpy(),rows['validation'])
  entry={'lr':lr,'positive_weight_cap':cap,'epoch':epoch,'auroc':score['macro_auroc'],'ap':score['macro_average_precision']};history.append(entry);print(json.dumps(entry),flush=True)
  if score['macro_auroc']>=baseline['macro_auroc'] and score['macro_average_precision']>=baseline['macro_average_precision'] and objective(score)>objective(best):
   best=score;best_selection=entry;torch.save(model.state_dict(),out/'cnn_model.pth')
  value=objective(score)
  if value>local_best:local_best=value;stale=0
  else:stale+=1
  (out/'progress.json').write_text(json.dumps({'phase':'training','latest':entry,'selected':best_selection}))
  if stale>=4:break
result={'parent_on_corrected_labels':baseline,'candidate_on_corrected_labels':best,'selected':best_selection,'selection':'Both validation macro AUROC and AP must not regress; maximize their mean among eligible candidates','history':history,'test_used':False,'labeler_sha256':hashlib.sha256(Path('chest_xpert/disease_targets.py').read_bytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-start,'auto_activated':False,'limitations':'Weak derived labels; combined label and head correction. Both models compared on the same corrected validation labels. Historical metrics before the negation fix are not comparable.'}
(out/'comparison.json').write_text(json.dumps(result,indent=2));(out/'validation_metrics.json').write_text(json.dumps(best,indent=2))
if best_selection:
 config.update({'model_name':'Chest Xpert negation-corrected per-finding CNN','disease_specific_queries':True,'selection':result['selection'],'labeler_sha256':result['labeler_sha256'],'positive_weight_cap':best_selection['positive_weight_cap'],'label_support':{d:bool(positive[i]>=5 and negative[i]>=20) for i,d in enumerate(DISEASES)}})
 (out/'training_config.json').write_text(json.dumps(config,indent=2))
(out/'progress.json').write_text(json.dumps({'phase':'complete','selected':best_selection,'auto_activated':False}))
print('COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['history','parent_on_corrected_labels','candidate_on_corrected_labels']}),flush=True)
