"""Validation-selected interpolation; requires both AUROC and AP nonregression."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from scripts.train_cnn_corrected import load_rows,metrics,predict_features
from chest_xpert.checkpoint_engine import CheckpointEngine
from chest_xpert.disease_targets import disease_targets
out=Path('checkpoints/cnn_validation_blend');out.mkdir(exist_ok=True)
source=Path('checkpoints/cnn_corrected');candidate=Path('checkpoints/cnn_validation_search')
config=json.loads((source/'training_config.json').read_text())
manifest=Path('data/report_training_v3/validation.jsonl')
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==config['manifest_sha256']['validation']
rows=load_rows(manifest);pairs=[disease_targets(r) for r in rows];y=np.array([p[0] for p in pairs]);m=np.array([p[1] for p in pairs])
features=torch.load(source/'features.pt',map_location='cpu',weights_only=True)['validation']
base=torch.load(source/'cnn_model.pth',map_location='cpu',weights_only=True)
other=torch.load(candidate/'cnn_model.pth',map_location='cpu',weights_only=True)
model=CheckpointEngine(checkpoint=source/'cnn_model.pth').model
baseline=metrics(predict_features(model,features,'cuda:0'),y,m,rows);best=baseline;chosen=0;trials=[]
for alpha in [.1,.25,.5,.75]:
 state={k:(base[k]*(1-alpha)+other[k]*alpha if k.startswith('alignment.') and base[k].is_floating_point() else base[k]) for k in base}
 model.load_state_dict(state)
 score=metrics(predict_features(model,features,'cuda:0'),y,m,rows)
 entry={'alpha':alpha,'macro_auroc':score['macro_auroc'],'macro_average_precision':score['macro_average_precision']};trials.append(entry);print(json.dumps(entry),flush=True)
 if score['macro_auroc']>=baseline['macro_auroc'] and score['macro_average_precision']>=baseline['macro_average_precision'] and sum([score['macro_auroc'],score['macro_average_precision']])>sum([best['macro_auroc'],best['macro_average_precision']]):
  best=score;chosen=alpha;torch.save(state,out/'cnn_model.pth')
report={'baseline':baseline,'candidate':best,'alpha':chosen,'trials':trials,'selection':'Both validation AUROC and AP must not regress; maximize their sum among eligible blends','test_used':False,'auto_activated':False}
(out/'comparison.json').write_text(json.dumps(report,indent=2))
if chosen:
 config.update({'model_name':'Chest Xpert validation-blended Stage1CNN','selection':report['selection'],'blend_alpha':chosen,'parent_checkpoints':[str(source/'cnn_model.pth'),str(candidate/'cnn_model.pth')]})
 (out/'training_config.json').write_text(json.dumps(config,indent=2));(out/'validation_metrics.json').write_text(json.dumps(best,indent=2))
print('COMPLETE',json.dumps({'alpha':chosen,'auroc':best['macro_auroc'],'ap':best['macro_average_precision']}),flush=True)
