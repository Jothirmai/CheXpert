"""Undo the known positive-class loss weighting in classifier logits.

This is a training-objective correction, not clinical probability calibration.
Validate both configurations on identical corrected, study-grouped labels.
"""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from chest_xpert.checkpoint_engine import CheckpointEngine
from chest_xpert.disease_targets import disease_targets,DISEASES
from scripts.train_cnn_corrected import load_rows,metrics,predict_features
source=Path('checkpoints/cnn_corrected');out=Path('checkpoints/cnn_weight_adjusted');out.mkdir(exist_ok=True)
config=json.loads((source/'training_config.json').read_text());audit=json.loads((source/'label_audit.json').read_text())
weights=np.array([min(20,max(1,audit[d]['negative_images']/max(audit[d]['positive_images'],1))) for d in DISEASES])
rows=load_rows('data/report_training_v3/validation.jsonl')
assert hashlib.sha256(Path('data/report_training_v3/validation.jsonl').read_bytes()).hexdigest()==config['manifest_sha256']['validation']
features=torch.load(source/'features.pt',map_location='cpu',weights_only=True)['validation']
y,m=zip(*(disease_targets(r) for r in rows));y=np.array(y);m=np.array(m)
engine=CheckpointEngine(checkpoint=source/'cnn_model.pth');before=predict_features(engine.model,features,'cuda:0')
with torch.no_grad():engine.model.alignment.disease_classifier.bias.sub_(torch.tensor(np.log(weights),device='cuda:0',dtype=torch.float32))
after=predict_features(engine.model,features,'cuda:0')
parent=metrics(before,y,m,rows);candidate=metrics(after,y,m,rows)
def proper_losses(scores):
 groups={}
 for i,r in enumerate(rows):groups.setdefault(r['study_id'],[]).append(i)
 s=np.stack([scores[ix].mean(0) for ix in groups.values()]);t=np.stack([y[ix[0]] for ix in groups.values()]);mask=np.stack([m[ix[0]] for ix in groups.values()])
 p=s.clip(1e-7,1-1e-7)
 return {'known_label_log_loss':float((-(t*np.log(p)+(1-t)*np.log1p(-p))*mask).sum()/mask.sum()),'known_label_brier':float((((p-t)**2)*mask).sum()/mask.sum())}
parent.update(proper_losses(before));candidate.update(proper_losses(after))
result={'parent_on_corrected_labels':parent,'candidate_on_corrected_labels':candidate,'formula':'classifier_bias -= log(training_pos_weight); pos_weight = clip(negative_images / positive_images, 1, 20) from the original run','training_positive_weights':dict(zip(DISEASES,weights.tolist())),'test_used':False,'auto_activated':False,'limitations':'Not clinically calibrated. Weak labels and class-dependent missingness remain. Per-image rankings are monotone, but averaging views can change study-level rankings.'}
(out/'comparison.json').write_text(json.dumps(result,indent=2))
config.update({'model_name':'Chest Xpert training-weight-adjusted CNN','score_semantics':'Training positive-class weighting removed from logits; not clinical probabilities','training_positive_weights':result['training_positive_weights'],'labeler_sha256':hashlib.sha256(Path('chest_xpert/disease_targets.py').read_bytes()).hexdigest(),'parent_checkpoint_sha256':hashlib.sha256((source/'cnn_model.pth').read_bytes()).hexdigest()})
(out/'training_config.json').write_text(json.dumps(config,indent=2));(out/'validation_metrics.json').write_text(json.dumps(candidate,indent=2));torch.save(engine.model.state_dict(),out/'cnn_model.pth')
print(json.dumps({k:{j:v[j] for j in ['macro_auroc','macro_average_precision','known_label_log_loss','known_label_brier']} for k,v in result.items() if k.endswith('corrected_labels')},indent=2))
