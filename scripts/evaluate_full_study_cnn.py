"""Compare checkpoints on unchanged validation studies with every original view."""
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from chest_xpert.study_images import resolve_study_images,index_study_images
from chest_xpert.checkpoint_engine import CheckpointEngine
from chest_xpert.disease_targets import disease_targets
from scripts.train_cnn_corrected import extract,predict_features,metrics

def main():
 out=Path('checkpoints/full_study_cnn_validation');out.mkdir(exist_ok=True)
 manifest=Path('data/report_training_v3/validation.jsonl');unique={}
 for row in map(json.loads,manifest.read_text().splitlines()):unique.setdefault(row['study_id'],row)
 rows=[];image_index=index_study_images('data/iu_xray/images')
 for study,row in unique.items():
  views=resolve_study_images(Path('data/iu_xray/reports/ecgen-radiology')/f'{study.removeprefix("CXR")}.xml','data/iu_xray/images',image_index=image_index)
  for image in views:rows.append({**row,'image_path':str(image)})
 engine=CheckpointEngine(checkpoint='checkpoints/cnn_corrected/cnn_model.pth')
 print('Resolved validation views',len(rows),flush=True)
 torch.set_num_threads(4)
 features=extract(engine.model,rows,'cuda:0',engine.pixel_scale,engine.crop,16)
 targets,masks=zip(*(disease_targets(row) for row in rows));targets=np.array(targets);masks=np.array(masks)
 output={}
 for variant,path in [('parent','checkpoints/cnn_corrected/cnn_model.pth'),('adjusted','checkpoints/cnn_weight_adjusted/cnn_model.pth')]:
  if variant=='adjusted':
   engine.model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True))
  scores=predict_features(engine.model,features,'cuda:0');measurement=metrics(scores,targets,masks,rows)
  groups={}
  for i,row in enumerate(rows):groups.setdefault(row['study_id'],[]).append(i)
  p=np.stack([scores[ix].mean(0) for ix in groups.values()]).clip(1e-7,1-1e-7);y=np.stack([targets[ix[0]] for ix in groups.values()]);m=np.stack([masks[ix[0]] for ix in groups.values()])
  measurement['known_label_log_loss']=float((-(y*np.log(p)+(1-y)*np.log1p(-p))*m).sum()/m.sum())
  measurement['known_label_brier']=float((((p-y)**2)*m).sum()/m.sum());measurement['checkpoint_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
  output[variant]=measurement
  np.savez_compressed(out/f'{variant}_scores.npz',scores=scores,targets=targets,masks=masks)
 result={'validation_studies':len(unique),'original_views':len(rows),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'labeler_sha256':hashlib.sha256(Path('chest_xpert/disease_targets.py').read_bytes()).hexdigest(),'rows':[{'study_id':r['study_id'],'image_path':r['image_path']} for r in rows],'metrics':output,'test_used':False,'existing_splits_modified':False,'limitations':'Same 563 validation studies, now all original XML views. Corrected weak labels, not clinical accuracy; training checkpoint still uses earlier weak labels.'}
 (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'original_views':len(rows),'metrics':output},indent=2),flush=True)
if __name__=='__main__':main()
