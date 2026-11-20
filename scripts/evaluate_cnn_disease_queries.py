"""Compare disease-query CNN with selected CNN on all original IU validation views."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from chest_xpert.study_images import resolve_study_images,index_study_images
from chest_xpert.checkpoint_engine import CheckpointEngine
from chest_xpert.disease_targets import disease_targets
from scripts.train_cnn_corrected import extract,predict_features,metrics

def main():
 root=Path('checkpoints/cnn_disease_queries_v1');candidate=root/'cnn_model.pth';assert candidate.exists()
 selected=json.loads(Path('checkpoints/full_study_cnn_validation/results.json').read_text());view_rows=selected['rows'];assert len(view_rows)==selected['original_views']==1124
 manifest=Path('data/report_training_v3/validation.jsonl');assert hashlib.sha256(manifest.read_bytes()).hexdigest()==selected['manifest_sha256']
 study_rows={r['study_id']:r for r in map(json.loads,manifest.read_text().splitlines())};rows=[{**study_rows[v['study_id']],'image_path':v['image_path']} for v in view_rows]
 engine=CheckpointEngine(checkpoint=candidate);torch.set_num_threads(4)
 feats=extract(engine.model,rows,'cuda:0',engine.pixel_scale,engine.crop,16)
 scores=predict_features(engine.model,feats,'cuda:0')
 targets,masks=zip(*(disease_targets(row) for row in rows));targets=np.array(targets);masks=np.array(masks)
 result=metrics(scores,targets,masks,rows);result['checkpoint_sha256']=hashlib.sha256(candidate.read_bytes()).hexdigest()
 parent=selected['metrics']['parent'];out={'cohort':'same IU X-Ray 563-study, 1124-view source-derived validation cohort','selected':{k:parent[k] for k in ['macro_auroc','macro_average_precision','evaluable_labels','checkpoint_sha256']},'disease_queries':result,'candidate_selected':bool(result['macro_auroc']>parent['macro_auroc'] and result['macro_average_precision']>=parent['macro_average_precision']),'test_used':False,'clinical_accuracy_claim':False}
 (root/'full_study_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='disease_queries'}|{'candidate_metrics':{k:result[k] for k in ['macro_auroc','macro_average_precision','evaluable_labels']}},indent=2))
if __name__=='__main__':main()
