"""Controlled end-to-end report test of original versus weight-adjusted CNN inputs."""
import hashlib,json,sys,time
from pathlib import Path
from uuid import uuid4
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.config import STORAGE_CONFIG
from backend.services.inference_service import inference_service
from backend.services import review_store
from chest_xpert.checkpoint_engine import CheckpointEngine
from scripts.assess_report_adapter import fact_states,analyze
from cliq_rrg.eval import bleu,rouge_l
from chest_xpert.study_images import resolve_study_images
out=Path('checkpoints/report_score_correction_multiview');out.mkdir(exist_ok=True)
rows=[json.loads(line) for line in Path('data/report_training_v3/validation.jsonl').read_text().splitlines() if line.strip()]
unique={}
for r in rows:unique.setdefault(r['study_id'],r)
positive=[]
for label in ['Cardiomegaly','Effusion']:
 for row in unique.values():
  if row['study_id'] in {r['study_id'] for r in positive}:continue
  if fact_states(row['target']['sections']).get(label)==1 and not any(term.lower()=='normal' for term in row.get('mesh_terms',[])):
   positive.append(row);break
# Keep the predeclared validation cohort; filenames do not identify projection.
cohort=json.loads(Path('checkpoints/report_score_correction/configuration.json').read_text())['studies']
selected=[unique[study] for study in cohort]
assert len(selected)==4 and len({r['study_id'] for r in selected})==4
(out/'configuration.json').write_text(json.dumps({'studies':[r['study_id'] for r in selected],'selection':'Same predeclared four-study validation cohort as the earlier score correction experiment; now every original XML view is required','reference_findings_supplied_to_generation':False,'test_used_for_selection':False,'input_contract':'actual mean CNN scores across every original study view + all original images + Grad-CAM contact sheet','labeler_sha256':hashlib.sha256(Path('chest_xpert/disease_targets.py').read_bytes()).hexdigest()},indent=2))
STORAGE_CONFIG.temp_dir=out/'temp';STORAGE_CONFIG.reports_dir=out/'reports';STORAGE_CONFIG.temp_dir.mkdir(exist_ok=True);STORAGE_CONFIG.reports_dir.mkdir(exist_ok=True)
results={}
for variant,checkpoint in [('parent','checkpoints/cnn_corrected/cnn_model.pth'),('adjusted','checkpoints/cnn_weight_adjusted/cnn_model.pth')]:
 service=inference_service;service._engine=CheckpointEngine(checkpoint=checkpoint);service._engine_device='cuda:0'
 cases=[]
 for row in selected:
  start=time.perf_counter();identifier=str(uuid4());accepted=False;review=None;error=None
  xml=Path('data/iu_xray/reports/ecgen-radiology')/f"{row['study_id'].removeprefix('CXR')}.xml"
  views=resolve_study_images(xml,'data/iu_xray/images')
  request={'additional_image_paths':[str(p) for p in views[1:]],'patient_id':row['study_id'],'clinical_indication':row.get('clinical_indication'),'comparison_study':row.get('comparison_study'),'template_id':'standard'}
  try:
   pipeline=service._run_pipeline_sync(identifier,str(views[0]),request,True)
   assert pipeline['status']=='completed',pipeline
   review=review_store.read(identifier);accepted=True
  except Exception as exc:error=str(exc)
  client=service._verification_engine.client if service._verification_engine else None
  raw=getattr(client,'_last_generation_output',None) or ''
  check=analyze(json.dumps({'sections':review['sections']}) if review else raw,row['target'],row.get('mesh_terms'))
  lexical={'bleu1':0,'bleu4':0,'rouge_l':0}
  if review:
   truth=' '.join(s['text'] for s in row['target']['sections'] if s['heading'] in ['Findings','Impression'])
   pred=' '.join(s['text'] for s in review['sections'] if s['heading'] in ['Findings','Impression'])
   lexical={'bleu1':bleu(truth,pred,1),'bleu4':bleu(truth,pred,4),'rouge_l':rouge_l(truth,pred)}
   assert review['model_provenance']['vision_inputs']['image_count']==len(views)+1
   assert len(review['model_provenance']['cnn_scores'])==14
  case={'study_id':row['study_id'],'reference':row['target'],'accepted':accepted,'error':error,'raw_output':raw,'review':review,'proxy':check,'lexical':lexical,'elapsed_seconds':time.perf_counter()-start}
  cases.append(case);results[variant]=cases
  (out/'results.json').write_text(json.dumps(results,indent=2))
  print(json.dumps({'variant':variant,'study':row['study_id'],'accepted':accepted,'proxy':check,'lexical':lexical,'error':error}),flush=True)
summary={}
for variant,cases in results.items():
 summary[variant]={'cases':len(cases),'accepted_reports':sum(c['accepted'] for c in cases),'reference_positive_terms':sum(c['proxy']['reference_positive_terms'] for c in cases),'matched_positive_terms':sum(c['proxy']['matched_positive_terms'] for c in cases),'unsupported_terms_against_known_negatives':sum(c['proxy']['positive_terms_in_normal_reference'] for c in cases),'mean_rouge_l':sum(c['lexical']['rouge_l'] for c in cases)/len(cases),'mean_bleu1':sum(c['lexical']['bleu1'] for c in cases)/len(cases)}
b,a=summary['parent'],summary['adjusted']
recommend=bool(a['accepted_reports']>=b['accepted_reports'] and a['matched_positive_terms']>=b['matched_positive_terms'] and a['unsupported_terms_against_known_negatives']<=b['unsupported_terms_against_known_negatives'] and (a['matched_positive_terms']>b['matched_positive_terms'] or a['unsupported_terms_against_known_negatives']<b['unsupported_terms_against_known_negatives']))
report={'metrics':summary,'recommend_experimental_selection':recommend,'auto_activated':False,'limitations':'Four validation studies; incomplete lexical proxies, not expert adjudication. Both input configurations generated with the same VLM and prompt, no reference findings/impression supplied. All views in the dataset XML are required; incomplete studies stop the benchmark.'}
(out/'summary.json').write_text(json.dumps(report,indent=2));print('COMPLETE',json.dumps(report),flush=True)
# Illustrative previously inspected example, deliberately excluded from selection.
row=json.loads(Path('reports/required_vlm_evidence/dataset_reference.json').read_text())
identifier=str(uuid4())
xml=Path('data/iu_xray/reports/ecgen-radiology')/f"{row['study_id'].removeprefix('CXR')}.xml"
views=resolve_study_images(xml,'data/iu_xray/images')
try:
 pipeline=inference_service._run_pipeline_sync(identifier,str(views[0]),{'additional_image_paths':[str(p) for p in views[1:]],'patient_id':row['study_id'],'clinical_indication':row.get('clinical_indication'),'comparison_study':row.get('comparison_study'),'template_id':'standard'},True)
 review=review_store.read(identifier)
 (out/'illustrative_report.json').write_text(json.dumps(review,indent=2))
 print('ILLUSTRATIVE_REPORT',json.dumps(review['sections']),flush=True)
except Exception as exc:
 (out/'illustrative_failure.json').write_text(json.dumps({'error':str(exc)},indent=2))
 print('ILLUSTRATIVE_FAILURE',str(exc),flush=True)
