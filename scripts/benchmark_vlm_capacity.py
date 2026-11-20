"""Paired validation benchmark with mandatory CNN scores and both images.

Run one model per process so CUDA memory is fully released between models.
Reference findings/impression are used only after generation for scoring.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chest_xpert.verification import LLMModelClient
from chest_xpert.label_mapping import LabelMapping
from chest_xpert.report_quality import parse_generated_sections
from scripts.assess_report_adapter import analyze
from cliq_rrg.eval import bleu,rouge_l
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--name',required=True);p.add_argument('--four-bit',action='store_true');args=p.parse_args()
out=Path('checkpoints/vlm_capacity_benchmark');out.mkdir(exist_ok=True)
rows=[json.loads(l) for l in Path('data/report_training_v3/evidence_fusion_validation.jsonl').read_text().splitlines() if l.strip()][:4]
records={r['study_id']:r for r in (json.loads(l) for l in Path('checkpoints/report_lora_required_evidence/evidence_manifest.jsonl').read_text().splitlines()) if r['split']=='validation'}
configuration={'model':args.model,'four_bit':args.four_bit,'studies':[r['study_id'] for r in rows],'validation_manifest_sha256':hashlib.sha256(Path('data/report_training_v3/evidence_fusion_validation.jsonl').read_bytes()).hexdigest(),'test_used_for_selection':False,'input_contract':'14 CNN scores + original image + labeled Grad-CAM contact sheet','generation_reference_findings_supplied':False,'generation_reference_impression_supplied':False}
(out/f'{args.name}_config.json').write_text(json.dumps(configuration,indent=2))
client=LLMModelClient(model_id=args.model,backend='transformers_local',load_in_4bit=args.four_bit)
if not client.is_configured():raise RuntimeError(client._load_error)
cases=[]
for row in rows:
 record=records[row['study_id']]
 assert hashlib.sha256(Path(record['original_image']).read_bytes()).hexdigest()==record['original_sha256']
 assert hashlib.sha256(Path(record['gradcam_contact_sheet']).read_bytes()).hexdigest()==record['gradcam_sha256']
 scores=record['cnn_scores'];assert len(scores)==14
 info={'clinical_indication':row.get('clinical_indication') or 'NOT PROVIDED','comparison_study':row.get('comparison_study') or 'NOT PROVIDED','technique':row.get('technique') or 'NOT PROVIDED','report_sections':[s['heading'] for s in row['target']['sections']]}
 started=time.perf_counter()
 result=client.generate_v7_report([LabelMapping(k,True,k,v) for k,v in scores.items()],'Project-trained Stage1CNN, 14 findings. Scores are uncalibrated, not probabilities.',info,record['original_image'],record['gradcam_contact_sheet'],self_correct=False)
 case={'study_id':row['study_id'],'reference':row['target'],'raw_output':result.v7_report,'error':result.error,'elapsed_seconds':time.perf_counter()-started,'input_audit':getattr(client,'_last_vision_inputs',None),'cnn_scores':scores}
 check=analyze(result.v7_report or '',row['target'],row.get('mesh_terms'));case['raw_clinical_valid']=check['valid_clinical_json']
 case['accepted']=False;case['lexical']={'bleu1':0.0,'bleu4':0.0,'rouge_l':0.0}
 try:
  sections,normalized=parse_generated_sections(result.v7_report or '',info['report_sections'])
  case.update(accepted=True,sections=sections,schema_normalized=normalized)
  check=analyze(json.dumps({'sections':sections}),row['target'],row.get('mesh_terms'))
  ref=' '.join(s['text'] for s in row['target']['sections'] if s['heading'] in ['Findings','Impression'])
  pred=' '.join(s['text'] for s in sections if s['heading'] in ['Findings','Impression'])
  case['lexical']={'bleu1':bleu(ref,pred,1),'bleu4':bleu(ref,pred,4),'rouge_l':rouge_l(ref,pred)}
 except (ValueError,KeyError,TypeError) as exc:case['rejection']=str(exc)
 case['proxy']=check
 assert case['input_audit'] and case['input_audit']['image_count']==2,case
 cases.append(case)
 (out/f'{args.name}_results.json').write_text(json.dumps({'configuration':configuration,'cases':cases},indent=2))
 print(json.dumps({'model':args.name,'study':row['study_id'],'accepted':case['accepted'],'proxy':check,'lexical':case['lexical'],'elapsed':case['elapsed_seconds']}),flush=True)
summary={'cases':len(cases),'accepted_reports':sum(c['accepted'] for c in cases),'reference_positive_terms':sum(c['proxy']['reference_positive_terms'] for c in cases),'matched_positive_terms':sum(c['proxy']['matched_positive_terms'] for c in cases),'unsupported_positive_terms_against_known_negatives':sum(c['proxy']['positive_terms_in_normal_reference'] for c in cases),'mean_bleu1':sum(c['lexical']['bleu1'] for c in cases)/len(cases),'mean_bleu4':sum(c['lexical']['bleu4'] for c in cases)/len(cases),'mean_rouge_l':sum(c['lexical']['rouge_l'] for c in cases)/len(cases),'mean_latency_seconds':sum(c['elapsed_seconds'] for c in cases)/len(cases)}
(out/f'{args.name}_summary.json').write_text(json.dumps(summary,indent=2))
print('COMPLETE',json.dumps(summary),flush=True)
