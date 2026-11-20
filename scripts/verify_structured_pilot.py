"""Verify completed pilot artifacts and input provenance without rerunning models."""
import hashlib,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chest_xpert.structured_pipeline import render_sections,score_prediction,summarize_predictions

def main():
 root=Path('checkpoints/structured_lora_pilot');r=json.loads((root/'guided_evaluation.json').read_text());records={v['study_id']:v for v in map(json.loads,Path(r['protocol']['dataset'],'validation.jsonl').read_text().splitlines())};inferences=json.loads((root/'guided_inference_partial.json').read_text())
 for protocol in ['protocol.json','guided_protocol.json']:
  for name,digest in json.loads((root/protocol).read_text())['source_sha256'].items():
   if name=='scripts/train_structured_pilot.py':
    repair=json.loads((root/'post_training_repair.json').read_text());assert hashlib.sha256(Path(repair['executed_source_copy']).read_bytes()).hexdigest()==digest==repair['executed_sha256'];assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==repair['repaired_sha256']
   else:assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
 for v in inferences:
  record=records[v['study_id']];audit=v['input_audit'];paths=record['inputs']['original_paths']+[record['inputs']['contact_sheet_path']]
  assert audit['image_paths']==paths
  assert audit['image_sha256']==[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]
  assert audit['cnn_scores']==record['inputs']['cnn_scores'] and len(audit['cnn_scores'])==14
  assert len(audit['image_grid_thw'])==len(paths)
  assert audit['generated_status_token_ids'] and v['prediction']['observations'][0]['status'] in ('present','absent','uncertain')
 for mode in ['base','candidate']:
  cases=r['classification'][mode]['cases']
  for case in cases:assert score_prediction(case['raw_output'],{'finding':case['finding'],'target':case['target']})==case['score']
  assert summarize_predictions(cases)==r['classification'][mode]['metrics']
  for report in r['reports'][mode]:
   study=report['study_id'];sections=render_sections([i['prediction'] for i in report['inferences']],records[study]['inputs']['clinical_metadata']);assert sections==report['sections']
   clinical=' '.join(s['text'] for s in sections);assert not re.search(r'grad.?cam|\bcnn\b|\bvlm\b|heatmap',clinical,re.I)
   assert (root/f'{study}_{mode}_guided_report.pdf').stat().st_size>1000
 active=json.loads(Path('checkpoints/active_models.json').read_text());assert active['report_adapter'] is None
 assert hashlib.sha256((root/'adapter_model.safetensors').read_bytes()).hexdigest()==r['candidate_weights_sha256']
 summary={'verified':True,'real_model_inferences':len(inferences),'input_types':['all original radiographs','CNN scores (14)','Grad-CAM contact sheet'],'classification_scores_recomputed':True,'clinical_sections_reconstructed_without_reference_findings':True,'technical_terms_absent_in_clinical_sections':True,'active_adapter_unchanged':True,'limitations':r['limitations']}
 Path('reports/structured_pilot/verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
