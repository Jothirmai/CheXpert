"""Score saved ablation text after deterministic JSON normalization.

Raw-schema success is reported separately; no generation is hidden or replaced.
"""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chest_xpert.report_quality import parse_generated_sections
from scripts.assess_report_adapter import analyze
from cliq_rrg.eval import bleu,rouge_l
folder=Path('checkpoints/vlm_evidence_ablation')
data=json.loads((folder/'results.json').read_text())
config=json.loads((folder/'configuration.json').read_text())
rows={r['study_id']:r for r in (json.loads(line) for line in Path('data/report_training_v3/validation.jsonl').read_text().splitlines() if line.strip())}
summary={}
for mode in ['cnn_evidence','image_only']:
 result={'cases':len(data['cases']),'raw_schema_valid':0,'raw_clinical_json_valid':0,'accepted_after_normalization':0,'rejected_reports':0,'reference_positive_terms':0,'matched_positive_terms':0,'missed_positive_terms':0,'unsupported_positive_terms_against_known_negatives':0,'false_normal_cases':0,'mean_bleu1':0,'mean_bleu4':0,'mean_rouge_l':0}
 for case in data['cases']:
  output=case['outputs'][mode];result['raw_clinical_json_valid']+=int(output['proxy']['valid_clinical_json'])
  try:
   from backend.api.v1.workflow import Section
   raw=json.loads(re.search(r'\{.*\}', output['text'], re.S).group())
   raw_sections=[Section.model_validate(s).model_dump() for s in raw['sections']]
   result['raw_schema_valid']+=int([s['heading'] for s in raw_sections]==[s['heading'] for s in case['reference']['sections']])
  except (ValueError,KeyError,TypeError,AttributeError):pass
  try:
   sections,changed=parse_generated_sections(output['text'],[s['heading'] for s in case['reference']['sections']])
   check=analyze(json.dumps({'sections':sections}),case['reference'],rows[case['study_id']].get('mesh_terms'))
   result['accepted_after_normalization']+=1
   ref=' '.join(s['text'] for s in case['reference']['sections'] if s['heading'] in ['Findings','Impression'])
   pred=' '.join(s['text'] for s in sections if s['heading'] in ['Findings','Impression'])
   result['mean_bleu1']+=bleu(ref,pred,1)/result['cases'];result['mean_bleu4']+=bleu(ref,pred,4)/result['cases'];result['mean_rouge_l']+=rouge_l(ref,pred)/result['cases']
   output['normalized_sections']=sections
  except (ValueError,KeyError,TypeError):
   result['rejected_reports']+=1
   check=output['proxy']
  output['normalized_proxy']=check
  result['reference_positive_terms']+=check['reference_positive_terms'];result['matched_positive_terms']+=check['matched_positive_terms'];result['unsupported_positive_terms_against_known_negatives']+=check['positive_terms_in_normal_reference'];result['false_normal_cases']+=int(check['false_normal_on_positive_reference'])
 result['missed_positive_terms']=result['reference_positive_terms']-result['matched_positive_terms']
 result['positive_term_recall']=result['matched_positive_terms']/result['reference_positive_terms'] if result['reference_positive_terms'] else None
 summary[mode]=result
(folder/'results_scored.json').write_text(json.dumps(data,indent=2))
(folder/'normalized_summary.json').write_text(json.dumps({'metrics':summary,'limitations':'Eight validation studies. Raw schema validity, raw clinical-output validity and acceptance after deterministic normalization are separate. Rejected reports score zero lexical overlap and zero matched positives; unsupported-finding counts cover accepted reports only. Assertion extraction is lexical, incomplete, and not independently adjudicated. Unmentioned reference findings are unknown except for normal-tagged reports. No clinical accuracy claim.'},indent=2))
print(json.dumps(summary,indent=2))
