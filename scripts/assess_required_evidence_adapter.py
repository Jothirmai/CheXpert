"""Score the three-input training pilot; preserve failures and input evidence."""
import hashlib,html,json,re,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chest_xpert.report_quality import parse_generated_sections
from scripts.assess_report_adapter import analyze
from cliq_rrg.eval import bleu,rouge_l
from chest_xpert.model_selection import selected_cnn
folder=Path('checkpoints/report_lora_required_evidence')
data=json.loads((folder/'evaluation.json').read_text());configuration=json.loads((folder/'run_config.json').read_text())
manifest=Path(configuration['validation'])
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==configuration['manifest_sha256']['validation']
rows={r['study_id']:r for r in (json.loads(line) for line in manifest.read_text().splitlines() if line.strip())}
summary={};details=[]
for mode in ['base','adapter']:
 totals={'cases':len(data['comparisons']),'accepted_reports':0,'raw_clinical_json_valid':0,'reference_positive_terms':0,'matched_positive_terms':0,'unsupported_terms_against_known_negatives':0,'mean_bleu1':0.0,'mean_bleu4':0.0,'mean_rouge_l':0.0}
 for case in data['comparisons']:
  raw=case[mode+'_output'];check=analyze(raw,case['reference'],rows[case['study_id']].get('mesh_terms'));totals['raw_clinical_json_valid']+=int(check['valid_clinical_json'])
  try:
   sections,normalized=parse_generated_sections(raw,[s['heading'] for s in case['reference']['sections']])
   check=analyze(json.dumps({'sections':sections}),case['reference'],rows[case['study_id']].get('mesh_terms'))
   totals['accepted_reports']+=1
   ref=' '.join(s['text'] for s in case['reference']['sections'] if s['heading'] in ['Findings','Impression'])
   pred=' '.join(s['text'] for s in sections if s['heading'] in ['Findings','Impression'])
   totals['mean_bleu1']+=bleu(ref,pred,1)/totals['cases'];totals['mean_bleu4']+=bleu(ref,pred,4)/totals['cases'];totals['mean_rouge_l']+=rouge_l(ref,pred)/totals['cases']
   case[mode+'_sections']=sections
  except (ValueError,KeyError,TypeError):pass
  totals['reference_positive_terms']+=check['reference_positive_terms'];totals['matched_positive_terms']+=check['matched_positive_terms'];totals['unsupported_terms_against_known_negatives']+=check['positive_terms_in_normal_reference']
  details.append({'study_id':case['study_id'],'model':mode,'proxy':check})
 totals['missed_reference_positive_terms']=totals['reference_positive_terms']-totals['matched_positive_terms']
 summary[mode]=totals
b,a=summary['base'],summary['adapter']
eligible=bool(data['validation_loss_improved'] and a['accepted_reports']==a['cases'] and a['matched_positive_terms']>b['matched_positive_terms'] and a['unsupported_terms_against_known_negatives']<=b['unsupported_terms_against_known_negatives'])
assessment={'input_contract':'CNN scores + original image + Grad-CAM contact sheet in every base, training and adapter generation','base_validation_loss':data['base_validation_loss'],'adapter_validation_loss':data['adapter_validation_loss'],'validation_loss_images':data['validation_images'],'training_images':data['train_images'],'optimizer_steps':data['steps'],'metrics':summary,'eligible_for_experimental_selection':eligible,'auto_activated':False,'details':details,'limitations':'Small 32-image/eight-step training pilot; token loss on eight validation images and generated comparison on four validation studies. Lexical finding proxies are incomplete and weak, not clinical accuracy. Normalized reports are counted separately from raw validity; failed reports score zero lexical overlap and matched positives.'}
(folder/'assessment.json').write_text(json.dumps(assessment,indent=2))
(folder/'scored_evaluation.json').write_text(json.dumps(data,indent=2))
out=Path('reports/required_evidence_training');out.mkdir(parents=True,exist_ok=True)
(out/'metrics.json').write_text(json.dumps(assessment,indent=2))
esc=lambda x:html.escape(str(x))
def table(headers,values):return '<table><tr>'+''.join('<th>'+esc(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in values)+'</table>'
body='<h1>Three-input VLM training pilot</h1><p>All training and inference calls use CNN scores, the original X-ray and Grad-CAM. No image-only mode. No reference findings or impression are supplied during generated validation comparisons.</p>'
cnn_validation=json.loads(Path('checkpoints/cnn_corrected/validation_metrics.json').read_text())
cnn_test=json.loads(Path('checkpoints/cnn_corrected/test_metrics.json').read_text())
body+='<h2>Existing CNN metrics</h2>'+table(['Split','Studies','Evaluable findings','Macro AUROC','Macro AP'],[[label,score['studies'],score['evaluable_labels'],f"{score['macro_auroc']:.4f}",f"{score['macro_average_precision']:.4f}"] for label,score in [('Validation',cnn_validation),('Test',cnn_test)]])+'<p>The CNN checkpoint was unchanged in this VLM pilot. Labels are weak report/MeSH-derived annotations, not independently adjudicated clinical outcomes. AUROC is not percentage accuracy.</p><h2>VLM pilot metrics</h2>'
body+=table(['Metric','Base Qwen','Candidate LoRA'],[['Validation token loss',data['base_validation_loss'],data['adapter_validation_loss']]]+[[k,b[k],a[k]] for k in b])
body+='<p>'+esc(assessment['limitations'])+'</p><p>Experimental selection check: '+esc(eligible)+'. Automatic activation: false.</p>'
checkpoint=selected_cnn();config=json.loads((checkpoint.parent/'training_config.json').read_text());fingerprint=hashlib.sha256(checkpoint.read_bytes()+json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
def display_sections(raw):
 try:
  parsed=json.loads(re.search(r'\{.*\}',raw,re.S).group())
  if 'sections' in parsed:return {s['heading']:s['text'] for s in parsed['sections']}
  return {key:(value if isinstance(value,str) else '\n'.join(f'{k}: {v}' for k,v in value.items())) for key,value in parsed.items()}
 except (ValueError,TypeError,KeyError,AttributeError):return {'Findings':raw}
for case in data['comparisons']:
 study=case['study_id'];row=rows[study];reference={s['heading']:s['text'] for s in case['reference']['sections']}
 base=display_sections(case['base_output']);adapt=display_sections(case['adapter_output'])
 body+='<h2>'+esc(study)+'</h2>'+table(['Section','Dataset reference','Base Qwen','Candidate LoRA'],[[heading,reference.get(heading,''),base.get(heading,'Rejected output; see raw JSON'),adapt.get(heading,'Rejected output; see raw JSON')] for heading in ['Findings','Impression']])
 original=out/f'{study}_original.png';sheet=out/f'{study}_gradcam.png'
 shutil.copyfile(row['image_path'],original)
 cache=Path(configuration['cache'])/fingerprint/row['image_sha256']
 shutil.copyfile(cache.with_suffix('.png'),sheet)
 scores=json.loads(cache.with_suffix('.json').read_text())
 body+='<details><summary>Original image, Grad-CAM and CNN score inputs</summary><img src="'+esc(original.name)+'"><img class="sheet" src="'+esc(sheet.name)+'">'+table(['Finding','CNN score'],[[k,f'{v:.6f}'] for k,v in scores.items()])+'</details>'
style='body{font:12px Arial;margin:25px;color:#17324a}table{border-collapse:collapse;width:100%;margin:15px 0}td,th{border:1px solid #abc;padding:9px;vertical-align:top}th{background:#edf4f6}h2{break-after:avoid}img{max-height:350px;max-width:90%}.sheet{max-height:500px}@page{size:A4 landscape;margin:12mm}'
report='<!doctype html><html><head><meta charset="utf-8"><style>'+style+'</style></head><body>'+body+'</body></html>'
(out/'comparison.html').write_text(report)
# PDF uses the report text and tables; the expandable input gallery remains in HTML.
import re
pdf_report=re.sub(r'<details>.*?</details>','',report,flags=re.S)
from weasyprint import HTML
HTML(string=pdf_report,base_url=str(out.resolve())).write_pdf(out/'comparison.pdf')
print(json.dumps(assessment,indent=2))
