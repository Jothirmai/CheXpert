"""Separate schema validity from class decisions using closed status decoding."""
import hashlib,html,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from chest_xpert.structured_decoding import decode_status
from chest_xpert.structured_pipeline import score_prediction,summarize_predictions,render_sections
from chest_xpert.report_scoring_v2 import score as report_score,summarize as report_summary
from backend.services import review_store
from weasyprint import HTML

def read(path):return list(map(json.loads,Path(path).read_text().splitlines()))
def main():
 out=Path('checkpoints/structured_lora_pilot');protocol=json.loads((out/'guided_protocol.json').read_text());
 for name,digest in protocol['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
 data=Path(protocol['dataset']);records={r['study_id']:r for r in read(data/'validation.jsonl')};original_metadata={k:dict(v['inputs']['clinical_metadata']) for k,v in records.items()}
 for r in records.values():r['inputs']['clinical_metadata']={}
 queries={q['query_id']:q for q in read(data/'validation_queries.jsonl')};probes=[queries[q] for q in protocol['probe_queries']]
 model_id='Qwen/Qwen2.5-VL-7B-Instruct';torch.set_num_threads(4)
 processor=AutoProcessor.from_pretrained(model_id,min_pixels=128*28*28,max_pixels=256*28*28,use_fast=False,local_files_only=True)
 base=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id,torch_dtype=torch.float16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True)
 model=PeftModel.from_pretrained(base,str(out)).eval();results={};reports={};cache={}
 def infer(mode,study,finding):
  key=(mode,study,finding)
  if key in cache:return cache[key]
  started=time.perf_counter()
  if mode=='base':
   with model.disable_adapter():pred,audit=decode_status(model,processor,records[study],finding)
  else:pred,audit=decode_status(model,processor,records[study],finding)
  r={'study_id':study,'finding':finding,'prediction':pred,'raw_output':json.dumps(pred,separators=(',',':')),'input_audit':audit,'seconds':time.perf_counter()-started};cache[key]=r
  (out/'guided_inference_partial.json').write_text(json.dumps(list(cache.values()),indent=2));print('GUIDED',len(cache),mode,study,finding,pred['observations'][0]['status'],flush=True);return r
 for mode in ['base','candidate']:
  cases=[]
  for q in probes:
   pred=infer(mode,q['study_id'],q['finding']);cases.append({**pred,'query_id':q['query_id'],'target':q['target'],'score':score_prediction(pred['raw_output'],q)})
  results[mode]={'metrics':summarize_predictions(cases),'cases':cases};(out/'guided_results_partial.json').write_text(json.dumps(results,indent=2))
  mode_reports=[]
  for study in protocol['report_studies']:
   preds=[infer(mode,study,finding) for finding in protocol['report_findings']];sections=render_sections([p['prediction'] for p in preds],original_metadata[study])
   html_report=review_store.render({'patient_id':study,'sections':sections,'signature':None});(out/f'{study}_{mode}_guided_report.html').write_text(html_report);HTML(string=html_report).write_pdf(out/f'{study}_{mode}_guided_report.pdf')
   proxy=report_score(json.dumps({'sections':sections}),{'target':records[study]['reference']})
   mode_reports.append({'study_id':study,'sections':sections,'reference':records[study]['reference'],'proxy':proxy,'inferences':preds})
  reports[mode]=mode_reports;(out/'guided_reports_partial.json').write_text(json.dumps(reports,indent=2))
 b,a=results['base']['metrics'],results['candidate']['metrics'];rb=report_summary([r['proxy'] for r in reports['base']]);ra=report_summary([r['proxy'] for r in reports['candidate']])
 passed=bool(a['positive_recovered']>b['positive_recovered'] and a['invented_positive']<=b['invented_positive'] and a['class_correct']>=b['class_correct'] and a['scope_correct']>=b['scope_correct'] and ra['matched_facts']>=rb['matched_facts'] and ra['unsupported_facts']<=rb['unsupported_facts'] and not a['absent_only'])
 result={'protocol':protocol,'classification':results,'reports':reports,'report_metrics':{'base':rb,'candidate':ra},'passed_bounded_research_check':passed,'automatically_activated':False,'decoding_scope':'Status-only. Laterality, region, degree and view are not generated or borrowed from references. Structural validity is guaranteed by decoding and is not a learned accuracy metric.','limitations':'12 finding queries and three demonstration studies, adaptively used validation data, not clinical validation or a reserved test result. The unassessable class remains masked.'}
 cnn=json.loads(Path('checkpoints/full_study_cnn_validation/results.json').read_text());result['cnn_validation']={'studies':cnn['validation_studies'],'original_images':cnn['original_views'],'metrics':cnn['metrics']['parent']}
 result['candidate_weights_sha256']=hashlib.sha256((out/'adapter_model.safetensors').read_bytes()).hexdigest()
 result['classification_interpretation']='Class agreement with source-derived labels, not diagnostic accuracy. invented_positive includes definite claims where the source is uncertain; it is not a clinical false-positive rate.'
 (out/'guided_evaluation.json').write_text(json.dumps(result,indent=2))
 public=Path('reports/structured_pilot');public.mkdir(parents=True,exist_ok=True);esc=lambda v:html.escape(str(v))
 def table(headers,rows):return '<table><tr>'+''.join('<th>'+esc(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
 body='<h1>Structured finding pilot — completed experiment</h1><p>'+esc(result['decoding_scope'])+'</p><p>'+esc(result['limitations'])+'</p>'
 body+='<h2>Unchanged CNN validation</h2>'+table(['Studies','Original images','Evaluable labels','Macro AUROC','Average precision'],[[cnn['validation_studies'],cnn['original_views'],cnn['metrics']['parent']['evaluable_labels'],cnn['metrics']['parent']['macro_auroc'],cnn['metrics']['parent']['macro_average_precision']]])+'<h2>Structured finding query results</h2><p>'+esc(result['classification_interpretation'])+'</p>'
 body+=table(['Metric','Base','Candidate'],[(k,b[k],a[k]) for k in b])+'<h2>Bounded research gate</h2><p>'+esc(passed)+'. Active model has not been replaced.</p>'
 free=out/'evaluation.json'
 if free.exists():body+='<h2>Original free-JSON inference (preserved)</h2><pre>'+esc(json.dumps(json.loads(free.read_text()),indent=2))+'</pre>'
 for study in protocol['report_studies']:
  source=next(r for r in reports['base'] if r['study_id']==study);candidate=next(r for r in reports['candidate'] if r['study_id']==study)
  asdict=lambda ss:{s['heading']:s['text'] for s in ss}
  ref=asdict(source['reference']['sections']);before=asdict(source['sections']);after=asdict(candidate['sections'])
  body+='<section><h2>'+esc(study)+'</h2>'+table(['Section','Dataset reference','Base structured pipeline','Trained structured pipeline'],[(h,ref[h],before[h],after[h]) for h in ['Findings','Impression']])+'</section>'
 doc='<html><head><meta charset="utf-8"><style>@page{size:A4 landscape;margin:12mm}body{font:11px Arial;color:#17324a}table{width:100%;table-layout:fixed;border-collapse:collapse;margin:10px 0}td,th{border:1px solid #abc;padding:7px;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}th{background:#edf4f6}pre{white-space:pre-wrap}section{break-before:page}tr{break-inside:avoid}</style></head><body>'+body+'</body></html>'
 (public/'comparison.html').write_text(doc);HTML(string=doc).write_pdf(public/'comparison.pdf');(public/'metrics.json').write_text(json.dumps(result,indent=2))
 print('COMPLETE_GUIDED',json.dumps({'base':b,'candidate':a,'research_check':passed,'report_metrics':result['report_metrics']}),flush=True)
if __name__=='__main__':main()
