"""Replay source spans, input hashes, masks and response schemas; export review pack."""
import argparse,csv,hashlib,html,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chest_xpert.structured_findings import STATES,CATALOG,bind_annotations,group_queries,build_structured_messages
from chest_xpert.structured_schema import FindingResponse
from chest_xpert.study_images import resolve_study_images,index_study_images
from weasyprint import HTML

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return list(map(json.loads,Path(path).read_text().splitlines()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='data/structured_findings_v1_1');p.add_argument('--output',default='reports/structured_targets');p.add_argument('--curation',default='reports/structured_targets/curation_v1_1.json');args=p.parse_args();root=Path(args.dataset);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((root/'manifest.json').read_text());records=read(root/'train.jsonl')+read(root/'validation.jsonl');queries=read(root/'train_queries.jsonl')+read(root/'validation_queries.jsonl');byid={r['study_id']:r for r in records};assert len(byid)==len(records)
 assert sha(args.curation)==manifest['curation_sha256'],'Curation changed after freeze'
 curated={c['study_id']:c for c in json.loads(Path(args.curation).read_text())['cases']}
 assert set(curated)==set(byid)
 schema=FindingResponse.model_json_schema();(root/'response_schema.json').write_text(json.dumps(schema,indent=2))
 for source,digest in manifest['source_files_sha256'].items():assert sha(source)==digest,source+' changed after freeze'
 ids={split:{r['study_id'] for r in records if r['split']==split} for split in ['train','validation']};assert not ids['train']&ids['validation']
 counts=Counter();coverage={};index=index_study_images('data/iu_xray/images')
 for r in records:
  for path,digest in zip(r['inputs']['original_paths']+[r['inputs']['contact_sheet_path']],r['input_provenance']['sha256']):assert sha(path)==digest,path
  assert sha(r['source_xml'])==r['source_xml_sha256']
  assert r['inputs']['original_paths']==list(map(str,resolve_study_images(r['source_xml'],'data/iu_xray/images',index)))
  sections={s['heading']:s['text'] for s in r['reference']['sections']}
  assert r['annotations']==bind_annotations(r['reference']['sections'],curated[r['study_id']]['annotations'])
  for a in r['annotations']:
   e=a['evidence'];assert sections[e['section']][e['start']:e['end']]==e['quote']
   target=a['target'];assert target['status'] in STATES
   assert all(a['field_loss_mask'][field]==(field in target) for field in a['field_loss_mask'])
   counts[(r['split'],target['status'])]+=1
  for fact,s in r['supervision'].items():
   if not s['loss_mask']:assert s['status'] is None and not s['supervised']
  for fact in CATALOG:
   coverage.setdefault(fact,{split:{state:0 for state in STATES} for split in ['train','validation']})
   for a in r['annotations']:
    if a['finding']==fact:coverage[fact][r['split']][a['target']['status']]+=1
 for q in queries:
  r=byid[q['record_id']];assert r['split']==q['split'];FindingResponse.model_validate(q['target'])
  expected=next(x for x in group_queries(r['study_id'],r['annotations']) if x['finding']==q['finding'])
  assert q['target']==expected['target'] and q['field_loss_masks']==expected['field_loss_masks'] and q['source_evidence']==expected['source_evidence']
  assert all('quote' not in obs and 'source_evidence' not in obs for obs in q['target']['observations'])
  messages,images=build_structured_messages(r,q);text=messages[1]['content'][-1]['text']
  for e in q['source_evidence']:
   if len(e['quote'])>35:assert e['quote'] not in text
  for image in images:image.close()
  assert q['sampling_weight']>0
 files={str(p):sha(p) for p in root.glob('*.json*')}
 report={'dataset':str(root),'studies':manifest['studies'],'query_counts':manifest['query_counts'],'observation_state_counts':manifest['observation_state_counts'],'original_images':manifest['original_images'],'all_source_spans_verified':True,'all_input_hashes_verified':True,'all_query_targets_pass_strict_schema':True,'source_answers_excluded_from_inputs':True,'unannotated_slots_masked':manifest['unannotated_slots_masked'],'explicit_supporting_view_labels':manifest['explicit_supporting_view_labels'],'independent_clinical_adjudication':False,'four_state_coverage_complete':manifest['four_state_coverage_complete'],'training_started':False,'automatic_training_release':False,'per_finding_coverage':coverage,'dataset_file_sha256':files,'verification_source_sha256':{f:sha(f) for f in ['chest_xpert/structured_schema.py','scripts/verify_structured_findings.py']},
  'next_labeling_needs':['Genuine finding-specific unassessable examples are missing; do not fabricate them from source silence or generic limitations.','Supporting-view supervision has only two explicit source examples.','Some findings lack positive validation labels; their recall cannot be estimated from this pilot.','Use a new versioned protocol and a checkpointed structured-query trainer before any GPU experiment.']}
 (out/'verification.json').write_text(json.dumps(report,indent=2))
 esc=lambda x:html.escape(str(x))
 def table(headers,rows):return '<table><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
 body='<h1>Structured finding targets — source verification</h1><p class="note">AI-assisted source-text review; not independent radiologist or image-based clinical adjudication. No model has been trained or activated from this dataset. Labels describe source assertions, not established patient diagnoses.</p>'
 body+=table(['Split','Studies','Finding queries','Present observations','Absent observations','Uncertain observations','Unassessable observations'],[(s,manifest['studies'][s],manifest['query_counts'][s],*[manifest['observation_state_counts'][s][state] for state in STATES]) for s in ['train','validation']])
 body+='<h2>Verification</h2>'+table(['Check','Result'],[['Original images',manifest['original_images']],['Source spans / input hashes / query schema','All verified'],['Unknown finding slots masked',manifest['unannotated_slots_masked']],['Finding-specific supporting-view labels',manifest['explicit_supporting_view_labels']],['Four-state coverage complete',False],['Active model changed / training started','No']])
 body+='<h2>Interpretation rules</h2><ul><li>Missing mention is unsupervised, never absent or unassessable.</li><li>Cannot exclude is uncertain; general limitations do not make every finding unassessable.</li><li>No large effusion preserves the large-only scope. No focal/definite/acute finding preserves its restriction.</li><li>Pericardial effusion is not a pleural-effusion positive. Low-normal volumes are not hypoinflation.</li><li>Supporting view is supervised only when the source explicitly attributes that finding to a view.</li><li>Evidence quotes are for audit only; neither user prompts nor model answer targets contain source report prose.</li><li>Sampling weights normalize study contribution and balance available supervision strata; they have not yet been used to train a model.</li></ul>'
 body+='<h2>Remaining label requirements</h2><ul>'+''.join('<li>'+esc(s)+'</li>' for s in report['next_labeling_needs'])+'</ul>'
 body+='<section><h1>Quarantine</h1>'+table(['Study','Original split','Reason','Replacement'],[(q['study_id'],q['original_split'],q['reason'],q['replacement']) for q in json.loads((root/'quarantine.json').read_text())])+'<h2>Per-finding coverage</h2>'+table(['Finding','Training present / absent / uncertain / unassessable','Validation present / absent / uncertain / unassessable'],[(k,' / '.join(str(v['train'][s]) for s in STATES),' / '.join(str(v['validation'][s]) for s in STATES)) for k,v in coverage.items() if any(v[split][s] for split in v for s in STATES)])+'</section>'
 for r in records:
  body+='<section><h1>'+esc(r['study_id'])+' — '+esc(r['split'])+'</h1><p>'+esc(' '.join(r['review']['notes']))+'</p>'+table(['Finding / status / scope','Location / degree / view','Exact source evidence'],[(a['finding']+' — '+a['target']['status']+'\nScope: '+json.dumps(a['target']['assessment_scope']),json.dumps({k:v for k,v in a['target'].items() if k not in ['status','assessment_scope']},ensure_ascii=False),a['evidence']['section']+f" [{a['evidence']['start']}:{a['evidence']['end']}]\n"+a['evidence']['quote']) for a in r['annotations']])+'<p>Source: '+esc(r['source_xml'])+'</p></section>'
 doc='<!doctype html><html><head><meta charset="utf-8"><style>@page{size:A4 landscape;margin:12mm}body{font:10px Arial;color:#17324a}h1{font-size:20px}h2{font-size:14px}.note{background:#fff4d9;padding:10px}table{width:100%;table-layout:fixed;border-collapse:collapse;margin:10px 0}td,th{border:1px solid #abc;padding:6px;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}th{background:#edf4f6}section{break-before:page}tr{break-inside:avoid}</style></head><body>'+body+'</body></html>'
 (out/'review.html').write_text(doc);HTML(string=doc).write_pdf(out/'review.pdf')
 # A blank worksheet supports independent review without suggesting a desired label.
 with (out/'clinical_review_worksheet.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['study_id','split','finding','source_context','original_paths','reviewed_status','supporting_view','reviewer','review_notes']);writer.writeheader()
  for r in records:
   for fact in ['Cardiomegaly','Effusion','Consolidation','Pneumothorax']:
    if any(a['finding']=='Technical limitation' for a in r['annotations']):
     writer.writerow({'study_id':r['study_id'],'split':r['split'],'finding':fact,'source_context':' '.join(a['evidence']['quote'] for a in r['annotations'] if a['finding']=='Technical limitation'),'original_paths':json.dumps(r['inputs']['original_paths'])})
 print(json.dumps({k:v for k,v in report.items() if k not in ['per_finding_coverage','dataset_file_sha256']},indent=2))
if __name__=='__main__':main()
