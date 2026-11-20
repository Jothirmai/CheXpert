"""Historical v1 gate, retained for reproduction. New audits use rescore_reports_v2.py.

Lexical proxies are explicitly not clinical accuracy.
"""
import argparse,html,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.assess_report_adapter import fact_states
from chest_xpert.report_quality import parse_generated_sections,report_warnings
from chest_xpert.disease_targets import NEG,UNCERTAIN
from cliq_rrg.eval import bleu,rouge_l
EXTRA={'Low lung volumes':r'low lung volumes?|hypoinflat\w*','Bronchovascular crowding':r'bronchovascular crowding','Rotation':r'rotat(?:ed|ion)','Kyphotic positioning':r'kyphotic','Limited inspiration':r'(?:limited|poor|suboptimal) inspirat\w*'}
def states(sections):
 result=fact_states(sections)
 for name,pattern in EXTRA.items():
  observations=[]
  for section in sections:
   if section['heading'].lower() not in ('findings','impression'):continue
   for sentence in re.split(r'[.;]|\bbut\b|\bhowever\b',section['text'].lower()):
    for match in re.finditer(pattern,sentence):
     prefix=sentence[:match.start()]
     observations.append(None if UNCERTAIN.search(prefix[-100:]) else 0 if NEG.search(prefix) else 1)
  result[name]=1 if 1 in observations else 0 if 0 in observations else None
 return result

def score(raw,row):
 truth=states(row['target']['sections']);positive={k for k,v in truth.items() if v==1}
 result={'accepted':False,'reference_positive_facts':len(positive),'matched_facts':0,'missed_facts':sorted(positive),'unsupported_facts':[],'contradictions':0,'bleu1':0,'bleu4':0,'rouge_l':0}
 try:sections,_=parse_generated_sections(raw,[s['heading'] for s in row['target']['sections']])
 except (ValueError,TypeError,KeyError):return result
 pred=states(sections);normal=any(t.lower()=='normal' for t in row.get('mesh_terms',[]))
 result.update(accepted=True,matched_facts=sum(pred.get(k)==1 for k in positive),missed_facts=sorted(k for k in positive if pred.get(k)!=1),unsupported_facts=sorted(k for k,v in pred.items() if v==1 and (truth.get(k)==0 or normal and truth.get(k) is None)),contradictions=len(report_warnings(sections,scorer_version="v1")))
 clinical=lambda ss:' '.join(s['text'] for s in ss if s['heading'].lower() in ('findings','impression'))
 ref=clinical(row['target']['sections']);text=clinical(sections)
 result.update(bleu1=bleu(ref,text,1),bleu4=bleu(ref,text,4),rouge_l=rouge_l(ref,text));return result

def summarize(cases):
 return {'cases':len(cases),'accepted_reports':sum(c['accepted'] for c in cases),'reference_positive_facts':sum(c['reference_positive_facts'] for c in cases),'matched_facts':sum(c['matched_facts'] for c in cases),'missed_facts':sum(len(c['missed_facts']) for c in cases),'unsupported_facts':sum(len(c['unsupported_facts']) for c in cases),'contradictions':sum(c['contradictions'] for c in cases),**{'mean_'+k:sum(c[k] for c in cases)/max(len(cases),1) for k in ('bleu1','bleu4','rouge_l')}}

def eligible_for_selection(data,totals,groups):
 b,a=totals['base'],totals['adapter']
 return bool(data['validation_loss_improved'] and a['cases']==12 and a['accepted_reports']==12 and b['reference_positive_facts']>=8 and a['missed_facts']<b['missed_facts'] and (a['unsupported_facts']<b['unsupported_facts'] if b['unsupported_facts'] else a['unsupported_facts']==0) and a['contradictions']<=b['contradictions'] and all(v['adapter']['missed_facts']<=v['base']['missed_facts'] and v['adapter']['unsupported_facts']<=v['base']['unsupported_facts'] for v in groups.values()))


def assess(folder):
 folder=Path(folder);data=json.loads((folder/'evaluation.json').read_text());audit=json.loads((folder/'cohort_audit.json').read_text())
 rows={r['study_id']:r for r in map(json.loads,(folder/'validation.jsonl').read_text().splitlines())};cases=data['comparisons']
 assert [c['study_id'] for c in cases]==audit['generated_validation_studies'],'Validation cohort changed'
 details=[{'study_id':c['study_id'],'category':audit['validation_categories'][c['study_id']],**{name:score(c[name+'_output'],rows[c['study_id']]) for name in ('base','adapter')}} for c in cases]
 totals={name:summarize([c[name] for c in details]) for name in ('base','adapter')}
 groups={group:{name:summarize([c[name] for c in details if c['category']==group]) for name in ('base','adapter')} for group in set(audit['validation_categories'].values())}
 b,a=totals['base'],totals['adapter']
 eligible=eligible_for_selection(data,totals,groups)
 result={'eligible_for_experimental_selection':eligible,'metrics':totals,'by_category':groups,'details':details,'base_validation_loss':data['base_validation_loss'],'adapter_validation_loss':data['adapter_validation_loss'],'limitations':'12 predeclared validation studies; conservative lexical proxies including low-volume and positioning descriptions. Incomplete factual vocabulary and weak annotations, not expert image-based clinical accuracy. Test reports are excluded from selection.'}
 cnn=Path('checkpoints/full_study_cnn_validation/results.json')
 if cnn.exists():
  cnn_data=json.loads(cnn.read_text());result['cnn_validation']={'studies':cnn_data['validation_studies'],'original_views':cnn_data['original_views'],'metrics':cnn_data['metrics']['parent']}
 test=folder/'reserved_test_results.json'
 if test.exists():result['reserved_test']={k:v for k,v in json.loads(test.read_text()).items() if k!='cases'}
 active=Path('checkpoints/active_models.json')
 if active.exists():result['active_report_adapter']=json.loads(active.read_text()).get('report_adapter')
 (folder/'assessment.json').write_text(json.dumps(result,indent=2))
 lines=['# Complete-study adapter comparison','','Reference reports were not supplied as generation answers. Raw outputs and failed cases are preserved.','','```json',json.dumps(result,indent=2),'```']
 for case in cases:
  lines.extend(['',f"## {case['study_id']}",'','### Reference',json.dumps(case['reference'],indent=2),'','### Base',case['base_output'],'','### Candidate',case['adapter_output']])
 (folder/'COMPARISON.md').write_text('\n'.join(lines))
 esc=lambda x:html.escape(str(x));body='<h1>Complete-study report training evaluation</h1><p>'+esc(result['limitations'])+'</p><pre>'+esc(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))+'</pre>'
 for case in cases:
  body+='<h2>'+esc(case['study_id'])+'</h2><table><tr><th>Reference</th><th>Base raw output</th><th>Candidate raw output</th></tr><tr>'+''.join('<td><pre>'+esc(v)+'</pre></td>' for v in [json.dumps(case['reference'],indent=2),case['base_output'],case['adapter_output']])+'</tr></table>'
 document='<html><head><meta charset="utf-8"><style>@page{size:A4 landscape;margin:12mm}body{font:11px Arial}table{width:100%;table-layout:fixed;border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style></head><body>'+body+'</body></html>'
 (folder/'comparison.html').write_text(document)
 from weasyprint import HTML
 HTML(string=document).write_pdf(folder/'comparison.pdf')
 import shutil
 public=Path('reports/full_study_training');public.mkdir(parents=True,exist_ok=True)
 for source,target in [('comparison.html','comparison.html'),('comparison.pdf','comparison.pdf'),('assessment.json','metrics.json'),('COMPARISON.md','COMPARISON.md')]:
  shutil.copyfile(folder/source,public/target)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);args=p.parse_args();print(json.dumps(assess(args.experiment),indent=2))
