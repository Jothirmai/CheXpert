"""Rescore frozen predictions with v2; preserve v1 and forbid post-hoc activation."""
import argparse,hashlib,html,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chest_xpert.report_scoring_v2 import VERSION,extract,score,summarize
from chest_xpert.report_quality import parse_generated_sections
from weasyprint import HTML

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',default='checkpoints/report_lora_full_study');p.add_argument('--output',default='reports/scorer_v2');args=p.parse_args();exp=Path(args.experiment);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 preserved={str(p):digest(p) for p in [exp/'evaluation.json',exp/'assessment.json',exp/'cohort_audit.json',Path('checkpoints/active_models.json')]}
 evaluation=json.loads((exp/'evaluation.json').read_text());old=json.loads((exp/'assessment.json').read_text());cohort=json.loads((exp/'cohort_audit.json').read_text());rows={r['study_id']:r for r in map(json.loads,(exp/'validation.jsonl').read_text().splitlines())}
 details=[]
 for c in evaluation['comparisons']:
  states=extract(c['reference']['sections']);category=cohort['validation_categories'][c['study_id']]
  revised=category
  if category=='technical' and states['Spinal kyphosis']['state']=='positive' and all(states[k]['state']=='unmentioned' for k in ['Technical limitation','Rotation','Kyphotic positioning','Limited inspiration']):revised='anatomy_chronic'
  details.append({'study_id':c['study_id'],'original_category':category,'revised_category':revised,**{mode:score(c[mode+'_output'],rows[c['study_id']]) for mode in ['base','adapter']}})
 totals={m:summarize([c[m] for c in details]) for m in ['base','adapter']}
 result={'scorer_version':VERSION,'scorer_sha256':digest('chest_xpert/report_scoring_v2.py'),'historical_v1_metrics':old['metrics'],'metrics':totals,'details':details,'post_hoc_rescoring':True,'automatic_activation_permitted':False,'historical_artifacts_preserved':preserved,'model_predictions_changed':False,'interpretation':'Same saved outputs under the same revised scoring rules for both models. Changed denominators and definitions are evaluator corrections, not model improvements. The adapter still misses all recognized positive and qualified reference facts and stays rejected.','limitations':'Lexical proxies with incomplete anatomy/relationship coverage; not independent clinical adjudication. Source silence is unknown; unverified extra claims are separate from claims against explicit negatives. Redacted source qualifiers cannot be reconstructed.'}
 (out/'assessment.json').write_text(json.dumps(result,indent=2))
 esc=lambda v:html.escape(str(v))
 def table(headers,rows):return '<table><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</table>'
 body='<h1>Version 2 report scoring</h1><p>'+esc(result['interpretation'])+'</p><p>'+esc(result['limitations'])+'</p>'
 keys=['cases','reference_positive_facts','reference_qualified_facts','matched_facts','missed_facts','missed_qualified_facts','explicit_denials','location_errors','qualification_errors','unsupported_facts','qualified_claims_against_reference_negatives','unverified_additional_positive_claims','contradictions']
 body+=table(['Measure','Base','Adapter'],[(k,totals['base'][k],totals['adapter'][k]) for k in keys])
 body+='<h2>What changed</h2><ul><li>Normal-heart and enlargement synonyms; low-volume word order; hyperexpanded lungs and crowding synonyms.</li><li>Probable/likely/suggesting uncertainty and unremarkable-for negation.</li><li>Spinal kyphosis separated from acquisition positioning; explicit technical limitations and hardware included.</li><li>Wrong side/region loses full fact credit; weakened/overstated certainty is separate.</li><li>Reference silence remains unknown, even with a normal MeSH tag.</li><li>All original v1 files and model selection are preserved. This post-hoc rescore cannot activate an adapter.</li></ul>'
 for c,d in zip(evaluation['comparisons'],details):
  body+='<section><h2>'+esc(c['study_id'])+'</h2><p>Original category: '+esc(d['original_category'])+'; revised semantic category: '+esc(d['revised_category'])+'</p>'
  body+=table(['Review item','Base','Adapter'],[(k,d['base'][k],d['adapter'][k]) for k in ['fully_matched_facts','missed_facts','missed_qualified_facts','explicit_denials','location_errors','qualification_errors','unsupported_facts','qualified_claims_against_reference_negatives','unverified_additional_positive_claims']])
  ref={s['heading']:s['text'] for s in c['reference']['sections']};modes={m:{s['heading']:s['text'] for s in parse_generated_sections(c[m+'_output'],[s['heading'] for s in c['reference']['sections']])[0]} for m in ['base','adapter']}
  body+=table(['Section','Reference','Base','Adapter'],[(h,ref[h],modes['base'][h],modes['adapter'][h]) for h in ['Findings','Impression']])+'</section>'
 doc='<!doctype html><html><head><meta charset="utf-8"><style>@page{size:A4 landscape;margin:12mm}body{font:10px Arial;color:#17324a}table{width:100%;border-collapse:collapse;table-layout:fixed;margin:10px 0}td,th{border:1px solid #abc;padding:6px;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}th{background:#edf4f6}section{break-before:page}tr{break-inside:avoid}</style></head><body>'+body+'</body></html>'
 (out/'comparison.html').write_text(doc);HTML(string=doc).write_pdf(out/'comparison.pdf')
 assert all(digest(p)==h for p,h in preserved.items())
 print(json.dumps({'scorer_version':VERSION,'metrics':totals,'historical_artifacts_unchanged':True},indent=2))
if __name__=='__main__':main()
