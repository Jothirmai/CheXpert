"""Describe controlled input sensitivity without claiming causal clinical accuracy."""
import hashlib,html,itertools,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chest_xpert.report_quality import parse_generated_sections
from chest_xpert.report_scoring_v2 import VERSION,extract,score
from weasyprint import HTML
FACTORS=['original_source','cam_source','score_source']
HEADINGS=['Clinical indication','Comparison','Technique','Findings','Impression']

def normalize(text):return re.sub(r'\s+',' ',text.casefold()).strip()
def profile(sections):
 assertions=extract(sections)
 return {k:{'state':v['state'],'locations':sorted({loc for m in v['mentions'] for loc in m.get('locations',[]) if m['state'] in ['positive','uncertain']}),'degrees':sorted({m['degree'] for m in v['mentions'] if m.get('degree') and m['state'] in ['positive','uncertain']})} for k,v in assertions.items()}

def sensitivity(records):
 lookup={(r['mode'],*(r[f] for f in FACTORS)):r for r in records};result={}
 for mode in ['base','adapter']:
  result[mode]={}
  for i,factor in enumerate(FACTORS):
   pairs=[]
   for other in itertools.product(['normal','abnormal'],repeat=2):
    keys=list(other);keys.insert(i,'normal');a=lookup[(mode,*keys)];keys[i]='abnormal';b=lookup[(mode,*keys)]
    valid=a['accepted'] and b['accepted']
    pairs.append({'normal_condition':a['condition_id'],'abnormal_condition':b['condition_id'],'both_reports_accepted':valid,'clinical_text_changed':a.get('clinical_text')!=b.get('clinical_text') if valid else None,'assertion_profile_changed':a.get('assertion_profile')!=b.get('assertion_profile') if valid else None})
   result[mode][factor]={'paired_comparisons':4,'valid_pairs':sum(p['both_reports_accepted'] for p in pairs),'clinical_text_changes':sum(p['clinical_text_changed'] is True for p in pairs),'assertion_profile_changes':sum(p['assertion_profile_changed'] is True for p in pairs),'pairs':pairs}
 return result

def main():
 out=Path('reports/grounding_diagnostic');data=json.loads((out/'results.json').read_text());raw=data['conditions'];assert len(raw)==16
 rows={r['study_id']:r for r in map(json.loads,Path('checkpoints/report_lora_full_study/enriched_validation.jsonl').read_text().splitlines())};pair={name:rows[study] for name,study in data['configuration']['pair'].items()};records=[]
 for i,r in enumerate(raw):
  audit=r['input_audit'];assert len(audit['image_grid_thw'])==3
  paths=audit['original_paths']+[audit['contact_sheet']]
  assert audit['original_paths']==pair[r['original_source']]['image_paths'] and audit['contact_sheet']==pair[r['cam_source']]['contact_sheet_path'] and audit['cnn_scores']==pair[r['score_source']]['cnn_scores']
  for path,digest in zip(paths,audit['image_sha256']):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
  try:
   sections,_=parse_generated_sections(r['raw_output'],HEADINGS);accepted=True
   clinical=normalize(' '.join(s['text'] for s in sections if s['heading'] in ['Findings','Impression']));pp=profile(sections)
  except (ValueError,KeyError,TypeError):sections=[];accepted=False;clinical=None;pp=None
  records.append({**r,'condition_id':i+1,'accepted':accepted,'sections':sections,'clinical_text':clinical,'assertion_profile':pp,'reference_proxy_against_original':score(r['raw_output'],pair[r['original_source']])})
 # Only visual input content changes in original/CAM contrasts; the prompt remains identical.
 for mode in ['base','adapter']:
  for factor in ['original_source','cam_source']:
   grouped={}
   for r in records:
    if r['mode']!=mode:continue
    key=tuple(r[f] for f in FACTORS if f!=factor);grouped.setdefault(key,set()).add(r['input_audit']['prompt_sha256'])
   assert all(len(values)==1 for values in grouped.values())
 analysis=sensitivity(records)
 unique={mode:{'unique_clinical_texts':len({r['clinical_text'] for r in records if r['mode']==mode and r['accepted']}),'unique_assertion_profiles':len({json.dumps(r['assertion_profile'],sort_keys=True) for r in records if r['mode']==mode and r['accepted']}),'accepted_reports':sum(r['accepted'] for r in records if r['mode']==mode)} for mode in ['base','adapter']}
 score_distance={name:abs(pair['normal']['cnn_scores'][name]-pair['abnormal']['cnn_scores'][name]) for name in pair['normal']['cnn_scores']}
 result={'scorer_version':VERSION,'configuration':data['configuration'],'sensitivity':analysis,'output_diversity':unique,'actual_inputs_verified':True,'reference_answers_not_supplied':True,'actual_score_differences':score_distance,'clinical_use':False,'model_selection_use':False,'limitations':['One pair of validation studies, with deliberately conflicting diagnostic inputs; not a clinical benchmark.','Both studies have high cardiomegaly scores, so the score contrast does not test a large positive-versus-negative cardiac-score range.','Greedy decoding is used, but no repeated identical-input control was run; numerical nondeterminism is not independently estimated.','Changed wording or extracted assertions does not establish correct grounding; unchanged covered assertions do not prove the model never uses pixels.','The lexical assertion profile has incomplete anatomy and uncertainty coverage.'],'conditions':records}
 (out/'analysis.json').write_text(json.dumps(result,indent=2));esc=lambda x:html.escape(str(x))
 def table(headers,rows):return '<table><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
 body='<h1>Controlled report input diagnostic</h1><p>Sixteen real generations: two models × eight combinations of originals, contact sheets and CNN scores. Every call retained two originals plus one contact sheet and all 14 scores. Clinical metadata is identical; reference answers are absent.</p>'
 body+=table(['Model','Changed factor','Valid paired comparisons','Changed clinical text','Changed covered assertion profile'],[(mode,factor,v['valid_pairs'],v['clinical_text_changes'],v['assertion_profile_changes']) for mode,ff in analysis.items() for factor,v in ff.items()])
 body+='<h2>Output diversity</h2>'+table(['Model','Accepted reports','Unique clinical texts','Unique covered assertion profiles'],[(m,v['accepted_reports'],v['unique_clinical_texts'],v['unique_assertion_profiles']) for m,v in unique.items()])
 body+='<h2>Limits</h2><ul>'+''.join('<li>'+esc(s)+'</li>' for s in result['limitations'])+'</ul>'
 for mode in ['base','adapter']:
  body+='<section><h2>'+mode+'</h2>'+table(['Condition','Originals','Contact sheet','Scores','Findings','Impression'],[(r['condition_id'],r['original_source'],r['cam_source'],r['score_source'],next((s['text'] for s in r['sections'] if s['heading']=='Findings'),r['raw_output']),next((s['text'] for s in r['sections'] if s['heading']=='Impression'),'INVALID')) for r in records if r['mode']==mode])+'</section>'
 doc='<html><head><meta charset="utf-8"><style>@page{size:A4 landscape;margin:12mm}body{font:10px Arial}table{width:100%;table-layout:fixed;border-collapse:collapse;margin:10px 0}td,th{border:1px solid #abc;padding:6px;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}th{background:#edf4f6}tr{break-inside:avoid}section{break-before:page}</style></head><body>'+body+'</body></html>'
 (out/'comparison.html').write_text(doc);HTML(string=doc).write_pdf(out/'comparison.pdf')
 print(json.dumps({'sensitivity':analysis,'output_diversity':unique},indent=2))
if __name__=='__main__':main()
