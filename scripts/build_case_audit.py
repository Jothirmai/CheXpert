"""Reproduce a separate twelve-case audit; never overwrite experiment scoring."""
import hashlib,html,json,re,shutil,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from chest_xpert.training_data import parse_clinical_xml,structured_target
from chest_xpert.study_images import resolve_study_images,index_study_images
from chest_xpert.report_quality import parse_generated_sections
from scripts.assess_full_study_adapter import states
from scripts.train_report_adapter import build_messages
from weasyprint import HTML
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'checkpoints/report_lora_full_study';OUT=ROOT/'reports/case_audit'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=True);images=OUT/'images';images.mkdir(exist_ok=True)
 protected=[EXP/name for name in ['evaluation.json','assessment.json','protocol.json','input_evidence.json','validation.jsonl','enriched_validation.jsonl','train.jsonl']]+[ROOT/'checkpoints/active_models.json',ROOT/'scripts/assess_full_study_adapter.py',ROOT/'chest_xpert/disease_targets.py']
 before={str(p.relative_to(ROOT)):digest(p) for p in protected}
 data=json.loads((EXP/'evaluation.json').read_text());old=json.loads((EXP/'assessment.json').read_text());annotations=json.loads((OUT/'case_annotations.json').read_text());notes={r['study_id']:r for r in annotations['cases']}
 rows={r['study_id']:r for r in map(json.loads,(EXP/'enriched_validation.jsonl').read_text().splitlines())};evidence={r['study_id']:r for r in json.loads((EXP/'input_evidence.json').read_text())['rows'] if r['split']=='validation'}
 index=index_study_images(ROOT/'data/iu_xray/images');cases=[];impressions=Counter();clear=0
 for c in data['comparisons']:
  study=c['study_id'];row=rows[study];xml=ROOT/f'data/iu_xray/reports/ecgen-radiology/{study.removeprefix("CXR")}.xml'
  record=parse_clinical_xml(xml);target=structured_target(record['sections']);assert target==c['reference'],study+' reference/source mismatch'
  originals=resolve_study_images(xml,ROOT/'data/iu_xray/images',index)
  assert [str(p) for p in originals]==row['image_paths']
  ev=evidence[study];assert ev['cnn_scores']==row['cnn_scores'] and len(ev['paths'])==len(originals)+1
  assert ev['paths']==row['image_paths']+[row['contact_sheet_path']]
  for path,expected in zip(ev['paths'],ev['sha256']):assert digest(Path(path))==expected,(study,path)
  messages,opened=build_messages(row);prompt=messages[1]['content'][-1]['text'];assert sum(v.get('type')=='image' for v in messages[1]['content'])==len(originals)+1
  # Full findings/impression fields are training targets, never generation answers.
  ref_sections={s['heading']:s['text'] for s in target['sections']}
  leaked=[heading for heading in ('Findings','Impression') if len(ref_sections[heading])>20 and ref_sections[heading] in prompt]
  assert not leaked,(study,leaked)
  assert all(str(v) in prompt for v in row['cnn_scores'].values())
  parsed={mode:parse_generated_sections(c[mode+'_output'],[s['heading'] for s in target['sections']])[0] for mode in ('base','adapter')}
  imp=next(s['text'].strip().casefold() for s in parsed['adapter'] if s['heading']=='Impression');impressions[imp]+=1
  findings=next(s['text'] for s in parsed['adapter'] if s['heading']=='Findings');clear+=bool(re.search(r'\blungs are clear\b',findings,re.I))
  copied=[]
  for path in originals:
   with Image.open(path) as im:im.load()
   dst=images/path.name;shutil.copyfile(path,dst);assert digest(dst)==digest(path);copied.append(str(dst.relative_to(OUT)))
  for im in opened:im.close()
  note=notes[study]
  profile='technical limitation only' if study=='CXR2726' else 'normal / upper-limit normal' if study in ('CXR626','CXR3223','CXR2796') else 'positive or qualified imaging findings in source'
  cases.append({**note,'reference_profile':profile,'originals':copied,'original_paths':[str(p) for p in originals],'source_xml':str(xml),'reference':c['reference'],'base_sections':parsed['base'],'adapter_sections':parsed['adapter'],'raw_base_output':c['base_output'],'raw_adapter_output':c['adapter_output'],'historical_score':next(d for d in old['details'] if d['study_id']==study),'cnn_scores':row['cnn_scores'],'integrity':{'all_xml_views_present':True,'hashes_match_saved_training_evidence':True,'scores_match_saved_evidence':True,'report_target_matches_source_xml':True,'message_image_count':len(originals)+1,'full_reference_answer_absent_from_prompt':True}})
 probes=[]
 definitions=[
 ('The heart is normal in size.','Cardiomegaly','negative','normal-size synonym not represented'),
 ('Heart size remains slightly large.','Cardiomegaly','positive, mildly enlarged','enlargement synonym not represented'),
 ('Lung volumes are low.','Low lung volumes','positive','word-order synonym missed'),
 ('Decreased lung volumes with bronchopulmonary crowding.','Low lung volumes','positive','decreased-volume synonym missed'),
 ('Decreased lung volumes with bronchopulmonary crowding.','Bronchovascular crowding','positive crowding description','bronchopulmonary synonym missed'),
 ('Lungs are overall hyperexpanded.','Hyperinflation','positive hyperexpansion description','hyperexpanded synonym missed'),
 ('Probable posterior recess effusions.','Effusion','uncertain / probable','probable certainty is lost'),
 ('Unremarkable for cardiomegaly.','Cardiomegaly','negative','unremarkable-for negation reversed'),
 ('Kyphotic degenerated osteopenic thoracic spine.','Kyphotic positioning','no positioning assertion; spinal kyphosis is present','anatomy confused with acquisition quality'),
 ('Technically limited study.','Limited inspiration','technical limitation present; inspiration cause unspecified','general technical limitation has no suitable fact type')]
 for text,key,expected,issue in definitions:probes.append({'text':text,'existing_fact_name':key,'existing_state':states([{'heading':'Findings','text':text}]).get(key),'reviewed_interpretation':expected,'issue':issue})
 counts=Counter();training=list(map(json.loads,(EXP/'train.jsonl').read_text().splitlines()))
 for row in training:
  ss=row['target']['sections'];f=' '.join(s['text'] for s in ss if s['heading']=='Findings');imp=' '.join(s['text'] for s in ss if s['heading']=='Impression');positive=any(v==1 for v in states(ss).values());noacute=bool(re.search(r'no (?:definite )?acute',imp,re.I))
  counts['studies']+=1;counts['normal_mesh_tag']+=any(t.casefold()=='normal' for t in row.get('mesh_terms',[]));counts['positive_existing_text_proxy']+=positive;counts['no_acute_impression']+=noacute;counts['positive_existing_text_proxy_and_no_acute_impression']+=positive and noacute
  counts['low_volume_text_pattern']+=bool(re.search(r'low lung|lung volumes? (?:are |is )?low|decreased lung volumes?|hypoinflat',f+' '+imp,re.I));counts['limited_quality_text_pattern']+=bool(re.search(r'technic\w* limit|study.{0,25}limit|rotated|poor inspir',f+' '+imp,re.I))
 after={str(p.relative_to(ROOT)):digest(p) for p in protected};assert before==after
 summary={'audited_cases':len(cases),'original_images_inspected':sum(len(c['originals']) for c in cases),'same_adapter_impression_count':impressions['no acute cardiopulmonary abnormality.'],'adapter_clear_lungs_count':clear,'source_positive_or_qualified_imaging_cases':sum(c['reference_profile']=='positive or qualified imaging findings in source' for c in cases),'reviewed_reference_details':sum(len(c['details']) for c in cases),'historical_metrics_unchanged':True,'model_selection_unchanged':True,'training_distribution_text_audit':dict(counts),'distribution_limitations':'Counts use text patterns and the existing weak fact parser, not adjudicated disease labels. Non-acute does not mean no chronic findings.','conclusion':'No missed equivalent positive wording was found that reverses adapter rejection. Real normal-style output collapse coexists with substantial scorer undercoverage and semantic errors.','causal_limitations':'Identical impressions alone do not prove image inputs were ignored, overfitting, or that CNN scores caused the failure. No image/score perturbation experiment or independent radiologist adjudication was performed.','input_limitations':'CXR3786 has only one lateral image in XML although caption-derived technique says two-view. All XML views are present; view completeness does not establish adequate clinical projections.','audit_scope':'AI-assisted source-to-report/software review, with visual inspection of all originals for coverage and integrity. Reference findings are source assertions, not independently verified diagnoses. The 41 detail rows are an illustrative reviewed checklist including qualifiers and uncertain findings, not a new accuracy denominator.'}
 result={'summary':summary,'scorer_probes':probes,'cases':cases,'source_artifact_sha256':before,'historical_metrics':old['metrics']}
 (OUT/'audit.json').write_text(json.dumps(result,indent=2))
 esc=lambda v:html.escape(str(v))
 def table(headers,rows):return '<table><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
 lines=['# Twelve-case audit','','This is an AI-assisted source-to-report and software audit, not an independent radiologist adjudication.','','## Findings','',summary['conclusion'],'',f"All {summary['audited_cases']} cases and {summary['original_images_inspected']} original images were reviewed. The adapter uses the same no-acute impression in all 12 reports and clear-lung wording in all 12. Eight source reports contain positive or qualified imaging findings. The original metric files and model selection remain unchanged.",'',summary['input_limitations'],'','## Training distribution','',json.dumps(dict(counts),indent=2),'','Text-pattern counts have incomplete coverage; no-acute impressions do not establish normal lungs.','']
 overview='<section class="page"><h1>Twelve-case report audit</h1><p class="note">'+esc(summary['audit_scope'])+'</p><h2>Conclusion</h2><p>'+esc(summary['conclusion'])+'</p>'
 overview+=table(['Check','Result'],[['Cases / original images',f"12 / {summary['original_images_inspected']}"],['Repeated adapter impression','12/12: No acute cardiopulmonary abnormality.'],['Adapter clear-lung wording','12/12'],['Source cases with positive or qualified imaging findings','8/12'],['Input/reference integrity','All XML views and saved hashes match; all 12 targets match source XML.'],['Input limitation',summary['input_limitations']],['Historical results and model selection','Preserved; rejected adapter remains inactive.']])
 overview+='<h2>Case index</h2>'+table(['Study','Audit finding'],[(c['study_id'],c['verdict']) for c in cases]).replace('<table>','<table class="index">',1)+'</section>'
 overview+='<section class="page"><h1>Scorer and training-data audit</h1><p>These probes reproduce scorer limitations without modifying historical results.</p>'+table(['Text probe','Existing fact / result','Reviewed interpretation'],[(p['text'],p['existing_fact_name']+' = '+str(p['existing_state']),p['reviewed_interpretation']) for p in probes])+'<h2>Training-target distribution</h2>'+table(['Text audit','Count'],counts.items())+'<p>'+esc(summary['distribution_limitations'])+'</p><p>'+esc(summary['causal_limitations'])+'</p><h2>Next actions</h2><ol><li>Version a corrected scorer: normal-heart language, volume synonyms, uncertainty, location, hardware and technical limitations; rescore both existing outputs under the same revised rules.</li><li>Rebuild the technical validation category with actual acquisition limitations; spinal kyphosis belongs to anatomy.</li><li>Run controlled input perturbations to distinguish visual-grounding failure from score anchoring. Keep the required three-input application unchanged during this diagnostic experiment.</li><li>Then change supervision around positive findings and chronic-versus-acute language; do not launch another blind long run.</li></ol></section>'
 body=overview
 for c in cases:
  source={s['heading']:s['text'] for s in c['reference']['sections']};base={s['heading']:s['text'] for s in c['base_sections']};adapter={s['heading']:s['text'] for s in c['adapter_sections']}
  body+='<section class="page"><h1>'+esc(c['study_id'])+'</h1><p class="verdict">'+esc(c['verdict'])+'</p><p>'+esc(c['visual_input_note'])+'</p><div class="images">'+''.join('<figure><img src="'+esc(src)+'"><figcaption>'+esc(Path(src).name)+'</figcaption></figure>' for src in c['originals'])+'</div>'
  body+=table(['Section','Dataset reference','Base output','Adapter output'],[(h,source[h],base[h],adapter[h]) for h in ('Findings','Impression')]).replace('<table>','<table class="clinical">',1)
  body+='<h2>Reference-detail review</h2>'+table(['Source detail / certainty','Base','Adapter'],[(d['reference_detail']+' ('+d['certainty']+')',d['base'],d['adapter']) for d in c['details']]) if c['details'] else '<h2>Reference-detail review</h2><p>Source report describes normal cardiopulmonary findings; no abnormal-positive checklist is assigned.</p>'
  body+='<h2>Additional base claims / qualification issues</h2><ul>'+''.join('<li>'+esc(v)+'</li>' for v in c['base_extra_claims'])+'</ul><h2>Scorer limitations</h2><ul>'+''.join('<li>'+esc(v)+'</li>' for v in c['scorer_issues'])+'</ul><p><b>Priority:</b> '+esc(c['priority'])+'</p><p class="small">Source XML: '+esc(c['source_xml'])+'<br>Scores and input integrity: audit.json. Full output and metadata are retained there.</p></section>'
  lines+=['## '+c['study_id'],'',c['verdict'],'',c['visual_input_note'],'','### Source reference',source['Findings'],source['Impression'],'','### Base',base['Findings'],base['Impression'],'','### Adapter',adapter['Findings'],adapter['Impression'],'','### Reviewed reference details']
  lines += [f"- {d['reference_detail']} [{d['certainty']}]: base={d['base']}; adapter={d['adapter']}" for d in c['details']]
  lines += ['','### Scorer limitations',*['- '+v for v in c['scorer_issues']],'','### Additional base claims / qualification issues',*['- '+v for v in c['base_extra_claims']],'','Priority: '+c['priority'],'']
 document='<!doctype html><html><head><meta charset="utf-8"><style>@page{size:A4 landscape;margin:11mm}body{font:9px Arial;color:#17324a}h1{font-size:18px;margin:0 0 6px}h2{font-size:11px;margin:6px 0 4px}p{line-height:1.2}table{width:100%;border-collapse:collapse;table-layout:fixed;margin:8px 0}th,td{border:1px solid #bccbd5;padding:4px;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}th{background:#eaf1f5}.page{break-before:page}.page:first-child{break-before:auto}.images{display:flex;justify-content:center;gap:10px}figure{margin:3px;text-align:center;max-width:31%}img{height:120px;max-width:100%;object-fit:contain}figcaption,.small{font-size:8px}.note,.verdict{padding:8px;background:#fff4d9}li{margin:3px 0}p.small{font-size:7px;margin:2px 0;line-height:1.1}tr{break-inside:avoid}.clinical th:first-child,.clinical td:first-child{width:8%}.index th:first-child,.index td:first-child{width:10%}</style></head><body>'+body+'</body></html>'
 (OUT/'audit.md').write_text('\n'.join(lines));(OUT/'audit.html').write_text(document);HTML(string=document,base_url=str(OUT)).write_pdf(OUT/'audit.pdf')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
