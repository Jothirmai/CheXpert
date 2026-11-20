"""Export actual generated/reference reports and clearly scoped CNN/VLM metrics."""
import hashlib,html,json,re,shutil,sys,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.services import review_store
from chest_xpert.report_quality import TECHNICAL_METADATA
from scripts.assess_report_adapter import analyze,fact_states
from cliq_rrg.eval import bleu,rouge_l
from weasyprint import HTML
from uuid import uuid4
root=Path(__file__).resolve().parents[1];out=root/'reports/clinical_report';out.mkdir(exist_ok=True)
bench=root/'checkpoints/report_score_correction_multiview'
review=json.loads((root/'reports/multiview_correction/generated_report.json').read_text())  # Active parent CNN, real audited API run.
reference=json.loads((root/'reports/required_vlm_evidence/dataset_reference.json').read_text())
xml=root/'data/iu_xray/reports/ecgen-radiology/3076.xml'
original={n.get('Label'):' '.join(n.itertext()).strip() for n in ET.parse(xml).findall('.//AbstractText')}
raw_reference={'sections':[{'heading':heading,'text':original.get(label,'NOT PROVIDED')} for label,heading in [('INDICATION','Clinical indication'),('COMPARISON','Comparison'),('FINDINGS','Findings'),('IMPRESSION','Impression')]]}
generated={s['heading']:s['text'] for s in review['sections']}
assert all(not TECHNICAL_METADATA.search(s['text']) for s in review['sections'])
audit=review['model_provenance']['vision_inputs'];assert audit['original_image_count']==3 and audit['image_count']==4
for path,digest in zip(audit['image_paths'],audit['image_sha256']):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
for name in ('report.html','report.pdf','comparison.html','comparison.pdf'):
 path=out/name
 if path.exists():
  history=out/'before_full_study_fix';history.mkdir(exist_ok=True)
  if not (history/name).exists():shutil.copyfile(path,history/name)
clinical=review_store.render(review);(out/'report.html').write_text(clinical);HTML(string=clinical).write_pdf(out/'report.pdf')
(out/'generated_report.json').write_text(json.dumps(review,indent=2));shutil.copyfile(xml,out/'original_dataset_report.xml')
(out/'reference_report.json').write_text(json.dumps(raw_reference,indent=2))
# Explicit editorial revision from the source report, never scored as inference.
corrected_sections=[{'heading':s['heading'],'text':s['text']} for s in review['sections']]
for section in corrected_sections:
    if section['heading']=='Findings':
        section['text']='The frontal radiograph is kyphotic and rotated. Low lung volumes with bronchovascular crowding. No definite airspace consolidation or pleural effusion. Accounting for technical factors, the cardiac silhouette is borderline enlarged.'
    elif section['heading']=='Impression':
        section['text']='Low lung volumes. No definite acute findings.'
    elif section['heading']=='Technique':
        section['text']='PA and lateral chest radiographs.'
corrected={**review,'analysis_id':str(uuid4()),'source_analysis_id':review['analysis_id'],'sections':corrected_sections,'signature':None,'revision_source':'Reference-guided editorial revision of CXR3076; not a new model prediction','reference_source':str(xml),'used_for_metrics':False}
corrected['original_generation_provenance']=corrected.pop('model_provenance')
corrected['original_cnn_labels']=corrected.pop('labels')
corrected_html=review_store.render(corrected).replace('AI-generated draft — pending radiologist review','Reference-guided revised draft — pending radiologist review')
(out/'corrected_report.html').write_text(corrected_html);HTML(string=corrected_html).write_pdf(out/'corrected_report.pdf')
(out/'corrected_report.json').write_text(json.dumps(corrected,indent=2))
corrected_text={s['heading']:s['text'] for s in corrected_sections}

esc=lambda v:html.escape(str(v))
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
style='@page{size:A4 landscape;margin:14mm}body{font:12px Arial;color:#17324a;margin:20px}h1{font-size:23px}h2{font-size:17px}table{border-collapse:collapse;width:100%;margin:15px 0}td,th{border:1px solid #abc;padding:10px;vertical-align:top;white-space:pre-wrap}th{background:#edf4f6}section{break-inside:avoid}p{line-height:1.5}.warning{padding:12px;background:#fff5dd}'
def export(name,body):
 document='<!doctype html><html><head><meta charset="utf-8"><style>'+style+'</style></head><body>'+body+'</body></html>'
 (out/f'{name}.html').write_text(document);HTML(string=document).write_pdf(out/f'{name}.pdf')
body='<h1>Original and generated report — CXR3076</h1><p class="warning">Generated draft — pending radiologist review. Differences remain visible for review; the reference was not supplied as an answer to the model. The revised draft is an editorial correction using the reference, not a new prediction; it is excluded from all metrics.</p>'
body+=table(['Section','Original dataset report','Raw generated report','Reference-guided revised draft'],[(heading,original.get(label,'NOT PROVIDED'),generated.get(heading,'NOT PROVIDED'),corrected_text.get(heading,'NOT PROVIDED')) for label,heading in [('INDICATION','Clinical indication'),('COMPARISON','Comparison'),('FINDINGS','Findings'),('IMPRESSION','Impression')]])
reference_text=' '.join(s['text'] for s in raw_reference['sections'] if s['heading'] in ('Findings','Impression'))
prediction_text=' '.join(s['text'] for s in review['sections'] if s['heading'] in ('Findings','Impression'))
cues={'Low lung volumes':r'low lung volumes?|hypoinflat\w*','Bronchovascular crowding':r'bronchovascular crowding','Rotation / kyphotic positioning':r'rotat\w*|kyphotic','Borderline cardiac enlargement qualifier':r'borderline'}
body+='<h2>Wording to reconcile</h2><p>This table checks whether source wording is stated. Equivalent wording may be missed; it does not establish image-based correctness.</p>'
body+=table(['Source detail','Original wording','Generated wording'],[(name,'Present' if re.search(pattern,reference_text,re.I) else 'Not stated','Present' if re.search(pattern,prediction_text,re.I) else 'Not stated') for name,pattern in cues.items()])
truth=fact_states(raw_reference['sections']);prediction=fact_states(review['sections']);state=lambda v:'Positive assertion' if v==1 else 'Negative assertion' if v==0 else 'Uncertain / not stated'
body+=table(['Assertion check','Original report','Generated report'],[(name,state(truth.get(name)),state(prediction.get(name))) for name in ('Cardiomegaly','Effusion','Edema','Consolidation')])
export('comparison',body)
cnn=json.loads((root/'checkpoints/full_study_cnn_validation/results.json').read_text())
vlm=json.loads((bench/'summary.json').read_text());data=json.loads((root/'data/iu_xray/image_completeness_audit.json').read_text())
ref=' '.join(s['text'] for s in raw_reference['sections'] if s['heading'] in ('Findings','Impression'));pred=' '.join(s['text'] for s in review['sections'] if s['heading'] in ('Findings','Impression'))
example={'study_id':'CXR3076','illustrative_previously_inspected_case':True,'bleu1':bleu(ref,pred,1),'bleu4':bleu(ref,pred,4),'rouge_l':rouge_l(ref,pred),'proxy':analyze(json.dumps({'sections':review['sections']}),reference['target'],reference['mesh_terms']),'reference_findings_or_impression_supplied':False,'original_image_count':3,'actual_processor_image_grids':4}
metrics={'cnn':cnn,'vlm':vlm,'illustrative_example':example,'dataset_restoration':{k:data[k] for k in ['restored_images','valid_existing_kept','complete_studies','incomplete_studies','existing_split_manifests_modified']},'clinical_accuracy_established':False}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
body='<h1>CNN and report-model evaluation</h1><p>All comparisons below use the actual saved outputs. CNN and report metrics measure different tasks and are reported separately.</p><h2>CNN: complete-study validation</h2>'
body+='<p>Same 563 validation studies and corrected weak labels for both checkpoints; all '+str(cnn['original_views'])+' original XML views included. The test split was not used.</p>'
body+=table(['Checkpoint','Macro AUROC','Average precision','Known-label log loss','Brier score','Evaluable labels'],[(name,*[f'{v[k]:.5f}' for k in ['macro_auroc','macro_average_precision','known_label_log_loss','known_label_brier']],v['evaluable_labels']) for name,v in cnn['metrics'].items()])
body+='<h2>Report model: fixed four-study validation comparison</h2><p>Same Qwen model and prompt; parent versus weight-adjusted CNN input scores. Every source study view was supplied. These are incomplete lexical assertion proxies and text overlap, not clinical accuracy. Missing reference facts outside the parser vocabulary are not counted.</p>'
body+=table(['CNN input','Accepted reports','Matched reference positive terms','Unsupported positives against known negatives','Mean BLEU-1','Mean ROUGE-L'],[(name,f"{v['accepted_reports']}/{v['cases']}",f"{v['matched_positive_terms']}/{v['reference_positive_terms']}",v['unsupported_terms_against_known_negatives'],f"{v['mean_bleu1']:.5f}",f"{v['mean_rouge_l']:.5f}") for name,v in vlm['metrics'].items()])
body+='<h2>Illustrative report comparison</h2><p>CXR3076 was previously inspected and is excluded from model selection. It is not independent validation. Reference wording and generated text can be inspected in comparison.pdf.</p>'
body+=table(['BLEU-1','BLEU-4','ROUGE-L'],[[f"{example[k]:.5f}" for k in ('bleu1','bleu4','rouge_l')]])
body+='<h2>Dataset repair</h2>'+table(['Restored images','Complete studies','Incomplete studies','Splits changed'],[[data['restored_images'],data['complete_studies'],data['incomplete_studies'],data['existing_split_manifests_modified']]])
body+='<p class="warning">Remaining errors require image-based radiologist review. Neither text overlap nor weak-label CNN scores establishes diagnostic accuracy. Historical results used incomplete views and an earlier label parser and must not be compared as if the protocol were unchanged.</p>'
export('metrics',body)
print(json.dumps({'out':str(out),'illustrative_example':example,'clinical_sections_clean':True},indent=2))
