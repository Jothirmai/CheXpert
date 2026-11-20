"""Generate and audit an illustrative report using all three mandatory inputs."""
import hashlib,html,json,shutil,sys,time
from pathlib import Path
from uuid import uuid4
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.services.inference_service import inference_service
from backend.services import review_store
from backend.config import STORAGE_CONFIG
from chest_xpert.checkpoint_engine import CheckpointEngine
from cliq_rrg.eval import bleu,rouge_l
from chest_xpert.study_images import resolve_study_images

out=Path('reports/full_study_evidence');out.mkdir(parents=True,exist_ok=True)
rows=[json.loads(line) for line in Path('data/report_training_v3/test.jsonl').read_text().splitlines() if line.strip()]
row=rows[0]  # Same previously inspected demo; not a new independent test metric.
xml=Path('data/iu_xray/reports/ecgen-radiology')/f"{row['study_id'].removeprefix('CXR')}.xml"
views=resolve_study_images(xml,'data/iu_xray/images')
identifier=str(uuid4());start=time.perf_counter()
service=inference_service;service._engine=CheckpointEngine(device='cuda:0');service._engine_device='cuda:0'
request={'additional_image_paths':[str(p) for p in views[1:]],'patient_id':row['study_id'],'clinical_indication':row.get('clinical_indication'),'comparison_study':row.get('comparison_study'),'template_id':'standard'}
result=service._run_pipeline_sync(identifier,str(views[0]),request,False)  # Legacy disable flag cannot remove mandatory evidence.
(out/'pipeline_result.json').write_text(json.dumps(result,indent=2,default=str))
assert result['status']=='completed',result
review=review_store.read(identifier);evidence=review['model_provenance']['vision_inputs']
assert evidence['image_count']==len(views)+1
assert evidence['original_image_count']==len(views)
assert len(evidence['image_grid_thw'])==len(views)+1
assert len(review['model_provenance']['cnn_scores'])==14
assert review['model_provenance']['evidence_mode']=='cnn_evidence'
for name,source,digest in zip([f'original_{i+1}.png' for i in range(len(views))]+['gradcam_contact_sheet.png'],evidence['image_paths'],evidence['image_sha256']):
 assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest
 shutil.copyfile(source,out/name)
client=service._verification_engine.client
(out/'input_prompt.txt').write_text(client._last_report_prompt)
(out/'raw_model_output.txt').write_text(result['v7_report'])
(out/'generated_report.json').write_text(json.dumps(review,indent=2))
(out/'dataset_reference.json').write_text(json.dumps(row,indent=2))
(out/'cnn_scores.json').write_text(json.dumps(review['model_provenance']['cnn_scores'],indent=2))
(out/'input_audit.json').write_text(json.dumps(evidence,indent=2))
xml=Path('data/iu_xray/reports/ecgen-radiology')/f"{row['study_id'].removeprefix('CXR')}.xml"
shutil.copyfile(xml,out/'original_dataset_report.xml')
original={n.get('Label'): ' '.join(n.itertext()).strip() for n in ET.parse(xml).findall('.//AbstractText')}
generated={s['heading']:s['text'] for s in review['sections']}
ref=' '.join(s['text'] for s in row['target']['sections'] if s['heading'] in ['Findings','Impression'])
pred=' '.join(s['text'] for s in review['sections'] if s['heading'] in ['Findings','Impression'])
metrics={'study_id':row['study_id'],'illustrative_previously_inspected_case':True,'input_contract_passed':True,'cnn_scores':14,'vision_images':len(views)+1,'original_image_count':len(views),'elapsed_seconds':time.perf_counter()-start,'bleu1':bleu(ref,pred,1),'bleu4':bleu(ref,pred,4),'rouge_l':rouge_l(ref,pred),'reference_findings_or_impression_supplied':False,'limitations':'Single previously inspected example, not an independent validation metric or clinical accuracy assessment.'}
(out/'verification.json').write_text(json.dumps(metrics,indent=2))
esc=lambda s:html.escape(str(s))
def table(headers,rows):return '<table><tr>'+''.join('<th>'+esc(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</table>'
body='<h1>Verified three-input VLM report — '+esc(row['study_id'])+'</h1><p>The model received 14 actual CNN scores, the original X-ray and the labeled Grad-CAM contact sheet. Every original study view was resolved from the dataset XML and retained by the image processor. File hashes and exact prompt are saved alongside this document. No reference findings or impression were supplied to the model.</p>'
body+='<h2>Dataset reference and newly generated report</h2>'+table(['Section','Original dataset report','Generated report'],[[heading,original.get(label,'Not supplied'),generated.get(heading,'Not supplied')] for label,heading in [('INDICATION','Clinical indication'),('COMPARISON','Comparison'),('FINDINGS','Findings'),('IMPRESSION','Impression')]])
body+='<h2>Original study images</h2>'+''.join(f'<img src="original_{i+1}.png" class="original">' for i in range(len(views)))+'<h2>Grad-CAM contact sheet input</h2><img src="gradcam_contact_sheet.png" class="sheet">'
body+='<h2>Exact CNN score input</h2>'+table(['Finding','Uncalibrated CNN score'],[[name,f'{value:.6f}'] for name,value in review['model_provenance']['cnn_scores'].items()])
body+='<h2>Run verification</h2><pre>'+esc(json.dumps(metrics,indent=2))+'</pre><p>Scores and attention do not establish a diagnosis. This is a reviewable draft; report-to-image correctness still requires assessment.</p>'
style='body{font:13px Arial;color:#17324a;margin:25px}table{border-collapse:collapse;width:100%;margin:15px 0}td,th{border:1px solid #abc;padding:10px;vertical-align:top}th{background:#edf4f6}h2{break-after:avoid}.original{max-height:400px;max-width:90%;display:block;margin:auto}.sheet{width:90%}pre{white-space:pre-wrap}@page{size:A4 landscape;margin:12mm}'
report='<!doctype html><html><head><meta charset="utf-8"><style>'+style+'</style></head><body>'+body+'</body></html>'
(out/'comparison.html').write_text(report)
from weasyprint import HTML
HTML(string=report,base_url=str(out.resolve())).write_pdf(out/'comparison.pdf')
HTML(string=review_store.render(review)).write_pdf(out/'generated_report.pdf')
print(json.dumps(metrics,indent=2),flush=True)
print(json.dumps(generated,indent=2),flush=True)
