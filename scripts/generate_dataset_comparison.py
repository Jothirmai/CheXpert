"""Generate one new held-out report without supplying reference findings."""
import json,sys,time,shutil
from pathlib import Path
from uuid import uuid4
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.services.inference_service import inference_service
from backend.services import review_store
from backend.config import STORAGE_CONFIG
from chest_xpert.checkpoint_engine import CheckpointEngine

out=Path('reports/model_comparison');out.mkdir(parents=True,exist_ok=True)
rows=[json.loads(line) for line in Path('data/report_training_v3/test.jsonl').read_text().splitlines() if line.strip()]
# Fixed, first held-out study; not selected for a flattering model output.
row=rows[0]
(out/'dataset_reference.json').write_text(json.dumps(row,indent=2))
shutil.copyfile(row['image_path'],out/'original.png')
service=inference_service
service._engine=CheckpointEngine(device='cuda:0');service._engine_device='cuda:0'
identifier=str(uuid4());start=time.perf_counter()
request={'patient_id':row['study_id'],'clinical_indication':row.get('clinical_indication'),'comparison_study':row.get('comparison_study'),'template_id':'standard'}
# Do not infer single-image acquisition details from a multi-view report.
result=service._run_pipeline_sync(identifier,row['image_path'],request,True)
(out/'pipeline_result.json').write_text(json.dumps(result,indent=2,default=str))
try:
 review=review_store.read(identifier)
 (out/'generated_report.json').write_text(json.dumps(review,indent=2))
 (out/'generated_report.html').write_text(review_store.render(review))
 from weasyprint import HTML
 HTML(string=review_store.render(review)).write_pdf(out/'generated_report.pdf')
 for suffix in ['gradcam_contact_sheet']:
  source=STORAGE_CONFIG.temp_dir/f'{identifier}_{suffix}.png'
  if source.exists():shutil.copyfile(source,out/f'{suffix}.png')
except FileNotFoundError:
 raise RuntimeError(f'No report generated: {result}')
(out/'generation_provenance.json').write_text(json.dumps({'study_id':row['study_id'],'split':'test','selection':'first row of fixed test manifest','elapsed_seconds':time.perf_counter()-start,'reference_findings_supplied_to_model':False,'reference_impression_supplied_to_model':False,'inputs':request},indent=2))
print(json.dumps({'output':str(out),'study':row['study_id'],'elapsed':time.perf_counter()-start}),flush=True)
