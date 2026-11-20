"""Real CUDA API smoke check after model training; temporary patient storage."""
import argparse,json,tempfile,time
from pathlib import Path
from fastapi.testclient import TestClient
from backend.config import STORAGE_CONFIG
from backend.main import app,inference_service

p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
result={'passed':False,'clinical_validation':False}
try:
    with tempfile.TemporaryDirectory(prefix='chest-final-verification-') as root:
        STORAGE_CONFIG.temp_dir=Path(root)/'temp';STORAGE_CONFIG.reports_dir=Path(root)/'reports'
        with TestClient(app) as client:
            readiness=client.get('/api/v1/ready').json();assert readiness['ready'],readiness
            result['readiness']=readiness
            for filename in ['fixture_color_photo.png','sample_frontal.png']:
                started=time.perf_counter()
                response=client.post('/api/v1/analyses',files={'file':(filename,Path('samples',filename).read_bytes(),'image/png')},data={'patient_id':'VERIFICATION-ONLY','template_id':'standard'})
                assert response.status_code==202,response.text
                identifier=response.json()['analysis_id']
                for _ in range(1500):
                    analysis=client.get(f'/api/v1/analyses/{identifier}').json()
                    if analysis['status'] in ['failed','completed']:break
                    time.sleep(.2)
                if filename.startswith('fixture'):
                    assert analysis['status']=='failed',analysis
                    assert analysis['modality']['reason']=='The uploaded image is not a chest X-ray. Please upload a valid chest X-ray.'
                    result['non_xray_rejected']=True
                    continue
                assert analysis['status']=='completed',analysis
                result['report_latency_s']=time.perf_counter()-started
                result['within_two_minute_target']=result['report_latency_s']<=120
                review=client.get(f'/api/v1/reviews/{identifier}').json()
                assert len(review['labels'])==14
                assert review['sections'][0]['text']=='NOT PROVIDED'
                assert review['sections'][1]['text']=='NOT PROVIDED'
                assert review['sections'][2]['text']=='NOT PROVIDED'
                for kind in ['original','gradcam']:
                    image=client.get(f'/api/v1/reviews/{identifier}/image/{kind}')
                    assert image.status_code==200
                    (output/f'verification_{kind}.png').write_bytes(image.content)
                response=client.get(f'/api/v1/report/{identifier}/download?format=pdf')
                assert response.status_code==200 and response.content.startswith(b'%PDF')
                (output/'verification_report.pdf').write_bytes(response.content)
                (output/'verification_report.json').write_text(json.dumps(review,indent=2))
                result['model_provenance']=review.get('model_provenance')
                result['adapter_loaded']=inference_service._verification_engine.client._lora_loaded
                result['passed']=True
except Exception as exc:
    result['error']=str(exc)
finally:
    (output/'application_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
if not result['passed']:raise SystemExit(1)
