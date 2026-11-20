from uuid import uuid4
from types import SimpleNamespace
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.api.v1 import workflow, report
from backend.services import review_store
from backend.services.inference_service import inference_service

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(review_store, 'STORAGE_CONFIG', SimpleNamespace(reports_dir=tmp_path))
    import backend.config
    monkeypatch.setattr(backend.config.STORAGE_CONFIG, 'reports_dir', tmp_path)
    app = FastAPI()
    app.include_router(workflow.router, prefix='/api/v1')
    app.include_router(report.router, prefix='/api/v1/report')
    return TestClient(app)


def seed():
    identifier = str(uuid4())
    review_store.write(identifier, {'analysis_id':identifier, 'created_at':'2026-11-09T00:00:00Z', 'patient_id':'TEST', 'template':review_store.DEFAULT_TEMPLATE, 'sections':[{'heading':'Findings','text':'Original text'}], 'signature':None, 'revision':1, 'labels':[]})
    return identifier


def test_review_sign_edit_invalidation_and_conflict(client):
    identifier = seed()
    root = f'/api/v1/reviews/{identifier}'
    assert client.post(root+'/sign',json={'name':'Test Reviewer','license':'TEST-123','confirmed':False,'revision':1}).status_code == 400
    signed = client.post(root+'/sign',json={'name':'Test Reviewer','license':'TEST-123','confirmed':True,'revision':1})
    assert signed.status_code == 200
    assert signed.json()['signature']['report_sha256']
    changed={'sections':[{'heading':'Findings','text':'Corrected <script>alert(1)</script>'}],'template_id':'standard','revision':2}
    saved=client.put(root,json=changed)
    assert saved.status_code==200 and saved.json()['signature'] is None
    assert client.put(root,json=changed).status_code==409
    preview=client.get(root+'/preview').text
    assert '&lt;script&gt;' in preview and '<script>' not in preview
    assert 'Corrected' in client.get(f'/api/v1/report/{identifier}/download?format=html').text
    assert client.get('/api/v1/reviews').json()[0]['analysis_id']==identifier


def test_pdf_and_template_save(client):
    identifier=seed()
    response=client.get(f'/api/v1/report/{identifier}/download?format=pdf')
    assert response.status_code==200 and response.content.startswith(b'%PDF')
    assert client.get(f'/api/v1/report/{identifier}/download?format=bad').status_code==400
    saved=client.put('/api/v1/templates/standard',json={'template_name':'Hospital','sections':['Findings','Impression'],'layout':{'header':'Hospital Radiology'}})
    assert saved.status_code==200 and saved.json()['id']!='standard'
    assert len(client.get('/api/v1/templates').json())==2


def test_unknown_reports_and_invalid_ids(client):
    assert client.get('/api/v1/reviews/not-a-uuid').status_code==404
    assert client.get(f'/api/v1/reviews/{uuid4()}').status_code==404


def test_rejected_and_uncertain_images_never_reach_cnn(monkeypatch):
    import torch
    from chest_xpert import preprocessing, modality
    from backend.services.inference_service import InferenceService
    from chest_xpert.modality import ModalityResult, ModalityDecision
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(preprocessing, 'validate_image', lambda path: SimpleNamespace(ok=True))
    service = InferenceService()
    for decision in (ModalityDecision.REJECTED, ModalityDecision.UNCERTAIN, ModalityDecision.TECHNICAL_FAILURE):
        monkeypatch.setattr(modality.modality_validator, 'validate', lambda path, d=decision: ModalityResult(decision=d, reason='test'))
        result = service._run_pipeline_sync(str(uuid4()), 'dummy.png', {}, True)
        assert result['status'] == 'rejected'
        assert result['labels'] == []
        if decision == ModalityDecision.REJECTED:
            assert result['reason'] == 'The uploaded image is not a chest X-ray. Please upload a valid chest X-ray.'


def test_bad_template_definition(client):
    response=client.put('/api/v1/templates/standard',json={'template_name':'Bad','sections':['Findings','Findings'],'layout':{'accent':'javascript:alert(1)'}})
    assert response.status_code==422


def test_cleanup_preserves_saved_reviews_and_templates(client, tmp_path):
    from backend.services.cleanup_service import CleanupService
    identifier=seed()
    service=CleanupService()
    service.temp_dir=tmp_path/'temp'
    service.temp_dir.mkdir()
    service.reports_dir=tmp_path
    uploaded=service.temp_dir/'expired-upload.png'
    uploaded.write_bytes(b'test')
    service.run(ttl_s=0)
    assert not uploaded.exists()
    assert review_store.read(identifier)['analysis_id']==identifier


def test_provisioned_reviewer_cannot_impersonate(client, tmp_path, monkeypatch):
    import hashlib, json
    credentials = tmp_path / 'reviewers.json'
    credentials.write_text(json.dumps([{'id':'reviewer-1','name':'Assigned Reviewer','license':'REG-1','key_sha256':hashlib.sha256(b'personal-key').hexdigest()}]))
    monkeypatch.setenv('CHESTXPERT_REVIEWERS_FILE', str(credentials))
    identifier = seed()
    root = f'/api/v1/reviews/{identifier}'
    assert client.get(root).status_code == 401
    headers = {'X-API-Key':'personal-key'}
    assert client.get('/api/v1/identity',headers=headers).json()['reviewer']['id']=='reviewer-1'
    body={'name':'Other Reviewer','license':'REG-1','confirmed':True,'revision':1}
    assert client.post(root+'/sign',json=body,headers=headers).status_code==403
    body['name']='Assigned Reviewer'
    response=client.post(root+'/sign',json=body,headers=headers)
    assert response.status_code==200
    assert response.json()['signature']['reviewer_id']=='reviewer-1'


def test_contradiction_blocks_signing_and_edit_resolves(client):
    identifier=seed()
    data=review_store.read(identifier)
    data['sections']=[{'heading':'Findings','text':'There is pneumothorax.'},{'heading':'Impression','text':'No pneumothorax.'}]
    review_store.write(identifier,data)
    root=f'/api/v1/reviews/{identifier}'
    body={'name':'Reviewer','license':'REG','confirmed':True,'revision':1}
    assert client.post(root+'/sign',json=body).status_code==409
    data['sections'][1]['text']='Pneumothorax.'
    review_store.write(identifier,data)
    assert client.post(root+'/sign',json=body).status_code==200


def test_unsupported_interval_claim_blocks_signing_until_edited(client):
    identifier = seed()
    data = review_store.read(identifier)
    data['sections'] = [
        {'heading': 'Comparison', 'text': 'NOT PROVIDED'},
        {'heading': 'Findings', 'text': 'Interval placement of a central venous catheter.'},
        {'heading': 'Impression', 'text': 'No pneumothorax.'},
    ]
    review_store.write(identifier, data)
    root = f'/api/v1/reviews/{identifier}'
    body = {'name': 'Reviewer', 'license': 'REG', 'confirmed': True, 'revision': 1}
    assert client.post(root + '/sign', json=body).status_code == 409
    data['sections'][1]['text'] = 'A central venous catheter is present.'
    review_store.write(identifier, data)
    assert client.post(root + '/sign', json=body).status_code == 200


def test_export_keeps_reviewer_warnings_outside_clinical_document(client):
    identifier=seed()
    data=review_store.read(identifier)
    data['quality_warnings']=['Check <unsafe> wording']
    review_store.write(identifier,data)
    response=client.get(f'/api/v1/report/{identifier}/download?format=html')
    assert response.status_code==200
    assert 'Check &lt;unsafe&gt; wording' not in response.text
    assert review_store.read(identifier)['quality_warnings']==['Check <unsafe> wording']


def test_malformed_reviewer_configuration_fails_closed(client, tmp_path, monkeypatch):
    path=tmp_path/'invalid-reviewers.json'
    path.write_text('{"unexpected":true}')
    monkeypatch.setenv('CHESTXPERT_REVIEWERS_FILE',str(path))
    assert client.get('/api/v1/identity',headers={'X-API-Key':'anything'}).status_code==503


def test_clean_report_export_does_not_include_supporting_evidence(client):
    identifier=seed()
    data=review_store.read(identifier)
    data['quality_warnings']=['Grad-CAM and CNN evidence requires review.']
    data['model_provenance']={'vision_inputs':{'image_count':2},'cnn_scores':{'Effusion':.8}}
    review_store.write(identifier,data)
    html=client.get(f'/api/v1/report/{identifier}/download?format=html').text
    assert 'Original text' in html
    for word in ['Grad-CAM','CNN','heatmap','0.8','model_provenance']:
        assert word not in html
    assert review_store.read(identifier)['model_provenance']['vision_inputs']['image_count']==2


def test_clinical_export_rejects_technical_narrative_without_deleting_claims(client):
    identifier=seed();data=review_store.read(identifier)
    data['sections']=[{'heading':'Findings','text':'Grad-CAM suggests a right upper lobe nodule.'}]
    review_store.write(identifier,data)
    response=client.get(f'/api/v1/report/{identifier}/download?format=pdf')
    assert response.status_code==422
    assert review_store.read(identifier)['sections']==data['sections']


def test_multiview_pipeline_averages_scores_and_supplies_every_original(client,tmp_path,monkeypatch):
    import json,torch
    from PIL import Image
    from chest_xpert import preprocessing,modality
    from chest_xpert.modality import ModalityResult,ModalityDecision
    from chest_xpert.disease_targets import DISEASES
    from chest_xpert.verification import VerificationResult
    from backend.services.inference_service import InferenceService
    from backend.config import STORAGE_CONFIG
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(preprocessing,'validate_image',lambda path:SimpleNamespace(ok=True))
    monkeypatch.setattr(modality.modality_validator,'validate',lambda path:ModalityResult(decision=ModalityDecision.ACCEPTED,reason='test'))
    monkeypatch.setattr(STORAGE_CONFIG,'temp_dir',tmp_path/'temp');STORAGE_CONFIG.temp_dir.mkdir()
    paths=[]
    for i,value in enumerate([20,120]):
        path=tmp_path/f'view{i}.png';Image.new('L',(32,32),value).save(path);paths.append(path)
    class Head(torch.nn.Module):
        def forward(self,features):return {'disease_logits':features.mean((2,3))}
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.encoder=torch.nn.Conv2d(1,14,1);self.alignment=Head()
    count=[0]
    def predict(tensor):
        count[0]+=1
        return SimpleNamespace(raw={d:(.2 if count[0]==1 else .6) for d in DISEASES},model_outputs=DISEASES,inference_only_ms=1.,model_name='Test model',unsupported_labels=[])
    engine=SimpleNamespace(resolution=32,pixel_scale=1.,crop='resize',predict=predict,model=Model(),model_name='Test model',fingerprint='test')
    seen={}
    model_client=SimpleNamespace(backend='transformers_local',model_id='Test VLM',lora_adapter_path=None,_last_vision_inputs={'image_count':3,'original_image_count':2})
    def generate(**kwargs):
        seen.update(kwargs)
        return VerificationResult(available=True,v7_report=json.dumps({'sections':[{'heading':heading,'text':'No focal consolidation.'} for heading in review_store.DEFAULT_TEMPLATE['sections']]}))
    service=InferenceService();service._engine=engine;service._engine_device='cpu';service._verification_engine=SimpleNamespace(client=model_client,generate_v7_report=generate)
    identifier=str(uuid4())
    result=service._run_pipeline_sync(identifier,str(paths[0]),{'additional_image_paths':[str(paths[1])]},True)
    assert result['status']=='completed'
    assert count[0]==2
    assert all(mapping.score==pytest.approx(.4) for mapping in seen['mappings'])
    assert len(seen['image_path'])==2 and all(__import__('pathlib').Path(path).is_file() for path in seen['image_path'])
    assert seen['patient_info']['original_image_count']==2
    assert __import__('pathlib').Path(seen['contact_sheet_path']).is_file()
    assert client.get(f'/api/v1/reviews/{identifier}/image/original_2').status_code==200
