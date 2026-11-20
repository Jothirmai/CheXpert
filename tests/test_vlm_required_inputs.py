import json
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from chest_xpert.verification import LLMModelClient
from chest_xpert.label_mapping import LabelMapping

@pytest.fixture
def client(tmp_path,monkeypatch):
    paths=[]
    for name,color in [('original.png',20),('gradcam.png',180)]:
        path=tmp_path/name;Image.new('RGB',(32,32),(color,color,color)).save(path);paths.append(str(path))
    c=LLMModelClient.__new__(LLMModelClient)
    c.backend='transformers_local';c.model_id='test-VL';c._is_vlm=True
    c.is_configured=lambda:True;c._cleanup_gpu=lambda:None
    c._model=SimpleNamespace(parameters=lambda:iter([torch.zeros(1)]),generate=lambda **kw:torch.tensor([[1,2,3,4,7]]))
    seen={}
    class Processor:
        def apply_chat_template(self,messages,**kwargs):
            seen['messages']=messages;return 'chat'
        def __call__(self,**kwargs):
            seen['images']=kwargs['images']
            return {'input_ids':torch.tensor([[1,2,3,4]]),'image_grid_thw':torch.tensor([[1,2,2],[1,2,2]])}
        def decode(self,*args,**kwargs):return '{"sections":[{"heading":"Findings","text":"No pleural effusion."}]}'
    c._processor=Processor()
    return c,paths,seen


def test_all_inputs_reach_processor_even_with_old_image_only_setting(client,monkeypatch):
    c,paths,seen=client
    monkeypatch.setenv('CHESTXPERT_REPORT_EVIDENCE','image_only')
    result=c.generate_v7_report([LabelMapping('Effusion',True,'Effusion',.73)],'CNN',{},*paths,self_correct=False)
    assert result.available,result.error
    assert len(seen['images'])==2
    assert seen['images'][0].getpixel((0,0))==(20,20,20)
    assert seen['images'][1].getpixel((0,0))==(180,180,180)
    content=seen['messages'][1]['content']
    prompt=content[-1]['text']
    assert json.loads(prompt.split('Supporting CNN scores: ')[1].split(' Model: ')[0])=={'Effusion':.73}
    assert c._last_vision_inputs['image_count']==2
    assert len(c._last_vision_inputs['image_sha256'])==2


@pytest.mark.parametrize('failure',['missing_sheet','missing_scores','nan_score','same_image','missing_original','remote','none_score','duplicate_label'])
def test_missing_evidence_never_calls_model(client,failure):
    c,paths,seen=client;scores=[LabelMapping('Effusion',True,'Effusion',.73)]
    if failure=='missing_sheet':paths[1]=None
    if failure=='missing_original':paths[0]=None
    if failure=='same_image':paths[1]=paths[0]
    if failure=='missing_scores':scores=[]
    if failure=='nan_score':scores=[LabelMapping('Effusion',True,'Effusion',float('nan'))]
    if failure=='remote':c.backend='hf_api'
    if failure=='none_score':scores=[LabelMapping('Effusion',True,'Effusion',None)]
    if failure=='duplicate_label':scores=scores+scores
    result=c.generate_v7_report(scores,'CNN',{},*paths,self_correct=False)
    assert not result.available and result.error
    assert seen=={}


def test_processor_cannot_silently_drop_an_image(client):
    c,paths,seen=client
    class DropImage:
        def apply_chat_template(self,*args,**kwargs):return 'chat'
        def __call__(self,**kwargs):return {'input_ids':torch.tensor([[1,2,3,4]]),'image_grid_thw':torch.tensor([[1,2,2]])}
    c._processor=DropImage()
    result=c.generate_v7_report([LabelMapping('Effusion',True,'Effusion',.73)],'CNN',{},*paths,self_correct=False)
    assert not result.available
    assert 'retain every supplied image' in result.error


def test_training_uses_both_images_scores_and_no_reference_findings(client):
    from scripts.train_report_adapter import build_messages
    c,paths,seen=client
    row={'image_path':paths[0],'contact_sheet_path':paths[1],'cnn_scores':{'Effusion':.73},'target':{'sections':[{'heading':'Findings','text':'REFERENCE_FINDINGS_MUST_NOT_LEAK'}]}}
    messages,images=build_messages(row)
    assert len(images)==2
    assert images[0].getpixel((0,0))==(20,20,20)
    assert images[1].getpixel((0,0))==(180,180,180)
    assert sum(c.get('type')=='image' for c in messages[1]['content'])==2
    prompt=messages[1]['content'][-1]['text']
    assert '0.73' in prompt and 'Effusion' in prompt
    assert 'REFERENCE_FINDINGS_MUST_NOT_LEAK' not in prompt


def test_self_correction_keeps_all_three_inputs(client):
    c,paths,seen=client
    result=c.generate_v7_report([LabelMapping('Effusion',True,'Effusion',.73)],'CNN',{},*paths,self_correct=True)
    assert result.available,result.error
    assert len(seen['images'])==2
    assert c._last_vision_inputs['image_count']==2
    assert '0.73' in seen['messages'][1]['content'][-1]['text']


def test_multiple_original_views_are_retained_before_contact_sheet(client,tmp_path):
    c,paths,seen=client
    extra=tmp_path/'extra.png';Image.new('RGB',(32,32),(120,120,120)).save(extra)
    old=c._processor
    class MultiProcessor:
        def apply_chat_template(self,*args,**kwargs):return old.apply_chat_template(*args,**kwargs)
        def __call__(self,**kwargs):
            result=old(**kwargs);result['image_grid_thw']=torch.tensor([[1,2,2],[1,2,2],[1,2,2]]);return result
        def decode(self,*args,**kwargs):return old.decode(*args,**kwargs)
    c._processor=MultiProcessor()
    result=c.generate_v7_report([LabelMapping('Effusion',True,'Effusion',.73)],'CNN',{},[paths[0],str(extra)],paths[1],self_correct=False)
    assert result.available,result.error
    assert [image.getpixel((0,0)) for image in seen['images']]==[(20,20,20),(120,120,120),(180,180,180)]
    assert c._last_vision_inputs['image_roles']==['original_xray','original_xray','gradcam_contact_sheet']
    assert c._last_vision_inputs['original_image_count']==2


def test_training_keeps_every_original_study_view(client, tmp_path):
    from scripts.train_report_adapter import build_messages
    from PIL import Image
    c,paths,seen=client
    extra=tmp_path/'another_view.png';Image.new('RGB',(8,8),(120,120,120)).save(extra)
    row={'image_path':paths[0],'image_paths':[paths[0],str(extra)],'contact_sheet_path':paths[1],
         'cnn_scores':{'Effusion':.73},'target':{'sections':[{'heading':'Findings','text':'NO_REFERENCE_LEAK'}]}}
    messages,images=build_messages(row)
    assert [im.getpixel((0,0)) for im in images]==[(20,20,20),(120,120,120),(180,180,180)]
    assert sum(part.get('type')=='image' for part in messages[1]['content'])==3
    prompt=messages[1]['content'][-1]['text']
    assert 'first 2 images' in prompt and '0.73' in prompt
    assert 'NO_REFERENCE_LEAK' not in prompt
