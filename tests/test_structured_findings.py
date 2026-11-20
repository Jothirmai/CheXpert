import json
from pathlib import Path
import pytest
from PIL import Image
from pydantic import ValidationError
from chest_xpert.disease_targets import DISEASES
from chest_xpert.structured_findings import Annotation,bind_annotations,group_queries,supervision_matrix,build_structured_messages
from chest_xpert.structured_schema import FindingResponse

def annotation(finding='Effusion',status='absent',quote='No pleural effusion.',**kw):
 return {'finding':finding,'status':status,'quote':quote,**kw}

def test_missing_is_masked_and_is_neither_absent_nor_unassessable():
 matrix=supervision_matrix([])
 assert all(v['status'] is None and not v['loss_mask'] and not v['whole_finding_supervised'] for v in matrix.values())

@pytest.mark.parametrize('text',['No acute findings.','Clear lungs.'])
def test_generic_normal_prose_is_not_a_negative_disease_label(text):
 with pytest.raises(ValidationError):Annotation(**annotation(quote=text))

@pytest.mark.parametrize('text',['Cannot exclude a pleural effusion.','Technically limited study with pleural effusion.'])
def test_uncertainty_and_general_limitations_are_not_unassessable(text):
 with pytest.raises(ValidationError):Annotation(**annotation(status='unassessable',quote=text))

def test_explicit_finding_unassessability_has_a_valid_schema():
 x=Annotation(**annotation('Cardiomegaly','unassessable','Heart size cannot be assessed.'))
 assert x.status=='unassessable'

@pytest.mark.parametrize('finding,quote,scope',[
 ('Effusion','No large pleural effusion.',{'size':'large'}),
 ('Consolidation','No focal consolidation.',{'pattern':'focal'}),
 ('Pneumonia','No definite pneumonia.',{'certainty_threshold':'definite'}),
 ('Pneumonia','No lobar pneumonia.',{'pattern':'lobar'}),
 ('Edema','No overt pulmonary edema.',{'degree':'overt'}),
])
def test_qualified_negative_cannot_silently_become_a_whole_finding_negative(finding,quote,scope):
 with pytest.raises(ValidationError):Annotation(**annotation(finding,quote=quote))
 x=Annotation(**annotation(finding,quote=quote,scope=scope));assert x.scope==scope


def test_pericardial_effusion_and_low_normal_are_not_false_positive_labels():
 with pytest.raises(ValidationError):Annotation(**annotation(status='present',quote='Possible pericardial effusion.'))
 with pytest.raises(ValidationError):Annotation(**annotation('Low lung volumes','present','Lung volumes are low normal.'))


def test_exact_spans_and_view_attribution_are_required():
 sections=[{'heading':'Findings','text':'The lateral view shows a small left pleural effusion.'}]
 x=annotation(status='present',quote=sections[0]['text'],supporting_view='lateral',laterality='left')
 bound=bind_annotations(sections,[x]);e=bound[0]['evidence']
 assert sections[0]['text'][e['start']:e['end']]==e['quote']
 assert bound[0]['field_loss_mask']['supporting_view']
 with pytest.raises(ValidationError):Annotation(**{**x,'supporting_view':'PA'})
 with pytest.raises(ValueError):bind_annotations(sections,[{**x,'quote':'The lateral view shows a right pleural effusion.'}])


def test_sparse_attributes_and_scoped_statuses_survive_grouping():
 ss=[{'heading':'Findings','text':'No large pleural effusion.'}]
 bound=bind_annotations(ss,[annotation(quote=ss[0]['text'],scope={'size':'large'})]);queries=group_queries('CXR1',bound)
 assert queries[0]['target']['observations'][0]=={'status':'absent','assessment_scope':{'size':'large'}}
 assert not queries[0]['field_loss_masks'][0]['supporting_view']
 assert not supervision_matrix(bound)['Effusion']['whole_finding_supervised']
 FindingResponse.model_validate(queries[0]['target'])


def test_input_builder_keeps_three_inputs_and_omits_answers_and_source_quotes(tmp_path):
 paths=[]
 for i in range(3):
  path=tmp_path/f'{i}.png';Image.new('RGB',(8,8),i*60).save(path);paths.append(str(path))
 record={'inputs':{'original_paths':paths[:2],'contact_sheet_path':paths[2],'cnn_scores':{k:.2 for k in DISEASES},'clinical_metadata':{}}}
 query={'finding':'Effusion','target':{'finding':'Effusion','observations':[{'status':'present','assessment_scope':{},'laterality':'left'}]},'source_evidence':[{'quote':'SOURCE_ANSWER_MUST_NOT_LEAK'}]}
 messages,images=build_structured_messages(record,query);prompt=messages[1]['content'][-1]['text']
 assert len(images)==3 and sum(p.get('type')=='image' for p in messages[1]['content'])==3
 assert 'SOURCE_ANSWER_MUST_NOT_LEAK' not in prompt and '"laterality": "left"' not in prompt and '"status": "present"' not in prompt
 for im in images:im.close()
 record['inputs']['cnn_scores']={}
 with pytest.raises(ValueError):build_structured_messages(record,query)


def test_response_schema_rejects_unmentioned_as_a_clinical_state_and_source_prose():
 with pytest.raises(ValidationError):FindingResponse.model_validate({'finding':'Effusion','observations':[{'status':'unmentioned','assessment_scope':{}}]})
 with pytest.raises(ValidationError):FindingResponse.model_validate({'finding':'Effusion','observations':[{'status':'present','assessment_scope':{},'source_quote':'answer text'}]})

def test_machine_readable_schema_has_the_same_finding_vocabulary():
 from chest_xpert.structured_findings import CATALOG
 assert set(FindingResponse.model_json_schema()['properties']['finding']['enum'])==set(CATALOG)

def test_scope_cannot_hide_report_answers_or_implementation_metadata():
 with pytest.raises(ValidationError):FindingResponse.model_validate({'finding':'Effusion','observations':[{'status':'absent','assessment_scope':{'source_quote':'unreviewed answer'}}]})
