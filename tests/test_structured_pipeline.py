import pytest
from chest_xpert.structured_pipeline import parse_prediction,score_prediction,render_sections

def response(status='present',scope=None,**attrs):return {'finding':'Effusion','observations':[{'status':status,'assessment_scope':scope or {},**attrs}]}

def test_unsupported_state_and_wrong_finding_are_rejected():
 import json
 with pytest.raises(ValueError):parse_prediction(json.dumps(response('unassessable')),'Effusion')
 with pytest.raises(ValueError):parse_prediction(json.dumps(response()),'Cardiomegaly')

def test_report_preserves_uncertainty_and_size_limited_negative():
 sections=render_sections([response('uncertain',laterality='left'),response('absent',{'size':'large'})],{})
 findings=next(s['text'] for s in sections if s['heading']=='Findings')
 assert 'Possible left pleural effusion.' in findings and 'No large pleural effusion.' in findings
 assert 'No acute' not in str(sections) and 'Grad-CAM' not in str(sections)


def test_report_refuses_empty_observations_and_technical_attribute_text():
 with pytest.raises(ValueError):render_sections([],{})
 with pytest.raises(ValueError):render_sections([response(region='Grad-CAM heatmap')],{})

def test_status_correct_is_not_scope_correct_if_qualified_negative_was_dropped():
 import json
 q={'finding':'Effusion','target':response('absent',{'size':'large'})}
 s=score_prediction(json.dumps(response('absent')),q)
 assert s['class_correct'] and not s['scope_correct']
