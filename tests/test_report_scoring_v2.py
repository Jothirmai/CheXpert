import json
import pytest
from chest_xpert.report_scoring_v2 import extract,score

def sections(text):return [{'heading':'Findings','text':text},{'heading':'Impression','text':'No acute findings.'}]
def state(text,fact):return extract(sections(text))[fact]['state']
def result(ref,pred,mesh=()):return score(json.dumps({'sections':sections(pred)}),{'target':{'sections':sections(ref)},'mesh_terms':mesh})

@pytest.mark.parametrize('text,fact,expected',[
 ('The heart is normal in size.','Cardiomegaly','negative'),
 ('Normal heart size.','Cardiomegaly','negative'),
 ('Cardiac and mediastinal contours are within normal limits.','Cardiomegaly','negative'),
 ('Heart size is on the upper limits of normal.','Cardiomegaly','negative'),
 ('Heart size remains slightly large.','Cardiomegaly','positive'),
 ('The heart is large.','Cardiomegaly','positive'),
 ('The cardiac silhouette appears enlarged.','Cardiomegaly','positive'),
 ('Lung volumes are low.','Low lung volumes','positive'),
 ('Decreased lung volumes with bronchopulmonary crowding.','Low lung volumes','positive'),
 ('Decreased lung volumes with bronchopulmonary crowding.','Bronchovascular crowding','positive'),
 ('Lungs are hyperexpanded.','Hyperinflation','positive'),
 ('Probable posterior recess effusions.','Effusion','uncertain'),
 ('Effusion cannot be excluded.','Effusion','uncertain'),
 ('Cannot exclude effusion.','Effusion','uncertain'),
 ('Unremarkable for cardiomegaly.','Cardiomegaly','negative'),
 ('Kyphotic degenerated osteopenic thoracic spine.','Spinal kyphosis','positive'),
 ('Kyphotic degenerated osteopenic thoracic spine.','Kyphotic positioning','unmentioned'),
 ('Frontal view is kyphotic and rotated.','Spinal kyphosis','unmentioned'),
 ('Frontal view is kyphotic and rotated.','Kyphotic positioning','positive'),
 ('Technically limited study.','Technical limitation','positive'),
 ('Technically limited study.','Limited inspiration','unmentioned'),
 ('Left humeral head bone anchors.','Hardware','positive'),
 ('No acute cardiomegaly.','Cardiomegaly','unmentioned'),
 ('Effusion: No effusion.','Effusion','negative'),
 ('No interval change in the pleural effusion.','Effusion','positive'),
 ('No effusion and the heart is enlarged.','Cardiomegaly','positive'),
 ('No pneumothorax, pleural effusion, or consolidation.','Consolidation','negative'),
 ('Prominent interstitial markings suggesting pulmonary fibrosis.','Fibrosis','uncertain'),
])
def test_assertion_states(text,fact,expected):assert state(text,fact)==expected

def test_location_cannot_receive_full_match_for_wrong_side_and_region():
 r=result('Left midlung focal atelectasis.','Bilateral lower-zone atelectasis.')
 assert r['matched_facts']==0 and 'Atelectasis' in r['location_errors']
 r=result('Left basal atelectasis.','Left lower-lobe atelectasis.')
 assert r['matched_facts']==1

def test_uncertainty_is_preserved_and_explicit_denials_count():
 r=result('Probable left pleural effusion.','No pleural effusion.')
 assert r['reference_positive_facts']==0 and r['reference_qualified_facts']==1
 assert 'Effusion' in r['explicit_denials']
 r=result('Probable left pleural effusion.','Left pleural effusion.')
 assert r['qualification_errors'] and r['missed_qualified_facts']==['Effusion']

def test_source_silence_is_not_a_negative_even_with_normal_tag():
 r=result('No acute findings.','Hernia.', ['normal'])
 assert r['unsupported_facts']==[]
 assert 'Hernia' in r['unverified_additional_positive_claims']

def test_normal_heart_detects_reference_conflict_and_valid_paraphrase():
 r=result('Normal heart size.','Cardiomegaly.')
 assert 'Cardiomegaly' in r['unsupported_facts']
 r=result('Low lung volumes.','Lung volumes are low.')
 assert r['matched_facts']==1

def test_invalid_report_has_no_credit():
 r=score('not JSON',{'target':{'sections':sections('Low lung volumes.')}})
 assert not r['accepted'] and r['matched_facts']==0

@pytest.mark.parametrize('text,expected',[
 ('The heart is not enlarged.','negative'),
 ('The heart may be enlarged.','uncertain'),
 ('The heart size is not normal.','unmentioned'),
 ('Pleural effusion is not present.','negative'),
])
def test_negation_and_uncertainty_inside_multiword_assertions(text,expected):
 fact='Effusion' if 'effusion' in text.lower() else 'Cardiomegaly'
 assert state(text,fact)==expected

def test_review_warnings_use_correct_negation_and_explicit_normal_heart():
 from chest_xpert.report_quality import report_warnings
 assert report_warnings(sections('Unremarkable for cardiomegaly.'))==[]
 assert report_warnings([{'heading':'Findings','text':'Heart is normal in size.'},{'heading':'Impression','text':'Cardiomegaly.'}])
 assert report_warnings(sections('Effusion: No effusion.'))==[]

def test_cardiac_enlargement_degree_is_not_silently_dropped():
 r=result('Heart size remains slightly large.','Cardiomegaly.')
 assert r['matched_facts']==0
 assert r['qualification_errors'][0]['type']=='cardiac_enlargement_degree'
 r=result('Heart size remains slightly large.','Mild cardiomegaly.')
 assert r['matched_facts']==1

def test_qualified_claim_against_negative_is_not_a_definite_false_positive():
 r=result('No pleural effusion.','Possible effusion.')
 assert r['unsupported_facts']==[]
 assert r['qualified_claims_against_reference_negatives']==['Effusion']

def test_normal_lungs_do_not_create_a_false_normal_heart_assertion():
 assert state('The heart is enlarged and lungs are normal.','Cardiomegaly')=='positive'
 assert state('The heart is enlarged with normal lungs.','Cardiomegaly')=='positive'
 assert state('The heart and mediastinum are within normal limits.','Cardiomegaly')=='negative'
 assert state('The heart and mediastinum are within normal limits. There is no acute cardiomegaly.','Cardiomegaly')=='negative'
