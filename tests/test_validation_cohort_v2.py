import pytest
from chest_xpert.validation_cohort import select_validation_cohort

def row(study,text,normal=False):return {'study_id':study,'target':{'sections':[{'heading':'Findings','text':text}]},'mesh_terms':['normal'] if normal else []}

def test_rare_overlapping_quality_cases_are_reserved_before_abnormal_cases():
 rows=[row('quality','Technically limited study. Low lung volumes.'),row('low','Low lung volumes.'),row('chronic','Kyphotic degenerated thoracic spine.'),row('normal','Normal heart size. Clear lungs.',True)]
 selected,categories,pools=select_validation_cohort(rows,quotas={k:1 for k in ['technical','low_volumes','normal','abnormal']})
 assert len(selected)==4 and len({r['study_id'] for r in selected})==4
 assert categories=={'low':'low_volumes','quality':'technical','chronic':'abnormal','normal':'normal'}
 assert pools['technical']==1

def test_insufficient_quality_pool_fails_instead_of_relabeling_spinal_kyphosis():
 rows=[row('chronic','Kyphotic thoracic spine.')]
 with pytest.raises(ValueError,match='genuine technical'):
  select_validation_cohort(rows,quotas={'technical':1,'low_volumes':0,'normal':0,'abnormal':0})
