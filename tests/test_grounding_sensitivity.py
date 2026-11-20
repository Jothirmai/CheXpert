import itertools
from scripts.analyze_report_grounding import sensitivity

def test_factorial_pairing_only_changes_one_input_at_a_time():
 records=[]
 for mode in ['base','adapter']:
  for i,(original,cam,scores) in enumerate(itertools.product(['normal','abnormal'],repeat=3)):
   records.append({'mode':mode,'original_source':original,'cam_source':cam,'score_source':scores,'accepted':True,'condition_id':mode+str(i),'clinical_text':original if mode=='base' else 'constant','assertion_profile':original if mode=='base' else 'constant'})
 result=sensitivity(records)
 assert result['base']['original_source']['assertion_profile_changes']==4
 assert result['base']['cam_source']['assertion_profile_changes']==0
 assert result['base']['score_source']['assertion_profile_changes']==0
 assert all(v['assertion_profile_changes']==0 for v in result['adapter'].values())
