"""Structured predictions -> conventional clinical prose; no reference answers."""
import json,re
from chest_xpert.structured_schema import FindingResponse
from chest_xpert.report_quality import validate_clinical_sections
SUPPORTED_STATES=('present','absent','uncertain')
NAMES={'Cardiomegaly':'cardiac enlargement','Effusion':'pleural effusion','Opacity':'pulmonary opacity','Hardware':'medical hardware','Low lung volumes':'low lung volumes','Technical limitation':'technical limitation','Lung collapse':'lung collapse'}

def parse_prediction(raw,finding):
 match=re.search(r'\{.*\}',raw or '',re.S)
 if not match:raise ValueError('No structured finding JSON')
 result=FindingResponse.model_validate_json(match.group()).model_dump(exclude_none=True)
 if result['finding']!=finding:raise ValueError('Model answered a different finding')
 if any(o['status'] not in SUPPORTED_STATES for o in result['observations']):raise ValueError('Untrained state is masked in this three-state experiment')
 return result

def score_prediction(raw,query):
 expected=query['target'];states={o['status'] for o in expected['observations']}
 result={'valid':False,'class_correct':False,'scope_correct':False,'positive_recovered':False,'invented_positive':False,'attributes_correct':False}
 try:prediction=parse_prediction(raw,query['finding'])
 except (ValueError,TypeError):return result
 predicted={o['status'] for o in prediction['observations']};result.update(valid=True,class_correct=predicted==states,positive_recovered=('present' in states and 'present' in predicted),invented_positive=('present' not in states and 'present' in predicted))
 result['scope_correct']=result['class_correct'] and all(any(p['status']==t['status'] and p['assessment_scope']==t['assessment_scope'] for p in prediction['observations']) for t in expected['observations'])
 result['attributes_correct']=result['scope_correct'] and all(any(all(p.get(k)==v for k,v in t.items()) for p in prediction['observations']) for t in expected['observations'])
 result['prediction']=prediction;return result

def summarize_predictions(cases):
 return {'queries':len(cases),'valid':sum(c['score']['valid'] for c in cases),'class_correct':sum(c['score']['class_correct'] for c in cases),'scope_correct':sum(c['score']['scope_correct'] for c in cases),'attributes_correct':sum(c['score']['attributes_correct'] for c in cases),'positive_targets':sum(any(o['status']=='present' for o in c['target']['observations']) for c in cases),'positive_recovered':sum(c['score']['positive_recovered'] for c in cases),'invented_positive':sum(c['score']['invented_positive'] for c in cases),'absent_only':bool(cases) and all(c['score'].get('prediction') and all(o['status']=='absent' for o in c['score']['prediction']['observations']) for c in cases)}

def observation_sentence(finding,obs):
 name=NAMES.get(finding,finding.lower());scope=obs['assessment_scope'];parts=[obs.get('degree'),obs.get('laterality'),obs.get('region'),name];description=' '.join(str(x) for x in parts if x)
 if obs['status']=='absent':
  if finding=='Cardiomegaly' and not scope:return 'Cardiac silhouette is not enlarged.'
  qualifiers=' '.join(scope[k] for k in ['size','degree','pattern','temporal','visibility'] if scope.get(k))
  description='No '+('definite findings of ' if scope.get('certainty_threshold')=='definite' else '')+(qualifiers+' ' if qualifiers else '')+name
 elif obs['status']=='uncertain':description='Possible '+description
 elif obs['status']!='present':raise ValueError('Unsupported clinical state')
 sentence=description[0].upper()+description[1:]+'.'
 # Reuse the clinical metadata/publication guard even for model attribute strings.
 validate_clinical_sections([{'heading':'Findings','text':sentence}],['Findings'])
 return sentence

def render_sections(predictions,metadata):
 sentences=[];positive=[];negative=[]
 for prediction in predictions:
  FindingResponse.model_validate(prediction)
  for obs in prediction['observations']:
   sentence=observation_sentence(prediction['finding'],obs)
   if sentence not in sentences:sentences.append(sentence)
   if obs['status'] in ('present','uncertain') and sentence not in positive:positive.append(sentence)
   elif obs['status']=='absent' and sentence not in negative:negative.append(sentence)
 if not sentences:raise ValueError('No valid structured findings; refusing a generic normal report')
 sections=[{'heading':'Clinical indication','text':metadata.get('clinical_indication') or 'NOT PROVIDED'},{'heading':'Comparison','text':metadata.get('comparison_study') or 'NOT PROVIDED'},{'heading':'Technique','text':metadata.get('technique') or 'NOT PROVIDED'},{'heading':'Findings','text':' '.join(sentences)},{'heading':'Impression','text':' '.join(positive or negative[:3])}]
 return validate_clinical_sections(sections,[s['heading'] for s in sections])
