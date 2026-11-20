"""Versioned, auditable reference-assertion scoring; never creates training labels.

States, uncertainty, explicit normal descriptions and location are kept separate.
Unmentioned reference findings remain unknown. This is a lexical software check,
not image-based diagnostic adjudication or a comprehensive clinical NLP system.
"""
from __future__ import annotations
import re
from collections import defaultdict
from chest_xpert.disease_targets import DISEASES,PATTERNS
from chest_xpert.report_quality import parse_generated_sections
VERSION='reference_assertions_v2.0'
PATTERNS_V2=dict(zip(DISEASES,PATTERNS))
PATTERNS_V2.update({
 'Cardiomegaly':r'cardiomegaly|cardiac enlargement|(?:heart|cardiac silhouette|cardiomediastinal silhouette|heart size)(?:\s+\w+){0,5}\s+(?:enlarged|large)|(?:enlarged|large)\s+(?:cardiac silhouette|heart)|increased cardiothoracic ratio',
 'Opacity':r'opacit(?:y|ies)|airspace disease',
 'Granuloma':r'granuloma\w*|granulomatous disease',
 'Calcification':r'calcifi\w*|calcinosis',
 'Hyperinflation':r'hyperinflation|hyperinflated|hyperexpanded|hyperexpansion|hyperdistention',
 'Low lung volumes':r'low lung volumes?|low lung volume study|(?:lung )?volumes? (?:are |is )?low|(?:decreased|reduced|diminished) lung volumes?|hypoinflat\w*',
 'Bronchovascular crowding':r'broncho(?:vascular|pulmonary) crowding|crowding of (?:the )?bronchovascular',
 'Rotation':r'rotated (?:frontal |chest )?(?:view|radiograph|study)|(?:view|radiograph|study) (?:is )?rotated|patient rotation',
 'Spinal kyphosis':r'kyphos\w*|kyphotic(?=[^.;]{0,45}(?:spine|vertebrae))',
 'Kyphotic positioning':r'(?:view|radiograph|position)[^.;]{0,25}kyphotic|kyphotic[^.;]{0,25}(?:view|radiograph|position)',
 'Technical limitation':r'technic\w* limit\w*|(?:study|examination|exam|radiograph) (?:is )?(?:somewhat |mildly )?limit\w*|suboptimal (?:study|examination|radiograph)',
 'Limited inspiration':r'(?:limited|poor|suboptimal|shallow) inspirat\w*',
 'Hardware':r'(?:bone )?anchors?|(?:(?:orthopedic|medical|implanted) )?(?:devices?|hardware)',
 'Elevated hemidiaphragm':r'elevation of (?:the )?(?:right |left )?hemidiaphragm|(?:elevated|raised) (?:right |left )?hemidiaphragm',
 'Flattened diaphragms':r'flatten(?:ing|ed) (?:of (?:the )?)?diaphragms?',
 'Aortic tortuosity':r'(?:aorta|aortic contour) (?:is )?[^.;]{0,25}tortuous|tortuous[^.;]{0,20}aorta',
 'Pulmonary venous hypertension':r'pulmonary venous hypertension|pulmonary venous congestion',
 'Interstitial markings':r'(?:prominent|increased)[^.;]{0,25}interstitial markings|interstitial markings[^.;]{0,30}(?:prominent|increased)',
 'Mediastinal prominence':r'(?:prominen\w*[^.;]{0,25}mediastinum|mediastin\w*[^.;]{0,25}prominen\w*)',
 'Atherosclerosis':r'atherosclero\w*',
 'Spinal degeneration':r'degenerative changes?[^.;]{0,30}(?:spine|vertebrae)|(?:degenerated|degenerative)[^.;]{0,25}(?:thoracic spine|vertebrae)|(?:spine|vertebrae)[^.;]{0,25}degenerative',
 'Osteopenia':r'osteopeni\w*',
 'Apical pleural capping':r'apical pleural capping|pleural capping',
 'Fracture':r'fractur\w*',
})
NEG=re.compile(r'\b(?:no|not|without|absent|negative for|free of|clear of|unremarkable for|resolution of|resolved)\b',re.I)
UNCERTAIN=re.compile(r'\b(?:possible|possibly|probable|probably|likely|may|might|could|suspect\w*|suggest(?:s|ing|ive|ion)?|cannot exclude|can not exclude|cannot rule out|question\w*|concern for|concerning for)\b',re.I)
NORMAL_HEART=re.compile(r'\b(?:normal (?:heart size|cardiac (?:size|contours?))|(?:heart(?: size)?|cardiac(?: and mediastinal)? contours?|cardiomediastinal silhouette|cardiac silhouette)(?:\s+(?:is|are|appears?|remains?|and|mediastinum|within|at|on|the|upper|limits?|of|in|size|contour)){0,8}\s+normal)\b',re.I)

LOCATION={
 'left':r'\bleft\b','right':r'\bright\b','bilateral':r'\bbilateral(?:ly)?\b|\bboth (?:lungs|lower lobes|lung fields)\b|\bbibasilar\b',
 'base':r'\bbas(?:e|es|al|ilar)\b|\bbibasilar\b|\blower\b','upper':r'\bupper\b','mid':r'\bmid(?:lung)?\b',
 'apex':r'\bapic\w*\b|\bapex\b','posterior':r'\bposterior\b','lingula':r'\blingul\w*\b','peripheral':r'\bperipheral\b',
 'aorta':r'\baort\w*\b','shoulder':r'\bhumer\w*\b|\bshoulder\b','thoracic spine':r'\bthoracic (?:spine|vertebrae)\b',
}
LOCATED={'Atelectasis','Consolidation','Effusion','Opacity','Granuloma','Nodule','Mass','Hardware','Elevated hemidiaphragm','Calcification'}

def _segments(text):
 # Remove assertion headings, including headings embedded in one JSON paragraph.
 names='|'.join(re.escape(n) for n in DISEASES)
 text=re.sub(r'\b(?:'+names+r')(?:\s*\([^)]*\))?\s*:\s*','',text,flags=re.I)
 text=re.sub(r'\bno (?:significant )?(?:interval )?change\b','stable',text,flags=re.I)
 for sentence in re.split(r'[.;!\n]|\bbut\b|\bhowever\b|\bwhereas\b',text,flags=re.I):
  # An explicit new anatomical subject ends coordinated negation scope.
  for clause in re.split(r',\s*(?=(?:the |there (?:is|are) )?(?:heart|aorta|mediastinum|lungs)\b)|\band\s+(?=(?:the )?(?:heart|aorta|lungs)\b)',sentence,flags=re.I):
   if clause.strip():yield clause.strip()

def _polarity(clause,start,end,fact):
 before=clause[:start];prefix=clause[:end];suffix=clause[end:]
 # Keep diagnostic uncertainty distinct from negation, including cannot exclude.
 if re.search(r'cannot (?:be )?(?:excluded|ruled out)|can not (?:be )?excluded',suffix,re.I):return 'uncertain'
 uncertain=list(UNCERTAIN.finditer(prefix))
 neg=list(NEG.finditer(prefix))
 if uncertain and (not neg or uncertain[-1].end()>=neg[-1].end()):return 'uncertain'
 if re.search(r'\b(?:is|are)\s+(?:absent|not seen|not present|resolved)\b',suffix,re.I):return 'negative'
 if neg:
  if fact=='Cardiomegaly' and re.search(r'\bno acute\s*$',before,re.I):return 'acute_only_negative'
  return 'negative'
 return 'positive'

def _degree(clause,start,end):
 context=clause[max(0,start-25):min(len(clause),end+15)]
 for pattern,degree in [(r'\bborderline\b','borderline'),(r'\b(?:slight(?:ly)?|mild(?:ly)?)\b','mild'),(r'\bmoderate(?:ly)?\b','moderate'),(r'\b(?:marked(?:ly)?|severe(?:ly)?|gross(?:ly)?)\b','severe')]:
  if re.search(pattern,context,re.I):return degree
 return None

def extract(sections):
 mentions=defaultdict(list)
 for section in sections:
  if section['heading'].casefold() not in ('findings','impression'):continue
  for clause in _segments(section['text']):
   for fact,pattern in PATTERNS_V2.items():
    for match in re.finditer(r'\b(?:'+pattern+r')\b',clause,re.I):
     state=_polarity(clause,match.start(),match.end(),fact)
     # Kyphotic spine is anatomy; no automatic positioning-quality label exists.
     locations=sorted(k for k,p in LOCATION.items() if re.search(p,clause,re.I)) if fact in LOCATED else []
     mentions[fact].append({'state':state,'section':section['heading'],'text':clause,'locations':locations,'degree':_degree(clause,match.start(),match.end()) if fact=='Cardiomegaly' else None,'redacted_context':'[REDACTED]' in clause or 'XXXX' in clause})
   if NORMAL_HEART.search(clause) and not re.search(r'\bnot (?:within )?normal\b',clause,re.I):mentions['Cardiomegaly'].append({'state':'negative','section':section['heading'],'text':clause,'locations':[]})
 result={}
 for fact in PATTERNS_V2:
  mm=mentions[fact];observed={m['state'] for m in mm}
  state='conflicting' if {'positive','negative'}<=observed else 'positive' if 'positive' in observed else 'uncertain' if 'uncertain' in observed else 'negative' if 'negative' in observed else 'unmentioned'
  result[fact]={'state':state,'mentions':mm}
 return result

def _location_match(fact,truth,prediction):
 if fact not in LOCATED:return True
 reference=[set(m['locations']) for m in truth['mentions'] if m['state'] in ('positive','uncertain') and m['locations']]
 predicted=[set(m['locations']) for m in prediction['mentions'] if m['state'] in ('positive','uncertain')]
 if not reference:return True
 # Don't discard detailed locations because an impression has a general mention.
 required=set().union(*reference)
 return any(required<=p for p in predicted)

def score(raw,row):
 truth=extract(row['target']['sections']);positive=[k for k,v in truth.items() if v['state']=='positive'];qualified=[k for k,v in truth.items() if v['state']=='uncertain']
 result={'scorer_version':VERSION,'accepted':False,'reference_positive_facts':len(positive),'reference_qualified_facts':len(qualified),'matched_facts':0,'fully_matched_facts':[],'missed_facts':positive.copy(),'missed_qualified_facts':qualified.copy(),'explicit_denials':[],'location_errors':[],'qualification_errors':[],'contradictions':0,'unsupported_facts':[],'qualified_claims_against_reference_negatives':[],'unverified_additional_positive_claims':[]}
 try:sections,_=parse_generated_sections(raw,[s['heading'] for s in row['target']['sections']])
 except (ValueError,KeyError,TypeError):return result
 prediction=extract(sections);result['accepted']=True;result['contradictions']=sum(v['state']=='conflicting' for v in prediction.values());matched=[];qmatched=[]
 for fact in positive+qualified:
  t,p=truth[fact],prediction[fact];ps=p['state'];ts=t['state']
  if ps=='negative':result['explicit_denials'].append(fact)
  elif ps in ('positive','uncertain'):
   location_ok=_location_match(fact,t,p)
   if not location_ok:result['location_errors'].append(fact)
   degree_ok=True
   if fact=='Cardiomegaly':
    degrees={m.get('degree') for m in t['mentions'] if m['state'] in ('positive','uncertain') and m.get('degree')}
    predicted_degrees={m.get('degree') for m in p['mentions'] if m['state'] in ('positive','uncertain') and m.get('degree')}
    degree_ok=not degrees or bool(degrees & predicted_degrees)
    if not degree_ok:result['qualification_errors'].append({'fact':fact,'type':'cardiac_enlargement_degree','reference':sorted(degrees),'prediction':sorted(predicted_degrees) or ['unspecified']})
   if ps!=ts:result['qualification_errors'].append({'fact':fact,'type':'certainty','reference':ts,'prediction':ps})
   if location_ok and degree_ok and ps==ts:(matched if ts=='positive' else qmatched).append(fact)
 result['fully_matched_facts']=matched;result['matched_facts']=len(matched);result['missed_facts']=[k for k in positive if k not in matched];result['missed_qualified_facts']=[k for k in qualified if k not in qmatched]
 for fact,p in prediction.items():
  t=truth[fact]['state'];state=p['state']
  if t=='negative' and state in ('positive','conflicting'):result['unsupported_facts'].append(fact)
  elif t=='negative' and state=='uncertain':result['qualified_claims_against_reference_negatives'].append(fact)
  elif t=='unmentioned' and state in ('positive','uncertain','conflicting'):result['unverified_additional_positive_claims'].append(fact)
 result['reference_assertions']=truth;result['prediction_assertions']=prediction
 return result

def summarize(cases):
 lists=['missed_facts','missed_qualified_facts','explicit_denials','location_errors','qualification_errors','unsupported_facts','qualified_claims_against_reference_negatives','unverified_additional_positive_claims']
 return {'cases':len(cases),'accepted_reports':sum(c['accepted'] for c in cases),'reference_positive_facts':sum(c['reference_positive_facts'] for c in cases),'reference_qualified_facts':sum(c['reference_qualified_facts'] for c in cases),'matched_facts':sum(c['matched_facts'] for c in cases),'contradictions':sum(c['contradictions'] for c in cases),**{k:sum(len(c[k]) for c in cases) for k in lists}}
