"""Source-cited pilot targets. Unmentioned is unsupervised, not a clinical state."""
from __future__ import annotations
import hashlib,json,math,re
from collections import defaultdict
from pathlib import Path
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator
from chest_xpert.report_scoring_v2 import PATTERNS_V2
from chest_xpert.disease_targets import DISEASES

VERSION='source_cited_findings_v1'
STATES=('present','absent','uncertain','unassessable')
CATALOG=tuple(PATTERNS_V2)+('Lung collapse',)
VIEWS=('PA','AP','frontal','lateral')
class Annotation(BaseModel):
    model_config=ConfigDict(extra='forbid')
    finding:str
    status:Literal['present','absent','uncertain','unassessable']
    quote:str=Field(min_length=3)
    section:Literal['Findings','Impression']='Findings'
    laterality:Literal['left','right','bilateral']|None=None
    region:str|None=None
    degree:str|None=None
    scope:dict[str,str]=Field(default_factory=dict)
    supporting_view:Literal['PA','AP','frontal','lateral']|None=None
    note:str|None=None
    @model_validator(mode='after')
    def source_constraints(self):
        if self.finding not in CATALOG:raise ValueError('Unknown finding')
        aliases={
          'Cardiomegaly':r'heart|cardiac|cardiomediastinal|cardiomegaly',
          'Effusion':r'(?:pleural )?effusion|pleural fluid',
          'Opacity':r'opacit|opaque|airspace disease',
          'Low lung volumes':r'lung volumes?|hypoinflat',
          'Technical limitation':r'limit|suboptimal',
          'Hardware':r'device|hardware|anchors?|clips?|catheter|PICC',
          'Aortic tortuosity':r'tortuous|tortuosity',
          'Spinal degeneration':r'degenerat',
          'Spinal kyphosis':r'kyphotic|kyphosis',
          'Lung collapse':r'collapse',
        }
        pattern=aliases.get(self.finding,PATTERNS_V2.get(self.finding))
        if not re.search(pattern,self.quote,re.I):raise ValueError('Evidence does not name this finding or an approved synonym')
        if self.status=='absent' and not re.search(r'\b(?:no|not|without|free of|clear of|normal|unremarkable for|absent)\b',self.quote,re.I):
            raise ValueError('Absent target needs an explicit source negative/normal assertion')
        if self.status=='absent':
            restrictions={
             'Consolidation':[(r'focal (?:airspace )?consolidat', 'pattern', ('focal','focal airspace')),(r'alveolar consolidat','pattern',('alveolar',))],
             'Opacity':[(r'focal airspace (?:disease|opacit)', 'pattern',('focal airspace',))],
             'Infiltration':[(r'focal infiltrat','pattern',('focal',)),(r'acute infiltrat','temporal',('acute',))],
             'Pneumonia':[(r'lobar pneumonia','pattern',('lobar',)),(r'definite pneumonia','certainty_threshold',('definite',))],
             'Edema':[(r'overt pulmonary edema','degree',('overt',))],
            }
            for pattern,key,allowed in restrictions.get(self.finding,[]):
                if re.search(pattern,self.quote,re.I) and self.scope.get(key) not in allowed:raise ValueError('Preserve the restricted negative scope')
        if self.finding=='Effusion' and self.status=='absent':
            if re.search(r'large (?:pleural )?effusion',self.quote,re.I) and self.scope.get('size')!='large':raise ValueError('No large effusion cannot become a whole-finding negative')
            if re.search(r'definite (?:pleural )?effusion',self.quote,re.I) and self.scope.get('certainty_threshold')!='definite':raise ValueError('Preserve definite-finding negative scope')
        if self.finding=='Effusion' and self.status=='present' and re.search(r'pericardial effusion',self.quote,re.I) and not re.search(r'pleural (?:effusion|fluid)',self.quote,re.I):
            raise ValueError('Pericardial fluid is not a positive pleural-effusion target')
        if self.finding=='Low lung volumes' and self.status=='present' and re.search(r'volumes? (?:are )?low[ -]normal',self.quote,re.I):raise ValueError('Low-normal is not hypoinflation')
        if self.supporting_view and not re.search(r'\b'+re.escape(self.supporting_view)+r'\b',self.quote,re.I):raise ValueError('Supporting view must be explicitly stated in the source quote')
        if self.status=='unassessable':
            explicit=r'(?:cannot|can not|unable to) (?:be )?(?:assess(?:ed)?|evaluat(?:e|ed)|determin(?:e|ed))|not assessable|unassessable'
            if not re.search(explicit,self.quote,re.I):raise ValueError('General limitations or cannot-exclude wording do not establish unassessable')
        return self

def bind_annotations(sections,annotations):
    """Bind manually reviewed decisions to exact, immutable source spans."""
    source={s['heading']:s['text'] for s in sections};bound=[]
    for item in annotations:
        a=Annotation.model_validate(item);text=source[a.section];start=text.find(a.quote)
        if start<0:raise ValueError(f'{a.finding}: quote not present in {a.section}: {a.quote}')
        evidence={'section':a.section,'quote':a.quote,'start':start,'end':start+len(a.quote),'redacted_context':'[REDACTED]' in a.quote}
        fields={'status':a.status,'assessment_scope':a.scope}
        for name in ['laterality','region','degree','supporting_view']:
            value=getattr(a,name)
            if value is not None:fields[name]=value
        bound.append({'finding':a.finding,'target':fields,'evidence':evidence,'review_note':a.note,
                      'field_loss_mask':{name:(name in fields) for name in ['status','assessment_scope','laterality','region','degree','supporting_view']}})
    return bound

def group_queries(study_id,annotations):
    by_finding=defaultdict(list)
    for a in annotations:by_finding[a['finding']].append(a)
    queries=[]
    for finding,items in by_finding.items():
        observations=[];masks=[];evidence=[]
        for a in items:
            if a['target'] not in observations:
                observations.append(a['target']);masks.append(a['field_loss_mask'])
            evidence.append(a['evidence'])
        queries.append({'query_id':study_id+':'+finding,'study_id':study_id,'finding':finding,
                        'target':{'finding':finding,'observations':observations},'field_loss_masks':masks,
                        'source_evidence':evidence,'review_method':'AI-assisted source-text review; no independent clinical adjudication'})
    return queries

def supervision_matrix(annotations):
    """No default absent or unassessable classes for unannotated concepts."""
    result={}
    for finding in CATALOG:
        items=[a for a in annotations if a['finding']==finding]
        whole=[a for a in items if not a['target']['assessment_scope']]
        result[finding]={'supervised':bool(items),'whole_finding_supervised':bool(whole),
                         'status':None if not whole else sorted({a['target']['status'] for a in whole}),
                         'loss_mask':bool(items),'reason':'explicit source-cited annotation' if items else 'not annotated; no absence or assessability inferred'}
    return result

def build_structured_messages(record,query):
    """Only actual images, scores and clinical metadata enter the user prompt."""
    from PIL import Image
    paths=record['inputs']['original_paths'];sheet=record['inputs']['contact_sheet_path'];scores=record['inputs']['cnn_scores']
    if not paths or not sheet or set(scores)!=set(DISEASES) or not all(isinstance(v,(float,int)) and math.isfinite(v) and 0<=v<=1 for v in scores.values()):raise ValueError('Every original, contact sheet and all 14 finite scores are required')
    allpaths=[*paths,sheet]
    if len({str(Path(p).resolve()) for p in allpaths})!=len(allpaths):raise ValueError('Originals and contact sheet must be distinct')
    images=[];content=[]
    for i,path in enumerate(allpaths):
        with Image.open(path) as image:images.append(image.convert('RGB'))
        label=f'Original study radiograph {i+1}/{len(paths)}.' if i<len(paths) else 'Final image: supporting attention contact sheet, not a clinical projection.'
        content.extend([{'type':'text','text':label},{'type':'image','image':images[-1]}])
    # No evidence quotes, target status, location, view labels, or target scope are inputs.
    prompt=('Inspect all original views and assess the requested finding. Return JSON with finding and observations. Each observation has status (present, absent, uncertain, unassessable), assessment_scope, and only supportable laterality, region, degree and supporting_view. Preserve the scope of any negative statement; uncertain is distinct from inability to assess. Do not write a normal-report template. Scores are uncalibrated supporting signals, not diagnoses. Finding: '+query['finding']+'. Clinical metadata: '+json.dumps(record['inputs']['clinical_metadata'])+'. Exact CNN scores: '+json.dumps(scores))
    content.append({'type':'text','text':prompt})
    return [{'role':'system','content':'Return structured finding observations only. Ground findings in originals; the contact sheet is supporting evidence.'},{'role':'user','content':content}],images
