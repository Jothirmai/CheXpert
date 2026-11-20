"""Disjoint, source-based validation strata; reserve rare quality cases first."""
import random
from chest_xpert.report_scoring_v2 import extract
QUALITY_FACTS=('Technical limitation','Rotation','Kyphotic positioning','Limited inspiration')
DEFAULT_QUOTAS={'low_volumes':6,'technical':5,'abnormal':7,'normal':6}

def select_validation_cohort(rows,seed=43,quotas=None):
 quotas=dict(DEFAULT_QUOTAS if quotas is None else quotas);rows=list(rows);random.Random(seed).shuffle(rows)
 facts={r['study_id']:extract(r['target']['sections']) for r in rows}
 positive=lambda r,name:facts[r['study_id']][name]['state'] in ('positive','uncertain')
 pools={
  'technical':[r for r in rows if any(positive(r,k) for k in QUALITY_FACTS)],
  'low_volumes':[r for r in rows if positive(r,'Low lung volumes')],
  'normal':[r for r in rows if any(t.lower()=='normal' for t in r.get('mesh_terms',[])) and not any(v['state'] in ('positive','uncertain','conflicting') for v in facts[r['study_id']].values())],
  'abnormal':[r for r in rows if any(v['state'] in ('positive','uncertain') for v in facts[r['study_id']].values())],
 }
 chosen={};seen=set()
 for category in ('technical','low_volumes','normal','abnormal'):
  options=[r for r in pools[category] if r['study_id'] not in seen]
  wanted=quotas[category]
  if len(options)<wanted:raise ValueError(f'Insufficient genuine {category} cases: need {wanted}, have {len(options)}; do not relabel another category')
  chosen[category]=options[:wanted];seen.update(r['study_id'] for r in chosen[category])
 selected=[];categories={}
 for i in range(max(quotas.values())):
  for category in ('low_volumes','technical','abnormal','normal'):
   if i<len(chosen[category]):
    row=chosen[category][i];selected.append(row);categories[row['study_id']]=category
 return selected,categories,{k:len(v) for k,v in pools.items()}
