"""Build a small, source-cited structured pilot without changing active models."""
import argparse,hashlib,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from chest_xpert.structured_findings import VERSION,STATES,CATALOG,bind_annotations,group_queries,supervision_matrix,build_structured_messages
from chest_xpert.study_images import resolve_study_images,index_study_images
from chest_xpert.training_data import parse_clinical_xml,structured_target
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return list(map(json.loads,Path(path).read_text().splitlines()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--curation',default='reports/structured_targets/curation.json');p.add_argument('--output',default='data/structured_findings_v1');args=p.parse_args()
 out=Path(args.output)
 if (out/'manifest.json').exists():raise RuntimeError('Structured pilot is frozen; use a new version instead of overwriting it')
 curation=json.loads(Path(args.curation).read_text());active=Path('checkpoints/active_models.json');active_hash=sha(active);selection=json.loads(active.read_text())
 rows={}
 for file in ['data/report_training_v3/validation.jsonl','checkpoints/report_lora_full_study_v2/validation.jsonl','checkpoints/report_lora_full_study/enriched_train.jsonl','checkpoints/report_lora_full_study/enriched_validation.jsonl']:
  rows.update({r['study_id']:r for r in read(file)})
 split_ids={split:{r['study_id'] for r in read(f'data/report_training_v3/{split}.jsonl')} for split in ['train','validation']}
 assert not split_ids['train']&split_ids['validation']
 selected={c['study_id'] for c in curation['cases']};assert len(selected)==len(curation['cases'])
 assert not selected&{c['study_id'] for c in curation['quarantine']}
 evidence=json.loads(Path('checkpoints/report_lora_full_study/input_evidence.json').read_text());assert evidence['cnn_sha256']==selection['cnn']['sha256']
 old_inputs={r['study_id']:r for r in evidence['rows']}
 inventory=index_study_images('data/iu_xray/images');records=[];queries={'train':[],'validation':[]};owners=defaultdict(set)
 for c in curation['cases']:
  study=c['study_id'];split=c['split'];assert study in split_ids[split]
  r=rows[study];xml=Path(f'data/iu_xray/reports/ecgen-radiology/{study.removeprefix("CXR")}.xml');source=structured_target(parse_clinical_xml(xml)['sections'])
  assert source==r['target'],study+' source mismatch'
  paths=resolve_study_images(xml,'data/iu_xray/images',inventory)
  if len(paths)<2 and any('two-view' in s['text'].lower() or 'pa and lateral' in s['text'].lower() for s in source['sections'] if s['heading']=='Technique'):raise ValueError(study+' caption/view-count mismatch')
  annotations=bind_annotations(source['sections'],c['annotations'])
  if 'cnn_scores' not in r or 'contact_sheet_path' not in r:
   from scripts.train_report_adapter import enrich
   print('Preparing missing actual evidence:',study,flush=True)
   r=enrich([dict(r)],Path('data/report_training_v3/evidence'),'cuda:0')[0]
  assert r['image_paths']==[str(p) for p in paths]
  allpaths=r['image_paths']+[r['contact_sheet_path']];hashes=[sha(path) for path in allpaths]
  assert len(r['cnn_scores'])==14 and all(math.isfinite(v) and 0<=v<=1 for v in r['cnn_scores'].values())
  for path in allpaths:
   with Image.open(path) as image:image.load()
  if study in old_inputs:
   assert allpaths==old_inputs[study]['paths'] and hashes==old_inputs[study]['sha256'] and r['cnn_scores']==old_inputs[study]['cnn_scores']
  for path,h in zip(paths,hashes):owners[h].add(split)
  record={'schema_version':VERSION,'study_id':study,'split':split,
    'inputs':{'original_paths':r['image_paths'],'contact_sheet_path':r['contact_sheet_path'],'cnn_scores':r['cnn_scores'],
      'clinical_metadata':{'clinical_indication':r.get('clinical_indication') or 'NOT PROVIDED','comparison_study':r.get('comparison_study') or 'NOT PROVIDED','technique':r.get('technique') or 'NOT PROVIDED'}},
    'input_provenance':{'sha256':hashes,'cnn_checkpoint_sha256':selection['cnn']['sha256'],'original_image_count':len(paths),'contact_sheet_count':1},
    'reference':source,'source_xml':str(xml),'source_xml_sha256':sha(xml),'annotations':annotations,'supervision':supervision_matrix(annotations),
    'review':{'method':curation['review_method'],'exact_source_spans_verified':True,'independent_image_adjudication':False,'notes':c['review_notes']}}
  qq=group_queries(study,annotations)
  # Verify that the training input helper cannot leak the answer or source quotes.
  for q in qq:
   messages,images=build_structured_messages(record,q);prompt=messages[1]['content'][-1]['text']
   assert sum(v.get('type')=='image' for v in messages[1]['content'])==len(paths)+1
   for text in [s['text'] for s in source['sections'] if s['heading'] in ['Findings','Impression']]:
    if len(text)>35:assert text not in prompt,study+' source answer leaked'
   for image in images:image.close()
   queries[split].append({**q,'split':split,'record_id':study})
  records.append(record)
 assert all(len(splits)==1 for splits in owners.values()),'Cross-split image duplicate'
 counts={split:Counter(a['target']['status'] for r in records if r['split']==split for a in r['annotations']) for split in queries}
 # Sampling is study-normalized first, then balanced by available target strata.
 sampler={}
 for split,qq in queries.items():
  per_study=Counter(q['study_id'] for q in qq);mass=Counter()
  for q in qq:
   obs=q['target']['observations'];status={o['status'] for o in obs}
   q['sampling_stratum']='present' if 'present' in status else 'uncertain' if 'uncertain' in status else 'unassessable' if 'unassessable' in status else 'absent' if any(not o['assessment_scope'] for o in obs) else 'scoped_absent'
   mass[q['sampling_stratum']]+=1/per_study[q['study_id']]
  for q in qq:q['sampling_weight']=(1/per_study[q['study_id']])/mass[q['sampling_stratum']]
  sampler[split]={'queries':len(qq),'stratum_counts':dict(Counter(q['sampling_stratum'] for q in qq)),'stratum_weight_mass':{s:sum(q['sampling_weight'] for q in qq if q['sampling_stratum']==s) for s in mass}}
 manifest={'schema_version':VERSION,'curation_version':curation['version'],'curation_sha256':sha(args.curation),'studies':dict(Counter(r['split'] for r in records)),'query_counts':{s:len(q) for s,q in queries.items()},'observation_state_counts':{s:{state:counts[s][state] for state in STATES} for s in counts},
  'original_images':sum(r['input_provenance']['original_image_count'] for r in records),'unannotated_slots_masked':sum(not v['loss_mask'] for r in records for v in r['supervision'].values()),
  'explicit_supporting_view_labels':sum('supporting_view' in a['target'] for r in records for a in r['annotations']),
  'four_state_coverage_complete':all(counts['train'][s]>0 for s in STATES),'unassessable_examples_missing':counts['train']['unassessable']==0,
  'source_text_verification_passed':True,'clinical_adjudication_completed':False,'training_started':False,'active_models_changed':False,'test_reports_used':False,
  'release_status':'source-text-verified pilot; incomplete four-state and view supervision; not clinically adjudicated',
  'source_manifest_sha256':{s:sha(f'data/report_training_v3/{s}.jsonl') for s in queries},
  'source_files_sha256':{f:sha(f) for f in ['chest_xpert/structured_findings.py','scripts/prepare_structured_findings.py']},
  'sampling':sampler,'limitations':['Source-derived text assertions are not independently verified image labels.','No unassessable labels are invented from source silence, cannot-exclude wording, or general limitations.','Only explicit source view statements supervise supporting_view; absent view attribution is masked.','Size-, certainty-, pattern- and temporal-qualified negatives keep their scope in the target.','The limited pilot does not establish model accuracy or readiness to train a four-state model.']}
 assert sha(active)==active_hash
 out.mkdir(parents=True,exist_ok=True)
 for split in queries:
  (out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records if r['split']==split))
  (out/f'{split}_queries.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in queries[split]))
 (out/'quarantine.json').write_text(json.dumps(curation['quarantine'],indent=2));(out/'ontology.json').write_text(json.dumps({'findings':CATALOG,'states':STATES,'Effusion_definition':'Pleural effusion, not pericardial fluid','missing_policy':'No label and no loss; not absent and not unassessable'},indent=2))
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':main()
