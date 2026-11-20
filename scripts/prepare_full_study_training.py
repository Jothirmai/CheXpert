"""Freeze full-study cohorts and audit restored views for cross-split duplicates."""
import argparse,hashlib,json,random,re,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chest_xpert.study_images import resolve_study_images,index_study_images
from scripts.train_report_adapter import select_rows
from chest_xpert.report_scoring_v2 import extract,VERSION
parser=argparse.ArgumentParser();parser.add_argument('--output',default='checkpoints/report_lora_full_study_v2');args=parser.parse_args()
OUT=Path(args.output);OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'cohort_audit.json').exists():raise SystemExit('Frozen cohorts already exist; choose a new experiment for changes')
index=index_study_images('data/iu_xray/images');splits={};owners=defaultdict(set);hashes={}
for split in ('train','validation','test'):
 unique={}
 for row in map(json.loads,Path(f'data/report_training_v3/{split}.jsonl').read_text().splitlines()):unique.setdefault(row['study_id'],row)
 splits[split]=list(unique.values())
 for row in unique.values():
  paths=resolve_study_images(Path('data/iu_xray/reports/ecgen-radiology')/f"{row['study_id'].removeprefix('CXR')}.xml",'data/iu_xray/images',index)
  row['image_paths']=[str(p) for p in paths]
  for p in paths:
   digest=hashlib.sha256(p.read_bytes()).hexdigest();hashes[str(p)]=digest;owners[digest].add((split,row['study_id']))
bad={study for groups in owners.values() if len({s for s,_ in groups})>1 for _,study in groups}
clean={s:[r for r in rows if r['study_id'] not in bad] for s,rows in splits.items()}
train=select_rows(clean['train'],512,'finding_balanced',42)
# Predeclared validation coverage; reference content is used for cohort/scoring only.
from chest_xpert.validation_cohort import select_validation_cohort,DEFAULT_QUOTAS
selected,coverage,available=select_validation_cohort(clean['validation'])
assert len(selected)==24
for name,rows in [('train',train),('validation',selected)]:
 (OUT/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
# Reserve the original test split; do not inspect its report facts for selection.
(OUT/'reserved_test_ids.json').write_text(json.dumps([r['study_id'] for r in clean['test']],indent=2))
audit={'scorer_version':VERSION,'validation_quotas':DEFAULT_QUOTAS,'available_validation_strata':available,'cohort_revision_reason':'Only five explicit quality/positioning studies are available; reserve these before overlapping strata rather than misclassifying spinal kyphosis.','technical_category_excludes_spinal_kyphosis':True,'train_studies':len(train),'validation_studies':len(selected),'generated_validation_studies':[r['study_id'] for r in selected[:12]],'validation_categories':coverage,'train_original_views':sum(len(r['image_paths']) for r in train),'validation_original_views':sum(len(r['image_paths']) for r in selected),'excluded_cross_split_duplicate_studies':sorted(bad),'original_manifests_modified':False,'test_content_used_for_training_or_selection':False,'reserved_test_studies':len(clean['test']),'input_hashes':hashes,'original_manifest_sha256':{s:hashlib.sha256(Path(f'data/report_training_v3/{s}.jsonl').read_bytes()).hexdigest() for s in splits}}
(OUT/'cohort_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='input_hashes'},indent=2))
