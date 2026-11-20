"""Freeze a patient-level development holdout; keep official valid untouched."""
import argparse,hashlib,json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--root',default='data/chexpert_plus_cnn_v1');args=parser.parse_args()
root=Path(args.root);train=root/'train.jsonl';assert train.exists();out=root/'development_split.json';
if out.exists():raise RuntimeError('Development split already frozen')
training=[];development=[]
for line in train.open():
 row=json.loads(line);bucket=int(hashlib.sha256(('chest-xpert-cnn-dev-v1:'+row['patient_id']).encode()).hexdigest()[:8],16)%10
 (development if bucket==0 else training).append(row)
assert {r['patient_id'] for r in training}.isdisjoint({r['patient_id'] for r in development})
for name,rows in [('fit',training),('development',development)]:
 p=root/f'{name}.jsonl';p.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
counts={name:{'images':len(rows),'patients':len({r['patient_id'] for r in rows}),'known_positive':[sum(r['labels'][j]==1 for r in rows) for j in range(14)],'known_negative':[sum(r['labels'][j]==0 for r in rows) for j in range(14)]} for name,rows in [('fit',training),('development',development)]}
assert min(counts['development']['known_positive'])>0
result={'rule':'SHA256 patient hash modulo 10; bucket 0 development; others fit','seed_string':'chest-xpert-cnn-dev-v1','original_train_sha256':hashlib.sha256(train.read_bytes()).hexdigest(),'official_valid_untouched':True,'test_used':False,'counts':counts,'sha256':{n:hashlib.sha256((root/f'{n}.jsonl').read_bytes()).hexdigest() for n in ['fit','development']}}
out.write_text(json.dumps(result,indent=2));print(json.dumps({k:{kk:v for kk,v in x.items() if kk in ('images','patients')} for k,x in counts.items()},indent=2))
