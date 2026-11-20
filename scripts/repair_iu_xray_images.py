"""Restore missing/corrupt official IU-Xray PNGs without replacing valid files."""
import hashlib,io,json,re,tarfile,time
from pathlib import Path
import requests
from PIL import Image
root=Path('data/iu_xray');cache=Path.home()/'.cache/chestxpert';cache.mkdir(parents=True,exist_ok=True)
archive=cache/'NLMCXR_png.tgz';partial=archive.with_suffix('.tgz.part')
if not archive.exists():
 print('Downloading official image collection',flush=True)
 with requests.get('https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz',stream=True,timeout=(30,120)) as response:
  response.raise_for_status();expected=int(response.headers.get('Content-Length',0));total=0;last=0
  with partial.open('wb') as handle:
   for chunk in response.iter_content(1<<20):
    if chunk:handle.write(chunk);total+=len(chunk)
    if total-last>=128*(1<<20):print('DOWNLOADED_MB',round(total/1e6),flush=True);last=total
  if expected and total!=expected:raise RuntimeError('Incomplete official archive download')
 partial.replace(archive)
images=root/'images';images.mkdir(exist_ok=True)
existing={p.stem:p for p in images.rglob('*.png')}
restored=[];valid_kept=0;members=0
with tarfile.open(archive,'r|gz') as collection:
 for member in collection:
  name=Path(member.name).name
  if not member.isfile() or not re.match(r'^CXR\d.*\.png$',name):continue
  members+=1;target=existing.get(Path(name).stem,images/name)
  try:
   with Image.open(target) as image:image.verify()
   valid_kept+=1;continue
  except (OSError,ValueError):pass
  handle=collection.extractfile(member);data=handle.read()
  with Image.open(io.BytesIO(data)) as image:image.verify()
  temporary=target.with_suffix('.repairtmp');temporary.write_bytes(data);temporary.replace(target);restored.append(name)
print('RESTORED',len(restored),'KEPT',valid_kept,'ARCHIVE_IMAGES',members,flush=True)
import xml.etree.ElementTree as ET
available={p.stem:str(p.resolve()) for p in images.rglob('*.png')};studies=[]
for xml in sorted((root/'reports').rglob('*.xml')):
 tree=ET.parse(xml).getroot();uid=tree.find('uId');study=uid.get('id') if uid is not None else 'CXR'+xml.stem
 expected=[node.get('id') for node in tree.findall('parentImage') if node.get('id')]
 studies.append({'study_id':study,'expected_image_ids':expected,'available_images':[available[key] for key in expected if key in available],'missing_image_ids':[key for key in expected if key not in available]})
report={'official_archive':'https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz','restored_images':len(restored),'restored_image_names':restored,'valid_existing_kept':valid_kept,'archive_images':members,'complete_studies':sum(bool(s['expected_image_ids']) and not s['missing_image_ids'] for s in studies),'reports_without_image_ids':sum(not s['expected_image_ids'] for s in studies),'incomplete_studies':sum(bool(s['missing_image_ids']) for s in studies),'studies':studies,'existing_split_manifests_modified':False}
(root/'image_completeness_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['studies','restored_image_names']}),flush=True)
