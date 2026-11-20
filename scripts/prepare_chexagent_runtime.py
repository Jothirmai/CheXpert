"""Create a pinned local runtime, preserving source and weights in the HF cache."""
import hashlib,json,shutil
from pathlib import Path
from huggingface_hub import snapshot_download
repo='StanfordAIMI/CheXagent-2-3b';revision='8f19b53a2eceda4c33b0acec6c81fbc293ad80d0'
source=Path(snapshot_download(repo,revision=revision,local_files_only=True))
out=Path('checkpoints/chexagent_runtime');out.mkdir(exist_ok=True)
for path in source.iterdir():
 if not path.is_file():continue
 target=out/path.name
 if path.suffix=='.safetensors':
  if not target.exists():target.symlink_to(path.resolve())
 else:shutil.copyfile(path,target)
p=out/'modeling_visual.py';original=p.read_text();patched=original
removed=['import albumentations as A\n','import cv2\n','import pyarrow as pa\n','from albumentations.pytorch import ToTensorV2\n']
for line in removed:
 assert line in patched;patched=patched.replace(line,'')
p.write_text(patched)
(out/'compatibility_patch.json').write_text(json.dumps({'repo':repo,'revision':revision,'removed_unused_imports':removed,'reason':'These names have no uses in the source; actual image processing uses PIL/torchvision and is unchanged.','source_sha256':hashlib.sha256(original.encode()).hexdigest(),'runtime_sha256':hashlib.sha256(patched.encode()).hexdigest(),'transformers':'4.40.0'},indent=2))
print(out.resolve())
