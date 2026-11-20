"""Diagnostic 2x2x2 input swaps; outputs never enter clinical report storage."""
import hashlib,itertools,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from scripts.train_report_adapter import build_messages
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/grounding_diagnostic';OUT.mkdir(exist_ok=True)
if (OUT/'results.json').exists():raise RuntimeError('Preserve existing diagnostic; use a new output folder')
exp=ROOT/'checkpoints/report_lora_full_study';rows={r['study_id']:r for r in map(json.loads,(exp/'enriched_validation.jsonl').read_text().splitlines())}
# Fixed same-view-count pair: one reference-normal and one reference-abnormal study.
pair={'normal':rows['CXR3223'],'abnormal':rows['CXR3286']}
assert all(len(r['image_paths'])==2 and len(r['cnn_scores'])==14 for r in pair.values())
config={'design':'Full 2x2x2 factorial: original views, contact sheet, CNN scores independently sourced from a normal/abnormal pair; both base and failed adapter','pair':{k:r['study_id'] for k,r in pair.items()},'metadata':'identical NOT PROVIDED in every condition','clinical_use':False,'model_selection_use':False,'reference_answers_in_prompt':False,'generation':{'min_pixels':256*28*28,'max_pixels':768*28*28,'max_new_tokens':768,'max_time':100,'do_sample':False}}
(OUT/'configuration.json').write_text(json.dumps(config,indent=2))
torch.set_num_threads(4);model_id='Qwen/Qwen2.5-VL-7B-Instruct'
processor=AutoProcessor.from_pretrained(model_id,min_pixels=256*28*28,max_pixels=768*28*28,use_fast=False,local_files_only=True)
base=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id,torch_dtype=torch.float16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True)
model=PeftModel.from_pretrained(base,str(exp)).eval();model.config.use_cache=True
results=[]
def generate(row):
 messages,images=build_messages(row);prompt=messages[1]['content'][-1]['text']
 for source in pair.values():
  for s in source['target']['sections']:
   if s['heading'] in ('Findings','Impression') and len(s['text'])>25:assert s['text'] not in prompt
 text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 inputs=processor(text=[text],images=images,return_tensors='pt').to('cuda:0');assert len(inputs['image_grid_thw'])==3
 audit={'original_paths':row['image_paths'],'contact_sheet':row['contact_sheet_path'],'cnn_scores':row['cnn_scores'],'image_grid_thw':inputs['image_grid_thw'].tolist(),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'input_tokens':int(inputs['input_ids'].shape[1])}
 paths=row['image_paths']+[row['contact_sheet_path']];audit['image_sha256']=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]
 with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=768,max_time=100,do_sample=False)
 result=processor.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
 for im in images:im.close()
 return result,audit
for mode in ('base','adapter'):
 for original,cam,scores in itertools.product(('normal','abnormal'),repeat=3):
  row={'image_path':pair[original]['image_paths'][0],'image_paths':pair[original]['image_paths'],'contact_sheet_path':pair[cam]['contact_sheet_path'],'cnn_scores':pair[scores]['cnn_scores'],'clinical_indication':None,'comparison_study':None,'technique':None,'target':{'sections':[{'heading':h,'text':''} for h in ['Clinical indication','Comparison','Technique','Findings','Impression']]}}
  started=time.perf_counter()
  if mode=='base':
   with model.disable_adapter():raw,audit=generate(row)
  else:raw,audit=generate(row)
  record={'mode':mode,'original_source':original,'cam_source':cam,'score_source':scores,'raw_output':raw,'input_audit':audit,'seconds':time.perf_counter()-started}
  results.append(record);(OUT/'partial_results.json').write_text(json.dumps(results,indent=2));print('CONDITION',len(results),'/16',mode,original,cam,scores,round(record['seconds'],1),flush=True)
(OUT/'results.json').write_text(json.dumps({'configuration':config,'conditions':results},indent=2));print('COMPLETE',flush=True)
