"""Medical VLM trial with audited CNN scores and both actual image inputs.

Run with the isolated CheXagent environment. Dataset agreement is illustrative;
OpenI is present in the upstream project's training-data processors.
"""
import argparse,hashlib,json,sys,time,types
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--model',default='checkpoints/chexagent_runtime');args=p.parse_args()
out=Path('checkpoints/chexagent_benchmark');out.mkdir(exist_ok=True)
model_path=str(Path(args.model).resolve())
print('Loading medical model',model_path,flush=True)
tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,trust_remote_code=True,low_cpu_mem_usage=True).to(device='cuda',dtype=torch.float16).eval()
encoded=[]
visual=model.model.visual;original_encode=visual.encode
original_forward=visual.forward
vision_batches=[]
def audited_encode(self,image_paths,training):
 encoded.extend(image_paths)
 return original_encode(image_paths,training)
def audited_forward(self,x):
 vision_batches.append(int(x.shape[0]));return original_forward(x)
visual.encode=types.MethodType(audited_encode,visual);visual.forward=types.MethodType(audited_forward,visual)
source=json.loads(Path('checkpoints/report_score_correction/results.json').read_text())
records=[]
for case in source['adjusted']:
 if case.get('review'):
  records.append({'study_id':case['study_id'],'reference':case['reference'],'review':case['review']})
demo=json.loads(Path('checkpoints/report_score_correction/illustrative_report.json').read_text())
reference=json.loads(Path('reports/required_vlm_evidence/dataset_reference.json').read_text())['target']
records.append({'study_id':'CXR3076','reference':reference,'review':demo,'illustrative_only':True})
results=[]
for record in records:
 provenance=record['review']['model_provenance'];paths=provenance['vision_inputs']['image_paths'];scores=provenance['cnn_scores']
 assert len(paths)==2 and len(scores)==14
 for path,digest in zip(paths,provenance['vision_inputs']['image_sha256']):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
 prompt=('Write a concise chest X-ray report for radiologist review. Image 1 is the original radiograph. Image 2 is a Grad-CAM contact sheet, not another clinical projection. CNN scores are uncalibrated supporting signals, not diagnoses or probabilities. Base observations on the original radiograph. Preserve limitations and uncertainty. Output only two sections with these headings: FINDINGS: and IMPRESSION:. Describe visible anatomy and findings; do not mention scores, models or heatmaps in the report. Supporting CNN scores: '+json.dumps(scores))
 query=tokenizer.from_list_format([{'image':paths[0]},{'image':paths[1]},{'text':prompt}])
 conv=[{'from':'system','value':'You are a helpful assistant.'},{'from':'human','value':query}]
 ids=tokenizer.apply_chat_template(conv,add_generation_prompt=True,return_tensors='pt').to('cuda')
 encoded.clear();vision_batches.clear();start=time.perf_counter()
 with torch.inference_mode():
  result=model.generate(ids,do_sample=False,num_beams=1,temperature=1.,top_p=1.,use_cache=True,max_new_tokens=384,max_time=150)
 text=tokenizer.decode(result[0,ids.shape[1]:],skip_special_tokens=True)
 assert encoded==paths,{'expected_paths':paths,'actual_paths':encoded}
 assert vision_batches and vision_batches[0]==2,vision_batches
 row={**record,'raw_output':text,'elapsed_seconds':time.perf_counter()-start,'input_audit':{'image_paths':paths,'image_sha256':provenance['vision_inputs']['image_sha256'],'cnn_scores':scores,'actual_vision_batch_sizes':vision_batches.copy(),'input_tokens':int(ids.shape[1]),'reference_findings_or_impression_supplied':False}}
 results.append(row);(out/'results.json').write_text(json.dumps({'model':model_path,'potential_pretraining_overlap':True,'cases':results},indent=2))
 print(json.dumps({'study':record['study_id'],'seconds':row['elapsed_seconds'],'text':text}),flush=True)
print('COMPLETE',flush=True)
