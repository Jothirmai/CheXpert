"""Audited native-task medical VLM experiment; never supplies reference findings."""
import hashlib,json,sys,time,types,warnings
from pathlib import Path
warnings.filterwarnings('ignore',message='.*copying from a non-meta parameter.*')
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
root=Path(__file__).resolve().parents[1];out=root/'checkpoints/chexagent_full_study';out.mkdir(exist_ok=True)
model_path=str(root/'checkpoints/chexagent_runtime')
tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
model,info=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,trust_remote_code=True,low_cpu_mem_usage=True,output_loading_info=True)
(out/'loading_info.json').write_text(json.dumps(info,indent=2));assert not info['missing_keys'] and not info['mismatched_keys'],info
model=model.to(device='cuda',dtype=torch.float16).eval()
review=json.loads((root/'reports/multiview_correction/generated_report.json').read_text());provenance=review['model_provenance'];audit=provenance['vision_inputs'];paths=audit['image_paths'];scores=provenance['cnn_scores']
assert len(paths)==4 and len(scores)==14
for path,digest in zip(paths,audit['image_sha256']):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
visual=model.model.visual;encode=visual.encode;forward=visual.forward;encoded=[];batches=[]
def audited_encode(self,paths,training):encoded.extend(paths);return encode(paths,training)
def audited_forward(self,x):batches.append(int(x.shape[0]));return forward(x)
visual.encode=types.MethodType(audited_encode,visual);visual.forward=types.MethodType(audited_forward,visual)
context='The first three images are original radiographs from one study. The final image is a Grad-CAM contact sheet, not another radiograph. Supporting CNN scores (uncalibrated; use originals to confirm findings): '+json.dumps(scores)+'. Do not mention heatmaps, scores or models in the clinical report. '
prompts=[context+'Given the indication: "Dyspnea", write a structured findings section for the CXR.',context+'Write the findings section for this chest X-ray study.']
results=[]
for prompt in prompts:
 query=tokenizer.from_list_format([*[{'image':p} for p in paths],{'text':prompt}]);conv=[{'from':'system','value':'You are a helpful assistant.'},{'from':'human','value':query}]
 ids=tokenizer.apply_chat_template(conv,add_generation_prompt=True,return_tensors='pt').to('cuda');encoded.clear();batches.clear();start=time.perf_counter()
 with torch.inference_mode():generated=model.generate(ids,do_sample=False,num_beams=1,temperature=1.,top_p=1.,use_cache=True,max_new_tokens=384,max_time=120)
 response=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True)
 assert encoded==paths and batches[0]==4,(encoded,batches)
 row={'prompt':prompt,'response':response,'elapsed_seconds':time.perf_counter()-start,'encoded_image_paths':encoded.copy(),'actual_vision_batch_sizes':batches.copy(),'input_tokens':int(ids.shape[1]),'reference_findings_or_impression_supplied':False}
 results.append(row);(out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
