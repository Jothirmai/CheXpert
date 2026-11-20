"""Bounded three-state structured LoRA trial with status-weighted supervision."""
import argparse,hashlib,json,math,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig,get_peft_model
from chest_xpert.structured_findings import build_structured_messages
from chest_xpert.structured_pipeline import SUPPORTED_STATES,score_prediction,summarize_predictions,parse_prediction,render_sections
from backend.services import review_store
from weasyprint import HTML

def read(p):return list(map(json.loads,Path(p).read_text().splitlines()))
def dump(p,data):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)

class Collator:
 def __init__(self,processor,records):self.processor=processor;self.records=records
 def messages(self,q):
  messages,images=build_structured_messages(self.records[q['study_id']],q)
  messages[1]['content'][-1]['text']+=' This bounded experiment supports only present, absent and uncertain. Do not output unassessable. Use exactly the requested finding name and an observations list. Each observation requires status and assessment_scope. Optional fields are laterality, region, degree and supporting_view; omit unknown optional fields. Do not copy explanatory placeholders.'
  return messages,images
 def batch(self,q):
  messages,images=self.messages(q);prefix=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  full=self.processor.apply_chat_template(messages+[{'role':'assistant','content':json.dumps(q['target'],separators=(',',':'))}],tokenize=False,add_generation_prompt=False)
  inputs=self.processor(text=[full],images=images,return_tensors='pt');pre=self.processor(text=[prefix],images=images,return_tensors='pt');n=pre['input_ids'].shape[1]
  assert torch.equal(inputs['input_ids'][0,:n],pre['input_ids'][0])
  assert len(inputs['image_grid_thw'])==len(images)
  labels=inputs['input_ids'].clone();labels[:,:n]=-100;inputs['labels']=labels
  weights=torch.zeros_like(labels,dtype=torch.float32);weights[:,n:]=.1
  ids=inputs['input_ids'][0].tolist();count=0
  for state in SUPPORTED_STATES:
   tokens=self.processor.tokenizer.encode(state,add_special_tokens=False)
   for i in range(n,len(ids)-len(tokens)+1):
    if ids[i:i+len(tokens)]==tokens:
     weights[0,i:i+len(tokens)]=10/len(tokens);count+=1
  assert count==len(q['target']['observations']),(count,q['query_id'])
  for image in images:image.close()
  return inputs,weights

def weighted_loss(model,batch,weights):
 labels=batch.pop('labels');output=model(**batch).logits[:,:-1,:];target=labels[:,1:];w=weights[:,1:]
 per=F.cross_entropy(output.float().reshape(-1,output.shape[-1]),target.reshape(-1),ignore_index=-100,reduction='none').reshape_as(target)
 return (per*w).sum()/w.sum().clamp_min(1)

def generate(model,collator,q):
 messages,images=collator.messages(q);prefix='{"finding":'+json.dumps(q['finding'])+',"observations":[';text=collator.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)+prefix
 inputs=collator.processor(text=[text],images=images,return_tensors='pt').to('cuda:0');assert len(inputs['image_grid_thw'])==len(images)
 paths=collator.records[q['study_id']]['inputs']['original_paths']+[collator.records[q['study_id']]['inputs']['contact_sheet_path']]
 audit={'image_paths':paths,'image_sha256':[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths],'image_grid_thw':inputs['image_grid_thw'].tolist(),'cnn_scores':collator.records[q['study_id']]['inputs']['cnn_scores'],'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'fixed_schema_prefix':prefix}
 model.eval();model.config.use_cache=True
 with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=256,max_time=50,do_sample=False)
 raw=prefix+collator.processor.decode(tokens[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
 for im in images:im.close()
 model.config.use_cache=False
 return {'query_id':q['query_id'],'study_id':q['study_id'],'finding':q['finding'],'target':q['target'],'raw_output':raw,'score':score_prediction(raw,q),'input_audit':audit}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='checkpoints/structured_lora_pilot');args=p.parse_args();out=Path(args.output);out.mkdir(exist_ok=True)
 if (out/'protocol.json').exists():raise RuntimeError('Use a fresh experiment; do not overwrite a frozen trial')
 data=Path('data/structured_findings_v1_1');records={r['study_id']:r for r in read(data/'train.jsonl')+read(data/'validation.jsonl')}
 # Remove history shortcuts for this controlled finding-grounding experiment.
 for record in records.values():record['inputs']['clinical_metadata']={}
 train=read(data/'train_queries.jsonl');validation=read(data/'validation_queries.jsonl');lookup={(q['study_id'],q['finding']):q for q in validation}
 probe_keys=[('CXR3286','Cardiomegaly'),('CXR1226','Consolidation'),('CXR3676','Fibrosis'),('CXR1050','Low lung volumes'),('CXR3632','Effusion'),('CXR3429','Hardware'),('CXR3223','Cardiomegaly'),('CXR1813','Pneumothorax'),('CXR1226','Effusion'),('CXR128','Fibrosis'),('CXR784','Effusion'),('CXR784','Atelectasis')]
 probes=[lookup[k] for k in probe_keys]
 sources=['scripts/train_structured_pilot.py','chest_xpert/structured_pipeline.py','chest_xpert/structured_findings.py','chest_xpert/structured_schema.py']
 protocol={'dataset':str(data),'dataset_manifest_sha256':hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),'supported_states':list(SUPPORTED_STATES),'unassessable_masked':True,'clinical_adjudication':False,'max_updates':16,'grad_accum':4,'checkpoints':[4,8,12,16],'probe_queries':[q['query_id'] for q in probes],'early_stop':'Two consecutive checkpoint probes with all four positive queries valid but absent-only. Invalid outputs fail selection.','selection':'All 12 valid, greater positive recovery, no more invented positives, no worse total class/scope correctness; then compare generated demonstration reports. No automatic clinical deployment from this pilot.','source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sources}}
 dump(out/'protocol.json',protocol);dump(out/'progress.json',{'phase':'loading','step':0,'target_steps':16})
 torch.set_num_threads(4);torch.manual_seed(47);rng=random.Random(47)
 processor=AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct',min_pixels=128*28*28,max_pixels=256*28*28,use_fast=False,local_files_only=True)
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct',torch_dtype=torch.float16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True)
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
 model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none',task_type='CAUSAL_LM'));model.config.use_cache=False
 collator=Collator(processor,records);base=[]
 dump(out/'progress.json',{'phase':'base_validation','step':0,'target_steps':16})
 for i,q in enumerate(probes):
  base.append(generate(model,collator,q));dump(out/'base_predictions.json',base);print('BASE',i+1,'/12',flush=True)
 parameters=[v for v in model.parameters() if v.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=5e-5);scaler=torch.amp.GradScaler('cuda');history=[];checkpoints=[];collapse_count=0;started=time.perf_counter()
 weights=[q['sampling_weight'] for q in train]
 step=0
 for attempt in range(1,25):
  model.train();optimizer.zero_grad(set_to_none=True);losses=[]
  for micro,q in enumerate(rng.choices(train,weights=weights,k=4),1):
   batch,w=collator.batch(q);batch={k:v.to('cuda:0') for k,v in batch.items()};w=w.to('cuda:0')
   with torch.autocast('cuda',dtype=torch.float16):loss=weighted_loss(model,batch,w)
   if not torch.isfinite(loss):raise RuntimeError('Non-finite structured loss')
   scaler.scale(loss/4).backward();losses.append(float(loss.detach()));print('MICRO',step+1,micro,losses[-1],flush=True)
  scaler.unscale_(optimizer);norm=float(torch.nn.utils.clip_grad_norm_(parameters,1));before_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
  if scaler.get_scale()<before_scale:
   print('SKIPPED_NONFINITE_GRADIENT',attempt,flush=True);continue
  step+=1
  history.append({'step':step,'loss':sum(losses)/4,'gradient_norm':norm,'elapsed_s':time.perf_counter()-started});dump(out/'progress.json',{'phase':'training','step':step,'target_steps':16,'history':history});print('UPDATE',json.dumps(history[-1]),flush=True)
  if step%4==0:
   model.save_pretrained(out/f'step_{step}');preds=[generate(model,collator,q) for q in probes[:4]];summary=summarize_predictions(preds)
   collapse_count=collapse_count+1 if summary['absent_only'] and summary['valid']==4 else 0
   checkpoints.append({'step':step,'metrics':summary,'predictions':preds});dump(out/'intermediate_evaluation.json',checkpoints)
   print('CHECKPOINT',step,json.dumps(summary),flush=True)
   if collapse_count>=2:print('EARLY_STOP_ABSENT_COLLAPSE',flush=True);break
  if step>=16:break
 dump(out/'progress.json',{'phase':'final_validation','step':step,'target_steps':16,'history':history})
 candidate=[]
 for i,q in enumerate(probes):
  candidate.append(generate(model,collator,q));dump(out/'candidate_predictions.json',candidate);print('FINAL',i+1,'/12',flush=True)
 model.save_pretrained(out);processor.save_pretrained(out)
 b=summarize_predictions(base);a=summarize_predictions(candidate)
 eligible=bool(a['valid']==12 and a['positive_recovered']>b['positive_recovered'] and a['invented_positive']<=b['invented_positive'] and a['class_correct']>=b['class_correct'] and a['scope_correct']>=b['scope_correct'] and not a['absent_only'])
 result={'base':b,'candidate':a,'updates':step,'optimizer_attempts':attempt,'intermediate_checkpoints':[{k:v for k,v in c.items() if k!='predictions'} for c in checkpoints],'eligible_for_further_validation':eligible,'automatically_activated':False,'supported_states':list(SUPPORTED_STATES),'limitations':'24 source-text-reviewed training studies; 12 fixed finding queries from existing validation studies. Not clinical validation; unassessable is unsupported and masked.'}
 dump(out/'evaluation.json',result)
 # Finish the observation -> prose path on three fixed studies, without references as answers.
 reports=[]
 demo_findings=['Cardiomegaly','Consolidation','Effusion','Pneumothorax','Atelectasis','Low lung volumes']
 for study in ['CXR3286','CXR1226','CXR3223']:
  preds=[];raws=[]
  for finding in demo_findings:
   q={'query_id':study+':'+finding,'study_id':study,'finding':finding,'target':{'finding':finding,'observations':[]}}
   cached=next((c for c in candidate if c['study_id']==study and c['finding']==finding),None)
   record=cached or generate(model,collator,q);raws.append(record)
   if record['score'].get('prediction'):preds.append(record['score']['prediction'])
  try:
   if len(preds)!=len(demo_findings):raise ValueError('Incomplete structured predictions; refusing a normal-report fallback')
   metadata=next(r for r in read(data/'validation.jsonl') if r['study_id']==study)['inputs']['clinical_metadata'];sections=render_sections(preds,metadata)
   document=review_store.render({'patient_id':study,'sections':sections,'signature':None});(out/f'{study}_report.html').write_text(document);HTML(string=document).write_pdf(out/f'{study}_report.pdf')
   reports.append({'study_id':study,'sections':sections,'raw_predictions':raws,'valid_finding_responses':len(preds),'reference':records[study]['reference']})
  except ValueError as exc:reports.append({'study_id':study,'error':str(exc),'raw_predictions':raws})
  dump(out/'generated_reports.json',reports)
 dump(out/'progress.json',{'phase':'complete','step':step,'target_steps':16,'eligible_for_further_validation':eligible});print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
