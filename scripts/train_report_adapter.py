"""CUDA QLoRA for real image + Grad-CAM + score -> clinical JSON reports.

Assistant-only loss, original Qwen image metadata preserved, study-held-out
validation, and separate candidate output. Never modifies the base model/CNN.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from chest_xpert.verification import _build_v7_report_prompt, build_report_messages
from chest_xpert.label_mapping import LabelMapping


def read_rows(path, limit=None):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def select_rows(rows, limit, sampling, seed):
    if sampling == 'first':
        return rows[:limit] if limit else rows
    # Sample different studies rather than spending the pilot on repeated views.
    unique = {}
    for row in rows:
        unique.setdefault(row['study_id'], row)
    if sampling == 'finding_balanced':
        from chest_xpert.disease_targets import disease_targets
        rng = random.Random(seed)
        candidates = list(unique.values())
        wanted = min(limit or len(candidates), len(candidates))
        flags = {row['study_id']: disease_targets(row)[0] for row in candidates}
        frequencies = [sum(flags[row['study_id']][i] for row in candidates) for i in range(14)]
        def weight(row):
            positives = flags[row['study_id']]
            return 1.0 + sum(1.0 / max(frequencies[i], 1) for i,v in enumerate(positives) if v) * len(candidates) / 14
        normals = [row for row in candidates if any(t.lower() == 'normal' for t in row.get('mesh_terms', []))]
        others = [row for row in candidates if row not in normals]
        # Weighted sampling without replacement; no validation/test content used.
        priority = sorted(others, key=lambda row: rng.random() ** (1.0 / weight(row)), reverse=True)
        rng.shuffle(normals)
        n = min(len(normals), wanted // 4)
        chosen = priority[:wanted-n] + normals[:n]
        if len(chosen) < wanted:
            chosen.extend(normals[n:n+wanted-len(chosen)])
        rng.shuffle(chosen)
        return chosen
    normal = [row for row in unique.values() if any(term.lower() == 'normal' for term in row.get('mesh_terms', []))]
    abnormal = [row for row in unique.values() if row not in normal]
    rng = random.Random(seed)
    rng.shuffle(normal); rng.shuffle(abnormal)
    wanted = min(limit or len(unique), len(unique))
    a = min(len(abnormal), wanted // 2)
    n = min(len(normal), wanted - a)
    selected = abnormal[:a] + normal[:n]
    selected.extend(abnormal[a:a + wanted - len(selected)])
    selected.extend(normal[n:n + wanted - len(selected)])
    rng.shuffle(selected)
    return selected


def enrich(rows, cache, device, cnn_evidence='legacy'):
    """Resolve full studies and cache actual scores/attributions for every view."""
    import hashlib
    from dataclasses import replace
    from chest_xpert.checkpoint_engine import CheckpointEngine, prepare_checkpoint_image
    from chest_xpert.explainability import GradCAM, generate_contact_sheet, save_contact_sheet
    from chest_xpert.study_images import resolve_study_images,index_study_images
    import numpy as np
    if cnn_evidence == 'selected_ensemble':
        from chest_xpert.selected_ensemble import SelectedEnsembleEngine
        engine = SelectedEnsembleEngine(device=device)
    else:
        engine=CheckpointEngine(device=device)
    cache = cache / engine.fingerprint[:16] / ('selected_ensemble_v1' if cnn_evidence == 'selected_ensemble' else 'full_study_v1')
    cache.mkdir(parents=True,exist_ok=True)
    image_index=index_study_images('data/iu_xray/images')
    for i,row in enumerate(rows):
        if row.get('dataset') == 'chexpert_plus':
            paths=[Path(path) for path in row['image_paths']]
        else:
            xml=Path('data/iu_xray/reports/ecgen-radiology')/f"{row['study_id'].removeprefix('CXR')}.xml"
            paths=resolve_study_images(xml,'data/iu_xray/images',image_index=image_index)
        if len(paths)>5:raise ValueError('Study exceeds the five-view report input limit')
        key=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in paths)).hexdigest()
        metadata=cache/f'{key}.json';sheet=cache/f'{key}.png'
        if metadata.exists() and sheet.exists():
            scores=json.loads(metadata.read_text())
        else:
            if cnn_evidence == 'selected_ensemble':
                scores,pairs=engine.predict_paths(paths)
                engine.contact_sheet(pairs,scores,sheet)
            else:
                tensors=[prepare_checkpoint_image(p,engine.resolution,engine.pixel_scale,engine.crop).unsqueeze(0).to(device) for p in paths]
                view_scores=[engine.predict(t).raw for t in tensors]
                scores={label:sum(v[label] for v in view_scores)/len(view_scores) for label in view_scores[0]}
                displays=[np.clip((t.detach().cpu().numpy()[0,0]/engine.pixel_scale+1)/2,0,1) for t in tensors]
                cam=GradCAM(engine.model,target_layer_name='encoder');results=[]
                try:
                    for view_index,(tensor,display) in enumerate(zip(tensors,displays)):
                        for label in sorted(scores,key=scores.get,reverse=True)[:6]:
                            result=cam.generate(tensor,label,engine.labels,display)
                            if len(paths)>1:result=replace(result,target_label=f'View {view_index+1}: {result.target_label}')
                            results.append(result)
                finally:cam.close()
                save_contact_sheet(generate_contact_sheet(results,displays[0],cols=6 if len(paths)>1 else 3,rows=len(paths) if len(paths)>1 else 2),str(sheet))
            metadata.write_text(json.dumps(scores))
        row['image_paths']=[str(p) for p in paths]
        row['cnn_scores']=scores;row['contact_sheet_path']=str(sheet.resolve())
        if (i+1)%32==0:print(f'EVIDENCE {i+1}/{len(rows)}',flush=True)
    del engine
    import gc
    gc.collect();torch.cuda.empty_cache()
    return rows


def build_messages(row):
    prompt=_build_v7_report_prompt([LabelMapping(name,True,name,score) for name,score in row['cnn_scores'].items()],
        'Project-trained CheXpert CNN, 14 findings. Scores are supporting signals, not calibrated clinical probabilities.',
        {'original_image_count':len(row.get('image_paths',[row['image_path']])),'clinical_indication':row.get('clinical_indication') or 'NOT PROVIDED',
         'comparison_study':row.get('comparison_study') or 'NOT PROVIDED',
         'technique':row.get('technique') or 'NOT PROVIDED',
         'report_sections':[section['heading'] for section in row['target']['sections']]},
        True,True)
    prompt+=' Never include publisher, author, article, DOI, license, URL, dataset title or citation metadata. If technique is not supplied, say NOT PROVIDED; do not invent positioning. Preserve negation and uncertainty. Missing redacted details must not be guessed.'
    images=[]
    for path in row.get('image_paths',[row['image_path']])+[row['contact_sheet_path']]:
        with Image.open(path) as image:images.append(image.convert('RGB'))
    messages=build_report_messages(prompt,images)
    return messages,images


class ReportCollator:
    def __init__(self,processor):self.processor=processor
    def __call__(self,rows):
        # Dynamic visual grids vary by image. Batch-size one preserves all fields
        # without stacking flattened image patch tensors incorrectly.
        if len(rows)!=1:raise ValueError('Use batch size 1 and gradient accumulation for dynamic image grids')
        row=rows[0]; messages,images=build_messages(row)
        prefix=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        complete=self.processor.apply_chat_template(messages+[{'role':'assistant','content':json.dumps(row['target'],ensure_ascii=False)}],tokenize=False,add_generation_prompt=False)
        inputs=self.processor(text=[complete],images=images,return_tensors='pt',padding=False)
        prompt_inputs=self.processor(text=[prefix],images=images,return_tensors='pt',padding=False)
        expected_images=len(images)
        if len(inputs['image_grid_thw'])!=expected_images or len(prompt_inputs['image_grid_thw'])!=expected_images:
            raise ValueError('Training processor dropped a required original view or contact sheet')
        length=prompt_inputs['input_ids'].shape[1]
        if not torch.equal(inputs['input_ids'][0,:length],prompt_inputs['input_ids'][0]):
            raise ValueError('Chat template prefix does not match the full training sequence')
        labels=inputs['input_ids'].clone();labels[:,:length]=-100
        labels[inputs['attention_mask']==0]=-100
        if not (labels!=-100).any():raise ValueError('No assistant tokens to train')
        inputs['labels']=labels
        return inputs


def evaluate_loss(model,rows,collator,device):
    model.eval();total=0;tokens=0
    with torch.no_grad():
        for row in rows:
            batch={key:value.to(device) for key,value in collator([row]).items()}
            count=int((batch['labels'][:,1:]!=-100).sum())
            loss=model(**batch).loss
            if not torch.isfinite(loss):raise RuntimeError('Non-finite validation loss')
            total+=float(loss)*count;tokens+=count
    return total/max(tokens,1)


def generate(model,processor,row,device):
    messages,images=build_messages(row)
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],images=images,return_tensors='pt').to(device)
    if len(inputs['image_grid_thw'])!=len(images):raise ValueError('Generation processor dropped a study view or contact sheet')
    model.eval();model.config.use_cache=True
    with torch.no_grad():out=model.generate(**inputs,max_new_tokens=768,do_sample=False,max_time=100)
    result=processor.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    model.config.use_cache=False
    return result


def main(args):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    device=args.device;torch.cuda.set_device(device)
    random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'adapter_config.json').exists():raise FileExistsError('Choose a fresh candidate output directory')
    import hashlib
    configuration = vars(args).copy()
    configuration['manifest_sha256'] = {name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('train',args.train),('validation',args.validation)]}
    (output/'run_config.json').write_text(json.dumps(configuration,indent=2))
    train=select_rows(read_rows(args.train),args.train_limit,args.sampling,args.seed)
    validation=select_rows(read_rows(args.validation),args.validation_limit,getattr(args, 'validation_sampling', 'balanced'),args.seed+1)
    (output/'selected_studies.json').write_text(json.dumps({'train':[row['study_id'] for row in train],'validation':[row['study_id'] for row in validation]},indent=2))
    (output/'progress.json').write_text(json.dumps({'phase':'preparing_evidence','train_images':len(train),'validation_images':len(validation)}))
    if {row['study_id'] for row in train}&{row['study_id'] for row in validation}:raise ValueError('Study leakage between splits')
    if {row['image_sha256'] for row in train}&{row['image_sha256'] for row in validation}:raise ValueError('Image leakage between splits')
    if not train or not validation:raise ValueError('Both training and held-out validation are required')
    train=enrich(train,Path(args.cache),device,args.cnn_evidence);validation=enrich(validation,Path(args.cache),device,args.cnn_evidence)
    if args.cnn_evidence == 'selected_ensemble':
        from chest_xpert.selected_ensemble import V6,V12
        paths=[V6,V12]
    else:
        from chest_xpert.model_selection import selected_cnn
        paths=[selected_cnn()]
    evidence={'cnn_sha256':hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in paths)).hexdigest(),
              'cnn_checkpoints':[str(p) for p in paths],
              'cnn_evidence':args.cnn_evidence,
              'required_inputs':['all_original_study_views','cnn_scores','gradcam_contact_sheet'],'rows':[]}
    for split,rows in [('train',train),('validation',validation)]:
        for row in rows:
            paths=row['image_paths']+[row['contact_sheet_path']]
            evidence['rows'].append({'split':split,'study_id':row['study_id'],'paths':paths,'sha256':[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths],'cnn_scores':row['cnn_scores']})
    (output/'input_evidence.json').write_text(json.dumps(evidence,indent=2))
    for split,rows in [('train',train),('validation',validation)]:
        (output/f'enriched_{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print('LOADING_QWEN',args.model,flush=True)
    processor=AutoProcessor.from_pretrained(args.model,min_pixels=128*28*28,max_pixels=args.max_pixels,use_fast=False,local_files_only=args.offline)
    config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type='nf4')
    options = {'torch_dtype':torch.float16, 'device_map':{'':device}, 'local_files_only':args.offline, 'attn_implementation':'sdpa'}
    if args.precision == '4bit':
        options['quantization_config'] = config
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,**options)
    if args.precision == '4bit':
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
    model=get_peft_model(model,LoraConfig(r=args.rank,lora_alpha=args.rank*2,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none',task_type='CAUSAL_LM'))
    model.config.use_cache=False
    model.print_trainable_parameters()
    inference_processor=AutoProcessor.from_pretrained(args.model,min_pixels=256*28*28,max_pixels=768*28*28,use_fast=False,local_files_only=args.offline)
    (output/'generation_settings.json').write_text(json.dumps({'min_pixels':256*28*28,'max_pixels':768*28*28,'max_new_tokens':768,'max_time':100,'do_sample':False,'base_weights_frozen':True},indent=2))
    collator=ReportCollator(processor)
    initial=evaluate_loss(model,validation,collator,device)
    comparisons=[{'study_id':row['study_id'],'image_path':row['image_path'],'reference':row['target']} for row in validation[:args.generate_examples]]
    print('BASE_VALIDATION_LOSS',initial,flush=True)
    loader=DataLoader(train,batch_size=1,shuffle=True,collate_fn=collator,num_workers=0)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.lr)
    scaler=torch.amp.GradScaler('cuda')
    model.train();optimizer.zero_grad(set_to_none=True)
    history=[];updates=0;micro=0;started=time.perf_counter()
    for epoch in range(args.epochs):
        for index,batch in enumerate(loader):
            batch={key:value.to(device) for key,value in batch.items()}
            with torch.autocast('cuda',dtype=torch.float16):
                loss=model(**batch).loss
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                group_size=min(args.grad_accum,len(loader)-(index//args.grad_accum)*args.grad_accum)
                scaled_loss=loss/group_size
            scaler.scale(scaled_loss).backward();micro+=1
            print('MICRO',micro,'loss',float(loss.detach()),flush=True)
            if (index+1)%args.grad_accum==0 or index==len(loader)-1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters,1)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                updates+=1
                history.append({'step':updates,'loss':float(loss.detach()),'elapsed_s':time.perf_counter()-started})
                print('TRAIN',json.dumps(history[-1]),flush=True)
                (output/'progress.json').write_text(json.dumps({'phase':'training','step':updates,'target_steps':args.steps,'history':history},indent=2))
                if updates % args.checkpoint_every == 0:
                    model.save_pretrained(output/f'step_{updates}')
                control_path = output/'control.json'
                stop_after = json.loads(control_path.read_text()).get('stop_after_step', args.steps) if control_path.exists() else args.steps
                if updates >= min(args.steps, stop_after):break
        if updates>=min(args.steps, stop_after):break
    (output/'progress.json').write_text(json.dumps({'phase':'evaluating','step':updates}))
    final=evaluate_loss(model,validation,collator,device)
    for comparison,row in zip(comparisons,validation):
        with model.disable_adapter():comparison['base_output']=generate(model,inference_processor,row,device)
        comparison['adapter_output']=generate(model,inference_processor,row,device)
        (output/'paired_generations.json').write_text(json.dumps(comparisons,indent=2))
        print('PAIRED_REPORT',row['study_id'],flush=True)
    model.save_pretrained(output);processor.save_pretrained(output)
    summary={'base_model':args.model,'train_images':len(train),'validation_images':len(validation),'steps':updates,'base_validation_loss':initial,'adapter_validation_loss':final,'validation_loss_improved':final<initial,'comparisons':comparisons,'history':history,'auto_activated':False,'limitations':'Small pilot; lower token loss is not proof of clinical accuracy. Review generated findings and abnormal held-out cases before activation.'}
    (output/'evaluation.json').write_text(json.dumps(summary,indent=2))
    (output/'progress.json').write_text(json.dumps({'phase':'complete','step':updates,'validation_loss_improved':final<initial},indent=2))
    print('DONE',json.dumps({key:value for key,value in summary.items() if key not in ['history','comparisons']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='Qwen/Qwen2.5-VL-7B-Instruct');p.add_argument('--train',default='data/report_training_v3/train.jsonl');p.add_argument('--validation',default='data/report_training_v3/validation.jsonl')
    p.add_argument('--output',default='checkpoints/report_lora_candidate');p.add_argument('--cache',default='data/report_training/evidence');p.add_argument('--device',default='cuda:0');p.add_argument('--offline',action='store_true')
    p.add_argument('--cnn-evidence', choices=['legacy','selected_ensemble'], default='legacy')
    p.add_argument('--train-limit',type=int,default=128);p.add_argument('--validation-limit',type=int,default=8);p.add_argument('--steps',type=int,default=16);p.add_argument('--epochs',type=int,default=1);p.add_argument('--grad-accum',type=int,default=8);p.add_argument('--rank',type=int,default=8);p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--max-pixels',type=int,default=256*28*28);p.add_argument('--generate-examples',type=int,default=1);p.add_argument('--seed',type=int,default=42);p.add_argument('--precision',choices=['4bit','fp16'],default='4bit');p.add_argument('--sampling',choices=['first','balanced','finding_balanced'],default='balanced');p.add_argument('--checkpoint-every',type=int,default=8);p.add_argument('--validation-sampling',choices=['first','balanced'],default='balanced')
    main(p.parse_args())
