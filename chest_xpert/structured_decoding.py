"""Closed-vocabulary status decoding; schema scaffolding contains no diagnosis."""
import hashlib,json
from pathlib import Path
from chest_xpert.structured_findings import build_structured_messages
from chest_xpert.structured_pipeline import SUPPORTED_STATES

def allowed_status_tokens(label_tokens,generated,eos_token_id):
    allowed=set()
    for tokens in label_tokens.values():
        if tokens[:len(generated)]==generated:
            allowed.add(tokens[len(generated)] if len(generated)<len(tokens) else eos_token_id)
    if not allowed:raise ValueError('Generated tokens left the closed status vocabulary')
    return sorted(allowed)

def decode_status(model,processor,record,finding,device='cuda:0'):
    import torch
    query={'finding':finding};messages,images=build_structured_messages(record,query)
    messages[1]['content'][-1]['text']+=' Classify this finding using only present, absent or uncertain. This experiment does not support unassessable. Assess the whole finding, not a size-qualified negative. Optional attributes are not predicted by this status-only decoder.'
    prefix='{"finding":'+json.dumps(finding)+',"observations":[{"status":"'
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)+prefix
    inputs=processor(text=[text],images=images,return_tensors='pt').to(device)
    assert len(inputs['image_grid_thw'])==len(images)
    length=inputs['input_ids'].shape[1];tokens={s:processor.tokenizer.encode(s,add_special_tokens=False) for s in SUPPORTED_STATES}
    eos=processor.tokenizer.eos_token_id
    def allowed(batch_id,ids):return allowed_status_tokens(tokens,ids[length:].tolist(),eos)
    model.eval();model.config.use_cache=True
    with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=max(map(len,tokens.values()))+1,do_sample=False,prefix_allowed_tokens_fn=allowed,eos_token_id=eos,pad_token_id=eos)
    generated=out[0,length:].tolist();generated=[token for token in generated if token!=eos]
    status=next((s for s,v in tokens.items() if v==generated),None)
    if status is None:raise RuntimeError('No supported status was decoded')
    paths=record['inputs']['original_paths']+[record['inputs']['contact_sheet_path']]
    result={'finding':finding,'observations':[{'status':status,'assessment_scope':{}}]}
    audit={'image_paths':paths,'image_sha256':[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths],'image_grid_thw':inputs['image_grid_thw'].tolist(),'cnn_scores':record['inputs']['cnn_scores'],'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'fixed_schema_prefix':prefix,'generated_status_token_ids':generated,'attributes_predicted':False}
    for image in images:image.close()
    model.config.use_cache=False
    return result,audit
