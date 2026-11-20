"""Evaluate a validation-approved adapter once on a reserved ID-only test sample."""
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from scripts.train_report_adapter import enrich,generate
from scripts.assess_full_study_adapter import score,summarize
def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);args=p.parse_args();out=Path(args.experiment)
    assert json.loads((out/'assessment.json').read_text())['eligible_for_experimental_selection']
    if (out/'reserved_test_results.json').exists():raise RuntimeError('Reserved test already evaluated; refusing adaptive rerun')
    ids=json.loads((out/'reserved_test_ids.json').read_text());random.Random(44).shuffle(ids);ids=ids[:32]
    # Only now, after validation approval, open the reserved report targets.
    lookup={}
    for row in map(json.loads,Path('data/report_training_v3/test.jsonl').read_text().splitlines()):lookup.setdefault(row['study_id'],row)
    rows=enrich([lookup[i] for i in ids],Path('data/report_training_v3/evidence'),'cuda:0');torch.set_num_threads(4)
    config=json.loads((out/'run_config.json').read_text());processor=AutoProcessor.from_pretrained(config['model'],min_pixels=256*28*28,max_pixels=768*28*28,use_fast=False,local_files_only=True)
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(config['model'],torch_dtype=torch.float16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True)
    model=PeftModel.from_pretrained(base,str(out)).eval();cases=[]
    for row in rows:
     with model.disable_adapter():before=generate(model,processor,row,'cuda:0')
     after=generate(model,processor,row,'cuda:0')
     cases.append({'study_id':row['study_id'],'reference':row['target'],'base_output':before,'adapter_output':after,'base':score(before,row),'adapter':score(after,row)})
     (out/'reserved_test_partial.json').write_text(json.dumps(cases,indent=2));print('TEST',row['study_id'],flush=True)
    result={'cases':cases,'metrics':{name:summarize([c[name] for c in cases]) for name in ('base','adapter')},'used_for_selection':False,'selection_preceded_test':True,'limitations':'32 deterministic reserved test studies, weak lexical proxies, not clinical validation. No tuning or reselection based on this result.'}
    (out/'reserved_test_results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
