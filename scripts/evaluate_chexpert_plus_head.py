"""Score frozen original and trained candidate on identical CheXpert Plus splits."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from torchvision import models
from scripts.train_chexpert_plus_head import ROOT,Rows,evaluate

def main():
 out=Path('checkpoints/chexpert_plus_imagenet_full_v1');protocol=json.loads((out/'protocol.json').read_text());assert (out/'progress.json').exists() and json.loads((out/'progress.json').read_text())['phase']=='complete'
 source=Path('/home/doli/HCT_LSR/hct14_ensemble/models/densenet121_hct_lsr_14_stage1_best.pt')
 for split,digest in protocol['data_hashes'].items():assert hashlib.sha256((ROOT/f'{split}.jsonl').read_bytes()).hexdigest()==digest
 torch.set_num_threads(4);model=models.densenet121(weights=None,num_classes=14).to('cuda:0');results={}
 for name,path in [('original',source),('candidate',out/'best_model.pth')]:
  state=torch.load(path,map_location='cpu',weights_only=True)
  model.load_state_dict(state['model_state_dict'] if name=='original' else state)
  for split in (['valid'] if name=='original' else ['development','valid']):
   loader=DataLoader(Rows(ROOT/f'{split}.jsonl'),batch_size=32,num_workers=8,pin_memory=True)
   results.setdefault(name,{})[split]=evaluate(model,loader,'cuda:0')
  print(name,results[name]['valid']['macro_auroc'],results[name]['valid']['macro_ap'],flush=True)
 gate=results['candidate']['valid']['macro_auroc']>results['original']['valid']['macro_auroc'] and results['candidate']['valid']['macro_ap']>=results['original']['valid']['macro_ap']
 result={'cohort':'patient-disjoint CheXpert Plus fit/development; original official valid 202 images','results':results,'improves_both_official_valid_metrics':gate,'active_app_unchanged':True,'paper_comparison_valid':False,'limitations':'Official valid has 202 images; uncertainty and missing labels masked. This does not establish superiority over publications using other test sets.'}
 (out/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps({'official_valid':{k:{m:v['valid'][m] for m in ['macro_auroc','macro_ap']} for k,v in results.items()},'gate':gate},indent=2))
if __name__=='__main__':main()
