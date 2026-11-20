"""Train a 14-label CheXpert Plus CNN from ImageNet weights on patient-disjoint fit/dev data."""
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path('data/chexpert_plus_cnn_v1')
class Rows(Dataset):
 def __init__(self,path):
  self.rows=[json.loads(s) for s in Path(path).read_text().splitlines()]
  self.transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['image_path']) as image:x=self.transform(image.convert('RGB'))
  vals=np.array([np.nan if v is None else v for v in r['labels']],dtype=np.float32)
  mask=np.isfinite(vals)&np.isin(vals,[0,1]);target=np.nan_to_num(vals,nan=0);target=np.clip(target,0,1)
  return x,torch.from_numpy(target),torch.from_numpy(mask)
def evaluate(model,loader,device):
 model.eval();scores=[];targets=[];masks=[]
 with torch.inference_mode():
  for x,y,m in loader:
   s=model(x.to(device)).sigmoid().cpu().numpy();scores.append(s);targets.append(y.numpy());masks.append(m.numpy())
 scores=np.concatenate(scores);targets=np.concatenate(targets);masks=np.concatenate(masks);per={};aucs=[];aps=[]
 for j,label in enumerate(json.loads((ROOT/'manifest.json').read_text())['labels']):
  keep=masks[:,j].astype(bool);y=targets[keep,j];s=scores[keep,j];pos=int(y.sum());neg=int(len(y)-pos)
  if pos>=3 and neg>=10:
   auroc=float(roc_auc_score(y,s));ap=float(average_precision_score(y,s));aucs.append(auroc);aps.append(ap)
  else:auroc=ap=None
  per[label]={'positive':pos,'negative':neg,'auroc':auroc,'ap':ap}
 return {'images':len(scores),'evaluable_labels':len(aucs),'macro_auroc':float(np.mean(aucs)),'macro_ap':float(np.mean(aps)),'per_label':per}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='checkpoints/chexpert_plus_imagenet_full_v1');p.add_argument('--epochs',type=int,default=2);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=8);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'protocol.json').exists():raise RuntimeError('Use fresh output directory')
 torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.set_num_threads(4)
 split=json.loads((ROOT/'development_split.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text())
 protocol={'data_hashes':split['sha256'],'initialization':'torchvision DenseNet121 ImageNet weights only; no CheXpert Plus patients in pretrained weights','uncertain_policy':'-1 masked, unlike original LSR smoothing','target':'14 CheXpert Plus labels','fit_images':split['counts']['fit']['images'],'development_images':split['counts']['development']['images'],'official_valid_untouched':True,'model':'torchvision DenseNet121 full-parameter fine-tuning','epochs':a.epochs,'batch_size':a.batch_size,'optimizer':'AdamW lr 1e-4 weight_decay 1e-4, masked BCE','selection':'development macro AUROC; report macro AP too','test_used':False};(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 fit=DataLoader(Rows(ROOT/'fit.jsonl'),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0);dev=DataLoader(Rows(ROOT/'development.jsonl'),batch_size=a.batch_size,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
 device='cuda:0';model=models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1);model.classifier=torch.nn.Linear(model.classifier.in_features,14);model.to(device)
 for par in model.parameters():par.requires_grad=True
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);best=-1;history=[];started=time.monotonic()
 for epoch in range(1,a.epochs+1):
  model.train();total=0;seen=0
  for step,(x,y,m) in enumerate(fit,1):
   x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);m=m.to(device,non_blocking=True)
   logits=model(x);loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*m).sum()/m.sum().clamp_min(1)
   optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=float(loss);seen+=1
   if step%200==0:
    status={'phase':'training','epoch':epoch,'epochs':a.epochs,'batch':step,'batches_per_epoch':len(fit),'fraction':((epoch-1)+step/len(fit))/a.epochs,'elapsed_s':time.monotonic()-started};(out/'progress.json').write_text(json.dumps(status));print('PROGRESS',json.dumps(status),flush=True)
  measure=evaluate(model,dev,device);entry={'epoch':epoch,'train_loss':total/seen,'development':measure};history.append(entry);print('EPOCH',json.dumps({'epoch':epoch,'train_loss':total/seen,'auroc':measure['macro_auroc'],'ap':measure['macro_ap']}),flush=True)
  if measure['macro_auroc']>best:best=measure['macro_auroc'];torch.save(model.state_dict(),out/'best_model.pth');(out/'best_metrics.json').write_text(json.dumps(measure,indent=2))
  (out/'history.json').write_text(json.dumps(history,indent=2))
 (out/'progress.json').write_text(json.dumps({'phase':'complete','epochs':a.epochs,'elapsed_s':time.monotonic()-started}));print('COMPLETE',best,flush=True)
if __name__=='__main__':main()
