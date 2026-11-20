"""Train/evaluate Stage1CNN with curated/negation-aware labels and correct pixels.

Freeze a CheXpert-only pretrained encoder; cache spatial features; fit the
existing attention head. MeSH/report-derived labels remain weak supervision.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from chest_xpert.disease_targets import DISEASES, disease_targets
from chest_xpert.checkpoint_engine import prepare_checkpoint_image, CheckpointEngine
from cliq_rrg.stage1 import Stage1CNN


def load_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class ImageRows(Dataset):
    def __init__(self,rows,scale,crop):self.rows=rows;self.scale=scale;self.crop=crop
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return prepare_checkpoint_image(self.rows[index]['image_path'],224,self.scale,self.crop)


def metrics(scores,targets,masks,rows):
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row['study_id']].append(i)
    indices=list(groups.values())
    scores=np.stack([scores[ix].mean(0) for ix in indices]);targets=np.stack([targets[ix[0]] for ix in indices]);masks=np.stack([masks[ix[0]] for ix in indices])
    per_label={};aucs=[];aps=[]
    for i,label in enumerate(DISEASES):
        known=masks[:,i]>0;truth=targets[known,i];prediction=scores[known,i]
        positive=int(truth.sum());negative=int(len(truth)-positive)
        result={'positive_studies':positive,'negative_studies':negative,'auroc':None,'average_precision':None}
        if positive>=3 and negative>=10:
            result['auroc']=float(roc_auc_score(truth,prediction));result['average_precision']=float(average_precision_score(truth,prediction));aucs.append(result['auroc']);aps.append(result['average_precision'])
        per_label[label]=result
    return {'studies':len(indices),'macro_auroc':float(np.mean(aucs)) if aucs else None,'macro_average_precision':float(np.mean(aps)) if aps else None,'evaluable_labels':len(aucs),'score_min':float(scores.min()),'score_max':float(scores.max()),'per_label':per_label}


def extract(model,rows,device,scale,crop,batch_size):
    loader=DataLoader(ImageRows(rows,scale,crop),batch_size=batch_size,num_workers=4,pin_memory=True)
    result=[];model.eval()
    with torch.no_grad():
        for i,images in enumerate(loader):
            result.append(model.encoder(images.to(device)).detach().half().cpu())
            if (i+1)%25==0:print('FEATURES',len(result)*batch_size,'/',len(rows),flush=True)
    return torch.cat(result)


def predict_features(model,features,device):
    result=[];model.alignment.eval()
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(features),batch_size=128):result.append(model.alignment(batch.to(device).float())['disease_probs'].cpu())
    return torch.cat(result).numpy()


def main(args):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.cuda.set_device(args.device);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    train=load_rows(args.train);validation=load_rows(args.validation)
    if {r['study_id'] for r in train}&{r['study_id'] for r in validation}:raise ValueError('Study leakage')
    labels_train=[disease_targets(row) for row in train];labels_val=[disease_targets(row) for row in validation]
    targets=torch.tensor([pair[0] for pair in labels_train]);masks=torch.tensor([pair[1] for pair in labels_train])
    val_targets=np.array([pair[0] for pair in labels_val]);val_masks=np.array([pair[1] for pair in labels_val])
    positives=(targets*masks).sum(0);negatives=((1-targets)*masks).sum(0)
    label_stats={label:{'positive_images':int(positives[i]),'negative_images':int(negatives[i]),'supported':bool(positives[i]>=5 and negatives[i]>=20)} for i,label in enumerate(DISEASES)}
    (output/'label_audit.json').write_text(json.dumps(label_stats,indent=2));print('LABELS',json.dumps(label_stats),flush=True)
    if not positives.sum():raise ValueError('No positive disease labels; refusing all-negative training')
    original=CheckpointEngine(device=args.device)
    baseline=[]
    with torch.no_grad():
        for images in DataLoader(ImageRows(validation,original.pixel_scale,original.crop),batch_size=args.batch_size,num_workers=4):
            baseline.append(original.model(images.to(args.device))['disease_probs'].cpu())
    baseline=metrics(torch.cat(baseline).numpy(),val_targets,val_masks,validation)
    del original;torch.cuda.empty_cache()
    print('OLD_CHECKPOINT',json.dumps({k:v for k,v in baseline.items() if k!='per_label'}),flush=True)
    # CheXpert-only pretraining avoids the known OpenI component of the old all-dataset weights.
    model=Stage1CNN(num_diseases=14,pretrained=True,pretrained_weights=args.backbone).to(args.device)
    model.alignment.disease_specific_queries = args.disease_specific_queries
    for parameter in model.encoder.parameters():parameter.requires_grad=False
    features_train=extract(model,train,args.device,1024.0,'center',args.batch_size)
    features_val=extract(model,validation,args.device,1024.0,'center',args.batch_size)
    torch.save({'train':features_train,'validation':features_val},output/'features.pt')
    weights=(negatives/positives.clamp_min(1)).clamp(1,20).to(args.device)
    dataset=TensorDataset(features_train,targets,masks)
    loader=DataLoader(dataset,batch_size=64,shuffle=True)
    optimizer=torch.optim.AdamW(model.alignment.parameters(),lr=args.lr,weight_decay=1e-4)
    best=-float('inf');stale=0;history=[];start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.alignment.train();losses=[]
        for feature,target,mask in loader:
            feature=feature.to(args.device).float();target=target.to(args.device);mask=mask.to(args.device)
            logits=model.alignment(feature)['disease_logits']
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=weights,reduction='none')*mask).sum()/mask.sum().clamp_min(1)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.alignment.parameters(),5);optimizer.step();losses.append(float(loss))
        score=metrics(predict_features(model,features_val,args.device),val_targets,val_masks,validation)
        current=score['macro_auroc']
        if current is None:raise ValueError('Insufficient validation support')
        entry={'epoch':epoch,'loss':float(np.mean(losses)),'macro_auroc':current,'macro_average_precision':score['macro_average_precision'],'elapsed_s':time.perf_counter()-start};history.append(entry);print('EPOCH',json.dumps(entry),flush=True)
        if current>best:
            best=current;stale=0;model.eval();model.save(output/'cnn_model.pth')
            (output/'validation_metrics.json').write_text(json.dumps(score,indent=2))
        else:stale+=1
        (output/'progress.json').write_text(json.dumps({'phase':'training','epoch':epoch,'best_validation_auroc':best,'history':history},indent=2))
        if stale>=args.patience:break
    selected=json.loads((output/'validation_metrics.json').read_text())
    config={'model_name':'Chest Xpert corrected Stage1CNN','num_diseases':14,'diseases':DISEASES,'resolution':224,'pixel_scale':1024.0,'crop':'center','backbone':args.backbone,'frozen_encoder':True,'disease_specific_queries':args.disease_specific_queries,'supervision':'MeSH positives plus conservative assertion rules; unknown targets masked','label_support':{k:v['supported'] for k,v in label_stats.items()},'epochs':len(history),'selection':'macro AUROC on study-grouped validation set','manifest_sha256':{name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('train',args.train),('validation',args.validation)]}}
    (output/'training_config.json').write_text(json.dumps(config,indent=2))
    result={'baseline':baseline,'corrected':selected,'improved_auroc':selected['macro_auroc']>baseline['macro_auroc'],'history':history,'limitations':'Weak MeSH/report-derived labels; sparse labels omitted from macro metrics. Validation model selection, not independent clinical validation. Final test split untouched.'}
    (output/'comparison.json').write_text(json.dumps(result,indent=2));(output/'progress.json').write_text(json.dumps({'phase':'complete','best_validation_auroc':best,'improved_auroc':result['improved_auroc']},indent=2))
    print('DONE',json.dumps({'baseline_macro_auroc':baseline['macro_auroc'],'candidate_macro_auroc':selected['macro_auroc'],'improved':result['improved_auroc']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',default='data/report_training_v3/train.jsonl');p.add_argument('--validation',default='data/report_training_v3/validation.jsonl');p.add_argument('--output',default='checkpoints/cnn_corrected');p.add_argument('--device',default='cuda:0');p.add_argument('--backbone',default='densenet121-res224-chex');p.add_argument('--epochs',type=int,default=30);p.add_argument('--patience',type=int,default=5);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--disease-specific-queries',action='store_true',help='Train a separate disease attention query for each class; saves its setting in training_config.json')
    main(p.parse_args())
