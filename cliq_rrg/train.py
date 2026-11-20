"""Compatibility entry point for corrected CUDA Stage1CNN training.

The former substring-based label builder and [-1,1] pretrained-backbone
inputs are retired. The original checkpoint is never overwritten by default.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_cnn_corrected import main as train_corrected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-dir',default='data/report_training_v3')
    parser.add_argument('--output-dir',default='checkpoints/cnn_corrected_new')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--patience',type=int,default=5)
    args=parser.parse_args()
    root=Path(args.manifest_dir)
    train_corrected(argparse.Namespace(train=str(root/'train.jsonl'),validation=str(root/'validation.jsonl'),output=args.output_dir,device=args.device,backbone='densenet121-res224-chex',epochs=args.epochs,patience=args.patience,batch_size=args.batch_size,lr=args.lr))


if __name__=='__main__':main()
