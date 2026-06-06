#!/usr/bin/env python3
"""Treina FrustumSegNet: cls + center + size + segmentação fg/bg (labels por-ponto on-the-fly
= pontos dentro da box GT normalizada). Mesmo dataset frustum (v4)."""
import sys, argparse, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
sys.path.insert(0,'/home/ericyos/airsim')
from train_pointnet2_frustum import FrustumPN2Dataset
from train_pointnet2_painted import FocalLoss
from frustum_segnet import FrustumSegNet
import torch.nn.functional as F


def fg_labels(pc, c_gt, s_gt):
    # pc:(B,N,3) c_gt:(B,3) s_gt:(B,3) → (B,N) 1 se ponto dentro da box
    half=torch.clamp(s_gt/2, min=0.02).unsqueeze(1)            # (B,1,3)
    d=torch.abs(pc - c_gt.unsqueeze(1))                        # (B,N,3)
    return (d<=half).all(dim=2).float()                        # (B,N)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_frustum_pn2_v4")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--output", default="runs/pointnet2_frustum/segnet")
    args=ap.parse_args()
    dev='cuda'
    tr=FrustumPN2Dataset(args.data, split="train"); va=FrustumPN2Dataset(args.data, split="val")
    posw=np.where(tr.is_drone>0.5, 1.0/max(1,tr.is_drone.sum()), 1.0/max(1,len(tr.is_drone)-tr.is_drone.sum()))
    sampler=WeightedRandomSampler(posw, len(posw))
    trl=DataLoader(tr,batch_size=args.batch,sampler=sampler,num_workers=args.workers,pin_memory=True,drop_last=True)
    val=DataLoader(va,batch_size=args.batch,shuffle=False,num_workers=args.workers,pin_memory=True)
    model=FrustumSegNet().to(dev); print(f"SegNet params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    focal=FocalLoss(alpha=0.5,gamma=2.0)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    Path(args.output).mkdir(parents=True,exist_ok=True); best=0.0
    for ep in range(1,args.epochs+1):
        model.train()
        for pc,isd,cg,sg in trl:
            pc=pc.to(dev);isd=isd.to(dev);cg=cg.to(dev);sg=sg.to(dev)
            cls,cp,sp,seg=model(pc,return_seg=True)
            loss=focal(cls,isd)
            pos=isd>0.5
            if pos.any():
                loss=loss+F.smooth_l1_loss(cp[pos],cg[pos])+F.smooth_l1_loss(sp[pos],sg[pos])
                fg=fg_labels(pc[pos],cg[pos],sg[pos])
                loss=loss+F.binary_cross_entropy_with_logits(seg[pos],fg)
            opt.zero_grad();loss.backward();opt.step()
        sched.step()
        # val
        model.eval();TP=FP=FN=0;cerr=[]
        with torch.no_grad():
            for pc,isd,cg,sg in val:
                pc=pc.to(dev);isd=isd.to(dev);cg=cg.to(dev);sg=sg.to(dev)
                cls,cp,sp=model(pc)
                pr=(torch.sigmoid(cls)>0.5).float()
                TP+=((pr==1)&(isd==1)).sum().item();FP+=((pr==1)&(isd==0)).sum().item();FN+=((pr==0)&(isd==1)).sum().item()
                pos=isd>0.5
                if pos.any(): cerr.append(((cp[pos]-cg[pos])**2).sum(1).mean().item())
        prec=TP/max(1,TP+FP);rec=TP/max(1,TP+FN);f1=2*prec*rec/max(1e-9,prec+rec)
        cm=np.mean(cerr) if cerr else 9
        print(f"E{ep:2d}/{args.epochs} va_f1={f1:.3f} va_prec={prec:.3f} va_rec={rec:.3f} va_c_mse={cm:.3f}")
        if f1>best:
            best=f1; torch.save({"model_state_dict":model.state_dict(),"val_f1":f1,"epoch":ep}, Path(args.output)/"best.pt")
    print(f"Best val_f1={best:.3f}")


if __name__=="__main__":
    main()
