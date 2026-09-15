#!/usr/bin/env python3
"""PointPillars-lite (center-based BEV) -- a whole-scene 3D voxel detector, WITHOUT YOLO.
Pillarises the cloud (BEV X=forward x Y=right), a mini-PointNet per pillar -> 2D pseudo-image ->
CNN -> heatmap of drone centres + regression (dx, dy, z, log size). The alternative paradigm to the frustum.
On-the-fly dataset from the recorded sessions (depth -> cloud in CV + ground truth in CV). Trained from scratch."""
import sys, json, glob, argparse, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
import common.pipeline as P
from tracking.ekf_3d import rotation_matrix

# BEV grid in CV coordinates: X = right [-40, 40], Z = forward [0, 90], 1.25 m cells
import os as _os
XMIN,XMAX,ZMIN,ZMAX = -40.0,40.0,2.0,90.0
CELL = float(_os.environ.get("VOXEL_CELL","1.25"))
GW=int((XMAX-XMIN)/CELL); GH=int((ZMAX-ZMIN)/CELL)   # grid W x H (right x fwd)


def gt_cv(xp, g):
    pf=rotation_matrix(xp[6],xp[7],xp[8]).T@(np.array(g)-xp[:3])  # FRD
    return np.array([pf[1],pf[2],pf[0]])  # CV: right,down,fwd


def cell_of(cx, cz):
    j=int((cx-XMIN)/CELL); i=int((cz-ZMIN)/CELL)
    return i,j


TEMPORAL_K = int(_os.environ.get("VOXEL_TK","0"))   # nº de frames passados a acumular (0=single)


def _pc_to_global(pc_cv, xp):
    frd=np.stack([pc_cv[:,2],pc_cv[:,0],pc_cv[:,1]],axis=1)   # CV→FRD
    return (xp[:3] + (rotation_matrix(xp[6],xp[7],xp[8]) @ frd.T).T)


def _global_to_cv(g, xp):
    frd=(rotation_matrix(xp[6],xp[7],xp[8]).T @ (g - xp[:3]).T).T
    return np.stack([frd[:,1],frd[:,2],frd[:,0]],axis=1)       # FRD→CV


class BEVDataset(Dataset):
    def __init__(self, sessions):
        self.items=[]; self.fmap={}
        for s in sessions:
            s=Path(s); meta=json.load(open(s/"meta.json"))
            for fr in meta["frames"]:
                self.items.append((s, fr["frame"], np.array(fr["x_plat"]), fr["gt_global"]))
                self.fmap[(str(s),fr["frame"])]=np.array(fr["x_plat"])
    def __len__(self): return len(self.items)
    def _load_pc(self, s, i):
        cfp=s/"pccv"/f"{i:05d}.npy"      # cache CV in-range subamostrado (build_pc_cache.py)
        if cfp.exists(): return np.load(cfp).astype(np.float32)
        dfp=s/"depth"/f"{i:05d}.npy"
        if not dfp.exists(): return None
        return P.depth_to_pointcloud_cv(np.load(dfp).astype(np.float32))
    def __getitem__(self, idx):
        s,i,xp,gt=self.items[idx]
        pc=self._load_pc(s,i)
        if pc is None: pc=np.zeros((1,3),np.float32)
        if TEMPORAL_K>0:   # acumula frames passados, compensando movimento do ego
            accs=[pc]
            for dk in range(1,TEMPORAL_K+1):
                xpk=self.fmap.get((str(s),i-dk))
                if xpk is None: continue
                pck=self._load_pc(s,i-dk)
                if pck is None or len(pck)<5: continue
                g=_pc_to_global(pck,xpk)        # frame antigo → global
                accs.append(_global_to_cv(g,xp)) # global -> CURRENT camera frame
            pc=np.concatenate(accs,axis=0).astype(np.float32)
        # pillarise: per-cell features = [n, mean_xyz, max_xyz], simplified to occupancy + height
        feat=np.zeros((4,GH,GW),dtype=np.float32)  # [count, mean_down, min_down, max_fwd]
        x=pc[:,0]; dn=pc[:,1]; z=pc[:,2]
        m=(x>=XMIN)&(x<XMAX)&(z>=ZMIN)&(z<ZMAX)
        x,dn,z=x[m],dn[m],z[m]
        jj=np.clip(((x-XMIN)/CELL).astype(int),0,GW-1); ii=np.clip(((z-ZMIN)/CELL).astype(int),0,GH-1)
        lin=ii*GW+jj; ncell=GH*GW
        cnt=np.bincount(lin,minlength=ncell).astype(np.float32)
        sdn=np.bincount(lin,weights=dn,minlength=ncell)
        mxz=np.full(ncell,0.0); np.maximum.at(mxz,lin,z)
        feat[0]=np.log1p(cnt).reshape(GH,GW)
        nzf=cnt>0; meandn=np.zeros(ncell); meandn[nzf]=sdn[nzf]/cnt[nzf]
        feat[1]=meandn.reshape(GH,GW); feat[3]=mxz.reshape(GH,GW)
        # ground-truth heatmap + regression targets
        hm=np.zeros((GH,GW),dtype=np.float32); reg=np.zeros((4,GH,GW),dtype=np.float32); mask=np.zeros((GH,GW),dtype=np.float32)
        for n,g in gt.items():
            c=gt_cv(xp,g)
            if not(XMIN<=c[0]<XMAX and ZMIN<=c[2]<ZMAX): continue
            ci,cj=cell_of(c[0],c[2])
            # gaussiana no heatmap (raio 2)
            for di in range(-2,3):
                for dj in range(-2,3):
                    yi,yj=ci+di,cj+dj
                    if 0<=yi<GH and 0<=yj<GW:
                        hm[yi,yj]=max(hm[yi,yj], math.exp(-(di*di+dj*dj)/2.0))
            reg[:,ci,cj]=[c[0]-( XMIN+(cj+0.5)*CELL ), c[2]-(ZMIN+(ci+0.5)*CELL), c[1], 0.0]
            mask[ci,cj]=1
        return (torch.from_numpy(feat), torch.from_numpy(hm), torch.from_numpy(reg), torch.from_numpy(mask))


class PPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(4,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
                               nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
                               nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.hm=nn.Conv2d(128,1,1); self.reg=nn.Conv2d(128,4,1)
    def forward(self,x):
        f=self.enc(x); return self.hm(f), self.reg(f)


def focal_hm(pred, gt):
    p=torch.sigmoid(pred).clamp(1e-4,1-1e-4)
    pos=(gt==1).float(); neg=(gt<1).float()
    negw=torch.pow(1-gt,4)
    lp=-torch.log(p)*torch.pow(1-p,2)*pos
    ln=-torch.log(1-p)*torch.pow(p,2)*negw*neg
    npos=pos.sum().clamp(min=1)
    return (lp.sum()+ln.sum())/npos


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=15); ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--out",default="runs/pointpillars"); ap.add_argument("--train_sessions",default="sessions/*_seq*")
    args=ap.parse_args()
    dev='cuda'
    sess=sorted(glob.glob(args.train_sessions))
    # train/val split by session (80/20)
    val_s=sess[::5]; tr_s=[x for x in sess if x not in val_s]
    tr=BEVDataset(tr_s); va=BEVDataset(val_s)
    print(f"BEV grid {GH}x{GW} | train {len(tr)} frames, val {len(va)} frames")
    trl=DataLoader(tr,batch_size=args.batch,shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
    val=DataLoader(va,batch_size=args.batch,shuffle=False,num_workers=8)
    m=PPNet().to(dev); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs)
    Path(args.out).mkdir(parents=True,exist_ok=True); best=1e9
    for ep in range(1,args.epochs+1):
        m.train()
        for feat,hm,reg,mask in trl:
            feat,hm,reg,mask=feat.to(dev),hm.to(dev),reg.to(dev),mask.to(dev)
            ph,pr=m(feat); lh=focal_hm(ph.squeeze(1),hm)
            mk=mask.unsqueeze(1)
            lr=(F.smooth_l1_loss(pr*mk,reg*mk,reduction='sum')/mk.sum().clamp(min=1)) if mk.sum()>0 else 0*ph.sum()
            loss=lh+lr; opt.zero_grad();loss.backward();opt.step()
        sched.step()
        m.eval(); vl=0;nb=0
        with torch.no_grad():
            for feat,hm,reg,mask in val:
                feat,hm,reg,mask=feat.to(dev),hm.to(dev),reg.to(dev),mask.to(dev)
                ph,pr=m(feat); vl+=focal_hm(ph.squeeze(1),hm).item();nb+=1
        vl/=max(1,nb); print(f"E{ep:2d}/{args.epochs} val_hm_loss={vl:.4f}")
        if vl<best: best=vl; torch.save({"model_state_dict":m.state_dict(),"epoch":ep,"val_loss":vl},Path(args.out)/"best.pt")
    print(f"best val_hm_loss={best:.4f}")


if __name__=="__main__":
    main()
