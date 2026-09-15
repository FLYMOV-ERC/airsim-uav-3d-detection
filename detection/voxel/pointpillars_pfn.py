#!/usr/bin/env python3
"""PointPillars with a PFN (Pillar Feature Network) -- LEARNED pillar features.
Instead of the hand-crafted [count, mean_down, max_fwd], every pillar passes through a
per-point mini-PointNet (MLP + max-pool over the points of the cell), giving C learned
features -> BEV pseudo-image -> the same center-based CNN (heatmap + regression).
This is the strong, fair voxel baseline, and the chapter's best-localizing model
(RMSE 0.87 m, Table 7.4).

Modes: train (default) and dump (--dump)."""
import sys, json, glob, argparse, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
import common.pipeline as P
from tracking.ekf_3d import rotation_matrix
from detection.voxel.pointpillars import XMIN,XMAX,ZMIN,ZMAX,CELL,GW,GH,gt_cv,cell_of

NCELL = GH*GW
MAXP  = 12000   # cap on points per frame (memory safety)
PC    = 64      # width of the learned pillar features


class PFNDataset(Dataset):
    def __init__(self, sessions):
        self.items=[]
        for s in sessions:
            s=Path(s); meta=json.load(open(s/"meta.json"))
            for fr in meta["frames"]:
                self.items.append((s, fr["frame"], np.array(fr["x_plat"]), fr["gt_global"]))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        s,i,xp,gt=self.items[idx]
        cfp=s/"pccv"/f"{i:05d}.npy"; dfp=s/"depth"/f"{i:05d}.npy"
        if cfp.exists(): pc=np.load(cfp).astype(np.float32)
        elif dfp.exists(): pc=P.depth_to_pointcloud_cv(np.load(dfp).astype(np.float32))
        else: pc=np.zeros((1,3),np.float32)
        x=pc[:,0]; dn=pc[:,1]; z=pc[:,2]
        m=(x>=XMIN)&(x<XMAX)&(z>=ZMIN)&(z<ZMAX)
        x,dn,z=x[m],dn[m],z[m]
        if len(x)>MAXP:
            sel=np.random.choice(len(x),MAXP,replace=False); x,dn,z=x[sel],dn[sel],z[sel]
        jj=np.clip(((x-XMIN)/CELL).astype(np.int64),0,GW-1); ii=np.clip(((z-ZMIN)/CELL).astype(np.int64),0,GH-1)
        gidx=ii*GW+jj
        pts=np.stack([x,dn,z],axis=1).astype(np.float32)
        # heatmap + regression (identical to the lite variant)
        hm=np.zeros((GH,GW),np.float32); reg=np.zeros((4,GH,GW),np.float32); mask=np.zeros((GH,GW),np.float32)
        for n,g in gt.items():
            c=gt_cv(xp,g)
            if not(XMIN<=c[0]<XMAX and ZMIN<=c[2]<ZMAX): continue
            ci,cj=cell_of(c[0],c[2])
            for di in range(-2,3):
                for dj in range(-2,3):
                    yi,yj=ci+di,cj+dj
                    if 0<=yi<GH and 0<=yj<GW: hm[yi,yj]=max(hm[yi,yj],math.exp(-(di*di+dj*dj)/2.0))
            reg[:,ci,cj]=[c[0]-(XMIN+(cj+0.5)*CELL), c[2]-(ZMIN+(ci+0.5)*CELL), c[1], 0.0]; mask[ci,cj]=1
        return (torch.from_numpy(pts), torch.from_numpy(gidx),
                torch.from_numpy(hm), torch.from_numpy(reg), torch.from_numpy(mask))


def collate(batch):
    pts=[]; gidx=[]; hms=[]; regs=[]; masks=[]
    for b,(p,g,hm,reg,mask) in enumerate(batch):
        pts.append(p); gidx.append(g + b*NCELL)   # global pillar index (batch*ncell + linear)
        hms.append(hm); regs.append(reg); masks.append(mask)
    return (torch.cat(pts,0), torch.cat(gidx,0).long(),
            torch.stack(hms), torch.stack(regs), torch.stack(masks), len(batch))


class PFNNet(nn.Module):
    def __init__(self):
        super().__init__()
        # per-point mini-PointNet: 6D augmented features -> PC
        self.pfn=nn.Sequential(nn.Linear(6,PC), nn.BatchNorm1d(PC), nn.ReLU(),
                               nn.Linear(PC,PC), nn.BatchNorm1d(PC), nn.ReLU())
        self.enc=nn.Sequential(nn.Conv2d(PC,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
                               nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
                               nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.hm=nn.Conv2d(128,1,1); self.reg=nn.Conv2d(128,4,1)
    def forward(self, pts, gidx, B):
        # pts:(M,3) xyz=right,down,fwd; gidx:(M,) global pillar id
        loc=(gidx % NCELL); j=(loc % GW).float(); i=(loc // GW).float()
        cxc=XMIN+(j+0.5)*CELL; czc=ZMIN+(i+0.5)*CELL   # pillar centre
        aug=torch.stack([pts[:,0],pts[:,1],pts[:,2], pts[:,0]-cxc, pts[:,2]-czc, pts[:,1]],dim=1)
        f=self.pfn(aug)                                 # (M,PC)
        grid=torch.full((B*NCELL,PC), -1e4, device=pts.device, dtype=f.dtype)
        grid=grid.scatter_reduce(0, gidx.unsqueeze(1).expand(-1,PC), f, reduce="amax", include_self=True)
        grid=torch.where(grid<-1e3, torch.zeros_like(grid), grid)   # pillars vazios -> 0
        grid=grid.view(B,GH,GW,PC).permute(0,3,1,2).contiguous()
        e=self.enc(grid); return self.hm(e), self.reg(e)


def focal_hm(pred, gt):
    p=torch.sigmoid(pred).clamp(1e-4,1-1e-4); pos=(gt==1).float(); neg=(gt<1).float()
    lp=-torch.log(p)*torch.pow(1-p,2)*pos
    ln=-torch.log(1-p)*torch.pow(p,2)*torch.pow(1-gt,4)*neg
    return (lp.sum()+ln.sum())/pos.sum().clamp(min=1)


def train(args):
    dev='cuda'; sess=sorted(glob.glob(args.train_sessions))
    val_s=sess[::5]; tr_s=[x for x in sess if x not in val_s]
    tr=PFNDataset(tr_s); va=PFNDataset(val_s)
    print(f"PFN BEV {GH}x{GW} | train {len(tr)} val {len(va)}")
    trl=DataLoader(tr,batch_size=args.batch,shuffle=True,num_workers=8,pin_memory=True,drop_last=True,collate_fn=collate)
    val=DataLoader(va,batch_size=args.batch,shuffle=False,num_workers=8,collate_fn=collate)
    m=PFNNet().to(dev); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs)
    Path(args.out).mkdir(parents=True,exist_ok=True); best=1e9
    for ep in range(1,args.epochs+1):
        m.train()
        for pts,gidx,hm,reg,mask,B in trl:
            pts,gidx,hm,reg,mask=pts.to(dev),gidx.to(dev),hm.to(dev),reg.to(dev),mask.to(dev)
            ph,pr=m(pts,gidx,B); lh=focal_hm(ph.squeeze(1),hm); mk=mask.unsqueeze(1)
            lr=(F.smooth_l1_loss(pr*mk,reg*mk,reduction='sum')/mk.sum().clamp(min=1)) if mk.sum()>0 else 0*ph.sum()
            loss=lh+lr; opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); m.eval(); vl=0;nb=0
        with torch.no_grad():
            for pts,gidx,hm,reg,mask,B in val:
                pts,gidx,hm=pts.to(dev),gidx.to(dev),hm.to(dev)
                ph,pr=m(pts,gidx,B); vl+=focal_hm(ph.squeeze(1),hm).item();nb+=1
        vl/=max(1,nb); print(f"E{ep:2d}/{args.epochs} val_hm_loss={vl:.4f}",flush=True)
        if vl<best: best=vl; torch.save({"model_state_dict":m.state_dict(),"val_loss":vl},Path(args.out)/"best.pt")
    print(f"best val_hm_loss={best:.4f}")


def dump(args):
    dev='cuda'; out=Path(args.out); out.mkdir(exist_ok=True)
    m=PFNNet().to(dev); m.load_state_dict(torch.load(args.ckpt,map_location=dev,weights_only=False)["model_state_dict"]); m.eval()
    for s in sorted(glob.glob(args.sessions)):
        s=Path(s); dp=out/f"{s.name}.json"
        if dp.exists(): continue
        meta=json.load(open(s/"meta.json")); fout=[]
        for fr in meta["frames"]:
            i=fr["frame"]; xp=np.array(fr["x_plat"])
            cfp=s/"pccv"/f"{i:05d}.npy"; dfp=s/"depth"/f"{i:05d}.npy"
            if cfp.exists(): pc=np.load(cfp).astype(np.float32)
            elif dfp.exists(): pc=P.depth_to_pointcloud_cv(np.load(dfp).astype(np.float32))
            else: continue
            x=pc[:,0];dn=pc[:,1];z=pc[:,2]; mm=(x>=XMIN)&(x<XMAX)&(z>=ZMIN)&(z<ZMAX); x,dn,z=x[mm],dn[mm],z[mm]
            jj=np.clip(((x-XMIN)/CELL).astype(np.int64),0,GW-1); ii=np.clip(((z-ZMIN)/CELL).astype(np.int64),0,GH-1)
            pts=torch.from_numpy(np.stack([x,dn,z],1).astype(np.float32)).to(dev)
            gidx=torch.from_numpy((ii*GW+jj)).long().to(dev)
            with torch.no_grad():
                ph,pr=m(pts,gidx,1); hm=torch.sigmoid(ph)[0,0]; reg=pr[0]
                pooled=F.max_pool2d(hm.unsqueeze(0).unsqueeze(0),3,1,1)[0,0]
                peaks=(hm==pooled)&(hm>args.thr)
            ys,xs=torch.where(peaks); dets=[]
            for yi,xj in zip(ys.tolist(),xs.tolist()):
                sc=float(hm[yi,xj]); rr=reg[:,yi,xj].cpu().numpy()
                cx=XMIN+(xj+0.5)*CELL+rr[0]; cz=ZMIN+(yi+0.5)*CELL+rr[1]; cdn=rr[2]
                if cz<=0.5: continue
                cc=np.array([cx,cdn,cz]); frd=np.array([cc[2],cc[0],cc[1]])
                g=xp[:3]+rotation_matrix(xp[6],xp[7],xp[8])@frd
                u=cx/cz*P.FX+P.CX; v=cdn/cz*P.FY+P.CY
                dets.append({'bbox':[u-25,v-20,u+25,v+20],'yolo':sc,'pn':sc,'gpos':[float(q) for q in g],'d':float(np.linalg.norm(cc))})
            Rm=rotation_matrix(xp[6],xp[7],xp[8]); gtv={}
            for n,gg in fr.get('gt_global',{}).items():
                pf=Rm.T@(np.array(gg)-xp[:3])
                if pf[0]<=0.1: continue
                u=pf[1]*P.FX/pf[0]+P.CX; vv=pf[2]*P.FY/pf[0]+P.CY
                if 0<=u<P.IMAGE_W and 0<=vv<P.IMAGE_H: gtv[n]=[float(q) for q in gg]
            fout.append({'frame':i,'ts':fr['ts'],'dets':dets,'gt':gtv})
        json.dump({'name':s.name,'frames':fout},open(dp,'w'))
        print(f"PFN-DUMP {s.name}: {len(fout)}f {sum(len(f['dets']) for f in fout)} dets",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dump",action="store_true")
    ap.add_argument("--epochs",type=int,default=15); ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--out",default="runs/pointpillars_pfn"); ap.add_argument("--train_sessions",default="sessions/*_seq*")
    ap.add_argument("--ckpt",default="runs/pointpillars_pfn/best.pt"); ap.add_argument("--sessions",default="sessions/*_seq*")
    ap.add_argument("--thr",type=float,default=0.1)
    a=ap.parse_args()
    if a.dump:
        if a.out=="runs/pointpillars_pfn": a.out="det_dumps_pfn"
        dump(a)
    else: train(a)
