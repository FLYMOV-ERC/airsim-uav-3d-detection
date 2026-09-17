#!/usr/bin/env python3
"""Voxel 3D denso (VoxelNet/SECOND-style, SEM spconv) — paradigma 3D completo.
Voxeliza a nuvem num grid 3D (X=right × Y=down × Z=fwd), backbone Conv3D, colapsa a
dimensão de altura (down) → pseudo-imagem BEV → cabeça center-based (heatmap+reg).
Diferente do PointPillars (que comprime a altura LOGO no início), aqui a altura é
processada por convoluções 3D antes de colapsar. Modos: train (default) e dump (--dump)."""
import sys, json, glob, argparse, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0,'/home/ericyos/airsim')
import inference_pipeline as P
from ekf_3d_global import rotation_matrix

# grid 3D em coords CV. célula 2.0m (denso 3D é caro → grade mais grossa que o PointPillars)
XMIN,XMAX = -40.0,40.0      # right
YMIN,YMAX = -24.0,24.0      # down
ZMIN,ZMAX =  2.0,90.0       # fwd
C3 = 2.0
GW=int((XMAX-XMIN)/C3); GD=int((YMAX-YMIN)/C3); GH=int((ZMAX-ZMIN)/C3)  # W(right) D(down) H(fwd)


def gt_cv(xp, g):
    pf=rotation_matrix(xp[6],xp[7],xp[8]).T@(np.array(g)-xp[:3])
    return np.array([pf[1],pf[2],pf[0]])  # right,down,fwd


class Vox3DDataset(Dataset):
    def __init__(self, sessions):
        self.items=[]
        for s in sessions:
            s=Path(s); meta=json.load(open(s/"meta.json"))
            for fr in meta["frames"]:
                self.items.append((s, fr["frame"], np.array(fr["x_plat"]), fr["gt_global"]))
    def __len__(self): return len(self.items)
    def voxelize(self, pc):
        x=pc[:,0]; dn=pc[:,1]; z=pc[:,2]
        m=(x>=XMIN)&(x<XMAX)&(dn>=YMIN)&(dn<YMAX)&(z>=ZMIN)&(z<ZMAX)
        x,dn,z=x[m],dn[m],z[m]
        jj=np.clip(((x-XMIN)/C3).astype(np.int64),0,GW-1)
        dd=np.clip(((dn-YMIN)/C3).astype(np.int64),0,GD-1)
        ii=np.clip(((z-ZMIN)/C3).astype(np.int64),0,GH-1)
        lin=(dd*GH+ii)*GW+jj; nc=GD*GH*GW
        cnt=np.bincount(lin,minlength=nc).astype(np.float32)
        vox=np.log1p(cnt).reshape(1,GD,GH,GW)   # (C=1, D, H, W)
        return vox
    def __getitem__(self, idx):
        s,i,xp,gt=self.items[idx]
        cfp=s/"pccv"/f"{i:05d}.npy"; dfp=s/"depth"/f"{i:05d}.npy"
        if cfp.exists(): pc=np.load(cfp).astype(np.float32)
        elif dfp.exists(): pc=P.depth_to_pointcloud_cv(np.load(dfp).astype(np.float32))
        else: pc=np.zeros((1,3),np.float32)
        vox=self.voxelize(pc).astype(np.float32)
        hm=np.zeros((GH,GW),np.float32); reg=np.zeros((3,GH,GW),np.float32); mask=np.zeros((GH,GW),np.float32)
        for n,g in gt.items():
            c=gt_cv(xp,g)
            if not(XMIN<=c[0]<XMAX and ZMIN<=c[2]<ZMAX): continue
            cj=int((c[0]-XMIN)/C3); ci=int((c[2]-ZMIN)/C3)
            for di in range(-2,3):
                for dj in range(-2,3):
                    yi,yj=ci+di,cj+dj
                    if 0<=yi<GH and 0<=yj<GW: hm[yi,yj]=max(hm[yi,yj],math.exp(-(di*di+dj*dj)/2.0))
            reg[:,ci,cj]=[c[0]-(XMIN+(cj+0.5)*C3), c[2]-(ZMIN+(ci+0.5)*C3), c[1]]; mask[ci,cj]=1
        return (torch.from_numpy(vox), torch.from_numpy(hm), torch.from_numpy(reg), torch.from_numpy(mask))


class Vox3DNet(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone 3D: reduz a altura (D) por stride, mantém H,W
        self.b3d=nn.Sequential(
            nn.Conv3d(1,16,3,padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16,32,3,stride=(2,1,1),padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32,32,3,stride=(2,1,1),padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32,64,3,stride=(2,1,1),padding=1), nn.BatchNorm3d(64), nn.ReLU())
        dd=GD
        for _ in range(3): dd=(dd+1)//2   # 3 strides 2 em D
        self.cflat=64*dd
        self.enc=nn.Sequential(nn.Conv2d(self.cflat,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
                               nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.hm=nn.Conv2d(128,1,1); self.reg=nn.Conv2d(128,3,1)
    def forward(self,x):
        f=self.b3d(x)                          # (B,64,d,H,W)
        B,C,d,H,W=f.shape
        f=f.reshape(B,C*d,H,W)                  # colapsa altura
        e=self.enc(f); return self.hm(e), self.reg(e)


def focal_hm(pred, gt):
    p=torch.sigmoid(pred).clamp(1e-4,1-1e-4); pos=(gt==1).float(); neg=(gt<1).float()
    lp=-torch.log(p)*torch.pow(1-p,2)*pos
    ln=-torch.log(1-p)*torch.pow(p,2)*torch.pow(1-gt,4)*neg
    return (lp.sum()+ln.sum())/pos.sum().clamp(min=1)


def train(args):
    dev='cuda'; sess=sorted(glob.glob(args.train_sessions))
    val_s=sess[::5]; tr_s=[x for x in sess if x not in val_s]
    tr=Vox3DDataset(tr_s); va=Vox3DDataset(val_s)
    print(f"VOX3D grid D{GD}xH{GH}xW{GW} | train {len(tr)} val {len(va)}")
    trl=DataLoader(tr,batch_size=args.batch,shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
    val=DataLoader(va,batch_size=args.batch,shuffle=False,num_workers=8)
    m=Vox3DNet().to(dev); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs)
    Path(args.out).mkdir(parents=True,exist_ok=True); best=1e9
    for ep in range(1,args.epochs+1):
        m.train()
        for vox,hm,reg,mask in trl:
            vox,hm,reg,mask=vox.to(dev),hm.to(dev),reg.to(dev),mask.to(dev)
            ph,pr=m(vox); lh=focal_hm(ph.squeeze(1),hm); mk=mask.unsqueeze(1)
            lr=(F.smooth_l1_loss(pr*mk,reg*mk,reduction='sum')/mk.sum().clamp(min=1)) if mk.sum()>0 else 0*ph.sum()
            loss=lh+lr; opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); m.eval(); vl=0;nb=0
        with torch.no_grad():
            for vox,hm,reg,mask in val:
                vox,hm=vox.to(dev),hm.to(dev); ph,pr=m(vox); vl+=focal_hm(ph.squeeze(1),hm).item();nb+=1
        vl/=max(1,nb); print(f"E{ep:2d}/{args.epochs} val_hm_loss={vl:.4f}",flush=True)
        if vl<best: best=vl; torch.save({"model_state_dict":m.state_dict(),"val_loss":vl},Path(args.out)/"best.pt")
    print(f"best val_hm_loss={best:.4f}")


def dump(args):
    dev='cuda'; out=Path(args.out); out.mkdir(exist_ok=True)
    m=Vox3DNet().to(dev); m.load_state_dict(torch.load(args.ckpt,map_location=dev,weights_only=False)["model_state_dict"]); m.eval()
    ds=Vox3DDataset([])  # só p/ reusar voxelize
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
            vox=torch.from_numpy(ds.voxelize(pc).astype(np.float32)).unsqueeze(0).to(dev)
            with torch.no_grad():
                ph,pr=m(vox); hm=torch.sigmoid(ph)[0,0]; reg=pr[0]
                pooled=F.max_pool2d(hm.unsqueeze(0).unsqueeze(0),3,1,1)[0,0]
                peaks=(hm==pooled)&(hm>args.thr)
            ys,xs=torch.where(peaks); dets=[]
            for yi,xj in zip(ys.tolist(),xs.tolist()):
                sc=float(hm[yi,xj]); rr=reg[:,yi,xj].cpu().numpy()
                cx=XMIN+(xj+0.5)*C3+rr[0]; cz=ZMIN+(yi+0.5)*C3+rr[1]; cdn=rr[2]
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
        print(f"VOX3D-DUMP {s.name}: {len(fout)}f {sum(len(f['dets']) for f in fout)} dets",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dump",action="store_true")
    ap.add_argument("--epochs",type=int,default=15); ap.add_argument("--batch",type=int,default=4)
    ap.add_argument("--out",default="runs/voxel3d"); ap.add_argument("--train_sessions",default="sessions/*_seq*")
    ap.add_argument("--ckpt",default="runs/voxel3d/best.pt"); ap.add_argument("--sessions",default="sessions/*_seq*")
    ap.add_argument("--thr",type=float,default=0.1)
    a=ap.parse_args()
    if a.dump:
        if a.out=="runs/voxel3d": a.out="det_dumps_vox3d"
        dump(a)
    else: train(a)
