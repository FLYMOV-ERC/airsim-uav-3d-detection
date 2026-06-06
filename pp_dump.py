#!/usr/bin/env python3
"""Dump de detecções do PointPillars-lite (formato compatível c/ metrics_from_dump).
Picos do heatmap → centro 3D (CV) → global + bbox 2D projetada. score=heatmap."""
import sys, json, glob, argparse, numpy as np, torch, torch.nn.functional as Fn
from pathlib import Path
sys.path.insert(0,'/home/ericyos/airsim')
import inference_pipeline as P
from ekf_3d_global import rotation_matrix
from pointpillars_lite import BEVDataset, PPNet, GH, GW, XMIN, ZMIN, CELL, TEMPORAL_K, _pc_to_global, _global_to_cv


def _load_cached_pc(s, i):
    cfp=s/"pccv"/f"{i:05d}.npy"
    if cfp.exists(): return np.load(cfp).astype(np.float32)
    dfp=s/"depth"/f"{i:05d}.npy"
    return P.depth_to_pointcloud_cv(np.load(dfp).astype(np.float32)) if dfp.exists() else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",default="runs/pointpillars/best.pt")
    ap.add_argument("--sessions",default="sessions/*_seq*")
    ap.add_argument("--out",default="det_dumps_pp"); ap.add_argument("--thr",type=float,default=0.1)
    args=ap.parse_args()
    dev='cuda'; out=Path(args.out); out.mkdir(exist_ok=True)
    m=PPNet().to(dev); m.load_state_dict(torch.load(args.ckpt,map_location=dev,weights_only=False)["model_state_dict"]); m.eval()
    for s in sorted(glob.glob(args.sessions)):
        s=Path(s); dp=out/f"{s.name}.json"
        if dp.exists(): continue
        meta=json.load(open(s/"meta.json")); frames_out=[]
        fmap={fr["frame"]:np.array(fr["x_plat"]) for fr in meta["frames"]}
        for fr in meta["frames"]:
            i=fr["frame"]; xp=np.array(fr["x_plat"])
            pc=_load_cached_pc(s,i)
            if pc is None: continue
            if TEMPORAL_K>0:   # acumula frames passados compensando ego (igual ao treino)
                accs=[pc]
                for dk in range(1,TEMPORAL_K+1):
                    xpk=fmap.get(i-dk); pck=_load_cached_pc(s,i-dk)
                    if xpk is None or pck is None or len(pck)<5: continue
                    accs.append(_global_to_cv(_pc_to_global(pck,xpk),xp))
                pc=np.concatenate(accs,0).astype(np.float32)
            # pillariza (mesma lógica do dataset)
            feat=np.zeros((4,GH,GW),np.float32); x=pc[:,0];dn=pc[:,1];z=pc[:,2]
            mm=(x>=XMIN)&(x<XMIN+GW*CELL)&(z>=ZMIN)&(z<ZMIN+GH*CELL); x,dn,z=x[mm],dn[mm],z[mm]
            jj=np.clip(((x-XMIN)/CELL).astype(int),0,GW-1); ii=np.clip(((z-ZMIN)/CELL).astype(int),0,GH-1)
            lin=ii*GW+jj; nc=GH*GW
            cnt=np.bincount(lin,minlength=nc).astype(np.float32); sdn=np.bincount(lin,weights=dn,minlength=nc)
            mxz=np.zeros(nc); np.maximum.at(mxz,lin,z)
            feat[0]=np.log1p(cnt).reshape(GH,GW); md=np.zeros(nc); nzf=cnt>0; md[nzf]=sdn[nzf]/cnt[nzf]
            feat[1]=md.reshape(GH,GW); feat[3]=mxz.reshape(GH,GW)
            with torch.no_grad():
                ph,pr=m(torch.from_numpy(feat).unsqueeze(0).to(dev))
                hm=torch.sigmoid(ph)[0,0]; reg=pr[0]
                pooled=Fn.max_pool2d(hm.unsqueeze(0).unsqueeze(0),3,1,1)[0,0]
                peaks=(hm==pooled)&(hm>args.thr)
            ys,xs=torch.where(peaks); dets=[]
            for yi,xj in zip(ys.tolist(),xs.tolist()):
                sc=float(hm[yi,xj]); rr=reg[:,yi,xj].cpu().numpy()
                cx=XMIN+(xj+0.5)*CELL+rr[0]; cz=ZMIN+(yi+0.5)*CELL+rr[1]; cdn=rr[2]
                center_cv=np.array([cx,cdn,cz])
                if cz<=0.5: continue
                frd=np.array([center_cv[2],center_cv[0],center_cv[1]])
                g=xp[:3]+rotation_matrix(xp[6],xp[7],xp[8])@frd
                u=cx/cz*P.FX+P.CX; v=cdn/cz*P.FY+P.CY
                bb=[u-25,v-20,u+25,v+20]
                dets.append({'bbox':bb,'yolo':sc,'pn':sc,'gpos':[float(q) for q in g],'d':float(np.linalg.norm(center_cv))})
            # GT visíveis
            Rm=rotation_matrix(xp[6],xp[7],xp[8]); gtv={}
            for n,gg in fr['gt_global'].items() if 'gt_global' in fr else fr.get('gt',{}).items():
                pf=Rm.T@(np.array(gg)-xp[:3])
                if pf[0]<=0.1: continue
                u=pf[1]*P.FX/pf[0]+P.CX; vv=pf[2]*P.FY/pf[0]+P.CY
                if 0<=u<P.IMAGE_W and 0<=vv<P.IMAGE_H: gtv[n]=[float(q) for q in gg]
            frames_out.append({'frame':i,'ts':fr['ts'],'dets':dets,'gt':gtv})
        json.dump({'name':s.name,'frames':frames_out},open(dp,'w'))
        nd=sum(len(f['dets']) for f in frames_out); print(f"PP-DUMP {s.name}: {len(frames_out)}f {nd} dets")


if __name__=="__main__":
    main()
