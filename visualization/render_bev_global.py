#!/usr/bin/env python3
"""Top-down bird's-eye view in a FIXED GLOBAL (world) frame: the ego itself moving,
plus the ground truth and the tracker, all in absolute coordinates. The frame is fixed and
sized from the scene bounds -- the moving-observer demonstration of Section 7.2.4."""
import sys, json, glob, argparse, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # repo root
from common.config import CONFIG
from tracking.sort import SortTracker

PAL=[(0,255,0),(255,128,0),(0,128,255),(255,0,255),(0,255,255),(128,0,255)]
GTCOL={'Drone3':(80,220,80),'Drone4':(80,160,255),'Intruder1':(255,160,80)}
W=H=820


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dumps", default="det_dumps_best")
    ap.add_argument("--fuse_w", type=float, default=0.5); ap.add_argument("--fuse_thr", type=float, default=0.5)
    ap.add_argument("--max_dist", type=float, default=90.0)
    ap.add_argument("--max_age", type=int, default=15); ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--outdir", default="videos_results")
    args=ap.parse_args()
    DST=CONFIG.resolve(args.outdir)   # relative to paths.data_root (configs/default.yaml)
    for dp in sorted(glob.glob(f"{args.dumps}/*.json")):
        name=Path(dp).stem; env=name.split('_')[0]
        sess=Path("sessions")/name
        meta=json.load(open(sess/"meta.json")); fps=meta["fps"]
        xplat={f['frame']:np.array(f['x_plat']) for f in meta['frames']}
        D=json.load(open(dp))['frames']
        # --- scene bounds (global): ego + ground truth ---
        xs=[]; ys=[]
        for f in meta['frames']: xs.append(f['x_plat'][0]); ys.append(f['x_plat'][1])
        for fr in D:
            for g in fr['gt'].values(): xs.append(g[0]); ys.append(g[1])
        if not xs: continue
        cx=(min(xs)+max(xs))/2; cy=(min(ys)+max(ys))/2
        span=max(max(xs)-min(xs), max(ys)-min(ys), 20)*1.15
        scale=(min(W,H)*0.88)/span
        def to_px(gx,gy):  # X (north) = up, Y (east) = right
            return int(W/2+(gy-cy)*scale), int(H/2-(gx-cx)*scale)
        out=DST/f"bevG_{env}"; out.mkdir(parents=True, exist_ok=True)
        w=cv2.VideoWriter(str(out/f"{name}_BEVglobal.mp4"),cv2.VideoWriter_fourcc(*'mp4v'),max(1.0,fps),(W,H))
        trk=SortTracker(iou_thr=0.3,max_age=args.max_age,min_hits=args.min_hits,nms_iou=0.4)
        gtr={}; ttr={}; egotr=[]
        for fr in D:
            i=fr['frame']; xp=xplat.get(i)
            if xp is None: continue
            dets=[]
            for d in fr['dets']:
                f=args.fuse_w*d['yolo']+(1-args.fuse_w)*d['pn']
                if f>=args.fuse_thr and d['d']<=args.max_dist:
                    dets.append({'bbox':tuple(d['bbox']),'fused':f,'d':d['d'],'gpos':np.array(d['gpos'])})
            tracks=trk.update(dets)
            cv=np.full((H,W,3),28,np.uint8)
            # world grid every 10 m
            g0x=np.floor(min(xs)/10)*10; g1x=np.ceil(max(xs)/10)*10
            g0y=np.floor(min(ys)/10)*10; g1y=np.ceil(max(ys)/10)*10
            for gx in np.arange(g0x,g1x+1,10):
                p1=to_px(gx,min(ys)); p2=to_px(gx,max(ys)); cv2.line(cv,p1,p2,(48,48,48),1)
            for gy in np.arange(g0y,g1y+1,10):
                p1=to_px(min(xs),gy); p2=to_px(max(xs),gy); cv2.line(cv,p1,p2,(48,48,48),1)
            # EGO (moving) + trail + yaw arrow
            eu,ev=to_px(xp[0],xp[1]); egotr.append((eu,ev)); egotr=egotr[-120:]
            for a in range(1,len(egotr)): cv2.line(cv,egotr[a-1],egotr[a],(0,0,255),2)
            yaw=xp[8]; hx,hy=eu+int(14*np.cos(-yaw+np.pi/2))*0, ev  # simple marker
            cv2.drawMarker(cv,(eu,ev),(0,0,255),cv2.MARKER_TRIANGLE_UP,16,2)
            cv2.putText(cv,"EGO",(eu+8,ev-8),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
            # ground truth (global, moving) + trail
            for n,g in fr['gt'].items():
                u,v=to_px(g[0],g[1]); gtr.setdefault(n,[]).append((u,v)); gtr[n]=gtr[n][-120:]
                col=GTCOL.get(n,(200,200,200))
                for a in range(1,len(gtr[n])): cv2.line(cv,gtr[n][a-1],gtr[n][a],col,1)
                cv2.circle(cv,(u,v),7,col,-1); cv2.putText(cv,n,(u+9,v),cv2.FONT_HERSHEY_SIMPLEX,0.34,col,1)
            # tracker (global) + trilha
            for t in tracks:
                u,v=to_px(t.det['gpos'][0],t.det['gpos'][1]); col=PAL[t.id%len(PAL)]
                ttr.setdefault(t.id,[]).append((u,v)); ttr[t.id]=ttr[t.id][-120:]
                for a in range(1,len(ttr[t.id])): cv2.line(cv,ttr[t.id][a-1],ttr[t.id][a],col,1)
                cv2.rectangle(cv,(u-7,v-7),(u+7,v+7),col,2)
            cv2.putText(cv,f"{name}  f{i}  GLOBAL  vermelho=EGO  o=GT  []=tracker",(8,20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)
            # escala
            cv2.line(cv,(20,H-20),(20+int(10*scale),H-20),(255,255,255),2)
            cv2.putText(cv,"10m",(20,H-26),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
            w.write(cv)
        w.release(); print(f"  {name}_BEVglobal.mp4")
    print("DONE ->",DST)


if __name__=="__main__":
    main()
