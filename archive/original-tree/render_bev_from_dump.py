#!/usr/bin/env python3
"""BEV (top-down) GT vs tracker a partir dos dumps. Rápido, sem GPU.
Ego na base, anéis de distância, GT (círculo cheio + trilha), tracker (quadrado + trilha)."""
import sys, json, glob, argparse, numpy as np, cv2
from pathlib import Path
sys.path.insert(0,'/home/ericyos/airsim')
from sort_tracker import SortTracker

PAL=[(0,255,0),(255,128,0),(0,128,255),(255,0,255),(0,255,255),(128,0,255)]
GTCOL={'Drone3':(80,220,80),'Drone4':(80,160,255),'Intruder1':(255,160,80)}
W=H=760; SCALE=7.0; OX,OY=W//2,H-80


def to_px(g, ego):
    dx=g[0]-ego[0]; dy=g[1]-ego[1]
    return int(OX+dy*SCALE), int(OY-dx*SCALE)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dumps", default="det_dumps_best")
    ap.add_argument("--fuse_w", type=float, default=0.5); ap.add_argument("--fuse_thr", type=float, default=0.5)
    ap.add_argument("--max_dist", type=float, default=90.0)
    ap.add_argument("--max_age", type=int, default=15); ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--outdir", default="videos_results")
    args=ap.parse_args()
    DST=Path("/mnt/c/Users/eric_/OneDrive/Área de Trabalho/Mestrado ITA/Simulação/drone_scales")/args.outdir
    for dp in sorted(glob.glob(f"{args.dumps}/*.json")):
        name=Path(dp).stem; env=name.split('_')[0]
        sess=Path("sessions")/name
        meta=json.load(open(sess/"meta.json")); fps=meta["fps"]
        xplat={f['frame']:np.array(f['x_plat']) for f in meta['frames']}
        D=json.load(open(dp))['frames']
        out=DST/f"bev_{env}"; out.mkdir(parents=True, exist_ok=True)
        w=cv2.VideoWriter(str(out/f"{name}_BEV.mp4"),cv2.VideoWriter_fourcc(*'mp4v'),max(1.0,fps),(W,H))
        trk=SortTracker(iou_thr=0.3,max_age=args.max_age,min_hits=args.min_hits,nms_iou=0.4)
        gtr={n:[] for n in GTCOL}; ttr={}
        for fr in D:
            i=fr['frame']; xp=xplat.get(i)
            if xp is None: continue
            ego=xp[:3]
            dets=[]
            for d in fr['dets']:
                f=args.fuse_w*d['yolo']+(1-args.fuse_w)*d['pn']
                if f>=args.fuse_thr and d['d']<=args.max_dist:
                    dets.append({'bbox':tuple(d['bbox']),'fused':f,'d':d['d'],'gpos':np.array(d['gpos'])})
            tracks=trk.update(dets)
            cv=np.full((H,W,3),32,np.uint8)
            for rng in (10,20,30,40,50,60):
                cv2.circle(cv,(OX,OY),int(rng*SCALE),(55,55,55),1)
                cv2.putText(cv,f"{rng}m",(OX+int(rng*SCALE)-14,OY-3),cv2.FONT_HERSHEY_SIMPLEX,0.32,(90,90,90),1)
            cv2.drawMarker(cv,(OX,OY),(255,255,255),cv2.MARKER_TRIANGLE_UP,18,2)
            cv2.putText(cv,"EGO",(OX-14,OY+20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
            for n,g in fr['gt'].items():
                u,v=to_px(np.array(g),ego); gtr.setdefault(n,[]).append((u,v)); gtr[n]=gtr[n][-50:]
                col=GTCOL.get(n,(200,200,200))
                for a in range(1,len(gtr[n])): cv2.line(cv,gtr[n][a-1],gtr[n][a],col,1)
                cv2.circle(cv,(u,v),7,col,-1); cv2.putText(cv,f"GT {n}",(u+9,v),cv2.FONT_HERSHEY_SIMPLEX,0.36,col,1)
            for t in tracks:
                u,v=to_px(t.det['gpos'],ego); col=PAL[t.id%len(PAL)]
                ttr.setdefault(t.id,[]).append((u,v)); ttr[t.id]=ttr[t.id][-50:]
                for a in range(1,len(ttr[t.id])): cv2.line(cv,ttr[t.id][a-1],ttr[t.id][a],col,1)
                cv2.rectangle(cv,(u-7,v-7),(u+7,v+7),col,2); cv2.putText(cv,f"T{t.id}",(u+9,v+4),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,1)
            cv2.putText(cv,f"{name}  f{i}  o=GT  []=tracker",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
            w.write(cv)
        w.release(); print(f"  {name}_BEV.mp4")
    print("DONE ->",DST)


if __name__=="__main__":
    main()
