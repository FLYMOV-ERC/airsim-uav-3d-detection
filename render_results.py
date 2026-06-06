#!/usr/bin/env python3
"""Renderiza vídeos de RESULTADO a partir dos dumps: RGB + dets + tracks SORT + GT.
Usa o best config (fuse_w 0.5, thr 0.5, max_age 15, min_hits 3). Saída mp4 por sequência."""
import sys, json, glob, argparse, numpy as np, cv2
from pathlib import Path
sys.path.insert(0,'/home/ericyos/airsim')
from sort_tracker import SortTracker
import inference_pipeline as P
from ekf_3d_global import rotation_matrix

PAL=[(0,255,0),(255,128,0),(0,128,255),(255,0,255),(0,255,255),(128,0,255)]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dumps", default="det_dumps_best")
    ap.add_argument("--fuse_w", type=float, default=0.5); ap.add_argument("--fuse_thr", type=float, default=0.5)
    ap.add_argument("--max_dist", type=float, default=90.0)
    ap.add_argument("--max_age", type=int, default=15); ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--outdir", default="videos_results")
    ap.add_argument("--only", default="")  # filtro opcional (ex: 'nh_seq01')
    args=ap.parse_args()
    DST_ROOT=Path("/mnt/c/Users/eric_/OneDrive/Área de Trabalho/Mestrado ITA/Simulação/drone_scales")/args.outdir
    for dp in sorted(glob.glob(f"{args.dumps}/*.json")):
        name=Path(dp).stem
        if args.only and args.only not in name: continue
        env=name.split('_')[0]
        sess=Path("sessions")/name
        meta=json.load(open(sess/"meta.json")); fps=meta["fps"]
        xplat={f['frame']:np.array(f['x_plat']) for f in meta['frames']}
        D=json.load(open(dp))['frames']
        out=DST_ROOT/f"videos_{env}"; out.mkdir(parents=True, exist_ok=True)
        w=cv2.VideoWriter(str(out/f"{name}_RESULT.mp4"),cv2.VideoWriter_fourcc(*'mp4v'),max(1.0,fps),(1280,720))
        trk=SortTracker(iou_thr=0.3,max_age=args.max_age,min_hits=args.min_hits,nms_iou=0.4)
        for fr in D:
            i=fr['frame']; xp=xplat.get(i)
            img=sess/"rgb"/f"{i:05d}.jpg"; bgr=cv2.imread(str(img))
            if bgr is None or xp is None: continue
            dets=[]
            for d in fr['dets']:
                f=args.fuse_w*d['yolo']+(1-args.fuse_w)*d['pn']
                if f>=args.fuse_thr and d['d']<=args.max_dist:
                    dets.append({'bbox':tuple(d['bbox']),'fused':f,'d':d['d'],'gpos':np.array(d['gpos'])})
            tracks=trk.update(dets)
            vis=bgr.copy()
            for d in dets:  # candidatos confirmados (cinza)
                x1,y1,x2,y2=[int(v) for v in d['bbox']]; cv2.rectangle(vis,(x1,y1),(x2,y2),(150,150,150),1)
            for t in tracks:
                col=PAL[t.id%len(PAL)]; x1,y1,x2,y2=[int(v) for v in t.bbox]
                cv2.rectangle(vis,(x1,y1),(x2,y2),col,2)
                cv2.putText(vis,f"ID{t.id} {t.det['d']:.0f}m F{t.det['fused']:.2f}",(x1,max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2)
                pts=[(int(a),int(b)) for a,b in t.history]
                for a in range(1,len(pts)): cv2.line(vis,pts[a-1],pts[a],col,2)
            # GT (cruz amarela)
            Rm=rotation_matrix(xp[6],xp[7],xp[8])
            for n,g in fr['gt'].items():
                pf=Rm.T@(np.array(g)-xp[:3])
                if pf[0]<=0.1: continue
                gu=pf[1]*P.FX/pf[0]+P.CX; gv=pf[2]*P.FY/pf[0]+P.CY
                if 0<=gu<1280 and 0<=gv<720: cv2.drawMarker(vis,(int(gu),int(gv)),(0,255,255),cv2.MARKER_CROSS,26,2)
            st=trk.stats()
            cv2.putText(vis,f"{name}  f{i}  tracks={st['confirmed']}  (amarelo=GT)",(10,26),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3)
            cv2.putText(vis,f"{name}  f{i}  tracks={st['confirmed']}  (amarelo=GT)",(10,26),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
            w.write(vis)
        w.release(); print(f"  {name}_RESULT.mp4")
    print("DONE ->", DST_ROOT)


if __name__=="__main__":
    main()
