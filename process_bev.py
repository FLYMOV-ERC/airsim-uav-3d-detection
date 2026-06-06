#!/usr/bin/env python3
"""Visualização TOP-DOWN (bird's-eye) do GT vs TRACKER em coords globais.
Processa sessão (best config) → para cada track, converte centro 3D (câmera) p/ global,
e desenha um mapa de cima: ego, drones reais (GT, círculo cheio) vs estimativa do tracker
(quadrado vazado), com trilhas. Gera vídeo BEV + plot estático final.
"""
import sys, json, argparse, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, '/home/ericyos/airsim')
from ultralytics import YOLO
from detector_frustum_pn2 import FrustumPN2Detector
from sort_tracker import SortTracker
import inference_pipeline as P
from ekf_3d_global import rotation_matrix

PAL = [(0,255,0),(255,128,0),(0,128,255),(255,0,255),(0,255,255),(128,0,255)]
GT_COL = {'Drone3':(80,220,80),'Drone4':(80,160,255),'Intruder1':(255,160,80)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session_rec2")
    ap.add_argument("--pointnet", default="runs/pointnet2_frustum/v5band/best.pt")
    ap.add_argument("--det_band", type=float, default=10.0)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--fuse_w", type=float, default=0.5); ap.add_argument("--fuse_thr", type=float, default=0.5)
    ap.add_argument("--max_dist", type=float, default=90.0)
    ap.add_argument("--max_age", type=int, default=15); ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sess = Path(args.session); meta = json.load(open(sess/"meta.json")); frames = meta["frames"]; fps = meta["fps"]
    yolo = YOLO("runs/drone/drone_synth_v1/weights/best.pt")
    det = FrustumPN2Detector(checkpoint_path=args.pointnet, cls_threshold=0.0, band_m=args.det_band)
    trk = SortTracker(iou_thr=0.3, max_age=args.max_age, min_hits=args.min_hits, nms_iou=0.4)

    # BEV canvas params (metros → pixels). X=fwd(p/ cima), Y=right(p/ direita)
    W = H = 760; SCALE = 7.0  # px por metro
    OX, OY = W//2, H - 80      # ego perto da base, centro horizontal

    def to_px(gx, gy, ego):
        # rel ao ego em FRD-ish global: usa X(fwd-north), Y(right-east) do NED global
        dx = gx - ego[0]; dy = gy - ego[1]
        u = int(OX + dy*SCALE); v = int(OY - dx*SCALE)
        return u, v

    out_path = args.out or str(sess/"bev_gt_vs_tracker.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), max(1.0,fps), (W,H))
    gt_trails = {n:[] for n in GT_COL}; trk_trails = {}
    all_gt = {n:[] for n in GT_COL}; all_trk = {}  # p/ plot estático
    err_by_drone = {n:[] for n in GT_COL}

    for fr in frames:
        i=fr["frame"]; xp=np.array(fr["x_plat"]); gt=fr["gt_global"]
        bgr=cv2.imread(str(sess/"rgb"/f"{i:05d}.jpg")); depth=np.load(sess/"depth"/f"{i:05d}.npy").astype(np.float32)
        pc=P.depth_to_pointcloud_cv(depth)
        r=yolo.predict(bgr,conf=args.conf,imgsz=1280,verbose=False); dets=[]
        if r and r[0].boxes is not None:
            for k in range(len(r[0].boxes.xyxy)):
                bb=tuple(map(float,r[0].boxes.xyxy.cpu().numpy()[k])); c=float(r[0].boxes.conf.cpu().numpy()[k])
                pr=det.predict(pc,depth,bb,c)
                if pr is None: continue
                f=args.fuse_w*c+(1-args.fuse_w)*pr['is_drone_prob']
                d3=float(np.linalg.norm(P.cv_to_frd(pr['center_cv'])))
                if f<args.fuse_thr or d3>args.max_dist: continue
                # centro global: ego + R(rpy) @ FRD(center_cv)
                frd=P.cv_to_frd(np.array(pr['center_cv']))
                g=xp[:3] + rotation_matrix(xp[6],xp[7],xp[8]) @ frd
                dets.append({'bbox':bb,'fused':f,'d':d3,'gpos':g})
        tracks=trk.update(dets)
        ego=xp[:3]

        canvas=np.full((H,W,3),32,np.uint8)
        # grid + anéis de distância
        for rng in (10,20,30,40,50):
            cv2.circle(canvas,(OX,OY),int(rng*SCALE),(55,55,55),1)
            cv2.putText(canvas,f"{rng}m",(OX+int(rng*SCALE)-14,OY-3),cv2.FONT_HERSHEY_SIMPLEX,0.35,(90,90,90),1)
        # ego (triângulo apontando p/ cima = fwd)
        cv2.drawMarker(canvas,(OX,OY),(255,255,255),cv2.MARKER_TRIANGLE_UP,18,2)
        cv2.putText(canvas,"EGO",(OX-16,OY+20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)

        # GT (círculo cheio + trilha)
        for n,g in gt.items():
            g=np.array(g); u,v=to_px(g[0],g[1],ego)
            gt_trails[n].append((u,v)); gt_trails[n]=gt_trails[n][-40:]; all_gt[n].append(g.copy())
            col=GT_COL.get(n,(200,200,200))
            for a in range(1,len(gt_trails[n])): cv2.line(canvas,gt_trails[n][a-1],gt_trails[n][a],col,1)
            cv2.circle(canvas,(u,v),7,col,-1)
            cv2.putText(canvas,f"GT {n}",(u+9,v),cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1)
        # TRACKER (quadrado vazado + trilha) + erro vs GT mais próximo
        for t in tracks:
            g=t.det['gpos']; u,v=to_px(g[0],g[1],ego)
            col=PAL[t.id%len(PAL)]
            trk_trails.setdefault(t.id,[]).append((u,v)); trk_trails[t.id]=trk_trails[t.id][-40:]
            all_trk.setdefault(t.id,[]).append(g.copy())
            for a in range(1,len(trk_trails[t.id])): cv2.line(canvas,trk_trails[t.id][a-1],trk_trails[t.id][a],col,1)
            cv2.rectangle(canvas,(u-7,v-7),(u+7,v+7),col,2)
            cv2.putText(canvas,f"T{t.id}",(u+9,v+4),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,1)
            # erro ao GT mais próximo (3D global)
            best=1e9; bn=None
            for n,gg in gt.items():
                e=np.linalg.norm(g-np.array(gg))
                if e<best: best=e; bn=n
            if bn and best<15: err_by_drone[bn].append(best)
        cv2.putText(canvas,"TOP-DOWN  o=GT (cheio)   []=Tracker",(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(canvas,f"f{i}",(W-60,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        writer.write(canvas)
    writer.release()
    print(f"[BEV] video: {out_path}")
    print("Erro 3D global tracker-vs-GT (m):")
    for n in err_by_drone:
        if err_by_drone[n]:
            e=np.array(err_by_drone[n]); print(f"  {n}: mediana={np.median(e):.1f}m  n={len(e)}")


if __name__ == "__main__":
    main()
