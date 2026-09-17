#!/usr/bin/env python3
"""Avalia o pipeline offline contra GROUND TRUTH (gt_global gravado).
Mede: (a) YOLO/track bate no pixel do drone? (b) distância 3D do frustum bate com GT?
Salva frames anotados: GT (amarelo) vs detecção YOLO+frustum (verde) + track (colorido).
"""
import sys, json, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, '/home/ericyos/airsim')
from ultralytics import YOLO
from detector_frustum_pn2 import FrustumPN2Detector
import inference_pipeline as P

sess = Path("session_rec")
meta = json.load(open(sess/"meta.json")); frames = meta["frames"]
yolo = YOLO("runs/drone/drone_synth_v1/weights/best.pt")
det = FrustumPN2Detector(checkpoint_path="runs/pointnet2_frustum/v4/best.pt", cls_threshold=0.0)
outdir = Path("eval_frames"); outdir.mkdir(exist_ok=True)

def box_center(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)

n_gt_visible=0; n_gt_detected=0; pix_errs=[]; dist_errs=[]
save_idx = set(range(0, len(frames), max(1, len(frames)//8)))
for fr in frames:
    i = fr["frame"]; x_plat = np.array(fr["x_plat"]); gt = fr["gt_global"]
    bgr = cv2.imread(str(sess/"rgb"/f"{i:05d}.jpg"))
    depth = np.load(sess/"depth"/f"{i:05d}.npy").astype(np.float32)
    pc = P.depth_to_pointcloud_cv(depth)
    # GT projetado + distância GT
    gt_proj = {}
    from ekf_3d_global import rotation_matrix
    Rm = rotation_matrix(x_plat[6], x_plat[7], x_plat[8])
    for name, g in gt.items():
        g = np.array(g)
        p_frd = Rm.T @ (g - x_plat[:3])   # FIX: transpor (mundo→câmera)
        xf, yf, zf = p_frd
        if xf <= 0.1: continue
        u = yf * P.FX / xf + P.CX; v = zf * P.FY / xf + P.CY
        if 0<=u<P.IMAGE_W and 0<=v<P.IMAGE_H:
            gt_dist = float(np.linalg.norm(g - x_plat[:3]))
            gt_proj[name] = (u, v, gt_dist)
    # YOLO + frustum
    res = yolo.predict(bgr, conf=0.25, imgsz=1280, verbose=False)
    dets = []
    if res and res[0].boxes is not None and len(res[0].boxes)>0:
        xyxy=res[0].boxes.xyxy.cpu().numpy(); confs=res[0].boxes.conf.cpu().numpy()
        for k in range(len(xyxy)):
            bb=tuple(map(float,xyxy[k])); conf=float(confs[k])
            pred=det.predict(pc,depth,bb,conf)
            if pred is None: continue
            d3=float(np.linalg.norm(P.cv_to_frd(pred['center_cv'])))
            dets.append({'bbox':bb,'conf':conf,'pn':pred['is_drone_prob'],'d':d3,
                         'fused':0.5*conf+0.5*pred['is_drone_prob']})
    # casa cada GT visível ao det mais próximo no pixel
    for name,(gu,gv,gd) in gt_proj.items():
        n_gt_visible += 1
        best=None; bestpx=1e9
        for de in dets:
            cu,cv_=box_center(de['bbox']); px=np.hypot(cu-gu,cv_-gv)
            if px<bestpx: bestpx, best = px, de
        if best is not None and bestpx < 80:   # dentro de 80px = mesmo drone
            n_gt_detected += 1; pix_errs.append(bestpx); dist_errs.append(abs(best['d']-gd))
    # salva frame anotado
    if i in save_idx:
        vis=bgr.copy()
        for de in dets:
            x1,y1,x2,y2=[int(v) for v in de['bbox']]
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(vis,f"det {de['d']:.0f}m F{de['fused']:.2f}",(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        for name,(gu,gv,gd) in gt_proj.items():
            cv2.drawMarker(vis,(int(gu),int(gv)),(0,255,255),cv2.MARKER_CROSS,40,3)
            cv2.putText(vis,f"GT {name} {gd:.0f}m",(int(gu)+12,int(gv)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
        cv2.imwrite(str(outdir/f"eval_{i:05d}.png"),vis)

print(f"GT drones visíveis (no quadro): {n_gt_visible}")
print(f"GT detectados por YOLO+frustum (<80px): {n_gt_detected} ({100*n_gt_detected/max(1,n_gt_visible):.0f}%)")
if pix_errs:
    print(f"Erro de PIXEL (det vs GT): media={np.mean(pix_errs):.1f}px mediana={np.median(pix_errs):.1f}px")
    print(f"Erro de DISTÂNCIA 3D (frustum vs GT): media={np.mean(dist_errs):.1f}m mediana={np.median(dist_errs):.1f}m")
print(f"frames anotados salvos em {outdir}/")
