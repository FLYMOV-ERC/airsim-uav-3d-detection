#!/usr/bin/env python3
"""ARCHIVED. Evaluate the offline pipeline against the recorded global ground truth.

Measures (a) whether the YOLO box / track lands on the drone's pixel, and (b)
whether the frustum's 3D range agrees with the ground truth. Writes annotated
frames: ground truth in yellow, YOLO + frustum detection in green, track in colour.

Superseded by evaluation/metrics.py plus visualization/render_results.py.
"""
import sys, json, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from ultralytics import YOLO
from detection.frustum.detector import FrustumPN2Detector
import common.pipeline as P
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
    # projected ground truth + ground-truth range
    gt_proj = {}
    from tracking.ekf_3d import rotation_matrix
    Rm = rotation_matrix(x_plat[6], x_plat[7], x_plat[8])
    for name, g in gt.items():
        g = np.array(g)
        p_frd = Rm.T @ (g - x_plat[:3])   # world -> camera is R^T, not R
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
    # match every visible ground-truth target to the nearest detection in pixels
    for name,(gu,gv,gd) in gt_proj.items():
        n_gt_visible += 1
        best=None; bestpx=1e9
        for de in dets:
            cu,cv_=box_center(de['bbox']); px=np.hypot(cu-gu,cv_-gv)
            if px<bestpx: bestpx, best = px, de
        if best is not None and bestpx < 80:   # within 80 px counts as the same drone
            n_gt_detected += 1; pix_errs.append(bestpx); dist_errs.append(abs(best['d']-gd))
    # save the annotated frame
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

print(f"Ground-truth drones visible in frame: {n_gt_visible}")
print(f"Ground-truth targets detected by YOLO+frustum (<80 px): {n_gt_detected} ({100*n_gt_detected/max(1,n_gt_visible):.0f}%)")
if pix_errs:
    print(f"PIXEL error (detection vs ground truth): mean={np.mean(pix_errs):.1f} px median={np.median(pix_errs):.1f} px")
    print(f"3D RANGE error (frustum vs ground truth): mean={np.mean(dist_errs):.1f} m median={np.median(dist_errs):.1f} m")
print(f"annotated frames written to {outdir}/")
