#!/usr/bin/env python3
"""SORT leve (2D IoU + Hungarian) para tracking de drones já confirmados pela PointNet.

- Associação por IoU da BBOX 2D (limpa, do YOLO) — não pelo 3D ruidoso do frustum.
- NMS prévio colapsa múltiplas caixas no mesmo objeto.
- Só entram detecções já confirmadas (fused>=thr) → tracks não vêm de lixo.
- Cada track carrega a info 3D (distância, centro) da última medida associada.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua > 0 else 0.0


def nms(dets, iou_thr=0.4):
    """dets: list de dict com 'bbox' e 'fused'. Mantém a de maior fused por grupo."""
    if not dets: return []
    order = sorted(range(len(dets)), key=lambda i: dets[i]['fused'], reverse=True)
    keep = []
    while order:
        i = order.pop(0); keep.append(i)
        order = [j for j in order if iou(dets[i]['bbox'], dets[j]['bbox']) < iou_thr]
    return [dets[i] for i in keep]


class Track:
    _next = 0
    def __init__(self, det):
        self.id = Track._next; Track._next += 1
        self.bbox = np.array(det['bbox'], dtype=float)
        self.vel = np.zeros(4)
        self.hits = 1; self.age = 0; self.time_since_update = 0
        self.det = det                      # info 3D da última medida
        self.history = [self._center()]
    def _center(self):
        return [(self.bbox[0]+self.bbox[2])/2, (self.bbox[1]+self.bbox[3])/2]
    def predict(self):
        self.bbox = self.bbox + self.vel
        self.age += 1; self.time_since_update += 1
    def update(self, det):
        nb = np.array(det['bbox'], dtype=float)
        self.vel = 0.5*self.vel + 0.5*(nb - self.bbox)  # vel suavizada
        self.bbox = nb; self.det = det
        self.hits += 1; self.time_since_update = 0
        self.history.append(self._center())
        self.history = self.history[-60:]


class SortTracker:
    def __init__(self, iou_thr=0.3, max_age=8, min_hits=3, nms_iou=0.4, coast=4):
        self.iou_thr = iou_thr; self.max_age = max_age
        self.min_hits = min_hits; self.nms_iou = nms_iou; self.coast = coast
        self.tracks = []; self.frame = 0

    def update(self, dets):
        """dets: list de dict {bbox,fused,d,center_cv,yolo_conf,pn_prob,...}"""
        self.frame += 1
        dets = nms(dets, self.nms_iou)
        for t in self.tracks: t.predict()

        if self.tracks and dets:
            cost = np.zeros((len(dets), len(self.tracks)))
            for i, d in enumerate(dets):
                for j, t in enumerate(self.tracks):
                    cost[i, j] = 1.0 - iou(d['bbox'], t.bbox)
            ri, ci = linear_sum_assignment(cost)
            matched = {(int(i), int(j)) for i, j in zip(ri, ci)
                       if (1.0 - cost[i, j]) >= self.iou_thr}
        else:
            matched = set()
        md = {i for i, _ in matched}; mt = {j for _, j in matched}
        for i, j in matched: self.tracks[j].update(dets[i])
        for i, d in enumerate(dets):
            if i not in md: self.tracks.append(Track(d))
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return self.confirmed()

    def confirmed(self):
        # inclui tracks em COAST curto (predição) → persiste na falha de 2D/3D
        return [t for t in self.tracks
                if t.hits >= self.min_hits and t.time_since_update <= self.coast]

    def stats(self):
        return {'active': len(self.tracks), 'confirmed': len(self.confirmed())}
