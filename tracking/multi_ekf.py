#!/usr/bin/env python3
"""MultiEKFTracker -- manages several EKFTracker3D instances, one per track.

Per-frame loop:
  1. predict() on every track
  2. Hungarian association in measurement space (Mahalanobis distance with chi-squared gating)
  3. Update de tracks matched
  4. spawn a new track for each unmatched detection
  5. Lifecycle (max_age, min_hits)
  6. return the confirmed tracks
"""
from __future__ import annotations
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # repo root

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Iterable, Sequence
import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.ekf_3d import EKFTracker3D, EKFParams, wrap_angle


# χ² inverso (4 dof) — gating
CHI2_INV_4DOF = {
    0.90: 7.78,
    0.95: 9.49,
    0.99: 13.28,
    0.995: 14.86,
}


@dataclass
class MultiTrackerParams:
    chi2_gate: float = 13.28
    max_age: int = 5
    min_hits: int = 3
    ekf_params: Optional[EKFParams] = None

    def __post_init__(self):
        if self.ekf_params is None:
            self.ekf_params = EKFParams()


class MultiEKFTracker:
    def __init__(self, params: Optional[MultiTrackerParams] = None):
        self.p = params if params is not None else MultiTrackerParams()
        self.tracks: List[EKFTracker3D] = []
        self.frame_count = 0
        self.total_created = 0
        self.total_deleted = 0

    # ─────────────────────────────────────────────────────────────────────
    def update(self,
               measurements: Sequence[Sequence],
               x_plat: np.ndarray,
               dt: float) -> List[EKFTracker3D]:
        """
        measurements: list of tuples/lists whose first four values
                       are (d, phi, theta, r). Optionally (..., conf, ...).
        x_plat: (9,) estado da plataforma
        dt: time (s) since the previous update
        return: list of confirmed tracks (references to the objects)
        """
        self.frame_count += 1

        for trk in self.tracks:
            trk.predict(dt)

        N = len(measurements)
        M = len(self.tracks)

        if M == 0 and N == 0:
            self._cull_dead()
            return self._collect_confirmed()
        elif M == 0:
            matched: List[Tuple[int, int]] = []
            unmatched_dets = list(range(N))
            unmatched_trks: List[int] = []
        elif N == 0:
            matched = []
            unmatched_dets = []
            unmatched_trks = list(range(M))
        else:
            matched, unmatched_dets, unmatched_trks = self._associate(
                measurements, x_plat
            )

        # Update matched
        for i_det, i_trk in matched:
            m = measurements[i_det]
            y_meas = np.array([m[0], m[1], m[2], m[3]])
            conf = float(m[4]) if len(m) > 4 else 1.0
            self.tracks[i_trk].update(y_meas, x_plat, confidence=conf)

        # Spawn new tracks for the unmatched detections
        for i_det in unmatched_dets:
            m = measurements[i_det]
            y_meas = np.array([m[0], m[1], m[2], m[3]])
            conf = float(m[4]) if len(m) > 4 else 1.0
            new_trk = EKFTracker3D(y_meas, x_plat, params=self.p.ekf_params)
            new_trk.last_confidence = conf
            self.tracks.append(new_trk)
            self.total_created += 1

        # Lifecycle
        self._cull_dead()

        return self._collect_confirmed()

    # ─────────────────────────────────────────────────────────────────────
    def _associate(self,
                   measurements: Sequence[Sequence],
                   x_plat: np.ndarray
                   ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        N = len(measurements)
        M = len(self.tracks)
        cost = np.full((N, M), np.inf)

        for j, trk in enumerate(self.tracks):
            y_pred = trk.compute_h(trk.x, x_plat)
            H = trk.compute_H(trk.x, x_plat)
            R = trk.build_R(y_pred[0])
            S = H @ trk.P @ H.T + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue
            for i, m in enumerate(measurements):
                y_m = np.array([m[0], m[1], m[2], m[3]])
                innov = y_m - y_pred
                innov[1] = wrap_angle(innov[1])
                innov[2] = wrap_angle(innov[2])
                mahal2 = float(innov @ S_inv @ innov)
                if mahal2 < self.p.chi2_gate:
                    cost[i, j] = mahal2

        big = 1e9
        cost_finite = np.where(np.isfinite(cost), cost, big)
        row_ind, col_ind = linear_sum_assignment(cost_finite)

        matched: List[Tuple[int, int]] = []
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < self.p.chi2_gate:
                matched.append((int(i), int(j)))

        matched_d = {i for i, _ in matched}
        matched_t = {j for _, j in matched}
        unmatched_dets = [i for i in range(N) if i not in matched_d]
        unmatched_trks = [j for j in range(M) if j not in matched_t]
        return matched, unmatched_dets, unmatched_trks

    def _cull_dead(self) -> None:
        survivors = []
        for trk in self.tracks:
            if trk.time_since_update <= self.p.max_age:
                survivors.append(trk)
            else:
                self.total_deleted += 1
        self.tracks = survivors

    def _collect_confirmed(self) -> List[EKFTracker3D]:
        out = []
        for trk in self.tracks:
            is_warming = (self.frame_count <= self.p.min_hits)
            is_confirmed = (trk.hits >= self.p.min_hits) or is_warming
            is_fresh = trk.time_since_update == 0
            is_coasting = (trk.hits >= self.p.min_hits and
                           trk.time_since_update <= self.p.max_age)
            if (is_confirmed and is_fresh) or is_coasting:
                out.append(trk)
        return out

    def statistics(self) -> dict:
        return {
            'frame': self.frame_count,
            'active': len(self.tracks),
            'confirmed': sum(1 for t in self.tracks if t.hits >= self.p.min_hits),
            'total_created': self.total_created,
            'total_deleted': self.total_deleted,
        }
