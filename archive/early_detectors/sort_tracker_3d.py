"""ARCHIVED. 3D SORT (Simple Online Realtime Tracking) for drone tracking.

Adapted from the 2D SORT implementation used for person tracking.
State: [x, y, z, vx, vy, vz] in world NED coordinates.
Measurement: [x, y, z], the 3D position from PointNet (or from ground truth).

Uses:
- a constant-velocity Kalman filter per track
- the Hungarian algorithm for detection-to-track association
- a Euclidean distance cost
- adaptive measurement noise based on the detection confidence
- outlier rejection via the Mahalanobis distance

PASSED OVER in favour of 2D association: Section 7.3.7 states that "performing
the association on the 2D boxes rather than on the noisier 3D positions keeps
identities stable". Kept as the evidence for that design choice.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanTracker3D:
    """
    Kalman filter tracker for a single object in 3D space.

    State vector: [x, y, z, vx, vy, vz]
      - x, y, z: position in NED world frame (meters)
      - vx, vy, vz: velocity (m/s)

    Measurement: [x, y, z] position only.
    Motion model: constant velocity.
    """

    _id_counter = 0

    def __init__(self, position_3d, confidence=1.0, dt=0.1,
                 sigma_acc=2.0, sigma_pos=1.0, outlier_threshold=16.0):
        self.id = KalmanTracker3D._id_counter
        KalmanTracker3D._id_counter += 1

        self.dt = dt
        self.sigma_acc = sigma_acc
        self.sigma_pos = sigma_pos
        self.outlier_threshold = outlier_threshold

        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.age = 0

        # State [x, y, z, vx, vy, vz]
        self.x = np.zeros((6, 1))
        self.x[0, 0] = position_3d[0]
        self.x[1, 0] = position_3d[1]
        self.x[2, 0] = position_3d[2]

        # State covariance - high initial velocity uncertainty
        self.P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])

        # Measurement matrix: observe [x, y, z]
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

    def predict(self, dt=None):
        """Advance state by dt using constant-velocity model."""
        if dt is None:
            dt = self.dt

        # State transition: position += velocity * dt
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Process noise via acceleration
        G = np.zeros((6, 3))
        G[0, 0] = dt ** 2 / 2
        G[1, 1] = dt ** 2 / 2
        G[2, 2] = dt ** 2 / 2
        G[3, 0] = dt
        G[4, 1] = dt
        G[5, 2] = dt

        Q = G @ (self.sigma_acc ** 2 * np.eye(3)) @ G.T

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += 1

        return self.x[:3, 0].copy()

    def update(self, position_3d, confidence=1.0):
        """Update state with a new 3D measurement."""
        # Adaptive measurement noise: lower confidence = higher noise
        adaptive_noise = self.sigma_pos / max(0.3, confidence)
        R = adaptive_noise ** 2 * np.eye(3)

        # Innovation (residual)
        z = np.array(position_3d[:3]).reshape(3, 1)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Outlier rejection via Mahalanobis distance
        mahal = float(y.T @ np.linalg.inv(S) @ y)
        if mahal > self.outlier_threshold:
            return False  # Rejected

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State and covariance update
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        return True

    def get_state(self):
        """Return current position [x, y, z]."""
        return self.x[:3, 0].copy()

    def get_velocity(self):
        """Return current velocity [vx, vy, vz]."""
        return self.x[3:6, 0].copy()

    def get_speed(self):
        """Return scalar speed (m/s)."""
        v = self.get_velocity()
        return float(np.linalg.norm(v))


def _euclidean_distance_matrix(positions_a, positions_b):
    """
    Compute pairwise Euclidean distances between two sets of 3D positions.
    Args:
        positions_a: (N, 3) array
        positions_b: (M, 3) array
    Returns:
        (N, M) distance matrix
    """
    a = np.asarray(positions_a)
    b = np.asarray(positions_b)
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def _associate(detections, tracker_positions, distance_threshold):
    """
    Assign detections to trackers using the Hungarian algorithm.
    Args:
        detections: (N, 4) array [x, y, z, confidence]
        tracker_positions: (M, 3) array [x, y, z]
        distance_threshold: max meters for valid association
    Returns:
        matched_pairs (K, 2), unmatched_det_indices, unmatched_trk_indices
    """
    if len(tracker_positions) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty(0, dtype=int))

    if len(detections) == 0:
        return (np.empty((0, 2), dtype=int),
                np.empty(0, dtype=int),
                np.arange(len(tracker_positions)))

    cost = _euclidean_distance_matrix(detections[:, :3], tracker_positions)
    row_idx, col_idx = linear_sum_assignment(cost)

    matched = []
    unmatched_dets = set(range(len(detections)))
    unmatched_trks = set(range(len(tracker_positions)))

    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= distance_threshold:
            matched.append([r, c])
            unmatched_dets.discard(r)
            unmatched_trks.discard(c)

    matched = np.array(matched, dtype=int).reshape(-1, 2) if matched else np.empty((0, 2), dtype=int)
    return matched, np.array(sorted(unmatched_dets), dtype=int), np.array(sorted(unmatched_trks), dtype=int)


class Sort3D:
    """
    3D multi-object tracker using SORT paradigm.

    Parameters:
        max_age: frames to keep a track alive without detections
        min_hits: minimum consecutive detections to confirm a track
        distance_threshold: max Euclidean distance (meters) for association
        dt: default time step (seconds)
        sigma_acc: acceleration process noise (m/s^2)
        sigma_pos: position measurement noise (m)
        outlier_threshold: Mahalanobis distance threshold for outlier rejection
    """

    def __init__(self, max_age=15, min_hits=2, distance_threshold=5.0,
                 dt=0.1, sigma_acc=3.0, sigma_pos=1.0, outlier_threshold=16.0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.dt = dt
        self.sigma_acc = sigma_acc
        self.sigma_pos = sigma_pos
        self.outlier_threshold = outlier_threshold
        self.trackers: list[KalmanTracker3D] = []
        self.total_tracks_created = 0

    def update(self, detections_3d, dt=None):
        """
        Run one tracking cycle.

        Args:
            detections_3d: (N, 3) or (N, 4) array.
                           Columns: [x, y, z] or [x, y, z, confidence].
            dt: elapsed time since last call (seconds).
        Returns:
            (M, 4) array of active tracks: [x, y, z, track_id].
        """
        if dt is None:
            dt = self.dt

        # Normalize input
        if len(detections_3d) > 0:
            detections_3d = np.atleast_2d(detections_3d)
            if detections_3d.shape[1] == 3:
                detections_3d = np.hstack([detections_3d, np.ones((len(detections_3d), 1))])
        else:
            detections_3d = np.empty((0, 4))

        # 1. Predict all existing trackers
        predicted_positions = []
        for trk in self.trackers:
            pos = trk.predict(dt)
            predicted_positions.append(pos)
        predicted_positions = np.array(predicted_positions) if predicted_positions else np.empty((0, 3))

        # 2. Associate detections ↔ trackers (Hungarian)
        matched, unmatched_dets, unmatched_trks = _associate(
            detections_3d, predicted_positions, self.distance_threshold
        )

        # 3. Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(
                detections_3d[det_idx, :3],
                confidence=detections_3d[det_idx, 3],
            )

        # 4. Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            trk = KalmanTracker3D(
                detections_3d[det_idx, :3],
                confidence=detections_3d[det_idx, 3],
                dt=dt,
                sigma_acc=self.sigma_acc,
                sigma_pos=self.sigma_pos,
                outlier_threshold=self.outlier_threshold,
            )
            self.trackers.append(trk)
            self.total_tracks_created += 1

        # 5. Mark unmatched trackers
        for trk_idx in unmatched_trks:
            self.trackers[trk_idx].time_since_update += 1
            self.trackers[trk_idx].hit_streak = 0

        # 6. Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # 7. Return confirmed tracks
        results = []
        for trk in self.trackers:
            if trk.hit_streak >= self.min_hits or trk.age <= self.min_hits:
                pos = trk.get_state()
                results.append([pos[0], pos[1], pos[2], float(trk.id)])

        return np.array(results) if results else np.empty((0, 4))

    def get_tracker_by_id(self, track_id):
        """Retrieve a KalmanTracker3D by its ID."""
        for trk in self.trackers:
            if trk.id == track_id:
                return trk
        return None

    def get_all_active_ids(self):
        """Return list of all active tracker IDs."""
        return [trk.id for trk in self.trackers]
