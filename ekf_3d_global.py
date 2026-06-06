#!/usr/bin/env python3
"""EKF 3D Tracker single-target — porta do ekf_filter.m (MATLAB).

Estado do alvo (mundo NED):
    x_OBJ = [x, y, z, vx, vy, vz, r] ∈ R^7

Estado da plataforma (vindo do AirSim):
    x_PLAT = [x_p, y_p, z_p, vx_p, vy_p, vz_p, phi_p, theta_p, psi_p] ∈ R^9

Medição esférica (frame câmera FRD, assumindo câmera = plataforma):
    y = [d, phi, theta, r] ∈ R^4
        d     = distância euclidiana
        phi   = azimute  = atan2(y_c, x_c)
        theta = elevação = atan2(z_c, sqrt(x_c^2 + y_c^2))
        r     = raio (passado direto do estado)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Utilities (rotação, conversões esférica/cartesiana)
# ─────────────────────────────────────────────────────────────────────────────

def rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """R_yaw(psi) @ R_pitch(theta) @ R_roll(phi).  Rotação mundo → câmera."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    R_roll  = np.array([[1, 0, 0], [0, cphi, -sphi], [0, sphi, cphi]])
    R_pitch = np.array([[cth, 0, sth], [0, 1, 0], [-sth, 0, cth]])
    R_yaw   = np.array([[cpsi, -spsi, 0], [spsi, cpsi, 0], [0, 0, 1]])
    return R_yaw @ R_pitch @ R_roll


def wrap_angle(a: float) -> float:
    """Wrap rad ∈ [-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def spherical_from_cartesian(p_c: np.ndarray) -> Tuple[float, float, float]:
    x_c, y_c, z_c = p_c
    d = float(np.sqrt(x_c * x_c + y_c * y_c + z_c * z_c))
    phi = float(np.arctan2(y_c, x_c))
    theta = float(np.arctan2(z_c, np.sqrt(x_c * x_c + y_c * y_c)))
    return d, phi, theta


def cartesian_from_spherical(d: float, phi: float, theta: float) -> np.ndarray:
    x = d * np.cos(theta) * np.cos(phi)
    y = d * np.cos(theta) * np.sin(phi)
    z = d * np.sin(theta)
    return np.array([x, y, z])


def measurement_to_global(meas: np.ndarray, x_plat: np.ndarray) -> np.ndarray:
    """Medição esférica + pose plataforma → posição global no mundo NED."""
    d, phi, theta, _ = meas
    p_c = cartesian_from_spherical(d, phi, theta)
    R_m = rotation_matrix(x_plat[6], x_plat[7], x_plat[8])
    return x_plat[:3] + R_m.T @ p_c


# ─────────────────────────────────────────────────────────────────────────────
# Parâmetros
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EKFParams:
    # Processo (random walk em aceleração)
    sigma_ax: float = 0.5
    sigma_ay: float = 0.5
    sigma_az: float = 0.5
    sigma_r_proc: float = 0.05

    # Medição
    sigma_phi: float = 0.005
    sigma_theta: float = 0.005
    sigma_r_meas: float = 0.5
    alpha_distance: float = 0.05
    sigma_d_min: float = 0.5

    # Inicialização
    sigma_pos_init: float = 5.0
    sigma_vel_init: float = 10.0
    sigma_r_init: float = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# EKFTracker3D
# ─────────────────────────────────────────────────────────────────────────────

class EKFTracker3D:
    """EKF 3D para um único alvo com medição esférica."""

    _id_counter = 0

    def __init__(self,
                 initial_meas: np.ndarray,
                 initial_x_plat: np.ndarray,
                 params: Optional[EKFParams] = None,
                 track_id: Optional[int] = None):
        self.p = params if params is not None else EKFParams()

        if track_id is None:
            EKFTracker3D._id_counter += 1
            self.id = EKFTracker3D._id_counter
        else:
            self.id = track_id

        d0, phi0, theta0, r0 = initial_meas
        p_global = measurement_to_global(initial_meas, initial_x_plat)
        self.x = np.zeros(7)
        self.x[0:3] = p_global
        self.x[3:6] = 0.0
        self.x[6] = max(r0, 0.1)

        self.P = np.diag([
            self.p.sigma_pos_init ** 2,
            self.p.sigma_pos_init ** 2,
            self.p.sigma_pos_init ** 2,
            self.p.sigma_vel_init ** 2,
            self.p.sigma_vel_init ** 2,
            self.p.sigma_vel_init ** 2,
            self.p.sigma_r_init ** 2,
        ])

        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0
        self.last_confidence = 1.0
        self.history: List[np.ndarray] = [p_global.copy()]

    # ─── Process model ────────────────────────────────────────────────────
    def build_F(self, dt: float) -> np.ndarray:
        F = np.eye(7)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def build_Q(self, dt: float) -> np.ndarray:
        Q = np.zeros((7, 7))
        for i, sigma_a in enumerate([self.p.sigma_ax, self.p.sigma_ay, self.p.sigma_az]):
            ipos = i
            ivel = i + 3
            q11 = sigma_a ** 2 * (dt ** 3) / 3.0
            q12 = sigma_a ** 2 * (dt ** 2) / 2.0
            q22 = sigma_a ** 2 * dt
            Q[ipos, ipos] = q11
            Q[ipos, ivel] = q12
            Q[ivel, ipos] = q12
            Q[ivel, ivel] = q22
        Q[6, 6] = self.p.sigma_r_proc ** 2
        return Q

    def predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        F = self.build_F(dt)
        Q = self.build_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 1:
            self.hit_streak = 0
        return self.x.copy(), self.P.copy()

    # ─── Measurement model ────────────────────────────────────────────────
    def compute_h(self, x_obj: np.ndarray, x_plat: np.ndarray) -> np.ndarray:
        R_m = rotation_matrix(x_plat[6], x_plat[7], x_plat[8])
        p_c = R_m @ (x_obj[:3] - x_plat[:3])
        d, phi, theta = spherical_from_cartesian(p_c)
        return np.array([d, phi, theta, x_obj[6]])

    def compute_H(self, x_obj: np.ndarray, x_plat: np.ndarray) -> np.ndarray:
        R_m = rotation_matrix(x_plat[6], x_plat[7], x_plat[8])
        p_c = R_m @ (x_obj[:3] - x_plat[:3])
        x_c, y_c, z_c = p_c
        eps = 1e-6
        d = float(np.sqrt(x_c * x_c + y_c * y_c + z_c * z_c))
        rho = float(np.sqrt(x_c * x_c + y_c * y_c))
        d_s = max(d, eps)
        rho_s = max(rho, eps)
        rho2 = max(x_c * x_c + y_c * y_c, eps * eps)
        d2 = max(d * d, eps * eps)

        dd_dpc = np.array([x_c / d_s, y_c / d_s, z_c / d_s])
        dphi_dpc = np.array([-y_c / rho2, x_c / rho2, 0.0])
        dtheta_dpc = np.array([
            -x_c * z_c / (d2 * rho_s),
            -y_c * z_c / (d2 * rho_s),
            rho_s / d2,
        ])

        H = np.zeros((4, 7))
        H[0, 0:3] = dd_dpc @ R_m
        H[1, 0:3] = dphi_dpc @ R_m
        H[2, 0:3] = dtheta_dpc @ R_m
        H[3, 6] = 1.0
        return H

    def build_R(self, d_pred: float) -> np.ndarray:
        sigma_d = max(self.p.alpha_distance * d_pred, self.p.sigma_d_min)
        return np.diag([
            sigma_d ** 2,
            self.p.sigma_phi ** 2,
            self.p.sigma_theta ** 2,
            self.p.sigma_r_meas ** 2,
        ])

    def innovation(self, y_meas: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        innov = y_meas - y_pred
        innov[1] = wrap_angle(innov[1])
        innov[2] = wrap_angle(innov[2])
        return innov

    # ─── Update ───────────────────────────────────────────────────────────
    def update(self, y_meas: np.ndarray, x_plat: np.ndarray,
               confidence: float = 1.0) -> None:
        y_pred = self.compute_h(self.x, x_plat)
        H = self.compute_H(self.x, x_plat)
        R = self.build_R(y_pred[0])
        innov = self.innovation(y_meas, y_pred)
        S = H @ self.P @ H.T + R

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ innov

        I = np.eye(7)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.last_confidence = float(confidence)
        self.history.append(self.x[:3].copy())
        if len(self.history) > 500:
            self.history = self.history[-500:]

    # ─── Accessors ────────────────────────────────────────────────────────
    def position_global(self) -> np.ndarray:
        return self.x[0:3].copy()

    def velocity_global(self) -> np.ndarray:
        return self.x[3:6].copy()

    def radius(self) -> float:
        return float(self.x[6])

    def covariance_diag(self) -> np.ndarray:
        return np.diag(self.P).copy()

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'position': self.position_global().tolist(),
            'velocity': self.velocity_global().tolist(),
            'radius': self.radius(),
            'P_diag': self.covariance_diag().tolist(),
            'hits': self.hits,
            'hit_streak': self.hit_streak,
            'age': self.age,
            'time_since_update': self.time_since_update,
            'confidence': self.last_confidence,
        }
