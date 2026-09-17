#!/usr/bin/env python3
"""Anexa um Quadrotor1 visual (4x maior) a cada multirotor AirSim.

IMPORTANTE: simGetVehiclePose retorna pose RELATIVA à origem do veículo
(declarada em settings.json X/Y/Z). Mas simSetObjectPose espera GLOBAL.
Por isso somamos vehicle_origin antes de setObjectPose, senão visual fica deslocado.
"""
from __future__ import annotations
import threading
import time
from typing import List, Dict, Optional
import cosysairsim as airsim


# Origens dos veículos do settings.json (X, Y, Z em NED)
VEHICLE_ORIGINS = {
    'Ego': (0.0, 0.0, -5.0),
    'Drone3': (15.0, -5.0, -5.0),
    'Drone4': (15.0, 5.0, -5.0),
    'Intruder1': (-15.0, 0.0, -5.0),
}


def vehicle_to_global(client, vehicle_name):
    """Converte vehicle_pose (relative-to-origin) → global pose somando origin."""
    vp = client.simGetVehiclePose(vehicle_name)
    ox, oy, oz = VEHICLE_ORIGINS.get(vehicle_name, (0.0, 0.0, 0.0))
    return airsim.Pose(
        airsim.Vector3r(vp.position.x_val + ox, vp.position.y_val + oy, vp.position.z_val + oz),
        vp.orientation,
    )


class DroneVisualizer:
    """Sincroniza objetos Quadrotor1 visuais com pose de multirotors."""

    VISUAL_PREFIX = "VisQuad"
    VISUAL_ASSET = "Quadrotor1"

    def __init__(self, client, drones: List[str], scale: float = 4.0,
                 sync_hz: float = 30.0):
        self.client = client
        self.drones = list(drones)
        self.scale = float(scale)
        self.sync_dt = 1.0 / float(sync_hz)
        self._visual_names: Dict[str, str] = {}   # drone_name → spawned obj name
        self._stop = threading.Event()
        self._thread = None

    # ─────────────────────────────────────────────────────────────────────
    def attach(self) -> Dict[str, str]:
        """Spawna 1 Quadrotor1 por drone na pose GLOBAL atual."""
        scale_v = airsim.Vector3r(self.scale, self.scale, self.scale)
        for d in self.drones:
            try:
                pose = vehicle_to_global(self.client, d)
            except Exception as e:
                print(f"[DroneVisualizer] skip {d}: {e}")
                continue
            vis_name = f"{self.VISUAL_PREFIX}_{d}"
            try:
                self.client.simDestroyObject(vis_name)
            except Exception:
                pass
            try:
                actual = self.client.simSpawnObject(
                    vis_name, self.VISUAL_ASSET, pose, scale_v,
                    physics_enabled=False, is_blueprint=False,
                )
                self._visual_names[d] = actual
                print(f"[DroneVisualizer] {d} -> {actual} scale={self.scale}x")
            except Exception as e:
                print(f"[DroneVisualizer] spawn {d} erro: {e}")
        return self._visual_names

    def detach(self) -> None:
        for d, name in self._visual_names.items():
            try:
                self.client.simDestroyObject(name)
            except Exception as e:
                print(f"[DroneVisualizer] destroy {name} erro: {e}")
        self._visual_names.clear()

    # ─────────────────────────────────────────────────────────────────────
    def sync_once(self) -> None:
        """Atualiza pose dos visuais. Bloqueante. Usa pose GLOBAL (vehicle + origin)."""
        for d, vis_name in list(self._visual_names.items()):
            try:
                pose = vehicle_to_global(self.client, d)
                self.client.simSetObjectPose(vis_name, pose, True)
            except Exception:
                pass

    def start_sync(self) -> None:
        """Inicia thread daemon que faz sync periodicamente."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="DroneVizSync")
        self._thread.start()
        print(f"[DroneVisualizer] sync thread started ({1.0/self.sync_dt:.0f}Hz)")

    def stop_sync(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self):
        import asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())
        # Cliente próprio pra essa thread (msgpack-rpc não compartilha)
        try:
            sub_client = airsim.MultirotorClient(
                ip=getattr(self.client, "ip", "172.19.80.1"),
                port=getattr(self.client, "port", 41451),
            )
            sub_client.confirmConnection()
        except Exception as e:
            print(f"[DroneVisualizer] sub_client erro: {e}")
            return
        while not self._stop.is_set():
            for d, vis_name in list(self._visual_names.items()):
                try:
                    pose = vehicle_to_global(sub_client, d)
                    sub_client.simSetObjectPose(vis_name, pose, True)
                except Exception:
                    pass
            self._stop.wait(self.sync_dt)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """Teste isolado: anexa visuais nos drones do AirSim e fica sincronizando."""
    import argparse, signal, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="172.19.80.1")
    ap.add_argument("--port", type=int, default=41451)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--drones", nargs="+", default=["Drone3", "Drone4", "Intruder1"])
    ap.add_argument("--duration", type=float, default=60.0)
    args = ap.parse_args()

    client = airsim.MultirotorClient(ip=args.ip, port=args.port)
    client.confirmConnection()
    viz = DroneVisualizer(client, args.drones, scale=args.scale, sync_hz=30)
    viz.attach()
    viz.start_sync()
    print(f"Visuals on. Press Ctrl+C to stop (or wait {args.duration}s).")
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        viz.stop_sync()
        viz.detach()
        print("Detached.")
