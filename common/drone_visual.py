#!/usr/bin/env python3
"""Attach an enlarged visual quadrotor mesh ("VisQuad") to each AirSim multirotor.

The native multirotor mesh has a sub-metre envelope and is neither detectable by
the 2D network nor densely sampled by the depth camera at 20-100 m.  Each target
therefore carries a rigidly attached ``Quadrotor1`` mesh scaled 4x in every linear
dimension, pose-synchronised to its carrier at 30 Hz.  This is the visual-mesh
augmentation of Section 7.2.2; the 2D detector is trained on it and assumes the
same mesh at inference, so it must be attached both when collecting data and when
running the pipeline live.

One AirSim asymmetry has to be corrected here: ``simGetVehiclePose`` returns a
pose RELATIVE to the vehicle's spawn origin (the X/Y/Z fields of settings.json),
whereas ``simSetObjectPose`` expects a GLOBAL pose.  ``vehicle_to_global`` adds
the origin back; without it the visual mesh is displaced from its carrier by the
spawn offset.
"""
from __future__ import annotations
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # repo root
from common.config import CONFIG, airsim_host, airsim_port

import threading
import time
from typing import List, Dict, Optional
# Optional: attaching a visual mesh needs a live simulator, but this module is
# reachable from common/pipeline.py, which the offline half imports.
from common.airsim_optional import airsim, require_airsim


# Vehicle spawn origins (X, Y, Z in NED metres), mirroring the "Vehicles" block
# of configs/settings.json.  Override them in configs/default.yaml, not here.
VEHICLE_ORIGINS = {
    name: tuple(float(v) for v in origin)
    for name, origin in CONFIG.get("airsim.vehicle_origins", {}).items()
}


def vehicle_to_global(client, vehicle_name):
    """Vehicle pose (relative to spawn origin) -> global pose, by adding the origin."""
    require_airsim()
    vp = client.simGetVehiclePose(vehicle_name)
    ox, oy, oz = VEHICLE_ORIGINS.get(vehicle_name, (0.0, 0.0, 0.0))
    return airsim.Pose(
        airsim.Vector3r(vp.position.x_val + ox, vp.position.y_val + oy, vp.position.z_val + oz),
        vp.orientation,
    )


class DroneVisualizer:
    """Keeps one spawned visual mesh per multirotor in sync with its carrier's pose."""

    VISUAL_PREFIX = "VisQuad"
    VISUAL_ASSET = str(CONFIG.get("target.visual_mesh", "Quadrotor1"))
    DEFAULT_SCALE = float(CONFIG.get("target.visual_mesh_scale", 4.0))

    def __init__(self, client, drones: List[str], scale: float = None,
                 sync_hz: float = 30.0):
        self.client = client
        self.drones = list(drones)
        self.scale = float(self.DEFAULT_SCALE if scale is None else scale)
        self.sync_dt = 1.0 / float(sync_hz)
        self._visual_names: Dict[str, str] = {}   # drone name -> spawned object name
        self._stop = threading.Event()
        self._thread = None

    # ─────────────────────────────────────────────────────────────────────
    def attach(self) -> Dict[str, str]:
        """Spawn one visual mesh per drone at that drone's current GLOBAL pose."""
        require_airsim()
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
                print(f"[DroneVisualizer] spawn {d} failed: {e}")
        return self._visual_names

    def detach(self) -> None:
        for d, name in self._visual_names.items():
            try:
                self.client.simDestroyObject(name)
            except Exception as e:
                print(f"[DroneVisualizer] destroy {name} failed: {e}")
        self._visual_names.clear()

    # ─────────────────────────────────────────────────────────────────────
    def sync_once(self) -> None:
        """Update the visual poses once, blocking. Uses the GLOBAL pose (vehicle + origin)."""
        for d, vis_name in list(self._visual_names.items()):
            try:
                pose = vehicle_to_global(self.client, d)
                self.client.simSetObjectPose(vis_name, pose, True)
            except Exception:
                pass

    def start_sync(self) -> None:
        """Start the daemon thread that re-mirrors the poses periodically."""
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
        # This thread needs its own client: msgpack-rpc connections are not shareable.
        try:
            sub_client = airsim.MultirotorClient(
                ip=getattr(self.client, "ip", airsim_host()),
                port=getattr(self.client, "port", airsim_port()),
            )
            sub_client.confirmConnection()
        except Exception as e:
            print(f"[DroneVisualizer] sub_client failed: {e}")
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
    """Standalone check: attach the visual meshes and keep them synchronised."""
    import argparse, signal, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=airsim_host())
    ap.add_argument("--port", type=int, default=airsim_port())
    ap.add_argument("--scale", type=float, default=DroneVisualizer.DEFAULT_SCALE)
    ap.add_argument("--drones", nargs="+",
                    default=list(CONFIG.get("airsim.target_vehicles", [])))
    ap.add_argument("--duration", type=float, default=60.0)
    args = ap.parse_args()

    require_airsim()
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
