#!/usr/bin/env python3
"""Runtime configuration loader.

Single place where the host/port of the AirSim RPC server, the sensor geometry
and the data locations are read from, so that no script in this repository
hardcodes a machine-specific address or an absolute path.

Resolution order (later wins):

1. ``configs/default.yaml`` next to the repository root;
2. the file named by the ``AIRSIM_UAV_CONFIG`` environment variable;
3. the ``--config`` flag of the calling script;
4. individual command-line flags (``--host``, ``--port``, ...).

Typical use in a script::

    from common.config import CONFIG, add_common_args, apply_common_args

    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    apply_common_args(args)
    client = airsim.MultirotorClient(ip=args.host, port=args.port)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_SETTINGS_PATH = REPO_ROOT / "configs" / "settings.json"

_FALLBACK: Dict[str, Any] = {
    "airsim": {
        "host": "127.0.0.1",
        "port": 41451,
        "ego_vehicle": "Ego",
        "target_vehicles": ["Drone3", "Drone4", "Intruder1"],
        "vehicle_origins": {
            "Ego": [0.0, 0.0, -5.0],
            "Drone3": [15.0, -5.0, -5.0],
            "Drone4": [15.0, 5.0, -5.0],
            "Intruder1": [-15.0, 0.0, -5.0],
        },
    },
    "camera": {
        "width": 1280,
        "height": 720,
        "fov_degrees": 90,
        "offset_body": [0.35, 0.0, -0.5],
        "pitch_degrees": -15.0,
        "min_depth": 0.1,
        "max_depth": 250.0,
    },
    "target": {
        "visual_mesh": "Quadrotor1",
        "visual_mesh_scale": 4.0,
        "extents": [3.01, 3.93, 2.79],
        "rotor_margin": 1.38,
    },
    "paths": {
        "data_root": ".",
        "datasets": "datasets",
        "sessions": "sessions",
        "runs": "runs",
        "dumps": "dumps",
    },
}


def _deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # PyYAML, already a transitive dependency of ultralytics
    except ImportError:  # pragma: no cover - the fallback keeps scripts runnable
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class Config:
    """Dictionary-backed configuration with dotted-path lookup."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = {}
        _deep_update(self._data, _FALLBACK)
        if data:
            _deep_update(self._data, data)

    # -- lookup ----------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, dotted: str) -> Any:
        return self.get(dotted)

    def as_dict(self) -> Dict[str, Any]:
        return self._data

    # -- mutation --------------------------------------------------------
    def load(self, path: os.PathLike | str) -> "Config":
        _deep_update(self._data, _read_yaml(Path(path)))
        return self

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    # -- convenience -----------------------------------------------------
    @property
    def host(self) -> str:
        return str(self.get("airsim.host"))

    @property
    def port(self) -> int:
        return int(self.get("airsim.port"))

    @property
    def data_root(self) -> Path:
        return Path(str(self.get("paths.data_root", "."))).expanduser()

    def path(self, key: str, *parts: str) -> Path:
        """Resolve a named data location, e.g. ``CONFIG.path("sessions")``."""
        base = self.data_root / str(self.get(f"paths.{key}", key))
        return base.joinpath(*parts) if parts else base

    def resolve(self, value: os.PathLike | str) -> Path:
        """Resolve a user-supplied path against ``paths.data_root``."""
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else self.data_root / candidate


def load_config(path: Optional[os.PathLike | str] = None) -> Config:
    cfg = Config()
    if DEFAULT_CONFIG_PATH.exists():
        cfg.load(DEFAULT_CONFIG_PATH)
    env_path = os.environ.get("AIRSIM_UAV_CONFIG")
    if env_path:
        cfg.load(env_path)
    if path:
        cfg.load(path)
    if os.environ.get("AIRSIM_HOST"):
        cfg.set("airsim.host", os.environ["AIRSIM_HOST"])
    if os.environ.get("AIRSIM_PORT"):
        cfg.set("airsim.port", int(os.environ["AIRSIM_PORT"]))
    if os.environ.get("AIRSIM_UAV_DATA_ROOT"):
        cfg.set("paths.data_root", os.environ["AIRSIM_UAV_DATA_ROOT"])
    return cfg


#: Module-level configuration, loaded once at import time.
CONFIG = load_config()


def add_common_args(parser) -> None:
    """Add ``--config``/``--host``/``--port``/``--data-root`` to a parser."""
    parser.add_argument("--config", default=None,
                        help="YAML overriding configs/default.yaml")
    parser.add_argument("--host", default=None,
                        help="AirSim RPC host (default: airsim.host from config)")
    parser.add_argument("--port", type=int, default=None,
                        help="AirSim RPC port (default: airsim.port from config)")
    parser.add_argument("--data-root", default=None,
                        help="Root for datasets/sessions/runs/dumps "
                             "(default: paths.data_root from config)")


def apply_common_args(args) -> Config:
    """Fold the common flags back into ``CONFIG`` and fill in the blanks."""
    if getattr(args, "config", None):
        CONFIG.load(args.config)
    if getattr(args, "data_root", None):
        CONFIG.set("paths.data_root", args.data_root)
    if getattr(args, "host", None):
        CONFIG.set("airsim.host", args.host)
    else:
        args.host = CONFIG.host
    if getattr(args, "port", None):
        CONFIG.set("airsim.port", int(args.port))
    else:
        args.port = CONFIG.port
    return CONFIG


def airsim_host() -> str:
    return CONFIG.host


def airsim_port() -> int:
    return CONFIG.port
