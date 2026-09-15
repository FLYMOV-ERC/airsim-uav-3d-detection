#!/usr/bin/env python3
"""Optional import of the AirSim Python client (`cosysairsim`).

The client is needed only to talk to a running simulator: dataset collection,
campaign recording, and the live pipeline.  Everything downstream of a recorded
session -- building the point-cloud cache, training and dumping the detectors,
fusing dumps, scoring metrics -- reads files from disk and needs no simulator,
yet those modules import `common.pipeline` for the camera model and the frame
conversions.  An unguarded `import cosysairsim` at the top of that module would
make the whole offline half of the repository unimportable without the client.

So the import is attempted here once, and its absence is reported only at the
point where a simulator connection is actually required:

    from common.airsim_optional import airsim, ImageResponse, require_airsim
    ...
    def run(self):
        require_airsim()
        client = airsim.MultirotorClient(...)

`airsim` and `ImageResponse` are None when the client is not installed.
"""
from __future__ import annotations

try:
    import cosysairsim as airsim
    from cosysairsim.types import ImageResponse
except ImportError as exc:                     # pragma: no cover - environment dependent
    airsim = None
    ImageResponse = None
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None


AIRSIM_AVAILABLE = airsim is not None


def require_airsim() -> None:
    """Raise if a code path that needs a live simulator is reached without the client."""
    if airsim is None:
        raise ImportError(
            "cosysairsim is not installed, and this code path needs a live AirSim "
            "connection.  Install it with `pip install cosysairsim`.  Offline paths "
            "-- pc_cache, the voxel and frustum trainers, dump_voxel, fuse_dumps, "
            "metrics -- do not need it."
        ) from _IMPORT_ERROR
