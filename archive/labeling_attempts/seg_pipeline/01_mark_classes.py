"""ARCHIVED. Step 1 of the numbered segmentation-labeling chain: assign stencil class IDs.

Part of route (i) of Section 7.2.3, which was abandoned because the pre-built
Unreal environment binaries did not permit setting unique identifiers on the drone
blueprints.
"""

# 01_mark_classes.py
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))  # repo root
from common.config import airsim_host, airsim_port

import re
try:
    import airsim
except Exception:
    import cosysairsim as airsim

IP, PORT = airsim_host(), airsim_port()
CLASS_RULES = [
    ("DRONE", r".*(Drone|Multirotor|Quad|UAV).*", 20),
    ("BIRD",  r".*(Bird|Crow|Pigeon|Seagull|Gull).*", 30),
    ("PLANE", r".*(Plane|Aircraft|Airliner|Jet|Boeing|Airbus).*", 40),
]

cli = airsim.MultirotorClient(ip=IP, port=PORT)
cli.confirmConnection()

# zera todo mundo
cli.simSetSegmentationObjectID(".*", 0, True)

# aplica classes
for label, pattern, seg_id in CLASS_RULES:
    ok = cli.simSetSegmentationObjectID(pattern, seg_id, True)
    print(f"[{label}] -> id={seg_id} aplicado={ok}")

# list a few objects, to inspect their names
objs = cli.simListSceneObjects(".*")
print("Total objetos:", len(objs))
for name in objs[:50]:
    print(" -", name)
