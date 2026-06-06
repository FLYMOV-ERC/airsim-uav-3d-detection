# 01_mark_classes.py
import re
try:
    import airsim
except Exception:
    import cosysairsim as airsim

IP, PORT = "172.19.80.1", 41451
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

# liste alguns objetos p/ inspecionar nomes
objs = cli.simListSceneObjects(".*")
print("Total objetos:", len(objs))
for name in objs[:50]:
    print(" -", name)
