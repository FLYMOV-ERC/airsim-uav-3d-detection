#!/bin/bash
# The 60-sequence evaluation campaign of Table 7.1: 20 varied sequences per environment.
#
# Run once per environment, with that environment open in AirSim:
#
#     bash dataset_generation/record_campaign.sh nh      # AirSimNH
#     bash dataset_generation/record_campaign.sh city    # CityEnviron
#     bash dataset_generation/record_campaign.sh coast   # Coastline
#
# Each sequence is recorded by dataset_generation/record_campaign.py in its own process
# (resilient to WSL crashes). Resumable: a sequence with more than 44 NON-empty JPEGs and a
# parseable meta.json is skipped; anything else is treated as corrupt and re-recorded.
#
# No dust: it kills visibility. The mix is clear-heavy, with light fog and light rain.
# The configuration list below is the one the campaign actually ran, kept verbatim.
cd "$(dirname "$0")/.." || exit 1
ENV_TAG="${1:-nh}"
mkdir -p sessions

# 20 configs: regime weather hour seed ndrones
CONFIGS=(
  "near clear 13 1 3"   "mid clear 13 2 3"    "far clear 13 3 3"
  "mid clear 9 4 3"     "mid clear 17 5 3"    "near clear 7 6 3"
  "far clear 15 7 3"    "mid clear 11 8 2"    "near clear 18 9 3"
  "far clear 9 10 3"    "mid fog_light 13 11 3" "near clear 15 12 3"
  "far clear 18 13 3"   "mid rain_light 13 14 3" "near clear 11 15 2"
  "far clear 7 16 3"    "mid clear 15 17 3"   "near rain_light 13 18 3"
  "far fog_light 13 19 3" "mid clear 18 20 3"
)
i=0
for cfg in "${CONFIGS[@]}"; do
  i=$((i+1)); read regime weather hour seed nd <<< "$cfg"
  name="sessions/${ENV_TAG}_seq$(printf %02d $i)_${regime}_${weather}_h${hour}"
  nvalid=$(find "$name/rgb" -name '*.jpg' -size +0c 2>/dev/null | wc -l)
  if [ "$nvalid" -gt 44 ] && python3 -c "import json;json.load(open('$name/meta.json'))" 2>/dev/null; then
    echo "SKIP $name ($nvalid valid frames)"; continue
  fi
  echo "=== REC $name (seed=$seed) ==="
  # remove VisQuad meshes orphaned by a crashed previous run
  timeout 15 python3 -c "
import sys; sys.path.insert(0, '.')
from common.config import airsim_host, airsim_port
import cosysairsim as airsim
c=airsim.MultirotorClient(ip=airsim_host(),port=airsim_port()); c.confirmConnection()
for v in ['Drone3','Drone4','Intruder1']:
    try: c.simDestroyObject(f'VisQuad_{v}')
    except: pass
" 2>/dev/null
  rm -rf "$name"
  python3 dataset_generation/record_campaign.py --duration 18 --out "$name" \
    --regime "$regime" --weather "$weather" --hour "$hour" --seed "$seed" --ndrones "$nd" \
    > "/tmp/camp_$i.log" 2>&1
  echo "  -> $(find $name/rgb -name '*.jpg' -size +0c 2>/dev/null | wc -l) valid frames"
done
DONE=$(for d in sessions/${ENV_TAG}_seq*; do [ "$(find $d/rgb -name '*.jpg' -size +0c 2>/dev/null|wc -l)" -gt 44 ] && echo x; done | wc -l)
echo "=== $ENV_TAG: $DONE/20 valid sequences ==="
