#!/bin/bash
# Campanha: 20 sequências variadas por ambiente, SEM dust (mata visibilidade).
# Resumível: pula prontas (>50 jpgs NÃO-vazios + meta válido). Re-grava corrompidas.
cd /home/ericyos/airsim
ENV_TAG="${1:-nh}"
mkdir -p sessions

# 20 configs: regime weather hour seed ndrones  (sem dust; clear-heavy + fog/rain leves)
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
    echo "SKIP $name ($nvalid frames válidos)"; continue
  fi
  echo "=== REC $name (seed=$seed) ==="
  timeout 15 python3 -c "
import cosysairsim as airsim
c=airsim.MultirotorClient(ip='172.19.80.1',port=41451); c.confirmConnection()
for v in ['Drone3','Drone4','Intruder1']:
    try: c.simDestroyObject(f'VisQuad_{v}')
    except: pass
" 2>/dev/null
  rm -rf "$name"
  python3 record_session.py --duration 18 --out "$name" \
    --regime "$regime" --weather "$weather" --hour "$hour" --seed "$seed" --ndrones "$nd" \
    > "/tmp/camp_$i.log" 2>&1
  echo "  -> $(find $name/rgb -name '*.jpg' -size +0c 2>/dev/null | wc -l) frames válidos"
done
DONE=$(for d in sessions/${ENV_TAG}_seq*; do [ "$(find $d/rgb -name '*.jpg' -size +0c 2>/dev/null|wc -l)" -gt 44 ] && echo x; done | wc -l)
echo "=== $ENV_TAG: $DONE/20 sequências válidas ==="
