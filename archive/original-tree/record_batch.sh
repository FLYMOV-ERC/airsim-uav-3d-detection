#!/bin/bash
# Grava um batch de sequências variadas p/ avaliação científica.
# Cada sequência é um processo isolado (resiliente a crash do WSL). Pula as já gravadas.
cd /home/ericyos/airsim
ENV_TAG="${1:-nh}"   # tag do ambiente atual aberto no AirSim
mkdir -p sessions

# seq: regime weather hour seed ndrones
CONFIGS=(
  "near clear 13 1 3"
  "mid clear 13 2 3"
  "far clear 13 3 3"
  "mid clear 9 4 3"
  "mid clear 17 5 3"
  "mid fog_light 13 6 3"
  "mid rain_light 13 7 3"
  "near clear 11 8 2"
  "far clear 15 9 3"
)

i=0
for cfg in "${CONFIGS[@]}"; do
  i=$((i+1))
  read regime weather hour seed nd <<< "$cfg"
  name="sessions/${ENV_TAG}_seq$(printf %02d $i)_${regime}_${weather}_h${hour}"
  if [ -f "$name/meta.json" ] && [ "$(ls $name/rgb 2>/dev/null | wc -l)" -gt 50 ]; then
    echo "SKIP $name (já existe)"; continue
  fi
  echo "=== REC $name ==="
  # limpa VisQuad órfão
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
    > "/tmp/rec_$i.log" 2>&1
  echo "  $name: $(ls $name/rgb 2>/dev/null | wc -l) frames"
done
echo "=== BATCH DONE ($ENV_TAG) ==="
ls -d sessions/${ENV_TAG}_* 2>/dev/null
