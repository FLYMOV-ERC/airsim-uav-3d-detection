#!/usr/bin/env bash
# Monitora results.csv do YOLO em busca de overfitting:
#   - Imprime cada epoch novo com train_loss, val_loss, mAP50
#   - Alerta se val_loss subir 3 epochs seguidos OU mAP50 cair 3 seguidos
CSV="runs/drone/drone_synth_v1/results.csv"
LAST_EPOCH=0
declare -a VAL_LOSSES MAPS
N_VAL_UP=0
N_MAP_DOWN=0

until [ -f "$CSV" ]; do sleep 5; done
echo "Monitor armado: $CSV"

while true; do
    # Pega última linha (header é linha 1)
    LINE=$(tail -1 "$CSV" 2>/dev/null)
    [ -z "$LINE" ] && sleep 5 && continue
    EPOCH=$(echo "$LINE" | awk -F',' '{print $1}' | tr -d ' ')
    [ "$EPOCH" = "epoch" ] && sleep 5 && continue
    [ "$EPOCH" -le "$LAST_EPOCH" ] && sleep 10 && continue
    LAST_EPOCH=$EPOCH
    # Colunas results.csv YOLO11:
    # epoch, time, train/box_loss, train/cls_loss, train/dfl_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B), val/box_loss, val/cls_loss, val/dfl_loss, lr/pg0...
    # Header pode variar — pega pelo nome
    HEADER=$(head -1 "$CSV")
    # Vou usar python pra parsing robusto
    python3 -c "
import csv, sys
rows = list(csv.DictReader(open('$CSV')))
if not rows: sys.exit()
r = rows[-1]
def g(k):
    for kk in r:
        if k in kk: return float(r[kk])
    return -1
ep = int(g('epoch'))
tr_box = g('train/box_loss')
tr_cls = g('train/cls_loss')
va_box = g('val/box_loss')
va_cls = g('val/cls_loss')
p = g('metrics/precision')
rec = g('metrics/recall')
map50 = g('metrics/mAP50(')
map95 = g('metrics/mAP50-95')
# Trend
val_losses = [(float(r2['        val/box_loss']) if '        val/box_loss' in r2 else next((float(r2[k]) for k in r2 if 'val/box_loss' in k), 0)) for r2 in rows]
maps = [(next((float(r2[k]) for k in r2 if 'mAP50(' in k), 0)) for r2 in rows]
flag = ''
if len(val_losses) >= 3:
    last3 = val_losses[-3:]
    if last3[2] > last3[1] > last3[0]:
        flag += ' [WARN: val_box_loss subiu 3 epochs]'
if len(maps) >= 3:
    last3m = maps[-3:]
    if last3m[2] < last3m[1] < last3m[0]:
        flag += ' [WARN: mAP50 caiu 3 epochs]'
gap = tr_box - va_box
gap_pct = abs(gap) / max(va_box, 0.01) * 100
if gap_pct > 30 and va_box > tr_box:
    flag += f' [WARN: val/train gap {gap_pct:.0f}%]'
print(f'E{ep:3d}: train_box={tr_box:.3f} val_box={va_box:.3f}  mAP50={map50:.3f} mAP95={map95:.3f}  P={p:.3f} R={rec:.3f}{flag}')
"
    sleep 30
done
