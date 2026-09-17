#!/bin/bash
# Roda eval científico (baseline vs best) por ambiente. Resumível (cache por seq).
cd /home/ericyos/airsim
for env in nh city coast; do
  ls -d sessions/${env}_seq* >/dev/null 2>&1 || { echo "SKIP $env (sem sessões)"; continue; }
  echo "######## ENV=$env ########"
  python3 eval_scientific.py --tag "BASE_${env}" --sessions "sessions/${env}_seq*" \
    --pointnet runs/pointnet2_frustum/v4/best.pt --conf 0.25 --max_age 8 --min_hits 3 \
    > /tmp/eval_BASE_${env}.log 2>&1
  echo "  BASE_$env done (rc=$?)"
  python3 eval_scientific.py --tag "BEST_${env}" --sessions "sessions/${env}_seq*" \
    --pointnet runs/pointnet2_frustum/v5band/best.pt --det_band 10 --conf 0.10 --max_age 15 --min_hits 3 \
    > /tmp/eval_BEST_${env}.log 2>&1
  echo "  BEST_$env done (rc=$?)"
done
echo "==== TODOS OS EVALS CONCLUÍDOS ===="
