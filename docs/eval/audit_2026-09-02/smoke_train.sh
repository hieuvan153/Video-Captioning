#!/bin/bash
# Task 0 Step 3: smoke 60 step voi train_refiner_v5.py da sua (eager). Gate: loss step10 in [1.8,4.0], step60 > 1.0, khong moc < 0.3.
cd /data/ndloc_bk/NLHV/ntVan
set -a; [ -f .env ] && . ./.env; set +a
nohup van_env/bin/python demo/LLM/train_refiner_v5.py \
  --data demo/output/train_v5/llm_data_scene_v5.json \
  --out demo/model_clean/smoke --epochs 1 --max_steps 60 > logs_smoke_clean.log 2>&1 &
echo $! > logs_smoke_clean.pid; echo "started $(cat logs_smoke_clean.pid)"
