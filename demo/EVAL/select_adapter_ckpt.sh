#!/bin/bash
# Task 4 Step 3 E0: cham moi checkpoint adapter sach tren 9 phim val (eval_ab), arm baseline, prompt v4.
# Dung: demo/EVAL/select_adapter_ckpt.sh [ckpt_root=demo/model_clean/adapter_clean]
cd /data/ndloc_bk/NLHV/ntVan
ROOT=${1:-demo/model_clean/adapter_clean}
VAL="movie_001 movie_008 movie_009 movie_015 movie_018 movie_045 movie_046 movie_059 movie_189"
free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for ck in $(ls -d $ROOT/checkpoint-* | sort -t- -k2 -n); do
  n=$(basename $ck | cut -d- -f2)
  while [ "$(free)" -lt 18000 ]; do echo "[wait_gpu] $(date +%H:%M)"; sleep 120; done
  van_env/bin/python demo/EVAL/run_arms.py --eval_dir demo/output/eval_ab --movies $VAL \
    --arms baseline --prefix clean_ck${n}_ --adapter $ck --prompt_style v4 \
    --log demo/output/eval_ab/select_ckpt.log
done
ARMS=$(for ck in $(ls -d $ROOT/checkpoint-*); do echo -n "clean_ck$(basename $ck | cut -d- -f2)_baseline.srt "; done)
van_env/bin/python demo/EVAL/thesis_score.py --eval_dir demo/output/eval_ab --movies $VAL --arms $ARMS --no_comet \
  --report docs/eval/clean_select_ckpt.json
echo SELECT_CKPT_DONE
