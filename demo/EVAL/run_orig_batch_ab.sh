#!/bin/bash
# A/B batch size tren tap E5 (co ground truth) voi refine_llm.py GOC cua anh ntVan:
# bs10 = mac dinh run_pipeline.py, bs1 = tung prompt mot. Output: demo/output/eval_e5/<m>/orig_bs{10,1}.srt
cd /data/ndloc_bk/NLHV/ntVan
PY=/data/ndloc_bk/ntVan/demo_env/bin/python3; ORIG=/data/ndloc_bk/ntVan/demo/LLM/refine_llm.py
E5="movie_054 movie_081 movie_090 movie_104 movie_124 movie_170 movie_291 movie_311 movie_312 movie_336"
export PYTHONUNBUFFERED=1
for bs in 10 1; do for m in $E5; do
  d=demo/output/eval_e5/$m; out=$d/orig_bs$bs.srt
  [ -s "$out" ] && { echo "skip $out"; continue; }
  echo "[$(date +%H:%M)] $m bs=$bs"
  demo/EVAL/orig_refine_cli.sh --en_srt $PWD/$d/en.srt --vinai_srt $PWD/$d/rough.srt --vlm_json $PWD/$d/captions.json --output_srt $PWD/$out --llm_batch_size $bs > $d/orig_bs$bs.log 2>&1 || echo "FAIL $m bs=$bs rc=$?"
done; done
echo "[$(date +%H:%M)] thesis_score"
van_env/bin/python demo/EVAL/thesis_score.py --eval_dir demo/output/eval_e5 --arms rough.srt orig_bs10.srt orig_bs1.srt --report docs/eval/orig_batch_ab.json > logs_orig_batch_ab_score.log 2>&1
echo "ORIG_BATCH_AB_DONE"
