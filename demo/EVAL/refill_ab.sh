#!/bin/bash
# Cho vong 1 ket thuc (marker thu 1) roi chay lai de bu 4 phim bs10 bi OOM va cham lai.
cd /data/ndloc_bk/NLHV/ntVan
until [ "$(grep -c ORIG_BATCH_AB_DONE logs_orig_batch_ab.log)" -ge 1 ]; do sleep 30; done
echo "[$(date +%H:%M)] round 1 done -> refill run" >> logs_orig_batch_ab.log
exec bash demo/EVAL/run_orig_batch_ab.sh >> logs_orig_batch_ab.log 2>&1
