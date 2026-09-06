#!/bin/bash
# Cho tien trinh Gemma bs1 cua phim (PID $1) ket thuc roi chay lai E5 A/B (script tu bo qua phim da co output).
cd /data/ndloc_bk/NLHV/ntVan
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M)] film bs1 PID $1 gone -> relaunch run_orig_batch_ab.sh" >> logs_orig_batch_ab.log
exec bash demo/EVAL/run_orig_batch_ab.sh >> logs_orig_batch_ab.log 2>&1
