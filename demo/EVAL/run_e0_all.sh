#!/bin/bash
# Dieu phoi E0 sau khi 3 model sach train xong: chon checkpoint adapter (Task 4.3), convert Whisper (Task 2),
# eval_clean 13 phim (Task 5), cac hang phu (Task 6), cham diem + bootstrap (Task 7, ket qua tho -> docs/eval/clean_eval_raw/).
# Idempotent: buoc nao co san ket qua thi bo qua. Chay: nohup demo/EVAL/run_e0_all.sh > logs_e0_all.log 2>&1 &
cd /data/ndloc_bk/NLHV/ntVan
set -a; [ -f .env ] && . ./.env; set +a
export PYTHONUNBUFFERED=1
PY=van_env/bin/python; RAW=docs/eval/clean_eval_raw; mkdir -p $RAW
E5="movie_054 movie_081 movie_090 movie_104 movie_124 movie_170 movie_291 movie_311 movie_312 movie_336"
EOUT="movie_002 movie_005 movie_006"
SUB=/data/ndloc_bk/ntVan/data/Movie/sub
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
waitfor() { while ! grep -q "$2" "$1" 2>/dev/null; do sleep 300; done; log "co $2 trong $1"; }

# --- 1. adapter: doi train xong, chon checkpoint tren val
waitfor logs_train_clean.log RUN_TRAIN_CLEAN_EXIT
grep -q LOSS_COLLAPSE logs_train_clean.log && { log "ADAPTER LOSS_COLLAPSE - dung"; exit 1; }
if [ ! -f docs/eval/clean_select_ckpt.json ]; then
  log "chon checkpoint adapter"; sed -i 's#docs/eval/clean_select_ckpt.md#docs/eval/clean_select_ckpt.json#' demo/EVAL/select_adapter_ckpt.sh
  demo/EVAL/select_adapter_ckpt.sh > logs_select_ckpt.log 2>&1
fi
BEST=$($PY - <<'PYEOF'
import json,glob,os
r=json.load(open("docs/eval/clean_select_ckpt.json"))["overall"]
best=max(r,key=lambda a:r[a]["bleu"]+100*r[a]["pronoun_f1"])
n=best.split("_ck")[1].split("_")[0]
ck=f"demo/model_clean/adapter_clean/checkpoint-{n}"
s=json.load(open("docs/eval/clean_split.json")); s["adapter_clean"]=ck
json.dump(s,open("docs/eval/clean_split.json","w"),ensure_ascii=False,indent=1)
print(ck)
PYEOF
)
log "adapter_clean = $BEST  ($(for a in $($PY -c "import json;r=json.load(open('docs/eval/clean_select_ckpt.json'))['overall'];print(' '.join(f'{k}:{v[\"bleu\"]}/{v[\"pronoun_f1\"]}' for k,v in r.items()))"))"

# --- 2. whisper: doi full xong, convert
waitfor logs_run_whisper_clean.log RUN_WHISPER_CLEAN_EXIT
grep -q SMOKE_FAIL logs_run_whisper_clean.log && { log "WHISPER SMOKE_FAIL - dung"; exit 1; }
WPT=demo/model_clean/whisper-medium-clean-openai.pt
[ -f $WPT ] || { log "convert whisper"; $PY ASR/convert_hf_whisper_to_openai.py demo/model_clean/whisper_hf $WPT > logs_convert_whisper.log 2>&1 || { log "CONVERT FAIL"; exit 1; }; }

# --- 3. mbart: doi luu xong
waitfor logs_mbart_clean.log "Model saved to"
[ -f demo/model_clean/mbart_clean/model.safetensors ] || { log "MBART thieu model.safetensors"; exit 1; }

# --- 4. Task 5: eval_clean 13 phim
log "prep_clean_dirs"; $PY demo/EVAL/prep_clean_dirs.py --whisper_pt $WPT --mbart_path demo/model_clean/mbart_clean --adapter $BEST > logs_prep_clean.log 2>&1 || log "prep_clean_dirs rc=$?"
for m in $E5 $EOUT; do d=demo/output/eval_clean/$m; for f in en.srt rough.srt captions.json context.srt; do [ -e $d/$f ] || log "THIEU $d/$f"; done; done

# --- 5. Task 6: hang phu
log "run_clean_rows"; $PY demo/EVAL/run_clean_rows.py --adapter $BEST > logs_clean_rows.log 2>&1 || log "run_clean_rows rc=$?"

# --- 6. Task 7: cham diem
log "export + bleu_episode"
$PY demo/EVAL/export_episode_srt.py --eval_dir demo/output/eval_clean --arms rough.srt nocontext.srt context.srt llm_direct.srt seamless.srt > $RAW/export.txt 2>&1
EP5=$(cd demo && ../$PY -c "from EVAL import split; print(' '.join(split.episode_name(m) for m in '$E5'.split()))")
EPO=$(cd demo && ../$PY -c "from EVAL import split; print(' '.join(split.episode_name(m) for m in '$EOUT'.split()))")
X=demo/output/eval_clean/export
{ echo "## E5 (10 tap)"; $PY demo/EVAL/bleu_episode.py $X/rough $X/nocontext $X/context $X/llm_direct ASR/seamless_output --movies $EP5;
  echo "## E-out (3 phim)"; $PY demo/EVAL/bleu_episode.py $X/rough $X/nocontext $X/context $X/seamless --movies $EPO; } > $RAW/bleu_episode.txt 2>&1
log "thesis_score"
$PY demo/EVAL/thesis_score.py --eval_dir demo/output/eval_clean --movies $E5 --arms rough.srt nocontext.srt context.srt llm_direct.srt --report docs/eval/clean_e5.json > $RAW/thesis_e5.txt 2>&1
$PY demo/EVAL/thesis_score.py --eval_dir demo/output/eval_clean --movies $EOUT --arms rough.srt nocontext.srt context.srt seamless.srt --report docs/eval/clean_out.json > $RAW/thesis_out.txt 2>&1
log "run_eval time-aligned"
: > $RAW/run_eval.txt
for m in $E5 $EOUT; do ep=$(cd demo && ../$PY -c "from EVAL import split; print(split.episode_name('$m'))"); ref="$SUB/$ep.Tiếng_Việt.srt"
  for arm in rough nocontext context llm_direct seamless; do h=demo/output/eval_clean/$m/$arm.srt; [ -f "$h" ] || continue
    echo "### $m $arm" >> $RAW/run_eval.txt; $PY demo/EVAL/run_eval.py --hyp_srt "$h" --ref_srt "$ref" --report $RAW/run_eval_${m}_${arm}.json >> $RAW/run_eval.txt 2>&1; done; done
log "bootstrap"
{ for set in "E5:$E5" "EOUT:$EOUT"; do name=${set%%:*}; mv=${set#*:}; echo "## $name"
    for pair in "rough.srt nocontext.srt" "nocontext.srt context.srt" "rough.srt context.srt"; do set -- $pair
      $PY demo/EVAL/bootstrap_scenes.py --eval_dir demo/output/eval_clean --movies $mv --a $1 --b $2; done; done; } > $RAW/bootstrap.txt 2>&1
log "WER whisper cu vs sach"
EVAL_DIR=demo/output/eval_clean $PY docs/eval/audit_2026-09-02/asr_wer_e5.py > $RAW/wer_clean.txt 2>&1 || log "asr_wer_e5 rc=$?"
log "E0_ALL_DONE"
