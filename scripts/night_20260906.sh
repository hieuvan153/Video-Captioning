#!/usr/bin/env bash
# Driver dem 2026-09-06: P0 -> P6 theo docs/superpowers/plans/2026-09-06-kien-truc-v2.md §7.
# CHAY KHONG NGUOI GIAM SAT. Nguyen tac: mot chang hong KHONG duoc giet ca dem.
#   set -u  (bat bien chua khoi tao) nhung KHONG set -e / KHONG pipefail.
#   Moi chang boc `timeout`, ghi $OUT/status.tsv: stage <TAB> OK|FAIL|SKIP|TIMEOUT <TAB> giay <TAB> ghi chu.
#   Chang sau kiem tra dau ra chang truoc ton tai + KHAC RONG; khong co -> SKIP, khong bao gio chay tren rong.
#   BAO_CAO_DEM.md ghi bang trap EXIT nen co ca khi moi chang deu hong.
#
# Chay that : nohup bash scripts/night_20260906.sh > .../logs/driver.log 2>&1 &
# Doc truoc : DRY_RUN=1 bash scripts/night_20260906.sh
# Chi lam bao cao lai tu artifact da co: bash scripts/night_20260906.sh verdict

set -u

# ---------------------------------------------------------------- duong dan
ROOT=/data/ndloc_bk/NLHV/ntVan
TP=/data/ndloc_bk/NLHV/third_party_test
PY=$ROOT/van_env/bin/python
OUT=${OUT:-$TP/arms_v2_20260906}
FILM=4810_HeVuiLaXiu_OdeToJoy_2019_CLEAN_HD_final_1785899894237

WAV="$TP/output_orig/$FILM.wav"
REF_EN="$TP/ref_3rd/en_3rd.clean.srt"
REF_VI="$TP/ref_3rd/vi_3rd.clean.srt"
OLD_EN="$TP/output_orig/$FILM.(Tiếng Anh).srt"
OLD_ROUGH="$TP/output_orig/$FILM.(Tiếng Việt_dich_tho).srt"
OLD_BS1="$TP/output_orig_bs1/$FILM.(Tiếng Việt_tinh_chinh).srt"
GOLD_ROUGH="$TP/output_gold_en/gold_rough.srt"

# CACHE HF: KHONG dat HF_HOME. Mac dinh /home/ndloc_bk/.cache/huggingface (symlink -> /data) chua
# GemmaX2 20 G + Unbabel/wmt22-comet-da 2,2 G. Doi HF_HOME sang /data/ndloc_bk/hf_cache se lam AN
# ca hai -> tai lai 22 G luc 2h sang va lam hong chinh chang cham diem.
# faster-whisper large-v3 nam o cache KHAC, nen truyen thang path snapshot (mac dinh trong
# asr_fw_infer.py), khong di qua HF_HOME.
export HF_HUB_OFFLINE=1        # moi thu can thiet da co tren dia -> bien tai ngam thanh loi nhanh
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

DRY_RUN=${DRY_RUN:-0}
DEADLINE=$(date -d "${DEADLINE_AT:-2026-09-08 12:00}" +%s)   # sau moc nay: bo qua chang optional (P5, P6 phu)

mkdir -p "$OUT/logs" "$OUT/scratch"

# --------------------------------------------------------- bien trang thai
RC=0
WER_MEDIUM=""; WER_FW3=""; WER_A2=""
G1="CHUA_CHAY"
BEST_EN="$OLD_EN"; ARM_PREFIX="asr"        # doi thanh fw3 khi G1 PASS/WEAK
NOTE_HARNESS_WER="chua chay"
declare -a LIVE_ARMS=()

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$OUT/status.tsv"; }

past_deadline() { [ "$(date +%s)" -ge "$DEADLINE" ]; }

# so sanh so thuc: flt A op B  -> exit 0 neu dung. Rong = khong so sanh duoc = false.
flt() {
  [ -n "${1:-}" ] && [ -n "${3:-}" ] || return 1
  awk -v a="$1" -v op="$2" -v b="$3" 'BEGIN{
    if(op=="<")  exit !(a< b); if(op=="<=") exit !(a<=b);
    if(op==">")  exit !(a> b); if(op==">=") exit !(a>=b); exit 1 }'
}

# dem cue that su parse duoc (khong dem dong so trong SRT - de sai)
n_cues() {
  [ "$DRY_RUN" = "1" ] && { echo 1767; return; }        # dry-run: gia dinh dat de in het lenh
  [ -s "${1:-}" ] || { echo 0; return; }
  "$PY" -c 'import srt,sys
try:
    print(sum(1 for _ in srt.parse(open(sys.argv[1],encoding="utf-8-sig",errors="replace").read())))
except Exception:
    print(0)' "$1" 2>/dev/null || echo 0
}

# dau vao hop le = file ton tai VA khac rong
ok_file() { [ "$DRY_RUN" = "1" ] && return 0; [ -s "${1:-}" ]; }

# doi GPU ranh. Poll 60s, toi da 30 phut roi van chay tiep (theo yeu cau).
wait_gpu() {
  local need=${1:-12000} waited=0 free
  if [ "$DRY_RUN" = "1" ]; then echo "[DRY] wait_gpu can ${need}MiB"; return 0; fi
  while [ "$waited" -lt 1800 ]; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    case "${free:-}" in ''|*[!0-9]*) free=999999 ;; esac
    [ "$free" -ge "$need" ] && { log "GPU ranh: ${free}MiB >= ${need}MiB"; return 0; }
    log "GPU chi con ${free}MiB (<${need}), doi... ${waited}s"
    sleep 60; waited=$((waited + 60))
  done
  log "GPU: het 1800s cho, van chay tiep"
  return 0
}

# run_stage <ten> <timeout giay> <chuoi lenh shell>
# Dat RC. Ghi status.tsv. Khong bao gio lam chet driver.
run_stage() {
  local name=$1 tmo=$2 cmd=$3 t0 dt st log_f
  log_f="$OUT/logs/$name.log"   # phai tach dong: bash khai trien het cac tu TRUOC khi chay
                                # `local`, nen "$name" trong cung dong se doc bien scope ngoai
  if [ "$DRY_RUN" = "1" ]; then
    printf '\n[DRY] %-6s timeout %-6ss  -> %s\n      %s\n' "$name" "$tmo" "$log_f" "$cmd"
    RC=0; return 0
  fi
  log "== $name bat dau (timeout ${tmo}s) =="
  t0=$(date +%s)
  { echo "=== $name $(date -Is) ==="; echo "CMD: $cmd"; } >> "$log_f"
  timeout "$tmo" bash -c "$cmd" >> "$log_f" 2>&1
  RC=$?
  dt=$(( $(date +%s) - t0 ))
  if   [ "$RC" -eq 0 ];   then st=OK
  elif [ "$RC" -eq 124 ]; then st=TIMEOUT
  else                         st=FAIL
  fi
  log "== $name $st rc=$RC ${dt}s =="
  record "$name" "$st" "$dt" "rc=$RC"
  [ "$RC" -eq 0 ]
}

skip_stage() { log "-- $1 SKIP: $2"; record "$1" SKIP 0 "$2"; }

# lay WER tu log film_wer.py (dong bat dau bang TONG, cot 2)
wer_from() {
  if [ "$DRY_RUN" = "1" ]; then    # so gia, chi de dry-run di qua nhanh PASS cua G1
    case "$1" in *1b.log) echo 16.73;; *1c.log) echo 14.50;; *4b.log) echo 16.00;; *) echo 12.00;; esac
    return
  fi
  awk '$1=="TONG"{print $2; exit}' "$1" 2>/dev/null
}

# =========================================================== BAO CAO (trap EXIT)
make_report() {
  [ "$DRY_RUN" = "1" ] && { echo "[DRY] make_report -> $OUT/verdict.json + $OUT/BAO_CAO_DEM.md"; return 0; }
  OUT="$OUT" G1="$G1" BEST_EN="$BEST_EN" ARM_PREFIX="$ARM_PREFIX" \
  WER_MEDIUM="$WER_MEDIUM" WER_FW3="$WER_FW3" WER_A2="$WER_A2" \
  NOTE_HARNESS_WER="$NOTE_HARNESS_WER" \
  "$PY" - <<'PYEOF' 2>&1 | tail -20
import json, os, re

OUT = os.environ["OUT"]
env = os.environ.get

def rd(p):
    try:
        return open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        return ""

# ---- 1. diem tuyet doi tu film_score_v2.json
scores = {}
try:
    scores = json.loads(rd(os.path.join(OUT, "film_score_v2.json")))["scores"]
except Exception:
    pass

# ---- 2. delta + CI tu log cham diem
RE_D = re.compile(r"^\s+(BLEU|chrF|COMET-DA):\s+(\S+)\s+-\s+(\S+)\s+=\s+([+-][\d.]+)\s+"
                  r"95%CI=\[\s*([+-][\d.]+),\s*([+-][\d.]+)\]")
deltas = {}
for lg in ("p3_score.log", "p3b_score.log"):
    for line in rd(os.path.join(OUT, "logs", lg)).splitlines():
        m = RE_D.match(line)
        if m:
            met, arm, base, d, lo, hi = m.groups()
            deltas.setdefault(arm, {})[met] = {
                "base": base, "delta": float(d), "ci_lo": float(lo), "ci_hi": float(hi)}

# ---- 3. toan ven harness: rough phai tai lap chrF 37,55 +/-0,05 va COMET-DA 0,7487 +/-0,001
integrity, notes = "OK", []
r = scores.get("rough")
if not r:
    integrity, _ = "UNTRUSTED", notes.append("khong co hang 'rough' trong film_score_v2.json")
else:
    if abs(r.get("chrf", 0) - 37.55) > 0.05:
        integrity = "UNTRUSTED"; notes.append(f"rough chrF {r.get('chrf')} != 37,55 +/-0,05")
    c = r.get("comet_da")
    if c is None or abs(c - 0.7487) > 0.001:
        integrity = "UNTRUSTED"; notes.append(f"rough COMET-DA {c} != 0,7487 +/-0,001")
if env("NOTE_HARNESS_WER", "") not in ("OK", "chua chay"):
    notes.append("WER moc: " + env("NOTE_HARNESS_WER", ""))
    integrity = "UNTRUSTED"

# ---- 4. cong G2: (chrF CI hoan toan >0 HOAC COMET CI hoan toan >0) VA CI metric con lai KHONG hoan toan <0
verdict = {"integrity": integrity, "integrity_notes": notes,
           "G1": env("G1", ""), "BEST_EN": env("BEST_EN", ""), "arm_prefix": env("ARM_PREFIX", ""),
           "wer": {k: env("WER_" + k.upper(), "") for k in ("medium", "fw3", "a2")},
           "wer_moc_medium_ft": 16.73, "arms": {}}
for arm, d in deltas.items():
    ch, co = d.get("chrF"), d.get("COMET-DA")
    win_ch = ch is not None and ch["ci_lo"] > 0
    win_co = co is not None and co["ci_lo"] > 0
    bad_ch = ch is not None and ch["ci_hi"] < 0
    bad_co = co is not None and co["ci_hi"] < 0
    passed = (win_ch and not bad_co) or (win_co and not bad_ch)
    verdict["arms"][arm] = {
        "chrF": ch, "COMET-DA": co, "BLEU": d.get("BLEU"),
        "G2": ("PASS" if passed else "FAIL") if integrity == "OK" else "UNTRUSTED",
        "abs": scores.get(arm, {})}
json.dump(verdict, open(os.path.join(OUT, "verdict.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# ---- 5. BAO_CAO_DEM.md
def f(x, n=2, dash="—"):
    return dash if x is None or x == "" else f"{float(x):.{n}f}".replace(".", ",")

L = ["# Báo cáo đêm 2026-09-06 — kiến trúc v2 (ASR + NMT, không train)", "",
     f"Thư mục artifact: `{OUT}`", ""]

L += ["## 0. Toàn vẹn harness", "",
      f"- Trạng thái: **{integrity}**"]
L += [f"  - {n}" for n in notes] or ["  - `rough` tái lập đúng chrF 37,55 / COMET-DA 0,7487."]
if integrity != "OK":
    L += ["", "> **Mọi so sánh dưới đây KHÔNG đáng tin.** Hàng `rough` không tái lập được mốc gốc "
          "(`third_party_test/film_score_ref_anchored.json`) nên chênh lệch giữa các arm có thể do "
          "đổi harness chứ không do đổi model. Cần người xử lý trước khi kết luận."]

L += ["", "## 1. Trạng thái từng chặng", "", "| Chặng | Kết quả | Giây | Ghi chú |", "|---|---|---|---|"]
for line in rd(os.path.join(OUT, "status.tsv")).splitlines():
    p = line.split("\t")
    if len(p) >= 4:
        L.append(f"| {p[0]} | {p[1]} | {p[2]} | {p[3]} |")

L += ["", "## 2. WER tầng ASR (thấp hơn = tốt hơn)", "",
      "| Arm | WER % | Mốc so |", "|---|---|---|",
      f"| medium-FT (tái lập mốc) | {f(env('WER_MEDIUM'))} | 16,73 |",
      f"| fw3 = large-v3 + fallback + beam 5 | {f(env('WER_FW3'))} | cổng G1 ≤ 15,00 |",
      f"| a2 = medium-FT + fallback + beam 5 | {f(env('WER_A2'))} | tách model / decode-config |",
      "", f"- **Cổng G1: {env('G1','')}** — `BEST_EN` = `{env('BEST_EN','')}`"]

e5 = [(k, v) for k, v in
      (("movie_054 (S01E017)", "11,9"), ("movie_081 (S02E021)", "8,5"), ("movie_090 (S03E011)", "10,5"))]
rows = []
for i, (nm, moc) in enumerate(e5):
    w = rd(os.path.join(OUT, f"wer_e5_{i}.txt")).strip()
    if w:
        rows.append(f"| {nm} | {f(w)} | {moc} |")
p5_state = rd(os.path.join(OUT, "p5_state.txt")).strip() or "SKIP: chua chay"
p5_ran = p5_state == "RUN" and bool(rows)
if p5_ran:
    L += ["", "### Test phụ E5 — đo trong phân phối", "",
          "| Tập | WER % large-v3 | Mốc medium-FT |", "|---|---|---|"] + rows + [
          "", "> movie_054 và movie_081 **nằm trong tập train của medium-FT** nên mốc của chúng "
          "được lợi rò rỉ — large-v3 thua ở hai tập này KHÔNG có nghĩa là kém. Chỉ **movie_090** "
          "là so sánh sạch."]
else:
    L += ["", "### Test phụ E5 — KHÔNG CHẠY", "",
          f"- Chặng P5 bị bỏ: `{p5_state}`",
          "- **Hệ quả: đêm nay không có số đo trong phân phối.** Mọi kết luận bên dưới chỉ dựa trên "
          "**một** phim ngoài phân phối là Ode to Joy. Chưa có gì được kiểm chứng hai chiều — một arm "
          "thắng ở đây vẫn có thể thua trên phim kiểu E5, và ngược lại. Trước khi đổi đường mặc định "
          "của `run_pipeline.py`, cần chạy lại P5 khi có media E5."]

L += ["", "## 3. Điểm dịch, neo theo cue tham chiếu (1767 đoạn)", "",
      "| Arm | BLEU | chrF | COMET-DA | Đoạn rỗng |", "|---|---|---|---|---|"]
for a, s in scores.items():
    L.append(f"| {a} | {f(s.get('bleu'))} | {f(s.get('chrf'))} | {f(s.get('comet_da'), 4)} | "
             f"{f(s.get('empty_rate', 0) * 100 if s.get('empty_rate') is not None else None, 1)}% |")

L += ["", "## 4. Cổng thắng G2 so với `rough`", "",
      "Cổng: chrF−rough có CI 95% hoàn toàn > 0 **hoặc** COMET-DA−rough có CI hoàn toàn > 0, "
      "**và** CI của metric còn lại không hoàn toàn < 0. BLEU không dùng làm cổng.", "",
      "| Arm | ΔchrF [CI] | ΔCOMET-DA [CI] | G2 |", "|---|---|---|---|"]
for a, v in verdict["arms"].items():
    ch, co = v["chrF"], v["COMET-DA"]
    sch = "—" if not ch else f"{f(ch['delta'])} [{f(ch['ci_lo'])}; {f(ch['ci_hi'])}]"
    sco = "—" if not co else f"{f(co['delta'],4)} [{f(co['ci_lo'],4)}; {f(co['ci_hi'],4)}]"
    L.append(f"| {a} | {sch} | {sco} | **{v['G2']}** |")

L += ["", "## 5. Kết luận: giữ hay bỏ từng arm", ""]
if not p5_ran:
    L += ["> Đọc mục 5 với điều kiện của mục 2: **chỉ đo trên Ode to Joy, không có đối chứng "
          "trong phân phối.** \"GIỮ\" ở đây nghĩa là *ứng viên đáng chạy tiếp*, không phải *đã "
          "kiểm chứng*.", ""]
if not verdict["arms"]:
    L += ["- Không arm nào được chấm điểm — xem cột Ghi chú ở bảng chặng (mục 1) để biết chỗ hỏng."]
for a, v in verdict["arms"].items():
    if v["G2"] == "PASS":
        L.append(f"- **`{a}` — GIỮ.** Qua cổng G2, thắng `rough` có ý nghĩa thống kê. Ứng viên "
                 f"thành đường mặc định v2 (sửa `run_pipeline.py` ở phiên CÓ giám sát, không sửa trong đêm).")
    elif v["G2"] == "UNTRUSTED":
        L.append(f"- `{a}` — **chưa kết luận được**, harness không tái lập mốc.")
    else:
        L.append(f"- `{a}` — **BỎ.** Không qua cổng G2; chênh lệch so với `rough` không tách khỏi 0.")

gg, gr = scores.get("gold_gx2"), scores.get("gold_rough")
L += ["", "### Trần tầng NMT (đọc N3)", ""]
if gg and gg.get("chrf") is not None:
    base_ch = (gr or {}).get("chrf", 40.92)
    base_co = (gr or {}).get("comet_da", 0.7677)
    better = gg["chrf"] > base_ch or (gg.get("comet_da") or 0) > base_co
    L.append(f"- `gold_gx2` (GemmaX2 trên EN chuẩn): chrF {f(gg.get('chrf'))} / "
             f"COMET-DA {f(gg.get('comet_da'),4)} so với `gold_rough` (mBART trên cùng EN chuẩn) "
             f"chrF {f(base_ch)} / COMET-DA {f(base_co,4)}.")
    L.append("- → **Tầng NMT còn dư địa theo hướng đổi model.**" if better else
             "- → **Tầng NMT hết dư địa theo hướng đổi model: v2 giữ mBART.** Đây là kết quả có giá "
             "trị kể cả khi mọi arm khác trượt — khỏi tốn công GemmaX2 về sau.")
else:
    L.append("- Chưa có `gold_gx2` → chưa đo được trần tầng NMT.")

L += ["", "## 6. Log", "",
      "| File | Nội dung |", "|---|---|",
      f"| `{OUT}/logs/driver.log` | toàn bộ dòng thời gian |",
      f"| `{OUT}/logs/<chặng>.log` | đầu ra thô từng chặng |",
      f"| `{OUT}/status.tsv` | bảng trạng thái máy đọc được |",
      f"| `{OUT}/verdict.json` | delta + CI + PASS/FAIL từng arm |", ""]

open(os.path.join(OUT, "BAO_CAO_DEM.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("da ghi", os.path.join(OUT, "verdict.json"), "va BAO_CAO_DEM.md")
PYEOF
}

on_exit() {
  log "### trap EXIT: ghi bao cao ###"
  make_report || log "make_report loi (bo qua)"
  log "### xong. Doc: $OUT/BAO_CAO_DEM.md ###"
}

# che do chi lam lai bao cao tu artifact da co
if [ "${1:-}" = "verdict" ]; then make_report; exit 0; fi

trap on_exit EXIT

log "###### DRIVER DEM 2026-09-06 bat dau. DRY_RUN=$DRY_RUN OUT=$OUT ######"
log "deadline optional: $(date -d @$DEADLINE)"

# ============================================================ P0: chuan bi + smoke
# 0a: kiem tra dau vao bat buoc co that (khong co thi dung han - moi thu sau deu vo nghia)
MISSING=""
for f in "$WAV" "$REF_EN" "$REF_VI" "$OLD_EN" "$OLD_ROUGH" "$GOLD_ROUGH"; do
  ok_file "$f" || MISSING="$MISSING $f"
done
if [ -n "$MISSING" ]; then
  log "0a THIEU DAU VAO:$MISSING"
  record 0a FAIL 0 "thieu dau vao:$MISSING"
  exit 1
fi
record 0a OK 0 "dau vao day du"

# 0b: preflight nap model - tha chet o phut thu 5 con hon phut thu 200.
# Kiem tra ca 3 thu se dung trong dem, TAT CA duoi HF_HUB_OFFLINE=1.
run_stage 0b 900 "$PY - <<'EOF'
import os, torch
ok = True
try:
    from faster_whisper import WhisperModel
    import sys; sys.path.insert(0, '$ROOT/demo/ASR')
    from asr_fw_infer import FW_LARGE_V3_LOCAL
    WhisperModel(FW_LARGE_V3_LOCAL, device='cuda', compute_type='float16')
    print('OK  faster-whisper large-v3 (local ct2):', FW_LARGE_V3_LOCAL)
except Exception as e:
    ok = False; print('FAIL faster-whisper:', repr(e))
try:
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained('xiaomi-research/GemmaX2-28-9B-v0.2', local_files_only=True)
    print('OK  tokenizer GemmaX2, vocab', len(t))
except Exception as e:
    ok = False; print('FAIL GemmaX2 tokenizer:', repr(e))
try:
    from comet import download_model
    print('OK  COMET-DA ckpt:', download_model('Unbabel/wmt22-comet-da'))
except Exception as e:
    ok = False; print('FAIL COMET-DA:', repr(e))
raise SystemExit(0 if ok else 1)
EOF"
PREFLIGHT_RC=$RC
[ "$PREFLIGHT_RC" -ne 0 ] && log "0b preflight CO LOI - van chay tiep, tung chang se tu bao loi"

# 0c: cat 5 phut audio smoke
run_stage 0c 120 "ffmpeg -y -ss 1200 -t 300 -i '$WAV' '$OUT/smoke.wav'"

# 0d: smoke ASR
if ok_file "$OUT/smoke.wav"; then
  wait_gpu 8000
  run_stage 0d 1800 "$PY '$ROOT/demo/ASR/asr_fw_infer.py' --audio '$OUT/smoke.wav' --out_srt '$OUT/smoke.en.srt'"
else
  skip_stage 0d "smoke.wav rong/khong co"
fi

SMOKE_CUES=$(n_cues "$OUT/smoke.en.srt")
log "0d smoke.en.srt: $SMOKE_CUES cue"

# 0e: smoke NMT
if ok_file "$OUT/smoke.en.srt"; then
  wait_gpu 8000
  run_stage 0e 900 "$PY '$ROOT/demo/NMT/run_nmt.py' --input_srt '$OUT/smoke.en.srt' --output_srt '$OUT/smoke.vi.srt'"
  log "0e smoke.vi.srt: $(n_cues "$OUT/smoke.vi.srt") cue (EN: $SMOKE_CUES)"
else
  skip_stage 0e "smoke.en.srt rong/khong co"
fi

# ============================================================ P1: ASR large-v3 ca phim
if ok_file "$OUT/smoke.en.srt" && [ "$SMOKE_CUES" -ge 20 ]; then
  wait_gpu 8000
  run_stage 1a 7200 "$PY '$ROOT/demo/ASR/asr_fw_infer.py' --audio '$WAV' --out_srt '$OUT/fw3.en.srt'"
else
  skip_stage 1a "smoke ASR truot (<20 cue) - khong chay ca phim"
fi

FW3_CUES=$(n_cues "$OUT/fw3.en.srt")
log "1a fw3.en.srt: $FW3_CUES cue"

# 1b: tai lap WER moc 16,73 (kiem tra dung ban film_wer.py)
run_stage 1b 600 "$PY '$ROOT/demo/EVAL/film_wer.py' --hyp '$OLD_EN' --ref '$REF_EN' --name medium_ft"
WER_MEDIUM=$(wer_from "$OUT/logs/1b.log")
if flt "$WER_MEDIUM" ">" 0; then
  if flt "$WER_MEDIUM" ">=" 16.43 && flt "$WER_MEDIUM" "<=" 17.03; then
    NOTE_HARNESS_WER=OK; log "1b WER moc $WER_MEDIUM (dat 16,73 +/-0,3)"
  else
    NOTE_HARNESS_WER="WER moc $WER_MEDIUM lech khoi 16,73 +/-0,3"
    log "1b CANH BAO: $NOTE_HARNESS_WER -> verdict se ghi UNTRUSTED"
  fi
else
  NOTE_HARNESS_WER="khong doc duoc WER moc"
fi
record 1b_gate "$([ "$NOTE_HARNESS_WER" = OK ] && echo OK || echo FAIL)" 0 "$NOTE_HARNESS_WER"

# 1c: WER arm moi + cong G1
if ok_file "$OUT/fw3.en.srt" && [ "$FW3_CUES" -ge 800 ]; then
  run_stage 1c 600 "$PY '$ROOT/demo/EVAL/film_wer.py' --hyp '$OUT/fw3.en.srt' --ref '$REF_EN' --name fw3"
  WER_FW3=$(wer_from "$OUT/logs/1c.log")
else
  skip_stage 1c "fw3.en.srt thieu hoac <800 cue (co $FW3_CUES)"
fi

if flt "$WER_FW3" "<=" 15.00; then
  G1=PASS;  BEST_EN="$OUT/fw3.en.srt"; ARM_PREFIX=fw3
elif flt "$WER_FW3" "<" 16.73; then
  G1=WEAK;  BEST_EN="$OUT/fw3.en.srt"; ARM_PREFIX=fw3
elif flt "$WER_FW3" ">=" 16.73; then
  G1=FAIL;  BEST_EN="$OLD_EN";         ARM_PREFIX=asr
else
  G1=KHONG_DO_DUOC; BEST_EN="$OLD_EN"; ARM_PREFIX=asr
fi
log "Cong G1 = $G1 (WER fw3=${WER_FW3:-?} vs moc ${WER_MEDIUM:-?}). BEST_EN=$BEST_EN, tien to arm=$ARM_PREFIX"
record 1c_gate "$G1" 0 "WER_fw3=${WER_FW3:-?} BEST_EN=$BEST_EN"

# ============================================================ P2: cac arm NMT
A_ROUGH="$OUT/${ARM_PREFIX}_rough.srt"
A_BEAM5="$OUT/${ARM_PREFIX}_beam5.srt"
A_GX2="$OUT/${ARM_PREFIX}_gx2.srt"
GOLD_GX2="$OUT/gold_gx2.srt"
BEST_EN_CUES=$(n_cues "$BEST_EN")
log "BEST_EN co $BEST_EN_CUES cue"

# 2a: greedy - dung giao thuc cua moc rough
if ok_file "$BEST_EN"; then
  wait_gpu 8000
  run_stage 2a 1800 "$PY '$ROOT/demo/NMT/run_nmt.py' --input_srt '$BEST_EN' --output_srt '$A_ROUGH'"
else
  skip_stage 2a "BEST_EN rong/khong co"
fi
[ "$(n_cues "$A_ROUGH")" -eq "$BEST_EN_CUES" ] && [ "$BEST_EN_CUES" -gt 0 ] \
  && LIVE_ARMS+=("${ARM_PREFIX}_rough=$A_ROUGH") \
  || { log "2a arm loai: $(n_cues "$A_ROUGH") cue != $BEST_EN_CUES"; record 2a_arm SKIP 0 "so cue khong khop"; }

# 2b: beam 5
if ok_file "$BEST_EN"; then
  wait_gpu 8000
  run_stage 2b 3600 "$PY '$ROOT/demo/NMT/run_nmt.py' --input_srt '$BEST_EN' --output_srt '$A_BEAM5' --num_beams 5"
else
  skip_stage 2b "BEST_EN rong/khong co"
fi
[ "$(n_cues "$A_BEAM5")" -eq "$BEST_EN_CUES" ] && [ "$BEST_EN_CUES" -gt 0 ] \
  && LIVE_ARMS+=("${ARM_PREFIX}_beam5=$A_BEAM5") \
  || { log "2b arm loai: $(n_cues "$A_BEAM5") cue != $BEST_EN_CUES"; record 2b_arm SKIP 0 "so cue khong khop"; }

# 2c: gold_gx2 - do TRAN tang NMT. Chay TRUOC 2d: khong phu thuoc P1, va neu GemmaX2 hong thi biet som.
REF_EN_CUES=$(n_cues "$REF_EN")
wait_gpu 26000
run_stage 2c 10800 "$PY '$ROOT/demo/NMT/run_gemmax2.py' --input_srt '$REF_EN' --output_srt '$GOLD_GX2'"
GX2_OK=$RC
[ "$(n_cues "$GOLD_GX2")" -eq "$REF_EN_CUES" ] && [ "$REF_EN_CUES" -gt 0 ] \
  && LIVE_ARMS+=("gold_gx2=$GOLD_GX2") \
  || { log "2c arm loai: $(n_cues "$GOLD_GX2") cue != $REF_EN_CUES"; record 2c_arm SKIP 0 "so cue khong khop"; GX2_OK=1; }

# 2d: GemmaX2 tren EN tot nhat - chi chay khi 2c chung minh GemmaX2 chay duoc
if [ "$GX2_OK" -eq 0 ] && ok_file "$BEST_EN"; then
  wait_gpu 26000
  run_stage 2d 10800 "$PY '$ROOT/demo/NMT/run_gemmax2.py' --input_srt '$BEST_EN' --output_srt '$A_GX2'"
  [ "$(n_cues "$A_GX2")" -eq "$BEST_EN_CUES" ] && [ "$BEST_EN_CUES" -gt 0 ] \
    && LIVE_ARMS+=("${ARM_PREFIX}_gx2=$A_GX2") \
    || { log "2d arm loai"; record 2d_arm SKIP 0 "so cue khong khop"; }
else
  skip_stage 2d "gold_gx2 that bai (GemmaX2 khong dung duoc) hoac BEST_EN rong"
fi

# ============================================================ P3: cham diem mot luot
ARMS_STR="'rough=$OLD_ROUGH' 'gold_rough=$GOLD_ROUGH'"
ok_file "$OLD_BS1" && ARMS_STR="$ARMS_STR 'gemma_bs1=$OLD_BS1'"
for a in ${LIVE_ARMS+"${LIVE_ARMS[@]}"}; do ARMS_STR="$ARMS_STR '$a'"; done
log "3a arm se cham: $ARMS_STR"

wait_gpu 12000
run_stage 3a 7200 "$PY '$ROOT/demo/EVAL/film_ref_anchored.py' --ref '$REF_VI' --src_en '$REF_EN' \
  --arms $ARMS_STR --base rough --n 1000 --report '$OUT/film_score_v2.json' 2>&1 | tee '$OUT/logs/p3_score.log'"

# 3b: verdict (make_report idempotent, chay lai o trap EXIT)
run_stage 3b 300 "true"
make_report

# ============================================================ P4: ablation decode-config tren model cu
if past_deadline; then
  skip_stage 4a "qua 12:00 07/09 - bo qua chang khong bat buoc"
else
  mkdir -p "$OUT/scratch/a2"
  # asr_movie_infer.py ghi vad_chunks/ + segment_info.json theo CWD -> phai cd vao scratch rieng
  wait_gpu 8000
  run_stage 4a 10800 "cd '$OUT/scratch/a2' && ASR_TEMPS=0,0.2,0.4,0.6,0.8,1.0 ASR_BEAM=5 \
    $PY '$ROOT/demo/ASR/asr_movie_infer.py' --audio_path '$WAV' --out_dir '$OUT' --out_name a2_fallback"
fi

A2_EN="$OUT/a2_fallback.(Tiếng Anh).srt"
if ok_file "$A2_EN"; then
  run_stage 4b 900 "$PY '$ROOT/demo/EVAL/film_wer.py' --hyp '$A2_EN' --ref '$REF_EN' --name a2"
  WER_A2=$(wer_from "$OUT/logs/4b.log")
  log "4b WER a2 = ${WER_A2:-?}"
else
  skip_stage 4b "a2_fallback EN rong/khong co"
fi

# 4c: chi khi a2 thang fw3 ve WER thi moi dang dich + cham bo sung
if ok_file "$A2_EN" && flt "$WER_A2" "<" "${WER_FW3:-99}" && ! past_deadline; then
  A2_ROUGH="$OUT/a2_rough.srt"
  A2_CUES=$(n_cues "$A2_EN")
  wait_gpu 8000
  run_stage 4c 1800 "$PY '$ROOT/demo/NMT/run_nmt.py' --input_srt '$A2_EN' --output_srt '$A2_ROUGH'"
  if [ "$(n_cues "$A2_ROUGH")" -eq "$A2_CUES" ] && [ "$A2_CUES" -gt 0 ]; then
    wait_gpu 12000
    run_stage 4d 5400 "$PY '$ROOT/demo/EVAL/film_ref_anchored.py' --ref '$REF_VI' --src_en '$REF_EN' \
      --arms 'rough=$OLD_ROUGH' 'a2_rough=$A2_ROUGH' --base rough --n 1000 \
      --report '$OUT/film_score_a2.json' 2>&1 | tee '$OUT/logs/p3b_score.log'"
  else
    skip_stage 4d "a2_rough so cue khong khop"
  fi
else
  skip_stage 4c "a2 khong thang fw3 ve WER (a2=${WER_A2:-?} fw3=${WER_FW3:-?}) hoac qua deadline"
fi

# ============================================================ P5: test phu E5 (optional, chi WER)
# Media goc cua E5 tung bi nghi la da xoa. Kiem tra SU TON TAI CUA CA 3 FILE TRUOC khi chay bat cu
# thu gi - thieu la SKIP sach se ngay, tuyet doi khong buoc vao ASR roi chet dan het timeout.
# Hai noi co the chua: data/Movie/video/<series>/<mua>/<tap>/<ten_tap>.wav (thuc te dang nam o day)
# va data/Movie/audio/<ten_tap>.wav (theo episode_name() trong demo/EVAL/split.py).
MV=/data/ndloc_bk/ntVan/data/Movie
E5_NAMES=(
  "S01E017_Nhu_Thuật_bọc_bong_bóng_và_chăm_lo"
  "S02E021_Trái_tim_tan_vỡ_và_quái_vật_đất_nung"
  "S03E011_Gà_sống_gà_rán_và_thánh_hôn"
)
E5_CODES=(S01E017 S02E021 S03E011)   # = movie_054 / movie_081 / movie_090

declare -a E5_WAVS=() E5_REFS=()
E5_MISSING=""
for i in 0 1 2; do
  nm=${E5_NAMES[$i]}; code=${E5_CODES[$i]}; found=""
  for cand in "$MV/audio/$nm.wav" $(ls "$MV/video"/*/*/*/"$nm.wav" 2>/dev/null); do
    [ -s "$cand" ] && { found=$cand; break; }
  done
  ref=$(ls "$ROOT/ASR/ground_truth_asr/${code}"*.srt 2>/dev/null | head -1)
  [ -n "$found" ] || E5_MISSING="$E5_MISSING $nm.wav"
  [ -s "${ref:-}" ] || E5_MISSING="$E5_MISSING ground_truth:$code"
  E5_WAVS+=("$found"); E5_REFS+=("${ref:-}")
done

if [ -n "$E5_MISSING" ]; then
  E5_STATE="SKIP: thieu media/ground truth:$E5_MISSING"
  log "P5 SKIP - thieu:$E5_MISSING"
  record P5 SKIP 0 "khong co audio E5"
elif past_deadline; then
  E5_STATE="SKIP: qua 12:00 07/09"
  record P5 SKIP 0 "qua deadline"
else
  E5_STATE=RUN
  log "P5: du ca 3 audio E5, chay WER trong-phan-phoi"
  for i in 0 1 2; do
    past_deadline && { skip_stage "5a_$i" "qua 12:00 07/09"; continue; }
    wait_gpu 8000
    run_stage "5a_$i" 3600 "$PY '$ROOT/demo/ASR/asr_fw_infer.py' --audio '${E5_WAVS[$i]}' --out_srt '$OUT/e5_$i.en.srt'"
    if ok_file "$OUT/e5_$i.en.srt"; then
      run_stage "5b_$i" 600 "$PY '$ROOT/demo/EVAL/film_wer.py' --hyp '$OUT/e5_$i.en.srt' --ref '${E5_REFS[$i]}' --name ${E5_CODES[$i]}"
      [ "$DRY_RUN" = "1" ] || wer_from "$OUT/logs/5b_$i.log" > "$OUT/wer_e5_$i.txt"
    else
      skip_stage "5b_$i" "e5_$i.en.srt rong"
    fi
  done
fi
[ "$DRY_RUN" = "1" ] || printf '%s\n' "$E5_STATE" > "$OUT/p5_state.txt"

# ============================================================ P6: bao cao (trap EXIT cung se chay)
make_report
record 6 OK 0 "bao cao da ghi"
log "###### DRIVER XONG ######"
