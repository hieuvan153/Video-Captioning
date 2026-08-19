# Plan dem 2026-08-18: rerun E5 sau prompt-fix (muc tieu vuot 40.51)

Nguoi dung quay lai 10:00 sang 19/08. Toan bo cong viec chay qua
`night_e5_rerun.sh` (nohup, PPID=1, song sot khi ngat SSH), log tai
`logs_night_e5.log`. Idempotent: file da co thi bo qua, chet giua chung
chay lai duoc.

## Boi canh

- Fix thoai hoa da qua code review (Ready to merge, 190/190 test):
  Scene Context chuyen tu system xuong user message + guard chan dong
  "-"/phi-Latin fallback ve rough (docs/eval/degeneration_e5.md).
- Validation giua chung cho thay prompt-fix CHUA diet het thoai hoa
  (110 dong fallback o 2/4 batch movie_336 baseline) nhung guard chan
  100% — dong hong lui ve chat luong rough thay vi pha diem.
- Moi SRT v1b_* cu da doi ten *.pre_promptfix.* — so sanh truoc/sau con
  nguyen.

## Cac buoc (moi buoc co kiem chung)

1. **A/B max_seq_length** tren movie_336 arm baseline: 4096 (config cu)
   vs 2048 (config goc cua anh ntVan). Do bang tong `fallback_used`
   trong debug JSON. Quyet dinh: it fallback hon thang; hoa -> 2048
   (khop config goc). Kiem chung: ca 2 file ab*.srt ton tai, verify
   khong con dong suy bien moi (khac dong rough cung vi tri).
2. **Rerun toan bo**: 10 phim x 2 arm voi MSL thang cuoc (movie_336
   baseline tai su dung file A/B thang). Sau moi file: verify SRT khong
   co dong suy bien nao KHONG bat nguon tu rough; log so fallback.
3. **Cham diem**:
   a. Xuat SRT theo ten tap (`source_episode.txt`) vao
      `demo/output/eval_e5/export/<arm>/<tap>.(Tiếng Việt).srt`.
   b. `bleu_episode.py` (giao thuc Bang 4.7, cot raw la cot bao cao)
      tren: rough, v1b_baseline, v1b_speaker + 3 artifact goc cua anh
      ntVan (E1 output_sentence_vinai, E4 output_pipeline_no_context,
      E5 output_full_pipeline) -> `docs/eval/e5_rerun_bleu.txt`.
   c. `thesis_score.py --arms rough.srt v1b_baseline.srt v1b_speaker.srt`
      (co COMET, GPU) -> `docs/eval/thesis_e5.json` (BLEU/chrF/COMET/
      pronoun P-R-F1 noi bo).
4. Marker `NIGHT_DONE` cuoi log.

## Uoc luong thoi gian

A/B ~50 phut + 19 luot refine ~25 phut/luot ~ 8 tieng + COMET ~30 phut
-> xong ~05:00-06:00, du du phong truoc 10:00. GPU chia se voi job cua
user khac (~57-60GB): moi luot cho `wait_gpu` >= 18GB trong.

## Rui ro & phuong an

- GPU ket dai: wait_gpu ngu 120s/vong, khong co timeout — sang mai neu
  chua xong thi log cho biet dang ket o dau, chay tiep duoc ngay.
- OOM giua batch: refine_llm tu chia doi batch + doi 30 phut; neu van
  fail thi driver log [FAIL] va lam phim tiep theo (khong pha ca dem).
- Thoai hoa van con sau fix: KHONG sao ve mat diem (guard chan het),
  nhung neu ty le fallback cao thi nghi pham con lai la lech whitespace
  giua base_system cua minh va system_prompt goc (reviewer da danh dau).
  Thu nghiem nay DOI user quyet vi dung den rang buoc byte-identical.

## Viec de lai (khong lam dem nay)

- Doi base_system sang dung tung byte prompt goc cua anh ntVan (cho
  user duyet — vi pham rang buoc hien tai nhung la khac biet cuoi cung
  so voi config ra so E5).
- Rerun eval_ab (356 phim) — user da dung queue nay.
- Chot giao thuc Bang 4.8 (dai tu).
- Commit: fix nay nen commit RIENG khoi phan V3 chua commit (goi y cua
  reviewer, cho user quyet).

## Cap nhat 22:40 — chuoi 3 pha dem (sau khi user duyet "cu thuc hien")

- Pha 1 `night_e5_rerun.sh` (log `logs_night_e5.log`): 10 phim x 2 arm,
  prompt hien tai, MSL=2048 (A/B 4096 vs 2048: hoa 295=295). Marker NIGHT_DONE.
- Pha 2 `night2_prompt_ab.sh` (log `logs_night2.log`): doi NIGHT_DONE ->
  A/B prompt goc 653-byte (patch `apply_author_prompt.py` trong scratchpad,
  backup `refine_llm.p1.py`) tren 336+090 baseline -> A/B --max_scene_lines 24
  -> neu config moi it fallback hon thi archive bo cu thanh *.p1.* va rerun
  het, cham lai vao *_final. Markers [P1]/[P2]/[P3]/[decision]/NIGHT2_DONE.
- Pha 3 `night3_retry.sh` (log `logs_night3.log`): doi NIGHT2_DONE -> retry
  2 vong moi file v1b thieu (FAIL/OOM giua dem), FLAG doc tu [decision],
  prompt lay nguyen trang refine_llm.py; cham lai neu co lap day. NIGHT3_DONE.
- Code moi phien nay: split_chunk (scene_assign.py) + flag --max_scene_lines
  (refine_llm.py), 194/194 test. RAM may 345GB kha dung — chi lo CUDA OOM,
  da co 3 lop: retry noi bo refine_llm (30'), wait_gpu 18GB, night3 retry.
- Ket qua cuoi nam o: docs/eval/e5_rerun_bleu.txt (P1),
  e5_rerun_bleu_final.txt + thesis_e5.json/thesis_e5_final.json (chot).
- CHUA COMMIT gi — user se chot khi quay lai.
