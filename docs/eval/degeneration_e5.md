# Suy bien output refine tren eval_e5/eval_ab (2026-08-18)

## Trieu chung

BLEU ca tap (giao thuc Bang 4.7) cua v1b_baseline/v1b_speaker thap bat thuong
so voi rough va E5 cua anh ntVan, du soi tung dong thi cho nao model lam viec
la lam TOT hon rough (vd movie_054 dong 36 khop nguyen van phu de goc).

Kiem ke: dong output chi con "-"/"- -"/"." hoac chua ky tu ngoai he Latin
(Bengali leak o movie_312 dong [0]). Nang nhat eval_e5/movie_336 v1b_speaker:
251/629 dong (39,9%). So dong van bao toan (khong lech hang).

## Co che

Soi debug JSON (`*.srt.json`): sap NGUYEN SCENE — moi scene hoac 100% dong
thanh "- -", hoac hoan toan binh thuong; `fallback_used=False` nen la model
tu sinh, khong phai parse lech. Moi scene sap deu >= 28 dong; scene <= 27
dong khong bao gio sap. Trong vung >= 28 dong thi knife-edge theo input
(cung scene sap o arm nay nhung khong sap o arm kia).

## Nguyen nhan goc

So doi voi setup goc cua anh ntVan (`ntVan/ASR/infer_LLM.py`):

- Adapter GIONG HET: `thevan2404/best_gemma_scene_context` md5 ==
  `ntVan/lora_model_scene_v4_4eps` (c753581554f4eeca9f2eae77129ed672).
- Decoding giong (greedy, max_new_tokens=1024, left padding).
- Khac DUY NHAT o cho dat <Scene Context>: anh ay de trong USER message
  (`<Scene Context>...<English Dialogue>...<Rough...>`), refine_llm.py cua
  minh nhet vao SYSTEM message. Voi context o user, 307 scene test cua anh
  ay (p90=52 dong, max 92) khong sap scene nao; voi context o system, scene
  lon sap hang loat.
- Data train v4 (`ntVan/data/llm_data_scene_v4`) cung de Scene Context trong
  user (system la prompt "post-editor" khac han, thu tu section
  Context/Rough/EN). Tuc la ca hai deu lech format train, nhung cho dat
  context o user la thu empirically hoat dong — va la thu sinh ra so E5.

Chuoi "- -" chinh la hoa van hoc tu data train (dong EN kieu "- -Oh, whoa!"),
model roi vao vong lap emit no khi prompt qua dai o vi tri system.

## Sua (feat/scene-registry-v2)

1. `demo/LLM/refine_llm.py`: chuyen <Scene Context> (gom ca registry tiem
   vao) tu system xuong dau user message, khop byte format infer_LLM.py goc;
   system = `base_system` nguyen ven.
2. `demo/LLM/output_guard.py`: `is_degenerate_line` bat them (a) dong khong
   co ky tu chu/so ("-", "- -", "."), (b) dong chua ky tu ngoai he Latin
   (Cyrillic/A Rap/Bengali/CJK/kana/Hangul) -> fallback dong rough.
3. `tests/test_output_guard.py`: 3 test moi (9/9 pass).

He qua: moi SRT v1b_* sinh truoc fix nay deu phai chay lai truoc khi so
sanh voi Bang 4.7.

## Ket qua cuoi (2026-08-19, sau chuoi A/B dem 18-19/08)

Config chot: prompt goc cua anh ntVan (653 byte, byte-identical voi
`ASR/infer_LLM.py`), Scene Context trong user message, `max_seq_length=2048`,
`--max_scene_lines 24` (sub-chunk can bang). Quyet dinh theo so dong fallback
tren 2 phim kho nhat (336+090 baseline): P1 (prompt cu) 380 -> P2 (prompt goc)
346 -> P3 (P2 + sub-chunk 24) **4**. Fallback toan cuc: 1738/9220 (18.8%) ->
66/9220 (0.7%); 0 dong suy bien lot ra 20 file SRT cuoi.

Bang 4.7 (BLEU corpus noi theo tap, cot raw, 10 tap, cham tren artifact con
song cua anh ntVan `ASR/output_full_pipeline`):

| arm            | raw   | nopunc | custom |
|----------------|-------|--------|--------|
| rough          | 38.24 | 37.73  | 40.06  |
| v1b_baseline   | 38.96 | 38.45  | 40.81  |
| v1b_speaker    | 38.32 | 37.62  | 40.15  |
| E5 (ntVan)     | 38.94 | 38.36  | 40.76  |

**KHONG duoc tuyen bo "vuot E5".** Chenh +0.02 raw la nhieu: paired bootstrap
2000 mau p=0.357; bootstrap doc lap cua reviewer 95% CI [-0.23, +0.30]. Thang
8/10 tap, thua 2 (S03E015 -0.57, S05E015 -0.41) — sign test p~0.11. Cau dung:
*"loai bo hoan toan suy bien dau ra (0 dong hong so voi te nhat 39.9% truoc do)
ma khong mat BLEU — ngang bang thong ke voi artifact E5"*. Metric noi bo
(scene-level): v1b_baseline BLEU 34.10 / COMET 0.7939 / F1 dai tu 0.6968 so
voi rough 33.49 / 0.7820 / 0.6506.

Han che phai ghi kem khi dua vao luan van:

1. **Chon config adaptive tren tap test**: A/B (MSL, prompt, sub-chunk) chay
   tren movie_336+090 — nam trong 10 tap E5; khong co held-out set. Giam nhe:
   tieu chi quyet dinh la so fallback (khong phai BLEU) va moi thay doi co can
   cu co che doc lap (o tren). Truoc khi dua so vao luan van nen xac nhan
   config tren cac tap ngoai E5 (hang doi expand).
2. Moc so sanh la artifact con song cua anh ntVan (38.94 raw / 40.76 custom),
   khong phai con so 40.51 trong luan van (SRT lan chay do khong con —
   `thesis_repro.md`); 40.51 gan cot custom hon cot raw.
3. Arm speaker (38.32) van thua E5 ro; chi arm baseline ngang duoc.
