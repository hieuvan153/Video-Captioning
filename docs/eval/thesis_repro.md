# Tai lap Bang 4.7 cua do an (2026-08-17)

Truoc do toi ket luan nham la artifact eval da bi xoa — sai, do tim sai ten
thu muc (`gemma-12b-...` gach ngang, thuc te `gemma_12b-...` gach duoi).
Artifact con day du.

## Giao thuc dung cua Bang 4.7

`/data/ndloc_bk/ntVan/ASR/calculate_bleu.py`:

- Don vi: **ca tap** (noi toan bo dong SRT thanh 1 chuoi), 10 tap -> corpus BLEU.
- Hypothesis: file `.srt` trong thu muc output cua tung phuong phap.
- Reference: phu de chinh thuc `data/Movie/sub/*.Tiếng_Việt.srt`, co buoc
  `get_optimized_reference_text` — hoan vi cac dong trung timestamp de chon
  thu tu cho BLEU cao nhat (exact neu <= 64 to hop, greedy 2 luot neu hon).
- Cot bao cao trong bang = **raw** (khong lowercase, khong bo ngoac).

## Doi chieu (chay lai 2026-08-17)

| Thu muc | raw | nopunc | custom | O trong Bang 4.7 |
|---|---|---|---|---|
| `ASR/seamless_output` | **23.33** | 21.28 | 24.53 | E3 SeamlessM4T 23.33 — khop |
| `ASR/output_pipeline_no_context_sentence_level` | **37.83** | 37.01 | 39.66 | E2 LLM dich 37.83 — khop |
| `ASR/output_sentence_vinai` | **38.23** | 37.72 | 40.05 | E1 NMT 38.23 — khop |
| `ASR/output_pipeline_no_context` | 37.40 | 36.82 | 39.19 | E4 (+LLM khong hinh anh) 38.93 |
| `ASR/output_full_pipeline` | 38.94 | 38.36 | 40.76 | E5 (+LLM co hinh anh) 40.51 |

3/5 o khop chinh xac den 2 chu so — giao thuc da xac dinh chac chan.

E4/E5 lech deu **+1.53 / +1.57** so voi bang, va do chenh do ngu canh hinh
anh gan nhu trung khop: tren o 38.94 - 37.40 = **+1.54**, trong bang
40.51 - 38.93 = **+1.58**. Ket luan: hai thu muc SRT tren o la ban chay cua
adapter doi truoc (mtime 2026-03-05 / 03-08), con bang bao cao ban chay lai
bang adapter v6 (mtime 2026-05-04 tro di) — SRT cua ban do khong duoc giu.
Ket qua co ban chat "visual context them ~1.5 BLEU" tai lap duoc.

Da quet toan bo cay `/data/ndloc_bk/ntVan` tim moi thu muc chua 10 file
`.(Tiếng Việt).srt`: chi co 5 thu muc tren + ban sao trong `upload/ASR/`
(diff -rq: giong het byte, mtime 2026-03-10). Khong con ung vien nao khac —
SRT cua ban chay v6 that su khong duoc giu.

## He qua cho track cua minh

Muon so truc tiep voi o 40.51: xuat arm cua minh ra SRT theo dung ten tap
(`<ten_tap>.(Tiếng Việt).srt`) roi cham bang chinh `calculate_bleu.py`
(hoac `demo/EVAL/bleu_episode.py` — ban chay nhieu thu muc). Do la so duy nhat
so sanh duoc apples-to-apples voi bang.

`demo/EVAL/thesis_score.py` (scene-level) KHONG phai giao thuc nay — chi dung
cho so sanh noi bo giua cac arm.

## Bang 4.8 (dai tu) — chua chot

Cham lai bang lexicon 24 dai tu cell-37 tren moi artifact deu ra cao hon bang:

| Nguon | multiset P/R/F1 | set P/R/F1 |
|---|---|---|
| xlsx v6-4eps (chunk) | 0.6388/0.7439/0.6873 | 0.6813/0.7090/0.6949 |
| xlsx v6_2-4eps (chunk) | 0.6617/0.7337/0.6958 | 0.6925/0.7021/0.6972 |
| bleu_report nmt_llm (chunk) | 0.6784/0.7488/0.7119 | 0.7259/0.7025/0.7140 |
| output_full_pipeline (ca tap) | 0.7694/0.8908/0.8257 | 0.8580/0.8910/0.8742 |

Bang 4.8 ghi E1 0.5384/0.5738/0.5556, E4 0.5752/0.6251/0.5991,
E5 0.5806/0.6411/0.6094 — thap hon han, dung dai voi so `pronoun_f1`
per-dong cua `demo/EVAL/run_eval.py` cham theo gold `pronouns_subject/object`
cua dataset. Gia thuyet: Bang 4.8 cham per-dong theo gold annotation, tuc
CUNG giao thuc `run_eval.py` cua minh. Chua kiem chung duoc vi truc thoi gian
ASR cua anh ay khong 1:1 voi dong dataset.
