# Error Analysis V0 — Character Registry A/B (2026-08-14)

Phân tích per-line 5 phim × 3 arm của `docs/eval/registry_ab_v0.json`
(output tại `demo/output/eval_ab/`), làm trước khi quyết định gate V1.

## TL;DR

1. **Con số headline +0.0223 ΔF1 là artifact.** Toàn bộ mức tăng của movie_015
   (+0.1184) đến từ **1 dòng suy biến** trong arm baseline (dòng 342: "Thằng
   nhóc," lặp ~253 lần — đúng failure mode adapter-degeneration đã ghi nhận ở
   Task 7). Thay dòng đó bằng dòng rough tương ứng: baseline movie_015 F1
   0.3873 → **0.5000**, BLEU 31.94 → **38.47**. Mean ΔF1 corrected =
   **−0.0003**, mean ΔBLEU corrected ≈ **0.00**. Registry V0 dạng hiện tại
   (1 dòng quan hệ toàn cục tiêm vào mọi scene) **không có tác dụng ròng**.
2. Tác dụng thật của registry **phụ thuộc nội dung**, hai phim có kết quả
   significant theo paired bootstrap (2000 resamples, per-line):
   - movie_046 (Young Sheldon — quan hệ gia đình dày): **+0.0460**, 95% CI
     [+0.0115, +0.0807] → giúp thật.
   - movie_045: **−0.0359**, 95% CI [−0.0677, −0.0043] → hại thật.
   - 008/009/015: CI đều chứa 0.
3. Cơ chế **hại**: model áp đúng edge cho **sai người nói/người nghe** —
   vocabulary của dòng registry leak vào các dòng thoại không liên quan
   (movie_008: 16/24 FP mới nằm trong vocab dòng tiêm; movie_045: 20/33).
   Registry extraction trên sitcom đông nhân vật còn rác (nhân vật "You"/"Me"/
   "I"/"Him", edge ngược chiều, giá trị "glenn", "(addressed as)", "anh/chị").
4. Cơ chế **giúp** (046): ctx sạch toàn edge thân tộc (bố/con, mẹ/con,
   bác sĩ/bệnh nhân) → model chuyển register đúng cho người nói là trẻ em
   (tôi→cháu, tôi→con) — đáng chú ý là 15/18 dòng cải thiện dùng term *không*
   có trong dòng tiêm → model tổng quát hóa từ quan hệ + tuổi, không tra bảng.
5. **Cấu trúc lỗi còn lại chỉ về speaker attribution**: FP của arm registry bị
   chi phối bởi generic "tôi" (194) / "anh" (176) trong khi FN toàn từ xưng hô
   có hướng (cô 72, anh 57, ông 51, cháu 45, bà 28, em 23, mẹ 23...). Model
   biết *quan hệ tồn tại* nhưng không biết *dòng này ai nói với ai* nên rơi về
   generic hoặc chọn nhầm hướng.

## Bảng corrected (sau khi vá dòng suy biến baseline movie_015)

| Movie | F1 base (raw→fixed) | F1 registry | ΔF1 fixed | 95% CI | Ghi chú |
|---|---|---|---|---|---|
| movie_008 | 0.4882 | 0.4853 | −0.0029 | [−0.033, +0.029] | không đổi |
| movie_009 | 0.6103 | 0.5961 | −0.0142 | [−0.039, +0.009] | không đổi |
| movie_015 | 0.3873 → 0.5000 | 0.5057 | **+0.0057** | [−0.021, +0.034] | +0.1184 là artifact |
| movie_045 | 0.3761 | 0.3402 | **−0.0359** | [−0.068, −0.004] | hại significant |
| movie_046 | 0.4325 | 0.4785 | **+0.0460** | [+0.012, +0.081] | giúp significant |
| **Mean** | | | **−0.0003** | | raw là +0.0223 |

Dòng suy biến phát hiện được (detector: 1–3-gram lặp ≥8 lần chiếm >50% dòng dài):
- `movie_015/refined_baseline.srt` dòng 342 — "Thằng nhóc," × ~253.
- `movie_009/rough.srt` dòng 203 — chuỗi "- " lặp (artifact NMT có sẵn, ảnh
  hưởng cả 3 arm như nhau nên không lệch A/B).
- Arm registry: **0 dòng suy biến** trên cả 5 phim.

## Bằng chứng chi tiết

### Regressed = đúng edge, sai chỗ áp

- movie_045 (13 dòng "tôi"→"em"): registry có edge `teacher->student self=tôi
  listener=em (high)` và `superior->subordinate ... listener=em`; model gán
  "em" cho cả những speaker không phải học trò: `EN "I have no idea." / REF
  "Tôi không biết." / REG "Em không biết."`. Dòng 179 còn tệ hơn: baseline
  đúng "cháu…bà" (1 lỗi) bị registry đẩy về "Tôi…Tôi" (2 lỗi).
- movie_008: reporter nữ bị gọi "anh" (`REF "Nếu cô muốn viết bài về tôi" →
  REG "Nếu anh quyết định..."`) **dù ctx có đúng edge** `Glenn calls Reporter
  "cô"` — vì 5/6 cặp còn lại trong ctx đều là anh/tôi và model không biết dòng
  nào là thoại với reporter.
- movie_009 (14 FP "cô"): "anh" đúng bị thay bằng "cô" hàng loạt — ctx nhắc
  nhiều nhân vật nữ (Cheyenne, Mom) nên model đổi listener sang "cô" ở thoại
  của người khác.
- Nhiễu thứ cấp: cùng seed, chỉ thêm 1 dòng ctx làm ~1/3 số dòng đổi text
  (tie_changed 49–56 dòng/phim, F1 không đổi) — paraphrase drift, và vài dòng
  lệch hẳn nội dung batch (movie_008 dòng 133 output "Ồ, J-O-N…").

### Improved (046) = đổi register người nói trẻ em

`EN "I'd like to speak with a loan officer." / BAS "Tôi muốn..." / REG "Cháu
muốn..."` (gold "cháu"); tương tự "tôi"→"cháu"/"con" ở 84, 223... Registry của
046 sạch nhất trong 5 phim: 6 cặp tiêm đều là quan hệ gia đình/bác sĩ
confidence high, không có cặp generic.

### Registry extraction quality (nguyên nhân gốc phía build)

21–34 characters và 40–60 edges/phim nhưng đa số confidence low nên chỉ 6 cặp
vào prompt; còn lại nhiều rác: nhân vật từ đại từ ("You", "Me", "I", "Him",
"Everyone", "She/her"), edge đảo chiều (`Speaker calls Dude "tôi" and self
"anh"`), vi_self/vi_listener không phải từ xưng hô ("glenn", "hubert",
"(addressed as)", "pen pal", "bệnh nhân", "anh/chị").

## Khuyến nghị gate V1

Theo outline "Sau V0": trường hợp "V0 ≈ 0" đã xảy ra (sau khi bỏ artifact),
nhưng error analysis chỉ ra nguyên nhân **không phải** "quan hệ vô dụng" mà là
**(a)** thiếu speaker per line và **(b)** cách tiêm toàn cục. Cụ thể:

1. **GO cho V1 Speaker Attribution** (CAM++ cluster + LLM map cluster→character,
   tiêm `[SPEAKER: X]` per line) — với điều kiện *đổi cơ chế dùng registry*:
   tra edge theo (speaker dòng này → listener) rồi tiêm per-line/per-scene,
   thay vì 1 dòng quan hệ chung cho cả phim. Bằng chứng ủng hộ: top-FP
   generic tôi/anh + top-FN từ xưng hô có hướng; case movie_008 có đúng edge
   nhưng vẫn chọn sai listener.
2. **3 guardrail rẻ làm ngay, trước V1** (không cần GPU để verify lại):
   - Chống suy biến trong `refine_llm.py`: detector n-gram lặp → fallback dòng
     rough. Diệt luôn class lỗi làm hỏng cả A/B lần này (và làm sạch eval V1).
   - Siết validate registry: drop character có name là đại từ/placeholder
     (mở rộng fix `dd6806c`), yêu cầu `vi_self`/`vi_listener` ∈ lexicon xưng hô
     (chặn "glenn", "(addressed as)", "anh/chị").
   - **Selective activation**: chỉ tiêm registry khi có ≥2 cặp kinship
     confidence high (kiểu 046); phim chỉ có edge generic/low (kiểu 045) thì
     bỏ qua — vì đo được tiêm generic là hại significant.
3. Gate V1 giữ nguyên +0.03 F1, nhưng chấm trên pipeline đã có guardrail chống
   suy biến (để không lặp lại artifact 015).

Script phân tích (per-line diff, sensitivity, bootstrap) nằm trong scratchpad
phiên làm việc 2026-08-14; số liệu gốc: `demo/output/eval_ab/<movie>/` +
`docs/eval/registry_ab_v0.json`.
