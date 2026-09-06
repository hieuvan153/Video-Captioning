## Định cỡ tín hiệu giọng điệu (Task 4)

Kiểm tra xem bản dịch phụ đề của người dịch có đổi thanh ghi xưng hô (thô vs lịch sự) trong cùng một cảnh hay không. Nếu tỉ lệ cảnh chứa cả hai thanh ghi nhỏ (< 15%), thì tín hiệu giọng điệu hầu như không tồn tại, khiến Phase 1 (mô tả âm thanh) mất phần lớn lý do tồn tại.

### Phương pháp

Định nghĩa **thanh ghi thô = xưng hô ngôi hai** (mày, tao, chúng mày, tụi mày, v.v.), không phải ngôi ba (nó, hắn, …). Lý do: người nói lịch sự vẫn dùng ngôi ba để nói về người khác; chỉ sử dụng ngôi hai mới là dấu hiệu đổi giọng điệu.

Phát hiện bằng regex trực tiếp trên text:
- Loại "mày" khi nó là "lông mày", "hàng mày", v.v. (các cụm từ ghép)
- Loại "mi" và "bay" vì chúng là đồng âm phổ biến (lông mi, động từ bay) → 2 giả dương / 0 đúng dương trên phim này

### Kết quả — Tập THO Mới (Chỉ `mày` Và `tao`, Loại Giả Dương)

Chạy trên 114 cảnh của phim tham chiếu:

- **Cảnh:** 1/73 = **1.4%**
- **Cửa sổ 6 cue:** 4/1451 = **0.3%** (sau khi sửa bug "mày ở đầu chuỗi")

### So Sánh — Tập THO Cũ (Có Ngôi Ba "nó")

Để chứng minh rằng 58.1% trước đây bị méo bởi "nó":

- **Cảnh:** 43/74 = **58.1%**
- **Cửa sổ 6 cue:** 334/1534 = **21.8%**

### Kiểm Soát Độ Dài

**THO mới (chỉ `mày` và `tao`, loại giả dương):**
- Nhóm "cả hai": 8.0 cue, 35.1 giây (chỉ 1 cảnh)
- Nhóm "chỉ một thanh ghi": 24.2 cue, 66.2 giây
- Tỉ lệ "cả hai" ở nửa cảnh ngắn: 3.1%; nửa dài: 0.0%

→ Tỉ lệ rất thấp ở cả hai nửa → Tín hiệu gần như không tồn tại, không bị chi phối độ dài.

**THO cũ (có ngôi ba):**
- Nhóm "cả hai": 32.1 cue, 84.3 giây
- Nhóm "chỉ một thanh ghi": 12.5 cue, 40.5 giây
- Tỉ lệ "cả hai" ở nửa cảnh ngắn: 9.4%; nửa dài: 72.7%

→ Tỉ lệ **bị chi phối lớn** bởi độ dài (2.6x và 72.7% ở nửa dài); sự khác biệt 9.4% vs 72.7% chứng tỏ metric đang đo độ dài, không phải giọng điệu.

### Giới Hạn & Cảnh Báo

1. **Nhánh THO rất hẹp:** Sau khi loại `mi` (đồng âm với "lông mi"), `bay` (động từ "bay"), và "lông mày", chỉ còn `mày` và `tao` thực tế được phát hiện.

2. **Phim tham chiếu này cực hiếm:** `tao` = 0 cue, `mày` = **đúng 1 cue** là đại từ (cue 1590 "Mày vừa nói gì đó?"). Con số 1.4% phần lớn là đặc điểm của **chính phim này** — một bộ phim tâm lý gia đình với giọng điệu phù hợp, chứ không nhất thiết mở rộng cho phim khác.

3. **Cần lặp lại trên các phim khác** — nhất là phim có xung đột gay gắt, các nhân vật với mối quan hệ khác nhau — trước khi coi là kết luận chung.

### Kết Luận

**Sửa lại kết luận (Important 4, review tổng) — số liệu đo được ở trên vẫn ĐÚNG, nhưng kết luận trước đây đi quá xa so với những gì đã đo:**

Phép đo này chỉ kiểm được **một trục hẹp**: xưng hô ngôi hai kiểu thô/tục
(mày, tao). Sau khi loại giả dương, tập THÔ thực tế đo được trên phim này
chỉ còn `{mày, tao}`, và bản thân phim tham chiếu gần như không có dữ liệu
để đo trục đó: `tao` = 0 cue, `mày` = 3 lần xuất hiện thô nhưng 2/3 là
"lông mày" (giả dương cụm từ), chỉ 1/3 là đại từ thật. Nói cách khác, **phép
đo gần như không có gì để đo** trên phim này — 1,4% không phải một con số
đo được đáng tin về "tín hiệu giọng điệu nói chung", mà là hệ quả trực tiếp
của việc trục mày-tao gần như vắng mặt trong phim.

**Trục thanh ghi THẬT của phim này chưa hề được đo.** Đây là một bộ phim tâm
lý gia đình, và trục xưng hô chi phối thực tế của nó là **anh / em / cô /
tôi / ông / bà** — số lần xuất hiện thô lần lượt khoảng **429 / 269 / 172 /
332** cho anh / em / cô / tôi (ông/bà chưa đếm riêng). Đây mới là nơi có thể
có biến đổi thanh ghi thật sự (vd nhân vật chuyển từ gọi "cô" sang gọi "mày"
khi giận dữ, hoặc từ "tôi" sang xưng "tao"), nhưng cũng có thể có biến đổi
tinh tế hơn (anh/em ↔ tôi/ông/bà) mà phép đo hiện tại — chỉ nhìn ngôi hai
thô/tục — hoàn toàn bỏ qua.

**Kết luận đúng với dữ liệu:** phép đo này chỉ cho biết trục thô/lịch-sự
kiểu mày-tao gần như không xuất hiện trong phim tham chiếu. Nó **KHÔNG đủ
căn cứ để bỏ Phase 1 (mô tả âm thanh)** hay để kết luận "tín hiệu giọng điệu
gần như không tồn tại" nói chung, vì trục xưng hô thực sự chi phối phim này
(anh/em/cô/tôi/ông/bà) chưa từng được kiểm. **Việc còn để ngỏ:** nếu sau này
có ai muốn xét lại hướng audio (Phase 1), việc cần làm trước là đo lại chính
phép kiểm tra "cảnh chứa cả hai thanh ghi" nhưng trên trục anh/em/cô/tôi/
ông/bà (hoặc trục thanh ghi phù hợp với phim đó), không phải trên mày/tao.

(Quyết định "dừng Phase 1" trong kế hoạch tổng thể vẫn đứng — nhưng dựa trên
đòn bẩy ASR lớn hơn nhiều lần và cổng G1 trượt, KHÔNG dựa trên kết quả Task
4 này. Task 4 không còn được dùng làm căn cứ độc lập để bỏ hướng audio.)

## Guard cấp chunk (Task 3c)

### Vấn đề

Arm `base_repro` (mốc chính của cổng G1) bị sập khối ở 2 chunk: `scene_index`
16 (`n_src`=62) và 52 (`n_src`=59) — cả hai model đều sinh ra 147 dòng và mọi
dòng đều collapse về đúng chuỗi `- Không. - Không.`. `output_guard.is_degenerate_line`
chỉ soi lặp n-gram TRONG một dòng nên không bắt được sập khối (mỗi dòng lặp
lại đều "hợp lệ" khi xét riêng lẻ). Kết quả: 121/1939 cue thành rác, không
chốt được cổng.

`refine_llm.py` dùng `do_sample=False` (greedy) nên chạy lại cho y hệt kết
quả — không cần GPU để sửa dữ liệu đã có, chỉ cần dựng lại SRT từ debug JSON
`*.srt.json` mà `refine_llm.py` đã lưu kèm mỗi arm.

### Hai luật guard cấp chunk (`is_degenerate_chunk`, `demo/LLM/output_guard.py`)

**Round 1 (2026-09-06) đã BỊ BÁC BỎ ở round 2, cùng ngày.** Round 1 từng chia
luật 1 thành 3 mức theo tỉ lệ lệch, cho phép lệch ±20% "an toàn" đi qua không
fallback — với lý do scene 28 (26 vs 27 dòng) kiểm tra vài dòng đầu thấy bình
thường. Reviewer đọc lại NGUYÊN VĂN nội dung `translations` trong debug JSON
(không suy đoán từ số dòng) và phát hiện kết luận đó SAI:

- **Scene 28 (base_repro, 26 vs 27 dòng, "lệch nhẹ" round 1 coi là an toàn):**
  model gộp 2 dòng nguồn thành 1 dòng ở vị trí 6 (`subtitle_index`=488).
  Reviewer báo cáo ~19/27 dòng (~70% chunk) từ vị trí 7 trở đi bị gán NHẦM
  câu thoại của cue kế tiếp. Tự kiểm lại xác nhận ĐÚNG hướng: đọc thủ công
  `translations[6:]` cho thấy nội dung từ vị trí 7 rõ ràng tương ứng với
  `rough_vietnamese` của vị trí kế tiếp (`refined[k]` ≈ `rough[k+1]`, có
  pronoun được refiner sửa nên không phải lúc nào cũng khớp CHÍNH XÁC từng
  ký tự — so khớp chuỗi y hệt chỉ bắt được 5/27 vị trí, phần còn lại khớp về
  nội dung/ngữ nghĩa nhưng khác vài từ do refiner paraphrase). Kết luận
  chung — chunk bị lệch cue thật sự sau điểm gộp — được xác nhận độc lập;
  con số phần trăm chính xác (19/27) đến từ reviewer, chưa đo lại bằng số
  đúng bằng phương pháp tự động. Round 0 "kiểm thủ công thấy bình thường" vì
  chỉ nhìn 5-6 dòng đầu — trước điểm gộp.
- **Scene 61 (base_repro, 86 vs 91 dòng, cũng "lệch nhẹ" ở round 1):** ĐÃ XÁC
  MINH ĐỘC LẬP, không còn là nghi vấn. Kiểm tra ban đầu (round 2, Task 3c)
  chỉ quét độ lệch vị trí -3..+3 nên bỏ lọt tín hiệu thật. Mở rộng quét ra
  -10..+10 (so khớp chuỗi y hệt `refined_vietnamese[i]` với
  `rough_vietnamese[i+shift]` sau chuẩn hoá khoảng trắng + lowercase, script
  kiểm tra ad-hoc trong Task 3b) cho kết quả: mức nền ở shift=0 chỉ 5/91 khớp
  (đúng như ghi nhận cũ), nhưng ở **shift=+5 có 22/86 khớp** — cao hơn hẳn
  mọi shift khác (shift kế cao nhất chỉ 1/84-1/86) — một đỉnh rõ ràng, cùng
  bản chất với scene 28. Kết luận: chunk 61 hỏng do **lệch offset hằng số
  +5** (5 dòng nguồn bị model bỏ/gộp ở đâu đó gần đầu chunk khiến mọi dòng từ
  điểm đó trở đi bị đẩy lệch 5 vị trí so với rough), không phải lệch rải rác
  như nghi ngờ ban đầu. Fallback cả chunk là quyết định đúng.
- Cả 9 chunk "lệch nhẹ" của round 1 (382 cue gộp 2 arm) đều có tỉ lệ dòng
  khớp CHÍNH XÁC với rough rất thấp khi tự kiểm lại: base_repro scene 28
  2/27, 50 5/37, 51 12/33, 61 5/91, 72 3/18; oracle scene 28 5/27, 51 12/33,
  52 0/59, 80 2/57 — dải quan sát được đúng như "0/59 đến 12/33" reviewer
  nêu. Tỉ lệ thấp này MỘT PHẦN là bản chất của refiner (nó được thiết kế để
  SỬA lời rough, không copy y nguyên), nên tự nó không chứng minh dòng nào
  cũng sai — bằng chứng cứng (offset rõ ràng, gán sai cue) chỉ được xác minh
  trực tiếp ở scene 28.
- Scene 80 (base_repro, ratio 0.75x — vùng "giữa" ở round 1, lúc đó tôi ghi
  "chưa kiểm tra thủ công"): reviewer đã kiểm — **hỏng thật**, fallback là
  quyết định đúng, không còn là câu hỏi mở.

**Bài học:** đếm SỐ LƯỢNG dòng không phát hiện được lệch VỊ TRÍ (offset).
Chỉ cần model gộp/tách MỘT dòng ở đầu hay giữa chunk là mọi cue sau điểm đó
gán sai nội dung, trong khi tỉ lệ lệch tổng thể có thể chỉ ~1% và rơi thẳng
vào bất kỳ vùng "an toàn" nào được định nghĩa bằng ngưỡng tỉ lệ. Không có
ngưỡng tỉ lệ nào an toàn — kết luận round 2: **quay lại luật chặt, không
ngoại lệ.**

1. **Lệch số dòng**: `n_model_lines != n_src` (đo trên `refined_lines`
   TRƯỚC khi pad/cắt về `n_src`) → fallback CẢ CHUNK, không có ngưỡng dung
   sai, không ngoại lệ. Chuỗi lý do vẫn ghi kèm `ratio` để dễ đọc log/phân
   loại thủ công về sau, nhưng **hành vi giống nhau cho mọi mức lệch**.
   Đường nâng cấp đúng đắn hơn (chưa làm — cổng đo hiện tại chỉ cần so sánh
   công bằng giữa các arm, không cần tối đa hoá recall bản dịch tốt): gióng
   đơn điệu (monotonic alignment) output với rough theo chrF từng dòng thay
   vì đếm số dòng — mục B1 bước 2,
   `docs/superpowers/plans/2026-09-02-roadmap-cai-tien.md`.
2. **Trùng lặp toàn khối**: chunk có >= `min_lines` (mặc định 5) dòng và
   >= `same_ratio` (mặc định 0.8) tỉ lệ dòng GIỐNG HỆT NHAU (sau chuẩn hoá
   khoảng trắng + lowercase) → sập khối, fallback cả chunk. Không đổi qua
   cả 2 round — chỉ áp dụng khi luật 1 không kích hoạt (số dòng model khớp
   đúng `n_src`).

Thêm `has_prompt_leak`: bắt dòng chứa nguyên văn tag của prompt (`<Scene
Context>`, `<Rough Vietnamese Translation>`, `<English Dialogue>`, …) — dấu
hiệu lỗi parse output của model, áp dụng ở cấp DÒNG cùng với
`is_degenerate_line`.

Script dựng lại post-hoc: `demo/EVAL/rebuild_guarded_srt.py` — đọc debug
JSON, áp `is_degenerate_chunk` cho từng chunk (cả chunk fallback về
`rough_vietnamese` nếu có lý do) rồi `has_prompt_leak`/`is_degenerate_line`
cho từng dòng còn lại, dựng SRT mới lấy timing từ `--rough_srt` theo
`subtitle_index`.

### Kiểm chéo bắt buộc

Dựng lại arm `oracle` với `--no_guard` (tắt toàn bộ guard) và so với
`output_oracle/oracle.srt` gốc: **1939/1939 cue khớp tuyệt đối, 0 cue lệch**
(`diff` cấp byte cũng cho hai file giống hệt nhau). Đã kiểm tra lại sau cả
round 1 và round 2, kết quả KHÔNG đổi (vẫn 0/1939 lệch) — script dựng lại
không phụ thuộc vào việc luật 1 có dung sai hay không.

### Kết quả áp guard cho từng arm (round 2 — luật chặt, giống round 0)

| Arm | Chunk fallback | Chunk tổng | Cue fallback | Cue tổng |
|---|---|---|---|---|
| `base_repro` | 8 | 95 | 384 (19.8%) | 1939 |
| `oracle` | 5 | 95 | 238 (12.3%) | 1939 |

Chi tiết chunk fallback (tất cả do luật 1 — lệch số dòng; không chunk nào do
luật 2 trùng lặp toàn khối):

| Arm | scene_index | n_src | n_model_lines | ratio | Ghi chú |
|---|---|---|---|---|---|
| `base_repro` | 16 | 62 | 147 | 2.37 | Sập khối thật — mọi dòng collapse `- Không. - Không.` |
| `base_repro` | 28 | 27 | 26 | 0.96 | **Đã kiểm nội dung, xác nhận: lệch 1 vị trí sau điểm model gộp 2 dòng nguồn thành 1 (~19/27 dòng theo reviewer)** |
| `base_repro` | 50 | 37 | 36 | 0.97 | Chưa kiểm nội dung chi tiết, cùng dạng lệch số dòng |
| `base_repro` | 51 | 33 | 32 | 0.97 | Chưa kiểm nội dung chi tiết |
| `base_repro` | 52 | 59 | 147 | 2.49 | Sập khối thật, giống scene 16 |
| `base_repro` | 61 | 91 | 86 | 0.95 | Reviewer báo cáo có mẫu rác lặp câu; tự kiểm lại KHÔNG thấy đúng chuỗi đó, xem "Nghi vấn" |
| `base_repro` | 72 | 18 | 17 | 0.94 | Chưa kiểm nội dung chi tiết |
| `base_repro` | 80 | 57 | 43 | 0.75 | **Đã kiểm: hỏng thật, fallback đúng** (đóng nghi vấn round 1) |
| `oracle` | 16 | 62 | 100 | 1.61 | Sập khối thật |
| `oracle` | 28 | 27 | 26 | 0.96 | Cùng chunk lệch vị trí như base_repro (bản dịch khác nhưng cùng cấu trúc gộp dòng) |
| `oracle` | 51 | 33 | 32 | 0.97 | — |
| `oracle` | 52 | 59 | 61 | 1.03 | Chunk chứa cue rò rỉ tag `<Rough Vietnamese Translation>` (xem mục dưới) |
| `oracle` | 80 | 57 | 56 | 0.98 | — |

**Hệ quả:** `base_repro` mất 384/1939 cue (19.8%) về rough, `oracle` mất
238/1939 (12.3%) — CHÊNH NHAU 7.5 điểm phần trăm. Vì luật guard chỉ phụ
thuộc lệch số dòng (một hiện tượng ngẫu nhiên của lần generate, không liên
quan gì đến việc arm nào đặt đại từ đúng hơn), con số cổng G1 tính trên
TOÀN BỘ 1939 cue của mỗi arm sẽ bị méo: arm nào "may mắn" lệch số dòng ít
hơn sẽ có vẻ tốt hơn dù chất lượng đại từ giống hệt. **Cổng chính phải tính
trên tập chunk mà CẢ HAI arm đều không fallback** (xem mục dưới) để loại bỏ
nhiễu này.

### Tập chunk chung giữa hai arm (`$T/common_chunks.json`)

Hợp 2 tập fallback: `base_repro` = {16, 28, 50, 51, 52, 61, 72, 80} (8 chunk),
`oracle` = {16, 28, 51, 52, 80} (5 chunk) → hợp = {16, 28, 50, 51, 52, 61, 72,
80} (8 chunk, vì tập fallback của `oracle` là tập con của `base_repro`).

**87/95 chunk chung, 1555/1939 cue chung** (8 chunk / 384 cue bị loại vì ít
nhất một arm fallback ở đó). Dùng tập 1555 cue này để tính cổng G1 — so sánh
công bằng "arm nào đặt đại từ đúng hơn" mà không lẫn "arm nào tránh được cú
sập/lệch dòng ở lần generate này".

### Cue rò rỉ prompt tag tìm được

Arm `oracle`, `scene_index`=52, `subtitle_index`=1002: `refined_vietnamese`
đúng bằng chuỗi `<Rough Vietnamese Translation>` — lỗi parse output của
model lọt thẳng ra `output_oracle/oracle.srt` gốc. Ở round 2 (luật chặt),
chunk 52 của `oracle` (59 vs 61 dòng, ratio 1.03) **fallback do luật 1** như
mọi lệch số dòng khác — cue 1002 được dọn về rough qua đường chunk-level,
không cần đến `has_prompt_leak`. (Ở round 1, chunk này từng rơi vào vùng
"lệch nhẹ" và không fallback ở cấp chunk — khi đó `has_prompt_leak` ở cấp
dòng là thứ duy nhất dọn được cue này; round 1 đã bị bác bỏ nên chi tiết đó
không còn áp dụng, nhưng vẫn là bằng chứng `has_prompt_leak` hoạt động đúng
độc lập với guard cấp chunk.)

### Nghi vấn / giới hạn

- File debug JSON thực tế (`*.srt.json`) lưu `translations` là JSON lồng
  hợp lệ (list dict), KHÔNG phải chuỗi repr Python như mô tả trong brief gốc
  của task này — `json.load` đọc thẳng được. `rebuild_guarded_srt.py` vẫn
  giữ nhánh `ast.literal_eval` để tương thích ngược nếu có bản debug JSON cũ
  ở định dạng chuỗi repr.
- Luật chặt (round 2) vứt cả dòng tốt nằm TRONG một chunk lệch số dòng (vd
  5-6 dòng đầu của scene 28 vẫn đúng trước điểm gộp) — đây là đánh đổi có
  chủ đích, chấp nhận mất một số cue tốt để đảm bảo KHÔNG cue nào bị gán sai
  cue lọt qua guard. Đường nâng cấp (chrF gióng đơn điệu từng dòng) đã ghi
  trong docstring `is_degenerate_chunk`, chưa triển khai.
  ponytail: cách thô (fallback cả chunk khi lệch số dòng) là global — nâng
  cấp lên chrF alignment từng dòng nếu cần tối đa hoá recall bản dịch tốt.
- Các chunk fallback chưa liệt kê "đã kiểm nội dung" ở bảng trên (scene 50,
  51, 72, oracle 51/28) chưa được đọc thủ công từng dòng như scene 28/80 —
  chỉ dựa trên lệch số dòng để suy luận, phù hợp với chủ trương "mọi lệch
  đều không đáng tin" của round 2 nên không cần kiểm để quyết định fallback,
  nhưng nếu cần định lượng chính xác mức độ hỏng thực tế của từng chunk thì
  đây là việc còn để ngỏ.
- **Claim "mẫu rác `- Tôi không. - Tôi không.`" ở scene 61 (base_repro):**
  vẫn CHƯA xác minh được đúng chuỗi rác đó — grep không tìm thấy. Nhưng phần
  còn lại của nghi vấn ("không tìm thấy độ lệch hằng số rõ ràng") ĐÃ ĐƯỢC GIẢI
  QUYẾT ở Task 3b: nguyên nhân là quét round 2 chỉ thử shift -3..+3, bỏ lọt
  tín hiệu thật ở shift=+5 (22/86 khớp, nền 5/91 ở shift=0 — xem mục Task 3c
  ở trên). Vậy scene 61 hỏng do lệch offset +5, không phải lệch rải rác; chỉ
  riêng câu trích dẫn nguyên văn "- Tôi không. - Tôi không." của reviewer là
  còn chưa xác minh được, không ảnh hưởng tới chẩn đoán chung.
- Kiểm nhanh scene 80 (base_repro) cho thấy dòng đầu tiên của
  `refined_vietnamese` bắt đầu bằng ký tự `<` mồ côi (`'<Cái gì? Để trốn
  sau, để thao túng.'`) — dấu hiệu mảnh vỡ của một tag prompt bị cắt ngang,
  cộng thêm 2 dòng liên tiếp gần trùng nội dung (1725, 1726) — củng cố độc
  lập cho kết luận "hỏng thật" của reviewer.

## Đo trần kênh Scene Context (Task 3)

Cổng G1 quyết định có tiếp tục hai hướng nghiên cứu đang cân nhắc (đổi VLM
sang mô tả chuyên nhân vật; thêm mô tả âm thanh + huấn luyện DPO) hay dừng cả
hai và chuyển hẳn sang cải thiện ASR. Cách đo: arm `oracle` bơm thẳng ĐÁP ÁN
xưng hô (từ `oracle.captions.json`, mục "Relationship") vào kênh
`<Scene Context>` của prompt Gemma. Nếu bơm đáp án mà PronF1 không tăng đủ
ngưỡng đặt trước, kênh này không còn trần để leo — mọi kỹ thuật lấy thông tin
THẬT (không phải đáp án) chỉ có thể đạt một phần con số đó.

**Đọc lại phạm vi của phép đo (Important 2 + 3, review tổng — quan trọng,
đừng bỏ qua):** cách viết ở trên dễ khiến người đọc hiểu là đã đo "trần của
kênh Scene Context" nói chung. Thực tế hẹp hơn nhiều: đã đo trần của **một
cách trình bày cụ thể** (danh sách từ xưng hô phẳng, không cấu trúc — "3.
Relationship: [Speakers] - address terms used in this scene: anh, em, …",
trung vị 7 từ/cảnh, tối đa 23 từ, có lẫn giả dương của lexicon đại từ như
`chúa`, `tên`, `các`, `người`) đi qua **một adapter cụ thể**
(`thevan2404/best_gemma_scene_context`) mà bằng chứng cho thấy gần như không
dùng đến loại thông tin này (xem mục "Đã đo trần của CÁCH TRÌNH BÀY nào,
không phải trần của KÊNH" ngay sau Step 4). Đây KHÔNG phải trần của *thông
tin* xưng hô nói chung — xem phần diễn giải đầy đủ ở đó và ở mục "Đọc kết
quả" bên dưới.

Ba arm, dùng bản `.guarded.srt` (đã qua guard cấp chunk, xem mục trên):

| arm | ý nghĩa |
|---|---|
| `rough` | dịch thô mBART, chưa qua LLM |
| `base_repro` | **MỐC**: Gemma refine với caption VLM gốc |
| `oracle` | Gemma refine với caption chứa đáp án xưng hô |

### Step 1 — Kiểm toàn vẹn trước khi chấm

Số cue của cả ba arm (dựng lại từ `.guarded.srt`) bằng nhau tuyệt đối:

| arm | n cue | dòng lặp nhiều nhất |
|---|---|---|
| rough | 1939 | 27 lần ("Ừ.") |
| base_repro | 1939 | 26 lần ("Ừ.") |
| oracle | 1939 | 24 lần ("Được rồi.") |

**Sửa lại lập luận (fix round 1):** 24, 26, 27 đều LỚN HƠN ngưỡng 15 mà brief
đặt ra — đọc đúng chữ của brief thì bước này lẽ ra phải kích hoạt STOP. Ngưỡng
tuyệt đối 15 không phù hợp với phim này: `rough` là dịch thô mBART, CHƯA hề
qua LLM nên không thể sập khối theo nghĩa Task 3c mô tả, vậy mà tự nó đã có
27 lần lặp "Ừ." — đơn giản vì đó là một từ đệm rất tự nhiên, lặp lại nhiều
trong hội thoại thật của phim. Ngưỡng 15 áp dụng máy móc sẽ coi cả `rough`
là "sập khối", vô lý. Căn cứ đúng để kết luận "không sập khối" là **so với
mốc tự nhiên của `rough`**: `base_repro` (26) và `oracle` (24) đều THẤP HƠN
mốc `rough` (27), tức guard không hề làm phát sinh lặp MỚI so với mức nền tự
nhiên của phim — không có dấu hiệu sập khối bổ sung nào từ hai arm qua LLM.
Ghi rõ để người dùng lại: ngưỡng "15 lần" trong brief là con số đặt ra khi
viết kế hoạch, KHÔNG dựa trên đặc điểm phim cụ thể — cần thay bằng so sánh
tương đối với `rough` (hoặc một ngưỡng riêng theo phim) ở các lần chấm sau.

Đối chứng `base_repro` (bản dev) với `.04_VI_tinh_chinh_Gemma_batch1.srt`
(bản refine_llm.py GỐC, chỗ DUY NHẤT dùng file `.04`):

| arm | PronF1 neo | delta_vs_base(`.04`) | CI95 |
|---|---|---|---|
| `.04` (old, refine_llm.py gốc) | 0,3666 | — | — |
| `base_repro` (dev, đã qua guard) | 0,3815 | **+0,0149** | **[0,0062, 0,0237]** |

**Lệch có ý nghĩa thống kê (CI95 không trùm 0)** — bản dev khác hành vi so
với bản gốc, KHÔNG "gần 0" như brief kỳ vọng. `base_repro` (dev, ĐÃ qua
guard) đo cao hơn `.04` (gốc, không qua guard) 0,0149 PronF1.

**Nguyên nhân đã xác định (fix round 1, không phải hành vi model khác
nhau):** so `.04` với `base_repro` bản **CHƯA guard** (`base_repro.srt` thô,
trước khi `rebuild_guarded_srt.py` chạy) thì dấu LẬT NGƯỢC hoàn toàn:
`delta = -0,0307`, CI95 `[-0,0426, -0,0199]` — tự kiểm chạy lại xác nhận
đúng con số này. Vậy toàn bộ cú lật dấu +0,0149 (guard) so với -0,0307
(chưa guard) là do RIÊNG guard cấp chunk gây ra, không phải do hai pipeline
sinh chữ khác nhau.

**Phát hiện đáng cảnh báo, NGOÀI PHẠM VI cổng G1 này:** bản dev sập
khối/lệch dòng nhiều hơn hẳn bản gốc của tác giả. Bằng chứng tự kiểm lại:
- `debug_Gemma_batch1.srt.json` (nguồn `.04`, không có field `n_src`/
  `n_model_lines` nên không kiểm trực tiếp được luật 1; áp luật 2 — trùng
  lặp nội dung — lên `refined_vietnamese` từng scene) cho **0/95 scene** bị
  gắn cờ trùng lặp/sập khối.
- `base_repro.srt.json` (dev, raw, có đủ field) có **8/95 scene lệch số
  dòng thật (luật 1)**, 2 trong số đó sập khối rõ ràng (scene 16, 52, tỉ lệ
  tới 2,37x và 2,49x).
- Fallback CẤP CUE (`fallback_used=True`, cờ gốc của chính
  `refine_llm.py` khi model sinh THIẾU dòng — không phải guard cấp chunk
  của Task 3c): `.04` = 89/1939 cue; `base_repro` dev raw = 23/1939 cue.
  Số fallback cấp cue của dev THẤP hơn (23 so với 89) — dev hiếm khi sinh
  THIẾU dòng hơn `.04` — nhưng dev lại là bản DUY NHẤT có kiểu lỗi sinh
  THỪA dòng nghiêm trọng (2,37x-2,49x, sập khối) mà `.04` không có case nào
  bị gắn cờ trùng lặp.

Kết luận: **nghi vấn KHÔNG làm hỏng cổng G1** — `oracle` và `base_repro`
chạy cùng codebase dev, cùng cơ chế guard, chỉ khác caption đầu vào, nên so
sánh `oracle` − `base_repro` là công bằng bất kể pipeline dev có dễ sập khối
hơn `.04` hay không. Nhưng đây LÀ một phát hiện độc lập cần điều tra tiếp,
ngoài phạm vi kế hoạch Task 3: **vì sao bản dev dễ sinh THỪA dòng/sập khối
cấp chunk hơn hẳn bản gốc của tác giả** — khác biệt code giữa hai bản
`refine_llm.py` (dev có thể đã đổi prompt, cách chia batch, hoặc
`max_new_tokens`) là nghi phạm hàng đầu, chưa xác minh trong task này.

### Step 2 & 2b — PronF1 neo theo cue tham chiếu

**Toàn arm** (1776 cue tham chiếu, `$T/g1_pron.json`):

| arm | PronF1 neo | delta_vs_base | CI95 (theo cue) | CI95 (cụm cảnh) |
|---|---|---|---|---|
| rough | 0,3667 | −0,0148 | [−0,0259, −0,0027] | [−0,0370, +0,0049] |
| base_repro | 0,3815 | — (mốc) | — | — |
| oracle | 0,3974 | **+0,0159** | **[+0,0084, +0,0230]** | **[+0,0047, +0,0278]** |

Cột "CI95 (theo cue)" lấy mẫu lại từng cue độc lập (`bootstrap`, mặc định). Cột
"CI95 (cụm cảnh)" lấy mẫu lại theo **cảnh** (`bootstrap_cluster`, cờ
`--cluster_by_scene oracle.captions.json`, dùng `start_time`/`end_time` của
114 cảnh VLM) — xem mục "Sửa bootstrap theo cụm cảnh" dưới bảng Step 2b để
biết vì sao cần cột này: ngữ cảnh oracle gán theo CẢNH, mọi cue trong một
cảnh dùng chung một khối ngữ cảnh nên tương quan với nhau, và cột "theo cue"
coi 1776 cue là 1776 đơn vị độc lập trong khi n hiệu dụng thật chỉ ~86 cảnh
(số cảnh khác nhau xuất hiện trong 1776 cue được chấm).

**Tập chunk CHUNG — con số cổng CHÍNH** (87/95 chunk, 1555 subtitle_index mà
CẢ HAI arm đều không fallback → lọc còn 1364/1776 cue tham chiếu giao thời
gian trọn vẹn với tập đó, `$T/g1_pron_common.json`, script mới
`demo/EVAL/pron_anchored_subset.py`):

| arm | PronF1 neo (tập chung) | delta_vs_base | CI95 (theo cue) | CI95 (cụm cảnh) |
|---|---|---|---|---|
| rough | 0,3686 | −0,0193 | [−0,0334, −0,0031] | [−0,0449, +0,0051] |
| base_repro | 0,3879 | — (mốc) | — | — |
| oracle | 0,3996 | **+0,0117** | **[+0,0038, +0,0197]** | **[+0,0021, +0,0233]** |

### Sửa bootstrap theo cụm cảnh (Important 1, review tổng)

`bootstrap()` gốc (và `pron_anchored_subset.py`) lấy mẫu lại theo **cue** độc
lập. Nhưng ngữ cảnh oracle được bơm theo **cảnh**: mọi cue trong cùng một
cảnh dùng chung một khối `<Scene Context>`, nên đúng/sai đại từ của các cue
đó tương quan với nhau — không phải 1364 (hay 1776) đơn vị độc lập như
bootstrap theo cue giả định, mà chỉ ~75-86 cảnh hiệu dụng. Bootstrap theo cue
đánh giá THẤP bất định thật, cho CI hẹp hơn thực tế.

Đã thêm `bootstrap_cluster()` (`demo/EVAL/pron_anchored.py`) và cờ CLI
`--cluster_by_scene <captions.json>` (cả `pron_anchored.py` và
`pron_anchored_subset.py`): lấy mẫu lại các **cảnh** có hoàn lại (dùng
`start_time`/`end_time` của captions JSON để gán cue vào cảnh, cùng quy tắc
thời gian với `scene_terms` trong `make_oracle_context.py`), rồi gộp toàn bộ
cue của các cảnh được chọn. Bootstrap theo cue vẫn là **mặc định** (không
đổi hành vi/con số khi không truyền cờ mới); cụm cảnh là tuỳ chọn thêm.

Kết quả trên tập chung (75 cảnh hiệu dụng / 1364 cue) khớp gần như tuyệt đối
với con số review tổng đã tính trước (`[+0,0021, +0,0239]`): tái lập được
`[+0,0021, +0,0233]`, lệch 0,0006 ở cận trên do khác biệt làm tròn/seed nhỏ,
không đổi kết luận. Trên toàn arm (86 cảnh hiệu dụng / 1776 cue), tái lập
`[+0,0047, +0,0278]` so với con số đối chiếu `[+0,0052, +0,0293]` — cùng
chiều, cùng kết luận (cận trên vượt ngưỡng +0,02), lệch nhiều hơn một chút so
với tập chung, có thể do 412 cue chênh lệch giữa hai tập rơi vào các cảnh
khác biệt về mức tương quan nội bộ; đã báo lại chênh lệch này thay vì âm
thầm sửa cho khớp.

**Hệ quả quan trọng nhất:** CI95 cụm cảnh của tập chung là `[+0,0021,
+0,0239]` — cận trên **VƯỢT** ngưỡng +0,02 đặt trước khi đo. Nghĩa là dữ
liệu **KHÔNG loại trừ được** khả năng trần thật của kênh (theo cách trình
bày và adapter này) cao hơn ngưỡng +0,02. Hiệu ứng vẫn khác 0 (cận dưới
+0,0021 > 0), nhưng khoảng tin cậy rộng hơn nhiều so với con số theo-cue đã
khiến tài liệu trước đây tuyên bố chắc chắn hơn mức dữ liệu cho phép — xem
lại cách đọc kết luận ở mục "Cổng G1 — Phán quyết" và "Đọc kết quả" bên
dưới.

Lý do cần tập chung: `base_repro` fallback 8/95 chunk (384 cue), `oracle`
fallback 5/95 chunk (238 cue) — hai tập fallback KHÁC NHAU (fallback của
`oracle` là tập con của `base_repro`, chênh 146 cue). Tính trên toàn bộ
1939/1776 cue sẽ lẫn "arm nào tránh được cú sập/lệch dòng ở lần generate
này" (ngẫu nhiên) vào "arm nào đặt đại từ đúng hơn" (cái cần đo). Tập chung
loại bỏ nhiễu này.

**Vì sao 1776 → 1364 (mất 412 cue tham chiếu):** tách theo lý do loại, **371
cue** bị loại vì giao thời gian với ÍT NHẤT một cue nguồn (rough) thuộc
chunk fallback (đúng mục đích lọc — tránh lẫn vùng fallback vào tập chung);
**41 cue** còn lại bị loại vì KHÔNG giao thời gian với BẤT KỲ cue rough nào
cả (lệch timing sẵn có giữa file phụ đề tham chiếu `vi_3rd.clean.srt` và
`rough` — khoảng trống giữa hai cue liền kề của rough rơi đúng vào một cue
tham chiếu). Nhóm 41 cue này ảnh hưởng CẢ BA arm như nhau (rough/base_repro/
oracle đều dùng chung một mốc thời gian nguồn) nên không thiên lệch kết quả
so sánh giữa các arm, chỉ làm giảm cỡ mẫu một chút.

Số trên tập chung (+0,0117) THẤP HƠN số toàn arm
(+0,0159) — chênh 0,0042, cùng chiều dương, không mâu thuẫn; cách đọc hợp lý
nhất là một phần chênh lệch toàn-arm đến từ đúng loại nhiễu mà tập chung
được thiết kế để loại bỏ (các chunk fallback khác nhau giữa hai arm), chứ
không phải bằng chứng oracle "thật ra tệ hơn" — cả hai con số đều dương và
CI95 đều không trùm 0.

### Step 3 — Chất lượng dịch neo theo tham chiếu (BLEU/chrF/COMET-DA)

1767/1776 cue tham chiếu có nguồn EN (`$T/g1_quality.json`):

| arm | BLEU | chrF | COMET-DA |
|---|---|---|---|
| rough | 18,34 | 37,55 | 0,7487 |
| base_repro | 18,40 | 37,74 | 0,7509 |
| oracle | 18,78 | 38,35 | 0,7541 |

Paired bootstrap 1000 vòng so với `base_repro`:

| so sánh | BLEU | chrF | COMET-DA |
|---|---|---|---|
| rough − base_repro | −0,05 [−0,47, +0,38] | −0,19 [−0,58, +0,19] | −0,0022 [−0,0045, +0,0001] |
| oracle − base_repro | +0,39 [+0,08, +0,70] | **+0,62 [+0,28, +0,94]** | **+0,0032 [+0,0011, +0,0054]** |

`oracle` cao hơn `base_repro` có ý nghĩa thống kê ở cả chrF và COMET-DA (CI95
không trùm 0), cùng chiều dương với PronF1. Không có tín hiệu nào cho thấy
oracle "đánh đổi" chất lượng dịch chung để lấy PronF1 — cả hai cùng tăng.

**Không đọc BLEU làm căn cứ chính** (theo ràng buộc của brief) — chỉ nêu để
đối chiếu; BLEU neo theo tham chiếu vốn kém tin cậy khi độ dài cue giữa các
arm khác nhau (xem `danh_gia_lai_3rd_party.md`).

### Step 4 — oracle thay đổi những gì (`demo/EVAL/oracle_delta.py`)

Trong 1776 cue tham chiếu, `oracle` khác `base_repro` (sau chuẩn hoá khoảng
trắng) ở **384 cue**. Trong số đó:

| | số cue |
|---|---|
| oracle khớp đại từ TỐT HƠN | 61 |
| oracle khớp đại từ TỆ HƠN | 29 |
| hoà (cùng F1, cả hai đều có đại từ) | 249 |
| hoà (cả hai đều không có đại từ nào) | 45 |

**Thay đổi RÒNG rất nhỏ: 61 tốt lên, 29 tệ đi, ròng +32/1776 cue (1,8% tổng
số cue)** — 384 cue oracle "làm khác" nhưng phần lớn (294/384, các dòng hoà)
không đổi độ đúng đại từ; phần còn lại nghiêng nhẹ về hướng tốt (tỉ lệ
tốt:tệ ≈ 2,1:1) nhưng số tuyệt đối nhỏ trên nền 1776 cue.

**Kiểm tra bổ sung (không có trong script, làm ad-hoc để diễn giải số 384):**
con số 384 trùng khớp CHÍNH XÁC với số cue mà `base_repro` fallback về rough
— không phải ngẫu nhiên hoàn toàn nhưng cũng không phải toàn bộ do fallback:
đối chiếu với `guard_report.json` của hai arm, trong 384 cue khác nhau đó có
**111 cue** chỉ khác nhau vì `base_repro` fell back về rough còn `oracle`
không (so sánh "rough vs refined", không phải "prompt A vs prompt B" —
kém sạch), và **272 cue** là so sánh SẠCH (cả hai arm đều refine bình
thường, không fallback — đây mới là bằng chứng trực tiếp cho tác dụng của
caption oracle). Trên 272 cue sạch: 38 tốt hơn, 20 tệ hơn, ròng +18/272
(6,6%). Trên 111 cue lẫn fallback: 23 tốt hơn, 9 tệ hơn. Cả hai tập con đều
nghiêng cùng chiều dương, nên kết luận "oracle nhỉnh hơn nhưng ròng nhỏ"
đứng vững dù tách nhiễu fallback hay không.

**Sửa lỗi cộng (Minor, review tổng):** 111 + 272 = 383, không phải 384 như
tổng ban đầu ngụ ý. Tự kiểm lại bằng cách gán từng cue trong 384 cue khác
nhau đó vào đúng 1 trong 3 nhóm (dùng `subtitle_index` nguồn giao thời gian
với mỗi cue tham chiếu, đối chiếu `cue_fallback_ranges` — khoảng ĐÓNG cả hai
đầu — của `guard_report.json` từng arm): xác nhận đúng 111 cue "chỉ
`base_repro` fallback" và đúng 272 cue "cả hai sạch", còn thiếu **1 cue**
(subtitle_index tham chiếu quanh 1042, giao thời gian với nguồn 1060–1061)
thuộc loại hỗn hợp — nguồn 1060 nằm trong vùng fallback của CẢ HAI arm (chunk
scene 52), còn 1061 thì không, nên cue này không thuộc gọn về nhóm "chỉ
base_repro fallback" hay nhóm "cả hai sạch". Tổng đúng: 111 + 272 + 1 (hỗn
hợp) = **384**.

**Theo nhóm cảnh có/không có đáp án oracle:** `oracle.captions.json` có 114
cảnh, 84 cảnh mục 3 có đáp án xưng hô, 30 cảnh ghi `[None]`. Khi gán 1776 cue
tham chiếu vào 114 cảnh này theo thời gian: **1772 cue rơi vào nhóm 84 cảnh
CÓ đáp án, chỉ 4 cue rơi vào nhóm 30 cảnh KHÔNG có đáp án.** Không so sánh
được hai nhóm — nhóm "không đáp án" không đủ cỡ mẫu (n=4, 0 cue khác nhau
giữa hai arm). Lý do: 30 cảnh "[None]" chiếm 602/5619 giây (10,7% thời
lượng phim) nhưng gần như không có thoại — VLM không gán quan hệ nhân vật
chính vì đó thường là cảnh không có người nói chuyện (record player, cảnh
toàn/không nhân vật chính), nên tự nhiên có rất ít cue phụ đề trong đó.
**Không rút kết luận "cải thiện tập trung ở nhóm có đáp án" từ số liệu này**
— đó là hệ quả của cỡ mẫu lệch, không phải bằng chứng về cơ chế.

### Đã đo trần của CÁCH TRÌNH BÀY nào, không phải trần của KÊNH (Important 2 + 3, review tổng)

Ba mảnh bằng chứng dưới đây giải thích vì sao con số +0,0117 phải được đọc
là trần của "túi từ phẳng + adapter hiện tại", không phải trần của kênh
`<Scene Context>` nói chung.

**1. Phần lớn lỗi còn lại của `base_repro` là lỗi ĐẶT SAI CHỖ, không phải lỗi
THIẾU TỪ — một túi từ cấp cảnh không sửa được lỗi đặt sai chỗ.** Phân rã lỗi
dương tính giả (FP) đại từ của `base_repro` so với đúng cái túi từ oracle
đã bơm cho cảnh đó:
- **885 cue (53,6%)** lỗi FP có từ đại từ sai đó **ĐÃ NẰM SẴN TRONG túi** của
  cảnh — model đã "nhìn thấy" đúng từ nhưng gán nhầm nhân vật/vị trí. Một túi
  từ không có cấu trúc (ai gọi ai) không thể sửa loại lỗi này dù có bơm đủ
  100% đáp án, vì đáp án đã ở đó rồi mà vẫn sai.
- **767 cue (46,4%)** còn lại nằm ngoài túi — đây là lỗi từ vựng thật, loại
  mà một túi từ đúng nghĩa có thể sửa.
- **100% đại từ bị thiếu (false negative)** đều nằm trong túi của cảnh — tức
  riêng lỗi "thiếu hẳn đại từ" thì túi có đủ thông tin cần thiết.

**2. Trần của chính cách trình bày này (nếu tiêu thụ hoàn hảo) cao hơn nhiều
so với những gì arm `oracle` lấy được.** Một "người tiêu thụ hoàn hảo" —
luôn chọn đúng đại từ nằm trong túi của cảnh mỗi khi có thể — đạt PronF1
**0,6050** so với `base_repro` **0,3815**, tức dư địa lý thuyết của riêng
cách trình bày này là **+0,2235**. Arm `oracle` (Gemma refine thật, không
phải người tiêu thụ hoàn hảo) chỉ lấy được +0,0117, tức khoảng **~7%** của
dư địa đó. Nói cách khác: đo được không phải "trần của thông tin xưng hô
không còn gì để lấy", mà là "adapter hiện tại lấy được rất ít từ một cách
trình bày vốn còn nhiều dư địa".

**3. Adapter hiện tại hấp thụ kênh này rất kém — gần như không đổi hành vi
khi có đáp án.** `base_repro` (không có đáp án, chỉ có caption VLM gốc) đã
tự sinh ra **69,3%** token đại từ trùng với túi từ đúng của cảnh — nghĩa là
phần lớn "đúng đại từ" không đến từ kênh `<Scene Context>` mà từ ngữ cảnh
hội thoại/thói quen sinh chữ khác của model. Đưa thẳng đáp án vào (`oracle`)
chỉ nâng tỉ lệ này lên **71,7%** (+2,4 điểm phần trăm), và chỉ đổi TEXT ở
**19,9%** số cue — tức hơn 80% cue, dù caption có đáp án hay không, model
vẫn sinh ra CHỮ Y HỆT NHAU. Đây là dấu hiệu adapter phần lớn không dùng đến
nội dung mới trong `<Scene Context>`, không phải dấu hiệu kênh đó vô dụng.

**Hệ quả quan trọng nhất của cả kế hoạch — đừng viết nhẹ đi:** một VLM
chuyên nhân vật (character-focused captioning), thứ mà quyết định của cổng
G1 này đang loại bỏ, không sinh ra "túi từ phẳng" giống oracle — nó có khả
năng sinh ra các **cặp xưng hô CÓ HƯỚNG** (ai gọi ai bằng gì, gắn với đúng
nhân vật/lượt lời), tức một **cách mã hoá thông tin khác hẳn**, có cấu trúc
đủ để sửa đúng loại lỗi "đặt sai chỗ" chiếm 53,6% lỗi còn lại ở mục (1) —
loại lỗi mà bản thân cấu trúc túi từ phẳng không thể sửa dù có đáp án 100%.
**Trần của cách mã hoá đó CHƯA HỀ ĐƯỢC ĐO trong kế hoạch này.** Quyết định
dừng nghiên cứu ở cổng G1 vẫn đứng (xem "Đọc kết quả" bên dưới), nhưng dựa
trên điểm ước lượng và đòn bẩy ASR lớn hơn nhiều, KHÔNG dựa trên việc đã
chứng minh hướng VLM chuyên nhân vật vô dụng — hướng đó đơn giản là chưa
được đo.

### Cổng G1 — Phán quyết

Điều kiện (đặt TRƯỚC khi đo, không phải hằng số tự nhiên — ghi rõ để người
đọc tự cân nhắc mức độ nghiêm ngặt của ngưỡng):

1. Trên **tập chunk chung** (con số CHÍNH): `oracle.delta_vs_base ≥ +0,02`
   VÀ `ci95[0] > 0`.
   Đo được: **+0,0117**, CI95 theo cue `[+0,0038, +0,0197]`, CI95 **cụm cảnh**
   `[+0,0021, +0,0233]` (xem mục "Sửa bootstrap theo cụm cảnh" ở Step 2b —
   đây là căn cứ bất định đúng, vì cue trong cùng cảnh tương quan với nhau).
   → `ci95[0] > 0` ĐÚNG ở cả hai cách tính (hiệu ứng có thật, không phải
   nhiễu). Nhưng `+0,0117 < +0,02` (điểm ước lượng) → **KHÔNG ĐẠT điều kiện
   1** (chỉ đạt ~59% ngưỡng đặt trước). **Lưu ý quan trọng:** CI95 cụm cảnh
   có cận TRÊN là `+0,0233`, VƯỢT ngưỡng +0,02 — nghĩa là bản thân khoảng
   tin cậy KHÔNG loại trừ được khả năng trần thật ≥ +0,02. Điều kiện 1 trượt
   vì ĐIỂM ƯỚC LƯỢNG (+0,0117) dưới ngưỡng, không phải vì đã chứng minh được
   rằng trần thật nằm dưới ngưỡng.
2. COMET-DA của `oracle` không thấp hơn `base_repro` quá 0,005.
   Đo được: `oracle` 0,7541 so với `base_repro` 0,7509 — **CAO HƠN** 0,0032,
   không hề thấp hơn.
   → **ĐẠT điều kiện 2.**

Cổng yêu cầu **CẢ HAI** đúng. Điều kiện 1 trượt → **G1: KHÔNG ĐẠT** (dựa trên
điểm ước lượng, xem giới hạn của cách đọc này ở mục "Đọc kết quả" ngay sau).

Con số toàn arm (`+0,0159`, CI95 theo cue `[+0,0084, +0,0230]`, CI95 cụm cảnh
`[+0,0047, +0,0278]`) cũng dưới ngưỡng +0,02 ở điểm ước lượng nếu dùng làm
căn cứ thay thế — không đổi phán quyết, và CI95 cụm cảnh của toàn arm cũng
vượt ngưỡng ở cận trên, cùng một kiểu giới hạn như tập chunk chung.

### Đọc kết quả — trung thực, không làm mềm

Hiệu ứng của caption oracle là **THẬT**: cả PronF1 neo (2 cách đo), chrF, và
COMET-DA đều tăng cùng chiều, CI95 đều không trùm 0 ở tập chung lẫn toàn arm
(kể cả khi tính lại theo cụm cảnh) — đây không phải nhiễu thống kê. Nhưng
hiệu ứng **NHỎ Ở ĐIỂM ƯỚC LƯỢNG**: khi bơm thẳng ĐÁP ÁN xưng hô theo cách
trình bày túi từ phẳng hiện tại, PronF1 neo trên tập chung chỉ tăng +0,0117
— chưa tới 60% ngưỡng +0,02 đã đặt trước khi đo.

**Sửa lại cách đọc kết luận (Important 1, review tổng) — đây là thay đổi
quan trọng so với bản trước của tài liệu này:** bản trước viết "dữ liệu cho
thấy trần dưới +0,02". Câu đó SAI với dữ liệu đã có: CI95 tính đúng theo
cụm cảnh (đơn vị lấy mẫu phải là CẢNH, vì ngữ cảnh oracle gán theo cảnh, xem
Step 2b) cho cận trên **+0,0233** (tập chung) và **+0,0278** (toàn arm) —
cả hai đều VƯỢT ngưỡng +0,02. Nói cách khác, **dữ liệu KHÔNG loại trừ được**
khả năng trần thật nằm trên ngưỡng; khoảng tin cậy quá rộng để kết luận theo
hướng đó. Cách đọc đúng: **G1 KHÔNG ĐẠT dựa trên ĐIỂM ƯỚC LƯỢNG trượt cổng ở
mọi cách phân tích lại** (tập chung, toàn arm, theo cue lẫn theo cụm cảnh),
**cộng với việc đòn bẩy ASR đo được trên cùng phim lớn hơn 5-6 lần** (xem
đoạn dưới) — chứ KHÔNG phải vì đã chứng minh được trần nằm dưới ngưỡng bằng
khoảng tin cậy. Đây là một suy luận yếu hơn "đã đo được trần thấp", nhưng là
suy luận ĐÚNG với dữ liệu hiện có; ngưỡng +0,02 không phải chân lý tuyệt đối,
nhưng đã thống nhất trước khi đo (đúng tinh thần pre-registration để tránh
tự thuyết phục bản thân sau khi thấy số).

**Giới hạn quan trọng hơn cả CI rộng (Important 2 + 3, review tổng — xem mục
đầy đủ ngay trước "Cổng G1 — Phán quyết"):** +0,0117 là trần của MỘT cách
trình bày ngữ cảnh (túi từ phẳng) qua MỘT adapter gần như không dùng đến
loại thông tin này (chỉ hấp thụ +2,4 điểm phần trăm trên tổng 69,3%→71,7%,
đổi text ở 19,9% cue) — KHÔNG phải trần của kênh `<Scene Context>` nói
chung, và càng không phải trần của mọi cách mã hoá thông tin xưng hô có thể
có. Một người tiêu thụ hoàn hảo cùng túi từ đó đạt F1 0,6050 (dư địa
+0,2235) trong khi arm `oracle` chỉ lấy được ~7% dư địa đó — phần lớn khoảng
cách nằm ở "adapter không dùng thông tin", không phải "thông tin không còn
gì để cho". Đặc biệt, một VLM chuyên nhân vật sinh cặp xưng hô CÓ HƯỚNG (một
cách mã hoá khác hẳn túi từ phẳng) có cấu trúc để sửa loại lỗi "đặt sai chỗ"
chiếm 53,6% lỗi còn lại — trần của hướng đó **chưa hề được đo**. Quyết định
dừng nghiên cứu dưới đây áp dụng cho "tiếp tục đầu tư dựa trên bằng chứng đã
có", không phải "đã chứng minh hướng VLM chuyên nhân vật vô dụng".

Điểm quan trọng: +0,0117 là **TRẦN CỦA CÁCH ĐO NÀY** — mức tốt nhất mà pipeline
hiện tại (túi từ phẳng + adapter hiện tại) đạt được khi bơm thẳng đáp án
đúng 100%. Trong phạm vi cách đo này, mọi kỹ thuật lấy thông tin THẬT bằng
CÙNG cách trình bày (túi từ) không thể vượt qua trần này, và trong thực tế
chỉ đạt được MỘT PHẦN của +0,0117 đó. Điều này KHÔNG áp dụng cho một cách
trình bày khác về chất (xem giới hạn ở trên).

So sánh với đòn bẩy khác đã đo trên đúng phim này
(`third_party_test/danh_gia_lai_3rd_party.md`, so `gold_rough` dùng phụ đề
EN chuẩn với `rough` dùng ASR): thay ASR bằng phụ đề chuẩn cho chrF
**+3,38** [+2,72, +4,01] và COMET-DA **+0,0190** [+0,0122, +0,0259] — lớn
hơn điểm ước lượng của trần kênh Scene Context (cách đo này) khoảng **5,4
lần theo chrF** (3,38/0,62) và **5,9 lần theo COMET-DA** (0,019/0,0032).
Đòn bẩy ASR áp dụng ngay không cần nghiên cứu thêm, và chênh lệch 5-6 lần
này — chứ không phải việc "đã chứng minh trần dưới ngưỡng" — là căn cứ chính
cho quyết định ưu tiên ASR ở mục "Hệ quả" dưới đây.

### Hệ quả

**G1 KHÔNG ĐẠT ⇒ dừng cả hai hướng nghiên cứu đang cân nhắc — dựa trên điểm
ước lượng trượt cổng ở mọi cách phân tích lại, cộng đòn bẩy ASR lớn hơn
5-6 lần, KHÔNG dựa trên việc đã chứng minh trần nằm dưới ngưỡng bằng CI
(xem "Đọc kết quả" ở trên):**
- Đổi VLM sang mô tả chuyên nhân vật (character-focused captioning).
- Thêm mô tả âm thanh + huấn luyện DPO (OmniDPO).

Chuyển sang mục **B2 (ASR và phân đoạn — đòn bẩy lớn nhất)** của
`docs/superpowers/plans/2026-09-02-roadmap-cai-tien.md`.

Ghi lại con số trần để không ai phải đo lại: **trần của cách trình bày túi
từ phẳng qua adapter hiện tại (tập chunk chung, oracle − base_repro) =
+0,0117 PronF1 neo, CI95 theo cue [+0,0038, +0,0197], CI95 cụm cảnh
[+0,0021, +0,0233] (cận trên vượt ngưỡng +0,02 — CI không loại trừ được
ngưỡng); chrF +0,62 [+0,28, +0,94]; COMET-DA +0,0032 [+0,0011, +0,0054].**
Đây là trần của MỘT cách mã hoá + MỘT adapter, không phải trần của kênh
`<Scene Context>` nói chung: 53,6% lỗi đại từ còn lại của `base_repro` là
lỗi đặt sai chỗ mà túi từ cấp cảnh không cấu trúc để sửa, một người tiêu thụ
hoàn hảo cùng túi đó có dư địa +0,2235 (arm `oracle` chỉ lấy được ~7%), và
adapter hiện tại chỉ hấp thụ thêm 2,4 điểm phần trăm token đại từ khi có đáp
án (69,3%→71,7%, đổi text 19,9% cue). **Trần của một VLM chuyên nhân vật
sinh cặp xưng hô có hướng (ai gọi ai) — hướng đang bị dừng ở đây — chưa hề
được đo.** Mọi cải tiến kênh `<Scene Context>` sau này (nếu có ai muốn thử
lại, đặc biệt nếu đổi hẳn cách trình bày sang có cấu trúc/có hướng thay vì
túi từ phẳng) nên được đọc theo tỉ lệ phần trăm của trần này VÀ xét lại từ
đầu chứ không giả định trần cũ vẫn áp dụng, và nên nhớ ASR là đòn bẩy lớn
hơn ~5-6 lần trên cùng phim.

### Phép thử lật cổng — arm `oracle_top4` (chạy sau review tổng)

Review tổng toàn nhánh nêu **Important 2**: arm `oracle` bơm một "túi từ
phẳng" (trung vị 7 từ xưng hô mỗi cảnh, tối đa 23), nên thứ đo được có thể là
trần của *cách trình bày* chứ không phải trần của *thông tin*. Reviewer chỉ ra
phép thử rẻ nhất có thể lật cổng: chạy lại **chỉ arm oracle** với `--top 4`
để nén túi từ về vùng ngữ cảnh sắc nét (các cảnh có ≤4 từ trong arm `oracle`
cho delta +0,1037, cao hơn hẳn trung bình), trong khi vẫn giữ 72,2% token đại
từ của tham chiếu.

**Tiêu chí ĐẶT TRƯỚC khi có số** (ghi vào sổ trước lúc phóng arm, không được
chỉnh sau): `delta ≥ +0,02` **VÀ** cận dưới CI95 **theo cụm cảnh** > 0 thì lật
G1.

Cách chạy: `make_oracle_context.py --top 4` (84/114 cảnh có từ xưng hô, 114/114
caption đúng lược đồ 3 mục) → `run_arm_context.sh oracle_top4` (61,9 phút, giữ
nguyên ASR/NMT, `--llm_batch_size 1`) → guard cấp chunk (7/95 chunk = 376/1939
cue rơi fallback, cảnh `[16, 28, 51, 52, 55, 61, 80]`) → tập chunk chung với
`base_repro` (hợp fallback `[16,28,50,51,52,55,61,72,80]`, còn **86 cảnh /
1508 cue**; sau khi neo tham chiếu là 74 cảnh hiệu dụng / 1325 cue).

| Arm | PronF1 neo (tập chung) | Δ vs `base_repro` | CI95 theo cue | CI95 theo cụm cảnh |
|---|---|---|---|---|
| `base_repro` | 0,3945 | — | — | — |
| `oracle_top4` | 0,4070 | **+0,0125** | [+0,0047, +0,0205] | [+0,0030, +0,0253] |

Toàn arm (không lọc tập chung): `oracle_top4` delta **+0,0132**, CI95 cụm cảnh
[+0,0049, +0,0240] — so với `oracle` (top-0) +0,0159, [+0,0047, +0,0278].

**Phán quyết: KHÔNG LẬT ĐƯỢC CỔNG G1.** Điểm ước lượng +0,0125 vẫn dưới ngưỡng
+0,02 đã đặt trước (đạt ~63%), y hệt kết luận của arm `oracle` gốc.

Ý nghĩa cho Important 2: giả thuyết "định dạng túi từ phẳng mới là thứ giới
hạn" **đã được kiểm ở điểm rẻ nhất và bị bác** — nén túi từ 7 xuống 4 từ cho
kết quả gần như y hệt (+0,0125 so với +0,0117). Kết luận dừng vì thế **mạnh hơn
trước, không yếu đi**.

Điều này **không** mở rộng thành "mọi cách mã hoá đều vô ích": top-4 vẫn là túi
từ phẳng, chỉ ngắn hơn. Giới hạn đã ghi ở mục "Đọc kết quả" vẫn nguyên giá trị
— **trần của một cách mã hoá CÓ HƯỚNG (ai gọi ai) chưa hề được đo.** Cái bị bác
ở đây chỉ là "độ dài/độ nhiễu của túi từ là nguyên nhân chính".
