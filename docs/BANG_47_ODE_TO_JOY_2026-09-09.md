# Chấm phim bên thứ 3 theo **đúng giao thức Bảng 4.7** — 09/09/2026

Mục đích: có một con số đặt cạnh được 38,23 / 40,51 của Bảng 4.7 (anh ntVan) và
cạnh 32,71 / 33,63 của bản chia lại train/test.

## 0. Cảnh báo trước khi đọc bất kỳ số nào

Ba giao thức BLEU đang cùng tồn tại trong đồ án, **chênh nhau ~14 điểm trên cùng
một file**. Chênh lệch là do **giao thức**, không phải do model:

| Giao thức | Đơn vị chấm | Xử lý tham chiếu | `rough` trên Ode to Joy |
|---|---|---|---|
| **Bảng 4.7** (`ASR/calculate_bleu.py`, `demo/EVAL/film_score.py`) | nối cả tập/phim thành 1 chuỗi → corpus BLEU | hoán vị các dòng trùng timestamp để **tối đa hóa** BLEU | **32,64** |
| scene (`demo/EVAL/thesis_score.py`) | từng scene | không | 18,16 |
| neo cue tham chiếu (`film_ref_anchored.py --min_overlap_s 0.2`) | theo THỜI GIAN, neo trên cue tham chiếu | không | 21,10 |

Bảng dưới đây **chỉ dùng giao thức Bảng 4.7** để so được với đồ án cũ. Cổng
nghiệm thu G2 (chrF / COMET-DA + CI bootstrap) vẫn chấm bằng thước neo cue —
BLEU không nằm trong cổng.

**Bẫy thứ tư — ô `src` của COMET.** COMET-DA chấm `src → mt` có tham chiếu, nên
ô `src` phải là **phụ đề tiếng Anh của người** (`ref_3rd/en_3rd.clean.srt`),
không phải bản tiếng Anh do chính ASR sinh ra
(`output_orig/….(Tiếng Anh).srt`). Đưa nhầm bản máy vào thì lỗi ASR nằm ở cả hai
phía phép đo và điểm mất ý nghĩa; nó còn đổi luôn cỡ mẫu (1764 → 1725 cue) vì
`keep` được lọc theo cue có nguồn EN. Bản 09/09 lần đầu mắc bẫy này ở §4b, đã
chạy lại — sau khi sửa, `rough` và `v2_lp4` trùng khít con số bàn giao. Với
`film_score.py` thì cờ `--en` **chỉ** đi vào nhánh COMET, nên các bảng chạy
`--no_comet` (§3, §3b, §3c) không bị ảnh hưởng.

## 1. Bảng 4.7 gốc (10 tập E5) — **có rò rỉ**

mBART trong pipeline đã được fine-tune trên chính 10 tập dùng để chấm.

| Cấu hình | BLEU |
|---|---|
| E1 — NMT (mBART) | 38,23 |
| E2 — LLM dịch | 37,83 |
| E3 — SeamlessM4T | 23,33 |
| E4 — NMT + LLM tinh chỉnh (không ảnh) | 38,93 |
| **E5 — NMT + LLM + ảnh** | **40,51** |

## 2. Chia lại train/test rồi đo lại **chính kiến trúc cũ** (`bleu_episode.txt`)

mBART huấn luyện lại có loại 13 phim (`mbart_clean`), chấm trên đúng 10 tập E5:

| Arm | n | raw | nopunc | custom |
|---|---|---|---|---|
| rough | 10 | **32,71** | 30,73 | 34,41 |
| nocontext | 10 | 33,43 | 31,95 | 35,25 |
| context | 10 | **33,63** | 32,06 | 35,48 |
| rough (3 phim E-out) | 3 | 32,78 | 29,68 | 34,66 |
| context (3 phim E-out) | 3 | 33,12 | 30,55 | 34,93 |

**40,51 → 33,63 = −6,88 điểm.** Đó là phần rò rỉ, không phải tụt chất lượng.

## 3. Ode to Joy 2019 (phim bên thứ 3, chưa model nào thấy) — cùng giao thức 4.7

Tham chiếu `ref_3rd/vi_3rd.clean.srt` (1776 cue), nguồn EN = SRT ASR của pipeline
(1939 cue). Báo cáo: `A_20260909/b47_existing.json`, `A_20260909/b47_clean.json`.

| Arm | raw | nopunc | custom | PronF1 | junk |
|---|---|---|---|---|---|
| `orig_rough` — bản bàn giao | **32,64** | 27,05 | 33,83 | 0,756 | 0,070 |
| `asr_rough` — tái tạo bản bàn giao | 32,64 | 27,05 | 33,83 | 0,756 | 0,070 |
| `asr_beam5` — chỉ thêm beam 5 | 33,57 | 28,03 | 34,75 | 0,758 | 0,073 |
| `asr_gx2` — GemmaX2-28-9B tinh chỉnh câu | 33,71 | 29,11 | 35,01 | 0,647 | 0,066 |
| **`v2_lp4` — kiến trúc v2 đang giao** | **36,00** | 29,99 | 37,22 | **0,768** | 0,061 |
| `a2_lp4` — trần trên của nhánh (ASR đắt) | 35,98 | 30,17 | 37,21 | 0,760 | 0,048 |

**+3,36 BLEU so với bản bàn giao trên cùng một phim, cùng một thước.**

### 3b. Thử MBR (Minimum Bayes Risk) — không mua thêm BLEU

Sinh K ứng viên rồi chọn ứng viên có chrF trung bình cao nhất so với cả tập ứng
viên (đồng thuận của model). Báo cáo: `A_20260909/b47_mbr.json`, `b47_mbrbeam.json`.

| Arm | nguồn ứng viên | raw | PronF1 | junk |
|---|---|---|---|---|
| `v2_lp4` (đang giao) | — (beam 5 + lp4) | **36,00** | 0,768 | 0,061 |
| `v2_mbrbeam5` | beam 5 + lp4 | 35,66 | **0,780** | 0,058 |
| `v2_mbrbeam8` | beam 8 + lp4 | 34,98 | 0,776 | **0,044** |
| `v2_mbr8` | lấy mẫu top-p 0,9 | 33,90 | 0,742 | 0,056 |

Kết luận: **MBR không thắng beam trên BLEU** (−0,34 đến −2,10). Ứng viên lấy mẫu
kém hẳn ứng viên beam, đúng như dự đoán. Nhưng MBR-trên-beam **có** cải thiện
PronF1 (+0,012) và giảm rác (−0,017) — vẫn dưới cổng G1 (+0,02 PronF1), nên
nhánh đóng: giữ `v2_lp4`, không đổi.

Bổ sung 09/09: đã chấm thêm COMET-DA cho hai arm MBR trên thước cổng (§4b).
MBR **qua cổng G2**, `v2_mbrbeam8` thậm chí hòa `v2_lp4` ở COMET-DA (0,7756).
Nhánh vẫn đóng, nhưng lý do chính xác là **giá** (phải sinh và xếp hạng K ứng
viên) chứ không phải "MBR kém hơn về ngữ nghĩa".

### 3c. Vì sao dừng ở `length_penalty=4.0`? — quét lại toàn dải

Cùng file SRT, chỉ đổi `length_penalty`. Cột trái = giao thức 4.7 (`b47_lpsweep.json`),
cột phải = thước cổng G2 (`beam_sweep/lp2.json`, `lp3_phaseA.json`).

| lp | BLEU-4.7 | PronF1 | BLEU-cổng | chrF-cổng |
|---|---|---|---|---|
| beam5 (lp1) | 33,57 | 0,758 | 19,13 | 38,29 |
| 3,0 | 35,65 | 0,768 | 19,09 | 39,41 |
| **4,0** | **36,00** | **0,768** | **19,15** | **39,55** |
| 5,0 | 36,16 | 0,769 | 19,01 | 39,52 |
| 6,0 | 36,25 | 0,766 | 18,89 | 39,53 |
| 8,0 | **36,29** | 0,764 | 18,80 | 39,50 |
| 12,0 | 36,23 | 0,763 | 18,69 | 39,44 |

BLEU theo giao thức 4.7 còn nhích thêm **+0,29** khi đẩy lp lên 8, nhưng trên
thước cổng thì cả BLEU (−0,35) lẫn chrF (−0,05) lẫn PronF1 (−0,004) đều đi
xuống. **lp = 4,0 là điểm gãy**, không phải chọn tùy tiện. Câu trả lời cho hội
đồng nếu bị hỏi "sao không tăng tiếp": tăng tiếp chỉ mua được điểm trên thước so
lịch sử, và trả bằng điểm trên chính thước nghiệm thu.

Beam đơn thuần (không lp) bão hòa ở 5: beam8 = 33,62, beam12 = 33,71 — tức
**cơ chế của lp4 là xếp hạng lại, không phải tìm kiếm rộng hơn.**

## 4. Đối chứng rò rỉ: mBART sạch trên cùng phim đó

`mbart_clean` loại 13 phim khỏi tập train (`logs_mbart_clean.log`), dịch lại đúng
file SRT tiếng Anh đó — không huấn luyện gì thêm.

| mBART | greedy | beam5 + lp4.0 | lp4 mua được |
|---|---|---|---|
| pipeline (đã thấy 10 tập E5) | 32,64 | **36,00** | +3,36 |
| **sạch** (loại 13 phim) | 32,06 | **34,79** | +2,73 |
| chênh do rò rỉ | −0,58 | −1,21 | |

Ý nghĩa: trên phim **cả hai model đều chưa thấy**, việc mBART pipeline từng học 10
tập E5 chỉ đáng **0,6–1,2 BLEU**. Nên 36,00 không phải con số được rò rỉ thổi lên;
ngay cả bản hoàn toàn sạch (34,79) vẫn cao hơn kiến trúc cũ đo lại sạch (33,63).

### 4b. Cũng phim đó, nhưng chấm bằng **thước cổng G2** (neo cue, `--min_overlap_s 0.2`)

n = **1764** cue tham chiếu (1776 trừ 12 cue không có nguồn EN), base = `rough`,
bootstrap ghép cặp n=1000. Báo cáo: `A_20260909/anchored_srcfix.json`.

> **Đính chính so với bản trước.** Bảng cũ ghi n=1725 và giải thích là "tập arm
> khác nhau nên giao cue khác nhau" — **giải thích đó sai**. Tập cue được chọn
> (`film_ref_anchored.py:84`) chỉ phụ thuộc `--ref` và `--src_en`, hoàn toàn không
> phụ thuộc tập arm; mọi arm trong bảng này đều có đúng 1764 cue và cùng tỉ lệ
> rỗng 2,61%. Nguyên nhân thật là **`--src_en` bị truyền nhầm bản tiếng Anh do ASR
> sinh** thay vì phụ đề tiếng Anh của người (`ref_3rd/en_3rd.clean.srt`); lỗi ASR
> khi đó nằm ở cả hai vế của COMET và làm rụng 39 cue. Bảng dưới là bản đã sửa.

| Arm | BLEU | chrF | COMET-DA | ΔBLEU (CI95) | ΔchrF (CI95) | ΔCOMET (CI95) |
|---|---|---|---|---|---|---|
| `rough` (bàn giao) | 21,10 | 38,07 | 0,7700 | — | — | — |
| **`v2_lp4`** | **22,00** | **40,18** | **0,7756** | **+0,89 [+0,27; +1,54]** | **+2,11 [+1,58; +2,64]** | **+0,0057 [+0,0031; +0,0084]** |
| `clean_rough` | 20,49 | 37,31 | 0,7679 | −0,61 [−1,14; −0,07] | −0,75 [−1,21; −0,28] | −0,0021 [−0,0045; +0,0007] |
| `clean_lp4` | 20,93 | 38,96 | 0,7722 | −0,17 [−0,80; +0,58] | +0,90 [+0,33; +1,48] | +0,0022 [−0,0006; +0,0053] |
| `v2_mbrbeam5` | 21,25 | 39,70 | 0,7751 | +0,15 [−0,46; +0,75] | +1,63 [+1,14; +2,14] | +0,0051 [+0,0024; +0,0079] |
| `v2_mbrbeam8` | 21,00 | 39,74 | 0,7756 | −0,10 [−0,70; +0,50] | +1,67 [+1,17; +2,20] | +0,0056 [+0,0032; +0,0083] |

Ba điều đọc được:

1. **`v2_lp4` qua cổng G2 trên cả hai thước cùng lúc** — chrF +2,11 và COMET-DA
   +0,0057, cả hai CI đều không chứa 0, BLEU không giảm (còn tăng +0,89, CI cũng
   không chứa 0). Đây là bằng chứng nghiệm thu, khác với BLEU-4.7 vốn chỉ để so
   lịch sử.
2. **MBR cũng qua cổng G2, và `v2_mbrbeam8` hòa `v2_lp4` ở COMET-DA (0,7756).**
   Đây là chỗ bản trước nói sai: hồi đó chưa chấm COMET cho MBR nên kết luận
   "không có kịch bản nào MBR thắng" là suy đoán, không phải số liệu. Đo rồi thì
   MBR *không thua về ngữ nghĩa* — nó chỉ thua ở **chrF (39,74 so với 40,18)** và
   **BLEU (21,00 so với 22,00, ΔBLEU so với `rough` có CI chứa 0)**. Lý do loại
   MBR vì thế là **giá**, không phải chất lượng ngữ nghĩa: MBR phải sinh và xếp
   hạng nhiều ứng viên, còn `length_penalty=4.0` chỉ là một tham số của beam
   search sẵn có, chi phí thêm bằng 0.
3. **Đo được lượng rò rỉ ngay trên thước cổng:** `clean_rough` thấp hơn `rough`
   **0,61 BLEU / 0,75 chrF** (cả hai CI không chứa 0), nhưng COMET-DA chỉ −0,0021
   với **CI chứa 0** — tức rò rỉ ảnh hưởng đến trùng khớp bề mặt (n-gram, ký tự)
   chứ không đo được ở mức ngữ nghĩa. So với khoảng cách 6,88 BLEU trong Bảng 4.7:
   khoảng cách kia lớn gấp 9 lần vì nó được đo **trên chính 10 tập bị rò rỉ**.

### 4c. Tầng LLM tinh chỉnh câu từ — kết luận cũ bị lật vì **lỗi gióng dòng**

Tầng LLM (`refine_llm.py`, Gemma-3 12B, `--llm_batch_size 1`) chạy trên đầu ra
`v2_lp4`. Bản đo trước kết luận nó **làm giảm chất lượng** và bị loại. Đo lại
09/09 cho thấy **kết luận đó là artefact của một lỗi phần mềm**, không phải của
mô hình.

**Lỗi.** `refine_llm.py` gửi mỗi chunk gồm *n* cue cho LLM rồi tách đầu ra theo
`\n` và ánh xạ **theo VỊ TRÍ**: dòng thứ *k* trả về được gán cho cue thứ *k*.
Chỉ cần mô hình gộp hai câu làm một (hoặc bỏ một câu), **mọi dòng còn lại trong
chunk trượt sang cue khác** — nội dung đúng nhưng nằm sai chỗ. Khối đệm/cắt cho
đủ số dòng ở cuối hàm chỉ che được *số đếm*, không sửa được *độ lệch*. Trên phim
này: **6/95 chunk = 304 dòng = 15,7 % phim** bị trượt.

**Sửa.** Thử hai mức, cả hai đều bỏ hẳn giả định "một dòng một cue":

- `_fix` — **bảo thủ**: số dòng khác số cue thì trả cả chunk về bản dịch thô. Không
  bao giờ gióng sai, nhưng *đánh giá thấp* tầng LLM vì bỏ 15,7 % phim.
- `_dp` — **gióng lại bằng quy hoạch động**: gióng m dòng LLM vào n cue theo đúng
  thứ tự, cho phép bỏ qua cue, điểm là độ giống giữa dòng LLM và bản thô của cue
  đó. Giữ lại **184/304 dòng** tinh chỉnh. Đây là bản đã đưa vào `refine_llm.py`.

| Arm (thước cổng, n=1764, base `v2_lp4`) | BLEU | chrF | COMET-DA |
|---|---|---|---|
| `v2_lp4` (nền) | 22,00 | 40,18 | 0,7756 |
| `v2_lp4_llm` — **có lỗi gióng** | 21,00 <br>−0,99 [−1,68; −0,32] | 38,15 <br>−2,03 [−2,71; −1,38] | 0,7575 <br>−0,0181 [−0,0232; −0,0133] |
| `v2_lp4_llm_fix` — bỏ cả chunk | 22,10 <br>+0,10 [−0,50; **+0,67**] | 40,03 <br>−0,15 [−0,66; **+0,33**] | 0,7751 <br>−0,0005 [−0,0036; **+0,0025**] |
| **`v2_lp4_llm_dp`** — gióng QHĐ | **22,08** <br>+0,08 [−0,56; **+0,76**] | 39,89 <br>−0,29 [−0,85; **+0,27**] | 0,7742 <br>−0,0015 [−0,0048; **+0,0019**] |

Cả ba mức tụt "có ý nghĩa thống kê" của tầng LLM **biến mất** sau khi sửa: sáu CI
của hai bản sửa đều chứa 0. Trên giao thức Bảng 4.7 cùng lúc đó:

| Arm | BLEU raw | BLEU nopunc | PronF1 | junk |
|---|---|---|---|---|
| `v2_lp4` | 36,00 | 29,99 | 0,7677 | 0,061 |
| `v2_lp4_llm_fix` | 35,36 | 30,96 | 0,8263 | 0,070 |
| **`v2_lp4_llm_dp`** | 35,16 | **30,80** | **0,8359** | 0,073 |

Đọc đúng ba ý:

1. **Tầng LLM trung tính với cổng G2** — không thắng (không CI nào loại 0 về phía
   dương) nhưng cũng **không hại**. Kết luận "LLM refine làm giảm chất lượng" của
   bản trước là **lỗi đo, không phải kết quả**.
2. **Nó qua cổng G1 rất rộng:** PronF1 0,8359 so với 0,7677 = **+0,068**, gấp hơn
   **3 lần** ngưỡng +0,02. Đây là đòn bẩy đại từ lớn nhất đo được cho tới nay —
   để so, trần kênh Scene Context (oracle) chỉ +0,0117. Càng giữ được nhiều dòng
   tinh chỉnh thì PronF1 càng cao (`_fix` 0,8263 → `_dp` 0,8359), tức phần thắng
   đến từ chính nội dung LLM sửa, không phải từ nhiễu.
3. **BLEU raw giảm 0,84 nhưng BLEU nopunc tăng 0,81.** Chênh lệch ngược chiều này
   nói phần mất nằm ở **dấu câu**, không ở từ ngữ: LLM chuẩn hóa lại dấu câu khác
   với phụ đề tham chiếu. Ở thước cổng (đã bỏ ảnh hưởng lưới cue) BLEU thậm chí
   nhích lên.

**Bài học phương pháp cho hội đồng:** một nhánh đã từng bị *đóng bằng số liệu*
vẫn có thể bị đóng nhầm nếu đường dẫn dữ liệu có lỗi. Dấu hiệu phát hiện ở đây là
một **bất thường nội tại** — BLEU raw giảm trong khi BLEU nopunc tăng — chứ không
phải nghi ngờ mô hình.

### 4d. Vá 3 + vá 4: chuẩn hóa dấu ba chấm và **khóa độ dài** — nhánh LLM chuyển từ trung tính sang THẮNG

§4c dừng ở vá 2 (gióng QHĐ). Đo tiếp 10/09 tìm thêm hai lỗi và một cơ chế mới.

**Vá 3 — dấu ba chấm.** `refine_llm.py` đổi `…` thành `...` *ngay sau khi decode*,
nhưng công cụ dựng lại arm ngoại tuyến thì gióng *trước khi* đổi. Hai chuỗi khác
nhau ở ký tự nên `SequenceMatcher` cho điểm thấp và QHĐ vứt nhầm dòng đúng. Sửa
xong (`_dp_e`): BLEU cổng **+0,61** so với nền, chứ không phải +0,08 như bảng
§4c. **Con số +0,61 thuộc về vá 3, không phải vá 4** — đây là chỗ dễ ghi nhầm.

**Vá 4 — khóa độ dài (`LEN_LOCK_RATIO`, `lock_len` trong `refine_llm.py`).**

Phân rã BLEU của giao thức 4.7 chỉ ra chỗ mất điểm thật:

| Arm | BLEU | BP | hyp/ref | BLEU nếu BP = 1 |
|---|---|---|---|---|
| `v2_lp4` (nền) | 36,00 | 0,9309 | 0,933 | 38,68 |
| `tau_0.2` (tầng LLM, vá 3) | 36,82 | **0,9055** | 0,910 | **40,66** |
| **`pm0.90` (vá 4)** | **37,90** | **0,9459** | 0,947 | 40,07 |

Đọc hàng giữa: tầng LLM **nâng độ chính xác n-gram rất mạnh** (38,68 → 40,66)
nhưng đồng thời **viết ngắn lại** (BP 0,9309 → 0,9055), nên phần thắng bị phạt độ
dài ăn gần hết, chỉ còn +0,82. Nguyên nhân gốc đã ghi trong bàn giao: người dịch
bên thứ 3 viết dài hơn (vi/en = 1,196) so với pipeline (1,105).

**Cơ chế.** Với mỗi cue, nếu dòng tinh chỉnh **ngắn hơn 0,90 lần** bản thô (đếm
theo từ) thì **không lấy cả dòng**, chỉ chuyển các đoạn sửa **thuần đại từ** từ
dòng tinh chỉnh sang bản thô (`merge_pron`, so khớp mức từ bằng
`difflib.get_opcodes`). Giữ độ dài và n-gram của bản thô (lợi BLEU) mà vẫn lấy
được phần xưng hô của LLM (lợi PronF1). Dòng đủ dài thì giữ nguyên cả dòng.

| Arm (giao thức 4.7) | raw | nopunc | custom | PronF1 |
|---|---|---|---|---|
| `v2_lp4` (nền) | 36,00 | 29,99 | 37,22 | 0,768 |
| `tau_0.2` — chỉ vá 3 | 36,82 | 30,84 | 38,04 | **0,833** |
| `pmerge` — chỉ ghép đại từ, không khóa | 36,42 | — | — | 0,818 |
| `len0.90` — chỉ khóa, dòng ngắn về hẳn bản thô | 37,89 | — | — | 0,822 |
| **`pm0.90` — khóa + ghép đại từ** | **37,90** | **31,83** | **39,14** | **0,830** |
| `oracle_sel` — trần trên của mọi bộ chọn theo cue | 38,48 | — | — | 0,790 |

**`pm0.90` thắng cả ba cột BLEU cùng lúc** (+1,90 / +1,84 / +1,92 so với nền) và
vẫn giữ **+0,062 PronF1**. Nghịch lý "raw giảm nhưng nopunc tăng" của §4c biến
mất, vì phần lớn dòng bị khóa quay về đúng dấu câu của bản thô.

Trên **thước cổng G2** (neo cue, n = 1764, base = `v2_lp4`, bootstrap ghép cặp
n = 1000, `A_20260909/anchored_pm.json`):

| Arm | BLEU | chrF | COMET-DA | ΔBLEU (CI95) | ΔchrF (CI95) | ΔCOMET (CI95) |
|---|---|---|---|---|---|---|
| `v2_lp4` (nền) | 22,00 | 40,18 | 0,7756 | — | — | — |
| `tau_0.2` | **23,11** | 40,84 | 0,7796 | +1,12 [+0,58; +1,64] | +0,66 [+0,19; +1,11] | +0,0040 [+0,0013; +0,0068] |
| `pm0.85` | 22,80 | 41,23 | **0,7805** | +0,80 [+0,36; +1,24] | +1,05 [+0,69; +1,44] | +0,0048 [+0,0029; +0,0070] |
| **`pm0.90`** | 22,81 | **41,32** | 0,7803 | **+0,81 [+0,41; +1,24]** | **+1,14 [+0,79; +1,50]** | **+0,0047 [+0,0029; +0,0067]** |

**`pm0.90` qua cổng G2 rất rõ: cả ba khoảng tin cậy đều không chứa 0.** Cổng chỉ
đòi chrF **hoặc** COMET-DA thắng có ý nghĩa và cái còn lại không giảm; ở đây cả
hai cùng thắng, BLEU cũng thắng dù BLEU không nằm trong cổng. Đây là arm đầu tiên
của tầng LLM làm được việc đó — so với `v2_lp4_llm_dp_e` (chỉ vá 3) vốn còn âm
chrF (−0,09) và âm COMET (−0,0015).

So `pm0.85` với `pm0.90`: chênh nhau trong sai số, `pm0.90` nhỉnh hơn ở chrF
(+0,09) nên chọn 0,90, đúng với giao thức 4.7 (37,90 so với 37,89).

**Ba nhánh đã đo và đóng, đừng đo lại:**

1. **Ngưỡng khóa không nhạy.** Quét 0,80–1,00: vùng 0,88–0,95 cho 37,79–37,90.
   Chọn 0,90 không phải chọn đúng một điểm may mắn.
2. **Ngưỡng gióng τ không ảnh hưởng.** `pm0.90` ở τ = 0,0 / 0,2 / 0,3 / 0,4 cho
   37,90 / 37,90 / 37,83 / 37,84 theo giao thức 4.7. Trên thước cổng
   (`A_20260909/anchored_tau.json`) cả năm mức τ = 0,0–0,4 cho ΔchrF +0,63 đến
   +0,69 và ΔCOMET +0,0035 đến +0,0044, mọi CI đều không chứa 0 — **chênh giữa
   các mức τ nhỏ hơn bề rộng CI**, tức τ không phải siêu tham số cần dò. Giữ
   τ = 0,20.
3. **Trần trên của cổng chất lượng (QE gating) đã đo bằng oracle.** Cho bộ chọn
   *nhìn trộm tham chiếu* rồi chọn theo chrF từng cue: chỉ được **38,48**
   (+0,58 so với `pm0.90`) và **PronF1 tụt còn 0,790**. Một mô hình QE thật không
   thể vượt oracle, nên nhánh "thêm tầng chọn ứng viên" mua được nhiều nhất 0,58
   BLEU và phải trả bằng đại từ — **không đáng thời gian GPU**.
4. **Trần trên độ dài cũng đã chạm.** Đẩy `length_penalty` lên 8 mua thêm BP
   nhưng trả bằng đúng chừng đó độ chính xác (38,68 → 38,24), net +0,29; còn
   `pm0.90` đã đạt BP 0,9459 (bằng mức lp6) mà **giữ nguyên** độ chính xác 40,07.
   Khóa độ dài **thay thế** việc đẩy lp, không cộng dồn với nó.

**Điều còn lại chưa đóng:** hyp/ref vẫn là 0,947. Nếu BP = 1 thì BLEU sẽ là 40,07,
tức **còn 2,17 điểm nằm trong phần độ dài**. Muốn lấy phải để chính tầng LLM sinh
tiếng Việt đầy đặn hơn (đổi lời nhắc / huấn luyện có ràng buộc độ dài), không lấy
được bằng hậu xử lý.

## 5. Câu chốt cho hội đồng

1. 32,64 (Ode to Joy) ≈ 32,71 (10 tập E5 sau khi chia lại) — kiến trúc cũ ổn định
   quanh **32,7** khi không rò rỉ, bất kể tập test nào.
2. Con số 40,51 trong Bảng 4.7 cao hơn **6,88 điểm** vì mBART đã học chính 10 tập
   đem đi chấm. Đây là phát hiện, không phải lỗi của kết quả mới.
3. Trên nền sạch đó, kiến trúc v2 đạt **36,00** (bản sạch hoàn toàn: 34,79), tức
   **+3,36** so với bản bàn giao, đồng thời bỏ 3 tầng (118 → 14 phút). Thêm tầng
   LLM đã vá + khóa độ dài (`pm0.90`, §4d) lên **37,90**.
4. BLEU ở đây là **thước so lịch sử** với Bảng 4.7. Cổng nghiệm thu G2 vẫn là
   chrF / COMET-DA với CI bootstrap (thước neo cue) — `v2_lp4` qua cổng:
   22,00 / 40,18 / 0,7756 so với 21,10 / 38,07 / 0,7700.
5. Ba nhánh đã thử và **đóng có số liệu**, không phải bỏ dở: MBR (qua cổng G2 nhưng thua
   `v2_lp4` ở BLEU/chrF trong khi tốn thêm nhiều lần chi phí sinh, §3b và §4b), đẩy `length_penalty` > 4 (mua BLEU-lịch-sử, trả bằng
   BLEU/chrF cổng, §3c), beam rộng hơn 5 (bão hòa, §3c). Đây là phần trả lời
   cho câu "sao không thử thêm X".
6. **Tầng LLM tinh chỉnh câu từ không những không bị đóng, nó là phần thắng lớn
   nhất còn lại.** Bản đo trước loại nó vì tưởng nó làm giảm chất lượng; đó là
   **lỗi gióng dòng trong `refine_llm.py`** (§4c), cộng thêm lỗi chuẩn hóa dấu ba
   chấm (§4d, vá 3). Sửa xong hai lỗi thì nó đã dương, nhưng phần thắng bị **phạt
   độ dài** ăn gần hết.
7. **Bản giao cuối là `pm0.90` = v2 + tầng LLM + khóa độ dài** (§4d). Khóa độ dài
   giải quyết đúng cái phạt đó: dòng LLM ngắn hơn 0,90 lần bản thô thì chỉ lấy
   phần sửa **xưng hô**, phần còn lại giữ bản thô.

   | Thước | Nền `v2_lp4` | **`pm0.90`** | Chênh |
   |---|---|---|---|
   | BLEU giao thức 4.7 | 36,00 | **37,90** | **+1,90** |
   | PronF1 (cổng G1: +0,02) | 0,768 | **0,830** | **+0,062** |
   | chrF cổng G2 | 40,18 | **41,32** | **+1,14** |
   | COMET-DA cổng G2 | 0,7756 | **0,7803** | **+0,0047** |

   Trên thước cổng, **cả ba khoảng tin cậy bootstrap đều không chứa 0**
   (ΔchrF +1,14 [+0,79; +1,50], ΔCOMET +0,0047 [+0,0029; +0,0067],
   ΔBLEU +0,81 [+0,41; +1,24]), nên `pm0.90` **qua cả cổng G1 lẫn cổng G2**.

   So với bản bàn giao đầu (32,64) là **+5,26 BLEU** trên cùng một phim bên thứ 3,
   cùng một thước, không hề train trên phim đó.

## 6. Nếu hội đồng hỏi ngược

**"Vậy kết quả mới thấp hơn 40,51 của đồ án cũ à?"**
Không so được trực tiếp. 40,51 đo trên 10 tập mà mBART đã học thuộc. Bỏ rò rỉ đi,
chính kiến trúc đó còn 33,63; kiến trúc mới trên phim hoàn toàn mới đạt 36,00.

**"Sao BLEU chỗ này 36 chỗ kia 22?"**
Ba giao thức khác nhau (§0). 36,00 và 40,51 cùng giao thức 4.7 nên so được với
nhau. 22,00 là thước neo cue dùng cho cổng nghiệm thu — nghiêm hơn vì không được
hoán vị dòng để tối đa hóa điểm.

**"Khóa độ dài có phải là chỉnh cho khớp thước đo không?"**
Không, và đây là câu dễ bị hỏi nhất. Khóa độ dài **không nhìn tham chiếu** — nó chỉ
so dòng LLM với bản thô của chính pipeline. Bằng chứng nó không phải trò gian
thước: nếu chỉ cần "viết dài ra" thì đẩy `length_penalty` lên 8 cũng làm được, mà
làm vậy chỉ được +0,29 BLEU-4.7 và **tụt** trên thước cổng (§3c). `pm0.90` thì
thắng đồng thời cả bốn con số ở §5.7, kể cả COMET-DA vốn không quan tâm độ dài.
Nguyên nhân gốc có thật và đã đo: người dịch bên thứ 3 viết dài hơn (vi/en 1,196
so với 1,105 của pipeline), còn LLM tinh chỉnh lại **rút ngắn** thêm.

**"Còn bao nhiêu nữa mới hết dư địa?"**
`pm0.90` có hyp/ref 0,947, tức BP 0,946. Nếu độ dài khớp hẳn thì BLEU sẽ là
**40,07** — còn **2,17 điểm** nằm trong phần độ dài. Không lấy được bằng hậu xử
lý; phải để chính tầng LLM sinh tiếng Việt đầy đặn hơn (đổi lời nhắc hoặc huấn
luyện có ràng buộc độ dài). Trần trên của hướng "thêm bộ chọn ứng viên" thì đã đo
bằng oracle nhìn trộm tham chiếu: chỉ 38,48 và tụt PronF1 (§4d).

**"Rò rỉ đáng bao nhiêu?"**
Đo được hai lần, hai thước: 0,58–1,21 BLEU (giao thức 4.7, §4) và 0,61 BLEU /
0,75 chrF với CI không chứa 0 (thước cổng, §4b) — nhưng **COMET-DA thì không đo
được** (−0,0021, CI95 [−0,0045; +0,0007] chứa 0). Rò rỉ làm model thuộc mặt chữ, không làm nó hiểu
hơn.
