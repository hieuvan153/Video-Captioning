# A/B V1 — Speaker Attribution (2026-08-16)

3 arm trên 5 phim eval, cùng một phiên bản pipeline (đã có guardrail chống suy
biến), paired bootstrap per-line 2000 lần. Số gốc: `docs/eval/speaker_ab_v1.json`,
SRT: `demo/output/eval_ab/<movie>/v1_{baseline,registry,speaker}.srt`.

Tái lập: `demo/EVAL/run_arms.py` rồi `demo/EVAL/compare_arms.py` (script bootstrap
của V0 đã mất trong scratchpad; lần này nằm trong repo).

## Kết quả

| phim | baseline | registry (V0) | speaker (V1) | Δ V1 vs baseline | Δ V1 vs V0 |
|---|---|---|---|---|---|
| movie_008 | 0,4744 | 0,4370 | 0,4934 | +0,0191 (p=0,102) | **+0,0564** |
| movie_009 | 0,6016 | 0,6016 | 0,5875 | −0,0141 (p=0,324) | −0,0141 |
| movie_015 | 0,4892 | 0,4786 | 0,5236 | **+0,0344 (p=0,000)** | +0,0450 |
| movie_045 | 0,3818 | 0,3142 | 0,3634 | −0,0184 (p=0,384) | **+0,0492** |
| movie_046 | 0,4221 | 0,4575 | 0,4878 | **+0,0657 (p=0,001)** | +0,0303 |
| **gộp** | **0,4789** | **0,4620** | **0,4954** | **+0,0164 (p=0,023)** | +0,0334 |

Trung bình delta theo phim (đại lượng của gate): registry **−0,0160**, speaker
**+0,0173**.

## Gate: KHÔNG đạt

Gate V1 là baseline + 0,03. Đo được +0,0173 (trung bình theo phim) / +0,0164
(gộp). Thiếu khoảng một nửa.

Điều **đạt được**: V0 gây hại có ý nghĩa thống kê (−0,0170, p=0,047); V1 đảo
thành có lợi có ý nghĩa thống kê (+0,0164, p=0,023), và tốt hơn V0 trên 4/5 phim.
Hồi quy của V0 đã được chữa, nhưng chưa đủ để mở V2.

## Điều quan trọng hơn con số: Bước 4 gần như không chạy

Log `run_arms.log` cho thấy registry theo scene chỉ render được:

| phim | scene có registry | lý do |
|---|---|---|
| movie_008 | 0/32 | tên trên cạnh thân tộc {Bo, Glenn, Mom} ∩ tên speaker gán được = {Glenn} — cạnh cần **cả hai** đầu |
| movie_009 | 0/32 | giao đủ {Cheyenne, Mom} nhưng gate cấp phim tắt (chỉ 1 cặp thân tộc) |
| movie_015 | 0/35 | giao {Big Daddy, Mommy} nhưng hai người này thuộc **hai cặp khác nhau** |
| movie_045 | 0/32 | giao {Dad, Meemaw, Shelly} đủ cặp, nhưng họ không bao giờ thoại **kề nhau** trong cùng scene |
| movie_046 | 6/26 | giao 6 tên → cặp hoàn chỉnh + kề nhau |

Nghĩa là trên 4/5 phim, arm "speaker" thực chất là **baseline + tag speaker,
registry tắt hẳn**. Tách ra:

- 4 phim chỉ có tag (008, 009, 015, 045): trung bình delta **+0,0053** — nhiễu.
- 1 phim có tag *và* registry theo scene (046): **+0,0657**, lớn nhất bộ.

Cảnh báo: 046 cũng là phim duy nhất V0 giúp được (+0,0354), nên nó có thể chỉ là
phim dễ. Nhưng hướng thì nhất quán với error analysis V0: giá trị nằm ở chỗ
**thu hẹp registry về đúng người đang nói**, không phải ở bản thân cái tag.

## Nút thắt cho V2

Registry theo scene chỉ kích hoạt khi **hội đủ ba điều kiện cùng lúc**:
cluster→tên đúng, tên trùng với đầu cạnh thân tộc trong registry, và hai người đó
thoại kề nhau trong cùng scene. Ba bước đều hao hụt nên tích lại gần như bằng 0.

Nguồn hao hụt đo được:

1. **Đặt tên cluster sai** — nút thắt lớn nhất. movie_045: SPK_106 (60 dòng, thoại
   về QCD) được gán "Missy"; SPK_036 (45 dòng, "mashed up avocados") gán
   "Dr. Sturgis". Đây cũng là phim duy nhất còn âm sau khi có tag.
2. **Registry và speaker gọi tên khác nhau** — registry sinh "Sheldon"/"Shelly"/
   "Dad"/"Boy" như các nhân vật riêng biệt; cluster_map lại chọn "Missy",
   "Dr. Sturgis", "Adult Sheldon". Cần hợp nhất alias giữa hai bước.
3. **Độ phủ speaker 70,6–83,7%** (sau khi loại sentinel `UNKNOWN` của CAM++), nên
   ~1/5 số dòng không có tag để bắt cặp.

## Audit sau A/B (2026-08-16, cùng ngày): sàn nhiễu và lỗi đã sửa

### Tính tất định — kết luận đã SỬA sau lần chạy v1b (cùng ngày 2026-08-16)

Ban đầu tôi so `v1_baseline.srt` với `refined_baseline.srt` (đợt 2026-08-13)
thấy 14–39 dòng/phim khác nhau và kết luận "sàn nhiễu run-to-run ±0,015".
**Kết luận đó sai về cơ chế.** Bằng chứng từ lần chạy lại v1b: cả 5 phim,
baseline và registry trùng `v1_` **từng dòng một (0 dòng khác)**; arm speaker
chỉ khác ở đúng 2 phim mà `speakers.json` đổi nội dung do fix code (009: 21
dòng, 015: 29 dòng). Tức pipeline **tất định hoàn toàn** khi cùng config +
cùng môi trường.

Chênh lệch với đợt 2026-08-13 do đó đến từ **khác biệt config/môi trường giữa
hai đợt** (khả năng cao nhất: llm_batch_size khác → thành phần batch và padding
đổi → số học fp16 đổi), không phải nhiễu ngẫu nhiên mỗi lần chạy. Quy tắc thực
hành giữ nguyên nhưng vì lý do khác: **chỉ so sánh arm trong cùng một đợt chạy
cùng config**; giữa hai đợt khác config thì khác biệt phản ánh config, không
phản ánh arm.

### Kiểm sạch (không có lỗi)

- Tag `[SPEAKER: X]` không rò rỉ sang output tiếng Việt: 0/1544 dòng.
- Prompt dài nhất ~472 token — xa trần 4096 và dưới cả ngưỡng suy giảm ~1200.
- Fallback thấp và đều giữa các arm (baseline 6, registry 2, speaker 3 /1544);
  guard suy biến không kích hoạt lần nào trong 15 lượt chạy.
- `max_seq_length` khác nhau giữa arm (2048/4096) đo được là vô hại
  (movie_009 trùng từng dòng) — vẫn đã đồng nhất về 4096 trong `run_arms.py`.

### Lỗi tìm thấy và đã sửa (115 test pass)

1. **`mention_exclusions` đếm nhãn kịch bản như một lần "nhắc tên"** — dòng
   "ADULT SHELDON: I was thrilled" vừa là bằng chứng người nói LÀ Adult Sheldon
   (anchor) vừa bị đếm là "nhắc đến Adult Sheldon" (exclusion): hai loại bằng
   chứng tự mâu thuẫn trên cùng một dòng. Sửa: xóa nhãn khỏi text trước khi đếm.
2. **`apply_evidence` để exclusion chạm vào cluster đã có anchor** — tên đúng bị
   "bỏ" rồi được anchor lặp lại ở cuối: kết quả tình cờ đúng nhưng bộ đếm sai.
   Sửa: thứ tự ưu tiên tường minh anchor > exclusion > LLM.
3. **`parse_registry` cho qua tên thuần số / 1 ký tự** ("1217" — mã cửa hàng
   trong Superstore, "G") — lọt vào tập đóng của cluster_map và substring-match
   mọi nơi trong `filter_registry_by_source`. Sửa tại parse; registry đã lưu
   được lọc lại khi load, không cần rebuild.
4. `character_names_from_registry` dedup phân biệt hoa thường → dedup casefold.
5. Chú thích `AlignStats.n_low_score` ghi "dòng" nhưng đếm chunk.

### A/B v1b — chạy lại toàn bộ sau fix (số chính thức, thay bảng v1 ở trên)

Số gốc: `docs/eval/speaker_ab_v1b.json`; SRT `v1b_*.srt`; speakers cũ giữ tại
`speakers.v1.json`.

| phim | arm | BLEU | chrF | P | R | F1 | ΔF1 vs baseline |
|---|---|---|---|---|---|---|---|
| movie_008 | baseline | 38,36 | 52,66 | 0,4091 | 0,5645 | 0,4744 | |
| | registry | 37,40 | 52,23 | 0,3772 | 0,5192 | 0,4370 | −0,0374 [−0,073;−0,002] p=0,037 |
| | speaker | 38,43 | 53,18 | 0,4246 | 0,5889 | 0,4934 | +0,0191 [−0,004;+0,043] p=0,102 |
| movie_009 | baseline | 46,21 | 60,13 | 0,5170 | 0,7192 | 0,6016 | |
| | registry | 46,21 | 60,13 | 0,5170 | 0,7192 | 0,6016 | 0 (gate tắt, SRT trùng từng dòng) |
| | speaker | 46,40 | 60,13 | 0,5011 | 0,7098 | 0,5875 | −0,0141 [−0,043;+0,014] p=0,316 |
| movie_015 | baseline | 38,51 | 53,99 | 0,4022 | 0,6243 | 0,4892 | |
| | registry | 37,33 | 52,76 | 0,3988 | 0,5983 | 0,4786 | −0,0106 [−0,040;+0,019] p=0,482 |
| | speaker | 38,63 | 54,00 | 0,4393 | 0,6590 | 0,5272 | **+0,0379 [+0,016;+0,060] p=0,000** |
| movie_045 | baseline | 35,92 | 51,03 | 0,3385 | 0,4377 | 0,3818 | |
| | registry | 34,83 | 50,27 | 0,2786 | 0,3603 | 0,3142 | −0,0675 [−0,128;−0,008] p=0,025 |
| | speaker | 35,72 | 50,32 | 0,3237 | 0,4141 | 0,3634 | −0,0184 [−0,058;+0,021] p=0,384 |
| movie_046 | baseline | 39,00 | 54,75 | 0,3675 | 0,4959 | 0,4221 | |
| | registry | 41,35 | 56,81 | 0,3988 | 0,5366 | 0,4575 | +0,0354 [−0,005;+0,076] p=0,082 |
| | speaker | 41,04 | 57,03 | 0,4268 | 0,5691 | 0,4878 | **+0,0657 [+0,024;+0,109] p=0,001** |
| **gộp 1544 dòng** | baseline | 40,17 | 54,68 | 0,4105 | 0,5747 | 0,4789 | |
| | registry | 39,81 | 54,48 | 0,3976 | 0,5512 | 0,4620 | −0,0170 [−0,034;−0,000] p=0,047 |
| | speaker | 40,50 | 55,02 | 0,4267 | 0,5928 | **0,4962** | **+0,0173 [+0,003;+0,031] p=0,020** |

Trung bình delta theo phim: registry −0,0160, speaker **+0,0180**. Gate +0,03:
vẫn KHÔNG đạt. Kết luận v1 giữ nguyên: V0 hại significant, V1 lợi significant,
cơ chế per-scene registry chỉ kích hoạt ở movie_046 (6/26 scene, 0 ở 4 phim
còn lại) — nút thắt V2 không đổi.

Tác động đo được của các fix lên mapping (`speakers.v1.json` → `speakers.json`):
movie_009 bỏ 6 cluster tên rác/đoán ẩu ('1217', 'Guys', 'Employee'…), 21 dòng
đổi; movie_015 SPK_007 'Big Daddy'→'Chief', 29 dòng đổi, F1 speaker nhích
+0,0036; movie_008/045/046 mapping không đổi, SRT trùng từng dòng.

## Ghi chú kỹ thuật phát hiện trong lúc chạy

- `UNKNOWN` là sentinel của CAM++ (`db_action: short_below_threshold`, 75–179
  chunk/phim), **không phải** một cluster. Coi nó là cluster thì 20–35% số dòng bị
  gom lại và gán chung một tên (movie_045: 67 dòng thành "Ingram"). Đã lọc ở
  `SPEAKER/align.py`.
- Prompt map cluster vượt ngưỡng ~1200 token đã ghi ở `CHARACTER/build_registry.py`
  làm Gemma-3 bỏ qua cả bằng chứng hiển nhiên (cluster có sẵn nhãn `GEORGE JR.:`
  vẫn bị trả `unknown`). Đã áp lại kỷ luật token trong `cluster_map.plan_batches`.
- SRT của A/B V0 được sinh **trước** guardrail chống suy biến (movie_015 vẫn còn
  dòng lặp), nên cả 3 arm lần này đều chạy lại; file V0 giữ nguyên, output mới
  mang tiền tố `v1_`.
- BLEU/chrF gần như không đổi giữa các arm (BLEU 35,7–46,6), tức thay đổi nằm
  đúng ở xưng hô chứ không phải model dịch lại nội dung.
