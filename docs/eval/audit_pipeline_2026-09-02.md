# Rà soát output & ground truth toàn pipeline (2026-09-02)

Phạm vi: 10 tập E5 (`demo/output/eval_e5`, arm `v1b_baseline` / `v1b_speaker`
sau prompt-fix 19/08), 19 phim `demo/output/eval_ab`, gold
`data/en-vi-speaker-with-time-pronouns`, phụ đề chính thức `data/Movie/sub`,
log train v5/v5b. Mọi số dưới đây tái lập bằng 4 script trong
`docs/eval/audit_2026-09-02/` (chạy từ thư mục `ntVan`, python `van_env`, không GPU).

## 0. Tóm tắt — xếp theo mức ảnh hưởng

| # | Module | Vấn đề | Số đo | Ảnh hưởng |
|---|---|---|---|---|
| 1 | ASR + phân đoạn | WER 7.9% (deletion chiếm 3.8%), 11% dòng là mảnh câu; NMT trên EN gold cao hơn NMT trên EN ASR **5.4 BLEU** cùng đơn vị | §1 | **Lớn nhất**, chưa ai đụng |
| 2 | Train LLM | v5 **và** v5b đều sụp loss về 0 → root cause ghi ở `train_v5_zero_loss.md` sai; thủ phạm là thiếu `attn_implementation="eager"` (probe 4: loss batch 8.93 → 3.97) | §4.5 | Chặn mọi retrain |
| 3 | LLM refine | 45% dòng bị sửa, 2/3 số đó sửa cả nội dung; sửa nội dung gần như hòa (748 tốt / 608 xấu); lợi ích thật nằm ở đại từ | §4.1–4.2 | Trung bình |
| 4 | LLM refine | Lệch dòng trong chunk (51 dòng/16 chunk) và sập cả khối kiểu "- Không." ×18 mà guard không bắt | §4.3 | Trung bình, sửa rẻ |
| 5 | LLM train data | Train trên rough SẠCH (EN gold + NMT của EN gold), chạy trên rough từ ASR | §4.4 | Trung bình |
| 6 | Gold đại từ | Nhãn sinh bằng Gemma-3 + luật, có lỗi loại từ ("con mồi" → "con"); Bảng 4.8 đếm trùng đại từ lồng nhau | §6.3 | Metric F1 đại từ chỉ tin được tương đối |
| 7 | Gold tuổi/giới | Là dự đoán từ face-track, 36% record null | §6.4 | Proxy speaker dùng nhãn máy |
| 8 | Đánh giá | ASR train chứa 9/10 tập E5; reference Bảng 4.7 tối ưu hoán vị | §6.5 | Số tuyệt đối lạc quan |
| 9 | Speaker arm | Thua baseline trên E5 chủ yếu vì sập chunk do tag lạ, không phải vì tag sai | §5 | Đổi cách kết luận về speaker |

## 1. ASR (Whisper-medium fine-tune + Silero VAD) — `asr_wer_e5.py`, `asr_cost_record_level.py`

WER so với phụ đề EN chính thức (`ASR/ground_truth_asr`), chuẩn hoá bỏ nhãn
`TÊN:`, `(âm thanh)`, dấu câu, lowercase:

| Tập | WER | sub | del | ins |
|---|---|---|---|---|
| movie_054 | 11.9 | 84 | 158 | 39 |
| movie_081 | 8.5 | 59 | 91 | 61 |
| movie_312 | 6.1 | 49 | 128 | 34 |
| movie_104 | 6.1 | 37 | 80 | 30 |
| movie_090 | **10.5** | 74 | 145 | 43 |
| movie_311 | 10.5 | 144 | 167 | 79 |
| movie_291 | 6.1 | 83 | 116 | 32 |
| movie_124 | 6.6 | 57 | 87 | 27 |
| movie_336 | 5.5 | 93 | 99 | 42 |
| movie_170 | 10.1 | 57 | 78 | 96 |
| **Tất cả** | **7.9** | 2.5% | **3.8%** | 1.6% |

- Deletion gấp rưỡi substitution: VAD/garbage-filter/`no_speech_threshold`
  đang cắt lời thật. Luận văn Bảng 4.6 ghi 8.61% (chuẩn hoá khác) — cùng cỡ.
- movie_090 là tập duy nhất **không** nằm trong tập train của Whisper
  (`finetune.py`), WER 10.5% so với trung bình 7.6% của 9 tập còn lại → WER thật
  ngoài train ước ~10–11%.
- Phân mảnh: 521/4614 dòng ASR (11%) không kết thúc bằng dấu câu, 467 (10%)
  bắt đầu bằng chữ thường → mBART dịch mảnh câu, mất chủ ngữ/đại từ.
- **Chi phí ASR đo cùng đơn vị** (3420 gold record, gộp dòng theo overlap thời gian):

| Đầu vào NMT | BLEU | chrF |
|---|---|---|
| EN gold đã gióng (`vietsub_raw`, cùng mBART, beam 1) | **35.34** | 51.25 |
| EN từ ASR (`rough.srt`) | 29.94 | 49.12 |
| EN từ ASR + LLM refine (`v1b_baseline`) | 30.25 | 49.30 |

  ASR (lỗi từ + cắt câu) làm mất **5.4 BLEU**; toàn bộ LLM refine chỉ bù +0.3
  ở đơn vị này. Đây là đòn bẩy lớn nhất còn lại.
- Artifact "NMT từ GT" của tác giả (`ASR/test_data_ground_truth_translated`)
  **không dùng được** để đo trần: 235/702 cue bị tách đôi (hai dòng hiển thị
  cùng timestamp, dịch riêng: "LUIS:" → "- Không."), nên BLEU scene-level chỉ
  28.23, thấp hơn cả ASR (33.49).

### 1b. Lỗi ASR đến từ đâu — `asr_cue_errors.py`, `asr_cue_audio.py`, `asr_missed_attribution.py`

Gióng từ toàn cục rồi quy từng lỗi về cue GT (7 185 cue, 29 876 từ):

| Nhóm cue | số từ | del% | sub% | % tổng deletion |
|---|---|---|---|---|
| tất cả | 29 876 | 3.8 | 2.5 | 100 |
| lời bài hát ♪ | 360 | **61.4** | 13.3 | 19.2 |
| cue ≤ 2 từ ("Cooper.", "Hello.", "Yes, sir.") | 2 249 | 7.7 | 3.7 | 15.1 |
| cue hai người nói "- …" | 1 169 | 6.8 | 4.6 | 7.0 |
| 60 s đầu / cuối tập | 2 587 | 4.8 | 3.0 | 10.9 |
| nói nhanh > 4 từ/s | 4 345 | 3.7 | 1.8 | 14.0 |

- 222 cue mất **hoàn toàn** (662 từ = 58% tổng deletion). Quy nguyên nhân bằng
  cách chạy lại ASR demo trên 10 tập và giữ mọi segment Whisper **trước** bộ
  lọc (`segment_info.json`), cộng VAD Silero cùng ngưỡng 0.2:

| Nguyên nhân | cue | Ghi chú |
|---|---|---|
| chạy lại hôm nay thì nhận ra | 93 | cùng script, greedy T=0 nhưng chunk VAD khác ⇒ **bất ổn theo cách cắt chunk**; WER lần chạy lại lệch −1.7…+1.6 điểm/tập |
| trong vùng VAD nhưng Whisper không sinh | 66 | đa số ≤ 2 từ, nằm giữa chunk dài (VAD gộp khi lặng < 3 s) |
| lời bài hát ♪ | 46 | Whisper bỏ hẳn phần hát; gold VI **có** dịch lời hát |
| Whisper sinh ra, bị garbage-list xoá | 8 | "Okay.", cảm thán |
| VAD bỏ (ngoài vùng speech) | 8 | |
| bị lọc logprob/no_speech/compression | 1 | |

- **Ảo giác lặp**: 2/10 tập có một chunk lặp vô hạn ("Oh, my God." ×29,
  compression ratio 13.9) ⇒ bộ lọc xoá cả chunk, mất luôn lời thật trong đó.
  Pipeline không dùng temperature fallback của Whisper nên không tự phục hồi.
- **Nhạc nền / âm lượng** (proxy: tỉ lệ năng lượng hài âm HPSS, RMS): deletion
  tăng đều 2.6% → 3.2% → 4.1% → 4.8% theo bốn mức hài âm, và 3.8% ở cue
  < −40 dB so với 2.6% ở cue bình thường ⇒ **có ảnh hưởng nhưng vừa phải**;
  khác với Quynh_NMT, ở đây thủ phạm chính không phải nhạc nền mà là
  (i) cách cắt chunk + câu ngắn bị nuốt, (ii) lời bài hát, (iii) ảo giác lặp.
  VAD chỉ chịu trách nhiệm 8/176 cue phi-nhạc bị mất.
- Substitution phần lớn là chuẩn hoá vô hại ("'cause"→"because" 27,
  "going"→"gonna" 12, "four"→"4") ⇒ WER thực chất thấp hơn 7.9%, thiệt hại
  thật nằm ở deletion.

## 2. Scene segmentation + VLM (`llm_errors_e5.py`, phần thống kê chunk)

- 330 scene/10 tập, **0 dòng ngoài scene**; chunk sau `--max_scene_lines 24`:
  median 14, p90 21, max 24.
- 5 caption `None`, trong đó 2 chunk có thoại không có ngữ cảnh.
- Caption có **hai template**: 200 kiểu `1. <mô tả>\n2. [Boy] - Male - Young...`
  và 120 kiểu `1. Summary: ...\n2. Main Characters: [ID] - [Gender] - [Age]...`,
  cộng 5 dạng tự do. Data train v4 dùng template thứ hai → 60% chunk E5 đưa
  ngữ cảnh lệch format so với lúc train.
- Không có gold ranh giới scene để đo.

## 3. NMT (vinai mBART fine-tune)

- Pipeline giải mã greedy (`num_beams=1`, `max_length=128`); khi tạo dataset
  tác giả dùng `num_beams=3` (`infer_nmt.py:146`) → không đồng nhất, và greedy
  là cấu hình yếu nhất.
- Rough: BLEU scene-level 33.49 / record-level 29.94; phần lớn thiệt hại đến từ
  đầu vào ASR (§1), không phải từ bản thân mBART.
- 51 dòng refine trùng nguyên văn EN đều là tên riêng/cảm thán ("Mary?",
  "Amen.") — hợp lệ, không phải lỗi.

## 4. LLM post-editor (Gemma-3-12B QLoRA, adapter `lora_model_scene_v4_4eps`) — `llm_errors_e5.py`

### 4.1 Model sửa gì

4614 dòng E5: LLM đổi **2076 (45%)**; trong đó 737 chỉ đổi đại từ/xưng hô,
**1339 đổi cả nội dung** (dù prompt yêu cầu giữ nguyên nội dung). 15 dòng ngắn
đi <50%, 15 dòng dài hơn 180%.

### 4.2 Sửa có ích không

Per gold record (3420, chrF so với gold): 748 tốt lên (>+1), 608 xấu đi (<−1),
2064 không đổi; trung bình **+0.14 chrF**. Lợi ích thật là đại từ
(F1 0.6506 → 0.6968), còn sửa nội dung là hòa.

Thử offline "chỉ nhận sửa của LLM khi …" (không GPU, chấm giao thức scene-level):

| Luật | dòng trả về rough | BLEU | chrF | F1 đại từ |
|---|---|---|---|---|
| rough | – | 33.49 | 54.11 | 0.6506 |
| v1b_baseline (nhận hết) | 0 | **34.10** | 54.54 | **0.6968** |
| chỉ khi đổi đúng đại từ | 1339 | 34.06 | 54.52 | 0.6846 |
| chrF(out, rough) ≥ 60 | 671 | 33.76 | 54.48 | 0.6868 |
| chrF ≥ 40 | 319 | 33.96 | 54.58 | 0.6941 |
| tỉ lệ độ dài 0.6–1.6 | 132 | 34.09 | 54.48 | 0.6961 |

Không luật nào thắng cả BLEU lẫn F1 → **không đáng làm gate hậu kiểm**; phải
làm model sửa đúng hơn chứ không phải lọc output.

### 4.3 Các lớp lỗi cụ thể (172 record sụt >15 chrF)

1. **Lệch dòng trong chunk** (output dòng k là nội dung dòng k±1): 51 dòng /
   16 chunk ở baseline, 238 dòng nằm trong các chunk lỗi; ví dụ movie_090
   "You know it takes two people…" → "Con biết. Georgie thì sao?". Code hiện
   chỉ xử lý lệch **số** dòng (cắt/độn), không phát hiện lệch **thứ tự**.
2. **Dịch lại từ EN thay vì sửa rough**: "Sorry?" rough "Sao cơ?" (= gold) →
   "Xin lỗi?"; "Hey." "Này." → "Chào."; đổi register sai "Ừ." → "Vâng.";
   paraphrase "Ôi, Chúa ơi!" → "Ôi trời ơi!"; dịch tên riêng "Falcon" → "Chim Ưng".
3. **Sập cả khối**: `output_guard` chỉ bắt dòng không chữ, phi-Latin, lặp
   n-gram hoặc >600 ký tự. Chunk 18 dòng toàn "- Không." (movie_312, arm
   speaker) lọt hết. Sau prompt-fix, E5 còn 2 chunk như vậy (đều arm speaker);
   `eval_ab` chạy **trước** prompt-fix (16–17/08) có 945/6103 dòng baseline
   (15.5%) và 1091/6103 dòng speaker (17.9%) nằm trong chunk sập — nhưng 5 phim
   dùng để quyết định gate V1–V3 (008/009/015/045/046) là **0/1544**, nên các
   kết luận gate vẫn đứng; 14 phim mở rộng thì phải chạy lại.

### 4.4 Lệch phân phối train/inference

`llm_data_scene_v4` (và v5 dẫn xuất): EN = `english` gold (10/10 dòng mẫu
khớp nguyên văn), rough = `vietsub_raw` = mBART dịch EN **gold** (7/10 khớp).
Tức adapter học sửa rough sạch, đủ câu; lúc chạy nhận rough từ EN ASR
(WER 7.9%, 11% mảnh câu). Adapter cũng giòn với mọi format chưa thấy khi train:
context ở system → sập (đã ghi 18/08); tag `[SPEAKER: X]` → sập chunk (mới);
prompt style v4 ≈ hòa (19/08).

### 4.5 Train v5/v5b sụp loss về 0 — root cause cũ sai

- v5 (có `train_on_responses_only`) và **v5b (không có)** đều: loss 8.16 → 0.0003
  trong ~160 step, 0.0 từ epoch 1; checkpoint sinh ra chép rough / in "```".
  Vậy `train_on_responses_only` không phải nguyên nhân.
- Log tác giả, cùng data cùng công thức: `finetune_gemma_v6.log` (Unsloth
  **2026.4.8** + Transformers **4.57.6**) loss 2.27 → 0.89 sau 266 mốc;
  `finetune_gemma_v6_8eps.log` (Unsloth **2026.5.2** + Transformers **5.5.0**)
  2.39 → 0.94. `van_env` hiện tại: unsloth 2026.5.2 (cài 09/05) + transformers
  4.57.6 (cài lại 08/07) + trl 0.24 — **tổ hợp chưa từng có run tốt**.
- Loss bước đầu 8.16 thay vì ~2.3 nghĩa là con số Trainer báo cáo không phải
  CE chuỗi thật ngay từ step 1. **Probe 1** (`probe_label_shift.py`, 4 mẫu
  train, logits thật): checkpoint-737 của v5b dự đoán token kế tiếp đúng
  55.9% (base 56.9%, adapter tác giả 63.9%), CE-shifted 3.67 (base 3.50,
  tác giả 2.36), acc "chép token hiện tại" 0.8% ⇒ adapter **gần như không
  học gì** (không phải học ánh xạ đồng nhất như giả thuyết ban đầu), và
  `model(labels=)` lúc inference vẫn cho CE bình thường. Vậy lỗi nằm trong
  vòng lặp train (collator/`compute_loss`/fused-CE của unsloth với
  transformers 4.57.6), không nằm ở model forward. Hành vi "chép rough" của
  checkpoint chính là hành vi của model gốc với instruction "giữ 90% từ".
  **Probe 2** (`probe_trainer_loss.py`): `SFTTrainer.compute_loss` =
  `model(labels=).loss` = CE tính tay = **8.09** trên batch train ⇒ Trainer
  không tính sai; chính forward của model trên batch train cho CE sai.
  **Probe 3** (`probe_double_bos.py`): batch train có `<bos>` nhân đôi
  (`[2, 2, …]`) nhưng CE chỉ 3.83 vs 3.75 ⇒ không phải nguyên nhân.
  **Probe 4** (`probe_causal_mask.py`, cùng 2 mẫu, cùng LoRA khởi tạo, batch
  có padding + attention_mask, cả eval lẫn train mode):

  | attention | loss batch | nhân quả (đổi token t+5, logits ≤ t) |
  |---|---|---|
  | mặc định (như `train_refiner_v5.py`) | **8.93** | không rò rỉ |
  | `attn_implementation="eager"` (như `finetune_gemma_v6.py`) | **3.97** | không rò rỉ |

  ⇒ **Nguyên nhân gốc**: đường attention mặc định của unsloth cho Gemma-3
  tính sai khi có attention_mask/padding trong chế độ train (tác giả đã ghi
  chú đúng điều này trong script: "bypass buggy SDPA path in unsloth Gemma-3
  patch" và giảm MAX_SEQ_LENGTH 3072→2048 vì "sliding-window attention mask
  bug"), còn `train_refiner_v5.py` không truyền `attn_implementation="eager"`.
  Với forward sai, optimizer tìm được nghiệm tầm thường (loss→0) mà không
  học được gì dùng được lúc inference (đường inference không có padding
  batch nên forward đúng ⇒ checkpoint ≈ base). Lệch phiên bản chỉ là nghi
  vấn phụ; sửa bắt buộc là `attn_implementation="eager"` + smoke test.
- Khác biệt còn lại với script tác giả (nên bám luôn): base
  `google/gemma-3-12b-it` (không phải `unsloth/...-bnb-4bit`), chat_template
  chép từ tokenizer Google, `lora_dropout=0`.

## 5. Speaker attribution (CAM++ + đặt tên bằng LLM)

| Tập | dòng có tên | cluster | coverage align |
|---|---|---|---|
| 054 | 76% | 16 | 0.76 |
| 081 | 74% | 25 | 0.74 |
| 090 | **35%** | 16 | 0.73 |
| 104 | 57% | 13 | 0.71 |
| 124 | 54% | 17 | 0.74 |
| 170 | 45% | 16 | 0.55 |
| 291 | 73% | 25 | 0.79 |
| 311 | 55% | 23 | 0.72 |
| 312 | 59% | 12 | 0.66 |
| 336 | 58% | 22 | 0.76 |

- Không có gold tên người nói trong dataset. Gold `age`/`gender` là dự đoán từ
  face-track (`speaker_data/*/results.json`: `age` float, `gender`, khớp theo
  `speaking_intervals`), 36.4% record null → `speaker_quality.py` đang so với
  nhãn máy.
- movie_312 arm speaker BLEU 39.9 → 35.7, COMET 0.819 → 0.756: 42 dòng
  fallback + 2 chunk sập ("- Không." ×18, "- Cái đó" ×17) với registry rỗng →
  nguyên nhân là tag lạ làm adapter sập, không phải tag sai.

## 6. Ground truth — `gold_quality.py`

1. Dataset 142 005 record / 356 phim, 0 record trống. 10 711 record (7.5%)
   chồng thời gian với record trước; 1 569 record tỉ lệ từ VI/EN ngoài
   [0.33, 3] (nghi gióng sai); 627 record EN còn nhãn `TÊN:`, 2 481 còn `(…)`
   (dòng đầu movie_054: "(View-Master clicks) ADULT SHELDON: you'll find…").
2. Trên 10 tập E5 gold phủ **94.2%** số từ phụ đề VI chính thức (mất ~1 950
   từ); 27–92 record/tập là gộp ≥2 cue (dấu hai khoảng trắng).
3. **Nhãn đại từ** sinh bằng Gemma-3-12B (`process_pronouns.py`) + luật +
   Gemma lọc + kiểm tra "có xuất hiện trong câu" (0/137 470 nhãn vắng mặt) —
   nhưng không kiểm tra ngữ nghĩa: ≥85 record gắn `con` là đại từ khi nó là
   loại từ ("con mồi", "con gà"; regex hẹp nên số thật cao hơn). Giao thức
   Bảng 4.8 (`thesis_score.py`) đếm trùng đại từ lồng nhau ("anh ấy" vừa khớp
   "anh ấy" vừa "anh"). Nhãn và model refine cùng họ Gemma-3 → bias chung.
4. Gold tuổi/giới = face-track (§5).
5. Đánh giá: ASR train chứa 9/10 tập E5 (`finetune.py`); `calculate_bleu.py`
   hoán vị dòng reference cùng timestamp để tối đa BLEU → số tuyệt đối Bảng
   4.7 lạc quan, chỉ so tương đối được; `eval_gemma.py` chấm trên train; XLM-R
   chọn checkpoint trên test (đã ghi 19/08).

## 7. Đã kiểm và không có vấn đề

Số dòng en/rough/refined khớp 10/10 tập; 0 dòng ngoài scene; guard bắt 0 dòng
lọt ở baseline sau prompt-fix; pipeline tất định; 5 phim gate speaker sạch.
