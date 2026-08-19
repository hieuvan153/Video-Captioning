# A/B V3 — làm sạch tag cấp dòng + address edges (2026-08-17)

4 arm trên 5 phim eval, cùng đợt `v1b_` (tái dùng `v1b_baseline`/`v1b_speaker`
của 2026-08-16 theo tiền lệ V2a — pipeline tất định, cùng config), paired
bootstrap per-line 2000 lần. Số gốc: `docs/eval/speaker_ab_v3.json` (ref
baseline) và `docs/eval/speaker_ab_v3_vs_speaker.json` (ref speaker); SRT:
`demo/output/eval_ab/<movie>/v1b_{speaker_v3,speaker_v3_addr}.srt`; speakers:
`speakers.v3.json` (label_override + min_score theo sweep
`docs/eval/speaker_quality_v3.md`, `address_edges` từ vocative EN).

Plan tiền đăng ký: `docs/superpowers/plans/2026-08-17-line-level-tags-v3.md`.
Tái lập: `demo/EVAL/run_arms.py --arms speaker_v3 speaker_v3_addr --prefix v1b_`
rồi `demo/EVAL/compare_arms.py`.

## Kết quả

| phim | baseline | speaker (V1) | speaker_v3 | v3_addr | Δ v3 vs baseline | Δ addr vs baseline |
|---|---|---|---|---|---|---|
| movie_008 | 0,4744 | 0,4934 | 0,4738 | 0,4738 | −0,0005 (p=0,979) | −0,0005 (p=0,979) |
| movie_009 | 0,6016 | 0,5875 | 0,5875 | 0,5856 | −0,0141 (p=0,316) | −0,0160 (p=0,282) |
| movie_015 | 0,4892 | 0,5272 | 0,5260 | 0,5350 | **+0,0367 (p=0,000)** | **+0,0458 (p=0,000)** |
| movie_045 | 0,3818 | 0,3634 | 0,3935 | 0,3935 | +0,0117 (p=0,608) | +0,0117 (p=0,608) |
| movie_046 | 0,4221 | 0,4878 | 0,4514 | 0,4514 | +0,0292 (p=0,232) | +0,0292 (p=0,232) |
| **gộp** | **0,4789** | **0,4962** | 0,4920 | 0,4938 | +0,0131 (p=0,088) | +0,0149 (p=0,060) |

Trung bình delta theo phim vs baseline (đại lượng của gate): speaker **+0,0180**,
speaker_v3 **+0,0126**, speaker_v3_addr **+0,0140**.

So trực tiếp với `speaker` (điều kiện phụ của gate, paired bootstrap riêng):

| phim | Δ v3 vs speaker | Δ addr vs speaker | dòng đổi speaker→v3 | dòng đổi v3→addr |
|---|---|---|---|---|
| movie_008 | **−0,0196 [−0,036;−0,005] p=0,007** | như v3 | 59 | 0 |
| movie_009 | 0,0000 [0;0] | −0,0018 (p=0,491) | 2 | 9 |
| movie_015 | −0,0012 (p=0,716) | +0,0078 (p=0,369) | 4 | 47 |
| movie_045 | +0,0301 [−0,015;+0,072] p=0,187 | như v3 | 92 | 0 |
| movie_046 | **−0,0364 [−0,066;−0,005] p=0,016** | như v3 | 78 | 0 |

Trung bình delta theo phim vs speaker: v3 **−0,0054**, addr **−0,0040**.

## Gate: KHÔNG đạt — rút cả hai arm

- Gate chính (baseline +0,03): v3 +0,0126, addr +0,0140 — chưa bằng cả arm
  `speaker` cũ (+0,0180).
- Điều kiện phụ 1 (`speaker_v3` không được thua `speaker`): **VI PHẠM** —
  −0,0054 trung bình theo phim, trong đó 008 (−0,0196, p=0,007) và 046
  (−0,0364, p=0,016) thua có ý nghĩa thống kê.
- Điều kiện phụ 2 (`addr` không được thua `v3`): đạt (+0,0140 ≥ +0,0126),
  nhưng vô nghĩa khi nền v3 đã thua.

Theo tiền đăng ký: rút `speaker_v3` và `speaker_v3_addr`. **Arm tốt nhất vẫn là
`speaker` (V1) +0,0180.**

## Cơ chế: Stage B sửa đúng chỗ nhắm nhưng phá hai chỗ khác

- **movie_045 — mục tiêu của Stage B — thành công đúng như nhắm:** bỏ anchor
  sai "Missy"→SPK_106 + label_override đổi 92 dòng, phim duy nhất còn âm ở V1
  chuyển +0,0301 so với speaker (dù chưa significant). Nguyên tắc "tag sai tệ
  hơn không tag" đúng ở đây.
- **movie_008 (−0,0196) và movie_046 (−0,0364), cả hai significant:** cùng
  đợt làm sạch đó đổi 59/78 dòng và xóa cả tag ĐÚNG (cluster mất anchor →
  vô danh → mất tag; 046 mất một nửa mức lợi +0,0657 của V1). Đây là mặt trái
  đã ghi ở mục Rủi ro của plan, nhưng offline proxy không nhìn thấy trước —
  xem bài học dưới.
- **Address edges (C.b) hoạt động đúng như đo offline nhưng quá nhỏ:** kích
  hoạt 009 (3 scene, 9 dòng đổi, −0,0018), 015 (15 scene, 47 dòng đổi,
  +0,0078 — arm tốt nhất của phim này, +0,0458 vs baseline), 046 (15 scene
  trùng tập pair đã có → 0 dòng đổi). Soi riêng movie_009 theo plan: film
  gate lần đầu bật ở đó nhưng kết quả phẳng/hơi âm — cạnh vocative không giúp
  phim này.

## Bài học cho proxy offline

Luật chọn sweep (max mean gender consistency, giữ ≥85% dòng có tên) chọn được
config làm 045 tốt lên, nhưng **proxy gender mù với hai chế độ lỗi quyết định
kết cục**: (1) mất tag đúng trên cluster cùng giới (008/046) không làm giảm
gender consistency; (2) F1 đại từ nhạy với MẤT coverage hơn mức ràng buộc 85%
cho phép. Nếu còn vòng sau: proxy phải phạt mất-tag-theo-dòng so với arm đang
thắng, không chỉ đo độ thuần của tag còn lại.

## Trạng thái sau V3

Ba đợt liên tiếp (V2a/b/c, V3) đều không vượt được `speaker` V1 +0,0180.
Điểm sáng duy nhất tái lập được: registry theo scene khi THẬT SỰ kích hoạt
(046 ở V1, 015 qua addr ở V3) đều cho phim tốt nhất bảng — cơ chế đúng hướng
nhưng độ phủ kích hoạt vẫn là nút thắt, và mọi cách nới kích hoạt tới nay đều
trả giá ở chỗ khác.
