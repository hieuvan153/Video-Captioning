# Sweep offline V3 — proxy gender/age cho chất lượng tag (2026-08-17)

Công cụ: `demo/EVAL/speaker_quality.py` (mới). Proxy: (gender, age-bucket) của
tên được gán, tra từ registry, so với (gender, age) vàng per-line của
`data/en-vi-speaker-with-time-pronouns/`. Gold **chỉ để đo** — không byte nào
vào pipeline. Số gốc: `speaker_quality_v3_sweep_base.json` (chỉ min_score),
`speaker_quality_v3_sweep_lo.json` (label_override + fix_anchors),
`speaker_quality_v0_speakers.json` (per-film chi tiết của speakers.json).

## Sanity proxy — ĐẠT cả hai anchor (điều kiện tin proxy)

1. Tái hiện đúng cross-tab đã biết: movie_045 "Missy" **41 nam / 12 nữ**;
   movie_046 "George Jr." **38 nam / 14 nữ**.
2. `speakers.season.json` (V2c đã sửa SPK_106→Sheldon) cho gender consistency
   movie_045 **0,5000 > 0,4286** của `speakers.json`. (Caveat: n giảm 140→28
   vì tên cấp mùa như "Sheldon Lee" không phải alias trong registry tập này —
   proxy mù với tên ngoài registry.)

Phát hiện phụ nhờ age-bucket: SPK_106 ("Missy", 73 dòng) không chỉ trộn
Missy+Sheldon như audit V2c ghi — age vàng của nó là elderly:20 / child:15 /
adult:12 / teen:6, tức trộn **cả Dr. Sturgis** (người giảng QCD trong tập này).
Cluster trộn 3+ người thì mọi cách đặt tên đều sai cho đa số dòng — củng cố
hướng làm sạch Ở CẤP DÒNG.

## Bảng sweep (mean theo 5 phim; floor coverage tiền đăng ký = 85% hiện tại
= 0,85 × 0,6818 = **0,5795**)

| cấu hình | gender | coverage | đạt floor? |
|---|---|---|---|
| hiện tại (không lọc) | 0,5373 | 0,6818 | ✓ |
| min_score 0,5 | 0,5421 | 0,5793 | ✗ (thiếu 0,0002) |
| min_score 0,55 | 0,5380 | 0,4735 | ✗ |
| min_score 0,6 | 0,5493 | 0,3370 | ✗ |
| min_score 0,65 | 0,5460 | 0,2181 | ✗ |
| **label_override + anchor mới** | **0,5561** | 0,5846 | ✓ |
| label_override + anchor mới + ms 0,5 | 0,5594 | 0,4900 | ✗ |

## Kết luận tiền đăng ký

1. **min_score: KHÔNG dùng (giữ None).** Giả thuyết V2 "chunk khớp embedding
   yếu chính là chunk thuộc cluster trộn" bị bác bằng số: nâng ngưỡng đến 0,65
   chỉ nhích gender +0,009 trong khi coverage sập 0,68→0,22, và movie_045 —
   phim cần cứu — còn TỆ ĐI (0,4286→0,2353). speaker_score không phải tín hiệu
   precision cho lỗi đặt tên.
2. **Điểm vận hành cho speakers.v3.json = thay đổi code B.1+B.2** (anchor chỉ
   nhận nhãn đầu-text + LABEL:: override cấp dòng), không flag thêm →
   `SPEAKER_BUILD_FLAGS["speakers.v3.json"] = []`. Hiệu ứng offline (xấp xỉ
   fix_anchors trên mapping cũ): movie_045 gender 0,4286→**0,5632** (đổi bằng
   coverage 57,7%→34,0% — đúng trade "tag sai tệ hơn không tag" đã đo);
   movie_046 gender 0,6643→0,6237 nhưng age 0,4336→0,5269.
3. Xấp xỉ offline không mô phỏng được LLM re-prompt khi anchor đổi (046 mất
   cả anchor giữa-chunk ĐÚNG mà rebuild thật có thể tự tìm lại tên). Số quyết
   định là chấm **file speakers.v3.json thật** sau rebuild — go/no-go B.4:
   thắng mean gender consistency và không làm movie_045 xấu đi.

## Rebuild thật (cùng ngày) — go/no-go B.4: ĐẠT

`speakers.v3.json` build bằng GPU (chỉ bước map cluster→tên), chấm bằng cùng
proxy (số gốc: `speaker_quality_v3_real.json`):

| phim | gender v1 → v3 | coverage v1 → v3 |
|---|---|---|
| movie_008 | 0,6000 → 0,5906 | 68,4% → 65,6% |
| movie_009 | 0,4724 → 0,4762 | 56,1% → 55,8% |
| movie_015 | 0,5212 → 0,5123 | 75,8% → 75,2% |
| movie_045 | 0,4286 → **0,6056** | 57,7% → **58,3%** |
| movie_046 | 0,6643 → **0,7163** | 82,9% → 81,3% |
| **mean** | 0,5373 → **0,5802** | 0,6818 → 0,6725 |

Xấp xỉ offline đã ĐÁNH GIÁ THẤP hiệu quả: nó dự đoán movie_045 mất coverage
(58%→34%) vì giả định cluster mất anchor sẽ vô danh; thực tế LLM được giải
phóng khỏi anchor sai đã đặt lại tên — SPK_106 (65 chunk giảng QCD, trước là
"Missy" 41M/12F) giờ là **"Dr. Sturgis"** (105 dòng, elderly:36 chiếm đa số
age vàng — bài giảng QCD trong tập này đúng là lớp của Dr. Sturgis, điều mà
cả audit tay V2c cũng đoán sai khi gán "Sheldon Lee"). movie_046: SPK_027
(trộn George Jr./Sheldon) giờ mang "Sheldon Lee Cooper".

Hiệu ứng dây chuyền lên registry theo scene: mode `pair` trên movie_046 kích
hoạt **15/26 scene** (tag v1: 6/26) — tên đúng hơn ⇒ nhiều cặp kinship khớp
được với người thật sự cầm mic (`registry_activation_v3_pair.json`).

## Stage C — quyết định offline (tiền đăng ký, cùng ngày)

- **C.b (đào cạnh vocative): RÚT, 0 GPU.** `mine_address_edges` trả **0 cạnh**
  trên cả 5 phim. Nguyên nhân đo được (debug movie_046): chỉ ~5 chunk có
  vocative thân tộc; các ứng viên đều bị guardrail giết ĐÚNG — "Mom, can this
  wait?" của Sheldon đứng cạnh cluster "Dad" (adjacency cấp chunk trỏ nhầm
  người nghe, guardrail gender chặn cạnh độc Sheldon→Dad "mẹ"); phần còn lại
  là chunk UNKNOWN vô danh cắt mạch kề. Từ vựng xung hô EN trong hai series
  này quá thưa cho cơ chế này.
- **C.a (nới loại cạnh, giữ pair-gating): GO.** Kích hoạt offline với tag v3:
  009: 3, 015: 15, 046: 15 scene (`registry_activation_v3_pair_addr.json`) —
  đạt tiêu chí "≥2 phim ngoài 046". Lưu ý rủi ro đã thấy: context 015 chứa
  cạnh dính tên placeholder ("Man", "Speaker 1") — arm GPU sẽ phân xử.
- Arm `speaker_v3_rel` bỏ (≡ `speaker_v3` từng byte khi address_edges rỗng).
