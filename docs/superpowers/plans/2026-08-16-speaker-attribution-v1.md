# Plan V1 — Speaker Attribution (2026-08-16)

Gate V1 đã mở theo khuyến nghị của `docs/eval/error_analysis_v0.md`: registry V0
đúng nội dung nhưng **không biết dòng nào ai nói với ai**, nên rơi về xưng hô
generic. Bằng chứng: FP của arm registry bị chi phối bởi "tôi" (194) / "anh"
(176) trong khi FN toàn từ có hướng (cô 72, anh 57, ông 51, cháu 45); case
movie_008 có **đúng** edge `Glenn calls Reporter "cô"` nhưng vẫn gọi nhầm "anh"
vì không biết dòng đó thoại với ai.

## Điều kiện đầu vào — đã có sẵn, không phải chạy lại CAM++

`data/speaker_verify_campp_en_full/` chứa kết quả pipeline CAM++ đã chạy xong:

- `series/<phim>/Mùa_N/episodes/movie_XXX/movie_XXX.tagged.json` — **356 tập**,
  mỗi VAD-chunk có `start`, `end`, `english`, `speaker_tag` (`SPK_029`),
  `speaker_score`, `speaker_score_details`.
- `series/<phim>/Mùa_N/db/season_speakers.json` — embedding bank theo mùa
  (`max_embeddings_per_speaker: 10`), giữ speaker nhất quán xuyên tập.

Đo trên 5 phim eval (2026-08-16): chunk mịn hơn dòng SRT (movie_045: 366 chunk /
312 dòng) nhưng **cùng trục thời gian và cùng text** — `chunk[0] 2.46–7.21` vs
`srt[0] 2.458–7.238`. Gióng theo overlap: **312/312 dòng có overlap**, chỉ 36
dòng (11,5%) chồng từ 2 speaker trở lên → cần tie-break theo overlap trội.

Vậy V1 **không cần GPU cho bước speaker**, chỉ cần GPU cho bước map cluster→tên
và cho eval refine.

## Kiến trúc

Thêm package `demo/SPEAKER/`, theo đúng convention `CHARACTER/`: logic thuần
tách khỏi GPU runner, `generate_fn` inject được để test.

```
SPEAKER/
  align.py        # tagged.json + en.srt -> speaker_tag cho tung dong (thuan, no-GPU)
  cluster_map.py  # SPK_* -> ten nhan vat bang LLM (generate_fn inject duoc)
  __init__.py
```

### Bước 1 — `align.py` (nền, không GPU)

`assign_speakers(chunks, subs) -> list[str | None]`

- Với mỗi dòng SRT, tính overlap thời gian với mọi chunk; chọn `speaker_tag` có
  **tổng overlap lớn nhất** (không phải chunk đầu tiên chạm vào).
- Dòng không overlap ai → `None`, và **phải** render thành không có tag chứ
  không đoán bừa — đoán sai speaker chính là failure mode V0 đang muốn sửa.
- Trả kèm thống kê để log: số dòng có tag, số dòng tranh chấp nhiều speaker.

### Bước 2 — `cluster_map.py` (LLM)

`SPK_029` vô nghĩa với model refine; cần map sang tên thật.

- Input: với mỗi cluster, lấy N dòng thoại tiêu biểu (ưu tiên `speaker_score`
  cao = khớp embedding chắc nhất) + danh sách nhân vật từ `registry.json`.
- LLM chọn tên trong **tập đóng** là `registry.characters` (không cho bịa tên
  mới) hoặc trả `unknown`.
- Xử lý câu ngắn ("You know.", "Me.") theo Huh & Zisserman: câu quá ngắn không
  đủ ngữ cảnh thì dựa vào các dòng dài cùng cluster, vì cluster đã gom sẵn.
- Validate cứng: tên không thuộc registry → `unknown`. Cluster `unknown` →
  dòng đó **không** có tag (xem lại nguyên tắc ở Bước 1).

### Bước 3 — tiêm `[SPEAKER: X]` per line

Prompt refine hiện tại (xem `LLM/refine_llm.py`) gom theo scene:
`raw_en` = các dòng EN nối bằng `\n`, `vinai_sub` tương ứng, output phải giữ
đúng số dòng. Tiêm vào **phía EN** để không đụng số dòng output:

```
<English Dialogue>
[SPEAKER: Sheldon] I'd like to speak with a loan officer.
[SPEAKER: Loan Officer] Sure, have a seat.
</English Dialogue>
```

An toàn vì system prompt đã nói "Use the English dialogue only to understand
speaker context" — tag nằm đúng chỗ nó phục vụ.

### Bước 4 — registry per-line thay cho 1 dòng toàn phim

Đây là **thay đổi cơ chế** mà error analysis yêu cầu, không chỉ là thêm tag.

- V0: `render_registry_context()` render 1 dòng quan hệ chung, tiêm vào **mọi**
  scene → vocab leak sang dòng không liên quan (movie_008: 16/24 FP mới nằm
  trong vocab dòng tiêm; movie_045: 20/33).
- V1: với mỗi scene, chỉ render edge giữa **các speaker thực sự xuất hiện trong
  scene đó**. Listener suy ra từ lượt thoại kề (turn-taking): dòng của speaker A
  nằm giữa các dòng của B → cặp (A→B).
- Giữ nguyên guardrail V0: chỉ edge thân tộc confidence high, ≥2 cặp
  (`KINSHIP_TERMS`, `min_kinship_pairs`) — đã đo được là edge generic gây hại
  significant.

## Kiểm chứng

- Unit test cho `align.py` và `cluster_map.py` (thuần, không GPU) — nối tiếp bộ
  47 test hiện có.
- A/B trên đúng 5 phim eval V0 (`demo/output/eval_ab/`), 3 arm:
  `baseline` / `registry` (V0) / `registry+speaker` (V1), chấm bằng
  `EVAL/run_eval.py` trên pipeline **đã có guardrail chống suy biến** để không
  lặp lại artifact movie_015.
- Báo cáo kèm paired bootstrap per-line như error analysis V0.

**Gate:** Pronoun F1 ≥ baseline + 0.03 (baseline paper E5 = 0.5806). Đích cuối
V2 ≥ 0.70.

## Rủi ro

- **Speaker sai còn tệ hơn không có speaker.** Tag sai đẩy model chọn dứt khoát
  một xưng hô sai, trong khi không tag thì nó rơi về generic (sai nhẹ hơn). Vì
  vậy mọi bước đều thiên về `None`/`unknown` khi không chắc, và cần đo riêng độ
  chính xác của cluster→tên trước khi tin vào số F1 cuối.
- Cluster của CAM++ có thể gộp 2 người giọng giống nhau, hoặc tách 1 người làm
  nhiều cluster. Cái sau vô hại (2 cluster cùng map về 1 tên), cái trước độc —
  cần nhìn `speaker_score` để lọc.
