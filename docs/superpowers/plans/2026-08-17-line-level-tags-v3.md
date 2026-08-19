# Plan V3 — làm sạch tag cấp dòng + đưa relation vào prompt (2026-08-17)

Tiếp nối tổng kết V2 (docs/superpowers/plans/2026-08-16-scene-registry-v2.md):
arm tốt nhất vẫn là `speaker` (V1) +0,0180, thiếu +0,012 tới gate +0,03;
movie_045 vẫn âm. Hai hướng V3 đã tiền đăng ký ở đó: (1) nhãn kịch bản trong
chunk phủ quyết tên cluster cho RIÊNG dòng đó; (2) đo `--min_score` (align hỗ
trợ, chưa từng đo). Thêm yêu cầu mới từ người dùng: relation phải THẬT SỰ vào
prompt Gemma — hiện chỉ 6/157 scene (toàn bộ ở movie_046).

## Stage A — bộ chấm attribution offline (không GPU)

Ground truth `data/en-vi-speaker-with-time-pronouns/` có `gender` (~72% dòng)
và `age` (float) per line, khớp 1:1 theo vị trí với SRT eval. Không có tên
nhân vật vàng, nhưng (gender, age-bucket) của tên được gán (tra registry) so
với (gender, age) vàng của dòng là proxy đủ mạnh: cross-tab thủ công đã bắt
đúng lỗi đã biết ("Missy" ở movie_045: 41 nam / 12 nữ).

`demo/EVAL/speaker_quality.py`: chấm một file speakers.json (coverage, gender
consistency, age consistency, cross-tab per-name) + mode sweep re-align CPU
(tái dùng mapping cluster→tên đã lưu, không gọi LLM).

**Ranh giới cứng: gold chỉ dùng để ĐO. Không một byte gold nào được làm input
pipeline** — vì vậy module nằm ở EVAL/, SPEAKER/ và LLM/ không import nó.

Sanity trước khi tin proxy: (1) tái hiện cross-tab đã biết; (2)
`speakers.season.json` (đã sửa SPK_106→Sheldon) phải cho gender consistency
movie_045 tốt hơn `speakers.json`. Sai (2) → dừng, proxy hỏng.

## Stage B — làm sạch tag cấp dòng

1. **Sửa ngữ nghĩa anchor** (`name_evidence.py`): chỉ nhãn ở ĐẦU text (sau
   noise mở đầu kiểu "(school bell rings)") mới anchor cả cluster. Nhãn giữa
   chunk là bằng chứng về phần ĐUÔI chunk — đo được nó đặt tên sai cả cluster:
   "Asked and answered. MISSY: Did you cry…" (c248) gán "Missy" cho SPK_106
   (65 chunk Sheldon giảng vật lý). Hệ quả chấp nhận: SPK_106 mất anchor sai,
   exclusion {Sheldon} (cluster không thuần tự nhắc tên người nói thật) có thể
   chặn LLM → cluster vô danh → ~60 dòng mất tag. Đúng nguyên tắc đã đo: tag
   sai còn tệ hơn không tag.
2. **Override nhãn cấp dòng** (`SPEAKER/label_override.py` mới): cắt chunk tại
   thời điểm nhãn (định vị bằng `english_word_timestamps`) thành vùng tag gốc
   + vùng `LABEL::Tên`, TRƯỚC khi align — vùng nhãn cạnh tranh trong chính máy
   overlap-trội của `assign_speakers`, dòng vắt ranh giới do overlap trội
   quyết, hòa → None. Tag LABEL:: miễn lọc `min_score` (bằng chứng text, không
   phải embedding). Bonus: nhãn đầu text trên chunk UNKNOWN của CAM++ ("INGRAM:
   Sheldon?") trước vô dụng, giờ thành vùng có tên.
3. **Đo `--min_score`**: nối qua `run_arms.py` với file riêng
   `speakers.v3.json` (cache theo tồn-tại-file). Sweep offline bằng Stage A:
   {None, 0.5, 0.55, 0.6, 0.65} × {label_override on/off} × 5 phim.
   **Luật chọn tiền đăng ký:** max mean gender consistency theo phim, ràng
   buộc giữ ≥85% số dòng có tên hiện tại; hòa → coverage cao hơn. Bảng ghi
   vào docs/eval/speaker_quality_v3.md.
4. **A/B GPU**: build `speakers.v3.json` thật (rẻ) → chấm lại bằng Stage A.
   **Go/no-go tiền đăng ký:** file v3 thắng file cũ về mean gender
   consistency/phim VÀ không làm movie_045 xấu đi; fail → lặp offline, không
   tốn refine. Đạt → arm `speaker_v3`, prefix `v1b_` (pipeline tất định, tái
   dùng baseline/speaker của v1b như tiền lệ V2a), so 3 arm bằng
   compare_arms.py → docs/eval/speaker_ab_v3.{json,md}.

## Stage C — đưa relation vào prompt (gate: chỉ khi speaker_v3 ≥ speaker)

Bài học V2a: relation khuếch đại tên — tên phải tốt lên trước. Bài học V2b:
cạnh kinship-high và cặp thực sự thoại kề nhau là hai tập rời nhau trên 4/5
phim → registry hiện tại KHÔNG THỂ kích hoạt thêm; cần nguồn cạnh mới nằm
đúng trên cặp đang nói chuyện.

- **C.0** `demo/EVAL/registry_activation.py`: đếm scene kích hoạt offline
  (tái hiện bảng V2 không cần GPU). Tách `find_best_scene` +
  `scene_line_indices` từ refine_llm.py sang `demo/LLM/scene_assign.py`
  (GPU-free) để hai bên dùng chung một logic.
- **C.a** nới LOẠI cạnh, giữ pair-gating V1: thêm cạnh `vi_listener ∈
  KINSHIP_TERMS`, vi_self bất kỳ, confidence high. Đo offline trước;
  **chỉ chạy GPU nếu kích hoạt ≥2 phim ngoài movie_046.** Ra ~0 = kết quả âm
  hợp lệ, biện minh cho C.b.
- **C.b** đào cạnh từ vocative EN (`SPEAKER/vocative_edges.py` mới): term thân
  tộc EN ("Mom", "Grandma", "Uncle"…) ở vị trí vocative trong chunk của
  cluster có tên A → vote cho hàng xóm kề có tên B (cả trước lẫn sau,
  min_votes=2). Cạnh theo cấu trúc nằm ĐÚNG trên cặp đang nói chuyện. Lexicon
  EN→VI chỉ thân tộc đợt đầu; guardrail: xung đột gender registry → bỏ, term
  mâu thuẫn cấu trúc trên cùng cặp → bỏ, hai đầu phải resolve qua alias index;
  cạnh ngược phát ở medium. Lưu vào `speakers.v3.json` trường `address_edges`
  (trơ với scope `pair` → một lần build phục vụ cả hai arm). Render qua
  `render_scene_registry_context(..., extra_edges=…)`, vẫn MỘT dòng prose,
  scope mới `pair_rel` trong refine_llm.py, arm `speaker_v3_rel`.

  **Go/no-go offline tiền đăng ký trước GPU:** kích hoạt ≥3/5 phim và ≥15/157
  scene, VÀ audit bằng proxy Stage A: <20% cạnh có term người-nghe xung đột
  với đa số gender vàng các dòng của speaker đó. Fail → rút C.b, 0 GPU.

## Gate

Không đổi: Pronoun F1 mean delta theo phim ≥ baseline + 0,03 (so trong cùng
đợt v1b, paired bootstrap như cũ). Điều kiện phụ: `speaker_v3` không được
thua `speaker`; `speaker_v3_rel` không được thua `speaker_v3` (thua → rút C,
giữ B). Soi riêng movie_009 (film gate có thể lần đầu bật ở đó qua C.b).

## Rủi ro

- Bỏ anchor sai của SPK_106 → ~60 dòng 045 có thể mất tag: chấp nhận theo
  nguyên tắc, hiệu ứng F1 thật chỉ biết ở arm GPU.
- Sweep offline tái dùng mapping cũ; rebuild thật (LLM thấy anchor mới) có
  thể lệch → luôn chấm lại file thật trước khi tốn refine.
- Proxy gender không thấy lỗi sai-tên-cùng-giới; tune trên đúng 5 phim eval
  dễ overfit → luật chọn đơn giản, tiền đăng ký, gate cuối vẫn là Pronoun F1.
- "Mom" trong lời kể lại (không phải gọi trực tiếp) → cạnh giả: min_votes +
  vị trí vocative + guardrail gender giảm thiểu.

## Addendum V3.1 (2026-08-17, sau khi V3 bi rut)

Hai viec, theo yeu cau nguoi dung sau bao cao V3:

1. **Sua bug mega-node merge_registries** (nguoi dung tim ra 2026-08-17):
   placeholder ("You", "I"...) trong names cua nhan vat ten-that tham gia
   union-find -> gop bac cau (movie_009 C38 = Steve+Glenn+Jonah+God+16
   placeholder; relation nhiem: 008 43/59, 015 24/54). Fix: bo qua key thuoc
   PLACEHOLDER_NAMES trong vong alias cua merge_registries + test hoi quy.
   Khong co cache registry per-chunk -> rebuild GPU (generate greedy tat
   dinh, chi buoc merge khac) ra `registry.fixed.json`, do offline:
   so nhan vat, quy mo node lon nhat, kich hoat pair/pair_addr.

2. **V3.1 — anchor giua-chunk theo phieu** (`mid_anchor_votes=2`): V3 bo het
   anchor giua-chunk lam 008/046 mat tag dung (−0,0196/−0,0364, ca hai
   significant). V3.1 giu label_override + anchor dau-text, nhung nhan
   giua-chunk van anchor ca cluster neu CUNG ten o >= 2 chunk khac nhau —
   phieu don le kieu SPK_106 ("MISSY" x1/65 chunk) van bi loai. Proxy duoc va
   theo dung bai hoc V3: them `tag_retention` (mat/them/doi ten so voi arm
   tham chieu `speakers.json`).

**Luat go/no-go tien dang ky (offline sweep, truoc khi nhin so):** build
`speakers.v3_1.json` (GPU) chi khi ca ba: (a) mean gender consistency v3.1
>= v3 − 0,005 cung khung sweep; (b) movie_045 gender >= v3 − 0,01; (c) tong
n_lost vs speakers.json tren 008+046 giam >= 30% so voi v3. Dat -> cham file
that (B.4 nhu cu: thang mean gender, 045 khong xau di, 008+046 lost giam) ->
moi chay arm `speaker_v3_1` (prefix v1b_, tai dung baseline/speaker cu).
Gate cuoi khong doi: mean delta/phim vs baseline, arm moi khong duoc thua
`speaker` V1 (+0,0180).
