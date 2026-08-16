# Plan V2 — mở rộng độ phủ của registry theo scene (2026-08-16)

Kết quả V1 (docs/eval/speaker_ab_v1.md): cơ chế per-scene registry chỉ kích
hoạt ở movie_046 (6/26 scene; 0/131 scene ở 4 phim còn lại) — và movie_046
chính là phim có delta lớn nhất bộ (+0,0657, p=0,001), trong khi 4 phim "chỉ
có tag" trung bình +0,0053 (nhiễu). Giá trị nằm ở chỗ thu hẹp registry về đúng
người đang nói; V2 phải làm cơ chế đó **kích hoạt được** trên các phim còn lại.

Điều kiện kích hoạt V1 cần ba điều kiện đồng thời, mỗi tầng đều hao hụt:
cluster→tên đúng, tên trùng đầu cạnh registry, hai người thoại kề nhau. Đo
2026-08-16 (số scene kích hoạt / tổng, theo từng mức nới):

| phim | V1 (cặp + kề nhau) | đồng hiện diện | chỉ cần 1 đầu |
|---|---|---|---|
| movie_008 | 0/32 | 0 | 0 |
| movie_015 | 0/35 | 0 | 19 |
| movie_045 | 0/32 | 1 | 19 |
| movie_046 | 6/26 | 11 | 24 |

(movie_009 gate cấp phim tắt — đúng thiết kế.)

## Thứ tự thí nghiệm

### V2a — chỉ cần một đầu cạnh (làm ngay, đo bằng arm mới `speaker_any`)

`render_speaker_registry_context()`: cạnh thân tộc được render khi ≥1 đầu là
speaker có tên trong scene. Guardrail giữ nguyên: gate cấp phim ≥2 cặp kinship
high, chỉ edge thân tộc, alias mơ hồ bị bỏ. Thêm mới: scene phải có ≥2 dòng
thoại; cặp đủ hai đầu xếp trước cặp một đầu khi vượt `max_pairs`.

Rủi ro có chủ đích: nới là tiến gần lại V0 (vốn hại significant). Khác biệt
then chốt so với V0: chỉ scene có người trong cạnh **thực sự cầm mic** mới
thấy quan hệ đó — V0 tiêm vào mọi scene kể cả khi cả hai vắng mặt.

Đo: thêm arm `speaker_any` vào cùng đợt v1b (pipeline tất định → tái dùng
baseline/registry/speaker của v1b, chỉ tốn 5 lượt refine GPU). So 4 arm bằng
`compare_arms.py`. Nếu `speaker_any` thua `speaker` → rút, cơ chế không scale
bằng cách nới; chuyển trọng tâm sang V2b/V2c.

**Kết quả (2026-08-16, docs/eval/speaker_ab_v2a.json): RÚT.** Trung bình delta
theo phim: speaker_any +0,0005 so với speaker +0,0180 (pooled F1 0,4788 vs
0,4962). Nguyên nhân nằm đúng chỗ dự đoán rủi ro: movie_045 — phim có tên
cluster sai nhiều nhất — sập từ −0,0184 xuống −0,0822 (p=0,000) khi số scene
kích hoạt tăng 0→19; movie_015 cũng giảm (+0,0379 → +0,0236) khi 4→19 scene.
Ngay cả movie_046 (tên tốt nhất bộ) cũng nhích xuống (+0,0657 → +0,0552).
Kết luận: độ phủ kích hoạt KHÔNG phải ràng buộc chính — độ đúng của tên mới
là; nới điều kiện chỉ khuếch đại tên sai. Xác nhận nguyên tắc "tag sai còn tệ
hơn không tag" ở cấp cơ chế. Trọng tâm chuyển sang V2b/V2c như luật đã định.

### V2b — hợp nhất danh tính nhân vật (logic thuần làm trước, GPU sau)

Registry sinh "Sheldon"/"Shelly"/"Moonpie"/"Adult Sheldon" thành 4 nhân vật;
cluster_map chọn tên khác với đầu cạnh thân tộc → không bao giờ khớp.
`CHARACTER/unify_cast.py`: LLM gộp alias trên **danh sách tên** (prompt ngắn,
xa dưới ngưỡng 1000 token), validate cứng: chỉ tên có sẵn, một tên một nhóm,
nhóm trộn giới tính rõ ràng bị bác. Hợp nhất bằng union-find, relation remap
id, giữ confidence cao nhất, drop self-loop sau gộp.

**Kết quả (2026-08-16): cơ học đúng, hiệu quả bằng 0 trên bộ eval — RÚT,
không đo arm GPU.** Hai vòng chạy trên 5 phim:

- Vòng 1 (chỉ danh sách tên): LLM quá dè dặt, chỉ gộp cặp hiển nhiên về mặt
  chuỗi (Dr. Sturgis / Dr. John Sturgis), bỏ sót Meemaw/Connie dù ví dụ nằm
  ngay trong prompt. Kích hoạt scene: y hệt registry gốc (0/32, 0/32, 0/35,
  0/32, 6/26).
- Vòng 2 (kèm 2 dòng thoại evidence mỗi nhân vật, lưu raw ra
  `unify.raw.txt`): đề xuất tốt hẳn — movie_045 đề xuất đúng
  Sheldon/Shelly/Moonpie nhưng bị gate giới tính bác (registry gán Shelly
  là nữ — nhãn sai từ thượng nguồn); cùng gate đó bác đúng nhóm Connie+John
  (LLM gộp nhầm cặp tình nhân thành một người). movie_008 gộp đúng
  Glenn Sturgis+Glenn. movie_009 (cast 32 người, prompt dài nhất) suy biến
  thành chép lại danh sách thay vì JSON — đúng kiểu degradation prompt dài
  của Gemma-3. Kích hoạt scene: vẫn y hệt (0, 0, 0, 0, 6); movie_046 chỉ đổi
  thứ tự/hướng cạnh trong 3 context, cùng nội dung.

Nguyên nhân gốc, đo từng tầng của chuỗi kích hoạt: **cặp có lượt thoại kề
nhau và cạnh kinship-high là hai tập người rời nhau.** Toàn phim: movie_008
18 cặp thoại / 2 cạnh, giao = 0; movie_009 21 cặp / 1 cạnh (dưới gate phim);
movie_015 24 cặp / 2 cạnh (me-Mommy, Big Daddy-Shorty — nhân vật phụ), giao
= 0; movie_045 12 cặp / 3 cạnh (Meemaw-Sheldon, Meemaw-Shelly, Shelly-Dad —
gia đình thật), giao = 0; movie_046 29 cặp / 9 cạnh, giao = 3 (đã kích hoạt
từ V1). Trần lý thuyết: ép gộp tay hoàn hảo cả Sheldon/Shelly/Moonpie lẫn
Meemaw/Connie ở movie_045 vẫn 0 scene kích hoạt — những người có cạnh không
có thoại kề nhau được tag, những cặp thật sự nói chuyện (Connie-Missy,
Meemaw-Dr. Sturgis, Amy-Cheyenne...) không có cạnh kinship-high. Gate giới
tính vì vậy cũng không tốn gì (chi phí = 0), giữ nguyên.

Bài học cộng dồn với V2a: kích hoạt registry theo scene bị chặn bởi (1) độ
đúng của tên cluster và (2) độ phủ cạnh trên đúng những người đang cầm mic —
không phải bởi alias phân mảnh. Giá trị còn lại của V1 (+0,0180) nằm ở tag
speaker, và đòn bẩy có bằng chứng nhất là làm tên tag đúng hơn → V2c.

### V2c — đặt tên cluster cấp mùa

`SPK_id` dùng chung xuyên tập (8–15 cluster trùng giữa mỗi cặp tập trong mùa
11 tập). Gom thoại tiêu biểu + nhãn kịch bản của một cluster qua cả mùa rồi
đặt tên MỘT lần: nhiều bằng chứng hơn, tập nghèo nhãn hưởng nhờ tập giàu,
và sửa được lỗi kiểu "cluster 60 dòng giảng QCD bị gán Missy" (movie_045 —
phim duy nhất còn âm sau khi có tag).

**Kết quả (2026-08-16, docs/eval/speaker_ab_v2c.json): RÚT.** Cài đặt:
`SPEAKER/season_map.py` gom bằng chứng cả mùa (anchor/exclusion/reps trên
chunk gộp, vocative theo từng tập rồi cộng phiếu), đặt tên một lần, rebuild
`speakers.season.json` từ line_tags có sẵn. Luật hybrid tiền đăng ký: chỉ
dùng cho mùa CÓ anchor kịch bản (Young Sheldon 16 anchor; Superstore 0 →
giữ tên per-episode, SRT trùng byte với arm speaker nhờ pipeline tất định).

Đo arm `speaker_season` (gate = trung bình delta theo phim): **+0,0028 so
với speaker +0,0180 — thua cả hai phim đổi tên.** movie_045: −0,0411
(p=0,047), tệ hơn speaker (−0,0184) dù đã sửa đúng cluster giảng bài
(SPK_106 Missy→Sheldon Lee) và cluster dẫn chuyện (SPK_001 Ingram→Adult
Sheldon); movie_046: +0,0123 (p=0,379), sập từ +0,0657 vì mất 147/213 dòng
có tên (66/257) khi anchor xung đột xuyên tập xoá tên hàng loạt.

Tiền đề bị bác bằng dữ liệu: danh tính SPK_id xuyên tập CÓ THẬT nhưng
NHIỄU. Nhất quán: SPK_043=George Sr. 208 chunk, SPK_078=George Jr. 137,
SPK_001=Adult Sheldon 10/11 tập có nhãn. Nhiễu: SPK_025 trộn cặp song sinh
Sheldon/Missy (nhãn Sheldon ở 042/043, Missy ở 045/046/058/060), SPK_027
mang 4 nhãn khác nhau, SPK_036 là phụ huynh ở 045 nhưng Meemaw ở 046.
Superstore 0 anchor toàn mùa → đặt tên mùa ở đó là đoán không dây bảo hiểm
(nhiễm chéo: "Glennda" của movie_008 bị gán vào cluster của movie_015).

Nút thắt thật sự lộ ra sau ba lần đo V2a/b/c: **cluster CAM++ không thuần
ngay TRONG một tập** — SPK_106 trộn chunk của Missy lẫn Sheldon trong chính
movie_045 (nhãn "MISSY:" nằm trong cluster giảng vật lý), SPK_027 của
movie_046 trộn George Jr. với Sheldon. Một cluster không thuần thì đặt tên
kiểu gì cũng sai cho một phần dòng của nó — trần của mọi cách đặt tên.

## Tổng kết V2 (2026-08-16)

Ba thí nghiệm, ba lần rút theo đúng luật tiền đăng ký, mỗi lần thu hẹp
không gian nguyên nhân:

| thí nghiệm | trung bình delta/phim | so với speaker +0,0180 | bài học |
|---|---|---|---|
| V2a speaker_any | +0,0005 | thua | nới kích hoạt chỉ khuếch đại tên sai |
| V2b unify (không cần arm) | trần = 0 trên bộ eval | — | cạnh kinship và người cầm mic là hai tập rời nhau |
| V2c speaker_season | +0,0028 | thua | danh tính cluster xuyên tập nhiễu; cluster không thuần trong tập |

Arm tốt nhất vẫn là `speaker` (V1): +0,0180, còn thiếu +0,012 tới gate
+0,03. Hướng V3 có bằng chứng nhất: làm sạch tag Ở CẤP DÒNG thay vì cấp
cluster — (1) nhãn kịch bản trong chunk phủ quyết tên cluster cho RIÊNG
dòng đó (SPK_106 giữ "Sheldon" cho dòng giảng bài nhưng dòng mang nhãn
"MISSY:" đổi thành Missy); (2) đo lại arm speaker với `--min_score` (align
đã hỗ trợ, chưa từng đo) để bỏ tag các chunk khớp embedding yếu — chính là
các chunk dễ thuộc về cluster trộn.

## Gate

Không đổi: Pronoun F1 trung bình theo phim ≥ baseline + 0,03. V1 đạt +0,0180;
còn thiếu +0,012. Mỗi thí nghiệm đo riêng bằng arm mới trong cùng đợt chạy,
paired bootstrap như cũ.
