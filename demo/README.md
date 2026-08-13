# Pipeline Tự Động Hóa Dịch Thuật & Tinh Chỉnh Phụ Đề Phim End-to-End

Dự án này tích hợp toàn bộ các mô hình trí tuệ nhân tạo (ASR, Scene Segmentation, VLM, NMT, LLM) vào một tiến trình duy nhất để tạo ra phụ đề tiếng Việt tự nhiên và chính xác dựa trên ngữ cảnh hình ảnh của từng cảnh phim.

---

## 🚀 Sơ đồ Hoạt động (Pipeline Flow)

```
[Video Đầu Vào] 
  │
  ├──► (FFmpeg) ──────► Trích xuất Audio (.wav) ──► (ASR Whisper) ────► Phụ đề Tiếng Anh (.srt)
  │                                                                           │
  │                                                                           ▼
  └──► (Scene Seg) ───► Phân đoạn Cảnh (JSON)                               (NMT MBart)
           │                                                                  │
           ▼ (Cắt video clip nhỏ)                                             ▼
     [Video Scene Clips] ──► (VLM VideoLLama3) ──► Context (JSON) ──► Phụ đề Việt Thô (.srt)
                                                      │                       │
                                                      ▼                       ▼
                                            [ Gemma 3 Tinh chỉnh ] ◄──────────┘
                                                      │
                                                      ▼
                                         [ Phụ đề Việt Tinh chỉnh ]
```

---

## 💻 Hướng dẫn Sử dụng

Chạy toàn bộ pipeline tự động từ đầu đến cuối bằng một câu lệnh duy nhất:

```bash
/data/ndloc_bk/ntVan/demo_env/bin/python3 /data/ndloc_bk/ntVan/demo/run_pipeline.py \
    --video_path "/data/ndloc_bk/ntVan/demo/test.mp4"
```

### 📋 Các tham số dòng lệnh (CLI Arguments)

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
| :--- | :--- | :--- | :--- |
| `--video_path` | `str` | *Bắt buộc* | Đường dẫn tới file video đầu vào cần dịch (`.mp4`, `.mkv`...). |
| `--output_dir` | `str` | `demo/output` | Thư mục lưu toàn bộ kết quả trung gian và sản phẩm phụ đề cuối cùng. |
| `--cache_dir` | `str` | `demo/cache` | Thư mục chứa mô hình cache để chạy offline. |
| `--seed` | `int` | `42` | Khóa trạng thái ngẫu nhiên giúp **đảm bảo kết quả dịch đồng nhất 100%** giữa các lần chạy. |
| `--vlm_fps` | `float` | `0.5` | Tần suất trích xuất khung hình từ video của VideoLLaMA3. |
| `--llm_batch_size` | `int` | `8` | Kích thước batch xử lý phụ đề đồng thời của Gemma 3. |

---

## 📦 Định dạng các File Đầu ra (Outputs)

Toàn bộ các file kết quả sẽ được xuất ra thư mục `--output_dir` (Ví dụ với đầu vào là `test.mp4`):

1.  **`test.wav`**: File âm thanh tách từ video gốc (16kHz Mono).
2.  **`test.(Tiếng Anh).srt`**: Phụ đề tiếng Anh trích xuất từ giọng nói (ASR).
3.  **`test.scenes.json`**: Danh sách mốc thời gian bắt đầu/kết thúc các phân cảnh.
4.  **`test_scenes/`**: Thư mục chứa các clip video cắt nhỏ cho từng phân cảnh.
5.  **`test.captions.json`**: Ngữ cảnh hình ảnh (VLM) của từng phân cảnh (nhân vật, quan hệ, mô tả hành động).
6.  **`test.(Tiếng Việt_dich_tho).srt`**: Bản dịch thô tiếng Việt (NMT).
7.  **`test.(Tiếng Việt_tinh_chinh).srt`**: Phụ đề tiếng Việt hoàn thiện đã được Gemma 3 tinh chỉnh theo ngữ cảnh.
8.  **`test.(Tiếng Việt_tinh_chinh).srt.json`**: File log JSON chi tiết đối chiếu dòng dịch để phục vụ kiểm thử.
