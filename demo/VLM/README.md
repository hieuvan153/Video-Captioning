# Hướng dẫn chạy VideoLLama3 phân tích ngữ cảnh Scene

Thư mục này chứa script chạy mô hình **VideoLLama3** (phiên bản 7B) để phân tích ngữ cảnh, nhân vật và mối quan hệ cho từng scene video của bộ phim.

---

## 📂 Các file trong thư mục
* [run_vlm.py](file:///data/ndloc_bk/ntVan/demo/VLM/run_vlm.py): Script Python chính để load mô hình và xử lý các video scene hàng loạt.
* [README.md](file:///data/ndloc_bk/ntVan/demo/VLM/README.md): File tài liệu hướng dẫn này.

---

## ⚙️ Cấu hình Môi trường chạy
Script yêu cầu sử dụng môi trường python chuyên biệt đã được cài đặt đầy đủ CUDA, PyTorch, và `flash_attn`:
* **Đường dẫn Python Virtualenv**: `/data/ndloc_bk/ntVan/demo_env/bin/python3`

---

## 🚀 Cách chạy Script

Sử dụng lệnh sau để khởi chạy script:

```bash
/data/ndloc_bk/ntVan/demo_env/bin/python3 /data/ndloc_bk/ntVan/demo/VLM/run_vlm.py \
    --input_dir /data/ndloc_bk/ntVan/demo/scene_seg/output/result_scenes \
    --output_file /data/ndloc_bk/ntVan/demo/VLM/result_captions.json \
    --min_duration 3.0
```

### 📋 Các tham số dòng lệnh (Arguments)

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
| :--- | :--- | :--- | :--- |
| `--input_dir` | `str` | `/data/ndloc_bk/ntVan/demo/scene_seg/output/result_scenes` | Thư mục chứa các file video scene (`.mp4`, `.avi`, v.v.) cần phân tích. |
| `--output_file` | `str` | `/data/ndloc_bk/ntVan/demo/VLM/result_captions.json` | Đường dẫn file `.json` đầu ra để lưu kết quả. |
| `--min_duration` | `float` | `3.0` | Thời lượng tối thiểu (giây) của scene để phân tích. Các scene ngắn hơn sẽ bị bỏ qua (đầu ra rỗng). |
| `--model_path` | `str` | `DAMO-NLP-SG/VideoLLaMA3-7B` | Tên/đường dẫn của mô hình VideoLLama3 trên HuggingFace. |
| `--cache_dir` | `str` | `/data/ndloc_bk/ntVan/hf_cache` | Thư mục cache tải và lưu mô hình HuggingFace. |
| `--device` | `str` | `cuda:0` | GPU chạy inference (ví dụ: `cuda:0`, `cuda:1`, v.v.). |
| `--max_frames` | `int` | `240` | Số lượng frame tối đa trích xuất từ mỗi scene truyền vào mô hình. |


