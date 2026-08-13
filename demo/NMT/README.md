# Hướng dẫn chạy Dịch Thô Phụ Đề bằng VinAI MBart

Thư mục này chứa script chạy mô hình dịch **VinAI MBart** (`vinai-translate-en2vi-v2`) để dịch thô toàn bộ file phụ đề `.srt` từ tiếng Anh sang tiếng Việt.

---

## 📂 Các file trong thư mục
* [run_nmt.py](file:///data/ndloc_bk/ntVan/demo/NMT/run_nmt.py): Script Python chính nạp mô hình dịch và dịch file phụ đề.
* [README.md](file:///data/ndloc_bk/ntVan/demo/NMT/README.md): File hướng dẫn này.

---

## ⚙️ Cấu hình Môi trường chạy
Script yêu cầu sử dụng môi trường python chuyên biệt đã cài đặt đầy đủ PyTorch, Transformers và các gói xử lý phụ đề:
* **Đường dẫn Python Virtualenv**: `/data/ndloc_bk/ntVan/demo_env/bin/python3`

---

## 🚀 Cách chạy Script

Sử dụng lệnh sau để chạy dịch thô:

```bash
/data/ndloc_bk/ntVan/demo_env/bin/python3 /data/ndloc_bk/ntVan/demo/NMT/run_nmt.py \
    --input_srt /ĐƯỜNG_DẪN/file_tieng_anh.srt \
    --output_srt /ĐƯỜNG_DẪN/file_tieng_viet_dich_tho.srt \
    --batch_size 64
```

### 📋 Các tham số dòng lệnh (Arguments)

| Tham số | Kiểu | Mặc định | Mô tả |
| :--- | :--- | :--- | :--- |
| `--input_srt` | `str` | *Bắt buộc* | Đường dẫn file phụ đề `.srt` tiếng Anh đầu vào. |
| `--output_srt` | `str` | *Bắt buộc* | Đường dẫn lưu file phụ đề `.srt` tiếng Việt đầu ra. |
| `--model_path` | `str` | Tự động dò tìm | Đường dẫn thư mục chứa model MBart cục bộ (dò tìm `/data/ndloc_bk/ntVan/infer/model/mbart_model` hoặc `/data/ndloc_bk/app/model/mbart_model`), nếu không thấy sẽ tự tải từ HF: `vinai/vinai-translate-en2vi-v2`. |
| `--cache_dir` | `str` | `/data/ndloc_bk/ntVan/hf_cache` | Thư mục cache lưu trữ các mô hình HuggingFace. |
| `--batch_size` | `int` | `64` | Kích thước batch khi thực hiện dịch qua model. |
| `--device` | `str` | cuda (nếu có) | Thiết bị phần cứng để chạy dịch (`cuda` hoặc `cpu`). |


