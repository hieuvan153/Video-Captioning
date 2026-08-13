# Hướng dẫn Chạy ASR (Whisper) Trích xuất Phụ đề Tiếng Anh

Thư mục này chứa script chạy mô hình **Whisper Medium** đã được tối ưu hóa sâu để trích xuất phụ đề tiếng Anh (`.srt`) từ file âm thanh phim (`.wav`) với tốc độ nhanh nhất và hiệu năng ổn định trên máy chủ GPU dùng chung.

---

## 📂 Các file trong thư mục
* [asr_movie_infer.py](file:///data/ndloc_bk/ntVan/demo/ASR/asr_movie_infer.py): Script Python chính chạy tách thoại (VAD) và nhận dạng giọng nói bằng Whisper.
* [README.md](file:///data/ndloc_bk/ntVan/demo/ASR/README.md): Hướng dẫn sử dụng chi tiết (file này).

---

## ⚙️ Cấu hình Môi trường chạy
Script yêu cầu sử dụng môi trường python chuyên biệt hỗ trợ PyTorch, Whisper, SoundFile và ffmpeg:
* **Đường dẫn Python Virtualenv**: `/data/ndloc_bk/ntVan/demo_env/bin/python3`

---

## 🚀 Hướng dẫn Sử dụng

Chạy trích xuất phụ đề từ một file âm thanh bằng lệnh sau:

```bash
/data/ndloc_bk/ntVan/demo_env/bin/python3 /data/ndloc_bk/ntVan/demo/ASR/asr_movie_infer.py \
    --audio_path "/data/ndloc_bk/ntVan/data/Movie/audio/S04E019_Thưa_tòa.wav" \
    --out_dir "/data/ndloc_bk/ntVan/demo/ASR/output" \
    --out_name "S04E019_Thưa_tòa"
```

### 📋 Các tham số dòng lệnh (Arguments)

| Tham số | Kiểu | Mặc định | Mô tả |
| :--- | :--- | :--- | :--- |
| `--audio_path` | `str` | *Bắt buộc* | Đường dẫn file âm thanh đầu vào dạng `.wav` hoặc `.mp4`. |
| `--out_dir` | `str` | `output` | Thư mục lưu kết quả file phụ đề `.srt` đầu ra. |
| `--out_name` | `str` | `None` | Tên của file đầu ra (không bao gồm phần mở rộng). Nếu không nhập, mặc định lấy tên file audio. |


