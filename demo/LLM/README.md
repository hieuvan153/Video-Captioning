# Hướng dẫn Tinh chỉnh Phụ đề bằng Gemma-12B và Ngữ cảnh Cảnh phim

Thư mục này chứa script chạy mô hình **Gemma-12B** (`thevan2404/best_gemma_scene_context`) để tinh chỉnh xưng hô, văn phong cho phụ đề tiếng Việt từ bản dịch thô dựa trên ngữ cảnh hình ảnh của từng cảnh phim.

---

## 📂 Các file trong thư mục
* [refine_llm.py](file:///data/ndloc_bk/ntVan/demo/LLM/refine_llm.py): Script Python chính nạp mô hình Gemma, phân bổ phụ đề theo cảnh phim và chạy tinh chỉnh tuần tự.
* [README.md](file:///data/ndloc_bk/ntVan/demo/LLM/README.md): File hướng dẫn này.

---

## ⚙️ Cấu hình Môi trường chạy
Script yêu cầu sử dụng môi trường python chuyên biệt hỗ trợ PyTorch, Unsloth và HuggingFace:
* **Đường dẫn Python Virtualenv**: `/data/ndloc_bk/ntVan/demo_env/bin/python3`

---

## 🚀 Cách chạy Script

Sử dụng lệnh sau để thực hiện tinh chỉnh phụ đề:

```bash
/data/ndloc_bk/ntVan/demo_env/bin/python3 /data/ndloc_bk/ntVan/demo/LLM/refine_llm.py \
    --en_srt "/data/ndloc_bk/ntVan/ASR/output_en_test/S03E015_Đồn_98.(Tiếng Anh).srt" \
    --vinai_srt "/data/ndloc_bk/ntVan/demo/NMT/S03E015_Đồn_98.(Tiếng Việt_dich_tho).srt" \
    --vlm_json "/data/ndloc_bk/ntVan/demo/VLM/result_captions.json" \
    --output_srt "/data/ndloc_bk/ntVan/demo/LLM/S03E015_Đồn_98.(Tiếng Việt_tinh_chinh).srt"
```

### 📋 Các tham số dòng lệnh (Arguments)

| Tham số | Kiểu | Mặc định | Mô tả |
| :--- | :--- | :--- | :--- |
| `--en_srt` | `str` | *Bắt buộc* | Đường dẫn file phụ đề `.srt` tiếng Anh gốc. |
| `--vinai_srt` | `str` | *Bắt buộc* | Đường dẫn file phụ đề `.srt` tiếng Việt dịch thô (bởi VinAI MBart). |
| `--vlm_json` | `str` | *Bắt buộc* | Đường dẫn file JSON chứa thông tin các cảnh phim (timestamps, captions) được trích xuất từ VideoLLama3. |
| `--output_srt` | `str` | *Bắt buộc* | Đường dẫn lưu file phụ đề `.srt` tiếng Việt đã tinh chỉnh. |
| `--adapter_model_name` | `str` | `thevan2404/best_gemma_scene_context` | Tên/Đường dẫn LoRA adapter mô hình trên HuggingFace Hub. |
| `--cache_dir` | `str` | `/data/ndloc_bk/ntVan/hf_cache` | Thư mục lưu cache mô hình. |
| `--max_seq_length` | `int` | `2048` | Độ dài sequence tối đa. |
| `--max_new_tokens` | `int` | `1024` | Số lượng token tối đa sinh ra cho mỗi phân cảnh. |
| `--llm_batch_size` | `int` | `20` | Kích thước batch chạy mô hình LLM (hiện tại chạy tuần tự). |


