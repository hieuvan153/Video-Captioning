# LLM: Gemma-3 12B LoRA tinh chỉnh bản dịch thô

Cần biến môi trường `HF_TOKEN` (đăng nhập Hugging Face ngay khi import) và GPU.

```bash
HF_TOKEN=... /data/ndloc_bk/ntVan/demo_env/bin/python3 demo/LLM/refine_llm.py \
    --en_srt "phim.(Tiếng Anh).srt" --vinai_srt "phim.(Tiếng Việt_dich_tho).srt" \
    --vlm_json phim.captions.json --output_srt "phim.(Tiếng Việt_tinh_chinh).srt" --llm_batch_size 1
```

| Tham số | Mặc định | Ghi chú |
| :--- | :--- | :--- |
| `--en_srt`, `--vinai_srt`, `--output_srt` | bắt buộc | hai SRT vào phải cùng số cue |
| `--vlm_json` | không | có: chia chunk theo cảnh VLM; không: theo khoảng lặng |
| `--chunk_gap_s`, `--chunk_target` | `2.0`, `20` | chỉ khi không có `--vlm_json` |
| `--adapter_model_name` | `thevan2404/best_gemma_scene_context` | |
| `--system_prompt` | không | cho adapter train không có khối `<Scene Context>` (v7) |
| `--cache_dir` | `demo/cache` | |
| `--max_seq_length`, `--max_new_tokens` | `2048`, `1024` | |
| `--llm_batch_size` | `1` | >1 làm lệch dòng, không tăng |

Sau khi sinh:
1. Gióng dòng LLM vào cue bằng quy hoạch động (`align_lines`, ngưỡng `ALIGN_MIN_RATIO = 0.20`), không bao giờ theo vị trí.
2. Khóa độ dài (`lock_len`, `LEN_LOCK_RATIO = 0.90`): dòng tinh chỉnh ngắn hơn 0,9 × bản thô (theo số từ)
   thì chỉ lấy phần sửa đại từ.
3. Cue không gióng được giữ bản thô. SRT dùng lưới cue của SRT tiếng Anh; file `.json` debug giữ dòng đã gióng, chưa khóa.
