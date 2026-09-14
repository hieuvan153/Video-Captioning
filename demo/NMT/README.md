# NMT: mBART (VinAI en2vi) → SRT tiếng Việt thô

```bash
/data/ndloc_bk/ntVan/demo_env/bin/python3 demo/NMT/run_nmt.py \
    --input_srt "phim.(Tiếng Anh).srt" --output_srt "phim.(Tiếng Việt_dich_tho).srt" \
    --num_beams 5 --length_penalty 4.0        # = cấu hình của run_pipeline.py
```

| Tham số | Mặc định | Ghi chú |
| :--- | :--- | :--- |
| `--input_srt`, `--output_srt` | bắt buộc | |
| `--model_path` | `demo/model/NMT/mbart_model` nếu có, không thì `vinai/vinai-translate-en2vi-v2` | |
| `--cache_dir` | `/data/ndloc_bk/ntVan/hf_cache` | |
| `--batch_size` | `64` | |
| `--num_beams` | `1` | pipeline dùng 5 |
| `--length_penalty` | `1.0` | pipeline dùng 4.0 (qua cổng G2) |
| `--mbr`, `--mbr_top_p`, `--seed` | `0`, `0.9`, `0` | MBR thí nghiệm, mặc định tắt |
| `--device` | `cuda` nếu có | |

Mỗi cue được tách câu (`nltk.sent_tokenize`), dịch từng câu rồi nối lại. `early_stopping=True` là bắt buộc
khi `length_penalty > 1` (bỏ đi thì beam lặp chữ).

Kiểm tra không cần GPU: `python demo/NMT/test_run_nmt_lp.py`. `run_gemmax2.py` là arm GemmaX2 để so sánh.
