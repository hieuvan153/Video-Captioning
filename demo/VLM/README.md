# VLM: VideoLLaMA3-7B mô tả từng cảnh (chỉ kiến trúc legacy)

```bash
/data/ndloc_bk/ntVan/demo_env/bin/python3 demo/VLM/run_vlm.py \
    --input_dir demo/output/phim_scenes --output_file demo/output/phim.captions.json
```

| Tham số | Mặc định |
| :--- | :--- |
| `--input_dir` | `demo/output/test_scenes` |
| `--output_file` | `demo/output/test.captions.json` |
| `--model_path` | `DAMO-NLP-SG/VideoLLaMA3-7B` |
| `--cache_dir` | `demo/cache` |
| `--device` | `cuda:0` |
| `--min_duration` | `3.0` (cảnh ngắn hơn: caption rỗng) |
| `--max_frames` | `240` |
| `--fps` | `1.0` |

Cần `flash_attn`. Script tự vá vài file mã nguồn của VideoLLaMA3 trong cache cho khớp bản transformers đang cài.
Khi chạy trong pipeline, mốc `start_time`/`end_time` của mỗi cảnh lấy từ `scenes.json` của scene_seg.
Lỗi khi sinh caption được ghi thành chuỗi `Error during analysis: ...` và sẽ đi vào prompt của tầng LLM.
