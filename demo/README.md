# Pipeline phụ đề phim Anh → Việt

Chạy từ thư mục gốc repo (`NLHV/ntVan`) bằng môi trường `/data/ndloc_bk/ntVan/demo_env/bin/python3`.

## Hai kiến trúc (`--arch`)

```
v2 (mặc định, ~14 phút cho phim 94 phút)
  video ─ffmpeg─> wav ─ASR (Whisper + VAD)─> SRT Anh ─NMT (mBART, beam 5, lp 4.0)─> SRT Việt thô

legacy (~118 phút): v2 + 3 tầng
  video ─scene_seg─> cảnh ─VLM─> caption ┐
  SRT Anh + SRT Việt thô ─────────────────┴─Gemma-3 12B LoRA (khóa độ dài 0,90)─> SRT Việt tinh chỉnh
```

Bản giao `pm0.90` trong Bảng 4.7 (Ode to Joy: BLEU 37,90, PronF1 0,830) là đầu ra kiểu legacy;
v2 cho 36,00. Chi tiết số đo: `docs/BANG_47_ODE_TO_JOY_2026-09-09.md`.

```bash
PY=/data/ndloc_bk/ntVan/demo_env/bin/python3
$PY demo/run_pipeline.py --video_path phim.mkv                                    # v2
HF_TOKEN=... $PY demo/run_pipeline.py --video_path phim.mkv --arch legacy          # legacy
```

| Tham số | Mặc định | Ghi chú |
| :--- | :--- | :--- |
| `--video_path` | bắt buộc | |
| `--output_dir` | `demo/output` | mọi file trung gian; bước nào đã có file đầu ra thì bỏ qua |
| `--cache_dir` | `demo/cache` | |
| `--seed` | `42` | |
| `--arch` | `v2` | `v2` hoặc `legacy` |
| `--vlm_fps` | `1` | chỉ legacy |
| `--llm_batch_size` | `1` | chỉ legacy; >1 làm lệch dòng, không tăng |

## File đầu ra (ví dụ `phim.mkv`)

| File | Tầng |
| :--- | :--- |
| `phim.wav` | ffmpeg, 16 kHz mono |
| `phim.(Tiếng Anh).srt` | ASR |
| `phim.(Tiếng Việt_dich_tho).srt` | NMT — **đầu ra cuối của v2** |
| `phim.scenes.json`, `phim_scenes/` | scene_seg (legacy) |
| `phim.captions.json` | VLM (legacy) |
| `phim.(Tiếng Việt_tinh_chinh).srt` (+ `.json` debug) | Gemma (legacy) — **đầu ra cuối của legacy** |

Mỗi module chạy riêng được; xem README trong `ASR/`, `NMT/`, `LLM/`, `VLM/`, `scene_seg/`.
Kiểm tra định tuyến không cần GPU: `python demo/test_pipeline_arch.py`.
