# ASR: Whisper + VAD → SRT tiếng Anh

`asr_movie_infer.py` nạp `demo/model/ASR/whisper-medium-13-openai.pt` ngay khi import.

```bash
/data/ndloc_bk/ntVan/demo_env/bin/python3 demo/ASR/asr_movie_infer.py \
    --audio_path phim.wav --out_dir demo/output --out_name phim
```

| Tham số | Mặc định |
| :--- | :--- |
| `--audio_path` | bắt buộc |
| `--out_dir` | `output` |
| `--out_name` | tên file audio |

Cách chạy:
1. Silero VAD (ngưỡng 0,2), đệm 0,2 s đầu / 1,3 s cuối; lặng > 3 s thì tách chunk.
2. Mỗi chunk (các đoạn có tiếng đã ghép) qua `whisper.transcribe`: greedy T=0, `word_timestamps=True`,
   `hallucination_silence_threshold=None`. Biến môi trường `ASR_BEAM`, `ASR_TEMPS` chỉ dùng cho arm thí nghiệm.
3. Bỏ segment có `avg_logprob < -1`, `no_speech_prob > 0.9` hoặc `compression_ratio > 6`; bỏ cue chỉ gồm tiếng đệm.
4. Đổi mốc từ audio đã ghép về trục thời gian gốc rồi ghi SRT.

Lưu ý đã biết:
- Ngoài SRT còn ghi `<tên>.(Tiếng Anh).segment_info.json` (mọi segment trước khi lọc, kèm `seek`) cạnh SRT.
- `whisper.load_model(<file .pt>)` không đặt `alignment_heads`, nên mốc từ dùng mọi head nửa trên decoder thay cho
  bộ head của medium.en. Mốc từ quyết định điểm seek và lưới cue; muốn đổi phải đo lại trên GPU.
- Trước 14/09 dùng `hallucination_silence_threshold=2.0`: Whisper nhảy trọn 30 s khi từ cuối nằm trong 2 s cuối cửa sổ và bỏ
  câu đang dở. Đổi sang `None` trên Ode to Joy: WER 16,61 → 15,92, từ mất ở đường nối 226 → 156, BLEU Bảng 4.7 +0,26.

Logic thuần (vùng VAD, ánh xạ thời gian, biến môi trường) nằm ở `asr_post.py`, test: `python demo/ASR/test_asr_post.py`.

`asr_fw_infer.py` là arm faster-whisper đã loại (VAD và bộ lọc khác, không so trực tiếp với file chính được).
