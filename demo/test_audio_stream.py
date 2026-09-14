"""Chon luong audio tieng Anh khi tach audio (14/09). Chay: van_env/bin/python demo/test_audio_stream.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_pipeline as rp  # noqa: E402

V = {"codec_type": "video"}
A = lambda lang=None, title=None: {"codec_type": "audio", "tags": {k: v for k, v in (("language", lang), ("title", title)) if v}}

# Ode to Joy: luong EN co nhan va dung truoc -> giu nguyen hanh vi (luong audio 0)
assert rp.english_audio_index([V, A("en", "English"), A(title="VI-ai-nu-2")]) == 0
# Luong long tieng dung truoc: ffmpeg mac dinh lay luong dau -> sai ngon ngu ma khong bao loi
assert rp.english_audio_index([V, A("vie"), A("eng")]) == 1
assert rp.english_audio_index([A(title="Vietnamese"), A(title="English")]) == 1
# Khong co nhan nao: giu luong dau
assert rp.english_audio_index([A(), A()]) == 0
print("OK: chon dung luong audio tieng Anh")
