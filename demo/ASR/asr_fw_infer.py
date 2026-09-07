"""ASR bang faster-whisper large-v3 (ct2, fp16) + VAD tich hop + temperature fallback mac dinh.

Khac voi demo/ASR/asr_movie_infer.py (whisper-medium fine-tune, temperature=0.0 co dinh, greedy):
day la model tong quat large-v3, beam 5, va GIU nguyen chuoi temperature fallback mac dinh cua
faster-whisper (0.0 -> 1.0) de chong ao giac lap chunk (Radford et al. 2022, muc 4.5).

CLI: van_env/bin/python demo/ASR/asr_fw_infer.py --audio a.wav --out_srt a.en.srt [--model large-v3] [--beam 5]
"""
from __future__ import annotations

import argparse
import datetime
import os
import time

import srt
from faster_whisper import WhisperModel


FW_LARGE_V3_LOCAL = ("/data/ndloc_bk/hf_cache/models--Systran--faster-whisper-large-v3/"
                     "snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_srt", required=True)
    # Mac dinh: snapshot ct2 CUC BO (/data/ndloc_bk/hf_cache). Khong di qua HF_HOME nen khong
    # can doi HF_HOME (doi se lam an GemmaX2 + COMET-DA o cache mac dinh). Van ghi de duoc
    # bang ten repo, vi du --model large-v3.
    ap.add_argument("--model", default=FW_LARGE_V3_LOCAL)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--language", default="en")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute_type", default="float16")
    ap.add_argument("--no_vad", action="store_true", help="tat VAD Silero (mac dinh van bat - giu nguyen hanh vi 1a)")
    a = ap.parse_args()

    t0 = time.time()
    print(f"[fw] nap model {a.model} ({a.device}/{a.compute_type})...", flush=True)
    model = WhisperModel(a.model, device=a.device, compute_type=a.compute_type)

    # temperature fallback: KHONG truyen -> giu mac dinh (0, 0.2, 0.4, 0.6, 0.8, 1.0) cua faster-whisper
    segments, info = model.transcribe(
        a.audio,
        language=a.language,
        beam_size=a.beam,
        vad_filter=not a.no_vad,
        condition_on_previous_text=False,
    )
    print(f"[fw] audio {info.duration:.1f}s, bat dau giai ma...", flush=True)

    subs: list[srt.Subtitle] = []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        start = float(s.start)
        end = max(float(s.end), start + 0.01)   # tranh cue rong 0 giay -> film_ref_anchored bo qua
        subs.append(srt.Subtitle(index=len(subs) + 1,
                                 start=datetime.timedelta(seconds=start),
                                 end=datetime.timedelta(seconds=end),
                                 content=text))
        if len(subs) % 100 == 0:
            print(f"[fw] {len(subs)} cue, toi {end:.0f}s / {info.duration:.0f}s", flush=True)

    out_dir = os.path.dirname(os.path.abspath(a.out_srt))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(a.out_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
        f.flush()
        os.fsync(f.fileno())

    print(f"XONG: {len(subs)} cue | audio {info.duration:.1f}s | chay {time.time() - t0:.1f}s -> {a.out_srt}",
          flush=True)


if __name__ == "__main__":
    main()
