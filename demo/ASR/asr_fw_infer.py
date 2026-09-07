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
    # Nup calibration cho VAD. Mac dinh cua faster-whisper (thr 0.5 / sil 2000ms / pad 400ms) la
    # GATING chat tay -> bo sot gap doi (do duoc 07/09: del 7.10 -> 14.39). asr_movie_infer.py cat-ghep
    # kieu WhisperX o thong so rong hon nhieu: thr 0.2, khoang lang 3000ms, dem duoi 1300ms.
    ap.add_argument("--vad_thr", type=float, default=None)
    ap.add_argument("--vad_min_sil_ms", type=int, default=None)
    ap.add_argument("--vad_pad_ms", type=int, default=None)
    # Moc thoi gian muc segment cua Whisper von long leo; bat word_timestamps thi faster-whisper
    # gong lai bien segment theo DTW cross-attention muc tu. Do 07/09: lech |start| trung binh cua
    # cut&merge la 0.48s, cua no-VAD la 1.14s - do lech nay moi la thu an diem chrF/COMET.
    ap.add_argument("--word_ts", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    print(f"[fw] nap model {a.model} ({a.device}/{a.compute_type})...", flush=True)
    model = WhisperModel(a.model, device=a.device, compute_type=a.compute_type)

    vad_params = {k: v for k, v in (("threshold", a.vad_thr),
                                    ("min_silence_duration_ms", a.vad_min_sil_ms),
                                    ("speech_pad_ms", a.vad_pad_ms)) if v is not None} or None
    print(f"[fw] vad={not a.no_vad} params={vad_params} word_ts={a.word_ts}", flush=True)

    # temperature fallback: KHONG truyen -> giu mac dinh (0, 0.2, 0.4, 0.6, 0.8, 1.0) cua faster-whisper
    segments, info = model.transcribe(
        a.audio,
        language=a.language,
        beam_size=a.beam,
        vad_filter=not a.no_vad,
        vad_parameters=vad_params,
        word_timestamps=a.word_ts,
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
