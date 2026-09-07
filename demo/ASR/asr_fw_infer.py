"""ASR bang faster-whisper large-v3 (ct2, fp16) + VAD tich hop + temperature fallback mac dinh.

Khac voi demo/ASR/asr_movie_infer.py (whisper-medium fine-tune, temperature=0.0 co dinh, greedy):
day la model tong quat large-v3, beam 5, va GIU nguyen chuoi temperature fallback mac dinh cua
faster-whisper (0.0 -> 1.0) de chong ao giac lap chunk (Radford et al. 2022, muc 4.5).

CLI: van_env/bin/python demo/ASR/asr_fw_infer.py --audio a.wav --out_srt a.en.srt [--model large-v3] [--beam 5]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time

import srt
from faster_whisper import WhisperModel


FW_LARGE_V3_LOCAL = ("/data/ndloc_bk/hf_cache/models--Systran--faster-whisper-large-v3/"
                     "snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478")


def split_words(words: list[dict], gap: float | None, max_s: float | None,
                min_s: float | None = 1.0) -> list[list[dict]]:
    """Gom moc tu thanh cum: ngat truoc mot tu neu khoang lang truoc no >= gap, hoac cum se dai qua max_s.

    min_s la chan an toan cho NGUOI XEM, khong phai cho thuoc do. Thuoc neo-theo-cue thuong cho viec
    cat nho vo han (do duoc: cat doi moi cue van chrF 100,00), nen toi uu theo rieng no se ra phu de
    nhay lien tuc khong ai doc kip. Cum ngan hon min_s duoc gop nguoc vao cum truoc.
    """
    out: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur and ((gap is not None and w["s"] - cur[-1]["e"] >= gap)
                    or (max_s is not None and w["e"] - cur[0]["s"] > max_s)):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    if min_s is None:
        return out
    keep: list[list[dict]] = []
    for g in out:
        if keep and g[-1]["e"] - g[0]["s"] < min_s:
            keep[-1].extend(g)          # qua ngan de doc kip -> gop nguoc vao cum truoc
        else:
            keep.append(g)
    return keep


def selftest() -> None:
    W = lambda s, e, t: {"s": s, "e": e, "w": t}
    ws = [W(0, .5, "a"), W(.6, 1., "b"), W(2., 2.4, "c")]        # khoang lang 1.0s truoc "c"
    assert [len(g) for g in split_words(ws, 0.4, None, None)] == [2, 1]
    assert [len(g) for g in split_words(ws, None, None, None)] == [3]  # khong nguong -> mot cum
    assert [len(g) for g in split_words(ws, None, 1.0, None)] == [2, 1]  # tran 1.0s cat truoc "c"
    assert split_words([], 0.4, 3.0, None) == []
    assert [len(g) for g in split_words(ws, 5.0, 99, None)] == [3]   # nguong qua rong -> khong cat
    # chan doc duoc: cum "c" chi dai 0.4s < 1.0s -> phai gop nguoc, khong duoc de rieng
    assert [len(g) for g in split_words(ws, 0.4, None, 1.0)] == [3]
    assert [len(g) for g in split_words(ws, 0.4, None, 0.3)] == [2, 1]  # ha chan thi lai tach
    print("selftest OK")


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
    # Nup calibration cho HINH DANG cue. Do 07/09: 60.8% cue cua Whisper trum >=2 cue tham chieu
    # (ban medium-FT thang chi 39.7%). Thuoc neo-theo-cue dan text cua mot cue hyp vao MOI cue tham
    # chieu no phu -> cue dai bi nhan ban text -> chrF tut. Cat lai theo moc tu chua khoang lang.
    ap.add_argument("--split_gap_s", type=float, default=None, help="tach cue khi khoang lang giua 2 tu >= nguong (can --word_ts)")
    ap.add_argument("--max_cue_s", type=float, default=None, help="tran do dai cue, tach tai tu ke tiep (can --word_ts)")
    # San 1.0s KHONG phai so doan: do tren chinh ban tham chieu nguoi lam cua Ode to Joy, EN co
    # 0.9% cue ngan hon 1.0s (p05 = 1.15s), VI co 7.8% (p05 = 0.96s). Tuc 1.0s dung o dung phan
    # vi 5 cua tay nghe nguoi that. Ha xuong 0.6s thi gan nhu bo chan (ref chi 0.1% duoi 0.6s).
    ap.add_argument("--min_cue_s", type=float, default=1.0, help="san do dai cue cho NGUOI XEM doc kip; manh ngan hon bi gop nguoc")
    ap.add_argument("--words_json", default=None, help="ghi kem moc tu ra JSON de cat lai offline, khoi chay lai ASR")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

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
    all_words: list[dict] = []
    for si, s in enumerate(segments):
        # "seg" = so hieu segment cua Whisper. BAT BUOC phai luu: do 07/09, xay lai cue tu danh sach
        # moc tu PHANG (vut bien segment) cho ra cue THO hon ban goc (1440 vs 1939 cue, chrF 67.59 vs
        # 73.24). Bo tach cua Whisper min hon luat "lang >= gap"; phep dung la CHIA THEM trong tung
        # segment, khong bao gio gop qua bien. Vong lap nay da lam dung; words_json phai giu duoc dau vet.
        words = [{"s": float(w.start), "e": float(w.end), "w": w.word, "seg": si} for w in (s.words or ())] \
            if a.word_ts and getattr(s, "words", None) else []
        all_words.extend(words)
        pieces = ([(g[0]["s"], g[-1]["e"], "".join(w["w"] for w in g).strip())
                   for g in split_words(words, a.split_gap_s, a.max_cue_s, a.min_cue_s)]
                  if words and (a.split_gap_s or a.max_cue_s)
                  else [(float(s.start), float(s.end), (s.text or "").strip())])
        for start, end, text in pieces:
            if not text:
                continue
            end = max(end, start + 0.01)   # tranh cue rong 0 giay -> film_ref_anchored bo qua
            subs.append(srt.Subtitle(index=len(subs) + 1,
                                     start=datetime.timedelta(seconds=start),
                                     end=datetime.timedelta(seconds=end),
                                     content=text))
            if len(subs) % 100 == 0:
                print(f"[fw] {len(subs)} cue, toi {end:.0f}s / {info.duration:.0f}s", flush=True)

    if a.words_json and all_words:
        os.makedirs(os.path.dirname(os.path.abspath(a.words_json)) or ".", exist_ok=True)
        with open(a.words_json, "w", encoding="utf-8") as f:
            json.dump(all_words, f, ensure_ascii=False)
        print(f"[fw] ghi {len(all_words)} moc tu -> {a.words_json}", flush=True)

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
