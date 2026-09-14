"""Logic thuan cua asr_movie_infer.py (khong nap Whisper) de test duoc: vung VAD, anh xa thoi gian, bien moi truong."""
import copy
import os

VAD_SR = 16000
CHUNK_GAP_S = 3.0
PAD_HEAD, PAD_TAIL = 3200, 20800  # 0,2 s / 1,3 s (don vi mau)


def group_regions(t, n_samples, chunk_gap_s=CHUNK_GAP_S, sr=VAD_SR):
    """Dem dau/duoi cho vung tieng noi VAD (chi so mau), bo chong lan, tach chunk khi lang > chunk_gap_s.
    Sua t tai cho va tra ve list chunk (moi chunk la list vung, van tinh bang MAU de ghep audio)."""
    for i in range(len(t)):
        t[i]["start"] = max(0, t[i]["start"] - PAD_HEAD)
        t[i]["end"] = min(n_samples - 16, t[i]["end"] + PAD_TAIL)
        if i > 0 and t[i]["start"] < t[i - 1]["end"]:
            t[i]["start"] = t[i - 1]["end"]
    u = [[]]
    for i in range(len(t)):
        if i > 0 and t[i]["start"] > t[i - 1]["end"] + chunk_gap_s * sr:
            u.append([])
        u[-1].append(t[i])
    return u


def annotate_seconds(u, sr=VAD_SR):
    """Doi vung sang giay va gan chunk_start/chunk_end (vi tri trong audio da ghep) + offset (cong vao de ve truc goc)."""
    for g in u:
        time_sec, offset = 0.0, 0.0
        for j, r in enumerate(g):
            r["start"] /= sr
            r["end"] /= sr
            r["chunk_start"] = time_sec
            time_sec += r["end"] - r["start"]
            r["chunk_end"] = time_sec
            offset += r["start"] if j == 0 else r["start"] - g[j - 1]["end"]
            r["offset"] = offset
    return u


def build_chunks(t, n_samples):
    return annotate_seconds(group_regions(copy.deepcopy(t), n_samples))


def map_segment(g, rs, re_):
    """Mốc segment (giay trong audio da ghep cua chunk g) -> giay tren truc goc.
    Mốc bat dau dung bang chunk_end cua mot vung thuoc ve vung KE TIEP (loi noi bat dau o do)."""
    start = rs + g[0]["offset"]
    for j, r in enumerate(g):
        if r["chunk_start"] <= rs and (rs < r["chunk_end"] or (j == len(g) - 1 and rs <= r["chunk_end"])):
            start = rs + r["offset"]
            break
    end = g[-1]["end"] + 0.5
    for r in g:
        if r["chunk_start"] <= re_ <= r["chunk_end"]:
            end = re_ + r["offset"]
            break
    return start, end


def decode_env(env):
    """ASR_TEMPS / ASR_BEAM cho arm thi nghiem; rong hoac 0 = mac dinh greedy T=0."""
    temps = tuple(float(x) for x in env.get("ASR_TEMPS", "").split(",") if x.strip()) or (0.0,)
    beam = int(env["ASR_BEAM"]) if env.get("ASR_BEAM", "").strip() else None
    return temps, (beam if beam and beam > 0 else None)


def segment_info_path(out_path):
    return os.path.splitext(out_path)[0] + ".segment_info.json"
