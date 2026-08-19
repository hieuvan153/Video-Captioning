"""Override nhan kich ban o CAP DONG: cat chunk tai thoi diem nhan.

Van de do duoc (movie_045/046, 2026-08-17): nhan kich ban ASR giu lai phan
lon nam GIUA chunk — chunk vat qua doi speaker va nhan chi danh dau phan
duoi. "Asked and answered. MISSY: Did you cry when you saw it?" (SPK_106)
nghia la nua sau la Missy, nhung anchor cu lay no dat ten CA cluster 65 chunk
cua Sheldon -> 60 dong SRT sai ten. Nhan/phim: 008/009/015 = 0; 045 = 6
(3 giua chunk); 046 = 12 (8 giua chunk).

Cach lam: cat chunk tai thoi diem nhan (dinh vi bang english_word_timestamps)
thanh vung tag goc + vung tag tong hop "LABEL::Ten", TRUOC khi align. Vung
nhan canh tranh trong chinh may overlap-troi cua align.assign_speakers: dong
vat qua ranh gioi do overlap troi quyet, hoa -> None — khong nhan ban logic
dominance. Bonus: nhan dau text tren chunk UNKNOWN cua CAM++ ("INGRAM:
Sheldon?") truoc day vo dung, gio thanh vung co ten.

Nguyen tac "khong chac thi im lang" giu nguyen: nhan khong dinh vi duoc
(thieu/lech word timestamps, khop >= 2 cho) thi BO QUA nhan do, khong doan.
"""
from __future__ import annotations

import re

from SPEAKER.align import LABEL_TAG_PREFIX
from SPEAKER.name_evidence import iter_script_labels, label_at_text_start

_WORD = re.compile(r"[A-Za-z0-9']+")


def _tokens(text: str) -> list[str]:
    return [w.casefold() for w in _WORD.findall(text)]


def is_label_tag(tag) -> bool:
    return isinstance(tag, str) and tag.startswith(LABEL_TAG_PREFIX)


def label_name(tag: str) -> str:
    return tag[len(LABEL_TAG_PREFIX):]


def _mid_label_time(chunk: dict, match: re.Match) -> float | None:
    """Thoi diem bat dau cua nhan giua text, tra tu english_word_timestamps.

    Word timestamps cua CAM++ da bo dau cau ("MISSY:" -> word "MISSY") nen
    doi chieu bang token [A-Za-z0-9']+ casefold. Token toan text khop 1:1 voi
    timestamps o ~99% chunk (045: 360/363); lech thi tim subsequence token
    cua nhan — 0 hoac >= 2 cho khop la mo ho, bo qua (im lang, khong doan).
    """
    wts = chunk.get("english_word_timestamps") or []
    if not wts:
        return None
    text = chunk.get("english") or ""
    wt_tokens: list[tuple[int, str]] = []          # (index wts, token)
    for i, w in enumerate(wts):
        for tok in _tokens(str(w.get("word", ""))):
            wt_tokens.append((i, tok))
    label_toks = _tokens(match.group(1))
    if not label_toks:
        return None

    if len(_tokens(text)) == len(wt_tokens):
        idx = len(_tokens(text[:match.start()]))   # token dau tien cua nhan
        if idx >= len(wt_tokens):
            return None
        wt = wts[wt_tokens[idx][0]]
    else:
        flat = [t for _, t in wt_tokens]
        hits = [
            k for k in range(len(flat) - len(label_toks) + 1)
            if flat[k:k + len(label_toks)] == label_toks
        ]
        if len(hits) != 1:
            return None
        wt = wts[wt_tokens[hits[0]][0]]
    try:
        return float(wt["start"])
    except (KeyError, TypeError, ValueError):
        return None


def label_positions(
    chunk: dict, valid_names: list[str]
) -> list[tuple[float, str]]:
    """[(giay, ten canonical)] cho moi nhan dinh vi duoc trong chunk.

    Nhan o dau text -> thoi diem = start cua chunk (khong can timestamps);
    nhan giua text -> tra timestamps, khong dinh vi duoc thi bo qua nhan do.
    """
    text = chunk.get("english") or ""
    canon = {n.casefold(): n for n in valid_names}
    out: list[tuple[float, str]] = []
    for m in iter_script_labels(text):
        real = canon.get(m.group(1).strip().casefold())
        if not real:
            continue
        if label_at_text_start(text, m):
            try:
                out.append((float(chunk["start"]), real))
            except (KeyError, TypeError, ValueError):
                continue
        else:
            t = _mid_label_time(chunk, m)
            if t is not None:
                out.append((t, real))
    return out


def split_chunks_at_labels(
    chunks: list[dict], valid_names: list[str]
) -> list[dict]:
    """Cat moi chunk co nhan thanh cac vung: [start, t1) giu tag goc,
    [t1, t2) = LABEL::Ten1, [t2, end) = LABEL::Ten2... Vung dai 0 bi bo.

    Chunk khong nhan giu nguyen (tra ve dung dict goc, khong copy).
    """
    out: list[dict] = []
    for c in chunks:
        pos = label_positions(c, valid_names)
        try:
            start, end = float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
            out.append(c)
            continue
        # Kep ve [start, end): lech nho cua timestamps khong tao vung ngoai
        # chunk; nhan tu >= end tro di khong co vung de phu.
        pos = sorted((max(t, start), name) for t, name in pos if t < end)
        if not pos:
            out.append(c)
            continue
        bounds = [start] + [t for t, _ in pos] + [end]
        tags = [c.get("speaker_tag")] + [
            f"{LABEL_TAG_PREFIX}{name}" for _, name in pos
        ]
        for (s0, s1), tag in zip(zip(bounds, bounds[1:]), tags):
            if s1 <= s0:
                continue
            seg = dict(c)
            seg["start"], seg["end"], seg["speaker_tag"] = s0, s1, tag
            out.append(seg)
    return out


def apply_label_names(
    line_tags: list[str | None], names: list[str | None]
) -> list[str | None]:
    """Dong duoc gan vung LABEL::X nhan thang ten X (khong qua mapping
    cluster->ten — nhan kich ban dinh danh truc tiep)."""
    out = list(names)
    for i, t in enumerate(line_tags):
        if t and is_label_tag(t):
            out[i] = label_name(t)
    return out
