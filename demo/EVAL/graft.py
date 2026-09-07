"""Ghep NOI DUNG cua arm A vao LUOI CUE cua arm B - de tach "chu dung" khoi "dat dung cho".

Bai toan: `--no_vad` cho WER tot nhat (13,63 vs 16,73) nhung moc thoi gian troi 1,14 s nen thua
o tang san pham. Cau hoi: neu co noi dung cua no tren luoi cue tot, co thang khong? Khong can
model gong hang (WhisperX dung wav2vec2) va khong can moc tu: hai arm doc CUNG mot audio nen hai
chuoi tu gan trung nhau - difflib gong duoc, roi moi tu cua A theo tu khop cua B ve dung cue cua B.

CLI: van_env/bin/python demo/EVAL/graft.py --text a.srt --grid b.srt --out c.srt
"""
from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher

import srt

NORM = re.compile(r"[^\w']+", re.UNICODE)


def words_of(subs: list[srt.Subtitle]) -> tuple[list[str], list[int]]:
    """Tra ve (danh sach tu, cue cua tung tu) - chi can THU TU tu, khong can moc thoi gian."""
    ws: list[str] = []
    owner: list[int] = []
    for i, s in enumerate(subs):
        for w in re.sub(r"\s+", " ", s.content).split():
            ws.append(w)
            owner.append(i)
    return ws, owner


def graft(text: list[srt.Subtitle], grid: list[srt.Subtitle],
          fill: bool = True) -> list[srt.Subtitle]:
    tw, _ = words_of(text)
    gw, gown = words_of(grid)
    key = lambda ws: [NORM.sub("", w).lower() for w in ws]
    own: list[int | None] = [None] * len(tw)
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, key(tw), key(gw), autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                own[i1 + k] = gown[j1 + k]
        elif tag == "replace":
            # Tu thay the CO doi tac ben luoi - gan theo ti le, dung de roi ve hang xom trai.
            # (Selftest bat duoc: "generel" thay "general" o cue 1 ma left-fill nem ve cue 0.)
            for k in range(i2 - i1):
                own[i1 + k] = gown[j1 + (k * (j2 - j1)) // (i2 - i1)]
        # "delete" = tu chi co ben text, khong co doi tac -> de None, hai vong duoi keo theo hang xom
    last = None
    for i, o in enumerate(own):
        if o is None:
            own[i] = last
        else:
            last = o
    nxt = None
    for i in range(len(own) - 1, -1, -1):
        if own[i] is None:
            own[i] = nxt
        else:
            nxt = own[i]
    buckets: dict[int, list[str]] = {}
    for w, o in zip(tw, own):
        if o is not None:
            buckets.setdefault(o, []).append(w)
    out = []
    for i, s in enumerate(grid):
        # Cue khong nhan duoc chu nao = tu cua luoi nam tron trong khoi "insert" (luoi noi ma ban
        # kia khong noi). Bo cue di thi doan RONG, ma doan rong cham 0 - do duoc: bo 93 cue day
        # doan rong 3,6% -> 5,9% va an het phan loi WER. Giu lai chu cua chinh luoi thi te nhat
        # cung bang arm luoi, khong bao gio te hon.
        t = " ".join(buckets.get(i, ())) or (re.sub(r"\s+", " ", s.content).strip() if fill else "")
        if t:
            out.append(srt.Subtitle(index=len(out) + 1, start=s.start, end=s.end, content=t))
    return out


def selftest() -> None:
    from datetime import timedelta as T
    C = lambda a, b, t: srt.Subtitle(index=0, start=T(seconds=a), end=T(seconds=b), content=t)
    grid = [C(0, 1, "hello there"), C(1, 2, "general kenobi")]
    # cung chu, cue chia khac -> phai ve dung luoi cua grid
    out = graft([C(0, 2, "hello there general kenobi")], grid)
    assert [s.content for s in out] == ["hello there", "general kenobi"], [s.content for s in out]
    assert [(s.start, s.end) for s in out] == [(T(seconds=0), T(seconds=1)),
                                               (T(seconds=1), T(seconds=2))]
    # co tu thua ("oh") va tu sai ("generel"): van phai bam theo hang xom, khong duoc mat chu
    out = graft([C(0, 2, "oh hello there generel kenobi")], grid)
    assert [s.content for s in out] == ["oh hello there", "generel kenobi"], [s.content for s in out]
    # moc thoi gian cua nguon bi troi hoan toan -> khong anh huong, vi chi dung THU TU tu
    out = graft([C(50, 99, "hello there general kenobi")], grid)
    assert [s.content for s in out] == ["hello there", "general kenobi"]
    # cue cua luoi khong nhan duoc chu -> mac dinh giu chu cua chinh luoi, --no_fill thi bo
    out = graft([C(0, 1, "hello there")], grid)
    assert [s.content for s in out] == ["hello there", "general kenobi"], [s.content for s in out]
    out = graft([C(0, 1, "hello there")], grid, fill=False)
    assert [s.content for s in out] == ["hello there"], [s.content for s in out]
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="SRT lay NOI DUNG (arm WER tot)")
    ap.add_argument("--grid", help="SRT lay LUOI CUE + moc thoi gian (arm dat cho tot)")
    ap.add_argument("--out")
    ap.add_argument("--no_fill", action="store_true",
                    help="bo cue khong nhan duoc chu, thay vi giu chu cua luoi (mac dinh: giu)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    L = lambda p: list(srt.parse(open(p, encoding="utf-8-sig", errors="replace").read()))
    text, grid = L(a.text), L(a.grid)
    out = graft(text, grid, fill=not a.no_fill)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(srt.compose(out))
    nt = sum(len(s.content.split()) for s in text)
    no = sum(len(s.content.split()) for s in out)
    print(f"XONG: {len(out)}/{len(grid)} cue giu lai | tu {nt} -> {no} (mat {nt - no}) -> {a.out}")


if __name__ == "__main__":
    main()
