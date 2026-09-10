"""VTT/SRT phu de NGUOI -> SRT tham chieu sach.

Bo dong loi bai hat (♪) va dong chu tren man hinh ([...]) — ASR khong sinh ra hai
loai nay nen giu lai chi lam nhieu phep do. Giong cach ref_3rd/vi_3rd.clean.srt
cua Ode to Joy da lam (bo [...]).

Doc luon VTT: chuan hoa sang dang SRT truoc khi parse (khong co thu vien webvtt).
"""
import re, sys, srt

BRACKET = re.compile(r"^\s*[\[\(].*[\]\)]\s*$", re.S)
TS = re.compile(r"^\d{1,2}:\d{2}:\d{2}[.,]\d{3}\s*-->")


def vtt_to_srt(text):
    """Bo header WEBVTT/NOTE/STYLE, doi '.' -> ',' o dau thoi gian, danh so lai cue."""
    text = text.replace("\r\n", "\n").lstrip("﻿")
    text = re.sub(r"(\d{2}:\d{2}:\d{2})\.(\d{3})", r"\1,\2", text)
    out, n = [], 0
    for block in re.split(r"\n\s*\n", text):
        lines = [l for l in block.split("\n") if l.strip()]
        while lines and not TS.match(lines[0]):
            lines.pop(0)                      # bo dong ten cue / header
        if len(lines) < 2:
            continue
        n += 1
        out.append(f"{n}\n" + "\n".join(lines))
    return "\n\n".join(out) + "\n"


def clean(text):
    lines = [l for l in text.splitlines()
             if l.strip() and "♪" not in l and not BRACKET.match(l)]
    joined = " ".join(" ".join(lines).split())
    # ngoac vuong hay trai qua HAI dong ("[Chuyen the tu ...\ncua tac gia ...]"),
    # loc theo tung dong khong bat duoc -> kiem lai ca cue sau khi ghep.
    return "" if BRACKET.match(joined) else joined


def run(src, dst):
    raw = open(src, encoding="utf-8-sig").read()
    if raw.lstrip().upper().startswith("WEBVTT"):
        raw = vtt_to_srt(raw)
    subs = list(srt.parse(raw))
    out, drop = [], 0
    for s in subs:
        c = clean(s.content)
        if c:
            out.append(srt.Subtitle(len(out) + 1, s.start, s.end, c))
        else:
            drop += 1
    open(dst, "w", encoding="utf-8").write(srt.compose(out))
    print(f"{src.split('/')[-1]}: giu {len(out)}/{len(subs)} cue, bo {drop} -> {dst}")
    return len(out)


def _demo():
    assert clean("♪ la la ♪") == ""
    assert clean("[Chu tren man hinh]") == ""
    assert clean("(ghi chu)") == ""
    assert clean("Xin chao\nban khoe khong") == "Xin chao ban khoe khong"
    assert clean("Toi [nghi] the") == "Toi [nghi] the"      # ngoac giua cau thi GIU
    assert clean('[Chuyen the tu "X"\ncua tac gia Y]') == ""   # ngoac trai 2 dong
    assert clean("  ") == ""
    v = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nchao\n\ncue-7\n00:00:03.000 --> 00:00:04.500\nban\n"
    s = list(srt.parse(vtt_to_srt(v)))
    assert [x.content for x in s] == ["chao", "ban"], s
    assert s[1].start.total_seconds() == 3.0
    assert s[0].index == 1 and s[1].index == 2
    print("clean_ref demo OK")


if __name__ == "__main__":
    _demo() if sys.argv[1] == "--demo" else run(sys.argv[1], sys.argv[2])
