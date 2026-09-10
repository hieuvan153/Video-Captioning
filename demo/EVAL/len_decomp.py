"""Phan ra cho THIEU do dai: loi tang dich hay loi ASR?

Phat penalty do dai (BP) la cho mat diem BLEU lon nhat tren phim ben thu 3. Cau hoi
la loi cua ai. Dem thang so tu tren CA PHIM (giong giao thuc Bang 4.7: noi ca phim
thanh mot chuoi), roi tach lam hai ti le doc lap:

    EN cua ASR / EN cua nguoi   -> ASR bo mat bao nhieu noi dung
    VI cua arm / EN cua ASR     -> tang dich gian no bao nhieu lan

So sanh ti le thu hai voi vi/en cua NGUOI DICH thi biet tang dich con du dia khong.

CANH BAO: dung neo cue theo thoi gian de do do dai thi SAI — mot cue tham chieu
thuong trum len nhieu cue gia thuyet nen chu bi dem lap, hyp/ref vot len tren 1,0.
Phai dem thang nhu day.

Dung:
    python len_decomp.py --ref_vi vi.srt --ref_en en.srt --src_en asr_en.srt \\
        arm1=a.srt arm2=b.srt
"""
import argparse, sys

import srt


def words(path):
    subs = list(srt.parse(open(path, encoding="utf-8").read()))
    return sum(len(s.content.split()) for s in subs), len(subs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_vi", required=True, help="phu de VI cua nguoi")
    ap.add_argument("--ref_en", required=True, help="phu de EN cua nguoi")
    ap.add_argument("--src_en", required=True, help="phu de EN do ASR sinh")
    ap.add_argument("arms", nargs="+", help="ten=duong_dan.srt")
    a = ap.parse_args()

    nvi, cvi = words(a.ref_vi)
    nen, cen = words(a.ref_en)
    nasr, casr = words(a.src_en)
    print(f"EN nguoi   {nen:6d} tu / {cen:5d} cue = {nen/cen:.2f} tu/cue")
    print(f"EN cua ASR {nasr:6d} tu / {casr:5d} cue = {nasr/casr:.2f} tu/cue"
          f"   -> ASR/nguoi = {nasr/nen:.3f}")
    print(f"VI nguoi   {nvi:6d} tu / {cvi:5d} cue = {nvi/cvi:.2f} tu/cue"
          f"   -> nguoi dich gian vi/en = {nvi/nen:.3f}")
    print()
    for arm in a.arms:
        name, path = arm.split("=", 1)
        nh, ch = words(path)
        print(f"{name:10s} {nh:6d} tu / {ch:5d} cue | hyp/ref {nh/nvi:.4f} | "
              f"gian no vi/en {nh/nasr:.3f} (nguoi {nvi/nen:.3f})")


def _demo():
    import io, os, datetime as dt, tempfile
    d = tempfile.mkdtemp()
    mk = lambda i, t: srt.Subtitle(i, dt.timedelta(seconds=i),
                                   dt.timedelta(seconds=i + 1), t)
    p = os.path.join(d, "x.srt")
    io.open(p, "w", encoding="utf-8").write(srt.compose([mk(1, "a b c"), mk(2, "d")]))
    assert words(p) == (4, 2), words(p)
    print("len_decomp demo OK")


if __name__ == "__main__":
    _demo() if sys.argv[1:] == ["--demo"] else main()
