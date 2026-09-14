"""Cat SRT lam doi theo THOI GIAN de kiem khoa do dai co on dinh tren ca hai nua phim.

Cat theo thoi gian (khong theo chi so cue) de moi arm va tham chieu cung roi vao
dung mot nua, du so cue khac nhau.
"""
import sys, srt

def cut(path, t_split, half, out):
    subs = list(srt.parse(open(path, encoding="utf-8").read()))
    keep = [s for s in subs
            if (s.start.total_seconds() < t_split) == (half == "a")]
    open(out, "w", encoding="utf-8").write(srt.compose(keep))
    return len(keep), len(subs)

if __name__ == "__main__":
    if sys.argv[1] == "--demo":
        import io, datetime as dt
        mk = lambda i, s: srt.Subtitle(i, dt.timedelta(seconds=s),
                                       dt.timedelta(seconds=s+1), f"c{i}")
        io.open("/tmp/_d.srt", "w").write(srt.compose([mk(1,1), mk(2,50), mk(3,99)]))
        assert cut("/tmp/_d.srt", 50, "a", "/tmp/_a.srt") == (1, 3)
        assert cut("/tmp/_d.srt", 50, "b", "/tmp/_b.srt") == (2, 3)
        print("split_half demo OK"); sys.exit()
    import os
    t = float(sys.argv[1]); half = sys.argv[2]
    bases = [os.path.basename(p) for p in sys.argv[3:]]
    dup = sorted({b for b in bases if bases.count(b) > 1})
    if dup:
        sys.exit(f"trung ten file {dup}: {half}_<ten> se ghi de nhau trong HALFDIR, doi ten arm truoc")
    for p in sys.argv[3:]:
        base = os.path.basename(p)
        o = f"{os.environ['HALFDIR']}/{half}_{base}"
        n, tot = cut(p, t, half, o)
        print(f"{base}: {n}/{tot} cue -> {o}")
