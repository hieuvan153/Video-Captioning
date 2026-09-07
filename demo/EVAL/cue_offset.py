"""Do do chinh xac moc thoi gian: voi moi cue tham chieu, lay cue hyp chong lan nhieu nhat
roi do |lech diem bat dau|. Thuoc nay giai thich chrF/COMET neo-theo-cue tot hon WER."""
import io, sys, statistics as st, srt
def load(p): return [(s.start.total_seconds(), s.end.total_seconds())
                     for s in srt.parse(io.open(p, encoding="utf-8").read())]
rs = load(sys.argv[1])
for p in sys.argv[2:]:
    hs = load(p); j = 0; offs = []; nohit = 0
    for b, e in rs:
        while j < len(hs) and hs[j][1] < b: j += 1
        best, bo, k = 0, None, j
        while k < len(hs) and hs[k][0] < e:
            ov = min(e, hs[k][1]) - max(b, hs[k][0])
            if ov > best: best, bo = ov, hs[k][0] - b
            k += 1
        (offs.append(abs(bo)) if bo is not None else None) or (nohit := nohit)
        if bo is None: nohit += 1
    print(f"{p.rsplit('/',1)[-1]:26s} {len(hs):5d} cue  khong khop {nohit/len(rs)*100:4.1f}%  "
          f"lech trung vi {st.median(offs):.2f}s  tb {sum(offs)/len(offs):.2f}s  "
          f">1s {sum(o>1 for o in offs)/len(offs)*100:4.1f}%", flush=True)
