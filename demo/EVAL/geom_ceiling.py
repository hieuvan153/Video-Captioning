"""Tran cua viec cat cue, do khong can GPU va khong con nhieu ve noi dung.
Lay chinh ban tham chieu VI lam 'ban dich', chi doi RANH GIOI cue roi cham lai voi chinh no.
chrF tut bao nhieu = gia phai tra thuan tuy cho hinh dang cue."""
import datetime as dt, io, sys, srt
src = list(srt.parse(io.open(sys.argv[1], encoding="utf-8").read()))
def dump(p, cues):
    io.open(p, "w", encoding="utf-8").write(srt.compose(
        [srt.Subtitle(index=i + 1, start=b, end=e, content=t) for i, (b, e, t) in enumerate(cues)]))
def merge(k):                       # gop k cue lien nhau -> cue dai gap k
    return [(g[0].start, g[-1].end, " ".join(s.content.replace("\n", " ") for s in g))
            for g in (src[i:i + k] for i in range(0, len(src), k))]
def halve():                        # cat doi moi cue: chia tu theo ty le, chia thoi gian giua
    out = []
    for s in src:
        w = s.content.replace("\n", " ").split()
        m = dt.timedelta(seconds=(s.start.total_seconds() + s.end.total_seconds()) / 2)
        if len(w) < 2: out.append((s.start, s.end, " ".join(w))); continue
        h = len(w) // 2
        out += [(s.start, m, " ".join(w[:h])), (m, s.end, " ".join(w[h:]))]
    return out
for name, cues in (("id", [(s.start, s.end, s.content.replace("\n", " ")) for s in src]),
                   ("merge2", merge(2)), ("merge3", merge(3)), ("halve", halve())):
    dump(f"{sys.argv[2]}/geo_{name}.srt", cues)
    print(f"geo_{name}: {len(cues)} cue")
