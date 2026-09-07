"""Tran cua viec cat cue, do khong can GPU va khong con nhieu ve noi dung.
Lay chinh ban tham chieu VI lam 'ban dich', chi doi RANH GIOI cue roi cham lai voi chinh no.
chrF tut bao nhieu = gia phai tra thuan tuy cho hinh dang cue.

Ket qua 07/09 tren vi_3rd.clean.srt (1767 cue cham duoc), chrF:
  id 100,00 | halve 100,00 | merge2 81,98 | merge3 69,49
  shift 0,2s 88,54 | 0,5s 86,67 | 1,0s 83,33
  grid 1,5s 89,82 | 2,0s 86,26 | 3,0s 80,28
Bai hoc: halve() = 100 KHONG co nghia "cat nho la mien phi". halve cat GIUA mot cue tham chieu nen
moi manh van cham dung mot cue. Cat lai that (grid) thi bien roi tuy y va mat 10-20 chrF du noi dung
hoan hao. Thu quyet dinh la BIEN CO TRUNG voi tham chieu khong, do dai cue chi anh huong gian tiep."""
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
def shift(d):                       # giu nguyen hinh dang cue, chi doi TOAN BO moc thoi gian
    return [(s.start + dt.timedelta(seconds=d), s.end + dt.timedelta(seconds=d),
             s.content.replace("\n", " ")) for s in src]

def grid(L):                        # cat lai theo luoi L giay, KHONG biet ranh gioi cue tham chieu
    """Kiem tra gia dinh cua halve(): halve cat GIUA mot cue nen moi manh van cham dung 1 cue tham
    chieu. Cat lai that thi bien roi tuy y -> co the cham 2 cue. Day la truong hop xau tuong ung."""
    bag = {}
    for s in src:
        w = s.content.replace("\n", " ").split()
        b0, e0 = s.start.total_seconds(), s.end.total_seconds()
        for i, tok in enumerate(w):
            t = b0 + (i + .5) / len(w) * (e0 - b0)
            bag.setdefault(int(t // L), []).append(tok)
    return [(dt.timedelta(seconds=k * L), dt.timedelta(seconds=(k + 1) * L), " ".join(v))
            for k, v in sorted(bag.items())]

for name, cues in (("id", [(s.start, s.end, s.content.replace("\n", " ")) for s in src]),
                   ("merge2", merge(2)), ("merge3", merge(3)), ("halve", halve()),
                   ("shift0.2", shift(0.2)), ("shift0.5", shift(0.5)), ("shift1.0", shift(1.0)),
                   ("grid1.5", grid(1.5)), ("grid2.0", grid(2.0)), ("grid3.0", grid(3.0))):
    dump(f"{sys.argv[2]}/geo_{name}.srt", cues)
    print(f"geo_{name}: {len(cues)} cue")
