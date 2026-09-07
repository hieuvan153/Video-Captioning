"""Cat lai cue cua mot ban ASR mà KHONG doi mot chu nao, de do RIENG gia hinh hoc tren noi dung that.

geom_ceiling.py do gia hinh hoc voi noi dung hoan hao (lay tham chieu lam ban dich). Con o day noi
dung la ban ASR that, chi ranh gioi bi thay. Hai bien the:
  oracle : dai lai text theo DUNG bien cua tham chieu  -> ref/cue = 1,00, phat hinh hoc = 0
  gridL  : dai lai theo luoi L giay, bo qua bien tham chieu
Hieu chrF giua `nguyen ban` va `oracle` = TOAN BO ngan sach hinh hoc con lai tren arm do.

CLI: recut.py ref.srt hyp.srt outdir [L1 L2 ...]
"""
import datetime as dt, io, os, sys, srt

ref = list(srt.parse(io.open(sys.argv[1], encoding="utf-8").read()))
hyp = list(srt.parse(io.open(sys.argv[2], encoding="utf-8").read()))
out = sys.argv[3]
os.makedirs(out, exist_ok=True)


def words():
    """(thoi diem, tu) — noi suy deu trong tung cue hyp; day la tat ca thong tin thoi gian ta co
    khi khong bat --word_ts. ponytail: du cho muc dich do RANH GIOI, khong dung de xuat ban."""
    for s in hyp:
        w = s.content.replace("\n", " ").split()
        b, e = s.start.total_seconds(), s.end.total_seconds()
        for i, tok in enumerate(w):
            yield b + (i + .5) / max(len(w), 1) * (e - b), tok


WS = list(words())


def dump(name, cues):
    cues = [(b, e, t) for b, e, t in cues if t]
    io.open(f"{out}/{name}.srt", "w", encoding="utf-8").write(srt.compose(
        [srt.Subtitle(index=i + 1, start=b, end=e, content=t) for i, (b, e, t) in enumerate(cues)]))
    rc = [(r.start.total_seconds(), r.end.total_seconds()) for r in ref]
    cov = [sum(1 for a, z in rc if a < e.total_seconds() and z > b.total_seconds()) for b, e, _ in cues]
    print(f"{name:10s} {len(cues):5d} cue  ref/cue {sum(cov)/len(cov):5.2f}  "
          f"tran~ {100 - 18.01 * (sum(cov)/len(cov) - 1):5.1f}")


def bucket(edges):
    """edges: danh sach (start, end) — gan moi tu vao khoang chua no; tu roi ngoai het thi bo vao
    khoang gan nhat de KHONG mat chu (mat chu se lan sang truc noi dung, hong phep do)."""
    bag = {}
    for t, tok in WS:
        k = next((i for i, (a, z) in enumerate(edges) if a <= t < z), None)
        if k is None:
            k = min(range(len(edges)), key=lambda i: min(abs(t - edges[i][0]), abs(t - edges[i][1])))
        bag.setdefault(k, []).append(tok)
    return [(dt.timedelta(seconds=edges[k][0]), dt.timedelta(seconds=edges[k][1]), " ".join(v))
            for k, v in sorted(bag.items())]


dump("orig", [(s.start, s.end, s.content.replace("\n", " ")) for s in hyp])
dump("oracle", bucket([(r.start.total_seconds(), r.end.total_seconds()) for r in ref]))
T = WS[-1][0] + 1
for L in (float(x) for x in sys.argv[4:]):
    dump(f"grid{L:g}", bucket([(k * L, (k + 1) * L) for k in range(int(T / L) + 1)]))
