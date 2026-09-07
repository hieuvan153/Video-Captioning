"""Cue dai gop nhieu cau tham chieu se lam text bi dan vao MOI cue tham chieu no phu -> phong dai.
Do: do dai cue, so cue tham chieu ma mot cue hyp phu len, va lech |start| khi CHI xet cue 1-1."""
import io, sys, statistics as st, srt
L = lambda p: [(s.start.total_seconds(), s.end.total_seconds()) for s in srt.parse(io.open(p, encoding="utf-8").read())]
rs = L(sys.argv[1])
# tran chrF ~ 100 - 18,0*(ref/cue - 1): khop tu 10 phep bien dang hinh hoc trong geom_ceiling.py
# (id/halve/merge2-3/shift0,2-1,0/grid1,5-3,0), RMSE 2,88. Day la gia HINH HOC thuan tuy khi noi
# dung hoan hao - tren noi dung that no bi suy giam manh, dung doc nhu du bao chrF cuoi cung.
TRAN = lambda r: 100 - 18.01 * (r - 1)
print(f"{'arm':22s} {'cue':>5s} {'dai tb':>7s} {'dai vi':>7s} {'ref/cue':>8s} {'tran~':>6s} {'>=2ref':>7s} {'tong che':>9s}")
print(f"{'THAM CHIEU':22s} {len(rs):5d} {sum(e-b for b,e in rs)/len(rs):6.2f}s "
      f"{st.median([e-b for b,e in rs]):6.2f}s {'-':>8s} {'-':>6s} {'-':>7s} "
      f"{sum(e-b for b,e in rs)/(rs[-1][1]-rs[0][0])*100:8.1f}%")
for p in sys.argv[2:]:
    hs = L(p); d = [e-b for b,e in hs]
    cov = [sum(1 for b,e in rs if b < he and e > hb) for hb,he in hs]   # so cue ref moi cue hyp phu
    print(f"{p.rsplit('/',1)[-1][:22]:22s} {len(hs):5d} {sum(d)/len(d):6.2f}s {st.median(d):6.2f}s "
          f"{sum(cov)/len(cov):8.2f} {TRAN(sum(cov)/len(cov)):6.1f} {sum(c>=2 for c in cov)/len(cov)*100:6.1f}% "
          f"{sum(d)/(hs[-1][1]-hs[0][0])*100:8.1f}%")
