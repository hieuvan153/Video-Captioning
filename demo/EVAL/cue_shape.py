"""Cue dai gop nhieu cau tham chieu se lam text bi dan vao MOI cue tham chieu no phu -> phong dai.
Do: do dai cue, so cue tham chieu ma mot cue hyp phu len, va lech |start| khi CHI xet cue 1-1."""
import io, sys, statistics as st, srt
L = lambda p: [(s.start.total_seconds(), s.end.total_seconds()) for s in srt.parse(io.open(p, encoding="utf-8").read())]
rs = L(sys.argv[1])
print(f"{'arm':22s} {'cue':>5s} {'dai tb':>7s} {'dai vi':>7s} {'ref/cue':>8s} {'>=2ref':>7s} {'tong che':>9s}")
print(f"{'THAM CHIEU':22s} {len(rs):5d} {sum(e-b for b,e in rs)/len(rs):6.2f}s "
      f"{st.median([e-b for b,e in rs]):6.2f}s {'-':>8s} {'-':>7s} "
      f"{sum(e-b for b,e in rs)/(rs[-1][1]-rs[0][0])*100:8.1f}%")
for p in sys.argv[2:]:
    hs = L(p); d = [e-b for b,e in hs]
    cov = [sum(1 for b,e in rs if b < he and e > hb) for hb,he in hs]   # so cue ref moi cue hyp phu
    print(f"{p.rsplit('/',1)[-1][:22]:22s} {len(hs):5d} {sum(d)/len(d):6.2f}s {st.median(d):6.2f}s "
          f"{sum(cov)/len(cov):8.2f} {sum(c>=2 for c in cov)/len(cov)*100:6.1f}% "
          f"{sum(d)/(hs[-1][1]-hs[0][0])*100:8.1f}%")
