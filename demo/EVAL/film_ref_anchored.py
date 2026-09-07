"""Cham nhieu arm tren CUNG mot tap doan, neo theo cue THAM CHIEU (khong theo cue cua arm).

Vi sao: neo theo cue cua arm (film_score.py) lam arm nao nuot noi dung se duoc mien cham
dung cho no nuot -> arm ASR duoc loi the gia. Neo theo tham chieu thi moi cue tham chieu
deu phai co doi chung o moi arm, arm nao thieu thi nhan chuoi rong va bi phat.

Moi cue tham chieu -> hyp = noi cac cue cua arm giao thoi gian (khu lap), src = tuong tu tren
phu de EN chuan. Xuat BLEU / chrF / COMET-DA + paired bootstrap so voi arm --base.

CLI: van_env/bin/python demo/EVAL/film_ref_anchored.py --ref vi.srt --src_en en_chuan.srt \
        --arms rough=a.srt gold_rough=b.srt --base rough [--no_comet] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sacrebleu  # noqa: E402
import srt  # noqa: E402


def load(p: str) -> list[srt.Subtitle]:
    with open(p, encoding="utf-8-sig", errors="replace") as f:
        return list(srt.parse(f.read()))


def overlap_text(ref: list[srt.Subtitle], other: list[srt.Subtitle],
                 min_ov: float = 0.0) -> list[str]:
    """Voi moi cue tham chieu, noi text cac cue cua `other` giao thoi gian voi no.

    min_ov = phan giao TOI THIEU (giay) moi tinh la trung. Mac dinh 0.0 = luat cu, giu nguyen
    moi con so da cong bo. Nup calibration, khong phai mac dinh moi: do 07/09, o nguong 0 thi
    chinh BAN THAM CHIEU EN cua NGUOI (phu de chuyen nghiep cung phim) cham voi thuoc VI ra
    ref/cue = 1,50 - te hon arm may 1,19 - chi vi duoi cue EN tran qua bien VI, 41% truong hop
    tran duoi 0,2 s. Nang nguong len 0,2 s thi con 0,97, dung bang moc nguoi. Tuc luat "giao
    mot phan nghin giay cung tinh" thuong cue NGAN va phat do dai cue dung nhip nguoi.
    """
    o = [(s.start.total_seconds(), s.end.total_seconds(),
          re.sub(r"\s+", " ", s.content).strip()) for s in other]
    out = []
    for r in ref:
        lo, hi = r.start.total_seconds(), r.end.total_seconds()
        hit = [t for (a, b, t) in o if min(b, hi) - max(a, lo) > min_ov and t]
        out.append(" ".join(hit))
    return out


def selftest() -> None:
    from datetime import timedelta as T
    C = lambda a, b, t: srt.Subtitle(index=0, start=T(seconds=a), end=T(seconds=b), content=t)
    ref = [C(0, 1, "A"), C(1, 2, "B")]
    hyp = [C(0, 1.05, "x")]                       # tran 0.05s sang cue B
    assert overlap_text(ref, hyp) == ["x", "x"], "nguong 0: tran mot chut van bi dan hai lan"
    assert overlap_text(ref, hyp, 0.2) == ["x", ""], "nguong 0.2s: tran nho khong con tinh"
    assert overlap_text(ref, [C(5, 6, "z")]) == ["", ""]          # roi hoan toan
    assert overlap_text(ref, [C(0, 2, "y")], 0.2) == ["y", "y"]   # trum that thi van tinh ca hai
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--src_en", required=True)
    ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--base")
    ap.add_argument("--no_comet", action="store_true"); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--report")
    ap.add_argument("--min_overlap_s", type=float, default=0.0,
                    help="phan giao toi thieu moi tinh la trung cue (0.0 = luat cu)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    ref = load(a.ref)
    R = [re.sub(r"\s+", " ", s.content).strip() for s in ref]
    S = overlap_text(ref, load(a.src_en), a.min_overlap_s)
    arms = {k: overlap_text(ref, load(v), a.min_overlap_s)
            for k, v in (x.split("=", 1) for x in a.arms)}
    keep = [i for i in range(len(R)) if R[i] and S[i]]
    print(f"doan cham: {len(keep)}/{len(R)} cue tham chieu (bo cue khong co nguon EN)", flush=True)

    comet = None
    if not a.no_comet:
        from comet import download_model, load_from_checkpoint
        comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    rep = {}
    seg = {}
    for name, H in arms.items():
        h = [H[i] for i in keep]; r = [R[i] for i in keep]
        row = {"bleu": round(sacrebleu.corpus_bleu(h, [r]).score, 2),
               "chrf": round(sacrebleu.corpus_chrf(h, [r]).score, 2),
               "empty_rate": round(sum(1 for x in h if not x) / len(h), 4),
               "n_seg": len(h)}
        if comet is not None:
            data = [{"src": S[i], "mt": H[i], "ref": R[i]} for i in keep]
            out = comet.predict(data, batch_size=32, gpus=1 if _cuda() else 0, progress_bar=False)
            seg[name] = list(out.scores)
            row["comet_da"] = round(sum(out.scores) / len(out.scores), 4)
        rep[name] = row
        print(f"{name:12s} BLEU {row['bleu']:6.2f}  chrF {row['chrf']:6.2f}  doan rong {row['empty_rate']:.1%}"
              + (f"  COMET-DA {row['comet_da']:.4f}" if "comet_da" in row else ""), flush=True)

    if a.base and a.base in arms:
        base = a.base
        for name in arms:
            if name == base:
                continue
            hb = [arms[base][i] for i in keep]; hn = [arms[name][i] for i in keep]; r = [R[i] for i in keep]
            for mname, fn in (("BLEU", sacrebleu.corpus_bleu), ("chrF", sacrebleu.corpus_chrf)):
                d0 = fn(hn, [r]).score - fn(hb, [r]).score
                random.seed(0); ds = []
                for _ in range(a.n):
                    idx = [random.randrange(len(r)) for _ in r]
                    ds.append(fn([hn[i] for i in idx], [[r[i] for i in idx]]).score
                              - fn([hb[i] for i in idx], [[r[i] for i in idx]]).score)
                ds.sort()
                print(f"  {mname}: {name} - {base} = {d0:+.2f}  95%CI=[{ds[int(.025*a.n)]:+.2f}, {ds[int(.975*a.n)]:+.2f}]", flush=True)
            if name in seg and base in seg:
                d = [x - y for x, y in zip(seg[name], seg[base])]
                random.seed(0); ds = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(a.n))
                print(f"  COMET-DA: {name} - {base} = {sum(d)/len(d):+.4f}"
                      f"  95%CI=[{ds[int(.025*a.n)]:+.4f}, {ds[int(.975*a.n)]:+.4f}]", flush=True)

    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump({"ref": a.ref, "src_en": a.src_en, "scores": rep}, f, ensure_ascii=False, indent=1)
        print("Report:", a.report)


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
