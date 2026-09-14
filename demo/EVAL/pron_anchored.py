"""Do dat dai tu theo TUNG cue tham chieu, khong phai multiset toan phim.

Vi sao: PronF1 toan phim (EVAL/pronoun_f1.py) chi thuong dung PHAN BO dai tu.
Bao cao 2026-09-04: arm Gemma bs1 duoc +0.039 PronF1 toan phim nhung khi neo theo
tham chieu thi am han (dai tu dung them 118 / sai them 117). Metric nay bit lo do.

CLI:
  van_env/bin/python demo/EVAL/pron_anchored.py --ref vi.srt \
      --arms base=a.srt oracle=b.srt --base base [--n 1000] [--report out.json]

CHU Y GIO HAN:
  Metric chi so sanh cong bang GIUA cac arm DUNG CHUNG cach chia cue (e.g., moi arm
  sinh tu cung file rough SRT). Khi hai arm co granularity cue khac nhau ro ret
  (arm ASR vs arm phu de chuan), overlap_text chia tu cua cue arm dai theo ty le thoi gian
  giao voi cac cue tham chieu (tu 14/09; truoc do nhan ban nguyen cau). Phep chia la xap xi,
  nen delta van phan anh mot phan do "nuot/gop cue" chu khong rieng do chinh xac dai tu.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EVAL.film_ref_anchored import load, overlap_text, percentile as _percentile  # noqa: E402
from EVAL.pronoun_lexicon import extract_pronouns  # noqa: E402


def cue_prf(ref_texts: list[str], hyp_texts: list[str]) -> list[tuple[int, int, int]]:
    """(tp, fp, fn) dai tu cho TUNG cue. Multiset trong pham vi mot cue."""
    rows = []
    for r, h in zip(ref_texts, hyp_texts):
        cr, ch = Counter(extract_pronouns(r)), Counter(extract_pronouns(h))
        tp = sum((cr & ch).values())
        rows.append((tp, sum(ch.values()) - tp, sum(cr.values()) - tp))
    return rows


def f1_of(rows: list[tuple[int, int, int]]) -> float:
    tp = sum(r[0] for r in rows); fp = sum(r[1] for r in rows); fn = sum(r[2] for r in rows)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0




def bootstrap(rows_a, rows_b, n=1000, seed=42):
    """CI95 cua F1(a) - F1(b), lay mau ghep cap THEO CUE tren cung tap chi so cue.

    CHU Y GIOI HAN: don vi lay mau la CUE doc lap. Khi cac cue duoc gan nhau boi mot
    dac diem CAP CAO HON (vd ngu canh oracle gan theo CANH, moi cue trong cung canh
    dung chung mot khoi ngu canh nen tuong quan voi nhau), n hieu dung THAT SU nho hon
    so cue va CI nay se HEP HON thuc te. Dung `bootstrap_cluster` khi co don vi cum ro
    rang (vd canh)."""
    rng = random.Random(seed)
    k = len(rows_a)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        diffs.append(f1_of([rows_a[i] for i in idx]) - f1_of([rows_b[i] for i in idx]))
    diffs.sort()
    return _percentile(diffs, 0.025), _percentile(diffs, 0.975)


def bootstrap_cluster(rows_a, rows_b, cluster_ids: list, n=1000, seed=42):
    """CI95 cua F1(a) - F1(b) bang CLUSTER BOOTSTRAP: don vi lay mau lai la CUM (vd
    canh), khong phai tung cue. Moi vong lap chon lai cac cum CO HOAN LAI (so cum bang
    so cum ban dau), roi gop TOAN BO cue cua moi cum duoc chon (chon lap thi lap ca
    khoi cue). Dung khi cac cue trong cung cum tuong quan voi nhau (vd chung mot khoi
    ngu canh) — bootstrap theo cue se cho CI hep hon thuc te trong truong hop do.

    rows_a, rows_b, cluster_ids phai cung do dai va cung thu tu (cluster_ids[i] la id
    cum cua rows_a[i]/rows_b[i])."""
    rng = random.Random(seed)
    by_cluster: dict = {}
    for i, cid in enumerate(cluster_ids):
        by_cluster.setdefault(cid, []).append(i)
    clusters = list(by_cluster.keys())
    m = len(clusters)
    diffs = []
    for _ in range(n):
        chosen = [clusters[rng.randrange(m)] for _ in range(m)]
        idx = [i for cid in chosen for i in by_cluster[cid]]
        diffs.append(f1_of([rows_a[i] for i in idx]) - f1_of([rows_b[i] for i in idx]))
    diffs.sort()
    return _percentile(diffs, 0.025), _percentile(diffs, 0.975)


def load_scenes(captions_path: str) -> list[tuple[float, float, object]]:
    """Doc scene start/end tu file captions JSON (vd oracle.captions.json), cung dinh
    dang voi make_oracle_context.py: moi phan tu la dict co start_time/end_time va
    scene_id (hoac chi so trong list neu thieu scene_id)."""
    with open(captions_path, encoding="utf-8") as f:
        scenes = json.load(f)
    out = []
    for i, sc in enumerate(scenes):
        st, en = sc.get("start_time"), sc.get("end_time")
        if st is None or en is None:
            continue
        out.append((float(st), float(en), sc.get("scene_id", i)))
    return out


def assign_scene_ids(ref_subs, scenes: list[tuple[float, float, object]]) -> list:
    """Gan moi cue THAM CHIEU vao id canh giao thoi gian NHIEU NHAT (cung quy tac thoi
    gian voi `scene_terms` trong make_oracle_context.py: scene.start < cue.end VA
    scene.end > cue.start). Cue khong giao voi canh nao duoc gan mot id rieng (cum don
    le, khong lan sang canh khac) de khong bi am tham loai khoi cluster bootstrap."""
    ids = []
    next_singleton = -1
    for r in ref_subs:
        lo, hi = r.start.total_seconds(), r.end.total_seconds()
        best_id, best_overlap = None, 0.0
        for st, en, sid in scenes:
            if st < hi and en > lo:
                ov = min(hi, en) - max(lo, st)
                if ov > best_overlap:
                    best_overlap, best_id = ov, sid
        if best_id is None:
            ids.append(next_singleton)
            next_singleton -= 1
        else:
            ids.append(best_id)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Danh gia dai tu neo theo cue tham chieu. CHU Y: chi hop le khi moi arm dung chung cach chia cue; neu granularity khac nhau, delta co the sai vao dung luong cue overlap.")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="ten=duong_dan.srt")
    ap.add_argument("--base", help="ten arm lam moc so sanh")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--cluster_by_scene", metavar="CAPTIONS_JSON",
                     help="Neu dat: THEM ci95_cluster_scene, CI95 tu CLUSTER BOOTSTRAP theo "
                          "canh (dung start_time/end_time cua file captions JSON nay, vd "
                          "oracle.captions.json) thay vi theo cue doc lap. Bootstrap theo cue "
                          "(ci95) van la mac dinh, khong bi thay doi.")
    ap.add_argument("--report")
    a = ap.parse_args()

    ref = load(a.ref)
    R = [re.sub(r"\s+", " ", s.content).strip() for s in ref]
    arms = {k: overlap_text(ref, load(v)) for k, v in (x.split("=", 1) for x in a.arms)}
    keep = [i for i in range(len(R)) if R[i]]
    print(f"cue cham: {len(keep)}/{len(R)}", flush=True)

    cluster_ids = None
    if a.cluster_by_scene:
        scenes = load_scenes(a.cluster_by_scene)
        cluster_ids = assign_scene_ids([ref[i] for i in keep], scenes)
        n_clusters = len(set(cluster_ids))
        print(f"cluster_by_scene: {n_clusters} canh hieu dung tren {len(keep)} cue "
              f"(tu {a.cluster_by_scene})", flush=True)

    rows = {k: cue_prf([R[i] for i in keep], [v[i] for i in keep]) for k, v in arms.items()}
    out = {k: {"pron_f1_anchored": round(f1_of(v), 4)} for k, v in rows.items()}
    if a.base and a.base in rows:
        for k in rows:
            if k == a.base:
                continue
            lo, hi = bootstrap(rows[k], rows[a.base], a.n)
            out[k]["delta_vs_base"] = round(f1_of(rows[k]) - f1_of(rows[a.base]), 4)
            out[k]["ci95"] = [round(lo, 4), round(hi, 4)]
            if cluster_ids is not None:
                clo, chi = bootstrap_cluster(rows[k], rows[a.base], cluster_ids, a.n)
                out[k]["ci95_cluster_scene"] = [round(clo, 4), round(chi, 4)]
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
