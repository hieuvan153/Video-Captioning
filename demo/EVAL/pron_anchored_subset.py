"""pron_anchored.py nhung CHI cham tren mot tap con cac cue nguon duoc giu lai
(vd tap chunk CHUNG giua nhieu arm co fallback khac nhau — xem
demo/EVAL/rebuild_guarded_srt.py (nay chi con tren nhanh research/base) va docs/eval/2026-09-06-tran-scene-context.md
muc "Guard cap chunk").

Vi sao can rieng file nay: khi cac arm fallback o cac chunk KHAC nhau, cham
PronF1 tren toan bo cue se lan "arm nao tranh duoc cu sap/lech dong" vao "arm
nao dat dai tu dung hon" — hai chuyen khac han nhau. Tap con "chung" (chunk ma
MOI arm dang so sanh deu khong fallback) loai bo nhieu do.

Cach loc: dung mot SRT "index" (vd rough SRT — cac arm guarded deu giu nguyen
gio cua rough nen chung so timing) de biet moi subtitle_index nam o khoang
thoi gian nao. Mot cue THAM CHIEU duoc giu lai chi khi MOI cue index-SRT giao
thoi gian voi no deu nam trong tap subtitle_index duoc giu (khong duoc "lan"
sang chunk bi loai) — tranh viec mot cue tham chieu nam vat qua ranh gioi
chunk bi tinh gop mo ho.

CLI:
  van_env/bin/python demo/EVAL/pron_anchored_subset.py --ref vi.srt \
      --index_srt rough.srt --keep_json common_chunks.json \
      --keep_key common_subtitle_index \
      --arms rough=a.srt base_repro=b.srt oracle=c.srt \
      --base base_repro --report out.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EVAL.film_ref_anchored import load, overlap_text  # noqa: E402
from EVAL.pron_anchored import (  # noqa: E402
    assign_scene_ids, bootstrap, bootstrap_cluster, cue_prf, f1_of, load_scenes,
)


def ref_keep_mask(ref: list, index_subs: list, keep_index: set[int]) -> list[bool]:
    """True cho cue tham chieu i neu MOI cue index_subs giao thoi gian voi no
    deu co .index nam trong keep_index (khong giao voi cue nao bi loai)."""
    idx = [(s.start.total_seconds(), s.end.total_seconds(), s.index) for s in index_subs]
    mask = []
    for r in ref:
        lo, hi = r.start.total_seconds(), r.end.total_seconds()
        hit = [i for (a, b, i) in idx if a < hi and b > lo]
        mask.append(bool(hit) and all(i in keep_index for i in hit))
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--index_srt", required=True, help="SRT dung timing/index de xac dinh cue nao thuoc chunk nao (vd rough SRT)")
    ap.add_argument("--keep_json", required=True, help="file JSON chua danh sach subtitle_index duoc giu")
    ap.add_argument("--keep_key", default="common_subtitle_index")
    ap.add_argument("--arms", nargs="+", required=True, help="ten=duong_dan.srt")
    ap.add_argument("--base", help="ten arm lam moc so sanh")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--cluster_by_scene", metavar="CAPTIONS_JSON",
                     help="Nhu pron_anchored.py: them ci95_cluster_scene, CI95 tu CLUSTER "
                          "BOOTSTRAP theo canh thay vi theo cue doc lap. Mac dinh (khong dat "
                          "co nay) khong doi hanh vi/con so cu.")
    ap.add_argument("--report")
    a = ap.parse_args()

    ref = load(a.ref)
    index_subs = load(a.index_srt)
    keep_index = set(json.load(open(a.keep_json, encoding="utf-8"))[a.keep_key])

    R = [re.sub(r"\s+", " ", s.content).strip() for s in ref]
    mask = ref_keep_mask(ref, index_subs, keep_index)
    arms = {k: overlap_text(ref, load(v)) for k, v in (x.split("=", 1) for x in a.arms)}
    keep = [i for i in range(len(R)) if R[i] and mask[i]]
    print(f"cue cham (tap con chung): {len(keep)}/{len(R)} "
          f"(sau loc trung khoang thoi gian voi {len(keep_index)} subtitle_index duoc giu)", flush=True)

    cluster_ids = None
    if a.cluster_by_scene:
        scenes = load_scenes(a.cluster_by_scene)
        cluster_ids = assign_scene_ids([ref[i] for i in keep], scenes)
        n_clusters = len(set(cluster_ids))
        print(f"cluster_by_scene: {n_clusters} canh hieu dung tren {len(keep)} cue "
              f"(tu {a.cluster_by_scene})", flush=True)

    rows = {k: cue_prf([R[i] for i in keep], [v[i] for i in keep]) for k, v in arms.items()}
    out = {k: {"pron_f1_anchored": round(f1_of(v), 4), "n_cue": len(keep)} for k, v in rows.items()}
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
