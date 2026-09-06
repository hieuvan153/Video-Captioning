"""Phan tich bo sung Step 4 (Task 3b, cong G1): oracle thay doi nhung gi so
voi base_repro, o muc do TUNG CUE thay vi chi so tong. Chi so tong (PronF1
neo) co the che mat chuyen dang xay ra: vd oracle co the dung dai tu TOT HON
o vai cue nhung TE HON o vai cue khac ma tong van tang nhe.

Dem 3 thu:
1. Bao nhieu cue tham chieu ma text overlap cua oracle khac base_repro (sau
   chuan hoa khoang trang) — cue "oracle co lam gi khac".
2. Trong so do, bao nhieu cue oracle khop dai tu tham chieu TOT HON / TE HON /
   HOA (so F1 dai tu tung cue, dung cue_prf cua demo/EVAL/pron_anchored.py).
3. Trong 84 canh CO tu xung ho oracle (oracle.captions.json, muc 3 khac
   "[None]") so voi 30 canh KHONG co — ti le cue duoc cai thien co tap trung
   o nhom 84 khong. Neu oracle that su co tac dung, cai thien phai LECH ve
   nhom co dap an; neu ti le gan nhau o ca hai nhom thi phan tang PronF1 tong
   co the la nhieu, khong phai do dung dung dap an xung ho.

CLI:
  van_env/bin/python demo/EVAL/oracle_delta.py --ref vi.srt \
      --base_repro base_repro.guarded.srt --oracle oracle.guarded.srt \
      --captions oracle.captions.json [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EVAL.film_ref_anchored import load, overlap_text  # noqa: E402
from EVAL.pron_anchored import cue_prf  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def cue_f1(row: tuple[int, int, int]) -> float | None:
    """F1 dai tu cua MOT cue. None neu ca ref va hyp deu khong co dai tu nao
    (khong so sanh duoc tot/te — coi la hoa o noi goi ham)."""
    tp, fp, fn = row
    if tp == 0 and fp == 0 and fn == 0:
        return None
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def load_scene_groups(captions_path: str) -> list[tuple[float, float, bool]]:
    """(start_time, end_time, has_answer) cho tung canh oracle.captions.json.
    has_answer = muc '3. Relationship: ...' khac '[None]'."""
    caps = json.load(open(captions_path, encoding="utf-8"))
    out = []
    for s in caps:
        m = re.search(r"3\.\s*Relationship:\s*(.*)", s["caption"])
        val = m.group(1).strip() if m else ""
        has_answer = bool(val) and "[None]" not in val
        out.append((s["start_time"], s["end_time"], has_answer))
    return out


def scene_group_of(t: float, scenes: list[tuple[float, float, bool]]) -> bool | None:
    for lo, hi, has in scenes:
        if lo <= t < hi:
            return has
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--base_repro", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--report")
    a = ap.parse_args()

    ref = load(a.ref)
    R = [norm(s.content) for s in ref]
    base_hyp = overlap_text(ref, load(a.base_repro))
    oracle_hyp = overlap_text(ref, load(a.oracle))
    scenes = load_scene_groups(a.captions)

    keep = [i for i in range(len(R)) if R[i]]
    rows_base = dict(zip(keep, cue_prf([R[i] for i in keep], [base_hyp[i] for i in keep])))
    rows_oracle = dict(zip(keep, cue_prf([R[i] for i in keep], [oracle_hyp[i] for i in keep])))

    diff = [i for i in keep if norm(base_hyp[i]) != norm(oracle_hyp[i])]

    better = worse = tie_scored = tie_no_pronoun = 0
    for i in diff:
        fb, fo = cue_f1(rows_base[i]), cue_f1(rows_oracle[i])
        if fb is None and fo is None:
            tie_no_pronoun += 1
        elif fb is None:
            fb = 0.0
        elif fo is None:
            fo = 0.0
        if fb is not None and fo is not None:
            if fo > fb:
                better += 1
            elif fo < fb:
                worse += 1
            else:
                tie_scored += 1

    # muc 3: theo nhom canh co/khong co dap an oracle
    group_stats = {True: {"n_cue": 0, "n_diff": 0, "n_better": 0, "n_worse": 0},
                   False: {"n_cue": 0, "n_diff": 0, "n_better": 0, "n_worse": 0}}
    unmatched = 0
    for i in keep:
        t = ref[i].start.total_seconds()
        g = scene_group_of(t, scenes)
        if g is None:
            unmatched += 1
            continue
        gs = group_stats[g]
        gs["n_cue"] += 1
        if i in diff:
            gs["n_diff"] += 1
            fb, fo = cue_f1(rows_base[i]), cue_f1(rows_oracle[i])
            fb = 0.0 if fb is None else fb
            fo = 0.0 if fo is None else fo
            if fo > fb:
                gs["n_better"] += 1
            elif fo < fb:
                gs["n_worse"] += 1

    out = {
        "n_cue_total": len(keep),
        "n_cue_diff_oracle_vs_base": len(diff),
        "diff_breakdown": {
            "oracle_better": better,
            "oracle_worse": worse,
            "tie_same_f1_both_have_pronoun": tie_scored,
            "tie_neither_has_pronoun": tie_no_pronoun,
        },
        "by_scene_answer_group": {
            "has_oracle_answer": {
                **group_stats[True],
                "improve_rate_of_diff": round(group_stats[True]["n_better"] / group_stats[True]["n_diff"], 4) if group_stats[True]["n_diff"] else None,
                "improve_rate_of_total": round(group_stats[True]["n_better"] / group_stats[True]["n_cue"], 4) if group_stats[True]["n_cue"] else None,
            },
            "no_oracle_answer": {
                **group_stats[False],
                "improve_rate_of_diff": round(group_stats[False]["n_better"] / group_stats[False]["n_diff"], 4) if group_stats[False]["n_diff"] else None,
                "improve_rate_of_total": round(group_stats[False]["n_better"] / group_stats[False]["n_cue"], 4) if group_stats[False]["n_cue"] else None,
            },
        },
        "n_cue_outside_any_scene_window": unmatched,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
