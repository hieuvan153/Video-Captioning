"""Task 8: mau 500 dong phan tang tu 13 movie holdout de nguoi gan nhan dai tu (B6).
CLI: van_env/bin/python demo/EVAL/sample_pronoun_annotation.py --out docs/eval/pronoun_annotation_500.csv"""
import argparse, csv, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from EVAL import split  # noqa: E402
from EVAL.pronoun_lexicon import extract_pronouns  # noqa: E402
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT = os.path.join(REPO, "data", "en-vi-speaker-with-time-pronouns")
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=500); a = ap.parse_args()
random.seed(0); pools = {}
for mv in split.load()["holdout"]:
    recs = json.load(open(os.path.join(GT, mv + ".json"), encoding="utf-8"))
    pools[mv] = [(i, r) for i, r in enumerate(recs) if extract_pronouns(r.get("vietnamese") or "")]
tot = sum(len(v) for v in pools.values()); rows = []
for mv, pool in pools.items():
    k = max(5, round(a.n * len(pool) / tot))
    for i, r in random.sample(pool, min(k, len(pool))):
        rows.append(dict(movie=mv, rec_index=i, chunk_id=r.get("chunk_id"), start=r["start"], end=r["end"], english=r["english"],
                         vietnamese=r["vietnamese"], machine_subject=r.get("pronouns_subject") or "", machine_object=r.get("pronouns_object") or "",
                         human_subject="", human_object="", note=""))
random.shuffle(rows); rows = rows[:a.n]
with open(a.out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), "rows,", len(set(r["movie"] for r in rows)), "movies ->", a.out)
