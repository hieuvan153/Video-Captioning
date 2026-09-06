"""Sinh thu muc eval per-movie (en/ref/rough SRT + gold_pronouns.json) tu dataset.

Nguon: data/en-vi-speaker-with-time-pronouns/<id>.json — moi record co english/
vietnamese/vietsub_raw/start/end/pronouns_subject/pronouns_object. rough.srt
chinh la cot vietsub_raw (da kiem chung 320/320 dong khop movie_008 —
khong can chay lai NMT). gold_pronouns.json = [subject, object] bo None,
dung thu tu do (khop file cu cua 5 phim eval V0).

Video/tagged khong sinh o day — chi bao thieu de driver GPU (run_expand_eval.sh)
biet bo qua buoc nao.

CLI:
    van_env/bin/python demo/EVAL/prep_eval_dirs.py --movies movie_001 ...
    van_env/bin/python demo/EVAL/prep_eval_dirs.py --check movie_008   # doi chieu
    van_env/bin/python demo/EVAL/prep_eval_dirs.py --list              # universe
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ROOT_DIR)
GT_DIR = os.path.join(REPO_ROOT, "data", "en-vi-speaker-with-time-pronouns")
PLAN = os.path.join(REPO_ROOT, "data", "speaker_verify_campp_en_full",
                    "manifests", "plan.json")
TAGGED_ROOT = os.path.join(REPO_ROOT, "data", "speaker_verify_campp_en_full")


def load_plan() -> dict[str, dict]:
    with open(PLAN, encoding="utf-8") as f:
        entries = json.load(f)
    return {os.path.splitext(e["json_name"])[0]: e for e in entries}


def find_video(entry: dict) -> str | None:
    """Video .mkv/.mp4 nam cung thu muc voi audio_path trong plan.json."""
    d = os.path.dirname(entry.get("audio_path") or "")
    if not os.path.isdir(d):
        return None
    for ext in (".mkv", ".mp4", ".avi", ".webm"):
        hits = glob.glob(os.path.join(d, "*" + ext))
        if hits:
            return sorted(hits)[0]
    return None


def find_tagged(movie: str) -> str | None:
    hits = glob.glob(os.path.join(
        TAGGED_ROOT, "*", "*", "*", "episodes", movie, f"{movie}.tagged.json"))
    hits += glob.glob(os.path.join(
        TAGGED_ROOT, "movies", "*", "files", movie, f"{movie}.tagged.json"))
    return hits[0] if hits else None


def _sorted_recs(recs: list[dict]) -> list[dict]:
    """Sap theo thoi gian nhu prep V0 (SRT phai tang dan; dataset co vai cap
    dong dao thu tu — movie_045 dong 21/22). Sort on dinh giu hoa cua cap
    trung start."""
    return sorted(recs, key=lambda r: (float(r.get("start") or 0.0),
                                       float(r.get("end") or 0.0)))


def _pronouns(rec: dict) -> list[str]:
    """Truong subject/object co the chua nhieu dai tu cach nhau dau phay
    ("co, ai") — gold V0 tach tung cai, giu thu tu subject truoc object."""
    out = []
    for field in (rec.get("pronouns_subject"), rec.get("pronouns_object")):
        if field:
            out.extend(p.strip() for p in str(field).split(",") if p.strip())
    return out


def _srt_block(i: int, start: float, end: float, text: str) -> str:
    def ts(sec: float) -> str:
        td = datetime.timedelta(seconds=max(0.0, sec))
        total = int(td.total_seconds())
        ms = int(round((td.total_seconds() - total) * 1000))
        return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d},{ms:03d}"
    return f"{i}\n{ts(start)} --> {ts(end)}\n{text.strip()}\n"


def write_movie(movie: str, out_root: str) -> dict:
    with open(os.path.join(GT_DIR, f"{movie}.json"), encoding="utf-8") as f:
        recs = json.load(f)
    recs = _sorted_recs(recs)
    mdir = os.path.join(out_root, movie)
    os.makedirs(mdir, exist_ok=True)
    cols = {"en.srt": "english", "ref.srt": "vietnamese",
            "rough.srt": "vietsub_raw"}
    for fname, key in cols.items():
        path = os.path.join(mdir, fname)
        if os.path.exists(path):
            continue
        blocks = [_srt_block(i + 1, r["start"], r["end"], r.get(key) or "")
                  for i, r in enumerate(recs)]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(blocks))
    gold_path = os.path.join(mdir, "gold_pronouns.json")
    if not os.path.exists(gold_path):
        gold = [_pronouns(r) for r in recs]
        with open(gold_path, "w", encoding="utf-8") as f:
            json.dump(gold, f, ensure_ascii=False)
    return {"n_lines": len(recs)}


def check_movie(movie: str, out_root: str) -> bool:
    """Doi chieu du lieu sinh ra voi thu muc eval da co (5 phim V0)."""
    sys.path.insert(0, ROOT_DIR)
    from EVAL.run_eval import load_srt_lines

    with open(os.path.join(GT_DIR, f"{movie}.json"), encoding="utf-8") as f:
        recs = json.load(f)
    recs = _sorted_recs(recs)
    mdir = os.path.join(out_root, movie)
    ok = True
    for fname, key in (("en.srt", "english"), ("ref.srt", "vietnamese"),
                       ("rough.srt", "vietsub_raw")):
        have = load_srt_lines(os.path.join(mdir, fname))
        want = [(r.get(key) or "").strip() for r in recs]
        # load_srt_lines gop newline noi bo thanh cach — so sanh cung chuan
        want = [" ".join(w.split()) for w in want]
        have = [" ".join(h.split()) for h in have]
        n_bad = sum(1 for a, b in zip(have, want) if a != b)
        status = "OK" if (len(have) == len(want) and n_bad == 0) else "LECH"
        if status != "OK":
            ok = False
        print(f"  {movie}/{fname}: {len(have)} vs {len(want)} dong, "
              f"{n_bad} dong lech -> {status}")
    with open(os.path.join(mdir, "gold_pronouns.json"), encoding="utf-8") as f:
        gold_have = json.load(f)
    gold_want = [_pronouns(r) for r in recs]
    n_bad = sum(1 for a, b in zip(gold_have, gold_want) if a != b)
    print(f"  {movie}/gold_pronouns.json: {n_bad} dong lech "
          f"-> {'OK' if n_bad == 0 else 'LECH'}")
    return ok and n_bad == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out_root", type=str,
                        default=os.path.join(ROOT_DIR, "output", "eval_ab"))
    parser.add_argument("--movies", type=str, nargs="*", default=None)
    parser.add_argument("--check", type=str, nargs="*", default=None,
                        help="Doi chieu thay vi ghi (thu muc phai co san).")
    parser.add_argument("--list", action="store_true",
                        help="In universe: id nao du video+gold+tagged.")
    args = parser.parse_args()

    plan = load_plan()
    if args.check:
        ok = all(check_movie(m, args.out_root) for m in args.check)
        raise SystemExit(0 if ok else 1)
    if args.list:
        for movie in sorted(plan):
            gold = os.path.exists(os.path.join(GT_DIR, f"{movie}.json"))
            video = find_video(plan[movie])
            tagged = find_tagged(movie)
            flags = (("G" if gold else "-") + ("V" if video else "-")
                     + ("T" if tagged else "-"))
            print(f"{movie}\t{flags}\t{plan[movie]['movie_name']}")
        return
    for movie in args.movies or []:
        if movie not in plan:
            print(f"{movie}: khong co trong plan.json — bo qua")
            continue
        info = write_movie(movie, args.out_root)
        video = find_video(plan[movie])
        tagged = find_tagged(movie)
        print(f"{movie}: {info['n_lines']} dong; "
              f"video={'co' if video else 'THIEU'}; "
              f"tagged={'co' if tagged else 'THIEU'}")


if __name__ == "__main__":
    main()
