"""So sanh nhieu arm A/B tren cung bo phim, kem paired bootstrap per-line.

Vi sao can bootstrap: headline cua A/B V0 (+0.0223 F1) hoa ra do DUNG MOT dong
suy bien o movie_015 sinh ra; bo dong do di thi delta that la −0.0003
(docs/eval/error_analysis_v0.md). Mot con so tong khong noi duoc dieu do, CI
per-line thi noi duoc — nen buoc so sanh phai di kem CI, va script phai nam
trong repo thay vi trong scratchpad nhu lan V0.

Resample theo DONG va giu cap (paired): cung tap dong duoc dung cho moi arm,
nen chenh lech quan sat duoc khong bi nhieu do chon phim/dong khac nhau.

CLI:
    python demo/EVAL/compare_arms.py --eval_dir demo/output/eval_ab \
        --arms baseline=refined_baseline.srt registry=refined_registry.srt \
               speaker=refined_speaker.srt \
        --ref_arm baseline --report docs/eval/speaker_ab_v1.json
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EVAL.pronoun_f1 import corpus_pronoun_f1, line_counts
from EVAL.pronoun_lexicon import extract_pronouns
from EVAL.run_eval import align_by_time, evaluate_lines, load_srt


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def line_stats(hyps: list[str], gold: list[list[str]]) -> list[tuple[int, int, int]]:
    """(tp, fp, fn) cua tung dong — don vi resample cua bootstrap."""
    return [line_counts(g, extract_pronouns(h)) for g, h in zip(gold, hyps)]


def paired_bootstrap(
    stats_a: list[tuple[int, int, int]],
    stats_b: list[tuple[int, int, int]],
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    """CI 95% cua F1(b) - F1(a), resample dong co hoan lai, giu cap a/b.

    p hai phia = ti le lan resample doi dau so voi delta quan sat duoc (uoc
    luong tho: bao nhieu de dang co the thay dau chi vi chon dong khac).
    """
    n = len(stats_a)
    if n == 0:
        return {"delta": 0.0, "ci95": [0.0, 0.0], "p_two_sided": 1.0, "n_lines": 0}
    observed = (_f1(*map(sum, zip(*stats_b)))
                - _f1(*map(sum, zip(*stats_a))))
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        a = [sum(stats_a[i][k] for i in idx) for k in range(3)]
        b = [sum(stats_b[i][k] for i in idx) for k in range(3)]
        deltas.append(_f1(*b) - _f1(*a))
    deltas.sort()
    lo = deltas[int(0.025 * n_resamples)]
    hi = deltas[min(int(0.975 * n_resamples), n_resamples - 1)]
    if observed >= 0:
        p = sum(1 for d in deltas if d <= 0) / n_resamples
    else:
        p = sum(1 for d in deltas if d >= 0) / n_resamples
    return {
        "delta": round(observed, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "p_two_sided": round(min(1.0, 2 * p), 4),
        "n_lines": n,
    }


def _gold_for(movie_dir: str, refs: list[str]) -> list[list[str]]:
    """Gold pronoun tu file da gan nhan neu co, khong thi rut tu ref bang lexicon."""
    path = os.path.join(movie_dir, "gold_pronouns.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            gold = json.load(f)
        if isinstance(gold, list) and len(gold) == len(refs):
            return [list(g) for g in gold]
        print(f"Warning: {path}: {len(gold)} dong vs ref {len(refs)} dong - "
              f"dung lexicon thay the", flush=True)
    return [extract_pronouns(r) for r in refs]


def _aligned_hyps(hyp_path: str, ref_subs: list) -> list[str]:
    hyp_subs = load_srt(hyp_path)
    if len(hyp_subs) == len(ref_subs):
        return [h.content for h in hyp_subs]
    return [h for h, _ in align_by_time(hyp_subs, ref_subs)]


def compare(
    eval_dir: str, arms: dict[str, str], ref_arm: str,
    movies: list[str] | None = None, n_resamples: int = 2000,
) -> dict:
    if movies is None:
        movies = sorted(
            d for d in os.listdir(eval_dir)
            if os.path.isdir(os.path.join(eval_dir, d))
        )
    per_movie: dict[str, dict] = {}
    pooled_stats: dict[str, list] = {a: [] for a in arms}
    pooled_lines: dict[str, list[str]] = {a: [] for a in arms}
    pooled_refs: list[str] = []
    pooled_gold: list[list[str]] = []

    for movie in movies:
        mdir = os.path.join(eval_dir, movie)
        ref_subs = load_srt(os.path.join(mdir, "ref.srt"))
        refs = [r.content for r in ref_subs]
        gold = _gold_for(mdir, refs)

        present = {a: os.path.join(mdir, fn) for a, fn in arms.items()
                   if os.path.exists(os.path.join(mdir, fn))}
        missing = sorted(set(arms) - set(present))
        if missing:
            print(f"Warning: {movie}: thieu arm {missing} - bo qua phim nay", flush=True)
            continue

        entry: dict = {"metrics": {}, "vs_" + ref_arm: {}}
        stats = {}
        for arm, path in present.items():
            hyps = _aligned_hyps(path, ref_subs)
            entry["metrics"][arm] = evaluate_lines(hyps, refs, gold)
            stats[arm] = line_stats(hyps, gold)
            pooled_stats[arm].extend(stats[arm])
            pooled_lines[arm].extend(hyps)
        for arm in present:
            if arm != ref_arm:
                entry["vs_" + ref_arm][arm] = paired_bootstrap(
                    stats[ref_arm], stats[arm], n_resamples
                )
        per_movie[movie] = entry
        pooled_refs.extend(refs)
        pooled_gold.extend(gold)

    overall = {"metrics": {}, "vs_" + ref_arm: {}}
    for arm in arms:
        if pooled_lines[arm]:
            overall["metrics"][arm] = evaluate_lines(
                pooled_lines[arm], pooled_refs, pooled_gold
            )
    for arm in arms:
        if arm != ref_arm and pooled_stats[arm]:
            overall["vs_" + ref_arm][arm] = paired_bootstrap(
                pooled_stats[ref_arm], pooled_stats[arm], n_resamples
            )
    # Gate V1 (+0.03) duoc phat bieu tren TRUNG BINH DELTA THEO PHIM — cung
    # dai luong headline cua A/B V0 — chu khong phai F1 gop toan corpus; hai
    # con so nay lech nhau dang ke (V0: +0.0223 vs +0.0311).
    mean_delta = {}
    for arm in arms:
        if arm == ref_arm:
            continue
        ds = [e["vs_" + ref_arm][arm]["delta"] for e in per_movie.values()
              if arm in e["vs_" + ref_arm]]
        if ds:
            mean_delta[arm] = round(sum(ds) / len(ds), 4)
    overall["mean_delta_per_movie"] = mean_delta
    return {"movies": per_movie, "overall": overall,
            "arms": arms, "ref_arm": ref_arm}


def print_table(report: dict) -> None:
    arms = list(report["arms"])
    ref_arm = report["ref_arm"]
    width = max(len(m) for m in list(report["movies"]) + ["overall"]) + 2
    head = "".join(f"{a:>14}" for a in arms)
    print(f"\n{'movie':<{width}}{head}   Δ vs " + ref_arm)
    print("-" * (width + 14 * len(arms) + 24))
    rows = list(report["movies"].items()) + [("overall", report["overall"])]
    for name, entry in rows:
        cells = "".join(
            f"{entry['metrics'].get(a, {}).get('pronoun_f1', float('nan')):>14.4f}"
            for a in arms
        )
        deltas = "  ".join(
            f"{a} {d['delta']:+.4f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] "
            f"p={d['p_two_sided']:.3f}"
            for a, d in entry["vs_" + ref_arm].items()
        )
        print(f"{name:<{width}}{cells}   {deltas}")
    mean = report["overall"].get("mean_delta_per_movie") or {}
    if mean:
        print("\nTrung binh delta theo phim (dai luong cua gate): "
              + ", ".join(f"{a} {d:+.4f}" for a, d in mean.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-arm subtitle A/B with paired bootstrap on Pronoun F1."
    )
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--arms", type=str, nargs="+", required=True,
                        help="name=filename.srt (relative to each movie dir)")
    parser.add_argument("--ref_arm", type=str, default="baseline")
    parser.add_argument("--movies", type=str, nargs="*", default=None)
    parser.add_argument("--n_resamples", type=int, default=2000)
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    arms = {}
    for spec in args.arms:
        if "=" not in spec:
            parser.error(f"--arms muon dang name=file.srt, nhan duoc {spec!r}")
        name, fn = spec.split("=", 1)
        arms[name] = fn
    if args.ref_arm not in arms:
        parser.error(f"--ref_arm {args.ref_arm!r} khong co trong --arms")

    report = compare(args.eval_dir, arms, args.ref_arm, args.movies,
                     args.n_resamples)
    print_table(report)
    if args.report:
        tmp = args.report + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        os.replace(tmp, args.report)
        print(f"\nSaved: {args.report}")


if __name__ == "__main__":
    main()
