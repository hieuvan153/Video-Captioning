"""Eval harness: BLEU / chrF / Pronoun F1.

Mode A (so 2 file SRT, gold pronoun rut tu ref bang lexicon hoac tu dataset json):
    python demo/EVAL/run_eval.py --hyp_srt hyp.srt --ref_srt ref.srt [--report out.json]

Mode B (calibration tren dataset da gan nhan — khong can GPU):
    python demo/EVAL/run_eval.py --dataset_dir data/en-vi-speaker-with-time-pronouns \
        --hyp_field vietsub_raw [--report out.json]
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sacrebleu
import srt

from EVAL.pronoun_f1 import corpus_pronoun_f1
from EVAL.pronoun_lexicon import PRONOUN_TERMS, extract_pronouns, parse_gold_pronouns


def load_srt(path: str) -> list:
    """Doc SRT, normalize whitespace trong content (giu style pipeline)."""
    with open(path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))
    for s in subs:
        s.content = re.sub(
            r"\s+", " ",
            " ".join(l.strip() for l in s.content.splitlines()),
        ).strip()
    return subs


def load_srt_lines(path: str) -> list[str]:
    return [s.content for s in load_srt(path)]


def align_by_time(hyp_subs: list, ref_subs: list) -> list[tuple[str, str]]:
    """Voi moi ref line, noi cac hyp line co overlap thoi gian > 0.

    Dung khi so dong hyp (do ASR cat) khac so dong reference.
    """
    pairs: list[tuple[str, str]] = []
    for r in ref_subs:
        r_start, r_end = r.start.total_seconds(), r.end.total_seconds()
        parts = [
            h.content for h in hyp_subs
            if min(r_end, h.end.total_seconds())
            - max(r_start, h.start.total_seconds()) > 0
        ]
        pairs.append((" ".join(parts).strip(), r.content))
    return pairs


def evaluate_lines(
    hyps: list[str],
    refs: list[str],
    gold_pronouns: list[list[str]] | None,
) -> dict:
    if len(hyps) != len(refs):
        raise ValueError(f"hyp/ref length mismatch: {len(hyps)} vs {len(refs)}")
    if gold_pronouns is None:
        gold_pronouns = [extract_pronouns(r) for r in refs]
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    scores = corpus_pronoun_f1(
        [(g, extract_pronouns(h)) for g, h in zip(gold_pronouns, hyps)]
    )
    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "pronoun_precision": round(scores.precision, 4),
        "pronoun_recall": round(scores.recall, 4),
        "pronoun_f1": round(scores.f1, 4),
        "n_lines": len(hyps),
    }


def eval_dataset_dir(dataset_dir: str, hyp_field: str) -> dict:
    hyps: list[str] = []
    refs: list[str] = []
    gold: list[list[str]] = []
    missing_terms: dict[str, int] = {}
    for path in sorted(glob.glob(os.path.join(dataset_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            hyp, ref = r.get(hyp_field), r.get("vietnamese")
            if not hyp or not ref:
                continue
            hyps.append(hyp)
            refs.append(ref)
            g = parse_gold_pronouns(r)
            gold.append(g)
            for t in g:
                if t not in PRONOUN_TERMS:
                    missing_terms[t] = missing_terms.get(t, 0) + 1
    report = evaluate_lines(hyps, refs, gold)
    report["lexicon_missing_terms"] = dict(
        sorted(missing_terms.items(), key=lambda kv: -kv[1])[:30]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Subtitle eval: BLEU/chrF/Pronoun F1")
    parser.add_argument("--hyp_srt", type=str)
    parser.add_argument("--ref_srt", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--hyp_field", type=str, default="vietsub_raw")
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    if args.dataset_dir:
        report = eval_dataset_dir(args.dataset_dir, args.hyp_field)
    elif args.hyp_srt and args.ref_srt:
        hyp_subs, ref_subs = load_srt(args.hyp_srt), load_srt(args.ref_srt)
        if len(hyp_subs) == len(ref_subs):
            pairs = [(h.content, r.content)
                     for h, r in zip(hyp_subs, ref_subs)]
        else:
            print(f"line counts differ ({len(hyp_subs)} vs {len(ref_subs)}); "
                  f"aligning by time overlap", flush=True)
            pairs = align_by_time(hyp_subs, ref_subs)
        report = evaluate_lines(
            [p[0] for p in pairs], [p[1] for p in pairs], None
        )
    else:
        parser.error("need either --dataset_dir or (--hyp_srt and --ref_srt)")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        tmp = args.report + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        os.replace(tmp, args.report)


if __name__ == "__main__":
    main()
