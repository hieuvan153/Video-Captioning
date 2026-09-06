"""Paired bootstrap (Koehn 2004) tren don vi CANH cho mot phim le khong co scene GT.

Canh lay tu debug JSON cua refine_llm.py (scene_index -> subtitle_index).
Tham chieu tung cue = cac cue cua phu de nguoi dich giao thoi gian voi cue do
(cung ham gold_src_lines nhu film_score.py), khu trung lap trong cung canh.

CLI: van_env/bin/python demo/EVAL/film_bootstrap.py --ref vi.srt --en en.srt \
        --debug debug.json --a rough.srt --b refined.srt [--n 2000]
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

from EVAL.film_score import gold_src_lines  # noqa: E402
from EVAL.thesis_score import pronoun_sets_scene  # noqa: E402


def lines(p: str) -> list[str]:
    with open(p, encoding="utf-8-sig", errors="replace") as f:
        return [re.sub(r"\s+", " ", s.content).strip() for s in srt.parse(f.read())]


def scenes(debug: str) -> list[list[int]]:
    with open(debug, encoding="utf-8") as f:
        return [[t["subtitle_index"] - 1 for t in sc["translations"]] for sc in json.load(f)]


def join_ref(ref_cue: list[str | None], idx: list[int]) -> str:
    out: list[str] = []
    for i in idx:                      # mot cue tham chieu co the trai nhieu cue cua ta -> khu lap
        t = ref_cue[i] if i < len(ref_cue) else None
        if t and (not out or out[-1] != t):
            out.append(t)
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--en", required=True)
    ap.add_argument("--debug", required=True)
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()

    ref_cue = gold_src_lines(a.en, a.ref, 1.0, 0.0)
    la, lb = lines(a.a), lines(a.b)
    pairs = []
    for idx in scenes(a.debug):
        r = join_ref(ref_cue, idx)
        if not r:
            continue
        pairs.append((" ".join(la[i] for i in idx if i < len(la)),
                      " ".join(lb[i] for i in idx if i < len(lb)), r))

    def bleu(tr): return sacrebleu.corpus_bleu([h for _, h, _ in tr], [[r for _, _, r in tr]]).score
    def chrf(tr): return sacrebleu.corpus_chrf([h for _, h, _ in tr], [[r for _, _, r in tr]]).score

    def pron_f1(tr):
        tp = fp = fn = 0
        for _, h, r in tr:
            a, b, c = pronoun_sets_scene(h, r); tp += a; fp += b; fn += c
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        return 100 * 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0

    for name, fn in (("BLEU", bleu), ("chrF", chrf), ("PronF1x100", pron_f1)):
        A = [(x, x, r) for x, _, r in pairs]; B = [(y, y, r) for _, y, r in pairs]
        d0 = fn(B) - fn(A); random.seed(0); ds = []; wins = 0
        for _ in range(a.n):
            idx = [random.randrange(len(pairs)) for _ in pairs]
            d = fn([B[i] for i in idx]) - fn([A[i] for i in idx]); ds.append(d); wins += d > 0
        ds.sort()
        print(f"{name}: b-a = {d0:+.2f}  95%CI=[{ds[int(.025*a.n)]:+.2f}, {ds[int(.975*a.n)]:+.2f}]"
              f"  p(b<=a)={1-wins/a.n:.3f}  n_scenes={len(pairs)}", flush=True)


if __name__ == "__main__":
    main()
