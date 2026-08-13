"""Micro-averaged Pronoun F1 giua hypothesis va gold (multiset per line)."""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class PronounScores:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def line_counts(gold: list[str], pred: list[str]) -> tuple[int, int, int]:
    g, p = Counter(gold), Counter(pred)
    tp = sum((g & p).values())
    return tp, sum(p.values()) - tp, sum(g.values()) - tp


def corpus_pronoun_f1(pairs: list[tuple[list[str], list[str]]]) -> PronounScores:
    tp = fp = fn = 0
    for gold, pred in pairs:
        t, f_p, f_n = line_counts(gold, pred)
        tp, fp, fn = tp + t, fp + f_p, fn + f_n
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PronounScores(precision, recall, f1, tp, fp, fn)
