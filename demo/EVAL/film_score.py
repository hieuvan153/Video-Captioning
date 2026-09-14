"""Cham mot phim le (khong co scene GT) cho nhieu arm SRT.

- BLEU: dung giao thuc Bang 4.7 (EVAL/bleu47.py, chep tu calculate_bleu.py cua anh ntVan): noi toan bo
  dong hyp thanh 1 chuoi, reference toi uu hoan vi dong trung timestamp,
  corpus BLEU tren 1 "tap". Cot raw / nopunc / custom nhu bleu_episode.py.
- Pronoun F1: multiset tren TOAN PHIM voi lexicon cua thesis_score.py.
- COMET-QE (khong tham chieu, Unbabel/wmt20-comet-qe-da): src = dong EN cung
  chi so (ASR), mt = dong hyp; trung binh dong. --no_comet de bo qua.
- junk: ti le dong khong co chu tieng Viet / refusal / chi dau cau.

Reference duoc lam sach: bo the <i>, bo dong chi co [am thanh], bo quang cao.

CLI:
    van_env/bin/python demo/EVAL/film_score.py --ref ref_vi.srt --en en.srt \
        --arms rough=path.srt bs10=path.srt [--no_comet] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sacrebleu  # noqa: E402
import srt  # noqa: E402
from EVAL.bleu47 import (  # noqa: E402
    clean_custom, clean_no_punc, get_optimized_reference_text,
    get_subs_from_srt, get_text_from_srt,
)
from EVAL.thesis_score import pronoun_sets_scene  # noqa: E402

VI = re.compile(r"[àáảãạăâđèéẻẽẹêìíỉĩịòóỏõọôơùúủũụưỳýỷỹỵ]")
TAG = re.compile(r"</?\s*[a-z]+\s*>", re.I)
BRACKET = re.compile(r"\[[^\]]*\]|\([^)]*\)")
AD = re.compile(r"opensubtitles|advertise|subtitlecat|www\.", re.I)


def load(path: str) -> list[srt.Subtitle]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return list(srt.parse(unicodedata.normalize("NFC", f.read())))


def clean_ref(path: str) -> str:
    """Tra ve duong dan SRT tam da lam sach (giu timestamp de toi uu hoan vi)."""
    out = []
    for s in load(path):
        t = BRACKET.sub(" ", TAG.sub(" ", s.content))
        t = re.sub(r"\s+", " ", t).strip(" -♪")
        if not t or AD.search(t):
            continue
        out.append(srt.Subtitle(index=len(out) + 1, start=s.start, end=s.end, content=t))
    fd, p = tempfile.mkstemp(suffix=".srt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(srt.compose(out))
    return p


def is_junk(s: str) -> bool:
    s = s.strip()
    return (not VI.search(s.lower())) or s.lower().startswith("i'm sorry") or s in {",", "-", ""}


def gold_src_lines(en_path: str, gold_path: str, a: float, b: float) -> list[str | None]:
    """src vang cho tung cue cua ta: noi cac cue gold giao voi [a*start+b, a*end+b]."""
    gold = load(gold_path)
    gs = [(g.start.total_seconds(), g.end.total_seconds(), re.sub(r"\s+", " ", g.content).strip()) for g in gold]
    out: list[str | None] = []
    for s in load(en_path):
        lo, hi = a * s.start.total_seconds() + b, a * s.end.total_seconds() + b
        hit = [t for (g0, g1, t) in gs if g0 < hi and g1 > lo]
        out.append(" ".join(hit) if hit else None)
    return out


def score(ref_path: str, en_path: str, arms: dict[str, str], use_comet: bool,
          gold_en: str | None = None, tmap: tuple[float, float] = (1.0, 0.0),
          use_comet_da: bool = False) -> dict:
    ref_clean = clean_ref(ref_path)
    ref_subs = get_subs_from_srt(ref_clean)
    ref_text = " ".join(s.content for s in load(ref_clean))
    en_subs = load(en_path)
    en_lines = [re.sub(r"\s+", " ", s.content).strip() for s in en_subs]
    if use_comet or use_comet_da:
        # src/ref cua COMET ghep theo CHI SO cue cua --en: arm khac luoi la lech dong ma khong bao loi
        # (--en = EN nguoi 1935 cue vs arm luoi ASR 1939 cue: chi 85 cap trung thoi gian -> COMET-DA 0,468).
        grid = [(s.start, s.end) for s in en_subs]
        bad = [k for k, p in arms.items() if [(s.start, s.end) for s in load(p)] != grid]
        if bad:
            raise SystemExit(f"COMET: arm {bad} khac luoi cue voi --en {en_path}. "
                             "Truyen --en = SRT EN cung luoi voi arm (SRT ASR cua pipeline).")
    gold_lines = gold_src_lines(en_path, gold_en, *tmap) if gold_en else None
    cfg = {"raw": lambda x: x, "nopunc": clean_no_punc, "custom": clean_custom}
    comet_model = da_model = ref_lines = None
    if use_comet or use_comet_da:
        from comet import download_model, load_from_checkpoint
    if use_comet:
        comet_model = load_from_checkpoint(download_model("Unbabel/wmt20-comet-qe-da"))
    if use_comet_da:   # COMET co tham chieu: ref = cue tham chieu giao thoi gian voi cue cua ta
        da_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        ref_lines = gold_src_lines(en_path, ref_clean, 1.0, 0.0)   # ref da bo the/[am thanh]/quang cao, nhu BLEU
    rep = {}
    for arm, hp in arms.items():
        hyp_raw = get_text_from_srt(hp)
        row = {}
        for k, fn in cfg.items():
            pred = fn(hyp_raw)
            ref = get_optimized_reference_text(ref_subs, pred, fn)
            row[f"bleu_{k}"] = round(sacrebleu.corpus_bleu([pred], [[ref]]).score, 2)
        hyp_lines = [re.sub(r"\s+", " ", s.content).strip() for s in load(hp)]
        tp, fp, fn_ = pronoun_sets_scene(" ".join(hyp_lines), ref_text)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn_) if tp + fn_ else 0.0
        row["pronoun_p"], row["pronoun_r"] = round(p, 4), round(r, 4)
        row["pronoun_f1"] = round(2 * p * r / (p + r), 4) if p + r else 0.0
        row["junk_rate"] = round(sum(map(is_junk, hyp_lines)) / len(hyp_lines), 4)
        row["n_lines"] = len(hyp_lines)
        if comet_model is not None:
            n = min(len(en_lines), len(hyp_lines))
            data = [{"src": en_lines[i], "mt": hyp_lines[i]} for i in range(n)]
            out = comet_model.predict(data, batch_size=32, gpus=1 if _cuda() else 0, progress_bar=False)
            row["comet_qe"] = round(sum(out.scores) / len(out.scores), 4)
            if gold_lines is not None:   # src = phu de EN chuan (giong thoi gian), bo dong khong khop
                data = [{"src": gold_lines[i], "mt": hyp_lines[i]} for i in range(n) if gold_lines[i]]
                out = comet_model.predict(data, batch_size=32, gpus=1 if _cuda() else 0, progress_bar=False)
                row["comet_qe_goldsrc"] = round(sum(out.scores) / len(out.scores), 4)
                row["n_goldsrc"] = len(data)
        if da_model is not None:
            n = min(len(en_lines), len(hyp_lines))
            data = [{"src": en_lines[i], "mt": hyp_lines[i], "ref": ref_lines[i]} for i in range(n) if ref_lines[i]]
            out = da_model.predict(data, batch_size=32, gpus=1 if _cuda() else 0, progress_bar=False)
            row["comet_da"] = round(sum(out.scores) / len(out.scores), 4)
            row["n_comet_da"] = len(data)
        rep[arm] = row
        print(f"{arm:10s} BLEU raw {row['bleu_raw']:6.2f} nopunc {row['bleu_nopunc']:6.2f} custom {row['bleu_custom']:6.2f}"
              f"  P {row['pronoun_p']:.3f} R {row['pronoun_r']:.3f} F1 {row['pronoun_f1']:.3f}"
              f"  junk {row['junk_rate']:.3f}" + (f"  COMET-QE {row['comet_qe']:.4f}" if 'comet_qe' in row else "")
              + (f"  COMET-QE(goldsrc) {row['comet_qe_goldsrc']:.4f} n={row['n_goldsrc']}" if 'comet_qe_goldsrc' in row else "")
              + (f"  COMET-DA {row['comet_da']:.4f} n={row['n_comet_da']}" if 'comet_da' in row else ""),
              flush=True)
    os.unlink(ref_clean)
    return rep


def _cuda() -> bool:
    if os.environ.get("FILM_SCORE_CPU"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--en", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="name=path.srt")
    ap.add_argument("--no_comet", action="store_true")
    ap.add_argument("--comet_da", action="store_true", help="them COMET co tham chieu (wmt22-comet-da), ref gióng theo thoi gian")
    ap.add_argument("--gold_en", default=None, help="phu de EN chuan (src vang cho COMET-QE)")
    ap.add_argument("--tmap", default="1,0", help="a,b: gold_time = a*our_time + b")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    arms = dict(x.split("=", 1) for x in a.arms)
    ta, tb = (float(v) for v in a.tmap.split(","))
    rep = score(a.ref, a.en, arms, not a.no_comet, a.gold_en, (ta, tb), a.comet_da)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump({"ref": a.ref, "en": a.en, "arms": arms, "scores": rep}, f, ensure_ascii=False, indent=1)
        print("Report:", a.report)


if __name__ == "__main__":
    main()
