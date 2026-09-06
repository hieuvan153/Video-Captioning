"""Cham cac arm theo DUNG giao thuc do an (Bang 4.7/4.8) tren tap eval_e5.

Don vi cham la SCENE nhu do an (khong phai per-dong nhu run_eval): moi scene
gop cac dong hyp/ref/src roi vao no theo midpoint (LLM/scene_assign — cung
logic gan scene voi refine). BLEU = sacrebleu corpus tren cac scene; COMET =
wmt22-comet-da trung binh scene; dai tu = per-scene, MULTISET voi lexicon
24 dai tu — verbatim code cell 37 infer_nmt.ipynb cua anh ntVan (nguon sinh
Bang 4.8), micro tren toan tap.

Reference: dataset gold (vietnamese, co timing cung truc voi audio) theo
movie_id — khong dung ref cua rieng anh ntVan vi khong con file.

CANH BAO: file nay KHONG phai giao thuc sinh Bang 4.7. Bang 4.7 duoc tai lap
chinh xac bang ASR/calculate_bleu.py (BLEU raw ca tap, reference toi uu hoan
vi timestamp) — xem docs/eval/thesis_repro.md. Dung file nay cho so sanh
scene-level noi bo; muon so truc tiep voi o 40.51 thi dung harness kia.

CLI:
    van_env/bin/python demo/EVAL/thesis_score.py \
        --arms rough.srt v1b_baseline.srt --report docs/eval/thesis_e5.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, "cache", "huggingface"))

from EVAL.run_eval import load_srt_lines  # noqa: E402
from LLM.scene_assign import find_best_scene  # noqa: E402

GT_DIR = os.path.join(os.path.dirname(ROOT_DIR), "data",
                      "en-vi-speaker-with-time-pronouns")


def _mid(a: float, b: float) -> float:
    return (a + b) / 2.0


def scene_texts(mdir: str, movie: str, arm: str):
    """(src, hyp, ref) da gop theo scene — chi scene co ca hyp lan ref."""
    with open(os.path.join(mdir, "captions.json"), encoding="utf-8") as f:
        scenes = json.load(f)
    import srt
    with open(os.path.join(mdir, "en.srt"), encoding="utf-8") as f:
        en_subs = list(srt.parse(f.read()))
    src_lines = load_srt_lines(os.path.join(mdir, "en.srt"))
    hyp_lines = load_srt_lines(os.path.join(mdir, arm))
    if len(hyp_lines) != len(src_lines):
        raise ValueError(f"{movie}/{arm}: {len(hyp_lines)} vs {len(src_lines)}")
    with open(os.path.join(GT_DIR, f"{movie}.json"), encoding="utf-8") as f:
        gold = json.load(f)

    per_scene: dict[int, dict[str, list[str]]] = {}
    for sub, s_text, h_text in zip(en_subs, src_lines, hyp_lines):
        si = find_best_scene(
            _mid(sub.start.total_seconds(), sub.end.total_seconds()), scenes)
        if si == -1:
            continue
        d = per_scene.setdefault(si, {"src": [], "hyp": [], "ref": []})
        d["src"].append(s_text)
        d["hyp"].append(h_text)
    for rec in gold:
        si = find_best_scene(
            _mid(float(rec["start"]), float(rec["end"])), scenes)
        if si == -1:
            continue
        if si in per_scene:
            per_scene[si]["ref"].append((rec.get("vietnamese") or "").strip())
    out = []
    for si in sorted(per_scene):
        d = per_scene[si]
        if d["hyp"] and d["ref"]:
            out.append(("\n".join(d["src"]), "\n".join(d["hyp"]),
                        "\n".join(d["ref"])))
    return out


# Lexicon 24 dai tu + multiset Counter: VERBATIM cell 37 cua
# /data/ndloc_bk/ntVan/infer_nmt.ipynb — code that sinh Bang 4.8 (van ban
# do an viet cong thuc dang tap hop, nhung CODE cua anh ay dung Counter).
# Luu y: giu nguyen ban notebook cell 37 cua anh ntVan — dai tu long nhau bi
# dem trung ("anh ấy" match ca "anh ấy" lan "anh"), nen P/R/F1 tuyet doi bi
# phong; ap dung deu moi arm nen so sanh A/B van cong bang.
VI_PRONOUNS = [
    "anh ấy", "cô ấy", "ông ấy", "bà ấy",
    "chúng tôi", "chúng ta", "chúng nó",
    "tụi nó",
    "tôi", "tao", "mình", "ta",
    "bạn", "cậu", "mày",
    "anh", "chị", "em", "ông", "bà",
    "hắn", "nó", "y", "họ",
]
_VI_PATTERNS = [re.compile(r"\b" + re.escape(t) + r"\b") for t in VI_PRONOUNS]


def _extract_thesis(text: str) -> list[str]:
    text = text.lower()
    found: list[str] = []
    for pat in _VI_PATTERNS:
        found.extend(pat.findall(text))
    return found


def pronoun_sets_scene(hyp: str, ref: str) -> tuple[int, int, int]:
    from collections import Counter
    p, g = Counter(_extract_thesis(hyp)), Counter(_extract_thesis(ref))
    tp = sum((p & g).values())
    return tp, sum(p.values()) - tp, sum(g.values()) - tp


def main() -> None:
    parser = argparse.ArgumentParser(description="Score arms per thesis protocol.")
    parser.add_argument("--eval_dir", type=str,
                        default=os.path.join(ROOT_DIR, "output", "eval_e5"))
    parser.add_argument("--movies", type=str, nargs="*", default=None)
    parser.add_argument("--arms", type=str, nargs="+", required=True)
    parser.add_argument("--no_comet", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    import sacrebleu

    eval_dir = os.path.abspath(args.eval_dir)
    movies = args.movies or sorted(
        d for d in os.listdir(eval_dir)
        if os.path.isdir(os.path.join(eval_dir, d)))

    per_arm: dict[str, dict[str, list]] = {}
    comet_data: list[dict] = []
    comet_index: list[tuple[str, str]] = []
    report = {"protocol": "scene-level (thesis 4.1/4.3.5)", "movies": {},
              "overall": {}}
    for movie in movies:
        mdir = os.path.join(eval_dir, movie)
        for arm in args.arms:
            if not os.path.exists(os.path.join(mdir, arm)):
                continue
            triples = scene_texts(mdir, movie, arm)
            a = per_arm.setdefault(arm, {"hyp": [], "ref": [], "tp_fp_fn":
                                         [0, 0, 0], "by_movie": {}})
            hyps = [h for _, h, _ in triples]
            refs = [r for _, _, r in triples]
            a["hyp"].extend(hyps)
            a["ref"].extend(refs)
            tp = fp = fn = 0
            for _, h, r in triples:
                t, p_, n_ = pronoun_sets_scene(h, r)
                tp, fp, fn = tp + t, fp + p_, fn + n_
            a["tp_fp_fn"][0] += tp
            a["tp_fp_fn"][1] += fp
            a["tp_fp_fn"][2] += fn
            bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
            a["by_movie"][movie] = {"bleu": round(bleu, 2),
                                    "n_scenes": len(triples)}
            comet_data.extend({"src": s, "mt": h, "ref": r}
                              for s, h, r in triples)
            comet_index.extend((movie, arm) for _ in triples)

    comet_by: dict[tuple[str, str], list[float]] = {}
    if comet_data and not args.no_comet:
        from comet import download_model, load_from_checkpoint
        import torch
        model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        out = model.predict(comet_data, batch_size=args.batch_size,
                            gpus=1 if torch.cuda.is_available() else 0,
                            progress_bar=True)
        for key, score in zip(comet_index, out.scores):
            comet_by.setdefault(key, []).append(float(score))

    for arm, a in per_arm.items():
        tp, fp, fn = a["tp_fp_fn"]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        comet_all = [s for (m, ar), ss in comet_by.items()
                     if ar == arm for s in ss]
        entry = {
            "bleu": round(sacrebleu.corpus_bleu(a["hyp"], [a["ref"]]).score, 2),
            "comet": round(sum(comet_all) / len(comet_all), 4)
            if comet_all else None,
            "pronoun_precision": round(prec, 4),
            "pronoun_recall": round(rec, 4),
            "pronoun_f1": round(f1, 4),
            "n_scenes": len(a["hyp"]),
        }
        report["overall"][arm] = entry
        for movie, row in a["by_movie"].items():
            cm = comet_by.get((movie, arm))
            row["comet"] = round(sum(cm) / len(cm), 4) if cm else None
            report["movies"].setdefault(movie, {})[arm] = row
        print(f"{arm}: BLEU {entry['bleu']}  COMET {entry['comet']}  "
              f"P {entry['pronoun_precision']}  R {entry['pronoun_recall']}  "
              f"F1 {entry['pronoun_f1']}  ({entry['n_scenes']} scene)",
              flush=True)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
