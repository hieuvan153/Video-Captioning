"""Cham COMET cho cac arm SRT tren eval set — bo sung chieu do cua bang E1-E5.

Model: Unbabel/wmt22-comet-da (reference-based, nhu so lieu COMET trong bang
thi nghiem cua anh ntVan). Load MOT lan, cham moi movie x arm; diem he thong
= trung binh diem cau. Report JSON de compare/aggregate ve sau.

CLI:
    van_env/bin/python demo/EVAL/comet_score.py \
        --arms rough.srt v1b_baseline.srt v1b_speaker.srt \
        --report docs/eval/comet_v1b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, "cache", "huggingface"))

from EVAL.run_eval import load_srt_lines  # noqa: E402

MODEL = "Unbabel/wmt22-comet-da"


def main() -> None:
    parser = argparse.ArgumentParser(description="COMET score SRT arms.")
    parser.add_argument("--eval_dir", type=str,
                        default=os.path.join(ROOT_DIR, "output", "eval_ab"))
    parser.add_argument("--movies", type=str, nargs="*", default=None)
    parser.add_argument("--arms", type=str, nargs="+", required=True,
                        help="Ten file SRT trong tung thu muc phim.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    eval_dir = os.path.abspath(args.eval_dir)
    movies = args.movies or sorted(
        d for d in os.listdir(eval_dir)
        if os.path.isdir(os.path.join(eval_dir, d)))

    # Gom het thanh MOT batch predict cho do phai load model nhieu lan.
    data: list[dict] = []
    index: list[tuple[str, str]] = []       # (movie, arm) per segment
    for movie in movies:
        mdir = os.path.join(eval_dir, movie)
        try:
            src = load_srt_lines(os.path.join(mdir, "en.srt"))
            ref = load_srt_lines(os.path.join(mdir, "ref.srt"))
        except FileNotFoundError:
            print(f"{movie}: thieu en/ref — bo qua", flush=True)
            continue
        for arm in args.arms:
            path = os.path.join(mdir, arm)
            if not os.path.exists(path):
                continue
            hyp = load_srt_lines(path)
            if len(hyp) != len(ref):
                print(f"{movie}/{arm}: lech so dong ({len(hyp)} vs "
                      f"{len(ref)}) — bo qua", flush=True)
                continue
            data.extend({"src": s, "mt": h, "ref": r}
                        for s, h, r in zip(src, hyp, ref))
            index.extend((movie, arm) for _ in ref)
    if not data:
        raise SystemExit("Khong co du lieu de cham.")

    from comet import download_model, load_from_checkpoint
    import torch

    ckpt = download_model(MODEL)
    model = load_from_checkpoint(ckpt)
    gpus = 1 if torch.cuda.is_available() else 0
    out = model.predict(data, batch_size=args.batch_size, gpus=gpus,
                        progress_bar=True)

    agg: dict[tuple[str, str], list[float]] = {}
    for (movie, arm), score in zip(index, out.scores):
        agg.setdefault((movie, arm), []).append(float(score))
    report = {"model": MODEL, "movies": {}, "overall": {}}
    for (movie, arm), scores in sorted(agg.items()):
        report["movies"].setdefault(movie, {})[arm] = {
            "comet": round(sum(scores) / len(scores), 4),
            "n_lines": len(scores),
        }
    for arm in args.arms:
        all_scores = [s for (m, a), ss in agg.items() if a == arm for s in ss]
        if all_scores:
            report["overall"][arm] = {
                "comet": round(sum(all_scores) / len(all_scores), 4),
                "n_lines": len(all_scores),
            }
    for movie, arms in report["movies"].items():
        row = "  ".join(f"{a}={v['comet']:.4f}" for a, v in arms.items())
        print(f"{movie}: {row}", flush=True)
    for arm, v in report["overall"].items():
        print(f"overall {arm}: {v['comet']:.4f} ({v['n_lines']} dong)",
              flush=True)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
