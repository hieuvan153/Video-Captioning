"""BLEU ca tap theo DUNG giao thuc Bang 4.7 — chay lai duoc tren nhieu thu muc.

Tai su dung truc tiep ASR/calculate_bleu.py cua anh ntVan (khong copy logic):
noi toan bo dong SRT cua mot tap thanh 1 chuoi, reference la phu de chinh thuc
voi buoc toi uu hoan vi cac dong trung timestamp, roi corpus BLEU tren 10 tap.
Cot **raw** la cot bao cao trong bang (da doi chieu: E1 38.23 / E2 37.83 /
E3 23.33 khop chinh xac — xem docs/eval/thesis_repro.md).

Ten file hypothesis phai la "<ten_tap>.(Tiếng Việt).srt" de khop phu de goc.

CLI:
    van_env/bin/python demo/EVAL/bleu_episode.py <dir> [<dir> ...]
"""
from __future__ import annotations

import glob
import os
import sys
import unicodedata

sys.path.insert(0, "/data/ndloc_bk/ntVan/ASR")

import sacrebleu  # noqa: E402
from calculate_bleu import (  # noqa: E402
    clean_custom,
    clean_no_punc,
    get_optimized_reference_text,
    get_subs_from_srt,
    get_text_from_srt,
)

SUB_DIR = "/data/ndloc_bk/ntVan/data/Movie/sub"


def score_dir(pred_dir: str, subs: dict[str, str]) -> dict[str, float]:
    cfg = {"raw": (lambda x: x, []), "nopunc": (clean_no_punc, []),
           "custom": (clean_custom, [])}
    pairs = {k: ([], []) for k in cfg}
    n = 0
    for pf in sorted(glob.glob(os.path.join(pred_dir, "*.srt"))):
        name = unicodedata.normalize("NFC", os.path.basename(pf))
        tgt = name.replace(".(Tiếng Việt).srt", ".Tiếng_Việt.srt")
        if tgt not in subs:
            print(f"  THIEU phu de goc cho {name}", flush=True)
            continue
        ref_subs = get_subs_from_srt(subs[tgt])
        raw = get_text_from_srt(pf)
        for key, (fn, _) in cfg.items():
            pred = fn(raw)
            pairs[key][0].append(pred)
            pairs[key][1].append(get_optimized_reference_text(ref_subs, pred, fn))
        n += 1
    out = {k: sacrebleu.corpus_bleu(h, [r]).score for k, (h, r) in pairs.items()}
    out["n"] = n
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    subs = {unicodedata.normalize("NFC", os.path.basename(s)): s
            for s in glob.glob(os.path.join(SUB_DIR, "*.srt"))}
    for d in sys.argv[1:]:
        r = score_dir(d, subs)
        print(f"{os.path.basename(os.path.normpath(d)):45s} n={r['n']:2d}  "
              f"raw {r['raw']:.2f}  nopunc {r['nopunc']:.2f}  "
              f"custom {r['custom']:.2f}", flush=True)


if __name__ == "__main__":
    main()
