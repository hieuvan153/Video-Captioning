"""WER cua ASR cho MOT phim le, dung dung chuan hoa cua docs/eval/audit_2026-09-02/asr_wer_e5.py
(bo nhan NGUOI NOI:, (am thanh), [..], the <i>, ky hieu nhac, lowercase, bo dau cau) de con so
so truc tiep duoc voi bang WER cua tap E5.

CLI: van_env/bin/python demo/EVAL/film_wer.py --hyp asr.srt --ref en_chuan.srt --name Ten_Phim
"""
from __future__ import annotations

import argparse
import re

import jiwer
import srt


def norm(t: str) -> str:      # giong het norm() trong asr_wer_e5.py
    t = re.sub(r'<[^>]+>', '', t); t = re.sub(r'\([^)]*\)', '', t); t = re.sub(r'\[[^\]]*\]', '', t)
    t = re.sub(r'\b[A-Z][A-Z .\']{1,30}:', '', t); t = t.replace('♪', '').lower()
    t = re.sub(r"[^a-z0-9' ]+", ' ', t); return re.sub(r'\s+', ' ', t).strip()


def text(path: str) -> str:
    with open(path, encoding='utf-8-sig') as f:
        return re.sub(r'\s+', ' ', ' '.join(norm(s.content) for s in srt.parse(f.read()))).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hyp', required=True); ap.add_argument('--ref', required=True)
    ap.add_argument('--name', default='PHIM')
    a = ap.parse_args()
    ref, hyp = text(a.ref), text(a.hyp)
    o = jiwer.process_words(ref, hyp)
    n = o.substitutions + o.deletions + o.hits          # so tu tham chieu
    row = (100 * (o.substitutions + o.deletions + o.insertions) / n,
           100 * o.substitutions / n, 100 * o.deletions / n, 100 * o.insertions / n)
    print(f"{'PHIM':22} {'WER':>7} {'THAY':>7} {'BO SOT':>8} {'THEM':>7}")
    print(f"{a.name:22} {row[0]:7.2f} {row[1]:7.2f} {row[2]:8.2f} {row[3]:7.2f}")
    print("-" * 56)
    print(f"{'TONG':22} {row[0]:7.2f} {row[1]:7.2f} {row[2]:8.2f} {row[3]:7.2f}")
    tot = o.substitutions + o.deletions + o.insertions
    print(f"\n  ty trong: thay {100*o.substitutions/tot:.1f}%  bo sot {100*o.deletions/tot:.1f}%"
          f"  them {100*o.insertions/tot:.1f}%")
    print(f"  tu tham chieu {n}, tu ASR {len(hyp.split())}")


if __name__ == '__main__':
    main()
