"""Sinh <Scene Context> ORACLE tu phu de THAM CHIEU cua nguoi dich.

Muc dich duy nhat: DO TRAN cua kenh Scene Context truoc khi dau tu vao VLM moi
hay DPO. Voi moi canh, lay cac tu xung ho ma ban dich nguoi that su da dung, nhet
vao muc "3. Relationship". Neu bom thang dap an ma diem khong len co y nghia thi
khong VLM nao / DPO nao cuu duoc kenh nay.

Giu NGUYEN schema 3 muc cua caption VLM: adapter gion voi moi format la, doi schema
la doi phan phoi dau vao chu khong phai doi thong tin.

CLI:
  van_env/bin/python demo/EVAL/make_oracle_context.py \
      --captions out/phim.captions.json --ref ref_3rd/vi_3rd.clean.srt \
      --out out/phim.captions.oracle.json [--top 6]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import srt  # noqa: E402

from EVAL.pronoun_lexicon import extract_pronouns  # noqa: E402


def oracle_line(terms: list[str]) -> str:
    if not terms:
        return "3. Relationship: [None]."
    return ("3. Relationship: [Speakers] - address terms used in this scene: "
            + ", ".join(terms) + ".")


def replace_rel(caption: str, new_line: str) -> str:
    """Thay dong '3. ...' trong caption, giu nguyen cac dong khac. Xu ly caption rong dac biet."""
    # Xu ly caption rong hoac chi toan khoang trang
    if not caption or not caption.strip():
        return "1. Summary: [None].\n2. Main Characters: [None].\n" + new_line

    lines = caption.splitlines()

    # Check if caption has section 1 and 2 (at column 0, no indent)
    has_1 = any(re.match(r"^1\s*\.", line) for line in lines)
    has_2 = any(re.match(r"^2\s*\.", line) for line in lines)

    # If missing BOTH sections 1 and 2 (malformed caption), wrap content in section 1
    if not has_1 and not has_2:
        # Join all original lines with space and wrap in section 1
        original_content = " ".join(lines).strip()
        return f"1. Summary: {original_content}\n2. Main Characters: [None].\n{new_line}"

    # Normal case: find and replace section 3 (no indent allowed)
    # Priority: prefer line with "Relationship" (case-insensitive), else first match
    rel_idx = -1
    first_idx = -1
    for i, line in enumerate(lines):
        if re.match(r"^3\s*\.", line):
            if first_idx == -1:
                first_idx = i
            if "relationship" in line.lower():
                rel_idx = i
                break

    if rel_idx >= 0:
        lines[rel_idx] = new_line
        return "\n".join(lines)
    elif first_idx >= 0:
        lines[first_idx] = new_line
        return "\n".join(lines)

    return "\n".join(lines + [new_line])


def scene_terms(ref_subs, start: float, end: float, top: int) -> list[str]:
    hit = [s.content for s in ref_subs
           if s.start.total_seconds() < end and s.end.total_seconds() > start]
    text = re.sub(r"\s+", " ", " ".join(hit))
    counts = Counter(extract_pronouns(text))
    # top=0 means no cutoff, return all terms
    return [t for t, _ in counts.most_common()] if top == 0 else [t for t, _ in counts.most_common(top)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=0,
                    help="max terms to keep per scene (0=no cutoff, all terms)")
    a = ap.parse_args()

    with open(a.captions, encoding="utf-8") as f:
        scenes = json.load(f)
    with open(a.ref, encoding="utf-8-sig", errors="replace") as f:
        ref_subs = list(srt.parse(f.read()))

    n_filled = 0
    n_valid_schema = 0
    for sc in scenes:
        st, en = sc.get("start_time"), sc.get("end_time")
        if st is None or en is None:
            continue
        terms = scene_terms(ref_subs, float(st), float(en), a.top)
        n_filled += bool(terms)
        sc["caption"] = replace_rel(sc.get("caption", "").strip(), oracle_line(terms))
        # Check if caption has sections 1, 2, 3 in correct order (use FIRST match, no indent)
        cap_lines = sc["caption"].splitlines()
        idx_1 = idx_2 = idx_3 = -1
        count_3_lines = 0  # ponytail: detect multiple unindented "3." lines (invalid)
        for i, line in enumerate(cap_lines):
            if idx_1 == -1 and re.match(r"^1\s*\.", line):
                idx_1 = i
            elif idx_2 == -1 and re.match(r"^2\s*\.", line):
                idx_2 = i
            elif re.match(r"^3\s*\.", line):
                if idx_3 == -1:
                    idx_3 = i
                count_3_lines += 1
        # Valid if: sections 1,2,3 found, in order, and exactly one line 3
        if (idx_1 >= 0 and idx_2 >= 0 and idx_3 >= 0 and idx_1 < idx_2 < idx_3
                and count_3_lines == 1):
            n_valid_schema += 1

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    print(f"{n_filled}/{len(scenes)} canh co tu xung ho oracle -> {a.out}", flush=True)
    print(f"schema 3 muc dung: {n_valid_schema}/{len(scenes)} canh", flush=True)


if __name__ == "__main__":
    main()
