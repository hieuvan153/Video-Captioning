"""Tiem speaker vao prompt refine: tag [SPEAKER: X] phia EN + cap hoi thoai.

Tag dat o phia English vi system prompt cua adapter da noi "Use the English
dialogue only to understand speaker context" — tag nam dung cho no phuc vu.
Quan trong hon: so dong OUTPUT (tieng Viet) khong doi, nen guardrail dem dong
trong refine_subtitles van hoat dong nguyen ven.

Dong khong biet speaker -> KHONG co tag, chu khong phai "[SPEAKER: unknown]":
mot tag rong van la tin hieu model phai dien giai, con thieu tag thi dong do
duoc xu ly y het baseline. Cung nguyen tac "khong chac thi im lang" cua
SPEAKER/align.py va CHARACTER/registry_schema.py.
"""
from __future__ import annotations

import json
import os


def tag_english_lines(
    lines: list[str], names: list[str | None] | None
) -> list[str]:
    """Them '[SPEAKER: X] ' vao dau dong co speaker; dong None giu nguyen."""
    if names is None:
        return list(lines)
    if len(names) != len(lines):
        raise ValueError(
            f"speaker/line mismatch: {len(names)} speaker vs {len(lines)} dong"
        )
    return [
        f"[SPEAKER: {n}] {ln}" if n else ln
        for n, ln in zip(names, lines)
    ]


def turn_pairs(names: list[str | None]) -> set[tuple[str, str]]:
    """Cap thuc su noi chuyen trong scene, suy tu luot thoai ke nhau.

    A noi roi B noi ngay sau -> (A, B) dang doi thoai. Dung de thu hep registry
    ve dung cap co mat, thay vi tiem quan he toan phim vao moi scene
    (docs/eval/error_analysis_v0.md: vocab leak sang dong khong lien quan
    chiem 16/24 FP moi o movie_008, 20/33 o movie_045).

    Dong khong ro speaker CAT mach luot thoai: A -> ? -> B khong dam bao A noi
    voi B, nguoi thu ba co the da xen vao.
    """
    pairs: set[tuple[str, str]] = set()
    prev: str | None = None
    for n in names:
        if not n:
            prev = None
            continue
        if prev and prev != n:
            pairs.add((prev, n) if prev < n else (n, prev))
        prev = n
    return pairs


def speakers_to_json(
    names: list[str | None],
    line_tags: list[str | None],
    mapping: dict[str, str],
    stats=None,
) -> dict:
    """Format file <base>.speakers.json — mot entry moi dong SRT, dung thu tu."""
    payload = {
        "n_lines": len(names),
        "speakers": names,
        "line_tags": line_tags,
        "clusters": dict(sorted(mapping.items())),
        "n_named": sum(1 for n in names if n),
    }
    if stats is not None:
        payload["align_stats"] = {
            "n_lines": stats.n_lines,
            "n_tagged": stats.n_tagged,
            "n_contested": stats.n_contested,
            "n_low_score": stats.n_low_score,
            "coverage": round(stats.coverage, 4),
        }
    return payload


def load_speakers(path: str, n_lines: int) -> list[str | None]:
    """Doc speakers.json, doi chieu do dai voi SRT.

    Lech do dai -> raise chu khong cat/dem: lech 1 dong lam LECH TOAN BO tag ve
    sau, tuc gan sai speaker cho gan het phim — dung failure mode "tag sai con
    te hon khong tag" ma V1 phai tranh.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("speakers")
    if not isinstance(raw, list):
        raise ValueError(f"{path}: thieu truong 'speakers'")
    if len(raw) != n_lines:
        raise ValueError(
            f"{path}: {len(raw)} dong speaker nhung SRT co {n_lines} dong — "
            f"speakers.json duoc build tu file SRT khac?"
        )
    return [n.strip() if isinstance(n, str) and n.strip() else None for n in raw]


def save_speakers(path: str, payload: dict) -> str:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path
