"""Chunking transcript, prompt trich xuat registry, va render block cho refine."""
from __future__ import annotations

import json
import re

from CHARACTER.registry_schema import (
    CONFIDENCE_LEVELS,
    PLACEHOLDER_NAMES,
    Registry,
)

EXTRACTION_SYSTEM = (
    "You are an expert film-script analyst. You read English movie dialogue and "
    "identify the characters and the DIRECTED relationships between them, so that "
    "correct Vietnamese pronouns (xưng hô) can be chosen later.\n"
    "Rules:\n"
    "- Relations are DIRECTED: A->B and B->A are two separate entries. A father "
    "speaking to his son uses different Vietnamese pronouns than the son speaking "
    "to the father.\n"
    "- For each direction give vi_self (how the speaker refers to themself) and "
    "vi_listener (how the speaker addresses the listener), using common Vietnamese "
    "pronoun/kinship terms (tôi, anh, em, chị, ông, bà, bố, mẹ, con, cháu, mày, "
    "tao, cậu, chú, bác, cô, ...).\n"
    "- Only include characters and relations supported by evidence: cite the "
    "dialogue line numbers in evidence_lines. If unsure, use confidence \"low\" "
    "or omit the relation entirely.\n"
    "- Output STRICT JSON only, compact (no pretty-printing). No markdown "
    "fences, no commentary."
)

_SCHEMA_EXAMPLE = {
    "characters": [
        {"id": "C1", "names": ["Meemaw", "Grandma"], "gender": "female",
         "age_range": "elderly", "evidence_lines": [35]},
        {"id": "C2", "names": ["Sheldon"], "gender": "male",
         "age_range": "child", "evidence_lines": [1]},
    ],
    "relations": [
        {"from_id": "C2", "to_id": "C1", "rel_type": "grandchild->grandmother",
         "vi_self": "cháu", "vi_listener": "bà", "confidence": "high",
         "evidence_lines": [35]},
        {"from_id": "C1", "to_id": "C2", "rel_type": "grandmother->grandchild",
         "vi_self": "bà", "vi_listener": "cháu", "confidence": "high",
         "evidence_lines": [35]},
    ],
}


def chunk_lines(
    lines: list[str], chunk_size: int = 200, overlap: int = 30
) -> list[tuple[int, list[str]]]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks: list[tuple[int, list[str]]] = []
    step = chunk_size - overlap
    for start in range(0, len(lines), step):
        chunks.append((start, lines[start:start + chunk_size]))
        if start + chunk_size >= len(lines):
            break
    return chunks


def number_lines(lines: list[str], start: int = 1) -> str:
    return "\n".join(f"{start + i}. {line}" for i, line in enumerate(lines))


def captions_summary(vlm_scenes: list[dict], max_chars: int = 4000) -> str:
    parts = []
    for i, sc in enumerate(vlm_scenes):
        caption = (sc.get("caption") or "").strip()
        # run_vlm.py luu error string vao caption khi crash → loc ra
        if caption and not caption.startswith("Error"):
            parts.append(f"Scene {i + 1}: {caption}")
    return "\n".join(parts)[:max_chars]


def build_extraction_prompt(
    numbered_dialogue: str, captions_text: str
) -> tuple[str, str]:
    user = (
        "<Visual Scene Captions>\n"
        f"{captions_text or 'None'}\n"
        "</Visual Scene Captions>\n\n"
        "<English Dialogue (numbered lines)>\n"
        f"{numbered_dialogue}\n"
        "</English Dialogue>\n\n"
        "Return JSON with EXACTLY this schema (keys: characters[], relations[]; "
        "relation keys: from_id, to_id, rel_type, vi_self, vi_listener, "
        "confidence, evidence_lines):\n"
        f"{json.dumps(_SCHEMA_EXAMPLE, ensure_ascii=False, separators=(',', ':'))}"
    )
    return EXTRACTION_SYSTEM, user


def parse_llm_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# Tu xung ho than toc/co huong — dung cho gate selective activation.
# "tôi"/"ta"/"bạn"... generic KHONG nam trong day: edge ma vi_self generic
# khong mang thong tin xung ho co huong (xem docs/eval/error_analysis_v0.md).
KINSHIP_TERMS: frozenset[str] = frozenset({
    "con", "mẹ", "má", "u", "bố", "ba", "cha", "bà", "ông", "cụ", "cháu",
    "chú", "bác", "cô", "dì", "cậu", "mợ", "thím", "anh", "chị", "em", "thầy",
})


def _display_names(reg: Registry) -> dict[str, str]:
    """id -> ten hien thi. Uu tien ten that: nhan vat merge tu nhieu chunk co
    the mang alias dai tu ("I", "Speaker") dung dau danh sach."""
    return {
        c.id: next(
            (n for n in c.names if n.casefold() not in PLACEHOLDER_NAMES),
            c.names[0],
        )
        for c in reg.characters
    }


def _kinship_pairs(
    reg: Registry, min_confidence: str
) -> dict[tuple[str, str], list]:
    """Gom relation than toc du confidence thanh cap khong huong {(idA, idB): [rel]}."""
    min_rank = CONFIDENCE_LEVELS[min_confidence]
    pairs: dict[tuple[str, str], list] = {}
    for r in reg.relations:
        if (CONFIDENCE_LEVELS[r.confidence] < min_rank
                or r.vi_self not in KINSHIP_TERMS
                or r.vi_listener not in KINSHIP_TERMS):
            continue
        pairs.setdefault(tuple(sorted((r.from_id, r.to_id))), []).append(r)
    return pairs


def _format_pairs(pairs: dict, name_of: dict[str, str], max_pairs: int) -> str:
    def best_rank(rels) -> int:
        return max(CONFIDENCE_LEVELS[r.confidence] for r in rels)

    entries = []
    for key, rels in sorted(pairs.items(), key=lambda kv: -best_rank(kv[1]))[:max_pairs]:
        a, b = key
        hints = "; ".join(
            f'{name_of[r.from_id]} calls {name_of[r.to_id]} "{r.vi_listener}" '
            f'and self "{r.vi_self}"'
            for r in rels
        )
        entries.append(f"{name_of[a]} & {name_of[b]} - {rels[0].rel_type} ({hints})")
    return " | ".join(entries)


def render_registry_context(
    reg: Registry, max_pairs: int = 6, min_confidence: str = "high",
    min_kinship_pairs: int = 2,
) -> str:
    """Render registry thanh 1 dong 'Relationship: ...' de noi VAO Scene Context.

    Adapter refine duoc train voi caption VLM co muc
    "3. Relationship: [A & B] - [Type]" ben trong <Scene Context>; block XML
    dat ngoai section nay lam adapter suy bien (output lap vo nghia), nen
    registry phai duoc dien dat dung style caption, 1 dong prose, khong tag.

    Selective activation (docs/eval/error_analysis_v0.md): CHI edge than toc
    confidence cao (ca vi_self lan vi_listener thuoc KINSHIP_TERMS) duoc
    render, va phai co >= min_kinship_pairs cap nhu vay thi moi tiem — nguoc
    lai tra "" va pipeline chay nhu baseline. Ly do: A/B do duoc edge generic
    (tôi/em, tôi/anh) leak vao sai speaker gay hai (movie_045 −0.036 F1),
    trong khi phim toan edge than toc high huong loi (movie_046 +0.046 F1).

    Day la che do V0 (1 dong chung cho MOI scene). Khi da co speaker per-line,
    dung render_scene_registry_context() thay the.
    """
    if not reg.characters:
        return ""
    pairs = _kinship_pairs(reg, min_confidence)
    if len(pairs) < min_kinship_pairs:
        return ""
    rendered = _format_pairs(pairs, _display_names(reg), max_pairs)
    return f"Relationship (whole-film analysis): {rendered}" if rendered else ""


def _alias_index(reg: Registry) -> dict[str, str]:
    """alias casefold -> character id. Alias trung o 2 nhan vat bi loai: khong
    phan giai duoc thi im lang, con hon gan nham quan he."""
    owner: dict[str, str | None] = {}
    for c in reg.characters:
        for n in c.names:
            key = n.casefold()
            if key in owner and owner[key] != c.id:
                owner[key] = None
            else:
                owner.setdefault(key, c.id)
    return {k: v for k, v in owner.items() if v is not None}


def render_scene_registry_context(
    reg: Registry, speaker_pairs, max_pairs: int = 4,
    min_confidence: str = "high", min_kinship_pairs: int = 2,
) -> str:
    """Render quan he cua RIENG cac cap dang noi chuyen trong scene nay.

    speaker_pairs: set cac cap ten (tu SPEAKER.inject.turn_pairs) that su co
    luot thoai ke nhau trong scene.

    Khac biet co che so voi V0 (docs/eval/error_analysis_v0.md): V0 tiem cung
    mot dong quan he vao MOI scene, nen tu xung ho cua cap nay leak sang dong
    cua cap khac — do duoc la nguon FP chinh (16/24 FP moi o movie_008, 20/33 o
    movie_045). O day moi scene chi thay quan he cua nguoi dang noi trong no.

    Gate min_kinship_pairs van tinh tren CA PHIM, khong phai tren scene: no la
    quyet dinh "phim nay co dang tiem registry khong" da do duoc o V0
    (movie_046 huong loi / movie_045 bi hai). Loc theo scene la buoc sau do.
    """
    if not reg.characters or not speaker_pairs:
        return ""
    pairs = _kinship_pairs(reg, min_confidence)
    if len(pairs) < min_kinship_pairs:
        return ""

    id_of = _alias_index(reg)
    wanted: set[tuple[str, str]] = set()
    for a, b in speaker_pairs:
        ia, ib = id_of.get(a.casefold()), id_of.get(b.casefold())
        if ia and ib and ia != ib:
            wanted.add(tuple(sorted((ia, ib))))
    scoped = {k: v for k, v in pairs.items() if k in wanted}
    if not scoped:
        return ""
    rendered = _format_pairs(scoped, _display_names(reg), max_pairs)
    return f"Relationship (speakers in this scene): {rendered}" if rendered else ""


def render_registry_block(
    reg: Registry, max_relations: int = 24, min_confidence: str = "medium"
) -> str:
    if not reg.characters:
        return ""
    min_rank = CONFIDENCE_LEVELS[min_confidence]
    relations = sorted(
        (r for r in reg.relations
         if CONFIDENCE_LEVELS[r.confidence] >= min_rank),
        key=lambda r: -CONFIDENCE_LEVELS[r.confidence],
    )[:max_relations]
    name_of = {c.id: c.names[0] for c in reg.characters}
    lines = ["<Character Registry>", "Characters:"]
    for c in reg.characters:
        aka = f" (aka {', '.join(c.names[1:])})" if len(c.names) > 1 else ""
        lines.append(f"- {c.names[0]}{aka}: {c.gender}, {c.age_range}")
    if relations:
        lines.append(
            "Directed relations (speaker -> listener; use these Vietnamese "
            "pronouns when this speaker addresses this listener):"
        )
        for r in relations:
            lines.append(
                f'- {name_of[r.from_id]} -> {name_of[r.to_id]} [{r.rel_type}]: '
                f'speaker calls self "{r.vi_self}", '
                f'calls listener "{r.vi_listener}"'
            )
    lines.append("</Character Registry>")
    return "\n".join(lines)
