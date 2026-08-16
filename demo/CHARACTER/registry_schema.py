"""Directed character-relationship registry: schema, validation, merge.

Chong poisoning: moi Relation BAT BUOC co evidence_lines hop le trong pham vi
transcript; edge khong evidence / id la / self-loop bi drop ngay khi parse.

Siet them sau error analysis V0 (docs/eval/error_analysis_v0.md):
- Character ma MOI ten deu la dai tu/placeholder tieng Anh ("You", "Me",
  "Him", "Speaker"...) bi drop — khong the map ve speaker that, chi gay nhieu.
- vi_self/vi_listener phai thuoc lexicon xung ho (EVAL.pronoun_lexicon) —
  chan gia tri rac tung sinh ra that: "glenn", "anh/chị", "(addressed as)".
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from EVAL.pronoun_lexicon import PRONOUN_TERMS

CONFIDENCE_LEVELS: dict[str, int] = {"high": 2, "medium": 1, "low": 0}
_GENDERS = {"male", "female", "unknown"}
_AGE_RANGES = {"child", "teen", "adult", "elderly", "unknown"}
PLACEHOLDER_NAMES = frozenset({
    "i", "you", "me", "he", "she", "him", "her", "it", "we", "us",
    "they", "them", "this", "that", "all",
    "everyone", "someone", "anyone", "anybody", "nobody", "no one",
    "speaker", "listener", "narrator", "person", "people", "guy", "dude",
})


@dataclass(frozen=True)
class Character:
    id: str
    names: tuple[str, ...]
    gender: str = "unknown"
    age_range: str = "unknown"
    evidence_lines: tuple[int, ...] = ()


@dataclass(frozen=True)
class Relation:
    from_id: str
    to_id: str
    rel_type: str
    vi_self: str        # speaker tu xung, vd "cháu"
    vi_listener: str    # speaker goi listener, vd "bà"
    confidence: str = "medium"
    evidence_lines: tuple[int, ...] = ()


@dataclass(frozen=True)
class Registry:
    characters: tuple[Character, ...] = ()
    relations: tuple[Relation, ...] = ()


def _valid_lines(value, n_lines: int) -> tuple[int, ...]:
    out = set()
    for x in value or []:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= n_lines:
            out.add(i)
    return tuple(sorted(out))


def parse_registry(raw: dict, n_lines: int) -> Registry:
    """Validate output tho cua LLM (hoac file da luu) thanh Registry sach."""
    characters: list[Character] = []
    ids: set[str] = set()
    for c in raw.get("characters") or []:
        cid = str(c.get("id", "")).strip()
        # Ten thuan so ("1217" — ma cua hang) hoac 1 ky tu ("G") khong dinh
        # danh duoc ai; te hon, chung lot vao tap dong cua cluster_map va
        # filter_registry_by_source ("G" substring-match moi noi).
        names = tuple(
            str(n).strip() for n in (c.get("names") or [])
            if str(n).strip() and not str(n).strip().isdigit()
            and len(str(n).strip()) >= 2
        )
        if not cid or not names or cid in ids:
            continue
        if all(n.casefold() in PLACEHOLDER_NAMES for n in names):
            continue
        ids.add(cid)
        characters.append(Character(
            id=cid,
            names=names,
            gender=c.get("gender") if c.get("gender") in _GENDERS else "unknown",
            age_range=(
                c.get("age_range") if c.get("age_range") in _AGE_RANGES
                else "unknown"
            ),
            evidence_lines=_valid_lines(c.get("evidence_lines"), n_lines),
        ))

    relations: list[Relation] = []
    seen: set[tuple[str, str]] = set()
    for r in raw.get("relations") or []:
        from_id = str(r.get("from_id", "")).strip()
        to_id = str(r.get("to_id", "")).strip()
        vi_self = str(r.get("vi_self", "")).strip().lower()
        vi_listener = str(r.get("vi_listener", "")).strip().lower()
        evidence = _valid_lines(r.get("evidence_lines"), n_lines)
        if (from_id not in ids or to_id not in ids or from_id == to_id
                or vi_self not in PRONOUN_TERMS
                or vi_listener not in PRONOUN_TERMS
                or not evidence or (from_id, to_id) in seen):
            continue
        seen.add((from_id, to_id))
        confidence = (
            r.get("confidence") if r.get("confidence") in CONFIDENCE_LEVELS
            else "low"
        )
        relations.append(Relation(
            from_id=from_id,
            to_id=to_id,
            rel_type=str(r.get("rel_type", "unknown")).strip() or "unknown",
            vi_self=vi_self,
            vi_listener=vi_listener,
            confidence=confidence,
            evidence_lines=evidence,
        ))
    return Registry(tuple(characters), tuple(relations))


def merge_registries(regs: list[Registry]) -> Registry:
    """Gop registry tu nhieu chunk: nhan vat trung alias (casefold) gop lam 1,
    relation trung (from, to) giu confidence cao nhat, evidence duoc union."""
    entries = [(ri, c) for ri, r in enumerate(regs) for c in r.characters]
    parent = list(range(len(entries)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    alias_owner: dict[str, int] = {}
    for i, (_, c) in enumerate(entries):
        for name in c.names:
            key = name.casefold()
            if key in alias_owner:
                union(i, alias_owner[key])
            else:
                alias_owner[key] = i

    groups: dict[int, list[int]] = {}
    for i in range(len(entries)):
        groups.setdefault(find(i), []).append(i)

    id_map: dict[tuple[int, str], str] = {}
    merged_chars: list[Character] = []
    for gi, (_, members) in enumerate(sorted(groups.items()), start=1):
        new_id = f"C{gi}"
        names: list[str] = []
        evidence: set[int] = set()
        gender = age_range = "unknown"
        for m in members:
            ri, c = entries[m]
            id_map[(ri, c.id)] = new_id
            for n in c.names:
                if n.casefold() not in {x.casefold() for x in names}:
                    names.append(n)
            evidence.update(c.evidence_lines)
            if gender == "unknown":
                gender = c.gender
            if age_range == "unknown":
                age_range = c.age_range
        merged_chars.append(Character(
            new_id, tuple(names), gender, age_range, tuple(sorted(evidence))
        ))

    best: dict[tuple[str, str], Relation] = {}
    for ri, reg in enumerate(regs):
        for rel in reg.relations:
            f = id_map.get((ri, rel.from_id))
            t = id_map.get((ri, rel.to_id))
            if f is None or t is None or f == t:
                continue
            cur = best.get((f, t))
            evidence = tuple(sorted(
                set(rel.evidence_lines)
                | set(cur.evidence_lines if cur else ())
            ))
            if (cur is None
                    or CONFIDENCE_LEVELS[rel.confidence]
                    > CONFIDENCE_LEVELS[cur.confidence]):
                best[(f, t)] = Relation(
                    f, t, rel.rel_type, rel.vi_self, rel.vi_listener,
                    rel.confidence, evidence,
                )
            else:
                best[(f, t)] = Relation(
                    cur.from_id, cur.to_id, cur.rel_type, cur.vi_self,
                    cur.vi_listener, cur.confidence, evidence,
                )
    relations = tuple(
        best[k] for k in sorted(best.keys())
    )
    return Registry(tuple(merged_chars), relations)


def filter_registry_by_source(reg: Registry, source_text: str) -> Registry:
    """Drop nhan vat khong co ten nao xuat hien trong source (transcript+captions).

    Chong few-shot leak: LLM thinh thoang chep nguyen nhan vat cua schema
    example (Meemaw/Sheldon) vao registry cua phim khong lien quan; evidence
    gate khong bat duoc vi so dong bia van nam trong pham vi hop le.
    Relation cham vao nhan vat bi drop cung bi drop theo.
    """
    haystack = source_text.casefold()
    kept = tuple(
        c for c in reg.characters
        if any(n.casefold() in haystack for n in c.names)
    )
    kept_ids = {c.id for c in kept}
    relations = tuple(
        r for r in reg.relations
        if r.from_id in kept_ids and r.to_id in kept_ids
    )
    return Registry(kept, relations)


def registry_to_json(reg: Registry) -> dict:
    return {
        "characters": [
            {"id": c.id, "names": list(c.names), "gender": c.gender,
             "age_range": c.age_range, "evidence_lines": list(c.evidence_lines)}
            for c in reg.characters
        ],
        "relations": [
            {"from_id": r.from_id, "to_id": r.to_id, "rel_type": r.rel_type,
             "vi_self": r.vi_self, "vi_listener": r.vi_listener,
             "confidence": r.confidence,
             "evidence_lines": list(r.evidence_lines)}
            for r in reg.relations
        ],
    }


def load_registry(path: str) -> Registry:
    """Doc registry da validate tu disk (evidence da check luc build)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return parse_registry(raw, n_lines=10**9)
