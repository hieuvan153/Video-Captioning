"""Hop nhat danh tinh nhan vat: 'Sheldon'/'Shelly'/'Moonpie' la MOT nguoi.

Van de do duoc o V1 (docs/eval/speaker_ab_v1.md): registry sinh cung mot nguoi
thanh nhieu nhan vat ('Sheldon'/'Shelly'/'Adult Sheldon', 'Meemaw'/'Connie',
'Dr. Sturgis'/'Dr. John Sturgis', 'Glenn'/'Glennda'), con cluster_map lai chon
mot ten khac voi ten tren canh than toc — hai buoc giai bai toan danh tinh hai
lan doc lap tren hai khong gian ten, nen canh khong bao gio khop va registry
theo scene gan nhu khong kich hoat (6/157 scene).

merge_registries chi gop alias TRUNG NHAU (casefold); o day can gop alias KHAC
NHAU cung chi mot nguoi. Viec do can hieu noi dung ('Moonpie' la biet danh
Meemaw goi Sheldon) nen giao cho LLM. Do lan dau (2026-08-16) cho thay danh
sach ten tran (khong thoai) lam LLM qua de dat: chi gop cap hien nhien ve mat
chuoi (Dr. Sturgis / Dr. John Sturgis), bo sot Meemaw/Connie du vi du nam ngay
trong prompt — nen kem theo toi da 2 dong thoai evidence cho moi nhan vat
(chay tren model goc, khong qua adapter, nen khong vuong nguong ~1000 token
cua duong refine).

Validate cung truoc khi tin LLM (cung tinh than evidence-gate):
- chi ten co trong registry; ten la bi bo qua
- mot ten chi duoc nam trong MOT nhom; xung dot -> bo ten do khoi moi nhom
- nhom tron gioi tinh ro rang (male + female) bi bac ca nhom
- nhom con lai < 2 ten thi khong con gi de gop

generate_fn(system, user) -> str inject duoc de test khong can GPU, giong
build_registry va cluster_map.
"""
from __future__ import annotations

import json
import re

from CHARACTER.registry_schema import (
    CONFIDENCE_LEVELS,
    Character,
    Registry,
    Relation,
)

UNIFY_SYSTEM = (
    "You are an expert film-script analyst. The character list below comes "
    "from an automatic extraction and often contains the SAME person under "
    "several entries: nicknames (Sheldon / Shelly / Moonpie), spelling or "
    "title variants (Dr. Sturgis / Dr. John Sturgis), a family role next to "
    "a real name (Meemaw / Connie), or an older self (Adult Sheldon).\n"
    "Group the names that refer to the same person.\n"
    "Rules:\n"
    "- Use ONLY names from the list, spelled exactly as given.\n"
    "- Only group names when the evidence (gender, age, common sense about "
    "the film) supports it. Different people with similar names stay apart.\n"
    "- A name belongs to at most one group. Leave singletons out entirely.\n"
    "- Output STRICT JSON only: a list of groups, each group a list of names. "
    "No markdown fences, no commentary."
)


def build_unify_prompt(reg: Registry, captions_text: str = "",
                       en_lines: list[str] | None = None) -> tuple[str, str]:
    lines = []
    for c in reg.characters:
        traits = ", ".join(
            t for t in (c.gender if c.gender != "unknown" else "",
                        c.age_range if c.age_range != "unknown" else "")
            if t
        )
        names = " / ".join(c.names)
        entry = f"- {names}" + (f" ({traits})" if traits else "")
        if en_lines:
            for i in c.evidence_lines[:2]:
                if 1 <= i <= len(en_lines):
                    entry += f'\n    "{en_lines[i - 1][:100]}"'
        lines.append(entry)
    example = [["Sheldon", "Shelly", "Moonpie"], ["Meemaw", "Connie"]]
    user = (
        "<Visual Scene Captions>\n"
        f"{captions_text or 'None'}\n"
        "</Visual Scene Captions>\n\n"
        "<Extracted characters>\n"
        + "\n".join(lines)
        + "\n</Extracted characters>\n\n"
        "Return the groups of names that are the same person. Example shape:\n"
        f"{json.dumps(example, ensure_ascii=False, separators=(',', ':'))}"
    )
    return UNIFY_SYSTEM, user


def parse_groups(text: str, reg: Registry) -> list[set[str]]:
    """Doc JSON tho cua LLM thanh cac nhom alias da validate."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []

    canonical = {n.casefold(): n for c in reg.characters for n in c.names}
    gender_of: dict[str, str] = {}
    for c in reg.characters:
        for n in c.names:
            gender_of[n.casefold()] = c.gender

    groups: list[set[str]] = []
    for g in raw:
        if not isinstance(g, list):
            continue
        names = {canonical[str(n).strip().casefold()] for n in g
                 if isinstance(n, str) and str(n).strip().casefold() in canonical}
        if len(names) < 2:
            continue
        genders = {gender_of[n.casefold()] for n in names} - {"unknown"}
        if len(genders) > 1:
            continue          # tron male + female -> chac chan gop nham
        groups.append(names)

    # Mot ten nam trong >= 2 nhom la mau thuan -> bo ten do khoi tat ca.
    seen: dict[str, int] = {}
    for g in groups:
        for n in g:
            seen[n] = seen.get(n, 0) + 1
    dup = {n for n, k in seen.items() if k > 1}
    return [g - dup for g in groups if len(g - dup) >= 2]


def unify_registry(reg: Registry, groups: list[set[str]]) -> Registry:
    """Gop cac nhan vat co ten cung nhom; relation remap id, giu confidence
    cao nhat cho moi (from, to), evidence union, self-loop sau gop bi bo."""
    if not groups:
        return reg

    owner_of_name = {n.casefold(): c.id for c in reg.characters for n in c.names}
    parent: dict[str, str] = {c.id: c.id for c in reg.characters}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in groups:
        ids = {owner_of_name[n.casefold()] for n in g}
        # sorted: thu tu duyet set chuoi doi theo hash seed cua tung process,
        # de nguyen thi root cua nhom (va danh so C{n} dau ra) het tat dinh.
        ids = sorted(i for i in ids if i in parent)
        for other in ids[1:]:
            ra, rb = find(ids[0]), find(other)
            if ra != rb:
                parent[rb] = ra

    root_members: dict[str, list[Character]] = {}
    for c in reg.characters:
        root_members.setdefault(find(c.id), []).append(c)

    id_map: dict[str, str] = {}
    merged: list[Character] = []
    for gi, (root, members) in enumerate(
        sorted(root_members.items(), key=lambda kv: kv[0]), start=1
    ):
        new_id = f"C{gi}"
        names: list[str] = []
        evidence: set[int] = set()
        gender = age_range = "unknown"
        for c in members:
            id_map[c.id] = new_id
            for n in c.names:
                if n.casefold() not in {x.casefold() for x in names}:
                    names.append(n)
            evidence.update(c.evidence_lines)
            if gender == "unknown":
                gender = c.gender
            if age_range == "unknown":
                age_range = c.age_range
        merged.append(Character(
            new_id, tuple(names), gender, age_range, tuple(sorted(evidence))
        ))

    best: dict[tuple[str, str], Relation] = {}
    for r in reg.relations:
        f, t = id_map[r.from_id], id_map[r.to_id]
        if f == t:
            continue          # canh giua hai alias cua cung mot nguoi
        cur = best.get((f, t))
        evidence = tuple(sorted(
            set(r.evidence_lines) | set(cur.evidence_lines if cur else ())
        ))
        if (cur is None or CONFIDENCE_LEVELS[r.confidence]
                > CONFIDENCE_LEVELS[cur.confidence]):
            best[(f, t)] = Relation(f, t, r.rel_type, r.vi_self,
                                    r.vi_listener, r.confidence, evidence)
        else:
            best[(f, t)] = Relation(cur.from_id, cur.to_id, cur.rel_type,
                                    cur.vi_self, cur.vi_listener,
                                    cur.confidence, evidence)
    relations = tuple(best[k] for k in sorted(best))
    return Registry(tuple(merged), relations)


def unify_cast(reg: Registry, generate_fn, captions_text: str = "",
               en_lines: list[str] | None = None) -> Registry:
    """Orchestrator: hoi LLM mot lan tren danh sach ten, validate, gop."""
    if len(reg.characters) < 2:
        return reg
    system, user = build_unify_prompt(reg, captions_text, en_lines)
    decoded = generate_fn(system, user)
    groups = parse_groups(decoded, reg)
    out = unify_registry(reg, groups)
    print(f"[unify] {len(reg.characters)} nhan vat -> {len(out.characters)} "
          f"({len(groups)} nhom gop)", flush=True)
    return out


def main() -> None:
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="Unify duplicate character identities in a registry."
    )
    parser.add_argument("--registry_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--vlm_json", type=str, default=None)
    parser.add_argument("--en_srt", type=str, default=None,
                        help="en.srt de trich dan thoai evidence vao prompt.")
    parser.add_argument("--model_name", type=str,
                        default="unsloth/gemma-3-12b-it-unsloth-bnb-4bit")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from CHARACTER.build_registry import _make_generate_fn
    from CHARACTER.registry_prompt import captions_summary
    from CHARACTER.registry_schema import load_registry, registry_to_json

    captions_text = ""
    if args.vlm_json and os.path.exists(args.vlm_json):
        with open(args.vlm_json, "r", encoding="utf-8") as f:
            captions_text = captions_summary(json.load(f), max_chars=600)

    en_lines = None
    if args.en_srt and os.path.exists(args.en_srt):
        import re as _re
        import srt as _srt
        with open(args.en_srt, "r", encoding="utf-8") as f:
            en_lines = [_re.sub(r"\s+", " ", s.content).strip()
                        for s in _srt.parse(f.read())]

    generate_fn, _ = _make_generate_fn(
        args.model_name, args.cache_dir or os.path.join(root, "cache"),
        max_seq_length=4096, max_new_tokens=512,
    )
    reg = unify_cast(load_registry(args.registry_json), generate_fn,
                     captions_text, en_lines)
    tmp = args.output_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry_to_json(reg), f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.output_json)
    print(f"[unify] Saved: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
