"""Dao canh xung ho co huong tu vocative EN trong chinh loi thoai.

Vi sao can (docs/superpowers/plans/2026-08-17-line-level-tags-v3.md, Stage C):
V2b da do — canh kinship-high cua registry va cap nguoi that su thoai ke nhau
la hai tap roi nhau tren 4/5 phim eval, nen registry theo scene KHONG THE kich
hoat them du hop nhat alias hoan hao. Canh o day duoc dao TU luot thoai ke
nhau, nen theo cau truc nam dung tren cap dang noi chuyen.

Co che: chunk cua cluster co ten A chua term xung ho EN ("Thanks, Mom.") o vi
tri vocative -> vote cho hang xom ke co ten B (ca truoc lan sau; min_votes=2
loc trung hop "Mom" trong loi ke lai). Moi guardrail "im lang khi nghi ngo":
gioi tinh registry cua B xung dot voi term -> bo; mot cap nhan 2 term mau
thuan cau truc (me vs ba) -> bo; hai dau phai resolve qua alias index.

Canh nguoc (B goi A la "con", tu xung "mẹ") gan nhu chac ve ngon ngu hoc
nhung la SUY RA, khong co bang chung truc tiep -> phat o confidence medium.

Output la list dict (from_name/to_name/vi_self/vi_listener/confidence/votes),
luu vao speakers.json truong "address_edges" — TRO voi registry_scope "pair"
(chi scope "pair_rel" cua LLM/refine_llm.py doc no), nen mot lan build phuc
vu ca hai arm speaker_v3 / speaker_v3_rel.
"""
from __future__ import annotations

from CHARACTER.registry_prompt import _alias_index
from SPEAKER.align import real_tag
from SPEAKER.name_evidence import _vocative_pattern, strip_script_labels

# term EN (casefold) -> (vi_listener, vi_self, gioi tinh nguoi nghe mong doi).
# Chi term than toc dot dau: "sir"/"ma'am"/ten rieng khong mang (hoac mang mo
# ho) thong tin xung ho tieng Viet — de danh cho vong sau.
# Moi gia tri vi_* deu thuoc KINSHIP_TERMS (CHARACTER/registry_prompt.py) va
# PRONOUN_TERMS (EVAL/pronoun_lexicon.py).
ADDRESS_TERMS: dict[str, tuple[str, str, str]] = {
    "mom":     ("mẹ", "con", "female"),
    "mommy":   ("mẹ", "con", "female"),
    "mama":    ("mẹ", "con", "female"),
    "dad":     ("bố", "con", "male"),
    "daddy":   ("bố", "con", "male"),
    "grandma": ("bà", "cháu", "female"),
    "granny":  ("bà", "cháu", "female"),
    "grandpa": ("ông", "cháu", "male"),
    "aunt":    ("cô", "cháu", "female"),
    "auntie":  ("cô", "cháu", "female"),
    "uncle":   ("chú", "cháu", "male"),
}


def mine_address_edges(
    chunks: list[dict],
    names_by_tag: dict[str, str],
    registry,
    min_votes: int = 2,
) -> list[dict]:
    """Canh xung ho co huong giua cac cluster DA CO TEN.

    names_by_tag: mapping cluster -> ten (ket qua cua build_speakers).
    registry: Registry — dung alias index de resolve va gender de gac cong.
    """
    if not names_by_tag or registry is None or not registry.characters:
        return []
    id_of = _alias_index(registry)
    gender_of = {c.id: c.gender for c in registry.characters}
    patterns = {t: _vocative_pattern(t) for t in ADDRESS_TERMS}

    votes: dict[tuple[str, str, str], int] = {}   # (A, B, term) -> phieu
    for i, c in enumerate(chunks):
        tag = real_tag(c)
        a = names_by_tag.get(tag) if tag else None
        if not a:
            continue
        text = strip_script_labels(c.get("english") or "").strip()
        if not text:
            continue
        hit_terms = [t for t, p in patterns.items() if p.search(text)]
        if not hit_terms:
            continue
        for j in (i - 1, i + 1):
            if not 0 <= j < len(chunks):
                continue
            btag = real_tag(chunks[j])
            b = names_by_tag.get(btag) if btag else None
            if not b or b == a:
                continue
            for term in hit_terms:
                key = (a, b, term)
                votes[key] = votes.get(key, 0) + 1

    per_pair: dict[tuple[str, str], dict[str, int]] = {}
    for (a, b, term), n in votes.items():
        if n >= min_votes:
            per_pair.setdefault((a, b), {})[term] = n

    edges: list[dict] = []
    for (a, b), terms in sorted(per_pair.items()):
        ia, ib = id_of.get(a.casefold()), id_of.get(b.casefold())
        if not ia or not ib or ia == ib:
            continue
        if len({ADDRESS_TERMS[t][0] for t in terms}) > 1:
            continue        # "mẹ" vs "bà" cho cung mot nguoi nghe -> im lang
        term, n = max(terms.items(), key=lambda kv: (kv[1], kv[0]))
        vi_listener, vi_self, expect_gender = ADDRESS_TERMS[term]
        g = gender_of.get(ib, "unknown")
        if g in ("male", "female") and g != expect_gender:
            continue        # registry noi B la nam ma term goi "mẹ" -> bo
        confidence = "high" if n >= 3 else "medium"
        edges.append({
            "from_name": a, "to_name": b,
            "rel_type": f"address:{term}",
            "vi_self": vi_self, "vi_listener": vi_listener,
            "confidence": confidence, "votes": n,
        })
        edges.append({
            "from_name": b, "to_name": a,
            "rel_type": f"address:{term}:reciprocal",
            "vi_self": vi_listener, "vi_listener": vi_self,
            "confidence": "medium", "votes": n,
        })
    return edges
