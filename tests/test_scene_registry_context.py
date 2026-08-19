"""V1 buoc 4: registry chi noi ve nguoi dang noi trong scene, thay vi 1 dong
quan he toan phim tiem vao moi scene (nguon FP chinh cua V0)."""
from CHARACTER.registry_prompt import (
    render_scene_registry_context,
    render_speaker_registry_context,
)
from CHARACTER.registry_schema import Character, Registry, Relation

# Hai cap than toc high => qua duoc gate cap phim cua V0.
CHARS = (
    Character("C1", ("Meemaw", "Grandma"), "female", "elderly", (35,)),
    Character("C2", ("Sheldon",), "male", "child", (1,)),
    Character("C3", ("George",), "male", "adult", (7,)),
)
RELS = (
    Relation("C2", "C1", "grandchild->grandmother", "cháu", "bà", "high", (35,)),
    Relation("C3", "C2", "father->son", "bố", "con", "high", (7,)),
)
REG = Registry(CHARS, RELS)


def test_renders_only_the_pair_speaking_in_this_scene():
    ctx = render_scene_registry_context(REG, {("Sheldon", "Meemaw")})
    assert ctx.startswith("Relationship")
    assert "Meemaw" in ctx and 'self "cháu"' in ctx
    assert "George" not in ctx          # co mat trong registry nhung khong noi


def test_scene_with_the_other_pair_gets_the_other_edge():
    ctx = render_scene_registry_context(REG, {("George", "Sheldon")})
    assert 'George calls Sheldon "con"' in ctx
    assert "Meemaw" not in ctx


def test_caption_style_stays_single_line_prose():
    # Adapter suy bien khi gap block XML ngoai <Scene Context>; giu dung style
    # caption 1 dong nhu render_registry_context.
    ctx = render_scene_registry_context(REG, {("Sheldon", "Meemaw")})
    assert "<" not in ctx and "\n" not in ctx


def test_scene_without_a_known_pair_renders_nothing():
    assert render_scene_registry_context(REG, set()) == ""
    assert render_scene_registry_context(REG, {("Sheldon", "Stranger")}) == ""


def test_pair_without_kinship_edge_renders_nothing():
    """George & Meemaw cung co mat nhung registry khong co edge giua ho."""
    assert render_scene_registry_context(REG, {("George", "Meemaw")}) == ""


def test_alias_resolves_to_the_same_character():
    ctx = render_scene_registry_context(REG, {("Sheldon", "Grandma")})
    assert 'Sheldon calls Meemaw "bà"' in ctx


def test_pair_order_does_not_matter():
    a = render_scene_registry_context(REG, {("Sheldon", "Meemaw")})
    b = render_scene_registry_context(REG, {("Meemaw", "Sheldon")})
    assert a == b != ""


def test_film_level_gate_still_applies_before_scene_scoping():
    """min_kinship_pairs la quyet dinh 'phim nay co dang tiem khong' (do duoc o
    V0), nen phai tinh tren ca phim chu khong tren scene."""
    one_pair = Registry(CHARS[:2], RELS[:1])
    assert render_scene_registry_context(one_pair, {("Sheldon", "Meemaw")}) == ""


def test_generic_edges_are_still_excluded():
    chars = (Character("C1", ("Amy",), "female", "adult", (1,)),
             Character("C2", ("Jonah",), "male", "adult", (2,)))
    rels = (Relation("C1", "C2", "colleague", "tôi", "anh", "high", (1,)),)
    assert render_scene_registry_context(Registry(chars, rels),
                                         {("Amy", "Jonah")}) == ""


def test_empty_registry_renders_nothing():
    assert render_scene_registry_context(Registry(), {("A", "B")}) == ""


def test_max_pairs_caps_the_rendered_edges():
    chars = tuple(Character(f"C{i}", (f"N{i}",), "male", "adult", (i,))
                  for i in range(1, 7))
    rels = tuple(
        Relation(f"C{i}", f"C{i + 1}", "father->son", "bố", "con", "high", (i,))
        for i in range(1, 6)
    )
    pairs = {(f"N{i}", f"N{i + 1}") for i in range(1, 6)}
    ctx = render_scene_registry_context(Registry(chars, rels), pairs, max_pairs=2)
    assert ctx.count(" & ") == 2


def test_ambiguous_alias_is_dropped_rather_than_guessed():
    """Hai nhan vat cung alias -> khong phan giai duoc, im lang con hon gan nham."""
    chars = CHARS + (Character("C4", ("Sheldon",), "male", "adult", (9,)),)
    ctx = render_scene_registry_context(Registry(chars, RELS),
                                        {("Sheldon", "Meemaw")})
    assert ctx == ""


# --- V2a: render_speaker_registry_context (chi can mot dau canh co mat) ---


def test_any_fires_with_one_named_endpoint():
    """Sheldon co ten, nguoi doi thoai chua dinh danh (None) — V1 im lang,
    V2a phai render duoc canh cua Sheldon."""
    names = ["Sheldon", None, "Sheldon", None]
    assert render_scene_registry_context(REG, set()) == ""
    ctx = render_speaker_registry_context(REG, names)
    assert 'Sheldon calls Meemaw "bà"' in ctx
    assert ctx.startswith("Relationship (speakers in this scene):")


def test_any_renders_only_edges_touching_present_speakers():
    """Meemaw cam mic -> canh Meemaw-Sheldon co mat, canh George-Sheldon
    (ca hai deu vang) thi khong."""
    ctx = render_speaker_registry_context(REG, ["Meemaw", None])
    assert "Meemaw" in ctx
    assert "George" not in ctx


def test_any_needs_at_least_two_lines():
    """Scene 1 dong la doc thoai — khong co doi dap thi khong can xung ho."""
    assert render_speaker_registry_context(REG, ["Sheldon"]) == ""


def test_any_stays_silent_without_named_speakers():
    assert render_speaker_registry_context(REG, [None, None, None]) == ""


def test_any_keeps_the_film_level_gate():
    one_pair = Registry(CHARS[:2], RELS[:1])
    assert render_speaker_registry_context(one_pair, ["Sheldon", None]) == ""


def test_any_both_present_pairs_outrank_one_present():
    """Vuot max_pairs thi cap du hai dau phai duoc giu lai truoc."""
    ctx = render_speaker_registry_context(
        REG, ["George", "Sheldon", None], max_pairs=1
    )
    # George & Sheldon du ca hai dau; Meemaw & Sheldon chi co mot dau.
    assert "George" in ctx and "Meemaw" not in ctx


def test_any_ignores_unknown_and_ambiguous_names():
    chars = CHARS + (Character("C4", ("Sheldon",), "male", "adult", (9,)),)
    ctx = render_speaker_registry_context(Registry(chars, RELS),
                                          ["Sheldon", "Stranger"])
    assert ctx == ""


def test_any_caption_style_single_line():
    ctx = render_speaker_registry_context(REG, ["Sheldon", None])
    assert "<" not in ctx and "\n" not in ctx


# --- V3/C.a: include_address_edges (chi doi vi_listener thuoc KINSHIP_TERMS) ---


def test_address_scope_allows_generic_self_with_directed_listener():
    """Edge "tôi"/"anh" bi loai o che do kinship (ca hai dau phai than toc)
    nhung duoc phep khi include_address_edges: nguoi nghe van co huong."""
    chars = (Character("C1", ("Amy",), "female", "adult", (1,)),
             Character("C2", ("Jonah",), "male", "adult", (2,)),
             Character("C3", ("Glenn",), "male", "adult", (3,)),)
    rels = (Relation("C1", "C2", "colleague", "tôi", "anh", "high", (1,)),
            Relation("C3", "C1", "boss", "tôi", "cô", "high", (3,)),)
    reg = Registry(chars, rels)
    assert render_scene_registry_context(reg, {("Amy", "Jonah")}) == ""
    ctx = render_scene_registry_context(reg, {("Amy", "Jonah")},
                                        include_address_edges=True)
    assert 'Amy calls Jonah "anh"' in ctx
    assert "Glenn" not in ctx           # cap khong noi trong scene nay


# --- V3/C.b: extra_edges (canh xung ho dao tu vocative) ---

_EXTRA = [{"from_name": "George", "to_name": "Meemaw",
           "rel_type": "address:grandma", "vi_self": "cháu",
           "vi_listener": "bà", "confidence": "high", "votes": 3}]


def test_extra_address_edges_render_only_under_turn_pair_gating():
    """Canh vocative George->Meemaw chi hien khi chinh cap do thoai ke nhau."""
    ctx = render_scene_registry_context(REG, {("George", "Meemaw")},
                                        extra_edges=_EXTRA)
    assert 'George calls Meemaw "bà"' in ctx
    other = render_scene_registry_context(REG, {("Sheldon", "Meemaw")},
                                          extra_edges=_EXTRA)
    assert "George" not in other and "Meemaw" in other
    assert render_scene_registry_context(REG, set(), extra_edges=_EXTRA) == ""


def test_extra_edges_count_toward_the_film_level_gate():
    """Gate hoi "phim co bang chung xung ho co huong dang tin khong" — canh
    vocative dat nguong phieu la mot nguon bang chung nhu the."""
    one_pair = Registry(CHARS, RELS[:1])    # 1 cap kinship: gate tat
    assert render_scene_registry_context(one_pair, {("Sheldon", "Meemaw")}) == ""
    ctx = render_scene_registry_context(one_pair, {("Sheldon", "Meemaw")},
                                        extra_edges=_EXTRA)
    assert 'Sheldon calls Meemaw "bà"' in ctx


def test_extra_edges_with_unresolvable_names_are_ignored():
    edges = [{"from_name": "Nobody", "to_name": "Meemaw", "vi_self": "cháu",
              "vi_listener": "bà", "confidence": "high"}]
    ctx = render_scene_registry_context(REG, {("George", "Meemaw")},
                                        extra_edges=edges)
    assert ctx == ""


def test_single_line_prose_is_preserved_with_extra_edges():
    ctx = render_scene_registry_context(REG, {("George", "Meemaw")},
                                        extra_edges=_EXTRA)
    assert "<" not in ctx and "\n" not in ctx
