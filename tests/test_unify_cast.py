import json

from CHARACTER.registry_schema import Character, Registry, Relation
from CHARACTER.unify_cast import (
    build_unify_prompt,
    parse_groups,
    unify_cast,
    unify_registry,
)

REG = Registry(
    characters=(
        Character("C1", ("Sheldon",), "male", "child", (1,)),
        Character("C2", ("Shelly",), "male", "child", (2,)),
        Character("C3", ("Meemaw",), "female", "elderly", (3,)),
        Character("C4", ("Connie",), "female", "elderly", (4,)),
        Character("C5", ("George",), "male", "adult", (5,)),
    ),
    relations=(
        Relation("C2", "C3", "grandchild->grandmother", "cháu", "bà",
                 "high", (2,)),
        Relation("C1", "C4", "grandchild->grandmother", "cháu", "bà",
                 "medium", (1,)),
        Relation("C5", "C1", "father->son", "bố", "con", "high", (5,)),
    ),
)
GROUPS = [{"Sheldon", "Shelly"}, {"Meemaw", "Connie"}]


def test_merge_collapses_grouped_characters():
    out = unify_registry(REG, GROUPS)
    assert len(out.characters) == 3
    names = {frozenset(c.names) for c in out.characters}
    assert frozenset({"Sheldon", "Shelly"}) in names
    assert frozenset({"Meemaw", "Connie"}) in names


def test_merge_remaps_relations_and_keeps_best_confidence():
    """C2->C3 (high) va C1->C4 (medium) sau gop la CUNG mot canh
    Sheldon->Meemaw: giu confidence cao nhat, evidence union."""
    out = unify_registry(REG, GROUPS)
    name_of = {c.id: c.names[0] for c in out.characters}
    rels = {(name_of[r.from_id], name_of[r.to_id]): r for r in out.relations}
    r = rels[("Sheldon", "Meemaw")]
    assert r.confidence == "high"
    assert r.evidence_lines == (1, 2)
    assert ("George", "Sheldon") in rels          # canh ngoai nhom giu nguyen


def test_merge_drops_edges_between_aliases_of_one_person():
    reg = Registry(
        REG.characters,
        REG.relations + (Relation("C1", "C2", "friend", "anh", "em",
                                  "high", (1,)),),
    )
    out = unify_registry(reg, GROUPS)
    name_of = {c.id: c.names[0] for c in out.characters}
    assert all(name_of[r.from_id] != name_of[r.to_id] for r in out.relations)


def test_empty_groups_is_identity():
    assert unify_registry(REG, []) == REG


def test_merge_output_ids_are_deterministic():
    """Root cua nhom la id NHO NHAT, khong phu thuoc thu tu duyet set (hash
    seed doi moi process) — dau ra phai trung tung byte giua hai lan chay."""
    out = unify_registry(REG, GROUPS)
    by_names = {frozenset(c.names): c.id for c in out.characters}
    assert by_names[frozenset({"Sheldon", "Shelly"})] == "C1"
    assert by_names[frozenset({"Meemaw", "Connie"})] == "C2"
    assert by_names[frozenset({"George"})] == "C3"


def test_parse_rejects_mixed_gender_groups():
    """'Meemaw' (female) + 'George' (male) — LLM gop nham thi phai bac."""
    raw = json.dumps([["Meemaw", "George"], ["Sheldon", "Shelly"]])
    assert parse_groups(raw, REG) == [{"Sheldon", "Shelly"}]


def test_parse_ignores_unknown_names_and_small_groups():
    raw = json.dumps([["Sheldon", "Gandalf"], ["Meemaw"]])
    assert parse_groups(raw, REG) == []


def test_parse_resolves_case_to_canonical_spelling():
    raw = json.dumps([["sheldon", "SHELLY"]])
    assert parse_groups(raw, REG) == [{"Sheldon", "Shelly"}]


def test_parse_drops_names_claimed_by_two_groups():
    """'Sheldon' nam trong 2 nhom hop le la mau thuan -> bo ten do khoi ca
    hai; phan con lai cua moi nhom khong du 2 ten thi rot theo."""
    raw = json.dumps([["Sheldon", "Shelly"], ["Sheldon", "George"]])
    out = parse_groups(raw, REG)
    assert all("Sheldon" not in g for g in out)
    assert out == []          # ca hai nhom chi con 1 ten sau khi bo Sheldon


def test_parse_survives_prose_and_garbage():
    assert parse_groups("no json", REG) == []
    assert parse_groups('{"a": 1}', REG) == []
    text = 'Here you go:\n[["Sheldon","Shelly"]]\nCheers.'
    assert parse_groups(text, REG) == [{"Sheldon", "Shelly"}]


def test_unify_cast_end_to_end_with_fake_llm():
    def fake_generate(system, user):
        assert "Sheldon" in user and "ONLY names" in system
        return json.dumps([["Sheldon", "Shelly"], ["Meemaw", "Connie"]])

    out = unify_cast(REG, fake_generate)
    assert len(out.characters) == 3


def test_unify_cast_skips_llm_for_tiny_casts():
    called = []
    reg = Registry((Character("C1", ("Solo",), "male", "adult", (1,)),), ())
    out = unify_cast(reg, lambda s, u: called.append(1) or "[]")
    assert out == reg and not called


def test_prompt_shows_traits_to_ground_the_grouping():
    _, user = build_unify_prompt(REG)
    assert "Sheldon (male, child)" in user
    assert "Meemaw (female, elderly)" in user


def test_prompt_quotes_evidence_lines_when_given():
    en_lines = ["Shelly, dinner is ready!", "Coming, Meemaw.",
                "I am Connie.", "Hi.", "George here."]
    _, user = build_unify_prompt(REG, en_lines=en_lines)
    # C1 evidence_lines=(1,) -> dong 1; chi so 1-based, cat 100 ky tu
    assert '"Shelly, dinner is ready!"' in user
    assert '"Coming, Meemaw."' in user
    # khong co en_lines thi khong co trich dan
    _, bare = build_unify_prompt(REG)
    assert '"Shelly, dinner is ready!"' not in bare
