from CHARACTER.registry_schema import (
    Registry,
    filter_registry_by_source,
    merge_registries,
    parse_registry,
    registry_to_json,
)

RAW = {
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
         "evidence_lines": [35, 40]},
    ],
}


def test_parse_valid_registry():
    reg = parse_registry(RAW, n_lines=100)
    assert len(reg.characters) == 2
    assert len(reg.relations) == 2
    assert reg.relations[0].vi_self == "cháu"


def test_parse_drops_relation_without_valid_evidence():
    raw = {
        "characters": RAW["characters"],
        "relations": [
            {"from_id": "C2", "to_id": "C1", "rel_type": "x",
             "vi_self": "cháu", "vi_listener": "bà", "confidence": "high",
             "evidence_lines": [999]},  # ngoai pham vi transcript
        ],
    }
    assert parse_registry(raw, n_lines=100).relations == ()


def test_parse_drops_unknown_ids_and_self_loops():
    raw = {
        "characters": [RAW["characters"][0]],
        "relations": [
            {"from_id": "C9", "to_id": "C1", "rel_type": "x",
             "vi_self": "a", "vi_listener": "b", "confidence": "high",
             "evidence_lines": [1]},
            {"from_id": "C1", "to_id": "C1", "rel_type": "x",
             "vi_self": "a", "vi_listener": "b", "confidence": "high",
             "evidence_lines": [1]},
        ],
    }
    assert parse_registry(raw, n_lines=100).relations == ()


def test_merge_dedups_characters_by_shared_alias():
    reg1 = parse_registry(RAW, n_lines=100)
    raw2 = {
        "characters": [
            # cung nhan vat, chunk khac dat id khac + alias trung "Grandma"
            {"id": "C1", "names": ["Grandma", "Constance"], "gender": "female",
             "age_range": "elderly", "evidence_lines": [210]},
        ],
        "relations": [],
    }
    reg2 = parse_registry(raw2, n_lines=300)
    merged = merge_registries([reg1, reg2])
    assert len(merged.characters) == 2  # Meemaw/Grandma/Constance gop lam 1
    names = {n for c in merged.characters for n in c.names}
    assert "Constance" in names and "Sheldon" in names


def test_merge_keeps_highest_confidence_relation():
    reg1 = parse_registry(RAW, n_lines=100)
    raw2 = {
        "characters": RAW["characters"],
        "relations": [
            {"from_id": "C2", "to_id": "C1", "rel_type": "grandchild->grandmother",
             "vi_self": "em", "vi_listener": "chị", "confidence": "low",
             "evidence_lines": [50]},
        ],
    }
    merged = merge_registries([reg1, parse_registry(raw2, n_lines=100)])
    rel = next(r for r in merged.relations
               if r.rel_type == "grandchild->grandmother")
    assert rel.vi_self == "cháu"  # high thang low
    assert 50 in rel.evidence_lines  # evidence van duoc gop


def test_filter_drops_characters_absent_from_source():
    # Chong few-shot leak: LLM chep nhan vat cua schema example (Meemaw/Sheldon)
    # vao registry cua phim khong he co ho => loc theo transcript+captions.
    reg = parse_registry(RAW, n_lines=100)
    source = "Sheldon said hi.\nA woman waves."  # co Sheldon, khong co Meemaw
    filtered = filter_registry_by_source(reg, source)
    assert [c.names[0] for c in filtered.characters] == ["Sheldon"]
    assert filtered.relations == ()  # moi relation cham vao Meemaw deu bi drop


def test_filter_keeps_all_when_names_present_case_insensitive():
    reg = parse_registry(RAW, n_lines=100)
    source = "MEEMAW plays cards with sheldon."
    filtered = filter_registry_by_source(reg, source)
    assert len(filtered.characters) == 2
    assert len(filtered.relations) == 2


def test_filter_matches_any_alias():
    reg = parse_registry(RAW, n_lines=100)
    # "Grandma" la alias thu 2 cua C1 => du "Meemaw" vang mat van giu
    filtered = filter_registry_by_source(reg, "Grandma and Sheldon talk.")
    assert len(filtered.characters) == 2


def test_json_roundtrip():
    reg = parse_registry(RAW, n_lines=100)
    raw = registry_to_json(reg)
    reg2 = parse_registry(raw, n_lines=100)
    assert reg == reg2


def test_parse_drops_placeholder_named_characters():
    # Registry that tung sinh nhan vat tu dai tu ("You", "Me", "I", "Him") —
    # khong the map speaker, chi gay nhieu (docs/eval/error_analysis_v0.md).
    raw = {
        "characters": [
            {"id": "C1", "names": ["You"], "gender": "unknown",
             "age_range": "adult", "evidence_lines": [1]},
            {"id": "C2", "names": ["Me", "I"], "gender": "unknown",
             "age_range": "adult", "evidence_lines": [2]},
            {"id": "C3", "names": ["Sheldon"], "gender": "male",
             "age_range": "child", "evidence_lines": [3]},
            {"id": "C4", "names": ["Him", "Rodger"], "gender": "male",
             "age_range": "adult", "evidence_lines": [4]},  # co ten that -> giu
        ],
        "relations": [
            {"from_id": "C1", "to_id": "C3", "rel_type": "unknown",
             "vi_self": "anh", "vi_listener": "em", "confidence": "high",
             "evidence_lines": [1]},
        ],
    }
    reg = parse_registry(raw, n_lines=10)
    assert {c.id for c in reg.characters} == {"C3", "C4"}
    assert reg.relations == ()  # from_id C1 da bi drop keo theo relation


def test_parse_drops_relation_with_non_lexicon_pronouns():
    # Gia tri rac quan sat trong registry that: "glenn", "anh/chị",
    # "(addressed as)" — vi_self/vi_listener phai la tu xung ho trong lexicon.
    raw = {
        "characters": RAW["characters"],
        "relations": [
            {"from_id": "C2", "to_id": "C1", "rel_type": "x",
             "vi_self": "glenn", "vi_listener": "tôi", "confidence": "high",
             "evidence_lines": [1]},
            {"from_id": "C1", "to_id": "C2", "rel_type": "x",
             "vi_self": "anh/chị", "vi_listener": "em", "confidence": "high",
             "evidence_lines": [1]},
        ],
    }
    assert parse_registry(raw, n_lines=100).relations == ()
    ok = {
        "characters": RAW["characters"],
        "relations": [
            {"from_id": "C2", "to_id": "C1", "rel_type": "lover",
             "vi_self": "em", "vi_listener": "anh", "confidence": "high",
             "evidence_lines": [1]},
        ],
    }
    reg = parse_registry(ok, n_lines=100)
    assert len(reg.relations) == 1 and reg.relations[0].vi_self == "em"
