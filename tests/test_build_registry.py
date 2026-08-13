import json

from CHARACTER.build_registry import extract_registry

_CHUNK_RESPONSE = json.dumps({
    "characters": [
        {"id": "C1", "names": ["Anna"], "gender": "female",
         "age_range": "adult", "evidence_lines": [1]},
        {"id": "C2", "names": ["Tom"], "gender": "male",
         "age_range": "adult", "evidence_lines": [2]},
    ],
    "relations": [
        {"from_id": "C1", "to_id": "C2", "rel_type": "wife->husband",
         "vi_self": "em", "vi_listener": "anh", "confidence": "high",
         "evidence_lines": [1, 2]},
    ],
})


def test_extract_registry_single_chunk():
    def fake_generate(system: str, user: str) -> str:
        assert "DIRECTED" in system
        return _CHUNK_RESPONSE

    reg = extract_registry(["Hello Tom.", "Hi Anna."], "", fake_generate)
    assert len(reg.characters) == 2
    assert len(reg.relations) == 1
    assert reg.relations[0].vi_self == "em"


def test_extract_registry_skips_unparseable_chunk():
    calls = {"n": 0}

    def flaky_generate(system: str, user: str) -> str:
        calls["n"] += 1
        return "GARBAGE NOT JSON" if calls["n"] == 1 else _CHUNK_RESPONSE

    lines = [f"line {i}" for i in range(300)]  # 2 chunk voi size=200/overlap=30
    reg = extract_registry(lines, "", flaky_generate)
    assert calls["n"] == 2
    assert len(reg.characters) == 2  # chunk 2 van duoc dung


def test_extract_registry_empty_transcript():
    reg = extract_registry([], "", lambda s, u: "{}")
    assert reg.characters == () and reg.relations == ()
