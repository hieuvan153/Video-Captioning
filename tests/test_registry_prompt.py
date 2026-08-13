from CHARACTER.registry_prompt import (
    build_extraction_prompt,
    captions_summary,
    chunk_lines,
    number_lines,
    parse_llm_json,
    render_registry_block,
)
from CHARACTER.registry_schema import Character, Registry, Relation


def test_chunk_lines_overlap_and_coverage():
    lines = [str(i) for i in range(450)]
    chunks = chunk_lines(lines, chunk_size=200, overlap=30)
    assert chunks[0][0] == 0 and len(chunks[0][1]) == 200
    assert chunks[1][0] == 170  # step = 200 - 30
    # moi line phai nam trong it nhat 1 chunk
    covered = set()
    for start, sub in chunks:
        covered.update(range(start, start + len(sub)))
    assert covered == set(range(450))


def test_chunk_lines_short_input_single_chunk():
    assert chunk_lines(["a", "b"], chunk_size=200, overlap=30) == [(0, ["a", "b"])]


def test_number_lines_uses_1_based_global_index():
    assert number_lines(["hi", "yo"], start=171) == "171. hi\n172. yo"


def test_captions_summary_skips_error_captions():
    scenes = [
        {"caption": "A boy talks to his grandmother."},
        {"caption": "Error during analysis: CUDA OOM"},
        {"caption": ""},
    ]
    text = captions_summary(scenes)
    assert "grandmother" in text
    assert "Error" not in text


def test_parse_llm_json_handles_fences_and_prose():
    text = 'Here you go:\n```json\n{"characters": [], "relations": []}\n```\nDone.'
    assert parse_llm_json(text) == {"characters": [], "relations": []}
    assert parse_llm_json("no json here") is None


def test_extraction_prompt_mentions_directed_and_schema():
    system, user = build_extraction_prompt("1. Hello", "Scene 1: two people")
    assert "DIRECTED" in system
    assert "vi_self" in user and "evidence_lines" in user
    assert "1. Hello" in user


def _sample_registry() -> Registry:
    chars = (
        Character("C1", ("Meemaw", "Grandma"), "female", "elderly", (35,)),
        Character("C2", ("Sheldon",), "male", "child", (1,)),
    )
    rels = (
        Relation("C2", "C1", "grandchild->grandmother", "cháu", "bà",
                 "high", (35,)),
        Relation("C1", "C2", "grandmother->grandchild", "bà", "cháu",
                 "low", (35,)),
    )
    return Registry(chars, rels)


def test_render_block_filters_by_confidence():
    block = render_registry_block(_sample_registry(), min_confidence="medium")
    assert "<Character Registry>" in block and "</Character Registry>" in block
    assert 'calls self "cháu"' in block          # high: giu
    assert 'calls self "bà"' not in block        # low: bi loc
    assert "Meemaw" in block and "Sheldon" in block


def test_render_block_empty_registry_returns_empty_string():
    assert render_registry_block(Registry()) == ""
