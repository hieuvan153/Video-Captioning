from CHARACTER.registry_prompt import (
    build_extraction_prompt,
    captions_summary,
    chunk_lines,
    number_lines,
    parse_llm_json,
    render_registry_block,
    render_registry_context,
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


def test_render_context_caption_style_single_line():
    # Adapter duoc train voi Scene Context dang "3. Relationship: [A & B] - [Type]"
    # => renderer phai ra 1 dong prose cung style, KHONG dung tag XML.
    ctx = render_registry_context(_sample_registry())
    assert ctx.startswith("Relationship")
    assert "<" not in ctx and "\n" not in ctx
    assert "Meemaw & Sheldon" in ctx or "Sheldon & Meemaw" in ctx
    assert 'self "cháu"' in ctx  # high-confidence edge Sheldon->Meemaw


def test_render_context_filters_low_confidence():
    ctx = render_registry_context(_sample_registry(), min_confidence="medium")
    # edge Meemaw->Sheldon la "low" => khong duoc xuat hien
    assert 'Meemaw calls Sheldon' not in ctx


def test_render_context_empty_registry_returns_empty_string():
    assert render_registry_context(Registry()) == ""


def test_render_context_skips_neutral_toi_ban_edges():
    # Edge "tôi"/"bạn" la mac dinh trung tinh cua NMT — khong mang thong tin,
    # chi day model paraphrase lech (quan sat tren movie_008), nen bo qua.
    chars = (
        Character("C1", ("Glenn",), "male", "adult", (1,)),
        Character("C2", ("Kitty",), "female", "adult", (2,)),
        Character("C3", ("Sal",), "male", "adult", (3,)),
    )
    rels = (
        Relation("C1", "C2", "coworker", "tôi", "bạn", "high", (1,)),
        Relation("C3", "C2", "lover", "anh", "em", "high", (3,)),
    )
    ctx = render_registry_context(Registry(chars, rels))
    assert "Glenn" not in ctx          # cap chi co edge trung tinh -> bien mat
    assert 'Sal calls Kitty "em"' in ctx


def test_render_context_all_neutral_returns_empty():
    chars = (
        Character("C1", ("A",), "male", "adult", (1,)),
        Character("C2", ("B",), "female", "adult", (2,)),
    )
    rels = (Relation("C1", "C2", "coworker", "tôi", "bạn", "high", (1,)),)
    assert render_registry_context(Registry(chars, rels)) == ""
