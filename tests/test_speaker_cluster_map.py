import json

from SPEAKER.cluster_map import (
    plan_batches,
    build_mapping_prompt,
    map_clusters,
    parse_mapping,
    tags_to_names,
)

NAMES = ["Sheldon", "Mary", "Meemaw"]


def test_parse_mapping_keeps_only_names_from_closed_set():
    raw = json.dumps({"SPK_001": "Sheldon", "SPK_002": "Gandalf",
                      "SPK_003": "unknown"})
    assert parse_mapping(raw, NAMES) == {"SPK_001": "Sheldon"}


def test_parse_mapping_is_case_insensitive_but_returns_canonical_name():
    assert parse_mapping('{"SPK_001":"sheldon"}', NAMES) == {"SPK_001": "Sheldon"}


def test_parse_mapping_survives_surrounding_prose():
    text = 'Sure! Here you go:\n{"SPK_007":"Mary"}\nHope that helps.'
    assert parse_mapping(text, NAMES) == {"SPK_007": "Mary"}


def test_parse_mapping_returns_empty_on_garbage():
    assert parse_mapping("no json here", NAMES) == {}
    assert parse_mapping("{broken", NAMES) == {}
    assert parse_mapping('["a","b"]', NAMES) == {}


def test_several_clusters_may_share_one_character():
    """CAM++ tach mot giong thanh nhieu cluster — day la binh thuong."""
    raw = json.dumps({"SPK_001": "Sheldon", "SPK_004": "Sheldon"})
    assert parse_mapping(raw, NAMES) == {"SPK_001": "Sheldon",
                                         "SPK_004": "Sheldon"}


def test_map_clusters_batches_and_merges():
    seen = []

    def fake_generate(system, user):
        tags = [t for t in ["SPK_001", "SPK_002", "SPK_003"] if t in user]
        seen.append(tuple(tags))
        return json.dumps({t: "Mary" for t in tags})

    lines = {f"SPK_00{i}": ["a long enough line here"] for i in (1, 2, 3)}
    out = map_clusters(lines, NAMES, fake_generate, batch_size=2)
    assert out == {"SPK_001": "Mary", "SPK_002": "Mary", "SPK_003": "Mary"}
    assert len(seen) == 2          # 3 cluster / batch_size 2 -> 2 lo


def test_map_clusters_ignores_clusters_absent_from_the_batch():
    def hallucinating_generate(system, user):
        return json.dumps({"SPK_001": "Mary", "SPK_999": "Meemaw"})

    out = map_clusters({"SPK_001": ["line one here"]}, NAMES,
                       hallucinating_generate)
    assert out == {"SPK_001": "Mary"}


def test_map_clusters_noop_without_characters():
    called = []
    out = map_clusters({"SPK_001": ["x"]}, [],
                       lambda s, u: called.append(1) or "{}")
    assert out == {}
    assert not called


def test_build_mapping_prompt_lists_names_and_clusters():
    system, user = build_mapping_prompt(
        {"SPK_001": ["hello there"]}, NAMES, captions_text="A kitchen."
    )
    assert "Never invent a name" in system
    assert "Sheldon" in user and "SPK_001" in user
    assert "hello there" in user and "A kitchen." in user


def test_prompt_warns_that_a_named_person_is_the_addressee():
    """Loi do duoc tren movie_045: cluster noi "Adrian, I can't believe you"
    bi gan chinh ten Adrian."""
    system, _ = build_mapping_prompt({"SPK_001": ["x"]}, NAMES)
    assert "NOT the speaker" in system
    assert "script label" in system      # bang chung manh nhat, phai duoc noi ro


def test_prompt_shows_line_counts_so_bit_parts_are_recognisable():
    _, user = build_mapping_prompt({"SPK_001": ["hi"]}, NAMES,
                                   line_counts={"SPK_001": 59})
    assert "59 dialogue lines" in user


def test_map_clusters_asks_about_the_biggest_clusters_first():
    """Batch dau phai chua nhan vat chinh: nhieu bang chung nhat."""
    seen = []

    def fake_generate(system, user):
        # "SPK_001"/"SPK_002" cung nam trong JSON example cua prompt, nen phai
        # tim theo header cua block cluster chu khong phai tim ten tran.
        seen.append([t for t in ("SPK_001", "SPK_002", "SPK_003")
                     if f"{t} (" in user])
        return "{}"

    lines = {t: ["a line long enough"] for t in
             ("SPK_001", "SPK_002", "SPK_003")}
    map_clusters(lines, NAMES, fake_generate, batch_size=1,
                 line_counts={"SPK_001": 2, "SPK_002": 90, "SPK_003": 40})
    assert [b[0] for b in seen] == ["SPK_002", "SPK_003", "SPK_001"]


def test_long_lines_are_truncated_so_one_line_cant_eat_the_budget():
    _, user = build_mapping_prompt({"SPK_001": ["word " * 200]}, NAMES)
    assert len(user) < 1500


def test_batches_are_packed_to_stay_under_the_token_budget():
    """Prompt vuot ~1200 token la Gemma-3 bo qua ca bang chung hien nhien."""
    lines = {f"SPK_{i:03d}": ["a moderately long dialogue line here"]
             for i in range(1, 7)}
    counts = {t: 10 for t in lines}
    # 10 "token" moi ky tu 40 -> dem gia: 1 token / 40 ky tu.
    batches = plan_batches(sorted(lines), lines, NAMES, "",
                           lambda s, u: (len(s) + len(u)) // 40, counts,
                           token_budget=30)
    assert len(batches) > 1
    assert sorted(t for b in batches for t in b) == sorted(lines)
    for b in batches:
        s, u = build_mapping_prompt({t: lines[t] for t in b}, NAMES,
                                    line_counts=counts)
        assert (len(s) + len(u)) // 40 <= 30 or len(b) == 1


def test_plan_batches_warns_when_a_single_cluster_exceeds_budget(capsys):
    """Lo don vuot nguong van phai gui, nhung phai keu len trong log —
    do la vung suy giam cua Gemma-3 (movie_009 unify tung suy bien im lang)."""
    batches = plan_batches(["SPK_001"], {"SPK_001": ["x"]}, NAMES, "",
                           lambda s, u: 5000, {}, token_budget=1000)
    assert batches == [["SPK_001"]]
    out = capsys.readouterr().out
    assert "Warning" in out and "5000" in out


def test_map_clusters_uses_the_token_budget_when_given_a_counter():
    calls = []
    map_clusters({f"SPK_{i:03d}": ["some dialogue line"] for i in range(1, 5)},
                 NAMES, lambda s, u: calls.append(u) or "{}",
                 count_tokens=lambda s, u: 10**6, batch_size=12)
    assert len(calls) == 4          # budget vuot ngay -> moi cluster mot lo


def test_tags_to_names_drops_unmapped_and_untagged_lines():
    tags = ["SPK_001", "SPK_777", None]
    assert tags_to_names(tags, {"SPK_001": "Mary"}) == ["Mary", None, None]
