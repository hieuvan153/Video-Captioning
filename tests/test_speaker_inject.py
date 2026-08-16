import json

import pytest

from SPEAKER.inject import (
    load_speakers,
    save_speakers,
    speakers_to_json,
    tag_english_lines,
    turn_pairs,
)


def test_tags_only_lines_that_have_a_speaker():
    out = tag_english_lines(["Hi there.", "Sure.", "Bye."],
                            ["Sheldon", None, "Mary"])
    assert out == ["[SPEAKER: Sheldon] Hi there.", "Sure.",
                   "[SPEAKER: Mary] Bye."]


def test_no_speakers_leaves_lines_untouched():
    lines = ["Hi there.", "Sure."]
    assert tag_english_lines(lines, None) == lines


def test_tagging_never_changes_the_line_count():
    """So dong output tieng Viet bam theo so dong EN — tag khong duoc dung vao."""
    lines = ["a", "b", "c", "d"]
    assert len(tag_english_lines(lines, ["X", None, "Y", "Z"])) == len(lines)


def test_length_mismatch_raises_instead_of_misaligning():
    with pytest.raises(ValueError):
        tag_english_lines(["a", "b"], ["X"])


def test_turn_pairs_from_adjacent_turns():
    assert turn_pairs(["A", "B", "A", "C"]) == {("A", "B"), ("A", "C")}


def test_turn_pairs_ignores_repeated_speaker():
    assert turn_pairs(["A", "A", "A"]) == set()


def test_untagged_line_breaks_the_turn_chain():
    """A -> ? -> B: nguoi thu ba co the da xen vao, khong ket luan A noi voi B."""
    assert turn_pairs(["A", None, "B"]) == set()


def test_turn_pairs_are_unordered_and_deduped():
    assert turn_pairs(["B", "A", "B", "A"]) == {("A", "B")}


def test_turn_pairs_empty_for_a_monologue():
    assert turn_pairs(["A"]) == set()
    assert turn_pairs([None, None]) == set()


def test_save_then_load_round_trips(tmp_path):
    names = ["Sheldon", None, "Mary"]
    path = str(tmp_path / "movie.speakers.json")
    save_speakers(path, speakers_to_json(names, ["SPK_001", None, "SPK_002"],
                                         {"SPK_001": "Sheldon"}))
    assert load_speakers(path, 3) == names


def test_load_speakers_rejects_a_length_mismatch(tmp_path):
    """Lech 1 dong lam lech toan bo tag ve sau — phai no, khong duoc cat bot."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"speakers": ["A", "B"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="SRT co 5 dong"):
        load_speakers(str(path), 5)


def test_load_speakers_normalises_blank_names(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"speakers": ["  ", "Mary", None]}),
                    encoding="utf-8")
    assert load_speakers(str(path), 3) == [None, "Mary", None]


def test_speakers_to_json_records_align_stats():
    from SPEAKER.align import AlignStats

    payload = speakers_to_json(["A"], ["SPK_001"], {"SPK_001": "A"},
                               AlignStats(1, 1, 0, 0))
    assert payload["n_lines"] == 1
    assert payload["n_named"] == 1
    assert payload["align_stats"]["coverage"] == 1.0
