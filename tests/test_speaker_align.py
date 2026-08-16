from SPEAKER.align import (
    assign_speakers,
    representative_lines,
)


def _chunk(start, end, tag, score=0.5, english="hello there friend"):
    return {
        "start": start, "end": end, "speaker_tag": tag,
        "speaker_score": score, "english": english,
    }


def test_assigns_tag_of_chunk_covering_the_line():
    chunks = [_chunk(0.0, 5.0, "SPK_001")]
    tags, stats = assign_speakers(chunks, [(1.0, 4.0)])
    assert tags == ["SPK_001"]
    assert stats.n_tagged == 1
    assert stats.coverage == 1.0


def test_picks_dominant_speaker_not_first_toucher():
    """Dong cham SPK_001 truoc nhung phan lon thoi luong la SPK_002."""
    chunks = [_chunk(0.0, 1.2, "SPK_001"), _chunk(1.2, 10.0, "SPK_002")]
    tags, stats = assign_speakers(chunks, [(1.0, 10.0)])
    assert tags == ["SPK_002"]
    assert stats.n_contested == 1


def test_exact_tie_returns_none_rather_than_guessing():
    chunks = [_chunk(0.0, 2.0, "SPK_001"), _chunk(2.0, 4.0, "SPK_002")]
    tags, stats = assign_speakers(chunks, [(0.0, 4.0)])
    assert tags == [None]
    assert stats.n_tagged == 0


def test_line_without_overlap_is_untagged():
    chunks = [_chunk(0.0, 1.0, "SPK_001")]
    tags, stats = assign_speakers(chunks, [(50.0, 60.0)])
    assert tags == [None]
    assert stats.coverage == 0.0


def test_grazing_overlap_below_threshold_is_ignored():
    """Cham nhau 10ms do lam tron thi khong tinh la thoai."""
    chunks = [_chunk(0.0, 5.01, "SPK_001"), _chunk(5.0, 9.0, "SPK_002")]
    tags, _ = assign_speakers(chunks, [(5.0, 9.0)])
    assert tags == ["SPK_002"]


def test_min_score_filters_weak_chunks():
    chunks = [_chunk(0.0, 5.0, "SPK_001", score=0.02)]
    tags, stats = assign_speakers(chunks, [(1.0, 4.0)], min_score=0.1)
    assert tags == [None]
    assert stats.n_low_score == 1


def test_chunks_without_tag_or_bad_times_are_skipped():
    chunks = [
        {"start": 0.0, "end": 5.0, "speaker_tag": None},
        {"start": "x", "end": 5.0, "speaker_tag": "SPK_002"},
        {"start": 5.0, "end": 5.0, "speaker_tag": "SPK_003"},   # do dai 0
    ]
    tags, stats = assign_speakers(chunks, [(0.0, 5.0)])
    assert tags == [None]
    assert stats.n_lines == 1


def test_unknown_sentinel_is_not_treated_as_a_cluster():
    """CAM++ ghi "UNKNOWN" cho chunk duoi nguong — gom chung lai thanh mot
    "nguoi" thi 20-35% so dong bi gan chung mot ten sai."""
    chunks = [_chunk(0.0, 5.0, "UNKNOWN"), _chunk(5.0, 9.0, "unknown")]
    tags, stats = assign_speakers(chunks, [(1.0, 4.0), (6.0, 8.0)])
    assert tags == [None, None]
    assert stats.n_tagged == 0
    assert representative_lines(chunks) == {}


def test_a_real_cluster_still_wins_over_an_unknown_neighbour():
    chunks = [_chunk(0.0, 6.0, "UNKNOWN"), _chunk(1.0, 5.0, "SPK_002")]
    tags, _ = assign_speakers(chunks, [(1.0, 5.0)])
    assert tags == ["SPK_002"]


def test_representative_lines_prefers_high_score_long_sentences():
    chunks = [
        _chunk(0, 1, "SPK_001", score=0.1, english="a b c d e"),
        _chunk(1, 2, "SPK_001", score=0.9, english="strong match sentence here"),
        _chunk(2, 3, "SPK_001", score=0.5, english="Me."),
    ]
    reps = representative_lines(chunks, per_cluster=2)
    assert reps["SPK_001"][0] == "strong match sentence here"
    assert "Me." not in reps["SPK_001"]


def test_representative_lines_falls_back_when_only_short_lines():
    chunks = [_chunk(0, 1, "SPK_009", english="Me."),
              _chunk(1, 2, "SPK_009", english="You know.")]
    reps = representative_lines(chunks, per_cluster=4)
    assert reps["SPK_009"] == ["Me.", "You know."]
