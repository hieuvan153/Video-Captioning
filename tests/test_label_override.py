"""Override nhan kich ban cap dong: cat chunk tai thoi diem nhan truoc khi
align, vung nhan mang tag LABEL::Ten va canh tranh overlap nhu moi vung khac."""
from SPEAKER.align import assign_speakers
from SPEAKER.label_override import (
    apply_label_names,
    is_label_tag,
    label_positions,
    split_chunks_at_labels,
)

NAMES = ["Missy", "Sheldon", "Ingram", "Meemaw"]

# 12 token, moi token 1 giay: "MISSY" la token thu 3 (0-based) -> nhan tai 3.0s.
_C248_TEXT = "Asked and answered. MISSY: Did you cry when you saw it? No."


def _wts(text, t0=0.0, step=1.0):
    import re
    words = re.findall(r"[A-Za-z0-9']+", text)
    return [{"word": w, "start": t0 + i * step, "end": t0 + (i + 1) * step}
            for i, w in enumerate(words)]


def _c248(score=0.9):
    return {"speaker_tag": "SPK_106", "english": _C248_TEXT,
            "start": 0.0, "end": 12.0, "speaker_score": score,
            "english_word_timestamps": _wts(_C248_TEXT)}


def test_mid_chunk_label_reassigns_only_the_post_label_region():
    """Hinh dang c248 movie_045: nhan MISSY giua chunk cua SPK_106 — dong
    truoc nhan giu cluster goc, dong sau nhan thanh Missy du cluster khong
    duoc map ra ten nao."""
    tags, _ = assign_speakers(split_chunks_at_labels([_c248()], NAMES),
                              [(0.0, 2.9), (3.1, 11.9)])
    assert tags == ["SPK_106", "LABEL::Missy"]
    assert apply_label_names(tags, [None, None]) == [None, "Missy"]


def test_label_at_text_start_covers_the_whole_chunk_without_word_timestamps():
    chunk = {"speaker_tag": "SPK_038", "start": 5.0, "end": 8.0,
             "english": "MEEMAW (quietly): There you go."}
    tags, _ = assign_speakers(split_chunks_at_labels([chunk], NAMES),
                              [(5.0, 8.0)])
    assert tags == ["LABEL::Meemaw"]


def test_label_on_an_unknown_tagged_chunk_still_yields_a_name():
    """Chunk sentinel UNKNOWN cua CAM++ truoc day vo dung; nhan dau text bien
    no thanh vung co ten (movie_045: "INGRAM: Sheldon?")."""
    chunk = {"speaker_tag": "UNKNOWN", "start": 0.0, "end": 2.0,
             "english": "INGRAM: Sheldon?"}
    tags, _ = assign_speakers(split_chunks_at_labels([chunk], NAMES),
                              [(0.0, 2.0)])
    assert apply_label_names(tags, [None]) == ["Ingram"]


def test_line_straddling_the_label_boundary_goes_to_the_dominant_side():
    split = split_chunks_at_labels([_c248()], NAMES)
    tags, _ = assign_speakers(split, [(2.0, 5.0)])   # 1s goc vs 2s nhan
    assert tags == ["LABEL::Missy"]


def test_line_straddling_the_label_boundary_with_an_exact_tie_is_none():
    split = split_chunks_at_labels([_c248()], NAMES)
    tags, _ = assign_speakers(split, [(2.0, 4.0)])   # 1s goc vs 1s nhan
    assert tags == [None]


def test_mid_text_label_without_word_timestamps_is_ignored():
    """Khong dinh vi duoc thi im lang, khong doan — chunk giu nguyen."""
    chunk = {"speaker_tag": "SPK_106", "english": _C248_TEXT,
             "start": 0.0, "end": 12.0}
    assert split_chunks_at_labels([chunk], NAMES) == [chunk]


def test_label_outside_the_character_list_does_not_split():
    chunk = {"speaker_tag": "SPK_001", "start": 0.0, "end": 2.0,
             "english": "NARRATOR: Once upon a time."}
    assert split_chunks_at_labels([chunk], NAMES) == [chunk]


def test_two_labels_in_one_chunk_create_two_regions():
    text = "MISSY: Did you cry? SHELDON: I did not."
    chunk = {"speaker_tag": "SPK_001", "english": text,
             "start": 0.0, "end": 9.0,
             "english_word_timestamps": _wts(text)}
    # 9 token: MISSY tai 0.0 (dau text), SHELDON tai token 4 -> 4.0s.
    tags, _ = assign_speakers(split_chunks_at_labels([chunk], NAMES),
                              [(0.0, 3.9), (4.1, 9.0)])
    assert tags == ["LABEL::Missy", "LABEL::Sheldon"]


def test_label_region_survives_the_min_score_filter():
    """Bang chung nhan la text; nguong embedding chi duoc loc vung tag goc."""
    tags, stats = assign_speakers(
        split_chunks_at_labels([_c248(score=0.1)], NAMES),
        [(0.0, 2.9), (3.1, 11.9)], min_score=0.5,
    )
    assert tags == [None, "LABEL::Missy"]
    assert stats.n_low_score == 1


def test_label_positions_locates_mid_text_label_by_word_timestamps():
    assert label_positions(_c248(), NAMES) == [(3.0, "Missy")]


def test_is_label_tag_distinguishes_synthetic_from_cluster_tags():
    assert is_label_tag("LABEL::Missy")
    assert not is_label_tag("SPK_106") and not is_label_tag(None)
