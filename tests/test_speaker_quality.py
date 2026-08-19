"""Bo cham attribution offline — proxy gender/age, CHI de do (khong vao pipeline)."""
import pytest

from CHARACTER.registry_schema import Character, Registry
from EVAL.speaker_quality import (
    age_bucket,
    registry_profiles,
    score_names,
    variant_names,
)

REG = Registry((
    Character("C1", ("Sheldon", "Shelly"), "male", "child", (1,)),
    Character("C2", ("Meemaw",), "female", "elderly", (2,)),
))
PROFILES = registry_profiles(REG)


def test_age_bucket_boundaries_map_child_teen_adult_and_elderly():
    assert age_bucket(11.0) == "child"
    assert age_bucket(13.0) == "teen"
    assert age_bucket(20.0) == "adult"
    assert age_bucket(63.8) == "elderly"
    assert age_bucket(None) is None


def test_registry_profiles_index_all_aliases_casefold():
    assert PROFILES["shelly"] == ("male", "child")
    assert PROFILES["meemaw"] == ("female", "elderly")


def test_matching_gender_and_age_score_as_consistent():
    scores = score_names(["Sheldon"], [{"gender": "male", "age": 11.0}], PROFILES)
    assert scores["gender_consistency"] == 1.0
    assert scores["age_consistency"] == 1.0


def test_gender_mismatch_counts_against_the_assigned_name_in_the_cross_tab():
    """Dong giong nam bi gan ten nu — dung loi "Missy" cua movie_045."""
    gold = [{"gender": "male", "age": 11.0}, {"gender": "male", "age": 11.2}]
    scores = score_names(["Meemaw", "Meemaw"], gold, PROFILES)
    assert scores["gender_consistency"] == 0.0
    assert scores["per_name"]["Meemaw"]["male"] == 2


def test_unknown_gold_or_registry_gender_is_unscored_not_a_mismatch():
    reg = Registry((Character("C1", ("Sam",), "unknown", "unknown", (1,)),))
    scores = score_names(
        ["Sam", "Sheldon"],
        [{"gender": "male", "age": None}, {"gender": None, "age": None}],
        {**PROFILES, **registry_profiles(reg)},
    )
    assert scores["n_gender_scored"] == 0
    assert scores["gender_consistency"] is None


def test_coverage_counts_only_named_lines():
    gold = [{"gender": "male"}, {"gender": "male"}, {"gender": None}]
    scores = score_names(["Sheldon", None, None], gold, PROFILES)
    assert scores["coverage"] == pytest.approx(1 / 3, abs=1e-4)
    assert scores["n_named"] == 1


def test_length_mismatch_between_gold_and_speakers_raises():
    with pytest.raises(ValueError):
        score_names(["Sheldon"], [], PROFILES)


def test_variant_names_reuses_the_stored_cluster_mapping_without_an_llm():
    chunks = [{"speaker_tag": "SPK_1", "start": 0.0, "end": 2.0,
               "speaker_score": 0.9, "english": "hi"}]
    names, _ = variant_names(chunks, [(0.0, 2.0)], {"SPK_1": "Sheldon"})
    assert names == ["Sheldon"]


def test_variant_names_min_score_drops_weak_chunks():
    chunks = [{"speaker_tag": "SPK_1", "start": 0.0, "end": 2.0,
               "speaker_score": 0.4, "english": "hi"}]
    names, stats = variant_names(chunks, [(0.0, 2.0)], {"SPK_1": "Sheldon"},
                                 min_score=0.6)
    assert names == [None] and stats.n_low_score == 1


def test_variant_names_label_override_names_the_post_label_region():
    text = "Asked and answered. MISSY: Did you cry when you saw it? No."
    import re
    wts = [{"word": w, "start": float(i), "end": float(i + 1)}
           for i, w in enumerate(re.findall(r"[A-Za-z0-9']+", text))]
    chunks = [{"speaker_tag": "SPK_106", "english": text, "start": 0.0,
               "end": 12.0, "speaker_score": 0.9,
               "english_word_timestamps": wts}]
    names, _ = variant_names(chunks, [(0.0, 2.9), (3.1, 11.9)], {},
                             label_override=True,
                             valid_names=["Missy", "Sheldon"])
    assert names == [None, "Missy"]


def test_tag_retention_counts_lost_gained_and_changed():
    from EVAL.speaker_quality import tag_retention

    ref = ["Sheldon", "Meemaw", None, "Missy"]
    cand = ["Sheldon", None, "George Jr.", "Mary"]
    r = tag_retention(cand, ref)
    assert r["n_ref_named"] == 3
    assert r["n_lost"] == 1        # Meemaw -> None
    assert r["n_gained"] == 1      # None -> George Jr.
    assert r["n_changed"] == 1     # Missy -> Mary
    assert r["retention"] == round(1 - 1 / 3, 4)


def test_tag_retention_rejects_length_mismatch():
    from EVAL.speaker_quality import tag_retention

    with pytest.raises(ValueError):
        tag_retention(["A"], ["A", "B"])
