from SPEAKER.name_evidence import (
    apply_evidence,
    mention_exclusions,
    script_label_anchors,
    vocative_hints,
)

NAMES = ["Sheldon", "Meemaw", "George Jr.", "Adrian"]


def _c(tag, text):
    return {"speaker_tag": tag, "english": text}


def test_script_label_names_its_own_cluster():
    chunks = [_c("SPK_078", 'GEORGE JR.: The first one is, "Most people."'),
              _c("SPK_038", "MEEMAW (quietly): There you go.")]
    assert script_label_anchors(chunks, NAMES) == {"SPK_078": "George Jr.",
                                                   "SPK_038": "Meemaw"}


def test_mid_text_label_does_not_anchor_the_whole_cluster():
    """Nhan giua chunk danh dau doi speaker BEN TRONG chunk — chi noi ve phan
    duoi. Chinh chuoi nay (movie_045 c248) tung dat ten "Missy" cho ca cluster
    65 chunk cua Sheldon; gio no la viec cua label_override cap dong."""
    chunks = [_c("SPK_106",
                 "Asked and answered. MISSY: Did you cry when you saw it? No.")]
    assert script_label_anchors(chunks, ["Missy", "Sheldon"]) == {}


def test_label_after_a_leading_stage_direction_still_anchors():
    chunks = [_c("SPK_001", "(school bell rings) SHELDON: Good morning.")]
    assert script_label_anchors(chunks, NAMES) == {"SPK_001": "Sheldon"}


def test_conflicting_labels_in_one_cluster_are_dropped():
    chunks = [_c("SPK_001", "SHELDON: hi"), _c("SPK_001", "MEEMAW: bye")]
    assert script_label_anchors(chunks, NAMES) == {}


def test_labels_outside_the_character_list_are_ignored():
    assert script_label_anchors([_c("SPK_001", "NARRATOR: Once...")], NAMES) == {}


def test_repeated_mention_excludes_that_name():
    """Nguoi ta goi ten nguoi khac, khong goi ten minh."""
    chunks = [_c("SPK_106", "Adrian, I can't believe you"),
              _c("SPK_106", "Adrian, please listen")]
    assert mention_exclusions(chunks, NAMES) == {"SPK_106": {"Adrian"}}


def test_a_single_mention_is_not_enough_to_exclude():
    chunks = [_c("SPK_106", "Adrian, I can't believe you")]
    assert mention_exclusions(chunks, NAMES) == {}


def test_vocative_votes_for_the_next_cluster_not_the_current_one():
    chunks = [_c("SPK_001", "Sheldon, are you okay?"), _c("SPK_002", "I'm fine.")]
    assert vocative_hints(chunks, NAMES) == {"SPK_002": {"Sheldon": 1}}


def test_no_vocative_vote_when_the_same_cluster_keeps_talking():
    chunks = [_c("SPK_001", "Sheldon, are you okay?"), _c("SPK_001", "Answer me.")]
    assert vocative_hints(chunks, NAMES) == {}


def test_name_merely_discussed_is_not_a_vocative():
    chunks = [_c("SPK_001", "I saw Sheldon at the store today."),
              _c("SPK_002", "Really?")]
    assert vocative_hints(chunks, NAMES) == {}


def test_script_label_is_not_counted_as_a_mention():
    """"ADULT SHELDON: I was thrilled" la bang chung nguoi noi LA Adult
    Sheldon; dem nhan nhu mot lan nhac ten se loai dung cai ten dung."""
    chunks = [
        {"speaker_tag": "SPK_001",
         "english": "ADULT SHELDON (V.O.): I was thrilled."},
        {"speaker_tag": "SPK_001",
         "english": "ADULT SHELDON: Meemaw had other plans."},
    ]
    exc = mention_exclusions(chunks, ["Adult Sheldon", "Meemaw"])
    assert exc.get("SPK_001", set()) == set()


def test_mentions_outside_the_label_still_count():
    chunks = [
        {"speaker_tag": "SPK_001", "english": "MEEMAW: Sheldon, come here."},
        {"speaker_tag": "SPK_001", "english": "MEEMAW: Sheldon, eat up."},
    ]
    exc = mention_exclusions(chunks, ["Sheldon", "Meemaw"])
    assert exc == {"SPK_001": {"Sheldon"}}


def test_anchor_outranks_exclusion_on_the_same_cluster():
    """Anchor va exclusion cham nhau (nhan kich ban tung bi dem la mention):
    anchor phai thang va khong duoc bao cao "dropped" ao."""
    out, fixed, dropped = apply_evidence(
        {"SPK_001": "Adult Sheldon"},
        {"SPK_001": "Adult Sheldon"},
        {"SPK_001": {"Adult Sheldon"}},
    )
    assert out == {"SPK_001": "Adult Sheldon"}
    assert fixed == 0 and dropped == 0


def test_anchor_overrides_the_llm():
    out, fixed, dropped = apply_evidence(
        {"SPK_078": "Meemaw"}, {"SPK_078": "George Jr."}, {}
    )
    assert out == {"SPK_078": "George Jr."} and fixed == 1 and dropped == 0


def test_excluded_name_is_dropped_rather_than_replaced():
    out, fixed, dropped = apply_evidence(
        {"SPK_106": "Adrian"}, {}, {"SPK_106": {"Adrian"}}
    )
    assert out == {} and dropped == 1


def test_anchor_fills_in_a_cluster_the_llm_left_unknown():
    out, fixed, _ = apply_evidence({}, {"SPK_078": "George Jr."}, {})
    assert out == {"SPK_078": "George Jr."} and fixed == 0


def test_evidence_leaves_agreeing_answers_alone():
    out, fixed, dropped = apply_evidence(
        {"SPK_078": "George Jr."}, {"SPK_078": "George Jr."}, {}
    )
    assert out == {"SPK_078": "George Jr."} and fixed == 0 and dropped == 0


def test_mid_label_with_two_votes_anchors_under_v3_1_policy():
    """V3.1: cung ten o vi tri giua-chunk tai >= 2 chunk khac nhau cua cluster
    thi van anchor — mot phieu don le (SPK_106 "MISSY" x1) van bi loai."""
    chunks = [_c("SPK_027", "Fine. GEORGE JR.: Give it back."),
              _c("SPK_027", "No way. GEORGE JR.: I mean it."),
              _c("SPK_106", "Asked and answered. MISSY: Did you cry? No.")]
    assert script_label_anchors(chunks, ["George Jr.", "Missy"],
                                mid_anchor_votes=2) == {"SPK_027": "George Jr."}
    # Mac dinh (None) van la V3: khong mid-anchor nao ca.
    assert script_label_anchors(chunks, ["George Jr.", "Missy"]) == {}


def test_mid_votes_count_chunks_not_occurrences():
    chunks = [_c("SPK_027", "Uh. GEORGE JR.: Stop. GEORGE JR.: Now.")]
    assert script_label_anchors(chunks, ["George Jr."],
                                mid_anchor_votes=2) == {}


def test_start_and_qualified_mid_conflict_drops_cluster():
    chunks = [_c("SPK_001", "SHELDON: hi"),
              _c("SPK_001", "Well. MEEMAW: one"),
              _c("SPK_001", "So. MEEMAW: two")]
    assert script_label_anchors(chunks, NAMES, mid_anchor_votes=2) == {}
