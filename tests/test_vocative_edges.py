"""Canh xung ho dao tu vocative EN — theo cau truc nam dung tren cap dang noi
chuyen (V2b da do: canh kinship registry va cap cam mic la hai tap roi nhau)."""
from CHARACTER.registry_schema import Character, Registry
from SPEAKER.vocative_edges import mine_address_edges

REG = Registry((
    Character("C1", ("Sheldon",), "male", "child", (1,)),
    Character("C2", ("Mary",), "female", "adult", (2,)),
    Character("C3", ("George",), "male", "adult", (3,)),
))
NAMES = {"SPK_1": "Sheldon", "SPK_2": "Mary", "SPK_3": "George"}


def _c(tag, text):
    return {"speaker_tag": tag, "english": text, "start": 0.0, "end": 1.0}


def test_mom_vocative_yields_a_directed_edge_to_the_adjacent_named_speaker():
    chunks = [_c("SPK_2", "Dinner is ready."),
              _c("SPK_1", "Thanks, Mom."),
              _c("SPK_2", "Eat up."),
              _c("SPK_1", "Mom, can I go to Tam's?")]
    edges = mine_address_edges(chunks, NAMES, REG)
    forward = [e for e in edges if e["from_name"] == "Sheldon"]
    assert len(forward) == 1
    e = forward[0]
    assert e["to_name"] == "Mary"
    assert e["vi_listener"] == "mẹ" and e["vi_self"] == "con"
    assert e["confidence"] == "high"        # 3 phieu (2 hang xom + 1)


def test_a_single_vote_is_not_enough_to_keep_an_edge():
    chunks = [_c("SPK_1", "Thanks, Mom."), _c("SPK_2", "You're welcome.")]
    assert mine_address_edges(chunks, NAMES, REG) == []


def test_listener_gender_conflict_with_the_registry_drops_the_edge():
    """Goi "Mom" nhung hang xom ke la George (registry: male) — im lang."""
    chunks = [_c("SPK_3", "Hey."), _c("SPK_1", "Thanks, Mom."),
              _c("SPK_3", "Sure."), _c("SPK_1", "Mom, please.")]
    assert mine_address_edges(chunks, NAMES, REG) == []


def test_conflicting_address_terms_for_one_pair_are_dropped():
    """Cung mot nguoi nghe nhan ca "mẹ" lan "bà" — mau thuan cau truc, bo."""
    chunks = [_c("SPK_2", "Hello."), _c("SPK_1", "Thanks, Mom."),
              _c("SPK_2", "Here."), _c("SPK_1", "Okay, Grandma."),
              _c("SPK_2", "Go on.")]
    assert mine_address_edges(chunks, NAMES, REG) == []


def test_unnamed_neighbours_produce_no_edge():
    chunks = [_c("SPK_9", "Dinner is ready."),   # cluster khong co ten
              _c("SPK_1", "Thanks, Mom."),
              _c("SPK_9", "Eat up."),
              _c("SPK_1", "Mom, can I go?")]
    assert mine_address_edges(chunks, NAMES, REG) == []


def test_reciprocal_edge_is_emitted_at_medium_confidence():
    """B->A (me goi con) gan chac ve ngon ngu nhung la suy ra -> medium."""
    chunks = [_c("SPK_2", "Dinner is ready."),
              _c("SPK_1", "Thanks, Mom."),
              _c("SPK_2", "Eat up."),
              _c("SPK_1", "Mom, can I go to Tam's?")]
    edges = mine_address_edges(chunks, NAMES, REG)
    recip = [e for e in edges if e["from_name"] == "Mary"]
    assert len(recip) == 1
    e = recip[0]
    assert e["to_name"] == "Sheldon"
    assert e["vi_self"] == "mẹ" and e["vi_listener"] == "con"
    assert e["confidence"] == "medium"


def test_a_script_label_is_not_a_vocative():
    """"MOM: We're leaving." nghia la nguoi noi LA me, khong phai dang goi me."""
    chunks = [_c("SPK_2", "Hurry up."), _c("SPK_1", "MOM: We're leaving."),
              _c("SPK_2", "Coming."), _c("SPK_1", "MOM: Now.")]
    assert mine_address_edges(chunks, NAMES, REG) == []


def test_mid_sentence_mention_is_not_a_vocative():
    chunks = [_c("SPK_2", "Hello."), _c("SPK_1", "I told Mom about it."),
              _c("SPK_2", "Good."), _c("SPK_1", "My mom knows already.")]
    assert mine_address_edges(chunks, NAMES, REG) == []
