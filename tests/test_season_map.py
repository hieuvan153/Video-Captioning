import json

from SPEAKER.season_map import (
    map_season,
    pooled_evidence,
    rebuild_speakers,
)


def _chunk(tag, text, start=0.0, end=1.0, score=0.9):
    return {"speaker_tag": tag, "english": text,
            "start": start, "end": end, "speaker_score": score}


NAMES = ["Sheldon", "Meemaw", "Missy"]


def test_vocative_hints_do_not_cross_episode_boundary():
    """Tap 1 ket thuc bang cau goi "Sheldon,"; tap 2 mo dau bang cluster khac.
    Ghep hai tap thanh mot chuoi se tao phieu gia cho SPK_B — phai khong co."""
    ep1 = [_chunk("SPK_A", "Sheldon, come here right now please.")]
    ep2 = [_chunk("SPK_B", "This has nothing to do with anyone.")]
    _, _, _, _, hints = pooled_evidence({"ep1": ep1, "ep2": ep2}, NAMES)
    assert "SPK_B" not in hints

    # Cung cau truc do NAM TRONG mot tap thi phai co phieu.
    _, _, _, _, hints_one = pooled_evidence({"ep1": ep1 + ep2}, NAMES)
    assert hints_one.get("SPK_B") == {"Sheldon": 1}


def test_anchor_conflict_across_episodes_is_dropped():
    """SPK_X mang nhan "MEEMAW:" o tap nay va "MISSY:" o tap khac -> khong tin."""
    ep1 = [_chunk("SPK_X", "MEEMAW: There you go, baby.")]
    ep2 = [_chunk("SPK_X", "MISSY: I want a horse for my birthday.")]
    _, _, anchors, _, _ = pooled_evidence({"ep1": ep1, "ep2": ep2}, NAMES)
    assert "SPK_X" not in anchors

    _, _, anchors_ok, _, _ = pooled_evidence({"ep1": ep1}, NAMES)
    assert anchors_ok == {"SPK_X": "Meemaw"}


def test_representative_lines_pool_across_episodes():
    """Tap ngheo bang chung huong nho tap giau: reps cua SPK_Y gom thoai ca mua."""
    ep1 = [_chunk("SPK_Y", "Short.", score=0.5)]
    ep2 = [_chunk("SPK_Y", "A much longer line with plenty of context here.",
                  score=0.99)]
    reps, counts, _, _, _ = pooled_evidence({"ep1": ep1, "ep2": ep2}, NAMES)
    assert reps["SPK_Y"] == [
        "A much longer line with plenty of context here."]
    assert counts["SPK_Y"] == 2


def test_map_season_end_to_end_anchor_overrides_llm():
    ep1 = [_chunk("SPK_1", "MEEMAW: There you go, baby."),
           _chunk("SPK_2", "I have twelve theorems to prove before dinner.")]

    def fake_generate(system, user):
        assert "SPK_1" in user and "SPK_2" in user
        return json.dumps({"SPK_1": "Missy", "SPK_2": "Sheldon"})

    mapping = map_season({"ep1": ep1}, NAMES, fake_generate)
    # LLM noi Missy nhung nhan kich ban noi Meemaw -> anchor thang.
    assert mapping == {"SPK_1": "Meemaw", "SPK_2": "Sheldon"}


def test_rebuild_speakers_applies_new_mapping():
    payload = {
        "n_lines": 4,
        "speakers": ["Connie", None, "Missy", None],
        "line_tags": ["SPK_066", None, "SPK_106", "SPK_999"],
        "clusters": {"SPK_066": "Connie", "SPK_106": "Missy"},
        "n_named": 2,
        "align_stats": {"n_lines": 4},
    }
    out = rebuild_speakers(payload, {"SPK_066": "Meemaw"})
    assert out["speakers"] == ["Meemaw", None, None, None]
    assert out["line_tags"] == payload["line_tags"]
    # clusters chi giu tag co mat tren dong VA co ten trong season map.
    assert out["clusters"] == {"SPK_066": "Meemaw"}
    assert out["n_named"] == 1
    assert out["align_stats"] == {"n_lines": 4}
