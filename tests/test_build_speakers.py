"""End-to-end orchestrator (khong GPU, generate_fn stub)."""
import re

from CHARACTER.registry_schema import Character, Registry
from SPEAKER.build_speakers import build_speakers


def _wts(text, t0=0.0, step=1.0):
    words = re.findall(r"[A-Za-z0-9']+", text)
    return [{"word": w, "start": t0 + i * step, "end": t0 + (i + 1) * step}
            for i, w in enumerate(words)]


def test_build_speakers_applies_label_override_end_to_end():
    """LLM khong map duoc gi ("{}") nhung dong sau nhan giua-chunk van ra ten
    truc tiep tu nhan; dong truoc nhan giu nguyen khong ten."""
    text = "Asked and answered. MISSY: Did you cry when you saw it? No."
    chunks = [{"speaker_tag": "SPK_106", "english": text,
               "start": 0.0, "end": 12.0, "speaker_score": 0.9,
               "english_word_timestamps": _wts(text)}]
    names, line_tags, mapping, stats, edges = build_speakers(
        chunks, [(0.0, 2.9), (3.1, 11.9)], ["Missy", "Sheldon"],
        lambda system, user: "{}",
    )
    assert names == [None, "Missy"]
    assert line_tags == ["SPK_106", "LABEL::Missy"]
    assert mapping == {} and edges == []


def test_build_speakers_mines_address_edges_when_given_a_registry():
    """Cluster co ten (qua anchor dau-text) goi "Mom" ke cluster co ten khac
    -> canh xung ho xuat hien trong phan tu thu 5."""
    registry = Registry((
        Character("C1", ("Sheldon",), "male", "child", (1,)),
        Character("C2", ("Mary",), "female", "adult", (2,)),
    ))
    chunks = [
        {"speaker_tag": "SPK_2", "english": "MARY: Dinner is ready.",
         "start": 0.0, "end": 1.0, "speaker_score": 0.9},
        {"speaker_tag": "SPK_1", "english": "SHELDON: Thanks, Mom.",
         "start": 1.0, "end": 2.0, "speaker_score": 0.9},
        {"speaker_tag": "SPK_2", "english": "Eat up.",
         "start": 2.0, "end": 3.0, "speaker_score": 0.9},
    ]
    names, line_tags, mapping, stats, edges = build_speakers(
        chunks, [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], ["Sheldon", "Mary"],
        lambda system, user: "{}", registry=registry,
    )
    assert mapping == {"SPK_1": "Sheldon", "SPK_2": "Mary"}
    forward = [e for e in edges if e["from_name"] == "Sheldon"]
    assert forward and forward[0]["vi_listener"] == "mẹ"
