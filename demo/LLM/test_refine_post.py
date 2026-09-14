"""Test logic hau xu ly cua tang LLM (khong nap Gemma). Chay: van_env/bin/python demo/LLM/test_refine_post.py"""
import datetime as dt
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import srt  # noqa: E402


def test_existing_alignment_and_lock_behaviour_kept():
    """Hanh vi da do cho pm0.90 phai giu nguyen voi dau vao binh thuong."""
    from refine_post import align_lines, lock_len, merge_pron
    assert align_lines(["a b c", "d e f"], ["a b", "c", "d e f"]) == ["a b", "d e f"]
    assert align_lines(["x", "y"], ["a", "b", "c"]) == [None, None]
    assert lock_len(["Cô thật xinh đẹp.", "b c d e"], ["Em thật đẹp.", "b c d e"]) == ["Em thật xinh đẹp.", "b c d e"]
    assert merge_pron("Cô ấy đến rồi.", "Chị ấy đến rồi.") == "Chị ấy đến rồi."


def test_drop_truncated_tail():
    """Sinh cham max_new_tokens (khong co EOS) thi dong cuoi la manh cut, khong duoc gan vao cue."""
    from refine_post import drop_truncated_tail
    assert drop_truncated_tail(["Anh đi đâu?", "Em về"], hit_limit=True) == ["Anh đi đâu?"]
    assert drop_truncated_tail(["Anh đi đâu?", "Em về."], hit_limit=False) == ["Anh đi đâu?", "Em về."]


def test_strip_numbering_only_when_all_lines_numbered():
    from refine_post import strip_numbering
    assert strip_numbering(["1. Anh đi đâu?", "2. Em về."]) == ["Anh đi đâu?", "Em về."]
    assert strip_numbering(["3. Tháng này", "Anh về."]) == ["3. Tháng này", "Anh về."]
    assert strip_numbering(["1. a", "3. b"]) == ["1. a", "3. b"]


def test_degenerate_chunk_detected():
    """v7 chunk 91 cue: model lap '- Tôi. - Tôi.' (khong co trong ban tho) va QHD gan vao 61 cue."""
    from refine_post import is_degenerate
    assert is_degenerate(["- Tôi. - Tôi."] * 30, ["Không.", "Anh..."] * 15)
    assert not is_degenerate(["Anh.", "Em.", "Ừ.", "Không.", "Đi thôi."], ["Anh.", "Em.", "Ừ.", "Không.", "Đi."])
    assert not is_degenerate(["Ừ.", "Ừ.", "Ừ."], ["Ừ.", "Ừ.", "Ừ."])


def test_repetition_already_in_rough_is_not_degenerate():
    """Ode to Joy chunk 30 ("Charlie!" x3) va chunk 122 (loi bai hat lap): ban tho da lap san."""
    from refine_post import is_degenerate
    assert not is_degenerate(["Charlie! Ôi Chúa ơi, Charlie!", "Charlie!", "Charlie!", "Charlie!"],
                             ["Charlie! Ôi Chúa ơi, Charlie!", "Charlie!", "Charlie!", "Charlie!"])
    song = ["Với cơ thể anh bên cạnh tôi", "Hơi thở hổn hển", "Với cơ thể anh bên cạnh tôi",
            "Tôi biết hết", "Với cơ thể anh bên cạnh tôi"]
    assert not is_degenerate(song, song + ["Tất cả dấu hiệu nguy hiểm"])


def test_short_rough_line_needs_closer_match():
    """Dong tho ngan ('Không.') gan duoc voi dong rac chi vi ty le giong 0,21 >= 0,2."""
    from refine_post import align_lines
    assert align_lines(["Không."], ["- Tôi. - Tôi."]) == [None]
    assert align_lines(["Không."], ["Không!"]) == ["Không!"]


def test_cap_chunks_splits_long_runs_evenly():
    from refine_post import cap_chunks, chunk_by_gap
    out = cap_chunks([list(range(45)), [45, 46]], 20)
    assert [len(c) for c in out] == [15, 15, 15, 2] and sum(out, []) == list(range(47))
    T = lambda s: dt.timedelta(seconds=s)
    subs = [srt.Subtitle(i + 1, T(i), T(i + 0.9), "x") for i in range(50)]  # khong co khoang lang
    assert max(len(c["indices"]) for c in chunk_by_gap(subs, 2.0, 20)) <= 20


def test_lock_len_catches_merged_lines():
    """Dong LLM gop hai cue gan vao cue sau -> cue sau hien noi dung cue truoc. Khoa hai phia."""
    from refine_post import lock_len
    assert lock_len(["Anh đi đâu?", "Em về."], [None, "Anh đi đâu? Em về."]) == [None, "Em về."]


def test_merge_pron_keeps_sentence_case_and_ignores_unaccented_words():
    from refine_post import merge_pron
    assert merge_pron("Anh có khỏe không?", "Có khỏe không?") == "Có khỏe không?"
    assert merge_pron("Nhà có ba con mèo.", "Nhà có bà cô mèo.") == "Nhà có ba con mèo."


def test_assign_scenes_uses_nearest_scene_outside_all_scenes():
    from refine_post import assign_scenes
    scenes = [{"start_time": 7.4, "end_time": 20.0}, {"start_time": 20.0, "end_time": 40.0}]
    idx, n_outside = assign_scenes([1.0, 10.0, 39.0, 55.0], scenes)
    assert idx == [0, 0, 1, 1] and n_outside == 2


def test_write_srt_keeps_every_cue():
    from refine_post import write_srt
    T = lambda s: dt.timedelta(seconds=s)
    p = os.path.join(tempfile.mkdtemp(), "o.srt")
    write_srt([srt.Subtitle(1, T(0), T(1), "a"), srt.Subtitle(2, T(1), T(1), "b")], p)
    assert len(list(srt.parse(open(p, encoding="utf-8").read()))) == 2


def test_system_prompt_matches_training_format():
    """Adapter mac dinh duoc train voi prompt da bo thut le (clean_prompt): NLL cau tra loi vang 1,0189 so voi
    1,0394 khi giu thut le nhu refine_llm dang lam (60 mau v3, 14/09)."""
    from refine_post import build_system_prompt
    sp = build_system_prompt("Hai nguoi dang cai nhau.")
    lines = sp.splitlines()
    assert lines[0].startswith("You are a professional Vietnamese subtitle editor")
    assert all(line == line.lstrip() for line in lines), [line for line in lines if line != line.lstrip()][:2]
    assert "<Scene Context>\nHai nguoi dang cai nhau.\n</Scene Context>" in sp
    assert build_system_prompt("x", system_prompt="Rewrite to natural Vietnamese subtitle.") == "Rewrite to natural Vietnamese subtitle."


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("OK  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, "->", type(e).__name__, e)
    sys.exit(1 if fails else 0)
