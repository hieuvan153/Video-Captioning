"""Test phan logic thuan cua ASR (khong nap Whisper). Chay: van_env/bin/python demo/ASR/test_asr_post.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SR = 16000


def _hand_example():
    # 100 s: A=[10,12], B=[13,14] (lien sau dem), C=[17,18] (cung chunk, khe bi cat), D=[30,31] (chunk moi)
    from asr_post import build_chunks
    raw = [dict(start=10 * SR, end=12 * SR), dict(start=13 * SR, end=14 * SR),
           dict(start=17 * SR, end=18 * SR), dict(start=30 * SR, end=31 * SR)]
    return build_chunks(raw, 100 * SR)


def test_build_chunks_padding_overlap_and_offsets():
    u = _hand_example()
    assert len(u) == 2
    A, B, C = u[0]
    assert (A["start"], A["end"]) == (9.8, 13.3) and (B["start"], B["end"]) == (13.3, 15.3)
    assert (C["start"], C["end"]) == (16.8, 19.3)
    for r in u[0]:
        assert abs(r["offset"] - (r["start"] - r["chunk_start"])) < 1e-9
    assert (u[1][0]["start"], u[1][0]["end"], u[1][0]["offset"]) == (29.8, 32.3, 29.8)


def test_map_segment_inside_straddle_and_fallback():
    from asr_post import map_segment
    g = _hand_example()[0]
    assert map_segment(g, 1.0, 2.0) == (10.8, 11.8)
    s, e = map_segment(g, 5.0, 6.0)
    assert (s, round(e, 6)) == (14.8, 17.3)
    s, e = map_segment(g, 7.0, 9.0)
    assert (round(s, 6), round(e, 6)) == (18.3, 19.8)


def test_map_segment_start_on_region_boundary_uses_next_region():
    """Segment bat dau DUNG chunk_end cua vung B (= chunk_start cua C): loi noi nam o C (16,8 s),
    truoc day khop vung B truoc -> cue bat dau som 1,5 s (15,3 s)."""
    from asr_post import map_segment
    s, _ = map_segment(_hand_example()[0], 5.5, 6.0)
    assert abs(s - 16.8) < 1e-9, s


def test_decode_env_rejects_empty_and_zero():
    from asr_post import decode_env
    assert decode_env({}) == ((0.0,), None)
    assert decode_env({"ASR_TEMPS": "", "ASR_BEAM": ""}) == ((0.0,), None)
    assert decode_env({"ASR_TEMPS": "0,0.2", "ASR_BEAM": "5"}) == ((0.0, 0.2), 5)
    assert decode_env({"ASR_BEAM": "0"}) == ((0.0,), None)


def test_segment_info_lives_next_to_srt():
    """segment_info.json ghi vao CWD bi lan chay sau ghi de; phai nam canh SRT dau ra."""
    from asr_post import segment_info_path
    assert segment_info_path("/x/y/phim.(Tiếng Anh).srt") == "/x/y/phim.(Tiếng Anh).segment_info.json"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("OK  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, "->", type(e).__name__, e)
    sys.exit(1 if fails else 0)
