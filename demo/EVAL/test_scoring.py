"""Test cac loi da xac minh trong khau cham diem (14/09). Chay: van_env/bin/python demo/EVAL/test_scoring.py"""
import os
import sys
import tempfile
import unicodedata
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import srt  # noqa: E402

NFC = "Anh ấy nói tôi đi."
NFD = unicodedata.normalize("NFD", NFC)


def _srt_file(texts):
    subs = [srt.Subtitle(i + 1, timedelta(seconds=2 * i), timedelta(seconds=2 * i + 1), t) for i, t in enumerate(texts)]
    fd, p = tempfile.mkstemp(suffix=".srt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
    return p


def test_loaders_normalize_nfc():
    """Tham chieu va hyp khac dang Unicode (NFC/NFD) phai thanh cung mot chuoi khi doc."""
    from EVAL import bleu47, film_ref_anchored, film_score
    p = _srt_file([NFD])
    assert bleu47.get_subs_from_srt(p)[0].content == NFC
    assert film_ref_anchored.load(p)[0].content == NFC
    assert film_score.load(p)[0].content == NFC


def test_pronoun_extractors_normalize_nfc():
    from EVAL.pronoun_lexicon import extract_pronouns
    from EVAL.thesis_score import _extract_thesis
    assert _extract_thesis(NFD) == _extract_thesis(NFC) and "tôi" in _extract_thesis(NFD)
    assert extract_pronouns(NFD) == extract_pronouns(NFC) and extract_pronouns(NFC)


def test_clean_custom_keeps_one_letter_interjection():
    """'Ừ.' la loi thoai, khong phai chu in hoa tren man hinh; dong in hoa nhieu chu van bi bo."""
    from EVAL.bleu47 import clean_custom
    assert clean_custom("Ừ.\nNHÀ THỜ BÁP-TÍT MEDFORD\nAnh đi đâu?") == "ừ. anh đi đâu?"


def test_ci95_interpolated():
    """Phan vi 2,5/97,5 noi suy nhu numpy, khong lay ds[int(p*n)] (lech mot order statistic)."""
    from EVAL.film_ref_anchored import ci95
    lo, hi = ci95(list(range(1000)))
    assert abs(lo - 24.975) < 1e-9 and abs(hi - 974.025) < 1e-9, (lo, hi)


def test_anchored_cli_defaults():
    """Giao thuc luan van dung --min_overlap_s 0.2; --selftest phai chay duoc khi khong co --ref."""
    from EVAL.film_ref_anchored import parse_args
    assert parse_args(["--ref", "r", "--src_en", "s", "--arms", "a=b"]).min_overlap_s == 0.2
    assert parse_args(["--selftest"]).selftest


def _C(a, b, t):
    return srt.Subtitle(0, timedelta(seconds=a), timedelta(seconds=b), t)


def _srt_times(rows):
    fd, p = tempfile.mkstemp(suffix=".srt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(srt.compose([srt.Subtitle(i + 1, timedelta(seconds=a), timedelta(seconds=b), t)
                             for i, (a, b, t) in enumerate(rows)]))
    return p


def test_overlap_text_splits_long_cue_without_duplication():
    """Mot cue arm trai hai cue tham chieu phai CHIA tu theo thoi gian giao, khong nhan ban sang ca hai."""
    from EVAL.film_ref_anchored import overlap_text
    ref = [_C(0, 1, "a b c d"), _C(1, 2, "e f g h")]
    assert overlap_text(ref, [_C(0, 2, "a b c d e f g h")]) == ["a b c d", "e f g h"]
    assert overlap_text(ref, [_C(0, 0.9, "x y"), _C(1.1, 2, "z")]) == ["x y", "z"]   # giao mot cue: nhu cu
    assert overlap_text(ref, [_C(0, 1.05, "p q")], 0.2) == ["p q", ""]             # tran duoi nguong
    out = overlap_text([_C(0, 2, "r"), _C(5, 7, "r"), _C(8, 10, "r")], [_C(0, 10, "w1 w2 w3 w4 w5 w6 w7")])
    assert sum(len(x.split()) for x in out) == 7, out                              # bao toan so tu


def test_split_half_rejects_basename_collision():
    """Hai arm cung ten file o hai thu muc khac nhau truoc day ghi de nhau trong HALFDIR."""
    import subprocess
    d1, d2, out = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
    for d in (d1, d2):
        open(os.path.join(d, "lp4.0.srt"), "w", encoding="utf-8").write(open(_srt_file(["x"]), encoding="utf-8").read())
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, "split_half.py"), "1", "a",
                        os.path.join(d1, "lp4.0.srt"), os.path.join(d2, "lp4.0.srt")],
                       env={**os.environ, "HALFDIR": out}, capture_output=True, text=True)
    assert r.returncode != 0 and "trung ten" in (r.stdout + r.stderr), (r.returncode, r.stdout, r.stderr)


def test_compare_arms_aligns_by_time_when_grids_differ():
    """Cung so cue KHONG co nghia la cung luoi cue: phai gioi theo thoi gian khi moc lech."""
    from EVAL.compare_arms import _aligned_hyps
    from EVAL.run_eval import align_by_time, load_srt
    ref_subs = load_srt(_srt_times([(0, 1, "a"), (2, 3, "b"), (4, 5, "c")]))
    hyp = _srt_times([(0, 1, "x"), (4, 5, "z"), (6, 7, "w")])
    assert _aligned_hyps(hyp, ref_subs) == [h for h, _ in align_by_time(load_srt(hyp), ref_subs)]


def test_film_wer_normalizes_numbers_and_apostrophes():
    """"twenty" va "20", dau nhay cong va thang la cung mot tu: truoc day bi tinh la loi thay the."""
    from EVAL.film_wer import norm
    assert norm("I've got twenty dollars.") == norm("I’ve got 20 dollars.")
    assert norm("(laughs) JOHN: Hello <i>there</i> ♪") == norm("hello there")


def test_thesis_score_scene_matching_has_no_missing_import():
    """scene_texts import LLM.scene_assign (khong ton tai tren nhanh nay) -> ModuleNotFoundError."""
    import inspect
    from EVAL import thesis_score
    assert "LLM.scene_assign" not in inspect.getsource(thesis_score.scene_texts)
    scenes = [{"start_time": 0.0, "end_time": 5.0}, {"start_time": None, "end_time": 9.0}, {"start_time": 5.0, "end_time": 9.0}]
    assert thesis_score.find_best_scene(2.0, scenes) == 0 and thesis_score.find_best_scene(7.0, scenes) == 2
    assert thesis_score.find_best_scene(20.0, scenes) == -1


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("OK  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, "->", type(e).__name__, e)
    sys.exit(1 if fails else 0)
