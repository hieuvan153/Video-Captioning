import datetime

import srt as srt_lib

from EVAL.run_eval import align_by_time, evaluate_lines, load_srt_lines


def _sub(i: int, s: float, e: float, text: str) -> srt_lib.Subtitle:
    return srt_lib.Subtitle(index=i, start=datetime.timedelta(seconds=s),
                            end=datetime.timedelta(seconds=e), content=text)


def test_evaluate_identical_lines():
    hyps = ["Bà ơi, cháu xin lỗi.", "Tôi không biết."]
    report = evaluate_lines(hyps, list(hyps), gold_pronouns=None)
    assert report["bleu"] == 100.0
    assert report["pronoun_f1"] == 1.0
    assert report["n_lines"] == 2


def test_evaluate_with_explicit_gold():
    # gold labels khac voi nhung gi lexicon rut tu ref → uu tien gold
    hyps = ["Mẹ ơi, con xin lỗi."]
    refs = ["Bà ơi, cháu xin lỗi."]
    report = evaluate_lines(hyps, refs, gold_pronouns=[["bà", "cháu"]])
    assert report["pronoun_f1"] == 0.0
    assert report["bleu"] < 100.0


def test_load_srt_lines_normalizes_whitespace(tmp_path):
    p = tmp_path / "x.srt"
    p.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nXin   chào\nbạn\n",
        encoding="utf-8",
    )
    assert load_srt_lines(str(p)) == ["Xin chào bạn"]


def test_align_by_time_merges_overlapping_hyp_lines():
    # ASR cat dong khac reference → align theo overlap thoi gian
    hyp = [_sub(1, 0, 2, "Xin chào"), _sub(2, 2.5, 4, "bạn khỏe không"),
           _sub(3, 10, 12, "Tạm biệt")]
    ref = [_sub(1, 0, 5, "Xin chào bạn khỏe không"), _sub(2, 9, 12, "Tạm biệt")]
    pairs = align_by_time(hyp, ref)
    assert pairs == [
        ("Xin chào bạn khỏe không", "Xin chào bạn khỏe không"),
        ("Tạm biệt", "Tạm biệt"),
    ]
