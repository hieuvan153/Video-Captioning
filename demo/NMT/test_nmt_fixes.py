"""Test cac loi da xac minh o tang NMT (14/09). Chay: van_env/bin/python demo/NMT/test_nmt_fixes.py"""
import datetime as dt
import inspect
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import srt  # noqa: E402


def test_write_srt_keeps_empty_and_zero_length_cues():
    """srt.compose mac dinh bo cue rong / dai 0 s va danh so lai -> SRT tho it cue hon SRT EN."""
    import run_nmt
    T = lambda s: dt.timedelta(seconds=s)
    subs = [srt.Subtitle(1, T(0), T(1), "a"), srt.Subtitle(2, T(2), T(3), ""), srt.Subtitle(3, T(4), T(4), "b")]
    p = os.path.join(tempfile.mkdtemp(), "x.srt")
    run_nmt.write_srt(subs, p)
    assert len(list(srt.parse(open(p, encoding="utf-8").read()))) == 3


def test_split_sentences_merges_punctuation_fragments():
    import run_nmt
    assert run_nmt.split_sentences("Wait... what?!") == ["Wait...", "what?!"]
    assert run_nmt.split_sentences("No! Yes.") == ["No!", "Yes."]
    assert run_nmt.split_sentences("  ") == []


def test_clean_cue_strips_tags_and_newlines():
    import run_nmt
    assert run_nmt.clean_cue("<i>Hello</i>\n  there ") == "Hello there"
    assert run_nmt.clean_cue("<font color='red'>Hi</font>") == "Hi"


def test_tokenizer_length_is_capped():
    """model_max_length cua mBART la 1e30 nen truncation=True khong cat gi; can max_length <= 1024."""
    import run_nmt
    seen = {}

    class Enc(dict):
        def to(self, d): return self

    class Tok:
        lang_code_to_id = {"vi_VN": 1}
        def __call__(self, batch, **kw):
            seen.update(kw); import torch; return Enc(input_ids=torch.zeros(len(batch), 1))
        def batch_decode(self, ids, **kw): return ["x"] * len(ids)

    class Model:
        def generate(self, **kw):
            import torch; return torch.zeros(kw["input_ids"].shape[0], 1)

    subs = [srt.Subtitle(1, dt.timedelta(0), dt.timedelta(seconds=1), "Hello.")]
    run_nmt.translate_en2vi(subs, Model(), Tok(), "cpu", batch_size=4)
    assert seen.get("max_length") == 1024 and seen.get("truncation") is True, seen


def test_gemmax2_clean_keeps_multiline_translation():
    import run_gemmax2
    assert run_gemmax2.clean("Xin chào.\nBạn khỏe không?\nEnglish: foo") == "Xin chào. Bạn khỏe không?"
    assert run_gemmax2.clean("\n  Dạ.  \n") == "Dạ."


def test_gemmax2_does_not_truncate_prompt_from_the_right():
    import run_gemmax2
    assert "max_length=512" not in inspect.getsource(run_gemmax2.translate)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("OK  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, "->", type(e).__name__, e)
    sys.exit(1 if fails else 0)
