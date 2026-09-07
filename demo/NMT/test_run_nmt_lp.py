"""Cho: --length_penalty phai di het duong tu CLI xuong model.generate().
Chay: van_env/bin/python demo/NMT/test_run_nmt_lp.py   (khong can GPU, khong can model)"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import srt, datetime as dt
import run_nmt

seen = {}


class FakeTok:
    lang_code_to_id = {"vi_VN": 7}

    def __call__(self, batch, **kw):
        return types.SimpleNamespace(to=lambda d: {"input_ids": batch})

    def batch_decode(self, ids, **kw):
        return list(ids)


class FakeModel:
    def generate(self, **kw):
        seen.update(kw)
        return kw["input_ids"]


subs = [srt.Subtitle(1, dt.timedelta(0), dt.timedelta(seconds=1), "Hello there.")]
run_nmt.translate_en2vi(subs, FakeModel(), FakeTok(), "cpu", batch_size=8,
                        num_beams=5, length_penalty=4.0)
assert seen["length_penalty"] == 4.0, seen
assert seen["num_beams"] == 5, seen

sys.argv = ["run_nmt.py", "--input_srt", "x", "--output_srt", "y", "--length_penalty", "4.0"]
assert run_nmt.parse_args().length_penalty == 4.0
sys.argv = ["run_nmt.py", "--input_srt", "x", "--output_srt", "y"]
assert run_nmt.parse_args().length_penalty == 1.0, "mac dinh phai giu hanh vi cu"
print("OK: length_penalty di duoc tu CLI xuong generate(), mac dinh 1.0")
