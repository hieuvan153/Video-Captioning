"""Kiem tra dinh tuyen --arch cua run_pipeline.py. Khong can GPU, khong nap model.

    python demo/test_pipeline_arch.py
"""
import os, sys, types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_pipeline as rp


def run(argv):
    """Chay main() voi moi buoc thay bang stub, tra ve danh sach buoc da goi."""
    called = []
    orig = {}
    for name, ret in (("step1_extract_audio", None), ("step2_run_asr", "EN.srt"),
                      ("step3_run_scene_seg", None), ("step4_run_vlm", "cap.json"),
                      ("step5_run_nmt", "tho.srt"), ("step6_run_llm", "tinh.srt")):
        orig[name] = getattr(rp, name)
        setattr(rp, name, (lambda n, r: lambda *a, **k: (called.append(n), r)[1])(name, ret))
    orig_exists = os.path.exists
    os.path.exists = lambda p: True if p == argv[argv.index("--video_path") + 1] else orig_exists(p)
    sys.argv = ["run_pipeline.py"] + argv
    try:
        rp.main()
    finally:
        os.path.exists = orig_exists
        for name, fn in orig.items():
            setattr(rp, name, fn)
    return called


V = ["--video_path", "/khong/co/that.mkv", "--output_dir", "/tmp/_arch_test"]

v2 = run(V)
assert v2 == ["step1_extract_audio", "step2_run_asr", "step5_run_nmt"], v2

legacy = run(V + ["--arch", "legacy"])
assert legacy == ["step1_extract_audio", "step2_run_asr", "step3_run_scene_seg",
                  "step4_run_vlm", "step5_run_nmt", "step6_run_llm"], legacy

sys.argv = ["run_pipeline.py"] + V
assert rp.parse_args().arch == "v2", "mac dinh phai la v2"
assert rp.parse_args().llm_batch_size == 1, "batch>1 lam hong 31,8% dong"

print("OK: v2 chay 3 buoc, legacy chay 6 buoc, mac dinh dung.")
