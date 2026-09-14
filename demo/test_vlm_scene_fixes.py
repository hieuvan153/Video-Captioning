"""Test cac loi da xac minh o VLM + scene_seg (14/09). Chay: demo_env/bin/python demo/test_vlm_scene_fixes.py"""
import inspect
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "VLM"))
sys.path.insert(0, os.path.join(ROOT, "scene_seg"))


def _run_vlm_with(model_cls):
    import run_vlm

    class Proc:
        @staticmethod
        def from_pretrained(*a, **k): return Proc()
        def __call__(self, **k): return {}
        def batch_decode(self, *a, **k): return [""]

    class Cap:
        def __init__(self, p): pass
        def get(self, prop): return 25.0 if prop == run_vlm.cv2.CAP_PROP_FPS else 250.0
        def release(self): pass

    d = tempfile.mkdtemp()
    open(os.path.join(d, "scene_0000.mp4"), "wb").close()
    out = os.path.join(d, "cap.json")
    saved = (run_vlm.AutoModelForCausalLM, run_vlm.AutoProcessor, run_vlm.cv2.VideoCapture, run_vlm.free_gpu_memory)
    run_vlm.AutoModelForCausalLM, run_vlm.AutoProcessor = model_cls, Proc
    run_vlm.cv2.VideoCapture, run_vlm.free_gpu_memory = Cap, (lambda *a, **k: None)
    try:
        run_vlm.run_vlm_captioning(d, out, cache_dir=d, device="cpu")
    finally:
        run_vlm.AutoModelForCausalLM, run_vlm.AutoProcessor, run_vlm.cv2.VideoCapture, run_vlm.free_gpu_memory = saved
    return out


def test_vlm_error_is_not_written_as_caption():
    """Loi khi sinh (vd OOM) truoc day thanh caption "Error during analysis: ..." va di vao prompt Gemma."""
    class Model:
        @staticmethod
        def from_pretrained(*a, **k): return Model()
        def generate(self, **k): raise RuntimeError("CUDA out of memory")
    rec = json.load(open(_run_vlm_with(Model)))[0]
    assert rec["caption"] == "" and "out of memory" in rec.get("error", ""), rec


def test_vlm_model_load_failure_raises():
    """Nap model loi truoc day chi in ra roi return -> pipeline chay tiep va do o buoc sau."""
    class Bad:
        @staticmethod
        def from_pretrained(*a, **k): raise OSError("khong co model")
    try:
        _run_vlm_with(Bad)
    except OSError:
        return
    raise AssertionError("phai nem loi khi nap model that bai")


def test_vlm_fps_default_matches_cli():
    import run_vlm
    assert inspect.signature(run_vlm.run_vlm_captioning).parameters["fps"].default == 1.0


def test_scene_cli_can_disable_cutting():
    """--cut_scenes type=bool: moi chuoi khac rong (ca "False") deu thanh True."""
    import predict_scenes
    assert predict_scenes.parse_args(["--video_path", "v.mp4"]).cut_scenes is True
    assert predict_scenes.parse_args(["--video_path", "v.mp4", "--no_cut_scenes"]).cut_scenes is False


def test_scene_seed_is_deterministic():
    import predict_scenes
    import torch.backends.cudnn as cudnn
    predict_scenes.setup_seed(100)
    assert cudnn.benchmark is False and cudnn.deterministic is True


def test_scene_temp_dir_is_per_run_and_env_untouched():
    """temp_dir mac dinh ./temp_predict_demo trong CWD bi rmtree cuoi lan chay -> hai lan chay dam nhau."""
    import predict_scenes
    src = inspect.getsource(predict_scenes.run_scene_segmentation)
    assert inspect.signature(predict_scenes.run_scene_segmentation).parameters["temp_dir"].default is None
    assert "CUDA_VISIBLE_DEVICES" not in src


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("OK  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, "->", type(e).__name__, str(e)[:150])
    sys.exit(1 if fails else 0)
