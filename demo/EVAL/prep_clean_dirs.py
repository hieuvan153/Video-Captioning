"""Task 5 Step 3 (E0): sinh demo/output/eval_clean/<movie>/ bang run_pipeline.py voi 3 model sach.

Moi movie trong holdout: tim .mkv theo ten tap trong data/Movie/video/**, chay run_pipeline
(--whisper_pt/--mbart_path/--adapter/--prompt_style v4 --max_scene_lines 24), roi doi ten
sang quy uoc eval: en.srt / rough.srt / captions.json / context.srt + source_episode.txt.
Voi 10 tap E5 copy captions.json tu eval_e5 truoc (skip scene-seg + VLM, giu cung ngu canh).
Idempotent; tuan tu; wait_gpu 18000 truoc moi phim.

CLI: van_env/bin/python demo/EVAL/prep_clean_dirs.py --whisper_pt P --mbart_path P --adapter P [--movies ...]
"""
from __future__ import annotations
import argparse, glob, os, shutil, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from EVAL import split  # noqa: E402
PY = sys.executable
OUT = os.path.join(ROOT, "output", "eval_clean")
E5_DIR = os.path.join(ROOT, "output", "eval_e5")


def gpu_free():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(o.strip().splitlines()[0])


def wait_gpu(mib=18000):
    while gpu_free() < mib:
        print(f"[wait_gpu] {time.strftime('%H:%M')} free={gpu_free()} < {mib}", flush=True); time.sleep(120)


def find_video(ep: str) -> str:
    c = glob.glob(os.path.join(REPO, "data", "Movie", "video", "**", ep + ".mkv"), recursive=True)
    if not c: raise FileNotFoundError(ep)
    return c[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--whisper_pt", required=True); ap.add_argument("--mbart_path", required=True)
    ap.add_argument("--adapter", required=True); ap.add_argument("--movies", nargs="*"); ap.add_argument("--llm_batch_size", type=int, default=8)
    a = ap.parse_args(); s = split.load(); movies = a.movies or s["holdout"]
    # run_pipeline chay voi cwd khac -> duong dan tuong doi khong giai duoc (whisper.load_model
    # tuong do la ten model va bao "not found"). Doi sang tuyet doi; giu nguyen neu la id tren hub.
    a.whisper_pt, a.mbart_path, a.adapter = (os.path.abspath(x) if os.path.exists(x) else x
                                             for x in (a.whisper_pt, a.mbart_path, a.adapter))
    for mv in movies:
        ep = split.episode_name(mv); mdir = os.path.join(OUT, mv); os.makedirs(mdir, exist_ok=True)
        with open(os.path.join(mdir, "source_episode.txt"), "w", encoding="utf-8") as f: f.write(ep + "\n")
        video = find_video(ep); base = os.path.splitext(os.path.basename(video))[0]
        names = {"en.srt": f"{base}.(Tiếng Anh).srt", "rough.srt": f"{base}.(Tiếng Việt_dich_tho).srt",
                 "captions.json": f"{base}.captions.json", "context.srt": f"{base}.(Tiếng Việt_tinh_chinh).srt"}
        if mv in s["test_e5"] and not os.path.exists(os.path.join(mdir, names["captions.json"])):
            shutil.copy(os.path.join(E5_DIR, mv, "captions.json"), os.path.join(mdir, names["captions.json"]))
            # run_pipeline skip scene-seg neu scenes.json + thu muc scene ton tai -> tao gia
            os.makedirs(os.path.join(mdir, f"{base}_scenes"), exist_ok=True)
            open(os.path.join(mdir, f"{base}_scenes", ".keep"), "w").close()
            shutil.copy(os.path.join(E5_DIR, mv, "captions.json"), os.path.join(mdir, f"{base}.scenes.json"))
        if all(os.path.exists(os.path.join(mdir, k)) for k in names):
            print(f"[skip] {mv} da du", flush=True); continue
        wait_gpu()
        cmd = [PY, os.path.join(ROOT, "run_pipeline.py"), "--video_path", video, "--output_dir", mdir,
               "--whisper_pt", a.whisper_pt, "--mbart_path", a.mbart_path, "--adapter", a.adapter,
               "--prompt_style", "v4", "--max_scene_lines", "24", "--llm_batch_size", str(a.llm_batch_size)]
        print(f"[run] {mv} {ep}", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"[FAIL] {mv} rc={r.returncode}", flush=True); continue
        for k, v in names.items():
            src = os.path.join(mdir, v)
            if os.path.exists(src) and not os.path.exists(os.path.join(mdir, k)):
                os.symlink(v, os.path.join(mdir, k))
        print(f"[done] {mv}", flush=True)
    print("PREP_CLEAN_DONE", flush=True)


if __name__ == "__main__":
    main()
