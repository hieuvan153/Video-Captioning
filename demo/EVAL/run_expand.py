"""Driver mo rong eval len toan bo phim — idempotent, tuan tu tren GPU chia se.

Moi phim trong queue: prep (CPU, tu dataset) -> captions (scene_seg + VLM)
-> registry (build_registry, code da sua merge placeholder) -> speakers
(build_speakers) -> arms baseline + speaker (run_arms, prefix v1b_). Moi buoc
skip neu output da co, nen giet/chay lai bao nhieu lan cung duoc; mot phim
hong khong giet hang doi (ghi vao expand_failures.log roi di tiep).

Truoc moi buoc GPU: doi VRAM trong >= 20GB (GPU dung chung, nhu run_task8.sh).
Khi khoi dong: doi cac job build_registry.fixed dang chay xong da.

CLI:
    nohup van_env/bin/python demo/EVAL/run_expand.py \
        --queue docs/eval/expand_queue.txt > demo/output/eval_ab/expand.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from EVAL.prep_eval_dirs import (  # noqa: E402
    find_tagged,
    find_video,
    load_plan,
    write_movie,
)

PY = sys.executable
MIN_FREE_MIB = 20000


def gpu_free_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def wait_gpu(tag: str) -> None:
    waited = 0
    while True:
        free = gpu_free_mib()
        if free >= MIN_FREE_MIB:
            return
        print(f"[wait_gpu:{tag}] VRAM trong {free}MiB < {MIN_FREE_MIB}MiB, "
              f"doi 60s (da doi {waited}s)", flush=True)
        time.sleep(60)
        waited += 60
        if waited >= 3600:
            print(f"[wait_gpu:{tag}] qua 60 phut, thu chay tiep", flush=True)
            return


def wait_other_registry_jobs() -> None:
    while True:
        r = subprocess.run(["pgrep", "-f", r"build_registry\.py.*registry\.fixed"],
                           capture_output=True, text=True)
        if not r.stdout.strip():
            return
        print("[wait] rebuild registry.fixed dang chay, doi 120s", flush=True)
        time.sleep(120)


def run_step(cmd: list[str], log_path: str) -> bool:
    print("$ " + " ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"  -> FAIL (rc={rc}), xem {log_path}", flush=True)
    return rc == 0


def process_movie(movie: str, plan: dict, eval_dir: str, cache_dir: str,
                  fail_log: str, no_prep: bool = False) -> bool:
    entry = plan.get(movie)
    if entry is None:
        return _fail(fail_log, movie, "khong co trong plan.json")
    mdir = os.path.join(eval_dir, movie)
    log_path = os.path.join(mdir, "expand_steps.log")
    if not no_prep:
        write_movie(movie, eval_dir)

    captions = os.path.join(mdir, "captions.json")
    if not os.path.exists(captions):
        video = find_video(entry)
        if not video:
            return _fail(fail_log, movie, "thieu video")
        wait_gpu("captions")
        if not run_step([PY, os.path.join(ROOT_DIR, "EVAL", "make_captions.py"),
                         "--video", video, "--mdir", mdir,
                         "--cache_dir", cache_dir], log_path):
            return _fail(fail_log, movie, "captions FAIL")

    registry = os.path.join(mdir, "registry.json")
    if not os.path.exists(registry):
        wait_gpu("registry")
        if not run_step([PY, os.path.join(ROOT_DIR, "CHARACTER",
                                          "build_registry.py"),
                         "--en_srt", os.path.join(mdir, "en.srt"),
                         "--vlm_json", captions,
                         "--output_json", registry], log_path):
            return _fail(fail_log, movie, "registry FAIL")

    speakers = os.path.join(mdir, "speakers.json")
    if not os.path.exists(speakers):
        tagged = find_tagged(movie)
        if not tagged:
            return _fail(fail_log, movie, "thieu tagged.json")
        wait_gpu("speakers")
        if not run_step([PY, os.path.join(ROOT_DIR, "SPEAKER",
                                          "build_speakers.py"),
                         "--tagged_json", tagged,
                         "--en_srt", os.path.join(mdir, "en.srt"),
                         "--registry_json", registry,
                         "--vlm_json", captions,
                         "--cache_dir", cache_dir,
                         "--output_json", speakers], log_path):
            return _fail(fail_log, movie, "speakers FAIL")

    need = [a for a in ("baseline", "speaker")
            if not os.path.exists(os.path.join(mdir, f"v1b_{a}.srt"))]
    if need:
        wait_gpu("arms")
        if not run_step([PY, os.path.join(ROOT_DIR, "EVAL", "run_arms.py"),
                         "--eval_dir", eval_dir, "--movies", movie,
                         "--arms", "baseline", "speaker",
                         "--prefix", "v1b_", "--cache_dir", cache_dir],
                        log_path):
            return _fail(fail_log, movie, "arms FAIL")
    return True


def _fail(fail_log: str, movie: str, reason: str) -> bool:
    print(f"  {movie}: {reason}", flush=True)
    with open(fail_log, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{movie}\t{reason}\n")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand eval to all movies.")
    parser.add_argument("--queue", type=str, required=True,
                        help="File text: moi dong mot movie id, theo thu tu.")
    parser.add_argument("--eval_dir", type=str,
                        default=os.path.join(ROOT_DIR, "output", "eval_ab"))
    parser.add_argument("--cache_dir", type=str,
                        default=os.path.join(ROOT_DIR, "cache"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Chi xu ly N phim dau (mac dinh het queue).")
    parser.add_argument("--no_wait_registry", action="store_true",
                        help="Khong doi job rebuild registry.fixed (dung khi "
                             "queue nay uu tien hon, VRAM con du).")
    parser.add_argument("--no_prep", action="store_true",
                        help="Khong sinh en/rough/ref tu dataset (eval_e5: "
                             "en/rough la ASR cua anh ntVan, khac truc dong).")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    plan = load_plan()
    with open(args.queue, encoding="utf-8") as f:
        queue = [ln.strip() for ln in f if ln.strip()
                 and not ln.startswith("#")]
    if args.limit:
        queue = queue[:args.limit]
    fail_log = os.path.join(args.eval_dir, "expand_failures.log")

    if not args.no_wait_registry:
        wait_other_registry_jobs()
    n_ok = 0
    t0 = time.time()
    for i, movie in enumerate(queue, 1):
        print(f"\n########## [{time.strftime('%H:%M:%S')}] "
              f"{i}/{len(queue)} {movie} ##########", flush=True)
        if process_movie(movie, plan, args.eval_dir, args.cache_dir, fail_log,
                         no_prep=args.no_prep):
            n_ok += 1
    print(f"\nXong: {n_ok}/{len(queue)} phim OK, "
          f"{(time.time() - t0) / 3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
