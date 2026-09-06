"""Scene segmentation + VLM captions cho MOT phim eval — GPU, chay tach process.

Tai su dung dung step3_run_scene_seg + step4_run_vlm cua demo/run_pipeline.py
(cung checkpoint, cung tham so: fps mac dinh 1.0, max_frames 240, min_duration
3.0) de captions.json cua phim moi cung phan phoi voi 5 phim eval V0.

Scene .mp4 cat tam bi XOA sau khi caption xong (phim le 2h cat ra vai GB;
356 phim thi disk khong chiu noi) — captions.json + scenes.json giu lai.

CLI:
    van_env/bin/python demo/EVAL/make_captions.py \
        --video <path.mkv> --mdir demo/output/eval_ab/movie_001
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from run_pipeline import step3_run_scene_seg, step4_run_vlm  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scene seg + VLM captions for one eval movie.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--mdir", type=str, required=True,
                        help="Thu muc eval cua phim (captions.json ghi vao day).")
    parser.add_argument("--cache_dir", type=str,
                        default=os.path.join(ROOT_DIR, "cache"))
    parser.add_argument("--vlm_fps", type=float, default=1.0)
    parser.add_argument("--keep_scenes", action="store_true",
                        help="Giu scene .mp4 da cat (mac dinh xoa sau caption).")
    args = parser.parse_args()

    mdir = os.path.abspath(args.mdir)
    captions = os.path.join(mdir, "captions.json")
    if os.path.exists(captions):
        print(f"{captions} da co, bo qua", flush=True)
        return
    if not os.path.exists(args.video):
        raise SystemExit(f"Video khong ton tai: {args.video}")
    os.makedirs(mdir, exist_ok=True)
    scenes_json = os.path.join(mdir, "scenes.json")
    scenes_dir = os.path.join(mdir, "_scenes_tmp")

    step3_run_scene_seg(args.video, scenes_json, scenes_dir)
    step4_run_vlm(scenes_dir, scenes_json, captions, args.cache_dir,
                  args.vlm_fps)

    if not os.path.exists(captions):
        raise SystemExit("VLM khong ghi duoc captions.json")
    if not args.keep_scenes:
        shutil.rmtree(scenes_dir, ignore_errors=True)
    print(f"Captions OK: {captions}", flush=True)


if __name__ == "__main__":
    main()
