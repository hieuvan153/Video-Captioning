"""Xuat <arm>.srt cua moi movie trong eval_dir sang export/<arm>/<ten_tap>.(Tiếng Việt).srt (giao thuc Bang 4.7 / bleu_episode.py).
CLI: van_env/bin/python demo/EVAL/export_episode_srt.py --eval_dir demo/output/eval_clean --arms rough.srt nocontext.srt context.srt"""
import argparse, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from EVAL import split  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--eval_dir", required=True); ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--movies", nargs="*")
a = ap.parse_args()
movies = a.movies or sorted(d for d in os.listdir(a.eval_dir) if d.startswith("movie_"))
for arm in a.arms:
    out = os.path.join(a.eval_dir, "export", arm[:-4]); os.makedirs(out, exist_ok=True); n = 0
    for m in movies:
        src = os.path.join(a.eval_dir, m, arm)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, f"{split.episode_name(m)}.(Tiếng Việt).srt")); n += 1
    print(arm, "->", out, n, "files")
