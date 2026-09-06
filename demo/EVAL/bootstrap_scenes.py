"""Paired bootstrap (Koehn 2004) tren don vi scene cho hai arm, cung eval_dir/movies (giao thuc thesis_score).
CLI: van_env/bin/python demo/EVAL/bootstrap_scenes.py --eval_dir demo/output/eval_clean --a rough.srt --b context.srt [--movies ...]"""
import argparse, os, random, sys
import sacrebleu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "demo/cache/huggingface")
from EVAL import thesis_score as TS  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--eval_dir", required=True); ap.add_argument("--movies", nargs="*")
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()
movies = a.movies or sorted(d for d in os.listdir(a.eval_dir) if d.startswith("movie_"))
pa, pb = [], []
for m in movies:
    ta = TS.scene_texts(os.path.join(a.eval_dir, m), m, a.a); tb = TS.scene_texts(os.path.join(a.eval_dir, m), m, a.b)
    assert len(ta) == len(tb), (m, len(ta), len(tb)); pa += ta; pb += tb


def bleu(tr): return sacrebleu.corpus_bleu([h for _, h, _ in tr], [[r for _, _, r in tr]]).score
def chrf(tr): return sacrebleu.corpus_chrf([h for _, h, _ in tr], [[r for _, _, r in tr]]).score


for name, fn in (("BLEU", bleu), ("chrF", chrf)):
    d0 = fn(pb) - fn(pa); random.seed(0); wins = 0; ds = []
    for _ in range(a.n):
        idx = [random.randrange(len(pa)) for _ in pa]
        d = fn([pb[i] for i in idx]) - fn([pa[i] for i in idx]); ds.append(d); wins += d > 0
    ds.sort()
    print(f"{name}: {a.b} - {a.a} = {d0:+.2f}  95%CI=[{ds[int(.025*a.n)]:+.2f}, {ds[int(.975*a.n)]:+.2f}]  p(b<=a)={1-wins/a.n:.3f}  n_scenes={len(pa)}")
