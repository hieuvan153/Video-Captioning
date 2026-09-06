"""Task 6 (E0): cac hang con lai trong eval_clean/<movie>/: nocontext.srt (E4), seamless.srt (E3, chi test_out),
llm_direct.srt (E2, chi test_e5). Idempotent, wait_gpu truoc moi buoc.
CLI: van_env/bin/python demo/EVAL/run_clean_rows.py --adapter <adapter_clean> [--rows nocontext seamless llm_direct] [--movies ...]"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from EVAL import split  # noqa: E402
from EVAL.prep_clean_dirs import wait_gpu  # noqa: E402
PY = sys.executable; SEAM_PY = os.path.join(REPO, "seamlessm4t", "bin", "python")
OUT = os.path.join(ROOT, "output", "eval_clean")


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n"); f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO).returncode


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True)
    ap.add_argument("--rows", nargs="*", default=["nocontext", "seamless", "llm_direct"]); ap.add_argument("--movies", nargs="*")
    a = ap.parse_args(); s = split.load(); movies = a.movies or s["holdout"]
    for mv in movies:
        mdir = os.path.join(OUT, mv); log = os.path.join(mdir, "rows.log")
        if "nocontext" in a.rows and not os.path.exists(os.path.join(mdir, "nocontext.srt")):
            caps = json.load(open(os.path.join(mdir, "captions.json"), encoding="utf-8"))
            for c in caps: c["caption"] = "None"
            json.dump(caps, open(os.path.join(mdir, "captions_none.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            wait_gpu(18000)
            rc = run([PY, os.path.join(ROOT, "LLM", "refine_llm.py"), "--en_srt", os.path.join(mdir, "en.srt"), "--vinai_srt", os.path.join(mdir, "rough.srt"),
                      "--vlm_json", os.path.join(mdir, "captions_none.json"), "--output_srt", os.path.join(mdir, "nocontext.srt"),
                      "--adapter_model_name", a.adapter, "--prompt_style", "v4", "--max_scene_lines", "24", "--max_seq_length", "2048",
                      "--cache_dir", os.path.join(ROOT, "cache")], log)
            print(f"[nocontext] {mv} rc={rc}", flush=True)
        if "seamless" in a.rows and mv in s["test_out"] and not os.path.exists(os.path.join(mdir, "seamless.srt")):
            wav = [l for l in open(os.path.join(mdir, "source_episode.txt"), encoding="utf-8")][0].strip()
            wavp = os.path.join(mdir, wav + ".wav")
            wait_gpu(18000)
            rc = run([SEAM_PY, os.path.join(REPO, "ASR", "run_seamless_one.py"), wavp, os.path.join(mdir, "seamless.srt")], log)
            print(f"[seamless] {mv} rc={rc}", flush=True)
        if "llm_direct" in a.rows and mv in s["test_e5"] and not os.path.exists(os.path.join(mdir, "llm_direct.srt")):
            wait_gpu(18000)
            rc = run([PY, os.path.join(REPO, "ASR", "run_llm_direct_one.py"), os.path.join(mdir, "en.srt"), os.path.join(mdir, "llm_direct.srt")], log)
            print(f"[llm_direct] {mv} rc={rc}", flush=True)
    print("ROWS_DONE", flush=True)


if __name__ == "__main__":
    main()
