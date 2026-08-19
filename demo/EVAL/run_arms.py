"""Chay cac arm A/B tren bo phim eval, moi arm mot file SRT rieng.

Arm:
    baseline  — refine tran, khong registry
    registry  — V0: 1 dong quan he toan phim tiem vao moi scene
    speaker   — V1: [SPEAKER: X] per-line + registry thu hep theo scene

Arm `speaker` can <movie>/speakers.json; driver tu build tu ket qua CAM++ da co
(khong chay lai speaker verification).

Chay lai TAT CA arm thay vi tai dung SRT cu: SRT A/B V0 duoc sinh truoc khi co
guardrail chong suy bien, va chinh mot dong suy bien o movie_015 da lam lech
headline V0 (+0.0223 -> that ra −0.0003, docs/eval/error_analysis_v0.md). Muon
so sanh cong bang thi ca 3 arm phai cung mot phien ban pipeline.

Idempotent: file da co thi bo qua, nen dut giua chung chay lai duoc.

CLI:
    van_env/bin/python demo/EVAL/run_arms.py --eval_dir demo/output/eval_ab \
        --arms baseline registry speaker
"""
import argparse
import glob
import os
import subprocess
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# arm -> (dung registry, dung speaker, registry_scope, file speaker)
ARMS: dict[str, tuple[bool, bool, str | None, str]] = {
    "baseline": (False, False, None, ""),
    "registry": (True, False, None, ""),
    "speaker": (True, True, "pair", "speakers.json"),      # V1: cap + ke nhau
    "speaker_any": (True, True, "any", "speakers.json"),   # V2a: mot dau canh
    # V2c: cung co che V1 nhung ten cluster dat o cap mua
    # (SPEAKER/season_map.py phai chay truoc de co speakers.season.json).
    "speaker_season": (True, True, "pair", "speakers.season.json"),
    # V3: tag lam sach cap dong — anchor chi nhan dau-text, vung LABEL::
    # override per-line, min_score tu sweep (docs/eval/speaker_quality_v3.md).
    "speaker_v3": (True, True, "pair", "speakers.v3.json"),
    # V3 + canh xung ho dao tu vocative (truong address_edges trong
    # speakers.v3.json — tro voi scope "pair" nen hai arm dung chung file).
    "speaker_v3_rel": (True, True, "pair_rel", "speakers.v3.json"),
    # V3/C.a: noi LOAI canh (vi_listener than toc, vi_self bat ky) duoi cung
    # pair-gating — do offline kich hoat 009: 3, 015: 15 scene.
    "speaker_v3_addr": (True, True, "pair_addr", "speakers.v3.json"),
    # V3.1: nhu speaker_v3 nhung nhan giua-chunk co >= 2 phieu cung ten van
    # anchor ca cluster (name_evidence.mid_anchor_votes) — nham lay lai tag
    # dung ma V3 xoa oan o 008/046 nhung van chan phieu don le kieu SPK_106.
    "speaker_v3_1": (True, True, "pair", "speakers.v3_1.json"),
}
TAGGED_GLOB = "data/speaker_verify_campp_en_full/series/*/*/episodes/{m}/{m}.tagged.json"

# File speakers nao driver nay build duoc -> flags them cho build_speakers.py.
# Ten file RIENG cho moi cau hinh la bat buoc: cache o day theo ton-tai-file,
# doi flag ma giu ten cu se lay nham file cu. speakers.v3.json khac
# speakers.json o CODE (anchor dau-text + LABEL:: override, V3), khong o flag:
# sweep offline da bac min_score (docs/eval/speaker_quality_v3.md — nang
# nguong den 0,65 chi nhich gender +0,009 ma coverage sap 0,68->0,22).
SPEAKER_BUILD_FLAGS: dict[str, list[str]] = {
    "speakers.json": [],
    "speakers.v3.json": [],
    "speakers.v3_1.json": ["--mid_anchor_votes", "2"],
}


def _run(cmd: list[str], log_path: str) -> None:
    print("$ " + " ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def ensure_speakers(movie: str, mdir: str, repo_root: str, log_path: str,
                    cache_dir: str, out_name: str = "speakers.json") -> str | None:
    out = os.path.join(mdir, out_name)
    if os.path.exists(out):
        print(f"  {out_name} da co, bo qua", flush=True)
        return out
    hits = glob.glob(os.path.join(repo_root, TAGGED_GLOB.format(m=movie)))
    if not hits:
        print(f"Warning: {movie}: khong tim thay tagged.json cua CAM++ - bo arm speaker",
              flush=True)
        return None
    _run([sys.executable, os.path.join(ROOT_DIR, "SPEAKER", "build_speakers.py"),
          "--tagged_json", hits[0],
          "--en_srt", os.path.join(mdir, "en.srt"),
          "--registry_json", os.path.join(mdir, "registry.json"),
          "--vlm_json", os.path.join(mdir, "captions.json"),
          "--cache_dir", cache_dir,
          "--output_json", out] + SPEAKER_BUILD_FLAGS[out_name], log_path)
    return out


def run_arm(arm: str, mdir: str, prefix: str, spk_built: dict,
            log_path: str, cache_dir: str, batch_size: int) -> str | None:
    use_registry, use_speaker, scope, spk_file = ARMS[arm]
    out_srt = os.path.join(mdir, f"{prefix}{arm}.srt")
    if os.path.exists(out_srt):
        print(f"  {os.path.basename(out_srt)} da co, bo qua", flush=True)
        return out_srt
    speakers_path = None
    if use_speaker:
        speakers_path = spk_built.get(spk_file)
        if speakers_path is None and spk_file not in SPEAKER_BUILD_FLAGS:
            # File dac thu khong build tu day (vd speakers.season.json cua
            # SPEAKER/season_map.py) — phai co san.
            speakers_path = os.path.join(mdir, spk_file)
            if not os.path.exists(speakers_path):
                print(f"Warning: {os.path.basename(mdir)}: thieu {spk_file} "
                      f"- bo arm {arm}", flush=True)
                return None
        if not speakers_path:
            return None
    # Config chot sau A/B dem 18-19/08 (docs/eval/degeneration_e5.md):
    # max_seq_length=2048 (A/B 4096 vs 2048 ra fallback y het, khop config
    # train cua anh ntVan) + --max_scene_lines 24 (sub-chunk: fallback
    # 336+090 tu 346 xuong 4). Giong nhau cho MOI arm.
    cmd = [sys.executable, os.path.join(ROOT_DIR, "LLM", "refine_llm.py"),
           "--en_srt", os.path.join(mdir, "en.srt"),
           "--vinai_srt", os.path.join(mdir, "rough.srt"),
           "--vlm_json", os.path.join(mdir, "captions.json"),
           "--output_srt", out_srt,
           "--cache_dir", cache_dir,
           "--max_seq_length", "2048",
           "--max_scene_lines", "24",
           "--llm_batch_size", str(batch_size)]
    if use_registry:
        cmd += ["--registry_json", os.path.join(mdir, "registry.json")]
    if use_speaker:
        cmd += ["--speaker_json", speakers_path, "--registry_scope", scope]
    _run(cmd, log_path)
    return out_srt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A/B arms on the eval set.")
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--arms", type=str, nargs="+", default=list(ARMS),
                        choices=list(ARMS))
    parser.add_argument("--movies", type=str, nargs="*", default=None)
    parser.add_argument("--prefix", type=str, default="v1_",
                        help="Tien to file output. Mac dinh 'v1_' de khong de "
                             "len SRT cua A/B V0 (bang chung cua error analysis).")
    parser.add_argument("--cache_dir", type=str,
                        default=os.path.join(ROOT_DIR, "cache"))
    parser.add_argument("--llm_batch_size", type=int, default=8)
    parser.add_argument("--log", type=str, default=None)
    args = parser.parse_args()

    repo_root = os.path.dirname(ROOT_DIR)
    eval_dir = os.path.abspath(args.eval_dir)
    movies = args.movies or sorted(
        d for d in os.listdir(eval_dir)
        if os.path.isdir(os.path.join(eval_dir, d))
    )
    log_path = args.log or os.path.join(eval_dir, "run_arms.log")
    print(f"Log chi tiet: {log_path}", flush=True)

    t_all = time.time()
    for movie in movies:
        mdir = os.path.join(eval_dir, movie)
        print(f"\n=== {movie} ===", flush=True)
        # Chi build file speakers ma mot arm duoc yeu cau se dung; arm dung
        # file ngoai SPEAKER_BUILD_FLAGS (speaker_season) tu kiem tra file
        # trong run_arm — khong dot GPU build mot file khong ai dung.
        spk_built: dict[str, str | None] = {}
        for fname in sorted({ARMS[a][3] for a in args.arms
                             if ARMS[a][1] and ARMS[a][3] in SPEAKER_BUILD_FLAGS}):
            spk_built[fname] = ensure_speakers(movie, mdir, repo_root, log_path,
                                               args.cache_dir, out_name=fname)
        for arm in args.arms:
            t0 = time.time()
            out = run_arm(arm, mdir, args.prefix, spk_built, log_path,
                          args.cache_dir, args.llm_batch_size)
            if out:
                print(f"  [{arm}] {os.path.basename(out)} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"\nXong trong {(time.time() - t_all) / 60:.1f} phut", flush=True)


if __name__ == "__main__":
    main()
