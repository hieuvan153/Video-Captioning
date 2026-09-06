"""Dung lai SRT post-hoc tu debug JSON cua refine_llm.py, ap dung guard cap
chunk (is_degenerate_chunk) va cap dong (has_prompt_leak, is_degenerate_line)
KHONG can chay lai GPU. refine_llm.py dung do_sample=False (greedy) nen chay
lai se ra y het — moi arm da luu du du lieu trong *.srt.json de dung lai.

CLI:
  van_env/bin/python demo/EVAL/rebuild_guarded_srt.py \
      --debug_json out/arm.srt.json --rough_srt rough.srt \
      --out out/arm.guarded.srt [--report out/arm.guard_report.json] [--no_guard]

--no_guard: tat toan bo guard, chi dung de KIEM CHEO voi SRT goc (phai giong
het), khong dung khi dung ban SRT that.
"""
import argparse
import ast
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import srt  # noqa: E402
from LLM.output_guard import has_prompt_leak, is_degenerate_chunk, is_degenerate_line  # noqa: E402


def _translations(chunk: dict) -> list:
    """translations co the la list (JSON dung) hoac chuoi repr Python (bug ghi debug
    JSON cu) -> phai dung ast.literal_eval, json.loads se hong voi chuoi repr."""
    tr = chunk["translations"]
    if isinstance(tr, str):
        return ast.literal_eval(tr)
    return tr


def _chunk_reason(chunk: dict, translations: list) -> str | None:
    """Goi is_degenerate_chunk dung ban chat luat 1 (so dong MODEL sinh, khong phai
    so dong da can chinh ve n_src). Debug JSON chi luu n_model_lines dang so, khong
    luu lai noi dung tho truoc khi can chinh, nen khi n_model_lines != n_src ta tao
    mot danh sach dung DO DAI do (noi dung khong quan trong: luat 1 tra ve ngay tu do
    dai, luat 2 khong bao gio duoc doc toi). Khi so dong khop nhau, kiem luat 2 tren
    noi dung refined_vietnamese that."""
    n_src = int(chunk["n_src"])
    n_model_lines = int(chunk["n_model_lines"])
    if n_model_lines != n_src:
        return is_degenerate_chunk([""] * n_model_lines, n_src)
    refined = [t["refined_vietnamese"] for t in translations]
    return is_degenerate_chunk(refined, n_src)


def _ranges(sorted_indices: list) -> list:
    if not sorted_indices:
        return []
    out = []
    start = prev = sorted_indices[0]
    for i in sorted_indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append([start, prev])
        start = prev = i
    out.append([start, prev])
    return out


def rebuild(debug_json_path: str, rough_srt_path: str, out_path: str,
            report_path: str = None, no_guard: bool = False):
    with open(debug_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(rough_srt_path, "r", encoding="utf-8") as f:
        rough_subs = list(srt.parse(f.read()))

    content = {s.index: s.content for s in rough_subs}  # mac dinh: giu nguyen rough
    fallback_cue_idx = []
    chunk_fallbacks = []

    for chunk in chunks:
        translations = _translations(chunk)
        indices = [t["subtitle_index"] for t in translations]
        n_src = int(chunk["n_src"])
        n_model_lines = int(chunk["n_model_lines"])
        reason = None if no_guard else _chunk_reason(chunk, translations)
        if reason:
            for t in translations:
                content[t["subtitle_index"]] = t["rough_vietnamese"]
            fallback_cue_idx.extend(indices)
            chunk_fallbacks.append({
                "scene_index": chunk["scene_index"],
                "n_src": n_src,
                "n_model_lines": n_model_lines,
                "reason": reason,
                "cue_range": [min(indices), max(indices)] if indices else None,
            })
            continue
        for t in translations:
            line = t["refined_vietnamese"]
            if not no_guard and (has_prompt_leak(line) or is_degenerate_line(line)):
                content[t["subtitle_index"]] = t["rough_vietnamese"]
                fallback_cue_idx.append(t["subtitle_index"])
            else:
                content[t["subtitle_index"]] = line

    out_subs = [srt.Subtitle(index=s.index, start=s.start, end=s.end,
                              content=content[s.index]) for s in rough_subs]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(out_subs))

    fallback_cue_idx = sorted(set(fallback_cue_idx))
    report = {
        "n_chunks": len(chunks),
        "n_chunk_fallback": len(chunk_fallbacks),
        "chunk_fallbacks": chunk_fallbacks,
        "n_cues": len(rough_subs),
        "n_cue_fallback": len(fallback_cue_idx),
        "cue_fallback_ranges": _ranges(fallback_cue_idx),
    }
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--debug_json", required=True)
    ap.add_argument("--rough_srt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--no_guard", action="store_true",
                     help="Tat toan bo guard (chi de kiem cheo voi SRT goc)")
    args = ap.parse_args()
    report = rebuild(args.debug_json, args.rough_srt, args.out, args.report, args.no_guard)
    print(f"chunk fallback: {report['n_chunk_fallback']}/{report['n_chunks']}, "
          f"cue fallback: {report['n_cue_fallback']}/{report['n_cues']}")


if __name__ == "__main__":
    main()
