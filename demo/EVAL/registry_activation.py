"""Dem so scene render duoc registry context — khong can GPU.

Tai hien offline con so "Per-scene registry rendered N/M scenes" cua
LLM/refine_llm.py (cung logic gan scene qua LLM/scene_assign.py, cung ham
render), de quyet dinh CO dang dot GPU cho mot arm khong truoc khi chay:
go/no-go cua V3 Stage C (docs/superpowers/plans/2026-08-17-line-level-tags-v3.md).

CLI:
    van_env/bin/python demo/EVAL/registry_activation.py \
        --eval_dir demo/output/eval_ab --speakers speakers.v3.json \
        --mode pair_rel
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from CHARACTER.registry_prompt import render_scene_registry_context
from CHARACTER.registry_schema import load_registry
from LLM.scene_assign import scene_line_indices
from SPEAKER.inject import load_address_edges, load_speakers, turn_pairs

MODES = ("pair", "pair_addr", "pair_rel")


def count_activation(
    registry, vlm_scenes, en_subs, speaker_names, mode: str,
    address_edges=(),
) -> tuple[int, int, list[str]]:
    """(so scene render duoc, so scene co dong thoai, cac context da render)."""
    n_scenes = n_fired = 0
    contexts: list[str] = []
    for indices in scene_line_indices(vlm_scenes, en_subs):
        if not indices:
            continue
        n_scenes += 1
        line_names = [speaker_names[j] for j in indices]
        pairs = turn_pairs(line_names)
        if mode == "pair_addr":
            ctx = render_scene_registry_context(
                registry, pairs, include_address_edges=True)
        elif mode == "pair_rel":
            ctx = render_scene_registry_context(
                registry, pairs, extra_edges=address_edges)
        else:
            ctx = render_scene_registry_context(registry, pairs)
        if ctx:
            n_fired += 1
            contexts.append(ctx)
    return n_fired, n_scenes, contexts


def main() -> None:
    import srt

    parser = argparse.ArgumentParser(
        description="Count scenes where the registry context fires (offline).")
    parser.add_argument("--eval_dir", type=str,
                        default=os.path.join(ROOT_DIR, "output", "eval_ab"))
    parser.add_argument("--speakers", type=str, default="speakers.json")
    parser.add_argument("--registry", type=str, default="registry.json",
                        help="Ten file registry trong tung thu muc phim "
                             "(vd registry.fixed.json sau fix mega-node).")
    parser.add_argument("--mode", type=str, default="pair", choices=MODES)
    parser.add_argument("--movies", type=str, nargs="*", default=None)
    parser.add_argument("--show_contexts", action="store_true")
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    eval_dir = os.path.abspath(args.eval_dir)
    movies = args.movies or sorted(
        d for d in os.listdir(eval_dir)
        if os.path.isdir(os.path.join(eval_dir, d))
    )
    total_fired = total_scenes = n_films_fired = 0
    report = {"mode": args.mode, "speakers_file": args.speakers,
              "registry_file": args.registry, "movies": {}}
    for movie in movies:
        mdir = os.path.join(eval_dir, movie)
        spk_path = os.path.join(mdir, args.speakers)
        if not os.path.exists(spk_path):
            print(f"{movie}: thieu {args.speakers} — bo qua", flush=True)
            continue
        with open(os.path.join(mdir, "en.srt"), encoding="utf-8") as f:
            en_subs = list(srt.parse(f.read()))
        with open(os.path.join(mdir, "captions.json"), encoding="utf-8") as f:
            vlm_scenes = json.load(f)
        registry = load_registry(os.path.join(mdir, args.registry))
        speaker_names = load_speakers(spk_path, len(en_subs))
        edges = load_address_edges(spk_path) if args.mode == "pair_rel" else ()
        fired, scenes, contexts = count_activation(
            registry, vlm_scenes, en_subs, speaker_names, args.mode, edges)
        print(f"{movie}: {fired}/{scenes} scene kich hoat"
              + (f" ({len(edges)} canh vocative)" if args.mode == "pair_rel"
                 else ""), flush=True)
        if args.show_contexts:
            for ctx in contexts:
                print(f"    {ctx}", flush=True)
        total_fired += fired
        total_scenes += scenes
        n_films_fired += 1 if fired else 0
        report["movies"][movie] = {"fired": fired, "scenes": scenes}
    print(f"tong: {total_fired}/{total_scenes} scene, "
          f"{n_films_fired}/{len(report['movies'])} phim co kich hoat", flush=True)
    report["total_fired"] = total_fired
    report["total_scenes"] = total_scenes
    report["n_films_fired"] = n_films_fired
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
