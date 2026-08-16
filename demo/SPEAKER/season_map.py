"""Dat ten cluster CAM++ o cap MUA thay vi cap tap (V2c).

SPK_id dung chung xuyen tap trong mot mua (do 2026-08-16: Superstore Mua 1
97 cluster / 11 tap, Young Sheldon Mua 1 113 cluster / 22 tap, moi cap tap
ke nhau giao 7-15 cluster). Dat ten per-tap co hai cai gia:

- Cung mot giong bi doan MOI TAP MOT TEN: movie_045 dang co SPK_021/028/038
  cung la "Meemaw" nhung SPK_066 lai thanh "Connie" — mot nguoi hai ten,
  turn_pairs va registry matching vo tac dung.
- Tap ngheo bang chung doan sai: cluster 60 dong giang vat ly bi gan "Missy"
  (movie_045 — phim duy nhat am sau khi co tag, docs/eval/speaker_ab_v1.md).

Gom bang chung CA MUA roi dat ten MOT lan:
- representative_lines / script_label_anchors / mention_exclusions chay tren
  chunk gop ca mua — cac ham nay per-chunk, khong phu thuoc thu tu, va anchor
  xung dot giua cac tap tu dong bi bo (dung tinh than "khong chac thi thoi").
- vocative_hints chay TUNG TAP roi cong phieu: "ke nhau" xuyen ranh tap la gia.
- Tap dong ten: union registry cua cac tap CO registry (bo eval); tap khong co
  registry van dong gop bang chung nhung khong them ten.
- Khong dung caption: caption ta canh cua mot tap, khong dai dien cho ca mua.

Dau ra: season map {SPK_id: ten} + speakers.season.json cho tung tap eval
(rebuild tu line_tags co san trong speakers.json, khong align lai).
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from SPEAKER.align import real_tag, representative_lines
from SPEAKER.cluster_map import map_clusters, tags_to_names
from SPEAKER.name_evidence import (
    apply_evidence,
    mention_exclusions,
    script_label_anchors,
    vocative_hints,
)


def load_season_chunks(season_dir: str) -> dict[str, list[dict]]:
    """movie_id -> chunks, tu <season_dir>/episodes/movie_XXX/movie_XXX.tagged.json."""
    episodes: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(
            os.path.join(season_dir, "episodes", "movie_*", "*.tagged.json"))):
        m = os.path.basename(os.path.dirname(f))
        with open(f, "r", encoding="utf-8") as fh:
            episodes[m] = json.load(fh)
    return episodes


def pooled_evidence(
    episodes: dict[str, list[dict]],
    valid_names: list[str],
    per_cluster: int = 8,
):
    """Gop bang chung ca mua. Tra (reps, counts, anchors, exclusions, hints);
    counts dem CHUNK VAD toan mua (per-episode dem dong SRT — cung la dong
    thoai, khac nguon)."""
    all_chunks = [c for m in sorted(episodes) for c in episodes[m]]
    reps = representative_lines(all_chunks, per_cluster)
    anchors = script_label_anchors(all_chunks, valid_names)
    exclusions = mention_exclusions(all_chunks, valid_names)
    hints: dict[str, dict[str, int]] = {}
    for m in sorted(episodes):
        for tag, votes in vocative_hints(episodes[m], valid_names).items():
            slot = hints.setdefault(tag, {})
            for name, k in votes.items():
                slot[name] = slot.get(name, 0) + k
    counts: dict[str, int] = {}
    for c in all_chunks:
        t = real_tag(c)
        if t:
            counts[t] = counts.get(t, 0) + 1
    return reps, counts, anchors, exclusions, hints


def map_season(
    episodes: dict[str, list[dict]],
    valid_names: list[str],
    generate_fn,
    count_tokens=None,
) -> dict[str, str]:
    """Mot lan dat ten cho moi cluster cua mua. Cung guardrail voi cap tap:
    LLM chi chon trong tap dong ten, anchor > exclusion > LLM."""
    reps, counts, anchors, exclusions, hints = pooled_evidence(
        episodes, valid_names)
    raw = map_clusters(reps, valid_names, generate_fn, "",
                       line_counts=counts, count_tokens=count_tokens,
                       hints=hints)
    mapping, n_fixed, n_dropped = apply_evidence(raw, anchors, exclusions)
    print(f"[season] {len(mapping)}/{len(reps)} cluster ra ten; "
          f"{len(anchors)} nhan kich ban, {n_fixed} sua, {n_dropped} bo",
          flush=True)
    return mapping


def rebuild_speakers(payload: dict, mapping: dict[str, str]) -> dict:
    """speakers.json cu (co line_tags) + season map -> payload moi cung schema."""
    line_tags = payload["line_tags"]
    names = tags_to_names(line_tags, mapping)
    used = {t for t in line_tags if t}
    out = {
        "n_lines": len(names),
        "speakers": names,
        "line_tags": line_tags,
        "clusters": {t: mapping[t] for t in sorted(used) if t in mapping},
        "n_named": sum(1 for n in names if n),
    }
    if payload.get("align_stats") is not None:
        out["align_stats"] = payload["align_stats"]
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Name CAM++ clusters once per season, then rebuild "
                    "speakers.season.json for eval episodes."
    )
    parser.add_argument("--season_dir", type=str, required=True,
                        help=".../series/<ten_phim>/<Mùa_N>")
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="demo/output/eval_ab — tap nao co mat o day thi "
                             "lay registry lam tap dong ten va duoc rebuild.")
    parser.add_argument("--output_map", type=str, default=None,
                        help="File season map. Mac dinh: "
                             "<eval_dir>/season_map.<series>.<mua>.json")
    parser.add_argument("--model_name", type=str,
                        default="unsloth/gemma-3-12b-it-unsloth-bnb-4bit")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    from SPEAKER.build_speakers import (
        _make_generate_fn,
        character_names_from_registry,
    )
    from SPEAKER.inject import save_speakers

    episodes = load_season_chunks(args.season_dir)
    if not episodes:
        raise SystemExit(f"{args.season_dir}: khong co tagged.json nao")
    eval_movies = [m for m in episodes
                   if os.path.isdir(os.path.join(args.eval_dir, m))]
    if not eval_movies:
        raise SystemExit(f"{args.eval_dir}: khong co tap nao thuoc mua nay")

    valid_names: list[str] = []
    seen: set[str] = set()
    for m in sorted(eval_movies):
        reg = os.path.join(args.eval_dir, m, "registry.json")
        if not os.path.exists(reg):
            continue
        for n in character_names_from_registry(reg):
            if n.casefold() not in seen:
                seen.add(n.casefold())
                valid_names.append(n)
    if not valid_names:
        raise SystemExit(
            f"{args.eval_dir}: khong tap eval nao cua mua nay co registry.json "
            f"— khong co tap dong ten, dung lai truoc khi ghi de "
            f"speakers.season.json thanh toan None."
        )
    print(f"[season] {len(episodes)} tap, {len(eval_movies)} tap eval, "
          f"{len(valid_names)} ten hop le", flush=True)

    generate_fn, count_tokens = _make_generate_fn(
        args.model_name, args.cache_dir or os.path.join(ROOT_DIR, "cache"),
        max_seq_length=8192, max_new_tokens=1024,
    )
    mapping = map_season(episodes, valid_names, generate_fn, count_tokens)

    season = os.path.basename(os.path.normpath(args.season_dir))
    series = os.path.basename(os.path.dirname(os.path.normpath(args.season_dir)))
    map_path = args.output_map or os.path.join(
        args.eval_dir, f"season_map.{series}.{season}.json")
    tmp = map_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(mapping.items())), f, ensure_ascii=False, indent=2)
    os.replace(tmp, map_path)
    print(f"[season] Saved: {map_path}", flush=True)

    for m in sorted(eval_movies):
        spk_path = os.path.join(args.eval_dir, m, "speakers.json")
        if not os.path.exists(spk_path):
            print(f"[season] {m}: chua co speakers.json, bo qua", flush=True)
            continue
        with open(spk_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        out = rebuild_speakers(payload, mapping)
        out_path = os.path.join(args.eval_dir, m, "speakers.season.json")
        save_speakers(out_path, out)
        changed = sum(1 for a, b in zip(payload["speakers"], out["speakers"])
                      if a != b)
        print(f"[season] {m}: {out['n_named']}/{out['n_lines']} dong co ten "
              f"(truoc {payload.get('n_named')}), {changed} dong doi",
              flush=True)


if __name__ == "__main__":
    main()
