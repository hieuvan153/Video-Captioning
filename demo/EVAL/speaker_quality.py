"""Cham chat luong gan speaker per-line bang gold gender/age — KHONG can GPU.

Ground truth (data/en-vi-speaker-with-time-pronouns/) khong co TEN nhan vat
vang, nhung co `gender` (~72% dong) va `age` (float uoc luong) cua nguoi noi
that, khop 1:1 theo vi tri voi SRT eval. So (gender, age_range) cua ten duoc
gan (tra tu registry) voi (gender, age-bucket) vang la proxy du manh de bat
loi dat ten cluster: cross-tab thu cong da bat dung "Missy" o movie_045
(41 nam / 12 nu — cluster Sheldon giang bai bi gan ten em gai).

RANH GIOI CUNG: gold o day CHI dung de do va tune offline. Khong mot byte
gold nao duoc lam input pipeline (leakage) — vi vay module nam trong EVAL/,
va SPEAKER/ / LLM/ khong bao gio import no.

Proxy mu voi loi sai-ten-CUNG-gioi-cung-tuoi, va gender/age_range cua registry
co the sai ("Shelly" bi gan female). No la tin hieu tuning; gate van la
Pronoun F1 (EVAL/compare_arms.py).

CLI (mode cham):
    van_env/bin/python demo/EVAL/speaker_quality.py \
        --eval_dir demo/output/eval_ab --speakers speakers.json --per_name

CLI (mode sweep, re-align CPU voi mapping da luu, khong goi LLM):
    ... --sweep_min_score 0 0.5 0.55 0.6 0.65 --label_override
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from CHARACTER.registry_prompt import _alias_index
from CHARACTER.registry_schema import Registry, load_registry
from EVAL.run_arms import TAGGED_GLOB
from SPEAKER.align import assign_speakers, spans_from_subs
from SPEAKER.cluster_map import tags_to_names

_SCORED_GENDERS = frozenset({"male", "female"})
_SCORED_AGES = frozenset({"child", "teen", "adult", "elderly"})


def age_bucket(age) -> str | None:
    """Doi tuoi float cua gold ve vocab age_range cua registry.

    Bien chon theo bo eval: Sheldon 11.0 -> child, Meemaw 63.8 -> elderly,
    gold max 74.2. Khong co tuoi -> None (khong cham, khong phai mismatch).
    """
    if age is None:
        return None
    try:
        a = float(age)
    except (TypeError, ValueError):
        return None
    if a < 13:
        return "child"
    if a < 20:
        return "teen"
    if a < 60:
        return "adult"
    return "elderly"


def registry_profiles(reg: Registry) -> dict[str, tuple[str, str]]:
    """alias casefold -> (gender, age_range). Alias mo ho (2 nhan vat) da bi
    _alias_index loai — khong phan giai duoc thi khong cham."""
    info = {c.id: (c.gender, c.age_range) for c in reg.characters}
    return {alias: info[cid] for alias, cid in _alias_index(reg).items()}


def score_names(
    names: list[str | None], gold: list[dict],
    profiles: dict[str, tuple[str, str]],
) -> dict:
    """Cham mot cot ten per-line so voi gold gender/age.

    Mot dong chi duoc cham gender khi CA gold lan registry deu biet gioi tinh;
    thieu mot phia -> "unscored", khong bao gio tinh la mismatch. Age tuong tu.
    """
    if len(names) != len(gold):
        raise ValueError(
            f"speaker/gold mismatch: {len(names)} ten vs {len(gold)} dong gold"
        )
    n_named = 0
    n_gender = n_gender_ok = 0
    n_age = n_age_ok = 0
    per_name: dict[str, dict] = {}
    for name, rec in zip(names, gold):
        if not name:
            continue
        n_named += 1
        entry = per_name.setdefault(
            name, {"lines": 0, "male": 0, "female": 0, "gender_unknown": 0,
                   "age_buckets": {}},
        )
        entry["lines"] += 1
        g_gold = rec.get("gender")
        if g_gold in _SCORED_GENDERS:
            entry[g_gold] += 1
        else:
            entry["gender_unknown"] += 1
        bucket = age_bucket(rec.get("age"))
        if bucket:
            entry["age_buckets"][bucket] = entry["age_buckets"].get(bucket, 0) + 1

        prof = profiles.get(name.casefold())
        if prof is None:
            continue
        g_reg, a_reg = prof
        if g_gold in _SCORED_GENDERS and g_reg in _SCORED_GENDERS:
            n_gender += 1
            if g_gold == g_reg:
                n_gender_ok += 1
        if bucket and a_reg in _SCORED_AGES:
            n_age += 1
            if bucket == a_reg:
                n_age_ok += 1
    return {
        "n_lines": len(names),
        "n_named": n_named,
        "coverage": round(n_named / len(names), 4) if names else 0.0,
        "n_gender_scored": n_gender,
        "gender_consistency": round(n_gender_ok / n_gender, 4) if n_gender else None,
        "n_age_scored": n_age,
        "age_consistency": round(n_age_ok / n_age, 4) if n_age else None,
        "per_name": per_name,
    }


def tag_retention(
    names: list[str | None], ref_names: list[str | None]
) -> dict:
    """Do MAT tag so voi cot ten tham chieu (arm dang thang).

    Bai hoc V3 (docs/eval/speaker_ab_v3.md): proxy chi do do thuan cua tag
    CON LAI thi mu voi viec xoa tag dung tren cluster cung gioi — chinh che
    do loi lam 008/046 thua significant. Luat chon vong sau phai phat
    mat-tag-theo-dong, khong chi gender consistency.
    """
    if len(names) != len(ref_names):
        raise ValueError(
            f"ref mismatch: {len(names)} ten vs {len(ref_names)} ref")
    n_ref = sum(1 for r in ref_names if r)
    n_lost = sum(1 for r, n in zip(ref_names, names) if r and not n)
    n_gained = sum(1 for r, n in zip(ref_names, names) if n and not r)
    n_changed = sum(1 for r, n in zip(ref_names, names) if r and n and r != n)
    return {"n_ref_named": n_ref, "n_lost": n_lost, "n_gained": n_gained,
            "n_changed": n_changed,
            "retention": round(1 - n_lost / n_ref, 4) if n_ref else None}


def variant_names(
    chunks: list[dict], spans, stored_mapping: dict[str, str],
    min_score: float | None = None, label_override: bool = False,
    valid_names: list[str] | None = None,
) -> tuple[list[str | None], object]:
    """Re-align CPU-only cho sweep: tai dung mapping cluster->ten DA LUU
    (truong "clusters" cua speakers.json), khong goi LLM.

    Xap xi co chu dich: rebuild that se re-prompt LLM voi anchor/cluster moi
    va ten co the doi — vi vay diem sweep chi de CHON operating point, con
    file speakers.v3.json that phai duoc cham lai truoc khi tin.
    """
    use = chunks
    if label_override:
        from SPEAKER.label_override import (
            apply_label_names, split_chunks_at_labels,
        )
        use = split_chunks_at_labels(chunks, valid_names or [])
    line_tags, stats = assign_speakers(use, spans, min_score=min_score)
    names = tags_to_names(line_tags, stored_mapping)
    if label_override:
        names = apply_label_names(line_tags, names)
    return names, stats


def drop_stale_anchor_names(
    mapping: dict[str, str], chunks: list[dict], valid_names: list[str],
) -> dict[str, str]:
    """Xap xi cho sweep: bo ten sinh ra tu anchor giua-chunk (ngu nghia cu).

    Mapping da luu duoc build khi nhan giua chunk con anchor ca cluster
    (SPK_106 -> "Missy"). Duoi ngu nghia moi (name_evidence.script_label_anchors
    chi nhan nhan dau text), nhung ten do khong con cho dua — bo ve khong ten.
    Chi la xap xi do luong; cau tra loi that la rebuild speakers.v3.json.
    """
    from SPEAKER.name_evidence import iter_script_labels, script_label_anchors

    canon = {n.casefold(): n for n in valid_names}
    old_votes: dict[str, set[str]] = {}
    for c in chunks:
        tag = c.get("speaker_tag")
        if not tag:
            continue
        for m in iter_script_labels(c.get("english") or ""):
            real = canon.get(m.group(1).strip().casefold())
            if real:
                old_votes.setdefault(str(tag), set()).add(real)
    old_anchors = {t: next(iter(v)) for t, v in old_votes.items() if len(v) == 1}
    new_anchors = script_label_anchors(chunks, valid_names)
    return {
        tag: name for tag, name in mapping.items()
        if not (old_anchors.get(tag) == name and new_anchors.get(tag) != name)
    }


def _load_gold(gt_dir: str, movie: str) -> list[dict]:
    with open(os.path.join(gt_dir, f"{movie}.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_spans(en_srt_path: str):
    import srt
    with open(en_srt_path, "r", encoding="utf-8") as f:
        return spans_from_subs(list(srt.parse(f.read())))


def _movie_dirs(eval_dir: str, movies) -> list[str]:
    if movies:
        return list(movies)
    return sorted(
        d for d in os.listdir(eval_dir)
        if os.path.isdir(os.path.join(eval_dir, d))
    )


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.4f}"


def _print_scores(movie: str, scores: dict, per_name: bool) -> None:
    print(f"{movie}: coverage {scores['coverage']:.1%} "
          f"({scores['n_named']}/{scores['n_lines']}), "
          f"gender {_fmt(scores['gender_consistency'])} "
          f"(n={scores['n_gender_scored']}), "
          f"age {_fmt(scores['age_consistency'])} "
          f"(n={scores['n_age_scored']})", flush=True)
    vr = scores.get("vs_ref")
    if vr and vr["retention"] is not None:
        print(f"    vs ref: giu {vr['retention']:.1%} "
              f"({vr['n_lost']} mat / {vr['n_gained']} them / "
              f"{vr['n_changed']} doi ten)", flush=True)
    if per_name:
        by_size = sorted(scores["per_name"].items(),
                         key=lambda kv: -kv[1]["lines"])
        for name, e in by_size:
            ages = ", ".join(f"{k}:{v}" for k, v in
                             sorted(e["age_buckets"].items(), key=lambda kv: -kv[1]))
            print(f"    {name}: {e['lines']} dong "
                  f"(M {e['male']} / F {e['female']} / ? {e['gender_unknown']}"
                  f"{'; ' + ages if ages else ''})", flush=True)


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score per-line speaker names against gold gender/age.")
    parser.add_argument("--eval_dir", type=str,
                        default=os.path.join(ROOT_DIR, "output", "eval_ab"))
    parser.add_argument("--gt_dir", type=str,
                        default=os.path.join(os.path.dirname(ROOT_DIR), "data",
                                             "en-vi-speaker-with-time-pronouns"))
    parser.add_argument("--speakers", type=str, default="speakers.json",
                        help="Ten file speakers trong tung thu muc phim.")
    parser.add_argument("--movies", type=str, nargs="*", default=None)
    parser.add_argument("--per_name", action="store_true")
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--sweep_min_score", type=float, nargs="*", default=None,
                        help="Mode sweep: re-align voi tung nguong (0 = khong "
                             "loc), tai dung mapping da luu, khong goi LLM.")
    parser.add_argument("--label_override", action="store_true",
                        help="Sweep: cat chunk tai nhan kich ban truoc khi "
                             "align (SPEAKER/label_override.py).")
    parser.add_argument("--fix_anchors", action="store_true",
                        help="Sweep: bo ten sinh tu anchor giua-chunk (xap xi "
                             "ngu nghia anchor moi tren mapping cu).")
    parser.add_argument("--ref_speakers", type=str, default=None,
                        help="Ten file speakers tham chieu (arm dang thang): "
                             "bao cao them so dong MAT ten so voi no.")
    parser.add_argument("--mid_anchor_votes", type=int, default=None,
                        help="Sweep: xap xi V3.1 — de anchor giua-chunk co "
                             ">= N phieu cung ten ghi de mapping da luu "
                             "(name_evidence.script_label_anchors).")
    args = parser.parse_args()

    eval_dir = os.path.abspath(args.eval_dir)
    repo_root = os.path.dirname(ROOT_DIR)
    movies = _movie_dirs(eval_dir, args.movies)
    report: dict = {"speakers_file": args.speakers, "movies": {}}

    if args.sweep_min_score is None:
        for movie in movies:
            mdir = os.path.join(eval_dir, movie)
            with open(os.path.join(mdir, args.speakers), encoding="utf-8") as f:
                data = json.load(f)
            profiles = registry_profiles(
                load_registry(os.path.join(mdir, "registry.json")))
            scores = score_names(data["speakers"], _load_gold(args.gt_dir, movie),
                                 profiles)
            if args.ref_speakers:
                from SPEAKER.inject import load_speakers
                scores["vs_ref"] = tag_retention(
                    [n if isinstance(n, str) and n.strip() else None
                     for n in data["speakers"]],
                    load_speakers(os.path.join(mdir, args.ref_speakers),
                                  len(data["speakers"])))
            _print_scores(movie, scores, args.per_name)
            report["movies"][movie] = scores
        report["mean_gender_consistency"] = _mean(
            [m["gender_consistency"] for m in report["movies"].values()])
        report["mean_coverage"] = _mean(
            [m["coverage"] for m in report["movies"].values()])
        print(f"mean: gender {_fmt(report['mean_gender_consistency'])}, "
              f"coverage {_fmt(report['mean_coverage'])}", flush=True)
    else:
        from SPEAKER.build_speakers import character_names_from_registry
        thresholds = [t if t > 0 else None for t in args.sweep_min_score]
        report["sweep"] = []
        loaded = {}
        for movie in movies:
            mdir = os.path.join(eval_dir, movie)
            hits = glob.glob(os.path.join(repo_root, TAGGED_GLOB.format(m=movie)))
            if not hits:
                print(f"Warning: {movie}: khong co tagged.json — bo qua", flush=True)
                continue
            with open(hits[0], encoding="utf-8") as f:
                chunks = json.load(f)
            with open(os.path.join(mdir, args.speakers), encoding="utf-8") as f:
                mapping = json.load(f)["clusters"]
            valid_names = character_names_from_registry(
                os.path.join(mdir, "registry.json"))
            if args.fix_anchors:
                mapping = drop_stale_anchor_names(mapping, chunks, valid_names)
            if args.mid_anchor_votes is not None:
                # Xap xi V3.1: anchor (start + mid >= N phieu) ghi de mapping
                # da luu — dung precedence that cua apply_evidence (anchor
                # phu quyet LLM lan exclusion). Khong mo phong duoc LLM
                # re-prompt; file that phai duoc cham lai truoc khi tin.
                from SPEAKER.name_evidence import script_label_anchors
                mapping = {**mapping, **script_label_anchors(
                    chunks, valid_names,
                    mid_anchor_votes=args.mid_anchor_votes)}
            spans = _load_spans(os.path.join(mdir, "en.srt"))
            ref_names = None
            if args.ref_speakers:
                from SPEAKER.inject import load_speakers
                ref_names = load_speakers(
                    os.path.join(mdir, args.ref_speakers), len(spans))
            loaded[movie] = (
                chunks, spans, mapping, valid_names,
                registry_profiles(load_registry(os.path.join(mdir, "registry.json"))),
                _load_gold(args.gt_dir, movie), ref_names,
            )
        for thr in thresholds:
            row = {"min_score": thr, "label_override": args.label_override,
                   "fix_anchors": args.fix_anchors,
                   "mid_anchor_votes": args.mid_anchor_votes, "movies": {}}
            for movie, (chunks, spans, mapping, valid_names, profiles,
                        gold, ref_names) in loaded.items():
                names, _ = variant_names(chunks, spans, mapping,
                                         min_score=thr,
                                         label_override=args.label_override,
                                         valid_names=valid_names)
                row["movies"][movie] = score_names(names, gold, profiles)
                if ref_names is not None:
                    row["movies"][movie]["vs_ref"] = tag_retention(names,
                                                                   ref_names)
            row["mean_gender_consistency"] = _mean(
                [m["gender_consistency"] for m in row["movies"].values()])
            row["mean_coverage"] = _mean(
                [m["coverage"] for m in row["movies"].values()])
            print(f"min_score={thr} label_override={args.label_override} "
                  f"fix_anchors={args.fix_anchors}: "
                  f"gender {_fmt(row['mean_gender_consistency'])}, "
                  f"coverage {_fmt(row['mean_coverage'])}", flush=True)
            for movie, sc in row["movies"].items():
                _print_scores(f"  {movie}", sc, args.per_name)
            report["sweep"].append(row)

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        # per_name chi de doc tren man hinh; report giu metric gon.
        slim = json.loads(json.dumps(report))
        for m in slim.get("movies", {}).values():
            m.pop("per_name", None)
        for row in slim.get("sweep", []):
            for m in row["movies"].values():
                m.pop("per_name", None)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(slim, f, ensure_ascii=False, indent=2)
        print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
