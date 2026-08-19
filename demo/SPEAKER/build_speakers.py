"""Build <base>.speakers.json: gan speaker CAM++ vao tung dong SRT + map ra ten.

Dau vao la ket qua CAM++ da chay san (`*.tagged.json` trong
`data/speaker_verify_campp_en_full/`), KHONG chay lai speaker verification.

GPU chi can cho buoc map cluster -> ten (Gemma-3-12B base, giong
CHARACTER/build_registry.py). Buoc gan theo overlap thoi gian la thuan CPU.

CLI:
    python build_speakers.py --tagged_json movie_045.tagged.json \
        --en_srt en.srt --registry_json movie.registry.json \
        --output_json movie.speakers.json
"""
import argparse
import json
import os
import sys

# Giam OOM do phan manh VRAM tren GPU dung chung (phai set truoc khi CUDA init).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from CHARACTER.registry_prompt import captions_summary
from CHARACTER.registry_schema import PLACEHOLDER_NAMES, load_registry
from SPEAKER.align import assign_speakers, representative_lines, spans_from_subs
from SPEAKER.cluster_map import map_clusters, tags_to_names
from SPEAKER.inject import save_speakers, speakers_to_json
from SPEAKER.label_override import (
    apply_label_names,
    is_label_tag,
    split_chunks_at_labels,
)
from SPEAKER.name_evidence import (
    apply_evidence,
    mention_exclusions,
    script_label_anchors,
    vocative_hints,
)
from SPEAKER.vocative_edges import mine_address_edges


def build_speakers(
    chunks: list[dict],
    spans: list[tuple[float, float]],
    character_names: list[str],
    generate_fn,
    captions_text: str = "",
    min_score: float | None = None,
    per_cluster: int = 8,
    count_tokens=None,
    registry=None,
    mid_anchor_votes: int | None = None,
):
    """Orchestrator thuan: generate_fn(system, user) -> str inject duoc de test.

    registry (Registry, optional): co no thi dao them canh xung ho tu vocative
    EN (SPEAKER/vocative_edges.py) — tra ve o phan tu thu 5.
    """
    # Cat chunk tai nhan kich ban TRUOC khi align: vung nhan mang tag tong hop
    # LABEL::Ten, canh tranh overlap voi vung tag goc trong assign_speakers.
    line_tags, stats = assign_speakers(
        split_chunks_at_labels(chunks, character_names), spans,
        min_score=min_score,
    )
    used = {t for t in line_tags if t and not is_label_tag(t)}
    # Chi hoi LLM ve cluster that su xuat hien tren dong SRT: cluster chi ton
    # tai trong chunk khong giong dong nao thi map cung vo dung, ma van ton
    # token va lam prompt dai them. Bang chung/reps van doc tu chunk GOC —
    # LLM khong can biet ve vung LABEL::.
    reps = {t: v for t, v in representative_lines(chunks, per_cluster).items()
            if t in used}
    counts: dict[str, int] = {}
    for t in line_tags:
        if t and not is_label_tag(t):
            counts[t] = counts.get(t, 0) + 1

    # Bang chung tu chinh loi thoai di truoc LLM va phu quyet no (xem
    # SPEAKER/name_evidence.py: LLM mot minh sai theo kieu du doan duoc).
    anchors = script_label_anchors(chunks, character_names,
                                   mid_anchor_votes=mid_anchor_votes)
    exclusions = mention_exclusions(chunks, character_names)
    hints = vocative_hints(chunks, character_names)

    raw = map_clusters(reps, character_names, generate_fn, captions_text,
                       line_counts=counts, count_tokens=count_tokens,
                       hints={t: v for t, v in hints.items() if t in reps})
    mapping, n_fixed, n_dropped = apply_evidence(raw, anchors, exclusions)
    names = apply_label_names(line_tags, tags_to_names(line_tags, mapping))
    address_edges = (
        mine_address_edges(chunks, mapping, registry)
        if registry is not None else []
    )
    print(f"[speaker] {stats.n_tagged}/{stats.n_lines} dong co cluster "
          f"({stats.coverage:.1%}), {stats.n_contested} dong tranh chap; "
          f"bang chung: {len(anchors)} nhan kich ban, {n_fixed} sua, "
          f"{n_dropped} bo; {len(mapping)}/{len(reps)} cluster ra ten; "
          f"{sum(1 for n in names if n)}/{len(names)} dong co ten; "
          f"{len(address_edges)} canh xung ho", flush=True)
    return names, line_tags, mapping, stats, address_edges


def character_names_from_registry(registry_json_path: str) -> list[str]:
    """Tap dong ten hop le cho cluster_map: moi alias that cua moi nhan vat.

    Bo alias dai tu ("You", "Speaker"): chung khong dinh danh duoc ai, ma neu
    lot vao tap dong thi LLM se chon chung nhu mot cai ten.
    """
    reg = load_registry(registry_json_path)
    names: list[str] = []
    seen: set[str] = set()
    for c in reg.characters:
        for n in c.names:
            if n.casefold() not in PLACEHOLDER_NAMES and n.casefold() not in seen:
                seen.add(n.casefold())
                names.append(n)
    return names


def _load_spans(en_srt_path: str):
    import srt
    with open(en_srt_path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))
    return spans_from_subs(subs)


def _make_generate_fn(model_name: str, cache_dir: str,
                      max_seq_length: int, max_new_tokens: int):
    import torch
    from unsloth import FastLanguageModel

    print(f"[speaker] Loading base model 4-bit: {model_name}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        cache_dir=cache_dir,
    )
    FastLanguageModel.for_inference(model)

    def generate_fn(system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            add_generation_prompt=True,
        ).to("cuda")
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True
        )

    def count_tokens(system: str, user: str) -> int:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            add_generation_prompt=True,
        ).shape[1]

    return generate_fn, count_tokens


def run(
    tagged_json_path: str,
    en_srt_path: str,
    registry_json_path: str,
    output_json_path: str,
    vlm_json_path: str | None = None,
    model_name: str = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    cache_dir: str | None = None,
    max_seq_length: int = 8192,
    max_new_tokens: int = 1024,
    min_score: float | None = None,
    mid_anchor_votes: int | None = None,
) -> str:
    if cache_dir is None:
        cache_dir = os.path.join(ROOT_DIR, "cache")
    with open(tagged_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    spans = _load_spans(en_srt_path)
    registry = load_registry(registry_json_path)
    character_names = character_names_from_registry(registry_json_path)
    if not character_names:
        raise ValueError(
            f"{registry_json_path}: khong co nhan vat nao — khong co tap dong "
            f"de map cluster, chay CHARACTER/build_registry.py truoc."
        )

    captions_text = ""
    if vlm_json_path and os.path.exists(vlm_json_path):
        with open(vlm_json_path, "r", encoding="utf-8") as f:
            # Cap ngan hon build_registry (1200): caption an vao cung budget
            # token voi cac dong thoai — von la bang chung manh hon nhieu de
            # nhan ra ai la ai.
            captions_text = captions_summary(json.load(f), max_chars=600)

    generate_fn, count_tokens = _make_generate_fn(
        model_name, cache_dir, max_seq_length, max_new_tokens
    )
    names, line_tags, mapping, stats, address_edges = build_speakers(
        chunks, spans, character_names, generate_fn, captions_text,
        min_score=min_score, count_tokens=count_tokens, registry=registry,
        mid_anchor_votes=mid_anchor_votes,
    )
    save_speakers(output_json_path,
                  speakers_to_json(names, line_tags, mapping, stats,
                                   address_edges=address_edges))
    print(f"[speaker] Saved: {output_json_path}", flush=True)
    return output_json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align CAM++ speaker tags to subtitle lines and name them."
    )
    parser.add_argument("--tagged_json", type=str, required=True)
    parser.add_argument("--en_srt", type=str, required=True)
    parser.add_argument("--registry_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--vlm_json", type=str, default=None)
    parser.add_argument("--model_name", type=str,
                        default="unsloth/gemma-3-12b-it-unsloth-bnb-4bit")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--min_score", type=float, default=None,
                        help="Bo qua chunk co speaker_score duoi nguong nay.")
    parser.add_argument("--mid_anchor_votes", type=int, default=None,
                        help="V3.1: nhan kich ban GIUA chunk van anchor ca "
                             "cluster neu cung ten xuat hien o >= N chunk "
                             "(mac dinh None = khong bao gio, nhu V3).")
    args = parser.parse_args()
    run(args.tagged_json, args.en_srt, args.registry_json, args.output_json,
        args.vlm_json, args.model_name, args.cache_dir, args.max_seq_length,
        args.max_new_tokens, args.min_score, args.mid_anchor_votes)


if __name__ == "__main__":
    main()
