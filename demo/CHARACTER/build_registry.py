"""Build directed character-relationship registry tu EN transcript + VLM captions.

GPU runner: load Gemma-3-12B base 4-bit (KHONG dung adapter refine), chay
extraction theo chunk, validate + merge, ghi JSON atomic.

LUU Y (do thuc nghiem 2026-08-13 tren A100 + unsloth 2026.5.2): generate cua
Gemma-3 qua unsloth suy giam dan khi prompt vuot ~1200 token (mat dau JSON,
tiep tuc "hoan thanh" schema example thay vi tra loi) — nghi do xu ly sliding
window attention 1024 cua Gemma-3 trong duong inference unsloth. Vi vay run()
do token that cua tung dong roi tu tinh chunk_size sao cho MOI prompt
<= _PROMPT_TOKEN_BUDGET (1000) token; khong dung chunk_size=200 mac dinh.

CLI:
    python build_registry.py --en_srt movie.srt --vlm_json captions.json \
        --output_json movie.registry.json
"""
import argparse
import json
import os
import re
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from CHARACTER.registry_prompt import (
    build_extraction_prompt,
    captions_summary,
    chunk_lines,
    number_lines,
    parse_llm_json,
)
from CHARACTER.registry_schema import (
    Registry,
    filter_registry_by_source,
    merge_registries,
    parse_registry,
    registry_to_json,
)


def extract_registry(
    en_lines: list[str],
    captions_text: str,
    generate_fn,
    chunk_size: int = 200,
    overlap: int = 30,
) -> Registry:
    """Orchestrator thuan: generate_fn(system, user) -> str duoc inject de test."""
    partials: list[Registry] = []
    chunks = chunk_lines(en_lines, chunk_size, overlap)
    for idx, (start, lines) in enumerate(chunks):
        system, user = build_extraction_prompt(
            number_lines(lines, start=start + 1), captions_text
        )
        decoded = generate_fn(system, user)
        raw = parse_llm_json(decoded)
        if raw is None:
            print(f"[registry] chunk {idx + 1}/{len(chunks)}: "
                  f"unparseable JSON, skipped", flush=True)
            continue
        partials.append(parse_registry(raw, n_lines=len(en_lines)))
        print(f"[registry] chunk {idx + 1}/{len(chunks)}: ok", flush=True)
    merged = merge_registries(partials)
    # Chong few-shot leak: bo nhan vat khong duoc nhac den trong input that.
    filtered = filter_registry_by_source(
        merged, "\n".join(en_lines) + "\n" + captions_text
    )
    if len(filtered.characters) < len(merged.characters):
        dropped = len(merged.characters) - len(filtered.characters)
        print(f"[registry] dropped {dropped} character(s) absent from "
              f"transcript/captions", flush=True)
    return filtered


def _load_en_lines(en_srt_path: str) -> list[str]:
    import srt
    with open(en_srt_path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))
    return [
        re.sub(r"\s+", " ",
               " ".join(l.strip() for l in s.content.splitlines())).strip()
        for s in subs
    ]


# Tran token cho MOI extraction prompt (system + captions + schema + dialogue).
# Vuot ~1200 la generate hong dan (xem docstring); 1000 de chua du phong.
_PROMPT_TOKEN_BUDGET = 1000


def _make_generate_fn(model_name: str, cache_dir: str,
                      max_seq_length: int, max_new_tokens: int):
    import torch
    from unsloth import FastLanguageModel

    print(f"[registry] Loading base model 4-bit: {model_name}", flush=True)
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

    return generate_fn, tokenizer


def _count_prompt_tokens(tokenizer, system: str, user: str) -> int:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt",
        add_generation_prompt=True,
    )
    return ids.shape[1]


def _fit_chunk_size(line_token_lens: list[int], budget: int,
                    max_chunk: int = 200, min_chunk: int = 8) -> int:
    """chunk_size lon nhat sao cho MOI cua so chunk_size dong lien tiep
    co tong token <= budget."""
    n = len(line_token_lens)
    for cs in range(min(max_chunk, n), min_chunk, -1):
        s = sum(line_token_lens[:cs])
        ok = s <= budget
        if ok:
            for i in range(cs, n):
                s += line_token_lens[i] - line_token_lens[i - cs]
                if s > budget:
                    ok = False
                    break
        if ok:
            return cs
    return min_chunk


def _plan_chunking(tokenizer, en_lines: list[str],
                   captions_text: str) -> tuple[int, int]:
    """Tinh (chunk_size, overlap) de moi prompt nam trong _PROMPT_TOKEN_BUDGET."""
    raw_tok = getattr(tokenizer, "tokenizer", tokenizer)
    system, user_empty = build_extraction_prompt("", captions_text)
    overhead = _count_prompt_tokens(tokenizer, system, user_empty)
    budget = max(_PROMPT_TOKEN_BUDGET - overhead, 80)
    line_lens = [
        len(raw_tok(f"{i + 1}. {line}", add_special_tokens=False)["input_ids"]) + 1
        for i, line in enumerate(en_lines)
    ]
    chunk_size = _fit_chunk_size(line_lens, budget)
    overlap = min(max(4, chunk_size // 6), chunk_size - 1)
    print(f"[registry] prompt overhead={overhead} tok, "
          f"chunk_size={chunk_size}, overlap={overlap}", flush=True)
    return chunk_size, overlap


def run(
    en_srt_path: str,
    vlm_json_path: str,
    output_json_path: str,
    model_name: str = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    cache_dir: str | None = None,
    max_seq_length: int = 8192,
    max_new_tokens: int = 2048,
) -> str:
    if cache_dir is None:
        cache_dir = os.path.join(ROOT_DIR, "cache")
    en_lines = _load_en_lines(en_srt_path)
    with open(vlm_json_path, "r", encoding="utf-8") as f:
        vlm_scenes = json.load(f)
    # Cap caption ngan hon mac dinh: captions an vao budget cua dialogue.
    captions_text = captions_summary(vlm_scenes, max_chars=1200)

    generate_fn, tokenizer = _make_generate_fn(
        model_name, cache_dir, max_seq_length, max_new_tokens
    )
    chunk_size, overlap = _plan_chunking(tokenizer, en_lines, captions_text)
    registry = extract_registry(
        en_lines, captions_text, generate_fn, chunk_size, overlap
    )
    print(f"[registry] {len(registry.characters)} characters, "
          f"{len(registry.relations)} directed relations", flush=True)

    tmp = output_json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry_to_json(registry), f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_json_path)
    print(f"[registry] Saved: {output_json_path}", flush=True)
    return output_json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build directed character-relationship registry."
    )
    parser.add_argument("--en_srt", type=str, required=True)
    parser.add_argument("--vlm_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default="unsloth/gemma-3-12b-it-unsloth-bnb-4bit")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()
    run(args.en_srt, args.vlm_json, args.output_json, args.model_name,
        args.cache_dir, args.max_seq_length, args.max_new_tokens)


if __name__ == "__main__":
    main()
