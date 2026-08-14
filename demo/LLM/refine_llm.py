import os
import re
import json
import time
import argparse
import srt

# Reduce fragmentation-driven OOM on the shared GPU (must be set before CUDA init).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# Disable torch.compile (Dynamo) to prevent compilation overhead for dynamic sequence lengths.
torch._dynamo.config.disable = True

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Dynamic root folder calculation (corresponds to the demo folder)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from LLM.output_guard import is_degenerate_line

# Login to Hugging Face (token from env; needed only if the adapter repo is private)
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

from unsloth import FastLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="Refine Vietnamese subtitles using Gemma-12B + scene context (BATCH).")
    parser.add_argument("--en_srt",   type=str, required=True)
    parser.add_argument("--vinai_srt",type=str, required=True)
    parser.add_argument("--vlm_json", type=str, required=True)
    parser.add_argument("--output_srt",type=str, required=True)
    parser.add_argument("--adapter_model_name", type=str, default="thevan2404/best_gemma_scene_context")
    parser.add_argument("--cache_dir", type=str, default=os.path.join(ROOT_DIR, "cache"))
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--llm_batch_size", type=int, default=8,
                        help="Number of prompts to process in one GPU batch.")
    parser.add_argument("--registry_json", type=str, default=None,
                        help="Optional character-relationship registry JSON "
                             "(built by CHARACTER/build_registry.py).")
    return parser.parse_args()


def find_best_scene(midpoint, scenes):
    """Strict scene matching by midpoint timestamp (no nearest neighbor fallback)."""
    for idx, sc in enumerate(scenes):
        s, e = sc.get('start_time'), sc.get('end_time')
        if s is None or e is None:
            continue
        if s <= midpoint <= e:
            return idx
    return -1


def _pad_and_generate(model, tokenizer, ids_list, max_new_tokens):
    """Left-pad a list of 1-D input_ids, generate, return decoded new tokens."""
    max_len = max(t.size(0) for t in ids_list)
    padded_list = []
    attention_mask_list = []
    for t in ids_list:
        pad_len = max_len - t.size(0)
        if pad_len > 0:
            pad_tensor = torch.full((pad_len,), tokenizer.pad_token_id, dtype=t.dtype)
            padded_t = torch.cat([pad_tensor, t], dim=0)
            mask_t = torch.cat([torch.zeros(pad_len, dtype=torch.long),
                                torch.ones(t.size(0), dtype=torch.long)], dim=0)
        else:
            padded_t = t
            mask_t = torch.ones(t.size(0), dtype=torch.long)
        padded_list.append(padded_t)
        attention_mask_list.append(mask_t)

    padded_batch = torch.stack(padded_list).to("cuda")
    attention_mask_batch = torch.stack(attention_mask_list).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=padded_batch,
            attention_mask=attention_mask_batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    # Slice from max_len because of left padding.
    return [tokenizer.decode(out[max_len:], skip_special_tokens=True)
            for out in outputs]


def _generate_batch_safe(model, tokenizer, ids_list, max_new_tokens,
                         oom_wait_s=60, oom_max_waits=30):
    """The GPU is shared with other users, so VRAM can vanish at any moment.
    On CUDA OOM: free the cache, halve the batch; if a single prompt still
    OOMs, wait for other processes to release VRAM and retry."""
    try:
        return _pad_and_generate(model, tokenizer, ids_list, max_new_tokens)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()

    if len(ids_list) > 1:
        mid = (len(ids_list) + 1) // 2
        print(f"⚠️ CUDA OOM (batch={len(ids_list)}) → retry as "
              f"{mid}+{len(ids_list) - mid}", flush=True)
        return (_generate_batch_safe(model, tokenizer, ids_list[:mid],
                                     max_new_tokens, oom_wait_s, oom_max_waits)
                + _generate_batch_safe(model, tokenizer, ids_list[mid:],
                                       max_new_tokens, oom_wait_s, oom_max_waits))

    for n in range(oom_max_waits):
        print(f"⚠️ CUDA OOM (batch=1), waiting {oom_wait_s}s for shared GPU "
              f"({n + 1}/{oom_max_waits})", flush=True)
        time.sleep(oom_wait_s)
        torch.cuda.empty_cache()
        try:
            return _pad_and_generate(model, tokenizer, ids_list, max_new_tokens)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
    raise torch.OutOfMemoryError(
        "CUDA OOM: still no free VRAM after waiting for other GPU processes")


def refine_subtitles(
    en_srt_path,
    vinai_srt_path,
    vlm_json_path,
    output_srt_path,
    adapter_model_name="thevan2404/best_gemma_scene_context",
    cache_dir=None,
    max_seq_length=2048,
    max_new_tokens=1024,
    llm_batch_size=8,
    registry_json_path=None
):
    if cache_dir is None:
        cache_dir = os.path.join(ROOT_DIR, "cache")
    # ── Read inputs ──────────────────────────────────────────────────────────
    print("📖 Reading subtitle files...")
    with open(en_srt_path,    "r", encoding="utf-8") as f:
        en_subs   = list(srt.parse(f.read()))
    with open(vinai_srt_path, "r", encoding="utf-8") as f:
        vinai_subs = list(srt.parse(f.read()))
    assert len(en_subs) == len(vinai_subs), "EN and VinAI subtitle counts mismatch!"

    for sub in en_subs:
        sub.content = re.sub(r'\s+', ' ',
            " ".join(l.strip() for l in sub.content.splitlines() if l.strip())).strip()
    for sub in vinai_subs:
        sub.content = re.sub(r'\s+', ' ',
            " ".join(l.strip() for l in sub.content.splitlines() if l.strip())).strip()

    out_subs = [srt.Subtitle(index=s.index, start=s.start, end=s.end, content="")
                for s in en_subs]

    print(f"📂 Reading scene captions: {vlm_json_path}")
    with open(vlm_json_path, "r", encoding="utf-8") as f:
        vlm_scenes = json.load(f)

    # ── Optional character registry ──────────────────────────────────────────
    # Tiem VAO trong <Scene Context> theo dung style caption VLM ("3. Relationship:
    # ..."); block/section la ngoai scene context lam adapter suy bien.
    registry_context = ""
    if registry_json_path and os.path.exists(registry_json_path):
        import sys
        if ROOT_DIR not in sys.path:
            sys.path.insert(0, ROOT_DIR)  # ROOT_DIR = demo/, chua package CHARACTER
        from CHARACTER.registry_prompt import render_registry_context
        from CHARACTER.registry_schema import load_registry
        registry_context = render_registry_context(load_registry(registry_json_path))
        if registry_context:
            print(f"📇 Character registry loaded: {registry_json_path}", flush=True)

    scenes_data = []
    for idx, sc in enumerate(vlm_scenes):
        scenes_data.append({
            "start_time": sc.get("start_time"),
            "end_time":   sc.get("end_time"),
            "caption":    sc.get("caption", "").strip() or "None",
            "indices":    []
        })

    # ── Assign subtitles → scenes ────────────────────────────────────────────
    for i, sub in enumerate(en_subs):
        mid = (sub.start.total_seconds() + sub.end.total_seconds()) / 2.0
        si  = find_best_scene(mid, scenes_data)
        if si != -1:
            scenes_data[si]["indices"].append(i)

    # ── Build prompt list ────────────────────────────────────────────────────
    prompts = []   # list of {"indices", "context", "raw_en", "vinai_sub"}
    for sc in scenes_data:
        if not sc["indices"]:
            continue
        cap = sc["caption"]
        chunk = sc["indices"]
        prompts.append({
            "indices":   chunk,
            "context":   cap,
            "raw_en":    "\n".join(en_subs[j].content    for j in chunk),
            "vinai_sub": "\n".join(vinai_subs[j].content for j in chunk),
        })
    print(f"✓ {len(prompts)} scene-prompt chunks to process.")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model via Unsloth 4-bit: {adapter_model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        cache_dir=cache_dir,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_system = (
        "You are a professional Vietnamese subtitle editor for a movie.\n"
        "    Given three sections:\n"
        "        <Scene Context> — A description of the characters, their relationships "
        "(e.g., lovers, enemies, boss/employee), and the mood of the scene.\n"
        "        <English Dialogue> — original English lines.\n"
        "        <Rough Vietnamese Translation> — rough Vietnamese translation with possible "
        "tone or pronoun issues.\n"
        "    Use the English dialogue only to understand speaker context.\n"
        "    Fix the Vietnamese translation so that pronouns, tone, and formality are natural "
        "and consistent with the context.\n"
        "    Keep meaning and structure unchanged.\n"
        "    Output only the corrected Vietnamese translation, line by line.   "
    )
    # ── Pre-tokenize all prompts ─────────────────────────────────────────────
    print("Tokenizing all prompts...")
    all_input_ids = []
    for item in prompts:
        scene_ctx = item['context']
        if registry_context:
            scene_ctx = f"{scene_ctx}\n    {registry_context}"
        full_sys = (f"{base_system}\n"
                    f"    <Scene Context>\n    {scene_ctx}\n    </Scene Context>")
        user_msg = (f"<English Dialogue>\n{item['raw_en']}\n</English Dialogue>\n"
                    f"<Rough Vietnamese Translation>\n{item['vinai_sub']}\n"
                    f"</Rough Vietnamese Translation>")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": full_sys}]},
            {"role": "user",   "content": [{"type": "text", "text": user_msg}]},
        ]

        # Keep on CPU; moved to GPU per batch inside _pad_and_generate.
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            add_generation_prompt=True
        )
        all_input_ids.append(input_ids)

    assert len(all_input_ids) == len(prompts)
    print("Tokenization done. Starting BATCH inference...")

    # ── Batch inference ───────────────────────────────────────────────────────
    t_total = time.time()
    debug_scenes = []
    batch_size = llm_batch_size
    num_prompts = len(prompts)

    for idx in range(0, num_prompts, batch_size):
        batch_slice = slice(idx, idx + batch_size)
        batch_prompts = prompts[batch_slice]
        batch_input_ids = [t.squeeze(0) for t in all_input_ids[batch_slice]]

        t0 = time.time()
        decoded_batch = _generate_batch_safe(
            model, tokenizer, batch_input_ids, max_new_tokens
        )
        elapsed = time.time() - t0
        print(f"Batch [{idx//batch_size + 1}/{(num_prompts - 1)//batch_size + 1}] finished in {elapsed:.1f}s", flush=True)

        # Process each response in the batch
        for i, item in enumerate(batch_prompts):
            decoded = decoded_batch[i]

            lines_out = [l.strip() for l in decoded.split("\n") if l.strip()]
            n_src = len(item["indices"])

            # Guardrail: dong suy bien (lap n-gram) -> fallback dong rough.
            degenerate_ks = set()
            for k in range(min(len(lines_out), n_src)):
                if is_degenerate_line(lines_out[k]):
                    degenerate_ks.add(k)
                    s_idx = item["indices"][k]
                    print(f"⚠️ [guard] degenerate output at subtitle "
                          f"{en_subs[s_idx].index}: {lines_out[k][:60]!r}… "
                          f"-> rough fallback", flush=True)
                    lines_out[k] = vinai_subs[s_idx].content

            # Record translations before adjustment for fallback tracking
            scene_translations = []
            for k, s_idx in enumerate(item["indices"]):
                refined_candidate = lines_out[k] if k < len(lines_out) else ""
                fallback_used = False
                if k >= len(lines_out):
                    refined_val = vinai_subs[s_idx].content
                    fallback_used = True
                else:
                    refined_val = lines_out[k]
                    fallback_used = k in degenerate_ks

                scene_translations.append({
                    "subtitle_index": en_subs[s_idx].index,
                    "english": en_subs[s_idx].content,
                    "rough_vietnamese": vinai_subs[s_idx].content,
                    "refined_vietnamese": refined_val,
                    "fallback_used": fallback_used
                })

            if len(lines_out) < n_src:
                for k in range(len(lines_out), n_src):
                    lines_out.append(vinai_subs[item["indices"][k]].content)
            elif len(lines_out) > n_src:
                lines_out = lines_out[:n_src]

            for k, line in enumerate(lines_out):
                out_subs[item["indices"][k]].content = line

            debug_scenes.append({
                "scene_index": idx + i,
                "scene_caption": item["context"],
                "translations": scene_translations
            })

    total = time.time() - t_total
    print(f"\n✅ Done in {total/60:.1f} min")

    # ── Write output ──────────────────────────────────────────────────────────
    for i, sub in enumerate(out_subs):
        sub.start = en_subs[i].start
        sub.end   = en_subs[i].end
        if not sub.content.strip():
            sub.content = vinai_subs[i].content   # fallback

    os.makedirs(os.path.dirname(os.path.abspath(output_srt_path)), exist_ok=True)
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(out_subs))
    print(f"💾 Saved: {output_srt_path}")

    # Save debug information as JSON
    output_json = output_srt_path + ".json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(debug_scenes, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved debug JSON: {output_json}")


def main():
    args = parse_args()
    refine_subtitles(
        en_srt_path=args.en_srt,
        vinai_srt_path=args.vinai_srt,
        vlm_json_path=args.vlm_json,
        output_srt_path=args.output_srt,
        adapter_model_name=args.adapter_model_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        llm_batch_size=args.llm_batch_size,
        registry_json_path=args.registry_json,
    )


if __name__ == "__main__":
    main()
