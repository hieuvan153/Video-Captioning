import os
import sys
import re
import json
import time
import argparse
import srt
import torch

# Tat torch.compile: do dai chuoi thay doi lien tuc, bien dich chi ton thoi gian.
torch._dynamo.config.disable = True

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from huggingface_hub import login
if os.environ.get("HF_TOKEN"):  # adapter/model cuc bo khong can token
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

from unsloth import FastLanguageModel  # phai import truoc transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine_post import (  # noqa: E402
    align_lines, assign_scenes, cap_chunks, chunk_by_gap, drop_truncated_tail, is_degenerate,
    lock_len, strip_numbering, write_srt,
)
















def parse_args():
    parser = argparse.ArgumentParser(description="Refine Vietnamese subtitles using Gemma-12B + scene context (BATCH).")
    parser.add_argument("--en_srt",   type=str, required=True)
    parser.add_argument("--vinai_srt",type=str, required=True)
    parser.add_argument("--vlm_json", type=str, default=None,
                        help="Caption cua VLM. Bo trong (hoac 'none') thi chia chunk theo khoang lang.")
    parser.add_argument("--chunk_gap_s", type=float, default=2.0,
                        help="Cat chunk khi khoang lang giua hai cue vuot nguong nay (giay).")
    parser.add_argument("--chunk_target", type=int, default=20,
                        help="Gop cac manh lien tiep cho toi khi dat co nay (so cue).")
    parser.add_argument("--max_chunk_cues", type=int, default=30,
                        help="Tran so cue moi canh VLM (p95 du lieu train ~31; canh 91 cue vuot max_seq_length).")
    parser.add_argument("--output_srt",type=str, required=True)
    parser.add_argument("--adapter_model_name", type=str, default="thevan2404/best_gemma_scene_context")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Ghi de system prompt va BO khoi <Scene Context>. Dung cho adapter "
                             "train khong co kenh caption, vi du v7: "
                             "'Rewrite to natural Vietnamese subtitle.'")
    parser.add_argument("--cache_dir", type=str, default=os.path.join(ROOT_DIR, "cache"))
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--llm_batch_size", type=int, default=1,
                        help="So prompt moi batch. >1 lam model tra ve lech dong (31,8%% dong hong o batch=10).")
    return parser.parse_args()






def refine_subtitles(
    en_srt_path,
    vinai_srt_path,
    vlm_json_path,
    output_srt_path,
    adapter_model_name="thevan2404/best_gemma_scene_context",
    cache_dir=None,
    max_seq_length=2048,
    max_new_tokens=1024,
    llm_batch_size=1,
    system_prompt=None,
    chunk_gap_s=2.0,
    chunk_target=20,
    max_chunk_cues=30
):
    if cache_dir is None:
        cache_dir = os.path.join(ROOT_DIR, "cache")
    print("Reading subtitle files...")
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

    if not vlm_json_path or vlm_json_path.lower() == "none":
        print(f"Khong co VLM caption -> chia chunk theo khoang lang "
              f"(gap>{chunk_gap_s}s, gop toi {chunk_target} cue).", flush=True)
        scenes_data = chunk_by_gap(en_subs, chunk_gap_s, chunk_target)
    else:
        print(f"Reading scene captions: {vlm_json_path}")
        with open(vlm_json_path, "r", encoding="utf-8") as f:
            vlm_scenes = json.load(f)

        scenes_data = []
        for idx, sc in enumerate(vlm_scenes):
            scenes_data.append({
                "start_time": sc.get("start_time"),
                "end_time":   sc.get("end_time"),
                "caption":    sc.get("caption", "").strip() or "None",
                "indices":    []
            })

        mids = [(s.start.total_seconds() + s.end.total_seconds()) / 2.0 for s in en_subs]
        scene_idx, n_outside = assign_scenes(mids, scenes_data)
        for i, si in enumerate(scene_idx):
            if si != -1:
                scenes_data[si]["indices"].append(i)
        if n_outside:
            print(f"{n_outside} cue nam ngoai moi canh VLM -> gan vao canh gan nhat", flush=True)

    prompts = []
    for sc in scenes_data:
        for chunk in cap_chunks([sc["indices"]], max_chunk_cues):
            prompts.append({
                "indices":   chunk,
                "context":   sc["caption"],
                "raw_en":    "\n".join(en_subs[j].content    for j in chunk),
                "vinai_sub": "\n".join(vinai_subs[j].content for j in chunk),
            })
    print(f"{len(prompts)} scene-prompt chunks to process.")

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

    print("Tokenizing all prompts...")
    all_input_ids = []
    for item in prompts:
        # --system_prompt (adapter v7+ train khong co kenh caption): dung nguyen chuoi, bo <Scene Context>.
        full_sys = system_prompt or (f"{base_system}\n"
                    f"    <Scene Context>\n    {item['context']}\n    </Scene Context>")
        user_msg = (f"<English Dialogue>\n{item['raw_en']}\n</English Dialogue>\n"
                    f"<Rough Vietnamese Translation>\n{item['vinai_sub']}\n"
                    f"</Rough Vietnamese Translation>")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": full_sys}]},
            {"role": "user",   "content": [{"type": "text", "text": user_msg}]},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda")
        all_input_ids.append(input_ids)

    print("Tokenization done. Starting BATCH inference...")

    t_total = time.time()
    n_mismatch = n_degenerate = 0
    debug_scenes = []
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos_ids = set(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos]) | {getattr(tokenizer, "eos_token_id", None)}
    eos_ids.discard(None)
    batch_size = llm_batch_size
    num_prompts = len(prompts)

    for idx in range(0, num_prompts, batch_size):
        batch_slice = slice(idx, idx + batch_size)
        batch_prompts = prompts[batch_slice]
        batch_input_ids = [t.squeeze(0) for t in all_input_ids[batch_slice]]
        
        # Dem trai de out[max_len:] la phan sinh moi cua moi dong.
        max_len = max(t.size(0) for t in batch_input_ids)
        padded_list = []
        attention_mask_list = []
        for t in batch_input_ids:
            pad_len = max_len - t.size(0)
            if pad_len > 0:
                pad_tensor = torch.full((pad_len,), tokenizer.pad_token_id, dtype=t.dtype, device=t.device)
                padded_t = torch.cat([pad_tensor, t], dim=0)
                mask_t = torch.cat([torch.zeros(pad_len, dtype=torch.long, device=t.device),
                                    torch.ones(t.size(0), dtype=torch.long, device=t.device)], dim=0)
            else:
                padded_t = t
                mask_t = torch.ones(t.size(0), dtype=torch.long, device=t.device)
            padded_list.append(padded_t)
            attention_mask_list.append(mask_t)

        padded_batch = torch.stack(padded_list).to("cuda")
        attention_mask_batch = torch.stack(attention_mask_list).to("cuda")

        t0 = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=padded_batch,
                attention_mask=attention_mask_batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        elapsed = time.time() - t0
        print(f"Batch [{idx//batch_size + 1}/{(num_prompts - 1)//batch_size + 1}] finished in {elapsed:.1f}s", flush=True)

        for i, item in enumerate(batch_prompts):
            new_tokens = outputs[i][max_len:]
            toks = new_tokens.tolist()
            hit_limit = len(toks) >= max_new_tokens and not any(t in eos_ids for t in toks)
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            # Gemma sinh U+2026, ban tho va phu de nguoi viet "..." (khong doi: -1,59 BLEU).
            decoded = decoded.replace("\u2026", "...")
            raw_lines = [l.strip() for l in decoded.split("\n") if l.strip()]
            lines_out = strip_numbering(drop_truncated_tail(raw_lines, hit_limit))
            rough = [vinai_subs[s_idx].content for s_idx in item["indices"]]
            if is_degenerate(lines_out, rough):
                n_degenerate += 1
                print(f"  ! chunk {idx + i}: dau ra lap -> giu ban tho ca chunk", flush=True)
                lines_out = []

            # LUON giong bang QHD, khong anh xa theo vi tri: LLM vua tach vua gop dong thi so dong
            # van bang nhau ma cac cue o giua truot 1 buoc (09/09: 30 cue).
            aligned = align_lines(rough, lines_out)
            locked = lock_len(rough, aligned)
            n_src, n_kept = len(rough), sum(v is not None for v in aligned)
            if len(lines_out) != n_src or n_kept != n_src:
                n_mismatch += 1
                print(f"  ! chunk {idx + i}: LLM tra {len(lines_out)} dong / "
                      f"{n_src} cue -> giong lai QHD, giu {n_kept} dong tinh chinh",
                      flush=True)

            # SRT lay ban da khoa do dai; JSON debug giu dong LLM da giong nhung CHUA khoa.
            scene_translations = []
            for s_idx, r, val, lv in zip(item["indices"], rough, aligned, locked):
                scene_translations.append({
                    "subtitle_index": en_subs[s_idx].index,
                    "english": en_subs[s_idx].content,
                    "rough_vietnamese": r,
                    "refined_vietnamese": val if val else r,
                    "fallback_used": val is None
                })
                out_subs[s_idx].content = lv if lv else r

            debug_scenes.append({
                "scene_index": idx + i,
                "scene_caption": item["context"],
                "translations": scene_translations,
                "raw_lines": raw_lines,
            })

    total = time.time() - t_total
    print(f"\nDone in {total/60:.1f} min")
    if n_mismatch:
        print(f"{n_mismatch}/{len(prompts)} chunk sai so dong -> da giong lai bang QHD", flush=True)
    if n_degenerate:
        print(f"{n_degenerate}/{len(prompts)} chunk dau ra lap -> giu ban tho", flush=True)

    for i, sub in enumerate(out_subs):
        sub.start = en_subs[i].start
        sub.end   = en_subs[i].end
        if not sub.content.strip():
            sub.content = vinai_subs[i].content

    write_srt(out_subs, output_srt_path)
    print(f"Saved: {output_srt_path}")

    output_json = output_srt_path + ".json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(debug_scenes, f, ensure_ascii=False, indent=2)
    print(f"Saved debug JSON: {output_json}")


def main():
    args = parse_args()
    refine_subtitles(
        en_srt_path=args.en_srt,
        vinai_srt_path=args.vinai_srt,
        vlm_json_path=args.vlm_json,
        output_srt_path=args.output_srt,
        adapter_model_name=args.adapter_model_name,
        system_prompt=args.system_prompt,
        chunk_gap_s=args.chunk_gap_s,
        chunk_target=args.chunk_target,
        max_chunk_cues=args.max_chunk_cues,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        llm_batch_size=args.llm_batch_size
    )


if __name__ == "__main__":
    main()
