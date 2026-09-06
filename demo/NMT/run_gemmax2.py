"""Dich SRT EN -> VI bang GemmaX2-28-9B-v0.2 (Xiaomi, arXiv:2502.02481), giu NGUYEN timestamp cue.

Model doc tu cache HF MAC DINH (/home/ndloc_bk/.cache/huggingface -> /data/ndloc_bk/.cache/...),
20 G da co san. KHONG doi HF_HOME: doi se lam an ca GemmaX2 lan COMET-DA.

CLI: van_env/bin/python demo/NMT/run_gemmax2.py --input_srt en.srt --output_srt vi.srt [--batch 16]
"""
from __future__ import annotations

import argparse
import os
import re
import time

import srt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "xiaomi-research/GemmaX2-28-9B-v0.2"
# Prompt dung y model card v0.2 - sai format la chat luong sup.
PROMPT = "Translate this from English to Vietnamese:\nEnglish: {text}\nVietnamese:"


def load_model(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tok.padding_side = "left"          # decoder-only: phai pad ben trai
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to(device)
    model.eval()
    return tok, model


def clean(gen: str) -> str:
    """Lay dong dau tien co chu cua phan SINH RA (da cat prompt o muc token)."""
    for line in gen.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            return line
    return ""


@torch.inference_mode()
def translate(texts: list[str], tok, model, device: str, batch: int, max_new_tokens: int) -> list[str]:
    out: list[str] = [""] * len(texts)
    todo = [i for i, t in enumerate(texts) if t.strip()]
    t0 = time.time()
    for b in range(0, len(todo), batch):
        idx = todo[b:b + batch]
        prompts = [PROMPT.format(text=texts[i]) for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                             use_cache=True, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]          # cat sach prompt theo token
        for i, s in zip(idx, tok.batch_decode(new, skip_special_tokens=True)):
            out[i] = clean(s)
        done = b + len(idx)
        print(f"[gx2] {done}/{len(todo)} cue, {time.time() - t0:.0f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_srt", required=True)
    ap.add_argument("--output_srt", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    with open(a.input_srt, encoding="utf-8-sig") as f:
        subs = list(srt.parse(f.read()))
    texts = [re.sub(r"\s+", " ", s.content).strip() for s in subs]
    print(f"[gx2] {len(subs)} cue tu {a.input_srt}", flush=True)

    tok, model = load_model(a.model, a.device)
    vi = translate(texts, tok, model, a.device, a.batch, a.max_new_tokens)

    n_prompt_leak = sum(1 for v in vi if "Translate this from English" in v or v.startswith("English:"))
    for s, v in zip(subs, vi):
        s.content = v                                     # timestamp giu nguyen

    out_dir = os.path.dirname(os.path.abspath(a.output_srt))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(a.output_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
        f.flush()
        os.fsync(f.fileno())

    n_empty = sum(1 for v in vi if not v)
    print(f"XONG: {len(subs)} cue | rong {n_empty} | lot prompt {n_prompt_leak} -> {a.output_srt}",
          flush=True)
    if n_prompt_leak:
        raise SystemExit(f"LOI: {n_prompt_leak} cue con nguyen van prompt")


if __name__ == "__main__":
    main()
