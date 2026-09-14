import os
import re
import sys
import time
import argparse
import srt
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
import nltk

for _pkg in ("punkt", "punkt_tab"):  # nltk >= 3.9 can punkt_tab cho sent_tokenize
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MBART = os.path.join(ROOT_DIR, "model/NMT/mbart_model")
DEFAULT_MODEL = LOCAL_MBART if os.path.exists(LOCAL_MBART) else "vinai/vinai-translate-en2vi-v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Translate English SRT subtitles to rough Vietnamese using VinAI MBart model.")
    parser.add_argument("--input_srt", type=str, required=True, help="Path to the input English .srt file.")
    parser.add_argument("--output_srt", type=str, required=True, help="Path to save the output Vietnamese .srt file.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL, help="Path or HuggingFace ID of the MBart model.")
    parser.add_argument("--cache_dir", type=str, default="/data/ndloc_bk/ntVan/hf_cache", help="HuggingFace cache directory.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for model inference.")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Mac dinh 1 (greedy) = giao thuc cua moc 'rough'. Pipeline dung 5.")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                        help="Pipeline dung 4.0 (mBART sinh tieng Viet ngan hon nguoi dich). Mac dinh 1.0 = hanh vi cu.")
    parser.add_argument("--mbr", type=int, default=0,
                        help="MBR: sinh K ung vien, chon ung vien co chrF trung binh cao nhat so voi cac ung vien con lai. "
                             "0 = tat.")
    parser.add_argument("--mbr_top_p", type=float, default=0.9, help="top-p khi lay mau ung vien MBR.")
    parser.add_argument("--seed", type=int, default=0, help="Seed cho lay mau MBR.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda or cpu).")
    return parser.parse_args()


def pick_mbr(cands):
    """Chon ung vien co chrF trung binh cao nhat so voi CA tap ung vien (ke ca ban sao:
    ung vien model sinh lai nhieu lan duoc cong diem, dung tinh than MBR)."""
    import sacrebleu
    cands = [c.strip() for c in cands if c.strip()]
    if not cands:
        return ""
    uniq = list(dict.fromkeys(cands))
    if len(uniq) == 1:
        return uniq[0]
    return max(uniq, key=lambda c: sum(sacrebleu.sentence_chrf(c, [o]).score for o in cands))


def _selftest_mbr():
    a = ["Toi yeu ban", "Toi yeu ban", "Con meo ngoi tren tham"]
    assert pick_mbr(a) == "Toi yeu ban", "ban sao phai keo ket qua ve phia dong thuan"
    assert pick_mbr(["", "  "]) == ""
    assert pick_mbr(["x"]) == "x"
    print("selftest pick_mbr OK")


def translate_en2vi(en_subs, model_en2vi, tokenizer_en2vi, device, batch_size=64, num_beams=1,
                    length_penalty=1.0, mbr=0, mbr_top_p=0.9, seed=0):
    """Dich tung cau (sent_tokenize) cua moi cue roi noi lai; ghi de sub.content."""
    all_sentences = []
    mapping = []  # (start_idx, end_idx) cua cac cau thuoc moi cue
    for sub in en_subs:
        sentences = [s for s in sent_tokenize(sub.content.strip()) if s.strip()]
        start_idx = len(all_sentences)
        all_sentences.extend(sentences)
        mapping.append((start_idx, len(all_sentences)))

    all_translations = []
    for i in tqdm(range(0, len(all_sentences), batch_size), desc="dich"):
        batch = all_sentences[i:i + batch_size]
        inputs = tokenizer_en2vi(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        if mbr:
            torch.manual_seed(seed + i)
            # num_beams >= mbr: ung vien tu beam (giu loi cua length_penalty); nguoc lai lay mau top-p.
            # num_beams=1 phai dat tuong minh vi generation_config.json cua mBART co san num_beams=5.
            extra = (dict(num_beams=num_beams, length_penalty=length_penalty, early_stopping=True)
                     if num_beams >= mbr else
                     dict(do_sample=True, num_beams=1, top_p=mbr_top_p))
            with torch.inference_mode():
                output_ids = model_en2vi.generate(
                    **inputs,
                    decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
                    num_return_sequences=mbr, max_length=128, **extra,
                )
            decoded = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
            all_translations.extend(pick_mbr(decoded[j * mbr:(j + 1) * mbr])
                                    for j in range(len(batch)))
            continue

        with torch.inference_mode():
            output_ids = model_en2vi.generate(
                **inputs,
                decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
                num_beams=num_beams,
                length_penalty=length_penalty,
                max_length=128,
                # DUNG GO: early_stopping=True giu length_penalty o buoc xep hang gia thuyet.
                # Bo di thi beam chay toi max_length va lap chu (08/09: BLEU 19,15 -> 0,99).
                early_stopping=True
            )
        all_translations.extend(tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True))

    for sub, (start_idx, end_idx) in zip(en_subs, mapping):
        sub.content = " ".join(all_translations[start_idx:end_idx]).strip()
    return en_subs


def main():
    args = parse_args()

    print(f"Doc phu de tieng Anh: {args.input_srt}")
    if not os.path.exists(args.input_srt):
        sys.exit(f"Loi: file khong ton tai -> {args.input_srt}")

    with open(args.input_srt, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))
    for sub in subtitles:
        sub.content = " ".join(line.strip() for line in sub.content.splitlines() if line.strip())
        sub.content = re.sub(r'\s+', ' ', sub.content).strip()
    print(f"Da load {len(subtitles)} phu de. Tai mo hinh {args.model_path} len {args.device}")

    tokenizer_en2vi = AutoTokenizer.from_pretrained(args.model_path, src_lang="en_XX", cache_dir=args.cache_dir)
    model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    model_en2vi.to(args.device)
    if "cuda" in args.device:
        model_en2vi.half()

    print(f"Dich (num_beams={args.num_beams}, length_penalty={args.length_penalty})")
    start_time = time.time()
    translated_subs = translate_en2vi(
        subtitles,
        model_en2vi,
        tokenizer_en2vi,
        args.device,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        mbr=args.mbr,
        mbr_top_p=args.mbr_top_p,
        seed=args.seed,
    )
    print(f"Dich xong trong {time.time() - start_time:.2f} giay.")

    os.makedirs(os.path.dirname(args.output_srt) or ".", exist_ok=True)
    with open(args.output_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(translated_subs))
    print(f"Da luu: {args.output_srt}")


if __name__ == "__main__":
    main()
