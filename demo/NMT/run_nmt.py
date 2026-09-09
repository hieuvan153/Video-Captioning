import os
import re
import time
import argparse
import srt
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
import nltk

# Đảm bảo đã tải gói tokenizer 'punkt' cho NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Dynamic root folder calculation (corresponds to the demo folder)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(description="Translate English SRT subtitles to rough Vietnamese using VinAI MBart model.")
    parser.add_argument(
        "--input_srt",
        type=str,
        required=True,
        help="Path to the input English .srt file."
    )
    parser.add_argument(
        "--output_srt",
        type=str,
        required=True,
        help="Path to save the output translated Vietnamese .srt file."
    )
    # Xác định mặc định của model path
    local_mbart = os.path.join(ROOT_DIR, "model/NMT/mbart_model")
    if os.path.exists(local_mbart):
        default_model = local_mbart
    elif os.path.exists("/data/ndloc_bk/ntVan/infer/model/mbart_model"):
        default_model = "/data/ndloc_bk/ntVan/infer/model/mbart_model"
    elif os.path.exists("/data/ndloc_bk/app/model/mbart_model"):
        default_model = "/data/ndloc_bk/app/model/mbart_model"
    else:
        default_model = "vinai/vinai-translate-en2vi-v2"
        
    parser.add_argument(
        "--model_path",
        type=str,
        default=default_model,
        help="Path or HuggingFace ID of the VinAI MBart model."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data/ndloc_bk/ntVan/hf_cache",
        help="HuggingFace cache directory."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for model inference."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="So beam khi giai ma. Mac dinh 1 (greedy) = dung giao thuc cua moc 'rough', dung doi."
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="mBART sinh tieng Viet chi gian 1.097 lan so voi tieng Anh, trong khi nguoi gian 1.196. "
             "Dat 4.0 de khop ti le -> chrF 38.29 -> 39.55, COMET 0.7512 -> 0.7547. Mac dinh 1.0 = hanh vi cu.",
    )
    parser.add_argument(
        "--mbr",
        type=int,
        default=0,
        help="Minimum Bayes Risk: sinh K ung vien bang lay mau roi chon ung vien co chrF trung binh "
             "cao nhat so voi cac ung vien con lai. 0 = tat (dung beam nhu cu). Bo qua --num_beams/"
             "--length_penalty khi bat."
    )
    parser.add_argument("--mbr_top_p", type=float, default=0.9, help="top-p khi lay mau ung vien MBR.")
    parser.add_argument("--seed", type=int, default=0, help="Seed cho lay mau MBR (tai lap duoc).")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu)."
    )
    return parser.parse_args()


def pick_mbr(cands):
    """Chon ung vien co chrF trung binh cao nhat so voi CA tap ung vien (ke ca ban sao).

    Ban sao duoc giu lam pseudo-reference nen ung vien nao model sinh lai nhieu lan
    duoc cong diem — dung tinh than MBR (ky vong tren phan bo mo hinh), khong phai bug.
    """
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
    """
    Dịch thô danh sách phụ đề từ tiếng Anh sang tiếng Việt dựa theo logic file gốc.
    """
    all_sentences = []
    mapping = []  # Lưu (start_idx, end_idx) của câu cho mỗi đoạn phụ đề

    # 1. Tách các câu đơn lẻ từ phụ đề bằng sent_tokenize
    for sub in en_subs:
        text = sub.content
        sentences = sent_tokenize(text.strip())
        sentences = [s for s in sentences if s.strip()]
        start_idx = len(all_sentences)
        all_sentences.extend(sentences)
        end_idx = len(all_sentences)
        mapping.append((start_idx, end_idx))
    
    # 2. Dịch tất cả các câu theo từng batch
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
            # num_beams >= mbr  -> lay ung vien tu BEAM (giu duoc loi cua length_penalty),
            # nguoc lai         -> lay mau top-p. Ung vien beam manh hon han tren phim thu 3.
            extra = (dict(num_beams=num_beams, length_penalty=length_penalty, early_stopping=True)
                     if num_beams >= mbr else
                     # num_beams=1 phai dat tuong minh: generation_config.json cua mBART co
                     # san num_beams=5 -> transformers bao num_return_sequences > num_beams.
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
                # DUNG GO: voi length_penalty=4.0, early_stopping=True giu hinh phat o buoc
                # XEP HANG cac gia thuyet da hoan chinh. Doi thanh False/"never" thi hinh phat
                # di vao tieu chi dung -> beam chay toi max_length va lap chu:
                # do 08/09/2026 tren Ode to Joy: 12885 tu -> 223477 tu, BLEU 19.15 -> 0.99.
                early_stopping=True
            )

        decoded = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
        all_translations.extend(decoded)

    # 3. Ghép nối lại các câu dịch thô và gán lại vào content của phụ đề gốc
    outputs = []
    for start_idx, end_idx in mapping:
        joined = " ".join(all_translations[start_idx:end_idx]).strip()
        outputs.append(joined)
        
    for sub, output in zip(en_subs, outputs):
        sub.content = output
        
    return en_subs

def main():
    args = parse_args()
    
    # 1. Đọc và tiền xử lý file SRT tiếng Anh
    print(f"📖 Đang đọc file phụ đề tiếng Anh: {args.input_srt}")
    if not os.path.exists(args.input_srt):
        print(f"❌ Lỗi: File không tồn tại -> {args.input_srt}")
        return
        
    with open(args.input_srt, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))
        
    for sub in subtitles:
        # Xóa xuống dòng thừa, chuẩn hóa khoảng trắng thành 1 dòng duy nhất
        sub.content = " ".join(line.strip() for line in sub.content.splitlines() if line.strip())
        sub.content = re.sub(r'\s+', ' ', sub.content).strip()
        
    print(f"✓ Đã load {len(subtitles)} phụ đề.")

    # 2. Khởi tạo Tokenizer và Model dịch VinAI
    print(f"🚀 Đang tải mô hình từ: {args.model_path}")
    print(f"⚙️ Thiết bị sử dụng: {args.device}")
    
    try:
        tokenizer_en2vi = AutoTokenizer.from_pretrained(
            args.model_path,
            src_lang="en_XX",
            cache_dir=args.cache_dir
        )
        model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir
        )
        model_en2vi.to(args.device)
        if args.device == "cuda" or "cuda" in args.device:
            model_en2vi.half()  # Dùng FP16 để tăng tốc độ và tiết kiệm VRAM
        print("✓ Tải mô hình thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")
        return

    # 3. Tiến hành dịch thô
    print(f"✍️ Đang thực hiện dịch thô phụ đề... (num_beams={args.num_beams}, length_penalty={args.length_penalty})")
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
    
    end_time = time.time()
    print(f"✓ Dịch xong trong {end_time - start_time:.2f} giây.")

    # 4. Ghi file phụ đề tiếng Việt đã dịch
    os.makedirs(os.path.dirname(args.output_srt) or ".", exist_ok=True)
    srt_content = srt.compose(translated_subs)
    
    with open(args.output_srt, "w", encoding="utf-8") as f:
        f.write(srt_content)
        
    print(f"💾 Kết quả phụ đề đã được lưu tại: {args.output_srt}")

if __name__ == "__main__":
    main()
