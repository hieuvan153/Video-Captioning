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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu)."
    )
    return parser.parse_args()

def translate_en2vi(en_subs, model_en2vi, tokenizer_en2vi, device, batch_size=64):
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
    for i in range(0, len(all_sentences), batch_size):
        batch = all_sentences[i:i + batch_size]
        inputs = tokenizer_en2vi(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            output_ids = model_en2vi.generate(
                **inputs,
                decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
                num_beams=1,
                max_length=128,
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
    print(f"Đang đọc file phụ đề tiếng Anh: {args.input_srt}")
    if not os.path.exists(args.input_srt):
        print(f"Lỗi: File không tồn tại -> {args.input_srt}")
        return
        
    with open(args.input_srt, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))
        
    for sub in subtitles:
        # Xóa xuống dòng thừa, chuẩn hóa khoảng trắng thành 1 dòng duy nhất
        sub.content = " ".join(line.strip() for line in sub.content.splitlines() if line.strip())
        sub.content = re.sub(r'\s+', ' ', sub.content).strip()
        
    print(f"Đã load {len(subtitles)} phụ đề.")

    # 2. Khởi tạo Tokenizer và Model dịch VinAI
    print(f"Đang tải mô hình từ: {args.model_path}")
    print(f"Thiết bị sử dụng: {args.device}")
    
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
        print("Tải mô hình thành công!")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    # 3. Tiến hành dịch thô
    print("Đang thực hiện dịch thô phụ đề...")
    start_time = time.time()
    
    translated_subs = translate_en2vi(
        subtitles,
        model_en2vi,
        tokenizer_en2vi,
        args.device,
        batch_size=args.batch_size
    )
    
    end_time = time.time()
    print(f"Dịch xong trong {end_time - start_time:.2f} giây.")

    # 4. Ghi file phụ đề tiếng Việt đã dịch
    os.makedirs(os.path.dirname(args.output_srt), exist_ok=True)
    srt_content = srt.compose(translated_subs)
    
    with open(args.output_srt, "w", encoding="utf-8") as f:
        f.write(srt_content)
        
    print(f"Kết quả phụ đề đã được lưu tại: {args.output_srt}")

if __name__ == "__main__":
    main()
