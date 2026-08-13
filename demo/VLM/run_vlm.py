import time
start_imports_t = time.time()
import os
import cv2
import json
import torch
import gc
import argparse
import glob
import re
import transformers.image_utils as iu

# Dynamic root folder calculation (corresponds to the demo folder)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Monkey patch VideoInput if not present in the transformers version
if not hasattr(iu, "VideoInput"):
    class VideoInput:
        pass
    iu.VideoInput = VideoInput
print("Monkey patch VideoInput OK")

from transformers import AutoModelForCausalLM, AutoProcessor
end_imports_t = time.time()
print(f"Thời gian nạp thư viện (Imports): {end_imports_t - start_imports_t:.4f}s")

def patch_videollama_processor_file(cache_dir):
    """
    Auto-patch the signature of _get_arguments_from_pretrained and the common_kwargs KeyErrors
    in processing_videollama3.py inside cache to match newer transformers versions.
    Also patches modeling_videollama3_encoder.py to support Flash Attention 2.
    """
    search_path = os.path.join(cache_dir, "modules", "transformers_modules", "**", "processing_videollama3.py")
    files = glob.glob(search_path, recursive=True)
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            
            modified = False
            
            # Patch 1: signature mismatch
            target_str = "def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):"
            replacement_str = "def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):"
            if target_str in content:
                print(f"Detect signature mismatch in cache file: {fpath}. Autopatching...")
                content = content.replace(target_str, replacement_str)
                modified = True
                
            # Patch 2: KeyError: 'common_kwargs' in default_kwargs loop
            t2 = "for modality in default_kwargs:\n            default_kwargs[modality]"
            t2_full = "for modality in default_kwargs:\n            default_kwargs[modality] = ModelProcessorKwargs._defaults.get(modality, {}).copy()"
            r2_full = "for modality in default_kwargs:\n            if modality == \"common_kwargs\":\n                continue\n            default_kwargs[modality] = ModelProcessorKwargs._defaults.get(modality, {}).copy()"
            if t2_full in content:
                print(f"Detect common_kwargs loop 1 bug in: {fpath}. Autopatching...")
                content = content.replace(t2_full, r2_full)
                modified = True
                
            # Patch 3: KeyError: 'common_kwargs' in output_kwargs loop
            t3_full = "for modality in output_kwargs:\n            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__.keys():"
            r3_full = "for modality in output_kwargs:\n            if modality == \"common_kwargs\":\n                continue\n            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__.keys():"
            if t3_full in content:
                print(f"Detect common_kwargs loop 2 bug in: {fpath}. Autopatching...")
                content = content.replace(t3_full, r3_full)
                modified = True
                
            if modified:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)
                print("Autopatch successful!")
        except Exception as e:
            print(f"Error while autopatching cached file {fpath}: {e}")

    # Patch 4: modeling_videollama3_encoder.py _supports_flash_attn_2 -> _supports_flash_attn
    encoder_path = os.path.join(cache_dir, "modules", "transformers_modules", "**", "modeling_videollama3_encoder.py")
    encoder_files = glob.glob(encoder_path, recursive=True)
    for fpath in encoder_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            
            target_str = "_supports_flash_attn_2 = True"
            replacement_str = "_supports_flash_attn = True\n    _supports_flash_attn_2 = True"
            
            if target_str in content and "_supports_flash_attn = True" not in content:
                print(f"Detect missing _supports_flash_attn in: {fpath}. Autopatching...")
                content = content.replace(target_str, replacement_str)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)
                print("Autopatch encoder successful!")
        except Exception as e:
            print(f"Error while autopatching cached encoder file {fpath}: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run VideoLLama3 to caption video scenes.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "output/test_scenes"),
        help="Directory containing the scene video files (e.g. .mp4)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(ROOT_DIR, "output/test.captions.json"),
        help="JSON file path to save the captions and metadata."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="DAMO-NLP-SG/VideoLLaMA3-7B",
        help="HuggingFace model path for VideoLLaMA3."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "cache"),
        help="HuggingFace cache directory."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to load model on (e.g. cuda:0, cpu)."
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=3.0,
        help="Minimum duration (in seconds) of scene to be processed. Scenes shorter than this are skipped."
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=240,
        help="Maximum frames to sample from each video scene for the model."
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame rate to sample from the video scene (e.g. 1.0 or 0.5)."
    )
    return parser.parse_args()

def free_gpu_memory(device=None):
    """
    Clear tensor references, force garbage collection, sync CUDA, and empty cache.
    """
    gc.collect()
    if device is not None and "cuda" in device:
        torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def extract_scene_id(filename):
    """
    Tách số ID scene từ tên file (ví dụ scene_0000.mp4 -> 0).
    """
    match = re.search(r'scene_(\d+)', filename)
    if match:
        return int(match.group(1))
    # Dự phòng nếu không khớp pattern trên
    match_any_digit = re.findall(r'\d+', filename)
    if match_any_digit:
        return int(match_any_digit[-1])
    return -1


def run_vlm_captioning(
    input_dir,
    output_file,
    model_path="DAMO-NLP-SG/VideoLLaMA3-7B",
    cache_dir=None,
    device="cuda:0",
    min_duration=3.0,
    max_frames=240,
    fps=0.5,
    scenes_json_path=None
):
    if cache_dir is None:
        cache_dir = os.path.join(ROOT_DIR, "cache")
    # Auto-patch the dynamic module if already cached
    patch_videollama_processor_file(cache_dir)
    
    # 1. Tìm tất cả các file video trong thư mục input
    video_extensions = ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
        # Thêm hỗ trợ chữ hoa/thường
        video_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    # Loại bỏ trùng lặp và sắp xếp theo tên file
    video_files = sorted(list(set(video_files)))
    
    if not video_files:
        print(f"Không tìm thấy file video nào trong thư mục: {input_dir}")
        return
        
    print(f"Tìm thấy {len(video_files)} file video cần xử lý.")
    
    # 2. Thử tìm file metadata để ánh xạ
    metadata_map = {}
    if scenes_json_path and os.path.exists(scenes_json_path):
        try:
            with open(scenes_json_path, "r", encoding="utf-8") as f:
                meta_list = json.load(f)
                for item in meta_list:
                    if "scene_id" in item:
                        metadata_map[int(item["scene_id"])] = item
            print(f"Đã tải thành công metadata từ: {scenes_json_path}")
        except Exception as e:
            print(f"Lỗi khi đọc file metadata {scenes_json_path}: {e}")
    else:
        # Dự phòng tìm result.json
        possible_metadata_paths = [
            os.path.join(input_dir, "result.json"),
            os.path.join(os.path.dirname(input_dir), "result.json")
        ]
        for meta_path in possible_metadata_paths:
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_list = json.load(f)
                        for item in meta_list:
                            if "scene_id" in item:
                                metadata_map[int(item["scene_id"])] = item
                    print(f"Đã tải thành công metadata từ: {meta_path}")
                    break
                except Exception as e:
                    print(f"Lỗi khi đọc file metadata {meta_path}: {e}")

    # 3. Khởi tạo mô hình VideoLLama3
    print(f"Đang tải mô hình {model_path} lên {device}...")
    t_load_start = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            cache_dir=cache_dir
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        t_load_end = time.time()
        print(f"Tải mô hình thành công! Thời gian load: {t_load_end - t_load_start:.4f}s")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    results = []

    # 4. Duyệt qua từng file video để phân tích
    for idx, video_path in enumerate(video_files):
        filename = os.path.basename(video_path)
        scene_id = extract_scene_id(filename)
        
        print(f"[{idx+1}/{len(video_files)}] Đang xử lý: {filename} (scene_id: {scene_id})")
        
        # Đọc thông tin video bằng OpenCV
        cap = cv2.VideoCapture(video_path)
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if fps_video == 0 or frame_count == 0:
            print(f"  -> Lỗi: Không thể mở video hoặc fps/frame_count bằng 0. Bỏ qua.")
            cap.release()
            continue
            
        duration_sec = frame_count / fps_video
        cap.release()
        
        # Kiểm tra độ dài tối thiểu
        if duration_sec < min_duration:
            print(f"  -> Bỏ qua scene do thời lượng ngắn ({duration_sec:.2f}s < {min_duration}s)")
            caption_text = ""
        else:
            t_infer_start = time.time()
            # Tạo conversation theo prompt chuẩn của VideoLLama3
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": {"video_path": video_path, "fps": fps, "max_frames": max_frames}},
                        {"type": "text", "text": """Analyze the video visual content.
                        
Output strict format:
1. Summary: One concise sentence describing the main action.
2. Main Characters: List prominent humans only. Format: [ID] - [Gender] - [Age Range].
3. Relationship: [Character A & Character B] - [Type] (or "None" if N/A).

Constraints: Use visual evidence only. Ignore background crowds. Strictly limit to maximum 6 characters.
"""},
                    ]
                },
            ]
            
            try:
                t_prep_start = time.time()
                inputs = processor(
                    conversation=conversation,
                    add_system_prompt=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                
                # Chuyển inputs lên device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                t_prep_end = time.time()
                
                # Sinh caption
                t_gen_start = time.time()
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, max_new_tokens=300)
                t_gen_end = time.time()
                
                caption_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                t_infer_end = time.time()
                print(f"  -> Caption thành công. Chi tiết thời gian:")
                print(f"     + Chuẩn bị inputs (processor & video extraction): {t_prep_end - t_prep_start:.4f}s")
                print(f"     + Sinh text (model.generate): {t_gen_end - t_gen_start:.4f}s")
                print(f"     + Tổng thời gian infer scene: {t_infer_end - t_infer_start:.4f}s")
            except Exception as e:
                print(f"  -> Lỗi khi phân tích video: {e}")
                caption_text = f"Error during analysis: {e}"
            finally:
                free_gpu_memory(device)

        # Lấy thông tin metadata sẵn có
        meta_info = metadata_map.get(scene_id, {})
        
        scene_result = {
            "scene_id": scene_id if scene_id != -1 else idx,
            "video_file": filename,
            "video_path": video_path,
            "duration": duration_sec,
            "start_time": meta_info.get("start_time", None),
            "end_time": meta_info.get("end_time", None),
            "caption": caption_text
        }
        
        results.append(scene_result)
        
        # Lưu kết quả tạm thời sau mỗi scene đề phòng lỗi nửa chừng
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nHoàn tất! Kết quả cuối cùng đã được lưu tại: {output_file}")


def main():
    args = parse_args()
    run_vlm_captioning(
        input_dir=args.input_dir,
        output_file=args.output_file,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        device=args.device,
        min_duration=args.min_duration,
        max_frames=args.max_frames,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
