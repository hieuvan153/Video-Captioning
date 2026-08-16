import os
import re
import gc
import sys
import json
import time
import argparse
import subprocess
import ffmpeg
import torch

# Dynamic root folder calculation (corresponds to the demo folder)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Force all model caches to go to /data/ mount to avoid filling up the home partition (which only has 2.0GB)
os.environ["HF_HOME"] = os.path.join(ROOT_DIR, "cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(ROOT_DIR, "cache/huggingface")
os.environ["TORCH_HOME"] = os.path.join(ROOT_DIR, "cache/torch")
os.environ["XDG_CACHE_HOME"] = os.path.join(ROOT_DIR, "cache")

def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set seed for determinism
set_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Video Subtitle Generation Pipeline")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file (.mp4, .mkv, etc.)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT_DIR, "output"), help="Directory to save all outputs")
    parser.add_argument("--cache_dir", type=str, default=os.path.join(ROOT_DIR, "cache"), help="Hugging Face cache directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic inference")
    parser.add_argument("--vlm_fps", type=float, default=1, help="Frame sampling rate for VideoLLaMA3")
    parser.add_argument("--llm_batch_size", type=int, default=10, help="Batch size for Gemma3 subtitle refinement")
    parser.add_argument("--use_registry", action="store_true",
                        help="Build a character-relationship registry (step 5b) "
                             "and inject it into the Gemma refinement prompt.")
    parser.add_argument("--speaker_tagged_json", type=str, default=None,
                        help="Existing CAM++ output (movie_XXX.tagged.json). "
                             "Enables step 5c: per-line speaker attribution, "
                             "[SPEAKER: X] tags, and per-scene registry scoping. "
                             "Requires --use_registry.")
    return parser.parse_args()

def free_gpu_memory():
    """Clear memory references, force garbage collection, and empty PyTorch CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("--- GPU Memory Cleared ---", flush=True)

def step1_extract_audio(video_path, audio_path):
    print("\n=== STEP 1: Extracting Audio ===", flush=True)
    if os.path.exists(audio_path):
        print(f"Audio file already exists: {audio_path}. Skipping extraction.", flush=True)
        return
    print(f"Extracting audio to {audio_path}...", flush=True)
    ffmpeg.input(video_path).output(
        audio_path,
        ar="16000",
        ac="1",
        acodec="pcm_s16le",
        map_metadata="-1",
        fflags="+bitexact",
    ).overwrite_output().run(quiet=True)
    print("Audio extraction complete.", flush=True)

def step2_run_asr(audio_path, output_dir, base_name):
    print("\n=== STEP 2: Running ASR (Whisper + VAD) ===", flush=True)
    english_srt_path = os.path.join(output_dir, f"{base_name}.(Tiếng Anh).srt")
    if os.path.exists(english_srt_path):
        print(f"English SRT already exists: {english_srt_path}. Skipping ASR.", flush=True)
        return english_srt_path

    # Dynamically import and run asr_movie_infer
    sys.path.append(os.path.join(ROOT_DIR, "ASR"))
    import asr_movie_infer
    
    print("Running Whisper transcription...", flush=True)
    asr_movie_infer.run(audio_path, out_dir=output_dir, out_name=base_name)
    
    # Unload ASR model to free VRAM
    print("Unloading ASR model...", flush=True)
    del asr_movie_infer.model_asr
    if "asr_movie_infer" in sys.modules:
        del sys.modules["asr_movie_infer"]
    free_gpu_memory()
    
    return english_srt_path

def step3_run_scene_seg(video_path, scenes_json_path, cut_scenes_dir):
    print("\n=== STEP 3: Running Scene Segmentation (SCRL + BiLSTM) ===", flush=True)
    if os.path.exists(scenes_json_path) and os.path.exists(cut_scenes_dir) and os.listdir(cut_scenes_dir):
        print(f"Scene segmentation results already exist. Skipping.", flush=True)
        return

    sys.path.append(os.path.join(ROOT_DIR, "scene_seg"))
    import predict_scenes
    
    predict_scenes.run_scene_segmentation(
        video_path=video_path,
        output_json=scenes_json_path,
        output_scenes_dir=cut_scenes_dir,
        gpu_id="0",
        scrl_checkpoint=os.path.join(ROOT_DIR, "model/scene_seg/checkpoint_0099.pth.tar"),
        bilstm_checkpoint=os.path.join(ROOT_DIR, "model/scene_seg/model_best.pth.tar")
    )
    
    # Unload predict_scenes from sys.modules and clear VRAM
    print("Unloading Scene Seg Model...", flush=True)
    if "predict_scenes" in sys.modules:
        del sys.modules["predict_scenes"]
    free_gpu_memory()
    print("Scene segmentation complete.", flush=True)

def step4_run_vlm(cut_scenes_dir, scenes_json_path, vlm_json_path, cache_dir, vlm_fps):
    print("\n=== STEP 4: Running VLM (VideoLLama3) ===", flush=True)
    if os.path.exists(vlm_json_path):
        print(f"VLM context captions already exist: {vlm_json_path}. Skipping VLM.", flush=True)
        return vlm_json_path

    sys.path.append(os.path.join(ROOT_DIR, "VLM"))
    import run_vlm
    
    run_vlm.run_vlm_captioning(
        input_dir=cut_scenes_dir,
        output_file=vlm_json_path,
        model_path="DAMO-NLP-SG/VideoLLaMA3-7B",
        cache_dir=cache_dir,
        device="cuda:0",
        min_duration=3.0,
        max_frames=240,
        fps=vlm_fps,
        scenes_json_path=scenes_json_path
    )

    # Clean VLM model from VRAM
    print("Unloading VLM Model...", flush=True)
    if "run_vlm" in sys.modules:
        del sys.modules["run_vlm"]
    free_gpu_memory()
    
    return vlm_json_path

def step5_run_nmt(english_srt_path, rough_srt_path, cache_dir):
    print("\n=== STEP 5: Running NMT (MBart Translation) ===", flush=True)
    if os.path.exists(rough_srt_path):
        print(f"Rough Vietnamese SRT already exists: {rough_srt_path}. Skipping NMT.", flush=True)
        return rough_srt_path

    sys.path.append(os.path.join(ROOT_DIR, "NMT"))
    import run_nmt
    import srt
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    # Decide model path
    local_mbart = os.path.join(ROOT_DIR, "model/NMT/mbart_model")
    if os.path.exists(local_mbart):
        model_path = local_mbart
    elif os.path.exists("/data/ndloc_bk/app/model/mbart_model"):
        model_path = "/data/ndloc_bk/app/model/mbart_model"
    elif os.path.exists("/data/ndloc_bk/ntVan/infer/model/mbart_model"):
        model_path = "/data/ndloc_bk/ntVan/infer/model/mbart_model"
    else:
        model_path = "vinai/vinai-translate-en2vi-v2"
        
    print(f"Loading MBart Model from {model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="en_XX", cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, cache_dir=cache_dir).to("cuda")
    model.half() # Use FP16 for efficiency
    
    with open(english_srt_path, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))
        
    for sub in subtitles:
        sub.content = " ".join(line.strip() for line in sub.content.splitlines() if line.strip())
        sub.content = re.sub(r'\s+', ' ', sub.content).strip()
        
    print("Translating subtitles...", flush=True)
    translated_subs = run_nmt.translate_en2vi(
        subtitles,
        model,
        tokenizer,
        "cuda",
        batch_size=64
    )
    
    with open(rough_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(translated_subs))
    print(f"Rough translation complete. Saved to: {rough_srt_path}", flush=True)

    # Clean NMT model from VRAM
    print("Unloading NMT Model...", flush=True)
    del model
    del tokenizer
    if "run_nmt" in sys.modules:
        del sys.modules["run_nmt"]
    free_gpu_memory()
    
    return rough_srt_path

def step5b_build_registry(english_srt_path, vlm_json_path, registry_json_path, cache_dir):
    print("\n=== STEP 5b: Building Character Registry (Gemma base) ===", flush=True)
    if os.path.exists(registry_json_path):
        print(f"Registry already exists: {registry_json_path}. Skipping.", flush=True)
        return registry_json_path

    sys.path.append(ROOT_DIR)
    from CHARACTER import build_registry

    build_registry.run(
        en_srt_path=english_srt_path,
        vlm_json_path=vlm_json_path,
        output_json_path=registry_json_path,
        cache_dir=cache_dir,
    )

    print("Unloading Registry Model...", flush=True)
    for mod in ("CHARACTER.build_registry",):
        if mod in sys.modules:
            del sys.modules[mod]
    free_gpu_memory()
    return registry_json_path

def step5c_build_speakers(tagged_json_path, english_srt_path, registry_json_path,
                          vlm_json_path, speaker_json_path, cache_dir):
    print("\n=== STEP 5c: Speaker Attribution (CAM++ align + naming) ===", flush=True)
    if os.path.exists(speaker_json_path):
        print(f"Speaker JSON already exists: {speaker_json_path}. Skipping.", flush=True)
        return speaker_json_path

    sys.path.append(ROOT_DIR)
    from SPEAKER import build_speakers

    build_speakers.run(
        tagged_json_path=tagged_json_path,
        en_srt_path=english_srt_path,
        registry_json_path=registry_json_path,
        output_json_path=speaker_json_path,
        vlm_json_path=vlm_json_path,
        cache_dir=cache_dir,
    )

    print("Unloading Speaker Naming Model...", flush=True)
    if "SPEAKER.build_speakers" in sys.modules:
        del sys.modules["SPEAKER.build_speakers"]
    free_gpu_memory()
    return speaker_json_path

def step6_run_llm(english_srt_path, rough_srt_path, vlm_json_path, output_srt_path, cache_dir, llm_batch_size, registry_json_path=None, speaker_json_path=None):
    print("\n=== STEP 6: Running LLM Refinement (Gemma3) ===", flush=True)
    if os.path.exists(output_srt_path):
        print(f"Refined Vietnamese SRT already exists: {output_srt_path}. Skipping LLM.", flush=True)
        return output_srt_path

    sys.path.append(os.path.join(ROOT_DIR, "LLM"))
    import refine_llm
    
    refine_llm.refine_subtitles(
        en_srt_path=english_srt_path,
        vinai_srt_path=rough_srt_path,
        vlm_json_path=vlm_json_path,
        output_srt_path=output_srt_path,
        adapter_model_name="thevan2404/best_gemma_scene_context",
        cache_dir=cache_dir,
        max_seq_length=4096 if registry_json_path else 2048,
        max_new_tokens=1024,
        llm_batch_size=llm_batch_size,
        registry_json_path=registry_json_path,
        speaker_json_path=speaker_json_path,
    )

    # Unload Gemma3 model from VRAM
    print("Unloading LLM Model...", flush=True)
    if "refine_llm" in sys.modules:
        del sys.modules["refine_llm"]
    free_gpu_memory()
    
    return output_srt_path

def main():
    args = parse_args()
    set_seed(args.seed)
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}", flush=True)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.video_path))[0]
    
    # Define intermediate paths
    audio_path = os.path.join(args.output_dir, f"{base_name}.wav")
    scenes_json_path = os.path.join(args.output_dir, f"{base_name}.scenes.json")
    cut_scenes_dir = os.path.join(args.output_dir, f"{base_name}_scenes")
    vlm_json_path = os.path.join(args.output_dir, f"{base_name}.captions.json")
    rough_srt_path = os.path.join(args.output_dir, f"{base_name}.(Tiếng Việt_dich_tho).srt")
    output_srt_path = os.path.join(args.output_dir, f"{base_name}.(Tiếng Việt_tinh_chinh).srt")
    registry_json_path = os.path.join(args.output_dir, f"{base_name}.registry.json")
    speaker_json_path = os.path.join(args.output_dir, f"{base_name}.speakers.json")

    t_start = time.time()
    durations = {}

    # Step 1: Extract Audio
    t0 = time.time()
    step1_extract_audio(args.video_path, audio_path)
    durations["Step 1: Extract Audio"] = time.time() - t0

    # Step 2: Run ASR (Whisper + VAD)
    t0 = time.time()
    english_srt_path = step2_run_asr(audio_path, args.output_dir, base_name)
    durations["Step 2: Run ASR (Whisper)"] = time.time() - t0

    # Step 3: Run Scene Segmentation (Subprocess)
    t0 = time.time()
    step3_run_scene_seg(args.video_path, scenes_json_path, cut_scenes_dir)
    durations["Step 3: Scene Segmentation"] = time.time() - t0

    # Step 4: Run VLM
    t0 = time.time()
    step4_run_vlm(cut_scenes_dir, scenes_json_path, vlm_json_path, args.cache_dir, args.vlm_fps)
    durations["Step 4: VLM Context Captioning"] = time.time() - t0

    # Step 5: Run NMT
    t0 = time.time()
    step5_run_nmt(english_srt_path, rough_srt_path, args.cache_dir)
    durations["Step 5: NMT Translation (MBart)"] = time.time() - t0

    # Step 5b: Character Registry (optional)
    if args.use_registry:
        t0 = time.time()
        step5b_build_registry(english_srt_path, vlm_json_path,
                              registry_json_path, args.cache_dir)
        durations["Step 5b: Character Registry"] = time.time() - t0

    # Step 5c: Speaker attribution (optional; needs the registry as its name set)
    use_speaker = bool(args.speaker_tagged_json) and args.use_registry
    if args.speaker_tagged_json and not args.use_registry:
        print("Warning: --speaker_tagged_json needs --use_registry (the registry "
              "supplies the closed set of names); skipping step 5c.", flush=True)
    if use_speaker:
        t0 = time.time()
        step5c_build_speakers(args.speaker_tagged_json, english_srt_path,
                              registry_json_path, vlm_json_path,
                              speaker_json_path, args.cache_dir)
        durations["Step 5c: Speaker Attribution"] = time.time() - t0

    # Step 6: Run LLM refinement
    t0 = time.time()
    step6_run_llm(english_srt_path, rough_srt_path, vlm_json_path, output_srt_path, args.cache_dir, args.llm_batch_size,
                  registry_json_path=registry_json_path if args.use_registry else None,
                  speaker_json_path=speaker_json_path if use_speaker else None)
    durations["Step 6: LLM Refinement (Gemma3)"] = time.time() - t0

    total_time = time.time() - t_start
    print(f"\n=========================================")
    print(f"Pipeline completed successfully in {total_time/60:.2f} minutes!")
    print(f"=========================================")
    print("Step Execution Time Breakdown:")
    for step_name, elapsed in durations.items():
        print(f" - {step_name}: {elapsed/60:.2f} min ({elapsed:.1f}s)")
    print(f"=========================================", flush=True)

if __name__ == "__main__":
    main()
