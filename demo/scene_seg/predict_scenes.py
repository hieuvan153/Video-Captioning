#!/usr/bin/env python
import os
import sys
import json
import subprocess
import argparse
import shutil
import pickle
from fractions import Fraction
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# Dynamic root folder calculation (corresponds to the demo folder)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(ROOT_DIR)

# Resolve local dependencies path inside this folder
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.join(LOCAL_DIR, "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)

import importlib.util

# Load shotdetect script
shotdetect_spec = importlib.util.spec_from_file_location("shotdetect_script", os.path.join(src_root, "shotdetect.py"))
shotdetect_module = importlib.util.module_from_spec(shotdetect_spec)
shotdetect_spec.loader.exec_module(shotdetect_module)

# Load extract_embeddings script
embeddings_spec = importlib.util.spec_from_file_location("extract_embeddings", os.path.join(src_root, "extract_embeddings.py"))
extract_embeddings = importlib.util.module_from_spec(embeddings_spec)
embeddings_spec.loader.exec_module(extract_embeddings)
from BiLSTM_protocol import BiLSTM
from movienet_seg_data import MovieNet_SceneSeg_Dataset_Embeddings_Val

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print(f"=> Loaded checkpoint from {ckpt_path} (epoch {checkpoint.get('epoch', 'N/A')})")

def extract_peaks_1d_all(probs):
    peaks = []
    n = len(probs)
    for i in range(n):
        left = probs[i - 1] if i - 1 >= 0 else -np.inf
        right = probs[i + 1] if i + 1 < n else -np.inf
        if probs[i] >= left and probs[i] >= right:
            peaks.append(i)
    return peaks

def suppress_close_boundaries(boundaries, probs, min_gap=2):
    suppressed = []
    i = 0
    n = len(boundaries)
    while i < n:
        j = i
        best = boundaries[i]
        while j + 1 < n and boundaries[j + 1] - boundaries[i] < min_gap:
            j += 1
            if probs[boundaries[j]] > probs[best]:
                best = boundaries[j]
        suppressed.append(best)
        i = j + 1
    return sorted(suppressed)

def refine_boundaries_from_probs(
    preb_all,
    thresh_high=0.5,
    max_scene_shots=40,
    split_prob_thresh=0.02
):
    num_shots = len(preb_all)
    B_high = [i for i, p in enumerate(preb_all) if p >= thresh_high]
    if 0 not in B_high:
        B_high.insert(0, 0)
    B_high = sorted(B_high)
    B_high = suppress_close_boundaries(B_high, preb_all, min_gap=2)

    B_peaks = extract_peaks_1d_all(preb_all)
    final_boundaries = set()

    for i, start in enumerate(B_high):
        end = B_high[i + 1] if i + 1 < len(B_high) else num_shots
        stack = [(start, end)]

        while stack:
            s, e = stack.pop()
            final_boundaries.add(s)

            if e - s <= max_scene_shots:
                continue

            candidates = [b for b in B_peaks if s < b < e]
            if candidates:
                best_b = max(candidates, key=lambda b: preb_all[b])
                if preb_all[best_b] >= split_prob_thresh:
                    split_b = best_b
                else:
                    split_b = (s + e) // 2
            else:
                split_b = (s + e) // 2

            if split_b <= s or split_b >= e:
                continue

            stack.append((s, split_b))
            stack.append((split_b, e))

    return sorted(final_boundaries)

@torch.no_grad()
def inference(args, model, loader):
    model.eval()
    stride = args.seq_len // 2
    for batch_idx, (data, target, imdb) in enumerate(loader):
        data = data.view(-1, args.dim).cuda(non_blocking=True)
        target = target.view(-1)
        data_len = data.size(0)
        gt_len = target.size(0)
        prob_all = []
        for w_id in range(data_len // stride):
            start_pos = w_id * stride
            _data = data[start_pos:start_pos + args.seq_len].unsqueeze(0)
            output = model(_data, None)
            output = output.view(-1, 2)
            prob = output[:, 1]
            prob = prob[stride // 2:stride + stride // 2].squeeze()
            prob_all.append(prob.cpu())
        
        preb_all = torch.cat(prob_all, axis=0)[:gt_len].numpy()
        refined_boundaries = refine_boundaries_from_probs(
            preb_all,
            thresh_high=args.thresh_high,
            max_scene_shots=args.max_scene_shots,
            split_prob_thresh=args.split_prob_thresh
        )
        return refined_boundaries, preb_all

def get_fps(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    fps_str = subprocess.check_output(cmd).decode().strip()
    return float(Fraction(fps_str))

def cut_scenes_ffmpeg(video_path, boundaries, shots, fps, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    num_shots = len(shots)
    print(f"Cutting scene videos and saving to: {output_dir}")
    for i in range(len(boundaries)):
        start_shot = boundaries[i]
        end_shot = (
            boundaries[i + 1] - 1
            if i + 1 < len(boundaries)
            else num_shots - 1
        )

        start_frame = shots[start_shot][0]
        end_frame = shots[end_shot][1]

        start_time = start_frame / fps
        end_time = (end_frame + 1) / fps
        duration = end_time - start_time

        out_path = os.path.join(output_dir, f"scene_{i:04d}.mp4")

        # Fast stream copying (-c copy) to avoid CPU/GPU re-encoding overhead
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-ss", f"{start_time:.6f}",
            "-i", video_path,
            "-t", f"{duration:.6f}",
            "-c", "copy",
            out_path
        ]
        subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Demo SCRL Video Scene Segmentation & Cutting")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_json", type=str, default="output/result.json", help="Path to save output scene timestamps JSON")
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA GPU ID")
    parser.add_argument("--temp_dir", type=str, default="./temp_predict_demo", help="Temporary directory for runtime artifacts")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary directory after processing")
    
    # Scene cutting options
    parser.add_argument("--cut_scenes", type=bool, default=True, help="Automatically cut the video into scene .mp4 files")
    parser.add_argument("--output_scenes_dir", type=str, default=None, 
                        help="Folder to save the cut mp4 scenes (default: output_json_basename_scenes)")
    
    # Model parameters
    parser.add_argument("--scrl_checkpoint", type=str, 
                        default=os.path.join(ROOT_DIR, "model/scene_seg/checkpoint_0099.pth.tar"),
                        help="Path to SCRL encoder checkpoint")
    parser.add_argument("--bilstm_checkpoint", type=str, 
                        default=os.path.join(ROOT_DIR, "model/scene_seg/model_best.pth.tar"),
                        help="Path to BiLSTM classifier checkpoint")
    
    # Thresholds
    parser.add_argument("--thresh_high", type=float, default=0.5, help="High confidence threshold for anchor boundary")
    parser.add_argument("--max_scene_shots", type=int, default=40, help="Maximum number of shots in a scene before splitting")
    parser.add_argument("--split_prob_thresh", type=float, default=0.02, help="Probability threshold for splitting long scenes")

    args = parser.parse_args()

    # Verify input video exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    run_scene_segmentation(
        video_path=args.video_path,
        output_json=args.output_json,
        gpu_id=args.gpu_id,
        temp_dir=args.temp_dir,
        keep_temp=args.keep_temp,
        cut_scenes=args.cut_scenes,
        output_scenes_dir=args.output_scenes_dir,
        scrl_checkpoint=args.scrl_checkpoint,
        bilstm_checkpoint=args.bilstm_checkpoint,
        thresh_high=args.thresh_high,
        max_scene_shots=args.max_scene_shots,
        split_prob_thresh=args.split_prob_thresh
    )


def run_scene_segmentation(
    video_path,
    output_json,
    gpu_id="0",
    temp_dir="./temp_predict_demo",
    keep_temp=False,
    cut_scenes=True,
    output_scenes_dir=None,
    scrl_checkpoint=None,
    bilstm_checkpoint=None,
    thresh_high=0.5,
    max_scene_shots=40,
    split_prob_thresh=0.02
):
    if scrl_checkpoint is None:
        scrl_checkpoint = os.path.join(ROOT_DIR, "model/scene_seg/checkpoint_0099.pth.tar")
    if bilstm_checkpoint is None:
        bilstm_checkpoint = os.path.join(ROOT_DIR, "model/scene_seg/model_best.pth.tar")

    # Set CUDA device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    setup_seed(100)

    # Setup directories
    video_abs_path = os.path.abspath(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_dir = os.path.abspath(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Set outputs paths
    output_json_abs = os.path.abspath(output_json)
    os.makedirs(os.path.dirname(output_json_abs), exist_ok=True)

    if output_scenes_dir is None:
        output_scenes_dir = os.path.splitext(output_json_abs)[0] + "_scenes"
    output_scenes_dir_abs = os.path.abspath(output_scenes_dir)

    print(f"--- Step 1: Run Shot Detection & Keyframe Extraction ---")
    shot_args = SimpleNamespace(
        video_path=video_abs_path,
        save_data_root_path=temp_dir,
        print_result=False,
        save_keyf=True,
        save_keyf_txt=True,
        split_video=False,
        keep_resolution=False,
        avg_sample=False,
        begin_time=None,
        end_time=120.0,
        begin_frame=None,
        end_frame=1000
    )
    print(f"Executing shotdetect.main directly in Python")
    shotdetect_module.main(shot_args, temp_dir)

    # Paths created by shotdetect.py
    shot_txt_file = os.path.join(temp_dir, "shot_txt", f"{video_name}.txt")
    shot_keyf_dir = os.path.join(temp_dir, "shot_keyf")

    if not os.path.exists(shot_txt_file):
        print(f"Error: Shot boundary file was not generated: {shot_txt_file}")
        return

    # Read shots and count lines
    with open(shot_txt_file, "r") as f:
        num_lines = sum(1 for line in f if line.strip())
    print(f"Detected {num_lines} shots.")
    
    fps = get_fps(video_abs_path)
    print(f"Video FPS: {fps}")

    # Read shots start/end frames
    shots = []
    with open(shot_txt_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            start_f, end_f, *_ = map(int, line.split())
            shots.append((start_f, end_f))
    num_shots = len(shots)

    if num_lines < 2:
        print("Warning: Too few shots detected. The video might be too short.")
        # Create minimal 1 scene output
        if shots:
            start_t = shots[0][0] / fps
            end_t = (shots[-1][1] + 1) / fps
            scenes = [{"scene_id": 0, "start_time": round(start_t, 6), "end_time": round(end_t, 6), "duration": round(end_t - start_t, 6)}]
        else:
            scenes = []
        with open(output_json_abs, "w", encoding="utf-8") as f:
            json.dump(scenes, f, indent=2)
        print(f"Saved fallback metadata to {output_json_abs}")
        
        if cut_scenes and shots:
            cut_scenes_ffmpeg(video_abs_path, [0], shots, fps, output_scenes_dir_abs)
        return

    print(f"--- Step 2: Create Dummy JSON file for Embeddings Extraction ---")
    dummy_json_path = os.path.join(temp_dir, f"{video_name}.v1.json")
    dummy_data = {
        "test": [
            {
                "name": video_name,
                "index": 0,
                "shot_count": num_lines,
                "label": [[f"{i:04d}", str(i)] for i in range(num_lines - 1)]
            }
        ]
    }
    with open(dummy_json_path, "w", encoding="utf-8") as f:
        json.dump(dummy_data, f, indent=4)
    print(f"Dummy metadata JSON written to {dummy_json_path}")

    print(f"--- Step 3: Run SCRL Embedding Feature Extraction ---")
    embeddings_save_dir = os.path.join(temp_dir, "embeddings")
    os.makedirs(embeddings_save_dir, exist_ok=True)

    cfg_embed = SimpleNamespace(
        model_path=scrl_checkpoint,
        shot_info_path=dummy_json_path,
        shot_img_path=shot_keyf_dir,
        Type="test",
        model_name="resnet50",
        frame_per_shot=3,
        shot_num=1,
        worker_num=0,
        bs=64,
        save_dir=embeddings_save_dir,
        gpu_id=gpu_id,
        log_file=os.path.join(embeddings_save_dir, "extraction.log")
    )
    print("Executing extract_embeddings.extract_features directly in Python")
    extract_embeddings.extract_features(cfg_embed)

    embeddings_pkl = os.path.join(embeddings_save_dir, "test.pkl")
    if not os.path.exists(embeddings_pkl):
        print(f"Error: Embeddings pkl file not found: {embeddings_pkl}")
        return

    print(f"--- Step 4: Run BiLSTM Scene Segmentation Predictor ---")
    # BiLSTM Config setup
    bilstm_args = SimpleNamespace(
        pkl_path_test=embeddings_pkl,
        test_bs=1,
        shot_num=10,
        gpu_id=gpu_id,
        input_drop_rate=0.2,
        workers=1,
        dim=2048,
        seq_len=40,
        thresh_high=thresh_high,
        max_scene_shots=max_scene_shots,
        split_prob_thresh=split_prob_thresh
    )

    # Initialize model
    model = BiLSTM(
        input_feature_dim=bilstm_args.dim,
        input_drop_rate=bilstm_args.input_drop_rate
    ).cuda()

    load_checkpoint(model, bilstm_checkpoint)

    # Load dataset
    test_dataset = MovieNet_SceneSeg_Dataset_Embeddings_Val(
        pkl_path=bilstm_args.pkl_path_test,
        sampled_shot_num=bilstm_args.seq_len
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, bilstm_args.test_bs, num_workers=bilstm_args.workers,
        shuffle=False, pin_memory=True, drop_last=False
    )

    boundaries, _ = inference(bilstm_args, model, test_loader)
    print(f"Predicted scene boundary shot indices: {boundaries}")

    print(f"--- Step 5: Convert Boundaries to Timestamps ---")
    scenes_metadata = []

    for i in range(len(boundaries)):
        start_shot = boundaries[i]
        end_shot = (
            boundaries[i + 1] - 1
            if i + 1 < len(boundaries)
            else num_shots - 1
        )

        start_frame = shots[start_shot][0]
        end_frame = shots[end_shot][1]

        start_time = start_frame / fps
        end_time = (end_frame + 1) / fps
        duration = end_time - start_time

        scenes_metadata.append({
            "scene_id": i,
            "start_time": round(start_time, 6),
            "end_time": round(end_time, 6),
            "duration": round(duration, 6)
        })

    # Save to JSON
    with open(output_json_abs, "w", encoding="utf-8") as f:
        json.dump(scenes_metadata, f, indent=2)
    print(f"Scene segmentation timestamps saved to {output_json_abs}")

    # Optional: Cut scene videos
    if cut_scenes:
        print(f"--- Step 6: Cut Scene Video Clips ---")
        cut_scenes_ffmpeg(video_abs_path, boundaries, shots, fps, output_scenes_dir_abs)

    print(f"[SUCCESS] Scene segmentation and video cutting completed successfully!")

    # Cleanup
    if not keep_temp:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
