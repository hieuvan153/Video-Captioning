import torch
torch.set_num_threads(4)
import whisper
import os
import ffmpeg
import srt
from tqdm import tqdm
import datetime
import json
import shutil

# Dynamic root folder calculation (corresponds to the demo folder)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Running Whisper...")
model_asr = whisper.load_model(os.path.join(ROOT_DIR, "model/ASR/whisper-medium-13-openai.pt"))

# Whisper Transcription Parameters
language = "english"
translation_mode = "No translation"
max_attempts = 1
verbose = False

# VAD Settings
vad_threshold = 0.2
chunk_threshold = 3.0

# Decoding Options
best_of = None
beam_size = None
patience = None
length_penalty = None
prefix = ""
suppress_tokens = "-1"
suppress_blank = True
without_timestamps = False
max_initial_timestamp = 1.0
fp16 = True

# Transcriber Settings
temperature = 0.0
compression_ratio_threshold = 2.4
logprob_threshold = -1.0
no_speech_threshold = 0.9
condition_on_previous_text = False
initial_prompt = ""
word_timestamps = True
clip_timestamps = "0"
hallucination_silence_threshold = 2.0

if translation_mode == "End-to-end Whisper (default)":
    task = "translate"
elif translation_mode == "Whisper -> DeepL":
    task = "transcribe"
elif translation_mode == "No translation":
    task = "transcribe"
else:
    raise ValueError("Invalid translation mode")

transcription_options = {
    "verbose": verbose,
    "compression_ratio_threshold": compression_ratio_threshold,
    "logprob_threshold": logprob_threshold,
    "no_speech_threshold": no_speech_threshold,
    "condition_on_previous_text": condition_on_previous_text,
    "initial_prompt": initial_prompt,
    "word_timestamps": word_timestamps,
    "clip_timestamps": clip_timestamps,
    "hallucination_silence_threshold": hallucination_silence_threshold
}

decoding_options = {
    "task": task,
    "language": language,
    "temperature": temperature,
    "best_of": best_of,
    "beam_size": beam_size,
    "patience": patience,
    "length_penalty": length_penalty,
    "prefix": prefix,
    "suppress_tokens": suppress_tokens,
    "suppress_blank": suppress_blank,
    "without_timestamps": without_timestamps,
    "max_initial_timestamp": max_initial_timestamp,
    "fp16": fp16,
}

def run(audio_path, out_dir = None, out_name = None):
    file_name = os.path.basename(audio_path)
    name_only = os.path.splitext(file_name)[0]
    
    if out_dir is None:
        if out_name is None:
            out_path = "output/"+name_only+".(Tiếng Anh).srt"
        else:
            out_path = "output/"+out_name+".(Tiếng Anh).srt"
    else:
        if out_name is None:
            out_path = os.path.join(out_dir, name_only+".(Tiếng Anh).srt")
        else:
            out_path = os.path.join(out_dir, out_name+".(Tiếng Anh).srt")

    print("Encoding audio...")
    if os.path.exists("vad_chunks"):
        shutil.rmtree("vad_chunks")

    os.mkdir("vad_chunks")
    ffmpeg.input(audio_path).output(
        "vad_chunks/silero_temp.wav",
        ar="16000",
        ac="1",
        acodec="pcm_s16le",
        map_metadata="-1",
        fflags="+bitexact",
    ).overwrite_output().run(quiet=True)

    print("Running VAD...")
    model, utils = torch.hub.load(
        repo_or_dir=os.path.join(ROOT_DIR, "model/ASR/silero-vad"),
        model="silero_vad",
        onnx=True,
        source="local"
    )

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Custom implementation of read_audio & save_audio to bypass torchaudio/torchcodec CUDA dependencies
    def custom_read_audio(path: str, sampling_rate: int = 16000) -> torch.Tensor:
        import soundfile as sf
        wav, sr = sf.read(path, dtype='float32')
        if sr != sampling_rate:
            raise ValueError(f"Sampling rate of {path} is {sr}, expected {sampling_rate}")
        return torch.from_numpy(wav)

    def custom_save_audio(path: str, tensor: torch.Tensor, sampling_rate: int = 16000):
        import soundfile as sf
        sf.write(path, tensor.numpy(), sampling_rate)

    VAD_SR = 16000
    wav = custom_read_audio("vad_chunks/silero_temp.wav", sampling_rate=VAD_SR)
    t = get_speech_timestamps(wav, model, sampling_rate=VAD_SR, threshold=vad_threshold)

    for i in range(len(t)):
        t[i]["start"] = max(0, t[i]["start"] - 3200)  # 0.2s head
        t[i]["end"] = min(wav.shape[0] - 16, t[i]["end"] + 20800)  # 1.3s tail
        if i > 0 and t[i]["start"] < t[i - 1]["end"]:
            t[i]["start"] = t[i - 1]["end"]  # Remove overlap

    u = [[]]
    for i in range(len(t)):
        if i > 0 and t[i]["start"] > t[i - 1]["end"] + (chunk_threshold * VAD_SR):
            u.append([])
        u[-1].append(t[i])

    # 1. Collect chunk audio tensors first (using raw integer sample indices)
    chunk_audio_tensors = []
    for i in range(len(u)):
        chunk_audio_tensors.append(collect_chunks(u[i], wav))

    os.remove("vad_chunks/silero_temp.wav")

    # 2. Now convert sample indices in u to float seconds for SRT time offset calculation
    for i in range(len(u)):
        time_sec = 0.0
        offset = 0.0
        for j in range(len(u[i])):
            u[i][j]["start"] /= VAD_SR
            u[i][j]["end"] /= VAD_SR
            u[i][j]["chunk_start"] = time_sec
            time_sec += u[i][j]["end"] - u[i][j]["start"]
            u[i][j]["chunk_end"] = time_sec
            if j == 0:
                offset += u[i][j]["start"]
            else:
                offset += u[i][j]["start"] - u[i][j - 1]["end"]
            u[i][j]["offset"] = offset

    subs = []
    segment_info = []
    sub_index = 1
    suppress_low = [
        "Thank you", "Thanks for", "ike and ", "Bye.", "Bye!", "Bye bye!", "lease sub", "The end.", "視聴",
    ]
    suppress_high = [
        "ubscribe", "my channel", "the channel", "our channel", "ollow me on", "for watching",
        "hank you for watching", "for your viewing", "r viewing", "Amara", "next video", "full video",
        "ranslation by", "ranslated by", "ee you next week", "ご視聴", "視聴ありがとうございました",
    ]

    for i in tqdm(range(len(u))):
        for x in range(max_attempts):
            chunk_audio_np = chunk_audio_tensors[i].numpy()
            result = model_asr.transcribe(
                chunk_audio_np,
                **transcription_options,
                **decoding_options,
            )

            if len(result["segments"]) == 0:
                break
            elif result["segments"][-1]["end"] < u[i][-1]["chunk_end"] + 10.0:
                break
            elif x+1 < max_attempts:
                print("Retrying chunk", i)

        for r in result["segments"]:
            if r["start"] > u[i][-1]["chunk_end"]:
                continue
            for s in suppress_low:
                if s in r["text"]:
                    r["avg_logprob"] -= 0.15
            for s in suppress_high:
                if s in r["text"]:
                    r["avg_logprob"] -= 0.35

            del r["tokens"]
            segment_info.append(r)

            if (
                r["avg_logprob"] < -1.0
                or r["no_speech_prob"] > 0.9
                or r["compression_ratio"] > 6
            ):
                continue

            start = r["start"] + u[i][0]["offset"]
            for j in range(len(u[i])):
                if (
                    r["start"] >= u[i][j]["chunk_start"]
                    and r["start"] <= u[i][j]["chunk_end"]
                ):
                    start = r["start"] + u[i][j]["offset"]
                    break

            if len(subs) > 0:
                last_end = datetime.timedelta.total_seconds(subs[-1].end)
                if last_end > start:
                    subs[-1].end = datetime.timedelta(seconds=start)

            end = u[i][-1]["end"] + 0.5
            for j in range(len(u[i])):
                if r["end"] >= u[i][j]["chunk_start"] and r["end"] <= u[i][j]["chunk_end"]:
                    end = r["end"] + u[i][j]["offset"]
                    break

            subs.append(
                srt.Subtitle(
                    index=sub_index,
                    start=datetime.timedelta(seconds=start),
                    end=datetime.timedelta(seconds=end),
                    content=r["text"].strip(),
                )
            )
            sub_index += 1

    with open("segment_info.json", "w", encoding="utf8") as f:
        json.dump(segment_info, f, indent=4)

    garbage_list = [
        "a", "aa", "ah", "ahh", "ha", "haa", "hah", "haha", "hahaha", "mmm",
        "mm", "m", "h", "o", "mh", "mmh", "hm", "hmm", "huh", "oh",
    ]
    need_context_lines = [
        "feelsgod", "godbye", "godnight", "thankyou",
    ]
    clean_subs = list()
    last_line_garbage = False
    for i in range(len(subs)):
        c = subs[i].content
        c = (
            c.replace(".", "")
            .replace(",", "")
            .replace(":", "")
            .replace(";", "")
            .replace("!", "")
            .replace("?", "")
            .replace("-", " ")
            .replace("  ", " ")
            .replace("  ", " ")
            .replace("  ", " ")
            .lower()
            .replace("that feels", "feels")
            .replace("it feels", "feels")
            .replace("feels good", "feelsgood")
            .replace("good bye", "goodbye")
            .replace("good night", "goodnight")
            .replace("thank you", "thankyou")
            .replace("aaaaaa", "a")
            .replace("aaaa", "a")
            .replace("aa", "a")
            .replace("aa", "a")
            .replace("mmmmmm", "m")
            .replace("mmmm", "m")
            .replace("mm", "m")
            .replace("mm", "m")
            .replace("hhhhhh", "h")
            .replace("hhhh", "h")
            .replace("hh", "h")
            .replace("hh", "h")
            .replace("oooooo", "o")
            .replace("oooo", "o")
            .replace("oo", "o")
            .replace("oo", "o")
        )
        is_garbage = True
        for w in c.split(" "):
            if w.strip() == "":
                continue
            if w.strip() in garbage_list:
                continue
            elif w.strip() in need_context_lines and last_line_garbage:
                continue
            else:
                is_garbage = False
                break
        if not is_garbage:
            clean_subs.append(subs[i])
        last_line_garbage = is_garbage

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf8") as f:
        f.write(srt.compose(clean_subs))
        f.flush()
        os.fsync(f.fileno())
    print("\nDone! Subs written to", out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ASR (Whisper) on audio file.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to input audio file (.wav)")
    parser.add_argument("--out_dir", type=str, default="output", help="Output directory for SRT")
    parser.add_argument("--out_name", type=str, default=None, help="Output name for SRT (excludes extension)")
    args = parser.parse_args()

    try:
        run(args.audio_path, out_dir=args.out_dir, out_name=args.out_name)
        os._exit(0)
    except Exception as e:
        print(f"Error processing {args.audio_path}: {e}")
        os._exit(1)
