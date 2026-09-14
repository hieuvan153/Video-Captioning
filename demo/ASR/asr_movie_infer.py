import sys
import torch
torch.set_num_threads(4)
import whisper
import os
import ffmpeg
import srt
import soundfile as sf
from tqdm import tqdm
import datetime
import json
import tempfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_post import VAD_SR, annotate_seconds, decode_env, group_regions, map_segment, segment_info_path  # noqa: E402
_TEMPS, _BEAM = decode_env(os.environ)

print("Running Whisper...")
model_asr = whisper.load_model(os.path.join(ROOT_DIR, "model/ASR/whisper-medium-13-openai.pt"))

vad_threshold = 0.2
chunk_threshold = 3.0  # lang dai hon so giay nay thi tach chunk moi

# Mac dinh greedy T=0; ASR_BEAM / ASR_TEMPS chi dat cho arm thi nghiem a2.
transcribe_options = {
    "task": "transcribe",
    "language": "english",
    "temperature": _TEMPS,
    "beam_size": _BEAM,
    "best_of": None,
    "patience": None,
    "length_penalty": None,
    "prefix": "",
    "suppress_tokens": "-1",
    "suppress_blank": True,
    "without_timestamps": False,
    "max_initial_timestamp": 1.0,
    "fp16": True,
    "verbose": False,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.9,
    "condition_on_previous_text": False,
    "initial_prompt": "",  # "" khac None: cua so dau moi chunk nhan prompt " "
    "word_timestamps": True,
    "clip_timestamps": "0",
    # None (14/09, Ode to Joy, cung dieu kien, chay lap giong het tung byte): voi 2.0 whisper nhay tron 30 s khi
    # tu cuoi nam trong 2 s cuoi cua so va bo cau dang do. WER 16,61 -> 15,92, tu mat o duong noi 226 -> 156,
    # Bang 4.7 36,04 -> 36,30 BLEU. (Da thu va BO: alignment_heads cua medium.en WER +0,26; initial_prompt=None +0,31.)
    "hallucination_silence_threshold": None,
}

def run(audio_path, out_dir = None, out_name = None):
    name = os.path.splitext(os.path.basename(audio_path))[0] if out_name is None else out_name
    srt_name = name + ".(Tiếng Anh).srt"
    out_path = "output/" + srt_name if out_dir is None else os.path.join(out_dir, srt_name)

    # WAV tam o thu muc rieng cua lan chay nay. Truoc day dung chung mot thu muc trong CWD: lan chay
    # sau xoa file cua lan truoc -> lan truoc nhan dang nham audio ma khong bao loi.
    out_dir_abs = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir_abs, exist_ok=True)
    print("Encoding audio...")
    with tempfile.TemporaryDirectory(dir=out_dir_abs) as tmp:
        tmp_wav = os.path.join(tmp, "silero_temp.wav")
        ffmpeg.input(audio_path).output(
            tmp_wav,
            ar="16000",
            ac="1",
            acodec="pcm_s16le",
            map_metadata="-1",
            fflags="+bitexact",
        ).overwrite_output().run(quiet=True)
        # Doc bang soundfile thay read_audio cua silero de khong phu thuoc torchaudio.
        wav, sr = sf.read(tmp_wav, dtype='float32')
    if sr != VAD_SR:
        raise ValueError(f"Sampling rate of converted audio is {sr}, expected {VAD_SR}")
    wav = torch.from_numpy(wav)

    print("Running VAD...")
    model, utils = torch.hub.load(
        repo_or_dir=os.path.join(ROOT_DIR, "model/ASR/silero-vad"),
        model="silero_vad",
        onnx=True,
        source="local"
    )
    get_speech_timestamps, _, _, _, collect_chunks = utils
    t = get_speech_timestamps(wav, model, sampling_rate=VAD_SR, threshold=vad_threshold)

    # Dem 0,2 s dau / 1,3 s duoi (don vi mau) roi bo phan chong lan.
    u = group_regions(t, wav.shape[0], chunk_threshold)

    # Cat audio theo chi so mau truoc, sau do moi doi u sang giay.
    chunk_audio_tensors = [collect_chunks(u[i], wav) for i in range(len(u))]

    # chunk_start/chunk_end: vi tri trong audio da ghep; offset: cong vao de ve thoi gian goc.
    annotate_seconds(u)

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
        result = model_asr.transcribe(chunk_audio_tensors[i].numpy(), **transcribe_options)

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
            r["chunk"] = i  # seek ve 0 khong du de biet segment thuoc chunk nao (chunk rong/ngan)
            segment_info.append(r)

            if (
                r["avg_logprob"] < -1.0
                or r["no_speech_prob"] > 0.9
                or r["compression_ratio"] > 6
            ):
                continue

            start, end = map_segment(u[i], r["start"], r["end"])

            if len(subs) > 0:
                last_end = datetime.timedelta.total_seconds(subs[-1].end)
                if last_end > start:
                    subs[-1].end = datetime.timedelta(seconds=start)


            subs.append(
                srt.Subtitle(
                    index=sub_index,
                    start=datetime.timedelta(seconds=start),
                    end=datetime.timedelta(seconds=end),
                    content=r["text"].strip(),
                )
            )
            sub_index += 1

    with open(segment_info_path(out_path), "w", encoding="utf8") as f:
        json.dump(segment_info, f, indent=4)

    # Bo cue chi gom tieng dem (oh, hmm...); "thankyou"/"godbye"... chi bo khi dong truoc cung la rac.
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
