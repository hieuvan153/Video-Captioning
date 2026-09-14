import os
import re
import difflib
import json
import time
import argparse
import srt
import torch

# Tat torch.compile: do dai chuoi thay doi lien tuc, bien dich chi ton thoi gian.
torch._dynamo.config.disable = True

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

from unsloth import FastLanguageModel  # phai import truoc transformers

# Diem cua mot cap (dong LLM, cue) = ty le giong - nguong; cap duoi nguong khong duoc gan
# (khong co nguong thi QHD gan bua dong lac vao cue bat ky). Quet 09/09 chon 0.20.
ALIGN_MIN_RATIO = 0.20


def align_lines(rough, out_lines, tau=ALIGN_MIN_RATIO):
    """Giong m dong LLM vao n cue theo THU TU bang QHD.

    Ba buoc chuyen: bo qua cue (cue giu ban tho), BO dong LLM (dong thua bi vut),
    hoac gan dong j cho cue i. Diem = ty le giong giua dong LLM va ban tho cua cue.

    Phai co CA HAI buoc bo: LLM tach mot cue thanh hai dong thi khong bo duoc
    dong thua, ca phan duoi chunk truot phai vinh vien (do 09/09: 30 cue mang
    dau hieu truot 1 buoc trong arm _dp cu). Co buoc bo dong thi m > n cung
    giong duoc, khong con phai vut ca chunk.

    Tra ve list dai len(rough): out_lines[j] hoac None (cue do giu ban tho).
    """
    n, m = len(rough), len(out_lines)
    NEG = float("-inf")
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == NEG:
                continue
            if j < m and dp[i][j] > dp[i][j + 1]:               # BO dong LLM j
                dp[i][j + 1], bt[i][j + 1] = dp[i][j], (i, j, None)
            if i < n:
                if dp[i][j] > dp[i + 1][j]:                    # bo qua cue i
                    dp[i + 1][j], bt[i + 1][j] = dp[i][j], (i, j, None)
                if j < m:                                      # gan dong j cho cue i
                    sc = dp[i][j] + difflib.SequenceMatcher(
                        None, rough[i], out_lines[j]).ratio() - tau
                    if sc > dp[i + 1][j + 1]:
                        dp[i + 1][j + 1], bt[i + 1][j + 1] = sc, (i, j, j)
    i, j, res = n, m, [None] * n
    while (i, j) != (0, 0):
        pi, pj, take = bt[i][j]
        if take is not None:
            res[pi] = out_lines[take]
        i, j = pi, pj
    return res


# Khoa do dai: dong tinh chinh ngan hon LEN_LOCK_RATIO x ban tho (tinh theo so tu) thi chi
# chuyen phan sua DAI TU sang ban tho. LLM viet ngan lai lam mat BLEU (BP); khoa nay dua
# Bang 4.7 Ode to Joy tu 36,82 len 37,90 BLEU, PronF1 0,833 -> 0,830 (nguong 0,88-0,95 deu tuong duong).
LEN_LOCK_RATIO = 0.90

# Dai tu / tu xung ho o muc TU, rong hon tu vung cua thuoc PronF1 de khong bam dinh thuoc do.
_PRON_TOK = set("anh chi em ong ba co cau may tao toi minh ta ban han no y ho ay "
                "chung tui con chau chu bac di mo thim ngai nang chang".split()
                + "anh ch\u1ecb em \u00f4ng b\u00e0 c\u00f4 c\u1eadu m\u00e0y tao t\u00f4i m\u00ecnh ta b\u1ea1n h\u1eafn n\u00f3 y h\u1ecd \u1ea5y "
                  "ch\u00fang t\u1ee5i con ch\u00e1u ch\u00fa b\u00e1c d\u00ec m\u1ee3 th\u00edm ng\u00e0i n\u00e0ng ch\u00e0ng".split())


def _pron_span(ws):
    """True neu MOI tu trong doan deu la dai tu (doan rong -> True)."""
    return all(w.strip(".,!?:;\u2026\"'?-\u2014-()").lower() in _PRON_TOK for w in ws)


def merge_pron(rough, refined):
    """Chi chuyen cac phep sua DAI TU tu refined sang rough; phan con lai giu rough."""
    a, b = rough.split(), refined.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, [w.lower() for w in a], [w.lower() for w in b]).get_opcodes():
        if tag != "equal" and _pron_span(a[i1:i2]) and _pron_span(b[j1:j2]):
            out.extend(b[j1:j2])
        else:
            out.extend(a[i1:i2])
    return " ".join(out)


def lock_len(rough, aligned, ratio=LEN_LOCK_RATIO):
    """Dong tinh chinh qua ngan -> chi giu phan sua dai tu, phan con lai ve ban tho."""
    return [v if v is None or len(v.split()) >= ratio * len(r.split())
            else merge_pron(r, v)
            for v, r in zip(aligned, rough)]


def _demo_align():
    assert align_lines(["a b c", "d e f"], ["a b c"]) == ["a b c", None]
    assert align_lines(["a b c", "d e f"], ["d e f"]) == [None, "d e f"]
    assert align_lines(["x", "y", "z"], ["y", "z"]) == [None, "y", "z"]
    # LLM tach cue 0 thanh 2 dong: phai VUT dong thua, khong duoc lam truot cue 1
    assert align_lines(["a b c", "d e f"], ["a b", "c", "d e f"]) == ["a b", "d e f"]
    # m > n van giong duoc (truoc day vut ca chunk)
    assert align_lines(["x", "y"], ["a", "b", "c"]) == [None, None]
    assert align_lines(["hello world", "bye"], ["zzz", "hello world"]) == ["hello world", None]
    assert "Ừ…".replace("\u2026", "...") == "Ừ..."
    # khoa do dai: dong ngan -> chi doi "Co"->"Em", giu do dai ban tho
    assert lock_len(["C\u00f4 th\u1eadt xinh \u0111\u1eb9p.", "b c d e"],
                    ["Em th\u1eadt \u0111\u1eb9p.", "b c d e"]) == ["Em th\u1eadt xinh \u0111\u1eb9p.", "b c d e"]
    assert lock_len(["a b c d"], [None]) == [None]      # khong co dong LLM -> giu None
    # doan doi KHONG phai dai tu -> giu nguyen ban tho
    assert merge_pron("T\u00f4i r\u1ea5t m\u1ec7t.", "T\u00f4i v\u00f4 c\u00f9ng m\u1ec7t.") == "T\u00f4i r\u1ea5t m\u1ec7t."
    # dai tu ghep hai tu
    assert merge_pron("C\u00f4 \u1ea5y \u0111\u1ebfn r\u1ed3i.", "Ch\u1ecb \u1ea5y \u0111\u1ebfn r\u1ed3i.") == "Ch\u1ecb \u1ea5y \u0111\u1ebfn r\u1ed3i."
    # chunk_by_gap: cat o khoang lang > gap_s, gop toi target
    import datetime as _dt
    _S = lambda i, a, b: srt.Subtitle(i, _dt.timedelta(seconds=a), _dt.timedelta(seconds=b), "x")
    _subs = [_S(1, 0, 1), _S(2, 1.2, 2), _S(3, 10, 11), _S(4, 11.2, 12), _S(5, 30, 31)]
    assert [c["indices"] for c in chunk_by_gap(_subs, 3.0, 99)] == [[0, 1, 2, 3, 4]]
    assert [c["indices"] for c in chunk_by_gap(_subs, 3.0, 2)] == [[0, 1], [2, 3], [4]]
    assert [c["indices"] for c in chunk_by_gap(_subs, 100.0, 99)] == [[0, 1, 2, 3, 4]]
    assert chunk_by_gap([], 3.0, 20) == []
    assert all(c["caption"] == "None" for c in chunk_by_gap(_subs, 3.0, 2))
    print("align_lines demo OK")


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
    parser.add_argument("--output_srt",type=str, required=True)
    parser.add_argument("--adapter_model_name", type=str, default="thevan2404/best_gemma_scene_context")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Ghi de system prompt va BO khoi <Scene Context>. Dung cho adapter "
                             "train khong co kenh caption, vi du v7: "
                             "'Rewrite to natural Vietnamese subtitle.'")
    parser.add_argument("--cache_dir", type=str, default=os.path.join(ROOT_DIR, "cache"))
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--llm_batch_size", type=int, default=8,
                        help="Number of prompts to process in one GPU batch.")
    return parser.parse_args()


def chunk_by_gap(subs, gap_s=2.0, target=20):
    """Chia cue thanh chunk khong can VLM: cat o khoang lang > gap_s, gop manh lien tiep toi target cue.
    Cung cau truc voi nhanh VLM (caption = "None"). gap 2,0 s / target 20 cho co chunk sat VLM
    tren Ode to Joy (104 chunk, trung vi 17 so voi 95 / 15). Luu y: mot doan khong co khoang
    lang nao van thanh MOT chunk dai hon target."""
    if not subs:
        return []
    cuts = [0]
    for i in range(len(subs) - 1):
        if subs[i + 1].start.total_seconds() - subs[i].end.total_seconds() > gap_s:
            cuts.append(i + 1)
    cuts.append(len(subs))
    pieces = [list(range(cuts[k], cuts[k + 1])) for k in range(len(cuts) - 1)]
    merged = []
    for pc in pieces:
        if merged and len(merged[-1]) + len(pc) <= target:
            merged[-1].extend(pc)
        else:
            merged.append(pc)
    return [{"start_time": None, "end_time": None, "caption": "None", "indices": ix}
            for ix in merged]


def find_best_scene(midpoint, scenes):
    """Strict scene matching by midpoint timestamp (no nearest neighbor fallback)."""
    for idx, sc in enumerate(scenes):
        s, e = sc.get('start_time'), sc.get('end_time')
        if s is None or e is None:
            continue
        if s <= midpoint <= e:
            return idx
    return -1


def refine_subtitles(
    en_srt_path,
    vinai_srt_path,
    vlm_json_path,
    output_srt_path,
    adapter_model_name="thevan2404/best_gemma_scene_context",
    cache_dir=None,
    max_seq_length=2048,
    max_new_tokens=1024,
    llm_batch_size=8,
    system_prompt=None,
    chunk_gap_s=2.0,
    chunk_target=20
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

        for i, sub in enumerate(en_subs):
            mid = (sub.start.total_seconds() + sub.end.total_seconds()) / 2.0
            si  = find_best_scene(mid, scenes_data)
            if si != -1:
                scenes_data[si]["indices"].append(i)

    prompts = []
    for sc in scenes_data:
        if not sc["indices"]:
            continue
        cap = sc["caption"]
        chunk = sc["indices"]
        prompts.append({
            "indices":   chunk,
            "context":   cap,
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
    n_mismatch = 0
    debug_scenes = []
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
            decoded = tokenizer.decode(outputs[i][max_len:], skip_special_tokens=True)
            # Gemma sinh U+2026, ban tho va phu de nguoi viet "..." (khong doi: -1,59 BLEU).
            decoded = decoded.replace("\u2026", "...")
            lines_out = [l.strip() for l in decoded.split("\n") if l.strip()]

            # LUON giong bang QHD, khong anh xa theo vi tri: LLM vua tach vua gop dong thi so dong
            # van bang nhau ma cac cue o giua truot 1 buoc (09/09: 30 cue).
            rough = [vinai_subs[s_idx].content for s_idx in item["indices"]]
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
                "translations": scene_translations
            })

    total = time.time() - t_total
    print(f"\nDone in {total/60:.1f} min")
    if n_mismatch:
        print(f"{n_mismatch}/{len(prompts)} chunk sai so dong -> da giong lai bang QHD", flush=True)

    for i, sub in enumerate(out_subs):
        sub.start = en_subs[i].start
        sub.end   = en_subs[i].end
        if not sub.content.strip():
            sub.content = vinai_subs[i].content

    os.makedirs(os.path.dirname(os.path.abspath(output_srt_path)), exist_ok=True)
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(out_subs))
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
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        llm_batch_size=args.llm_batch_size
    )


if __name__ == "__main__":
    main()
