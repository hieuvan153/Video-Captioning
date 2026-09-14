"""Giao thuc BLEU cua Bang 4.7 (ban goc: /data/ndloc_bk/ntVan/ASR/calculate_bleu.py cua anh ntVan,
ngoai git). Chep vao repo de giao thuc duoc quan ly phien ban. Khac ban goc: chuan hoa NFC khi doc
(SRT tu nguon khac nhau tron NFC/NFD lam BLEU cua cung mot cau tu 100 xuong 4,5) va clean_custom
khong coi dong mot chu cai ("Ừ.") la chu in hoa tren man hinh.
"""
import datetime
import itertools
import re
import unicodedata

import sacrebleu
import srt


def get_subs_from_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8") as f:
        content = unicodedata.normalize("NFC", f.read())
    try:
        return list(srt.parse(content))
    except Exception:
        # Fallback to parsing lines and creating mock subtitle objects
        lines = content.strip().split('\n')
        subs = []
        text_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.isdigit():
                continue
            if '-->' in line:
                continue
            text_lines.append(line)
        for idx, text in enumerate(text_lines):
            subs.append(srt.Subtitle(
                index=idx,
                start=datetime.timedelta(seconds=idx),
                end=datetime.timedelta(seconds=idx+1),
                content=text
            ))
        return subs

def get_text_from_srt(srt_path):
    # Returns raw text string from SRT file, separated by newlines
    subs = get_subs_from_srt(srt_path)
    return "\n".join(sub.content for sub in subs)

def clean_no_punc(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_custom(text):
    # 1. Remove bracketed content like [Georgie] or (tiếng cười)
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    text = re.sub(r'\([^)]*\)', ' ', text)
    
    # 2. Split into lines and filter out all-caps lines like NHÀ THỜ BÁP-TÍT MEDFORD
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Chi xet tu co >= 2 chu cai: "Ừ.", "À…" la loi thoai, khong phai chu in hoa tren man hinh
        words = [w for w in re.split(r'\s+', stripped) if sum(c.isalpha() for c in w) >= 2]
        # If the line consists only of uppercase words, discard it
        if words and all(w.isupper() for w in words):
            continue
        filtered_lines.append(line)
        
    text = '\n'.join(filtered_lines)
    
    # 3. Only lowercase, keeping all punctuation
    text = text.lower()
    
    # Clean up double/multiple spaces but preserve spacing and punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_optimized_reference_text(ref_subs, pred_text, clean_func):
    if not ref_subs:
        return ""
    # 1. Group consecutive subtitles with identical start time
    groups = []
    i = 0
    n = len(ref_subs)
    while i < n:
        j = i + 1
        while j < n and ref_subs[j].start == ref_subs[i].start:
            j += 1
        groups.append(ref_subs[i:j])
        i = j

    # Generate permutations for each group
    group_perms = []
    for g in groups:
        perms = list(itertools.permutations(g))
        group_perms.append(perms)

    active_indices = [0] * len(groups)

    def construct_text(indices):
        parts = []
        for idx, perm_list in zip(indices, group_perms):
            chosen_perm = perm_list[idx]
            for sub in chosen_perm:
                parts.append(sub.content)
        raw_joined = "\n".join(parts)
        return clean_func(raw_joined)

    # Calculate search space size
    total_combinations = 1
    for perms in group_perms:
        total_combinations *= len(perms)

    if total_combinations == 1:
        # No duplicate timestamps, return standard cleaned text
        return construct_text(active_indices)

    if total_combinations <= 64:
        # Exact search for maximum BLEU score
        best_score = -1.0
        best_indices = active_indices
        ranges = [range(len(perms)) for perms in group_perms]
        for comb in itertools.product(*ranges):
            text = construct_text(comb)
            score = sacrebleu.sentence_bleu(pred_text, [text]).score
            if score > best_score:
                best_score = score
                best_indices = comb
        active_indices = list(best_indices)
    else:
        # Greedy search: 2 passes to optimize groups one-by-one
        for pass_num in range(2):
            for g_idx in range(len(groups)):
                if len(group_perms[g_idx]) <= 1:
                    continue
                best_g_score = -1.0
                best_g_idx = 0
                for p_idx in range(len(group_perms[g_idx])):
                    temp_indices = list(active_indices)
                    temp_indices[g_idx] = p_idx
                    text = construct_text(temp_indices)
                    score = sacrebleu.sentence_bleu(pred_text, [text]).score
                    if score > best_g_score:
                        best_g_score = score
                        best_g_idx = p_idx
                active_indices[g_idx] = best_g_idx

    return construct_text(active_indices)
