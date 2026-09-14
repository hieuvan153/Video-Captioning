"""Hau xu ly cua tang LLM (khong nap Gemma) de test duoc: gioi dong, khoa do dai, chia chunk, chan dau ra suy bien."""
import difflib
import os
import re
from collections import Counter

import srt

# Diem mot cap (dong LLM, cue) = ty le giong - nguong; cap duoi nguong khong duoc gan. Quet 09/09 chon 0.20.
ALIGN_MIN_RATIO = 0.20
# Nguong rieng cho dong tho ngan (< SHORT_WORDS tu), CHI bat cho adapter v7 (--system_prompt): "Không." vs "- Tôi. - Tôi."
# dat 0,21 >= 0,20 (24 cue rac). Bat cho adapter mac dinh thi 31 cue ngan ve ban tho: neo cue -0,06 BLEU so voi pm0.90 (14/09).
ALIGN_MIN_RATIO_SHORT = 0.50
SHORT_WORDS = 4
# Tran cue moi chunk: dau ra ~10 token/cue phai lot max_new_tokens 1024 (canh 91 cue cua Ode to Joy ~897 token); unsloth khong
# cat prompt Gemma-3 dai hon max_seq_length. Tran 30 cat canh lam mat ngu canh xung ho: -0,50 BLEU 4.7 / -0,025 PronF1 (14/09).
MAX_CHUNK_CUES = 100
# Khoa do dai: dong tinh chinh ngan hon LEN_LOCK_RATIO x ban tho (so tu) chi duoc chuyen phan sua dai tu.
LEN_LOCK_RATIO = 0.90
# Dong LLM giong CAP cue ke nhau hon cue duoc gan it nhat chung nay -> la dong gop hai cue.
MERGE_MARGIN = 0.15

# Dai tu / tu xung ho co dau (dang khong dau nhu "ba", "con", "may", "no" la tu thuong -> khong dua vao).
_PRON_TOK = set("anh chị em ông bà cô cậu mày tao tôi mình ta bạn hắn nó y họ ấy "
                "chúng tụi con cháu chú bác dì mợ thím ngài nàng chàng".split())
_NUM = re.compile(r"^(\d+)[.)]\s+")


def align_lines(rough, out_lines, tau=ALIGN_MIN_RATIO, short_tau=None):
    """Gioi m dong LLM vao n cue theo THU TU bang QHD: bo qua cue (giu ban tho), BO dong LLM, hoac gan.
    short_tau: nguong cho dong tho ngan (None = dung tau). Tra ve list dai len(rough): out_lines[j] hoac None."""
    n, m = len(rough), len(out_lines)
    NEG = float("-inf")
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    taus = [short_tau if short_tau is not None and len(r.split()) < SHORT_WORDS else tau for r in rough]
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
                    sc = dp[i][j] + difflib.SequenceMatcher(None, rough[i], out_lines[j]).ratio() - taus[i]
                    if sc > dp[i + 1][j + 1]:
                        dp[i + 1][j + 1], bt[i + 1][j + 1] = sc, (i, j, j)
    i, j, res = n, m, [None] * n
    while (i, j) != (0, 0):
        pi, pj, take = bt[i][j]
        if take is not None:
            res[pi] = out_lines[take]
        i, j = pi, pj
    return res


def _pron_span(ws):
    """True neu MOI tu trong doan deu la dai tu (doan rong -> True)."""
    return all(w.strip(".,!?:;…\"'?-—-()").lower() in _PRON_TOK for w in ws)


def merge_pron(rough, refined):
    """Chi chuyen cac phep sua DAI TU tu refined sang rough; phan con lai giu rough (ca chu hoa dau cau)."""
    a, b = rough.split(), refined.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, [w.lower() for w in a], [w.lower() for w in b]).get_opcodes():
        if tag != "equal" and _pron_span(a[i1:i2]) and _pron_span(b[j1:j2]):
            out.extend(b[j1:j2])
        else:
            out.extend(a[i1:i2])
    if a and out and a[0][:1].isupper() and out[0][:1].islower():
        out[0] = out[0][:1].upper() + out[0][1:]
    return " ".join(out)


def _looks_merged(v, rough, aligned, i):
    """Dong LLM chua ca noi dung mot cue ke ben ma cue do khong duoc gan dong nao."""
    base = difflib.SequenceMatcher(None, rough[i], v).ratio()
    for j in (i - 1, i + 1):
        if 0 <= j < len(rough) and aligned[j] is None and rough[j].strip():
            pair = f"{rough[j]} {rough[i]}" if j < i else f"{rough[i]} {rough[j]}"
            if difflib.SequenceMatcher(None, pair, v).ratio() > base + MERGE_MARGIN:
                return True
    return False


def lock_len(rough, aligned, ratio=LEN_LOCK_RATIO):
    """Dong tinh chinh qua ngan, hoac la dong gop hai cue, thi chi giu phan sua dai tu tren nen ban tho."""
    return [v if v is None or (len(v.split()) >= ratio * len(r.split()) and not _looks_merged(v, rough, aligned, i))
            else merge_pron(r, v)
            for i, (v, r) in enumerate(zip(aligned, rough))]


def drop_truncated_tail(lines, hit_limit):
    """Sinh cham max_new_tokens ma khong co EOS: dong cuoi la manh bi cat."""
    return lines[:-1] if hit_limit and lines else lines


def strip_numbering(lines):
    """Bo tien to '1. ', '2. ' chi khi MOI dong danh so lien tiep tu 1 (giu '3. Tháng...' trong loi thoai)."""
    m = [_NUM.match(x) for x in lines]
    if lines and all(m) and [int(x.group(1)) for x in m] == list(range(1, len(lines) + 1)):
        return [_NUM.sub("", x, count=1) for x in lines]
    return lines


def is_degenerate(lines, rough):
    """Dau ra lap: mot dong chiem hon nua chunk (tu 4 dong) VA lap nhieu hon han trong ban tho
    (ban tho da lap san, vd loi bai hat hay "Charlie!" x3, thi model chep lai la dung)."""
    if len(lines) < 4:
        return False
    top, cnt = Counter(lines).most_common(1)[0]
    return cnt * 2 > len(lines) and cnt > 2 * max(1, rough.count(top))


def cap_chunks(chunks, target):
    """Chia chunk dai hon target thanh cac manh gan bang nhau."""
    out = []
    for c in chunks:
        if not c:
            continue
        k = -(-len(c) // target)
        size = -(-len(c) // k)
        out.extend(c[i:i + size] for i in range(0, len(c), size))
    return out


def chunk_by_gap(subs, gap_s=2.0, target=20, max_cues=MAX_CHUNK_CUES):
    """Chia cue thanh chunk khong can VLM: cat o khoang lang > gap_s, gop manh lien tiep toi target cue,
    chi chia nho doan dai hon max_cues. Cung cau truc voi nhanh VLM (caption "None")."""
    if not subs:
        return []
    cuts = [0]
    for i in range(len(subs) - 1):
        if subs[i + 1].start.total_seconds() - subs[i].end.total_seconds() > gap_s:
            cuts.append(i + 1)
    cuts.append(len(subs))
    merged = []
    for pc in (list(range(cuts[k], cuts[k + 1])) for k in range(len(cuts) - 1)):
        if merged and len(merged[-1]) + len(pc) <= target:
            merged[-1].extend(pc)
        else:
            merged.append(pc)
    return [{"start_time": None, "end_time": None, "caption": "None", "indices": ix}
            for ix in cap_chunks(merged, max_cues)]


def find_best_scene(midpoint, scenes):
    """Canh chua midpoint (so khop chat); -1 neu khong co."""
    for idx, sc in enumerate(scenes):
        s, e = sc.get("start_time"), sc.get("end_time")
        if s is not None and e is not None and s <= midpoint <= e:
            return idx
    return -1


def assign_scenes(mids, scenes):
    """Canh cho moi cue; cue nam ngoai moi canh (vd truoc canh dau) gan vao canh gan nhat thay vi bi bo qua."""
    valid = [(k, s["start_time"], s["end_time"]) for k, s in enumerate(scenes)
             if s.get("start_time") is not None and s.get("end_time") is not None]
    idx, outside = [], 0
    for m in mids:
        k = find_best_scene(m, scenes)
        if k == -1 and valid:
            outside += 1
            k = min(valid, key=lambda v: min(abs(m - v[1]), abs(m - v[2])))[0]
        idx.append(k)
    return idx, outside


def write_srt(subs, path):
    """reindex=False: srt.compose mac dinh bo cue rong / dai 0 s va danh so lai."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs, reindex=False))


# Noi dung giu nguyen tu ban cu cua refine_llm (dong "mood of the scene." khong co dau cach cuoi, khac du lieu v3 mot dau
# cach; do NLL cho thay dau cach khong dang ke). THUT LE thi co: adapter duoc train voi prompt da qua clean_prompt.
BASE_SYSTEM = 'You are a professional Vietnamese subtitle editor for a movie.\n    Given three sections:\n        <Scene Context> — A description of the characters, their relationships (e.g., lovers, enemies, boss/employee), and the mood of the scene.\n        <English Dialogue> — original English lines.\n        <Rough Vietnamese Translation> — rough Vietnamese translation with possible tone or pronoun issues.\n    Use the English dialogue only to understand speaker context.\n    Fix the Vietnamese translation so that pronouns, tone, and formality are natural and consistent with the context.\n    Keep meaning and structure unchanged.\n    Output only the corrected Vietnamese translation, line by line.   '


def clean_prompt(prompt):
    """Bo thut le dau dong va dong trong o hai dau, giong cong thuc train (ALMA/finetune_gemma_v6.py)."""
    lines = [line.lstrip() for line in prompt.splitlines()]
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def build_system_prompt(caption, system_prompt=None):
    """--system_prompt (adapter train khong co kenh caption): dung nguyen chuoi. Mac dinh: BASE_SYSTEM + <Scene Context>,
    qua clean_prompt nhu luc train (NLL cau tra loi vang 1,0394 -> 1,0189)."""
    if system_prompt:
        return system_prompt
    return clean_prompt(f"{BASE_SYSTEM}\n<Scene Context>\n{caption}\n</Scene Context>")
