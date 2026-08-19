"""Guardrail chong dong output suy bien cua LLM refine.

Failure mode ghi nhan trong docs/eval/error_analysis_v0.md: adapter thinh
thoang sinh 1 dong lap 1 n-gram hang tram lan ("Thằng nhóc, thằng nhóc, ..."),
mot dong nhu vay du pha precision/BLEU ca phim. Module thuan Python (khong
torch) de test local; refine_llm goi is_degenerate_line tren tung dong va
fallback ve dong rough khi phat hien.
"""
import re
from collections import Counter

# Ky tu ngoai he chu Latin (Hy Lap, Cyrillic, Hebrew, A Rap, Devanagari+
# Bengali, Thai, Hangul Jamo, kana, CJK ExtA+chinh, Hangul, fullwidth):
# khong bao gio xuat hien trong phu de tieng Viet hop le; quan sat adapter
# leak token Bengali o movie_312 (docs/eval/degeneration_e5.md).
_NON_LATIN = re.compile(
    r"[\u0370-\u03ff\u0400-\u04ff\u0590-\u05ff\u0600-\u06ff"
    r"\u0900-\u09ff\u0e00-\u0e7f\u1100-\u11ff\u3040-\u30ff"
    r"\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff00-\uffef]"
)


def is_degenerate_line(line: str, min_repeats: int = 8,
                       max_chars: int = 600) -> bool:
    """True neu dong la output suy bien (lap n-gram ap dao hoac dai bat thuong).

    Lap ngan kieu thoai that ("Không, không, không!") duoc giu: chi xet dong
    >= 20 tu, va 1-3-gram pho bien nhat phai vua lap >= min_repeats lan vua
    chiem > 50% so tu cua dong.

    Ngoai ra: dong khong co ky tu chu/so nao (vd "-", "- -", ".") va dong
    chua ky tu ngoai he Latin deu la suy bien — hai failure mode quan sat
    khi scene sap nguyen khoi (docs/eval/degeneration_e5.md).
    """
    if not any(c.isalnum() for c in line):
        return True
    if _NON_LATIN.search(line):
        return True
    if len(line) > max_chars:
        return True
    words = line.split()
    if len(words) < 20:
        return False
    for n in (1, 2, 3):
        grams = Counter(
            tuple(words[i:i + n]) for i in range(len(words) - n + 1)
        )
        top = max(grams.values())
        if top >= min_repeats and top * n > 0.5 * len(words):
            return True
    return False
