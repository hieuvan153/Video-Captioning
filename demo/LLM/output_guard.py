"""Guardrail chong dong output suy bien cua LLM refine.

Failure mode ghi nhan trong docs/eval/error_analysis_v0.md: adapter thinh
thoang sinh 1 dong lap 1 n-gram hang tram lan ("Thằng nhóc, thằng nhóc, ..."),
mot dong nhu vay du pha precision/BLEU ca phim. Module thuan Python (khong
torch) de test local; refine_llm goi is_degenerate_line tren tung dong va
fallback ve dong rough khi phat hien.
"""
from collections import Counter


def is_degenerate_line(line: str, min_repeats: int = 8,
                       max_chars: int = 600) -> bool:
    """True neu dong la output suy bien (lap n-gram ap dao hoac dai bat thuong).

    Lap ngan kieu thoai that ("Không, không, không!") duoc giu: chi xet dong
    >= 20 tu, va 1-3-gram pho bien nhat phai vua lap >= min_repeats lan vua
    chiem > 50% so tu cua dong.
    """
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
