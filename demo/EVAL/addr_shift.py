"""Dem xem ban dich cua NGUOI co doi thanh ghi xung ho trong cung mot canh khong.

Neu ty le canh chua CA hai thanh ghi (tho: xung ho ngoi hai; lich su: anh/em/chi/ong/ba)
la nho, thi tin hieu giong dieu khong co gi de ban -> huong 2 (audio) khong dang
dau tu du cong G1 co dat.

CLI: van_env/bin/python demo/EVAL/addr_shift.py --ref vi.srt --captions caps.json [--window 6] [--legacy_tho]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import srt  # noqa: E402

from EVAL.pronoun_lexicon import extract_pronouns  # noqa: E402

# THO cũ (có nó) để đối chứng
THO_OLD = {"mày", "tao", "nó", "hắn", "thằng", "con bé", "lũ", "bọn nó", "tụi nó", "gã", "ả", "lão"}
LICH_SU = {"anh", "em", "chị", "ông", "bà", "cô", "chú", "bác", "ngài", "quý vị", "cậu", "thầy"}

# Danh sách từ không được phép đứng ngay trước "mày" trong cụm "X mày"
MAY_EXCLUDE_PREFIXES = {"lông", "hàng", "chân", "kẻ", "nhổ", "cạo"}


def has_tho_new(text: str) -> bool:
    """Phát hiện thanh ghi thô (ngôi hai) bằng regex, loại giả dương.

    Giữ lại: tao, mày, chúng mày, tụi mày, bọn mày, chúng tao, tụi tao, bọn tao, mày tao.
    Loại: mi (đồng âm với "lông mi"), bay (đồng âm với động từ "bay"); trên phim này
    chúng đóng góp 2 giả dương / 0 đúng dương (cue 761 "bay hơi", 1295 "bay lên").
    Loại: "mày" khi nó là "lông mày", "hàng mày", v.v.
    """
    # Kiếm các từ (bỏ mi, bay)
    if re.search(r'\btao\b', text, re.IGNORECASE):
        return True
    if re.search(r'\b(?:chúng|tụi|bọn)\s+(?:mày|tao)\b', text, re.IGNORECASE):
        return True
    if re.search(r'\bmày\s+tao\b', text, re.IGNORECASE):
        return True

    # Phát hiện "mày" nhưng loại giả dương (lông mày, v.v.)
    for match in re.finditer(r'\bmày\b', text, re.IGNORECASE):
        start = match.start()
        # Tìm từ đứng ngay trước
        if start > 0:
            # Lùi lại để tìm khoảng trắng trước đó
            before_text = text[:start].rstrip()
            if before_text:
                # Tách từ cuối cùng
                words = re.findall(r'\w+', before_text)
                if words:
                    last_word = words[-1].lower()
                    if last_word not in MAY_EXCLUDE_PREFIXES:
                        return True
                else:
                    return True
            else:
                # "mày" ở đầu text sau khoảng trắng
                return True
        else:
            # "mày" ở ĐẦU chuỗi (start == 0) → không có từ trước → đúng là đại từ
            return True

    return False


def has_lich_su(text: str) -> bool:
    """Phát hiện thanh ghi lịch sự bằng extract_pronouns."""
    terms = set(extract_pronouns(re.sub(r"\s+", " ", text)))
    return bool(terms & LICH_SU)


def analyze_by_scene(subs: list, scenes: list, use_legacy_tho: bool) -> dict:
    """Phân tích theo cảnh."""
    n_ca_hai = n_chi_tho = n_chi_lich_su = n_khong_co = 0
    n_bat_ky = 0

    # Cho nhóm "cả hai" và nhóm "chỉ một thanh ghi"
    cua_hai_cues = []
    cua_hai_seconds = []
    chi_mot_cues = []
    chi_mot_seconds = []

    # Dùng cho việc chia nửa
    all_scene_cues = []

    for sc in scenes:
        st, en = sc.get("start_time"), sc.get("end_time")
        if st is None or en is None:
            continue

        text = " ".join(s.content for s in subs
                        if s.start.total_seconds() < float(en)
                        and s.end.total_seconds() > float(st))
        text_normalized = re.sub(r"\s+", " ", text)

        # Đếm số dòng và thời lượng trong cảnh
        subs_in_scene = [s for s in subs
                         if s.start.total_seconds() < float(en)
                         and s.end.total_seconds() > float(st)]
        n_cues = len(subs_in_scene)
        duration = float(en) - float(st) if n_cues > 0 else 0

        # Phát hiện THO và LICH_SU
        if use_legacy_tho:
            has_tho = bool(set(extract_pronouns(text_normalized)) & THO_OLD)
        else:
            has_tho = has_tho_new(text)

        has_any_pronoun = bool(set(extract_pronouns(text_normalized)))
        has_lich_su_val = has_lich_su(text)

        if has_any_pronoun:
            n_bat_ky += 1

        all_scene_cues.append(n_cues)

        if has_tho and has_lich_su_val:
            n_ca_hai += 1
            cua_hai_cues.append(n_cues)
            cua_hai_seconds.append(duration)
        elif has_tho and not has_lich_su_val:
            n_chi_tho += 1
            chi_mot_cues.append(n_cues)
            chi_mot_seconds.append(duration)
        elif has_lich_su_val and not has_tho:
            n_chi_lich_su += 1
            chi_mot_cues.append(n_cues)
            chi_mot_seconds.append(duration)
        elif not has_tho and not has_lich_su_val:
            n_khong_co += 1

    n_co_tu = n_ca_hai + n_chi_tho + n_chi_lich_su
    total_scenes = n_co_tu + n_khong_co

    # Tính thống kê cho nhóm "chỉ một thanh ghi"
    chi_mot_stats = None
    if chi_mot_cues:
        chi_mot_stats = {
            'avg_cues': statistics.mean(chi_mot_cues),
            'avg_seconds': statistics.mean(chi_mot_seconds),
        }

    # Chia nửa cảnh theo trung vị số cue
    if all_scene_cues:
        median_cues = statistics.median(all_scene_cues)
        ca_hai_short = sum(1 for cues in cua_hai_cues if cues <= median_cues)
        ca_hai_long = sum(1 for cues in cua_hai_cues if cues > median_cues)
        co_tu_short = sum(1 for cues in all_scene_cues if cues <= median_cues and cues > 0)
        co_tu_long = sum(1 for cues in all_scene_cues if cues > median_cues)

        pct_short = 100.0 * ca_hai_short / co_tu_short if co_tu_short > 0 else 0
        pct_long = 100.0 * ca_hai_long / co_tu_long if co_tu_long > 0 else 0
    else:
        ca_hai_short = ca_hai_long = co_tu_short = co_tu_long = 0
        pct_short = pct_long = 0

    return {
        'n_bat_ky': n_bat_ky,
        'n_co_tu': n_co_tu,
        'n_ca_hai': n_ca_hai,
        'n_chi_tho': n_chi_tho,
        'n_chi_lich_su': n_chi_lich_su,
        'n_khong_co': n_khong_co,
        'total': total_scenes,
        'cua_hai_cues': cua_hai_cues,
        'cua_hai_seconds': cua_hai_seconds,
        'chi_mot_stats': chi_mot_stats,
        'ca_hai_short': ca_hai_short,
        'ca_hai_long': ca_hai_long,
        'co_tu_short': co_tu_short,
        'co_tu_long': co_tu_long,
        'pct_short': pct_short,
        'pct_long': pct_long,
    }


def analyze_by_window(subs: list, window_size: int, use_legacy_tho: bool) -> dict:
    """Phân tích theo cửa sổ liên tiếp."""
    if window_size <= 0 or window_size > len(subs):
        return None

    n_windows_ca_hai = 0
    n_windows_co_tu = 0

    for i in range(len(subs) - window_size + 1):
        window = subs[i:i + window_size]
        text = " ".join(s.content for s in window)
        text_normalized = re.sub(r"\s+", " ", text)

        # Phát hiện THO và LICH_SU
        if use_legacy_tho:
            has_tho = bool(set(extract_pronouns(text_normalized)) & THO_OLD)
        else:
            has_tho = has_tho_new(text)

        has_lich_su_val = has_lich_su(text)

        if has_tho or has_lich_su_val:
            n_windows_co_tu += 1

        if has_tho and has_lich_su_val:
            n_windows_ca_hai += 1

    return {
        'n_windows_co_tu': n_windows_co_tu,
        'n_windows_ca_hai': n_windows_ca_hai,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--window", type=int, default=6, help="Kích thước cửa sổ cue (0 = tắt)")
    ap.add_argument("--legacy_tho", action="store_true", help="Dùng tập THO cũ (có nó, hắn, v.v.) để đối chứng")
    a = ap.parse_args()

    with open(a.ref, encoding="utf-8-sig", errors="replace") as f:
        subs = list(srt.parse(f.read()))
    with open(a.captions, encoding="utf-8") as f:
        scenes = json.load(f)

    tho_name = "THO (cũ, có 'nó')" if a.legacy_tho else "THO (mới, ngôi hai, loại lông mày)"

    # === Phân tích theo cảnh ===
    result_scene = analyze_by_scene(subs, scenes, a.legacy_tho)

    print(f"=== PHÂN TÍCH THEO CẢNH ({tho_name}) ===")
    print(f"Cảnh có từ xưng hô bất kỳ (THO hoặc LICH_SU): {result_scene['n_bat_ky']}")
    print(f"Cảnh có từ THO hoặc LICH_SU: {result_scene['n_co_tu']}")
    print(f"Cảnh chứa CẢ hai thanh ghi: {result_scene['n_ca_hai']}")
    if result_scene['n_co_tu'] > 0:
        pct_of_co_tu = 100.0 * result_scene['n_ca_hai'] / result_scene['n_co_tu']
        pct_of_bat_ky = 100.0 * result_scene['n_ca_hai'] / result_scene['n_bat_ky'] if result_scene['n_bat_ky'] > 0 else 0
        print(f"  → {result_scene['n_ca_hai']}/{result_scene['n_co_tu']} = {pct_of_co_tu:.1f}% (của có THO/LICH_SU)")
        print(f"  → {result_scene['n_ca_hai']}/{result_scene['n_bat_ky']} = {pct_of_bat_ky:.1f}% (của bất kỳ từ xưng hô)")
    print()

    # === Kiểm soát độ dài ===
    print(f"=== KIỂM SOÁT ĐỘ DÀI ===")
    if result_scene['cua_hai_cues']:
        avg_cues_ca_hai = statistics.mean(result_scene['cua_hai_cues'])
        avg_secs_ca_hai = statistics.mean(result_scene['cua_hai_seconds'])
        print(f"Nhóm 'cả hai': trung bình {avg_cues_ca_hai:.1f} cue, {avg_secs_ca_hai:.1f} giây")

    if result_scene['chi_mot_stats']:
        stats = result_scene['chi_mot_stats']
        print(f"Nhóm 'chỉ một thanh ghi': trung bình {stats['avg_cues']:.1f} cue, {stats['avg_seconds']:.1f} giây")

    # Chia nửa cảnh
    if result_scene['co_tu_short'] > 0 or result_scene['co_tu_long'] > 0:
        print(f"Nửa cảnh ngắn (≤ trung vị {result_scene['co_tu_short']} cảnh): {result_scene['ca_hai_short']} cả hai ({result_scene['pct_short']:.1f}%)")
        print(f"Nửa cảnh dài (> trung vị {result_scene['co_tu_long']} cảnh): {result_scene['ca_hai_long']} cả hai ({result_scene['pct_long']:.1f}%)")
    print()

    # === Phân tích theo cửa sổ (nếu có) ===
    if a.window > 0:
        result_window = analyze_by_window(subs, a.window, a.legacy_tho)
        print(f"=== PHÂN TÍCH THEO CỬA SỔ {a.window} CUE ({tho_name}) ===")
        print(f"Cửa sổ có từ THO hoặc LICH_SU: {result_window['n_windows_co_tu']}")
        print(f"Cửa sổ chứa CẢ hai thanh ghi: {result_window['n_windows_ca_hai']}")
        if result_window['n_windows_co_tu'] > 0:
            pct_window = 100.0 * result_window['n_windows_ca_hai'] / result_window['n_windows_co_tu']
            print(f"  → {result_window['n_windows_ca_hai']}/{result_window['n_windows_co_tu']} = {pct_window:.1f}%")
        print()


def _selftest() -> None:
    """Kiểm tra hàm has_tho_new với 5 trường hợp."""
    tests = [
        ("Nó giống như một tổ hợp chất dễ bay hơi", False),  # "bay" = động từ
        ("Anh ấy gần như muốn bay lên... - Cooper. Cooper.", False),  # "bay" = động từ
        ("Mày vừa nói gì đó?", True),  # "mày" ở đầu, đại từ
        ("Anh ta còn tẩy lông mày.", False),  # "lông mày" = cụm từ
        ("Tao với mày ra ngoài nói chuyện", True),  # cả "tao" và "mày"
    ]
    print("=== SELF-TEST ===")
    for text, expected in tests:
        result = has_tho_new(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} has_tho_new('{text}') = {result} (expected {expected})")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
