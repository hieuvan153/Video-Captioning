"""Convert tap test E5 cua anh ntVan sang thu muc eval cua minh (CPU).

Nguon: /data/ndloc_bk/ntVan/ASR/test_data_translated/<tap>.json — moi scene co
caption VLM + subtitles (ASR Whisper cua anh ay, co timing) + rough_vietsub
(NMT VinAI fine-tune, list 1:1 voi subtitles). Day chinh la input E5 that cua
Bang 4.7/4.8 trong do an — dung lai de tai lap E5 tren dung du lieu test,
cung nguon ASR, khong can chay lai ASR/scene-seg/VLM.

Ghi ra demo/output/eval_e5/<movie_id>/: en.srt, rough.srt, captions.json.
Reference tieng Viet KHONG ghi o day — thesis_score.py doc thang tu dataset
gold theo movie_id (cung video nen truc thoi gian trung nhau).

Mapping tap -> movie_id do ten tap trong plan.json (da doi chieu thu cong
2026-08-17, 10/10 khop duy nhat).
"""
from __future__ import annotations

import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from EVAL.prep_eval_dirs import _srt_block  # noqa: E402

SRC_DIR = "/data/ndloc_bk/ntVan/ASR/test_data_translated"
OUT_ROOT = os.path.join(ROOT_DIR, "output", "eval_e5")

# tap test cua do an -> movie_id trong dataset (khop qua movie_name plan.json)
E5_TEST: dict[str, str] = {
    "S01E017_Nhu_Thuật_bọc_bong_bóng_và_chăm_lo": "movie_054",
    "S02E021_Trái_tim_tan_vỡ_và_quái_vật_đất_nung": "movie_081",
    "S03E001_Đồn_trưởng_mới": "movie_312",
    "S03E002_Tủ_đựng_chổi_và_cờ_tỷ_phú_của_Satan": "movie_104",
    "S03E011_Gà_sống_gà_rán_và_thánh_hôn": "movie_090",
    "S03E015_Đồn_98": "movie_311",
    "S03E017_Adrian_Pimento": "movie_291",
    "S04E014_Đậu_má_và_sự_chấp_thuận_vô_điều_kiện_của_cơ_quan_c": "movie_124",
    "S05E015_Bá_chủ_ô_chữ": "movie_336",
    "S07E002_Bàn_quay_roulette_và_chó_biết_chơi_piano": "movie_170",
}


def convert(test_name: str, movie: str) -> dict:
    with open(os.path.join(SRC_DIR, f"{test_name}.json"), encoding="utf-8") as f:
        scenes = json.load(f)
    scenes = sorted(scenes, key=lambda s: float(s["start_time"]))
    mdir = os.path.join(OUT_ROOT, movie)
    os.makedirs(mdir, exist_ok=True)

    en_lines: list[tuple[float, float, str]] = []
    rough_lines: list[tuple[float, float, str]] = []
    captions = []
    for sc in scenes:
        subs, roughs = sc["subtitles"], sc["rough_vietsub"]
        if len(subs) != len(roughs):
            raise ValueError(
                f"{test_name} scene {sc['scene_id']}: "
                f"{len(subs)} sub vs {len(roughs)} rough")
        for s, r in zip(subs, roughs):
            en_lines.append((float(s["start"]), float(s["end"]),
                             str(s["text"] or "")))
            rough_lines.append((float(r["start"]), float(r["end"]),
                                str(r["text"] or "")))
        captions.append({"start_time": float(sc["start_time"]),
                         "end_time": float(sc["end_time"]),
                         "caption": str(sc.get("caption") or "None")})

    # Thu tu SRT phai tang dan theo thoi gian (nhu prep_eval_dirs)
    order = sorted(range(len(en_lines)), key=lambda i: en_lines[i][:2])
    for fname, lines in (("en.srt", en_lines), ("rough.srt", rough_lines)):
        blocks = [_srt_block(k + 1, *lines[i]) for k, i in enumerate(order)]
        with open(os.path.join(mdir, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(blocks))
    with open(os.path.join(mdir, "captions.json"), "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    with open(os.path.join(mdir, "source_episode.txt"), "w",
              encoding="utf-8") as f:
        f.write(test_name + "\n")
    return {"n_lines": len(en_lines), "n_scenes": len(captions),
            "n_dialogue_scenes": sum(1 for sc in scenes if sc["subtitles"])}


def main() -> None:
    total_lines = total_scenes = 0
    for test_name, movie in E5_TEST.items():
        info = convert(test_name, movie)
        total_lines += info["n_lines"]
        total_scenes += info["n_dialogue_scenes"]
        print(f"{movie} <- {test_name}: {info['n_lines']} dong, "
              f"{info['n_dialogue_scenes']}/{info['n_scenes']} scene co thoai")
    print(f"tong: {total_lines} dong, {total_scenes} scene co thoai "
          f"(do an: 1067 ASR chunks / 587 NMT scenes tren 9h)")


if __name__ == "__main__":
    main()
