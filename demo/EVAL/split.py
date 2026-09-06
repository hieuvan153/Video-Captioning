"""Split sach co dinh cho danh gia lai E0 (docs/eval/clean_split.json)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
_PATH = os.path.join(REPO, "docs", "eval", "clean_split.json")
_NAMES = os.path.join(REPO, "data", "movie_to_name_mapping.json")


def load() -> dict:
    with open(_PATH, encoding="utf-8") as f:
        s = json.load(f)
    s["holdout"] = s["test_e5"] + s["test_out"]
    return s


def episode_name(movie_id: str) -> str:
    with open(_NAMES, encoding="utf-8") as f:
        return json.load(f)[movie_id].replace(" ", "_")
