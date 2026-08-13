# Character Relationship Registry (V0) + Eval Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây eval harness (BLEU/chrF/Pronoun F1) và module V0 "Character Registry" — đồ thị quan hệ **có hướng** giữa nhân vật, trích xuất bằng Gemma từ transcript EN + caption VLM, rồi tiêm vào prompt refine của Gemma-12B — để nâng Pronoun F1 từ 0.5806.

**Architecture:** Thêm 2 module mới vào `demo/`: `EVAL/` (metric thuần Python, chạy local không cần GPU) và `CHARACTER/` (schema + prompt thuần Python, test local; runner GPU chạy trên server). Registry được build 1 lần/phim ở step 5b (Gemma base, không dùng adapter refine), lưu `{base}.registry.json`, rồi `refine_llm.py` render thành block `<Character Registry>` trong system prompt. Toàn bộ là **opt-in** (`--use_registry`) — pipeline mặc định chạy y như cũ để so sánh A/B sạch.

**Tech Stack:** Python 3.10+, pytest, sacrebleu, srt, Unsloth FastLanguageModel (Gemma-3-12B-IT 4-bit), server GPU tại `/data/ndloc_bk/ntVan`.

**Cơ sở từ survey (paper → quyết định thiết kế):**
- **PRIDE (EMNLP 2021)** — quan hệ liên nhân phải là **directed edges**: schema `Relation(from_id, to_id, vi_self, vi_listener)`, mỗi cặp nhân vật là 2 edge riêng.
- **DramaSR-LRM (2026)** — prompt trích xuất character-relationship-graph bằng LLM, yêu cầu quan hệ có hướng + evidence: chính là prompt của Task 5.
- **Hermes the Polyglot** — đưa tuổi/giới tính vào prompt dịch để chọn đại từ: field `gender`/`age_range` trong Character.
- **OmniScript Character Profile Manager** — registry là persistent memory xuyên phim, build 1 lần rồi tiêm vào mọi prompt.
- **Look-Listen-Recognise / Huh & Zisserman context-LLM** — gán speaker per-line bằng audio exemplar + LLM context: **hoãn sang V1** (xem cuối plan), vì V0 text-only validate hướng đi rẻ nhất trước (đúng roadmap V0→V1→V2 đã chốt).

## Global Constraints

- **KHÔNG đổi hành vi mặc định:** mọi thay đổi ở `run_pipeline.py`/`refine_llm.py` chỉ kích hoạt khi có `--use_registry` / `--registry_json`. Không flag → output byte-identical logic cũ.
- **KHÔNG hardcode secret:** token HF chỉ đọc từ env `HF_TOKEN`. Token cũ `hf_alBY...` đã lộ — user PHẢI tự revoke tại https://huggingface.co/settings/tokens (ngoài phạm vi code).
- **Module thuần (`demo/EVAL/*`, `demo/CHARACTER/registry_schema.py`, `demo/CHARACTER/registry_prompt.py`) KHÔNG được import torch/unsloth/transformers** — để pytest chạy được trên máy local Windows không GPU.
- **Atomic write** cho mọi file output mới: ghi `path + ".tmp"` rồi `os.replace`.
- Task 1–5, 7 (phần thuần) chạy + test local. Task 6 (GPU smoke) và Task 8 (E2E + eval A/B) chạy trên server.
- Style theo repo hiện có: argparse CLI, `print(..., flush=True)`, sys.path.append theo `ROOT_DIR`.
- Commit theo conventional commits (`feat:`, `fix:`, `test:`), không attribution footer.
- Quality gate V0: Pronoun F1 trên eval set ≥ **baseline + 0.03** (baseline paper E5 = 0.5806; đích cuối V2 ≥ 0.70).

---

### Task 1: Gỡ HF token hardcode + fix CLI chết của refine_llm.py

**Files:**
- Modify: `demo/LLM/refine_llm.py:18-20` (token), cuối file (main guard)

**Interfaces:**
- Consumes: env var `HF_TOKEN` (optional).
- Produces: `refine_llm.py` chạy được như CLI (`python refine_llm.py --en_srt ...`); không còn secret trong source.

- [x] **Step 1: Tạo branch làm việc**

```bash
git checkout -b feat/character-registry-v0
```

- [x] **Step 2: Thay block login hardcode bằng env var**

Sửa `demo/LLM/refine_llm.py` dòng 18–20, từ:

```python
# Login to Hugging Face
from huggingface_hub import login
login(token="hf_alBY[REDACTED]", add_to_git_credential=False)
```

thành:

```python
# Login to Hugging Face (token from env; needed only if the adapter repo is private)
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)
```

- [x] **Step 3: Thêm main guard cuối file**

`main()` được định nghĩa ở dòng 279 nhưng không bao giờ được gọi → CLI trong README là no-op. Thêm vào cuối file (sau `main()`):

```python
if __name__ == "__main__":
    main()
```

- [x] **Step 4: Verify không còn token trong toàn repo**

```bash
grep -rn "hf_alBY" demo/ && echo "FAIL: token still present" || echo "OK: token removed"
```

Expected: `OK: token removed`

- [x] **Step 5: Commit**

```bash
git add demo/LLM/refine_llm.py
git commit -m "fix: read HF token from env and add missing __main__ guard in refine_llm"
```

**Lưu ý cho user (ngoài code):** revoke token cũ trên huggingface.co, tạo token mới, `export HF_TOKEN=...` trên server trước khi chạy step 6 của pipeline.

---

### Task 2: Pronoun lexicon + Pronoun F1 metric

**Files:**
- Create: `demo/EVAL/__init__.py` (rỗng)
- Create: `demo/EVAL/pronoun_lexicon.py`
- Create: `demo/EVAL/pronoun_f1.py`
- Create: `tests/conftest.py`
- Test: `tests/test_pronoun_metrics.py`

**Interfaces:**
- Produces: `extract_pronouns(text: str) -> list[str]`; `parse_gold_pronouns(record: dict) -> list[str]`; `corpus_pronoun_f1(pairs: list[tuple[list[str], list[str]]]) -> PronounScores` (dataclass có `.precision .recall .f1 .tp .fp .fn`). Task 3 và Task 8 dùng đúng các tên này.

- [x] **Step 1: Cài dependency test (local)**

```bash
pip install pytest sacrebleu
```

- [x] **Step 2: Viết conftest cho pytest tìm được package trong demo/**

`tests/conftest.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
```

- [x] **Step 3: Viết failing test**

`tests/test_pronoun_metrics.py`:

```python
from EVAL.pronoun_lexicon import extract_pronouns, parse_gold_pronouns
from EVAL.pronoun_f1 import corpus_pronoun_f1


def test_extract_single_and_multiword():
    # "chúng tôi" phải match nguyên cụm, không tách thành "tôi"
    assert extract_pronouns("Chúng tôi đổi linh kiện.") == ["chúng tôi"]
    assert extract_pronouns("Ông tôi mua nó năm 1957.") == ["ông", "tôi", "nó"]


def test_extract_no_partial_word_match():
    # "bà" không được match bên trong "bàn"
    assert extract_pronouns("Cái bàn này của bà.") == ["bà"]


def test_parse_gold_pronouns_comma_separated():
    record = {"pronouns_subject": "ông, bố", "pronouns_object": "tôi"}
    assert parse_gold_pronouns(record) == ["ông", "bố", "tôi"]


def test_parse_gold_pronouns_null_fields():
    assert parse_gold_pronouns({"pronouns_subject": None, "pronouns_object": None}) == []


def test_corpus_f1_perfect_and_zero():
    perfect = corpus_pronoun_f1([(["tôi"], ["tôi"]), (["bà", "cháu"], ["bà", "cháu"])])
    assert perfect.f1 == 1.0
    zero = corpus_pronoun_f1([(["bà"], ["mẹ"])])
    assert zero.f1 == 0.0
    assert zero.fp == 1 and zero.fn == 1


def test_corpus_f1_partial_multiset():
    # gold có 2 "con", pred chỉ có 1 → tp=1, fn=1
    scores = corpus_pronoun_f1([(["con", "con", "mẹ"], ["con", "mẹ"])])
    assert scores.tp == 2 and scores.fn == 1 and scores.fp == 0
```

- [x] **Step 4: Chạy test, xác nhận FAIL**

```bash
python -m pytest tests/test_pronoun_metrics.py -v
```

Expected: FAIL / ERROR với `ModuleNotFoundError: No module named 'EVAL'`

- [x] **Step 5: Implement lexicon**

`demo/EVAL/pronoun_lexicon.py`:

```python
"""Vietnamese pronoun/kinship-term lexicon and extraction utilities.

Lexicon khoi tao tu quan sat dataset data/en-vi-speaker-with-time-pronouns;
Task 3 co buoc calibration de bo sung term con thieu.
"""
import re

PRONOUN_TERMS: frozenset[str] = frozenset({
    # ngoi 1
    "tôi", "ta", "tao", "tớ", "mình",
    "chúng tôi", "chúng ta", "chúng mình", "bọn tôi", "bọn ta", "bọn tao",
    "tụi tôi", "tụi tao", "tụi mình",
    # ngoi 2/3 + quan he than toc
    "anh", "em", "chị", "mày", "bạn", "cậu", "mợ", "cô", "dì", "chú", "bác",
    "thím", "ông", "bà", "cụ", "bố", "ba", "cha", "mẹ", "má", "u", "con",
    "cháu", "thầy", "ngài", "quý vị", "người ta",
    "các bạn", "các anh", "các em", "các chị", "các cô", "các chú",
    "các con", "các cháu", "các ông", "các bà",
    "nó", "hắn", "y", "gã", "ả", "họ", "chúng", "chúng nó", "bọn nó", "tụi nó",
    "anh ấy", "cô ấy", "chị ấy", "ông ấy", "bà ấy", "em ấy",
    "chú ấy", "bác ấy", "cậu ấy",
})

# Match dai-nhat-truoc; \w cua Python mac dinh Unicode-aware nen boundary
# hoat dong dung voi tieng Viet co dau.
_PATTERN = re.compile(
    r"(?<!\w)("
    + "|".join(re.escape(t) for t in sorted(PRONOUN_TERMS, key=len, reverse=True))
    + r")(?!\w)"
)


def extract_pronouns(text: str) -> list[str]:
    """Tra ve list term (da lowercase) tim thay trong text, theo thu tu xuat hien."""
    return [m.group(1) for m in _PATTERN.finditer(text.lower())]


def parse_gold_pronouns(record: dict) -> list[str]:
    """Doc nhan gold tu 1 record cua data/en-vi-speaker-with-time-pronouns."""
    terms: list[str] = []
    for field in ("pronouns_subject", "pronouns_object"):
        raw = record.get(field)
        if raw:
            terms.extend(t.strip().lower() for t in str(raw).split(",") if t.strip())
    return terms
```

- [x] **Step 6: Implement metric**

`demo/EVAL/pronoun_f1.py`:

```python
"""Micro-averaged Pronoun F1 giua hypothesis va gold (multiset per line)."""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class PronounScores:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def line_counts(gold: list[str], pred: list[str]) -> tuple[int, int, int]:
    g, p = Counter(gold), Counter(pred)
    tp = sum((g & p).values())
    return tp, sum(p.values()) - tp, sum(g.values()) - tp


def corpus_pronoun_f1(pairs: list[tuple[list[str], list[str]]]) -> PronounScores:
    tp = fp = fn = 0
    for gold, pred in pairs:
        t, f_p, f_n = line_counts(gold, pred)
        tp, fp, fn = tp + t, fp + f_p, fn + f_n
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PronounScores(precision, recall, f1, tp, fp, fn)
```

Tạo thêm `demo/EVAL/__init__.py` rỗng.

- [x] **Step 7: Chạy test, xác nhận PASS**

```bash
python -m pytest tests/test_pronoun_metrics.py -v
```

Expected: 6 passed

- [x] **Step 8: Commit**

```bash
git add demo/EVAL/ tests/
git commit -m "feat: Vietnamese pronoun lexicon and micro-F1 metric for eval harness"
```

---

### Task 3: Eval CLI (BLEU/chrF/Pronoun F1) + calibration trên dataset

**Files:**
- Create: `demo/EVAL/run_eval.py`
- Test: `tests/test_run_eval.py`

**Interfaces:**
- Consumes: `extract_pronouns`, `parse_gold_pronouns`, `corpus_pronoun_f1` (Task 2).
- Produces: CLI `python demo/EVAL/run_eval.py` với 2 mode; hàm `evaluate_lines(hyps: list[str], refs: list[str], gold_pronouns: list[list[str]] | None) -> dict` (keys: `"bleu"`, `"chrf"`, `"pronoun_precision"`, `"pronoun_recall"`, `"pronoun_f1"`, `"n_lines"`). Task 8 gọi CLI này.

- [x] **Step 1: Viết failing test**

`tests/test_run_eval.py`:

```python
import datetime

import srt as srt_lib

from EVAL.run_eval import align_by_time, evaluate_lines, load_srt_lines


def _sub(i: int, s: float, e: float, text: str) -> srt_lib.Subtitle:
    return srt_lib.Subtitle(index=i, start=datetime.timedelta(seconds=s),
                            end=datetime.timedelta(seconds=e), content=text)


def test_evaluate_identical_lines():
    hyps = ["Bà ơi, cháu xin lỗi.", "Tôi không biết."]
    report = evaluate_lines(hyps, list(hyps), gold_pronouns=None)
    assert report["bleu"] == 100.0
    assert report["pronoun_f1"] == 1.0
    assert report["n_lines"] == 2


def test_evaluate_with_explicit_gold():
    # gold labels khac voi nhung gi lexicon rut tu ref → uu tien gold
    hyps = ["Mẹ ơi, con xin lỗi."]
    refs = ["Bà ơi, cháu xin lỗi."]
    report = evaluate_lines(hyps, refs, gold_pronouns=[["bà", "cháu"]])
    assert report["pronoun_f1"] == 0.0
    assert report["bleu"] < 100.0


def test_load_srt_lines_normalizes_whitespace(tmp_path):
    p = tmp_path / "x.srt"
    p.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nXin   chào\nbạn\n",
        encoding="utf-8",
    )
    assert load_srt_lines(str(p)) == ["Xin chào bạn"]


def test_align_by_time_merges_overlapping_hyp_lines():
    # ASR cat dong khac reference → align theo overlap thoi gian
    hyp = [_sub(1, 0, 2, "Xin chào"), _sub(2, 2.5, 4, "bạn khỏe không"),
           _sub(3, 10, 12, "Tạm biệt")]
    ref = [_sub(1, 0, 5, "Xin chào bạn khỏe không"), _sub(2, 9, 12, "Tạm biệt")]
    pairs = align_by_time(hyp, ref)
    assert pairs == [
        ("Xin chào bạn khỏe không", "Xin chào bạn khỏe không"),
        ("Tạm biệt", "Tạm biệt"),
    ]
```

- [x] **Step 2: Chạy test, xác nhận FAIL**

```bash
python -m pytest tests/test_run_eval.py -v
```

Expected: FAIL với `ModuleNotFoundError` hoặc `ImportError`

- [x] **Step 3: Implement CLI**

`demo/EVAL/run_eval.py`:

```python
"""Eval harness: BLEU / chrF / Pronoun F1.

Mode A (so 2 file SRT, gold pronoun rut tu ref bang lexicon hoac tu dataset json):
    python demo/EVAL/run_eval.py --hyp_srt hyp.srt --ref_srt ref.srt [--report out.json]

Mode B (calibration tren dataset da gan nhan — khong can GPU):
    python demo/EVAL/run_eval.py --dataset_dir data/en-vi-speaker-with-time-pronouns \
        --hyp_field vietsub_raw [--report out.json]
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sacrebleu
import srt

from EVAL.pronoun_f1 import corpus_pronoun_f1
from EVAL.pronoun_lexicon import PRONOUN_TERMS, extract_pronouns, parse_gold_pronouns


def load_srt(path: str) -> list:
    """Doc SRT, normalize whitespace trong content (giu style pipeline)."""
    with open(path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))
    for s in subs:
        s.content = re.sub(
            r"\s+", " ",
            " ".join(l.strip() for l in s.content.splitlines()),
        ).strip()
    return subs


def load_srt_lines(path: str) -> list[str]:
    return [s.content for s in load_srt(path)]


def align_by_time(hyp_subs: list, ref_subs: list) -> list[tuple[str, str]]:
    """Voi moi ref line, noi cac hyp line co overlap thoi gian > 0.

    Dung khi so dong hyp (do ASR cat) khac so dong reference.
    """
    pairs: list[tuple[str, str]] = []
    for r in ref_subs:
        r_start, r_end = r.start.total_seconds(), r.end.total_seconds()
        parts = [
            h.content for h in hyp_subs
            if min(r_end, h.end.total_seconds())
            - max(r_start, h.start.total_seconds()) > 0
        ]
        pairs.append((" ".join(parts).strip(), r.content))
    return pairs


def evaluate_lines(
    hyps: list[str],
    refs: list[str],
    gold_pronouns: list[list[str]] | None,
) -> dict:
    if len(hyps) != len(refs):
        raise ValueError(f"hyp/ref length mismatch: {len(hyps)} vs {len(refs)}")
    if gold_pronouns is None:
        gold_pronouns = [extract_pronouns(r) for r in refs]
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    scores = corpus_pronoun_f1(
        [(g, extract_pronouns(h)) for g, h in zip(gold_pronouns, hyps)]
    )
    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "pronoun_precision": round(scores.precision, 4),
        "pronoun_recall": round(scores.recall, 4),
        "pronoun_f1": round(scores.f1, 4),
        "n_lines": len(hyps),
    }


def eval_dataset_dir(dataset_dir: str, hyp_field: str) -> dict:
    hyps: list[str] = []
    refs: list[str] = []
    gold: list[list[str]] = []
    missing_terms: dict[str, int] = {}
    for path in sorted(glob.glob(os.path.join(dataset_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            hyp, ref = r.get(hyp_field), r.get("vietnamese")
            if not hyp or not ref:
                continue
            hyps.append(hyp)
            refs.append(ref)
            g = parse_gold_pronouns(r)
            gold.append(g)
            for t in g:
                if t not in PRONOUN_TERMS:
                    missing_terms[t] = missing_terms.get(t, 0) + 1
    report = evaluate_lines(hyps, refs, gold)
    report["lexicon_missing_terms"] = dict(
        sorted(missing_terms.items(), key=lambda kv: -kv[1])[:30]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Subtitle eval: BLEU/chrF/Pronoun F1")
    parser.add_argument("--hyp_srt", type=str)
    parser.add_argument("--ref_srt", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--hyp_field", type=str, default="vietsub_raw")
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    if args.dataset_dir:
        report = eval_dataset_dir(args.dataset_dir, args.hyp_field)
    elif args.hyp_srt and args.ref_srt:
        hyp_subs, ref_subs = load_srt(args.hyp_srt), load_srt(args.ref_srt)
        if len(hyp_subs) == len(ref_subs):
            pairs = [(h.content, r.content)
                     for h, r in zip(hyp_subs, ref_subs)]
        else:
            print(f"line counts differ ({len(hyp_subs)} vs {len(ref_subs)}); "
                  f"aligning by time overlap", flush=True)
            pairs = align_by_time(hyp_subs, ref_subs)
        report = evaluate_lines(
            [p[0] for p in pairs], [p[1] for p in pairs], None
        )
    else:
        parser.error("need either --dataset_dir or (--hyp_srt and --ref_srt)")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        tmp = args.report + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        os.replace(tmp, args.report)


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Chạy test, xác nhận PASS**

```bash
python -m pytest tests/test_run_eval.py -v
```

Expected: 4 passed

- [x] **Step 5: Calibration trên dataset thật (local, không cần GPU)**

```bash
mkdir -p docs/eval
python demo/EVAL/run_eval.py --dataset_dir "data/en-vi-speaker-with-time-pronouns" --hyp_field vietsub_raw --report docs/eval/baseline_vietsub_raw.json
```

Kiểm tra output:
1. Chạy hết 356 file không crash.
2. `lexicon_missing_terms` — nếu có term gold xuất hiện >50 lần mà lexicon thiếu → thêm vào `PRONOUN_TERMS`, chạy lại đến khi các term tần suất cao được cover (đây là bước calibrate lexicon, chấp nhận đuôi dài term hiếm).
3. Ghi lại `pronoun_f1` của `vietsub_raw` — đây là **baseline trước-refine** để đối chiếu xu hướng với số liệu paper (E5 full-pipeline = 0.5806; số ở đây đo trên tập khác nên chỉ cần cùng bậc, không cần khớp).

- [x] **Step 6: Commit (kèm report baseline)**

```bash
git add demo/EVAL/run_eval.py tests/test_run_eval.py demo/EVAL/pronoun_lexicon.py docs/eval/
git commit -m "feat: eval CLI with BLEU/chrF/PronounF1 and dataset calibration baseline"
```

---

### Task 4: Registry schema — validate + merge (chống poisoning)

**Files:**
- Create: `demo/CHARACTER/__init__.py` (rỗng)
- Create: `demo/CHARACTER/registry_schema.py`
- Test: `tests/test_registry_schema.py`

**Interfaces:**
- Produces (Task 5, 6, 7 dùng đúng các tên này):
  - `Character(id: str, names: tuple[str, ...], gender: str, age_range: str, evidence_lines: tuple[int, ...])` — frozen dataclass
  - `Relation(from_id: str, to_id: str, rel_type: str, vi_self: str, vi_listener: str, confidence: str, evidence_lines: tuple[int, ...])` — frozen dataclass
  - `Registry(characters: tuple[Character, ...], relations: tuple[Relation, ...])` — frozen dataclass
  - `parse_registry(raw: dict, n_lines: int) -> Registry`
  - `merge_registries(regs: list[Registry]) -> Registry`
  - `registry_to_json(reg: Registry) -> dict` / `load_registry(path: str) -> Registry`
  - `CONFIDENCE_LEVELS: dict[str, int]` (`{"high": 2, "medium": 1, "low": 0}`)

- [x] **Step 1: Viết failing test**

`tests/test_registry_schema.py`:

```python
from CHARACTER.registry_schema import (
    Registry,
    merge_registries,
    parse_registry,
    registry_to_json,
)

RAW = {
    "characters": [
        {"id": "C1", "names": ["Meemaw", "Grandma"], "gender": "female",
         "age_range": "elderly", "evidence_lines": [35]},
        {"id": "C2", "names": ["Sheldon"], "gender": "male",
         "age_range": "child", "evidence_lines": [1]},
    ],
    "relations": [
        {"from_id": "C2", "to_id": "C1", "rel_type": "grandchild->grandmother",
         "vi_self": "cháu", "vi_listener": "bà", "confidence": "high",
         "evidence_lines": [35]},
        {"from_id": "C1", "to_id": "C2", "rel_type": "grandmother->grandchild",
         "vi_self": "bà", "vi_listener": "cháu", "confidence": "high",
         "evidence_lines": [35, 40]},
    ],
}


def test_parse_valid_registry():
    reg = parse_registry(RAW, n_lines=100)
    assert len(reg.characters) == 2
    assert len(reg.relations) == 2
    assert reg.relations[0].vi_self == "cháu"


def test_parse_drops_relation_without_valid_evidence():
    raw = {
        "characters": RAW["characters"],
        "relations": [
            {"from_id": "C2", "to_id": "C1", "rel_type": "x",
             "vi_self": "cháu", "vi_listener": "bà", "confidence": "high",
             "evidence_lines": [999]},  # ngoai pham vi transcript
        ],
    }
    assert parse_registry(raw, n_lines=100).relations == ()


def test_parse_drops_unknown_ids_and_self_loops():
    raw = {
        "characters": [RAW["characters"][0]],
        "relations": [
            {"from_id": "C9", "to_id": "C1", "rel_type": "x",
             "vi_self": "a", "vi_listener": "b", "confidence": "high",
             "evidence_lines": [1]},
            {"from_id": "C1", "to_id": "C1", "rel_type": "x",
             "vi_self": "a", "vi_listener": "b", "confidence": "high",
             "evidence_lines": [1]},
        ],
    }
    assert parse_registry(raw, n_lines=100).relations == ()


def test_merge_dedups_characters_by_shared_alias():
    reg1 = parse_registry(RAW, n_lines=100)
    raw2 = {
        "characters": [
            # cung nhan vat, chunk khac dat id khac + alias trung "Grandma"
            {"id": "C1", "names": ["Grandma", "Constance"], "gender": "female",
             "age_range": "elderly", "evidence_lines": [210]},
        ],
        "relations": [],
    }
    reg2 = parse_registry(raw2, n_lines=300)
    merged = merge_registries([reg1, reg2])
    assert len(merged.characters) == 2  # Meemaw/Grandma/Constance gop lam 1
    names = {n for c in merged.characters for n in c.names}
    assert "Constance" in names and "Sheldon" in names


def test_merge_keeps_highest_confidence_relation():
    reg1 = parse_registry(RAW, n_lines=100)
    raw2 = {
        "characters": RAW["characters"],
        "relations": [
            {"from_id": "C2", "to_id": "C1", "rel_type": "grandchild->grandmother",
             "vi_self": "em", "vi_listener": "chị", "confidence": "low",
             "evidence_lines": [50]},
        ],
    }
    merged = merge_registries([reg1, parse_registry(raw2, n_lines=100)])
    rel = next(r for r in merged.relations
               if r.rel_type == "grandchild->grandmother")
    assert rel.vi_self == "cháu"  # high thang low
    assert 50 in rel.evidence_lines  # evidence van duoc gop


def test_json_roundtrip():
    reg = parse_registry(RAW, n_lines=100)
    raw = registry_to_json(reg)
    reg2 = parse_registry(raw, n_lines=100)
    assert reg == reg2
```

- [x] **Step 2: Chạy test, xác nhận FAIL**

```bash
python -m pytest tests/test_registry_schema.py -v
```

Expected: FAIL với `ModuleNotFoundError: No module named 'CHARACTER'`

- [x] **Step 3: Implement schema**

`demo/CHARACTER/registry_schema.py`:

```python
"""Directed character-relationship registry: schema, validation, merge.

Chong poisoning: moi Relation BAT BUOC co evidence_lines hop le trong pham vi
transcript; edge khong evidence / id la / self-loop bi drop ngay khi parse.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

CONFIDENCE_LEVELS: dict[str, int] = {"high": 2, "medium": 1, "low": 0}
_GENDERS = {"male", "female", "unknown"}
_AGE_RANGES = {"child", "teen", "adult", "elderly", "unknown"}


@dataclass(frozen=True)
class Character:
    id: str
    names: tuple[str, ...]
    gender: str = "unknown"
    age_range: str = "unknown"
    evidence_lines: tuple[int, ...] = ()


@dataclass(frozen=True)
class Relation:
    from_id: str
    to_id: str
    rel_type: str
    vi_self: str        # speaker tu xung, vd "cháu"
    vi_listener: str    # speaker goi listener, vd "bà"
    confidence: str = "medium"
    evidence_lines: tuple[int, ...] = ()


@dataclass(frozen=True)
class Registry:
    characters: tuple[Character, ...] = ()
    relations: tuple[Relation, ...] = ()


def _valid_lines(value, n_lines: int) -> tuple[int, ...]:
    out = set()
    for x in value or []:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= n_lines:
            out.add(i)
    return tuple(sorted(out))


def parse_registry(raw: dict, n_lines: int) -> Registry:
    """Validate output tho cua LLM (hoac file da luu) thanh Registry sach."""
    characters: list[Character] = []
    ids: set[str] = set()
    for c in raw.get("characters") or []:
        cid = str(c.get("id", "")).strip()
        names = tuple(
            str(n).strip() for n in (c.get("names") or []) if str(n).strip()
        )
        if not cid or not names or cid in ids:
            continue
        ids.add(cid)
        characters.append(Character(
            id=cid,
            names=names,
            gender=c.get("gender") if c.get("gender") in _GENDERS else "unknown",
            age_range=(
                c.get("age_range") if c.get("age_range") in _AGE_RANGES
                else "unknown"
            ),
            evidence_lines=_valid_lines(c.get("evidence_lines"), n_lines),
        ))

    relations: list[Relation] = []
    seen: set[tuple[str, str]] = set()
    for r in raw.get("relations") or []:
        from_id = str(r.get("from_id", "")).strip()
        to_id = str(r.get("to_id", "")).strip()
        vi_self = str(r.get("vi_self", "")).strip().lower()
        vi_listener = str(r.get("vi_listener", "")).strip().lower()
        evidence = _valid_lines(r.get("evidence_lines"), n_lines)
        if (from_id not in ids or to_id not in ids or from_id == to_id
                or not vi_self or not vi_listener or not evidence
                or (from_id, to_id) in seen):
            continue
        seen.add((from_id, to_id))
        confidence = (
            r.get("confidence") if r.get("confidence") in CONFIDENCE_LEVELS
            else "low"
        )
        relations.append(Relation(
            from_id=from_id,
            to_id=to_id,
            rel_type=str(r.get("rel_type", "unknown")).strip() or "unknown",
            vi_self=vi_self,
            vi_listener=vi_listener,
            confidence=confidence,
            evidence_lines=evidence,
        ))
    return Registry(tuple(characters), tuple(relations))


def merge_registries(regs: list[Registry]) -> Registry:
    """Gop registry tu nhieu chunk: nhan vat trung alias (casefold) gop lam 1,
    relation trung (from, to) giu confidence cao nhat, evidence duoc union."""
    entries = [(ri, c) for ri, r in enumerate(regs) for c in r.characters]
    parent = list(range(len(entries)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    alias_owner: dict[str, int] = {}
    for i, (_, c) in enumerate(entries):
        for name in c.names:
            key = name.casefold()
            if key in alias_owner:
                union(i, alias_owner[key])
            else:
                alias_owner[key] = i

    groups: dict[int, list[int]] = {}
    for i in range(len(entries)):
        groups.setdefault(find(i), []).append(i)

    id_map: dict[tuple[int, str], str] = {}
    merged_chars: list[Character] = []
    for gi, (_, members) in enumerate(sorted(groups.items()), start=1):
        new_id = f"C{gi}"
        names: list[str] = []
        evidence: set[int] = set()
        gender = age_range = "unknown"
        for m in members:
            ri, c = entries[m]
            id_map[(ri, c.id)] = new_id
            for n in c.names:
                if n.casefold() not in {x.casefold() for x in names}:
                    names.append(n)
            evidence.update(c.evidence_lines)
            if gender == "unknown":
                gender = c.gender
            if age_range == "unknown":
                age_range = c.age_range
        merged_chars.append(Character(
            new_id, tuple(names), gender, age_range, tuple(sorted(evidence))
        ))

    best: dict[tuple[str, str], Relation] = {}
    for ri, reg in enumerate(regs):
        for rel in reg.relations:
            f = id_map.get((ri, rel.from_id))
            t = id_map.get((ri, rel.to_id))
            if f is None or t is None or f == t:
                continue
            cur = best.get((f, t))
            evidence = tuple(sorted(
                set(rel.evidence_lines)
                | set(cur.evidence_lines if cur else ())
            ))
            if (cur is None
                    or CONFIDENCE_LEVELS[rel.confidence]
                    > CONFIDENCE_LEVELS[cur.confidence]):
                best[(f, t)] = Relation(
                    f, t, rel.rel_type, rel.vi_self, rel.vi_listener,
                    rel.confidence, evidence,
                )
            else:
                best[(f, t)] = Relation(
                    cur.from_id, cur.to_id, cur.rel_type, cur.vi_self,
                    cur.vi_listener, cur.confidence, evidence,
                )
    relations = tuple(
        best[k] for k in sorted(best.keys())
    )
    return Registry(tuple(merged_chars), relations)


def registry_to_json(reg: Registry) -> dict:
    return {
        "characters": [
            {"id": c.id, "names": list(c.names), "gender": c.gender,
             "age_range": c.age_range, "evidence_lines": list(c.evidence_lines)}
            for c in reg.characters
        ],
        "relations": [
            {"from_id": r.from_id, "to_id": r.to_id, "rel_type": r.rel_type,
             "vi_self": r.vi_self, "vi_listener": r.vi_listener,
             "confidence": r.confidence,
             "evidence_lines": list(r.evidence_lines)}
            for r in reg.relations
        ],
    }


def load_registry(path: str) -> Registry:
    """Doc registry da validate tu disk (evidence da check luc build)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return parse_registry(raw, n_lines=10**9)
```

Tạo thêm `demo/CHARACTER/__init__.py` rỗng.

- [x] **Step 4: Chạy test, xác nhận PASS**

```bash
python -m pytest tests/test_registry_schema.py -v
```

Expected: 6 passed

- [x] **Step 5: Commit**

```bash
git add demo/CHARACTER/ tests/test_registry_schema.py
git commit -m "feat: directed character-relationship registry schema with evidence-gated validation and merge"
```

---

### Task 5: Chunking + extraction prompt + render block cho refine

**Files:**
- Create: `demo/CHARACTER/registry_prompt.py`
- Test: `tests/test_registry_prompt.py`

**Interfaces:**
- Consumes: `Registry`, `CONFIDENCE_LEVELS` (Task 4).
- Produces (Task 6, 7 dùng):
  - `chunk_lines(lines: list[str], chunk_size: int = 200, overlap: int = 30) -> list[tuple[int, list[str]]]` — trả `(start_index_0_based, sublist)`
  - `number_lines(lines: list[str], start: int = 1) -> str`
  - `captions_summary(vlm_scenes: list[dict], max_chars: int = 4000) -> str`
  - `build_extraction_prompt(numbered_dialogue: str, captions_text: str) -> tuple[str, str]` — `(system, user)`
  - `parse_llm_json(text: str) -> dict | None`
  - `render_registry_block(reg: Registry, max_relations: int = 24, min_confidence: str = "medium") -> str`

- [x] **Step 1: Viết failing test**

`tests/test_registry_prompt.py`:

```python
from CHARACTER.registry_prompt import (
    build_extraction_prompt,
    captions_summary,
    chunk_lines,
    number_lines,
    parse_llm_json,
    render_registry_block,
)
from CHARACTER.registry_schema import Character, Registry, Relation


def test_chunk_lines_overlap_and_coverage():
    lines = [str(i) for i in range(450)]
    chunks = chunk_lines(lines, chunk_size=200, overlap=30)
    assert chunks[0][0] == 0 and len(chunks[0][1]) == 200
    assert chunks[1][0] == 170  # step = 200 - 30
    # moi line phai nam trong it nhat 1 chunk
    covered = set()
    for start, sub in chunks:
        covered.update(range(start, start + len(sub)))
    assert covered == set(range(450))


def test_chunk_lines_short_input_single_chunk():
    assert chunk_lines(["a", "b"], chunk_size=200, overlap=30) == [(0, ["a", "b"])]


def test_number_lines_uses_1_based_global_index():
    assert number_lines(["hi", "yo"], start=171) == "171. hi\n172. yo"


def test_captions_summary_skips_error_captions():
    scenes = [
        {"caption": "A boy talks to his grandmother."},
        {"caption": "Error during analysis: CUDA OOM"},
        {"caption": ""},
    ]
    text = captions_summary(scenes)
    assert "grandmother" in text
    assert "Error" not in text


def test_parse_llm_json_handles_fences_and_prose():
    text = 'Here you go:\n```json\n{"characters": [], "relations": []}\n```\nDone.'
    assert parse_llm_json(text) == {"characters": [], "relations": []}
    assert parse_llm_json("no json here") is None


def test_extraction_prompt_mentions_directed_and_schema():
    system, user = build_extraction_prompt("1. Hello", "Scene 1: two people")
    assert "DIRECTED" in system
    assert "vi_self" in user and "evidence_lines" in user
    assert "1. Hello" in user


def _sample_registry() -> Registry:
    chars = (
        Character("C1", ("Meemaw", "Grandma"), "female", "elderly", (35,)),
        Character("C2", ("Sheldon",), "male", "child", (1,)),
    )
    rels = (
        Relation("C2", "C1", "grandchild->grandmother", "cháu", "bà",
                 "high", (35,)),
        Relation("C1", "C2", "grandmother->grandchild", "bà", "cháu",
                 "low", (35,)),
    )
    return Registry(chars, rels)


def test_render_block_filters_by_confidence():
    block = render_registry_block(_sample_registry(), min_confidence="medium")
    assert "<Character Registry>" in block and "</Character Registry>" in block
    assert 'calls self "cháu"' in block          # high: giu
    assert 'calls self "bà"' not in block        # low: bi loc
    assert "Meemaw" in block and "Sheldon" in block


def test_render_block_empty_registry_returns_empty_string():
    assert render_registry_block(Registry()) == ""
```

- [x] **Step 2: Chạy test, xác nhận FAIL**

```bash
python -m pytest tests/test_registry_prompt.py -v
```

Expected: FAIL với `ImportError`

- [x] **Step 3: Implement**

`demo/CHARACTER/registry_prompt.py`:

```python
"""Chunking transcript, prompt trich xuat registry, va render block cho refine."""
from __future__ import annotations

import json
import re

from CHARACTER.registry_schema import CONFIDENCE_LEVELS, Registry

EXTRACTION_SYSTEM = (
    "You are an expert film-script analyst. You read English movie dialogue and "
    "identify the characters and the DIRECTED relationships between them, so that "
    "correct Vietnamese pronouns (xưng hô) can be chosen later.\n"
    "Rules:\n"
    "- Relations are DIRECTED: A->B and B->A are two separate entries. A father "
    "speaking to his son uses different Vietnamese pronouns than the son speaking "
    "to the father.\n"
    "- For each direction give vi_self (how the speaker refers to themself) and "
    "vi_listener (how the speaker addresses the listener), using common Vietnamese "
    "pronoun/kinship terms (tôi, anh, em, chị, ông, bà, bố, mẹ, con, cháu, mày, "
    "tao, cậu, chú, bác, cô, ...).\n"
    "- Only include characters and relations supported by evidence: cite the "
    "dialogue line numbers in evidence_lines. If unsure, use confidence \"low\" "
    "or omit the relation entirely.\n"
    "- Output STRICT JSON only. No markdown fences, no commentary."
)

_SCHEMA_EXAMPLE = {
    "characters": [
        {"id": "C1", "names": ["Meemaw", "Grandma"], "gender": "female",
         "age_range": "elderly", "evidence_lines": [35]},
        {"id": "C2", "names": ["Sheldon"], "gender": "male",
         "age_range": "child", "evidence_lines": [1]},
    ],
    "relations": [
        {"from_id": "C2", "to_id": "C1", "rel_type": "grandchild->grandmother",
         "vi_self": "cháu", "vi_listener": "bà", "confidence": "high",
         "evidence_lines": [35]},
        {"from_id": "C1", "to_id": "C2", "rel_type": "grandmother->grandchild",
         "vi_self": "bà", "vi_listener": "cháu", "confidence": "high",
         "evidence_lines": [35]},
    ],
}


def chunk_lines(
    lines: list[str], chunk_size: int = 200, overlap: int = 30
) -> list[tuple[int, list[str]]]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks: list[tuple[int, list[str]]] = []
    step = chunk_size - overlap
    for start in range(0, len(lines), step):
        chunks.append((start, lines[start:start + chunk_size]))
        if start + chunk_size >= len(lines):
            break
    return chunks


def number_lines(lines: list[str], start: int = 1) -> str:
    return "\n".join(f"{start + i}. {line}" for i, line in enumerate(lines))


def captions_summary(vlm_scenes: list[dict], max_chars: int = 4000) -> str:
    parts = []
    for i, sc in enumerate(vlm_scenes):
        caption = (sc.get("caption") or "").strip()
        # run_vlm.py luu error string vao caption khi crash → loc ra
        if caption and not caption.startswith("Error"):
            parts.append(f"Scene {i + 1}: {caption}")
    return "\n".join(parts)[:max_chars]


def build_extraction_prompt(
    numbered_dialogue: str, captions_text: str
) -> tuple[str, str]:
    user = (
        "<Visual Scene Captions>\n"
        f"{captions_text or 'None'}\n"
        "</Visual Scene Captions>\n\n"
        "<English Dialogue (numbered lines)>\n"
        f"{numbered_dialogue}\n"
        "</English Dialogue>\n\n"
        "Return JSON with EXACTLY this schema (keys: characters[], relations[]; "
        "relation keys: from_id, to_id, rel_type, vi_self, vi_listener, "
        "confidence, evidence_lines):\n"
        f"{json.dumps(_SCHEMA_EXAMPLE, ensure_ascii=False, indent=1)}"
    )
    return EXTRACTION_SYSTEM, user


def parse_llm_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def render_registry_block(
    reg: Registry, max_relations: int = 24, min_confidence: str = "medium"
) -> str:
    if not reg.characters:
        return ""
    min_rank = CONFIDENCE_LEVELS[min_confidence]
    relations = sorted(
        (r for r in reg.relations
         if CONFIDENCE_LEVELS[r.confidence] >= min_rank),
        key=lambda r: -CONFIDENCE_LEVELS[r.confidence],
    )[:max_relations]
    name_of = {c.id: c.names[0] for c in reg.characters}
    lines = ["<Character Registry>", "Characters:"]
    for c in reg.characters:
        aka = f" (aka {', '.join(c.names[1:])})" if len(c.names) > 1 else ""
        lines.append(f"- {c.names[0]}{aka}: {c.gender}, {c.age_range}")
    if relations:
        lines.append(
            "Directed relations (speaker -> listener; use these Vietnamese "
            "pronouns when this speaker addresses this listener):"
        )
        for r in relations:
            lines.append(
                f'- {name_of[r.from_id]} -> {name_of[r.to_id]} [{r.rel_type}]: '
                f'speaker calls self "{r.vi_self}", '
                f'calls listener "{r.vi_listener}"'
            )
    lines.append("</Character Registry>")
    return "\n".join(lines)
```

- [x] **Step 4: Chạy test, xác nhận PASS**

```bash
python -m pytest tests/test_registry_prompt.py -v
```

Expected: 8 passed

- [x] **Step 5: Commit**

```bash
git add demo/CHARACTER/registry_prompt.py tests/test_registry_prompt.py
git commit -m "feat: registry extraction prompt, transcript chunking, and refine-prompt rendering"
```

---

### Task 6: build_registry.py — GPU runner (extraction bằng Gemma base)

**Files:**
- Create: `demo/CHARACTER/build_registry.py`
- Test: `tests/test_build_registry.py` (phần orchestrator, generate_fn giả — chạy local)

**Interfaces:**
- Consumes: toàn bộ Task 4 + Task 5.
- Produces:
  - `extract_registry(en_lines: list[str], captions_text: str, generate_fn, chunk_size: int = 200, overlap: int = 30) -> Registry` — `generate_fn(system: str, user: str) -> str`
  - `run(en_srt_path: str, vlm_json_path: str, output_json_path: str, model_name: str = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit", cache_dir: str | None = None, max_seq_length: int = 8192, max_new_tokens: int = 2048) -> str` — Task 8 (run_pipeline) gọi hàm này.
  - CLI: `python demo/CHARACTER/build_registry.py --en_srt ... --vlm_json ... --output_json ...`

**Lý do dùng Gemma BASE (không dùng adapter refine):** adapter `thevan2404/best_gemma_scene_context` được finetune cho task sửa phụ đề line-by-line — bắt nó sinh JSON graph là off-distribution. Base model instruct làm task extraction tổng quát tốt hơn, và không tốn thêm VRAM (cùng base 4-bit).

- [x] **Step 1: Viết failing test (orchestrator với generate_fn giả)**

`tests/test_build_registry.py`:

```python
import json

from CHARACTER.build_registry import extract_registry

_CHUNK_RESPONSE = json.dumps({
    "characters": [
        {"id": "C1", "names": ["Anna"], "gender": "female",
         "age_range": "adult", "evidence_lines": [1]},
        {"id": "C2", "names": ["Tom"], "gender": "male",
         "age_range": "adult", "evidence_lines": [2]},
    ],
    "relations": [
        {"from_id": "C1", "to_id": "C2", "rel_type": "wife->husband",
         "vi_self": "em", "vi_listener": "anh", "confidence": "high",
         "evidence_lines": [1, 2]},
    ],
})


def test_extract_registry_single_chunk():
    def fake_generate(system: str, user: str) -> str:
        assert "DIRECTED" in system
        return _CHUNK_RESPONSE

    reg = extract_registry(["Hello Tom.", "Hi Anna."], "", fake_generate)
    assert len(reg.characters) == 2
    assert len(reg.relations) == 1
    assert reg.relations[0].vi_self == "em"


def test_extract_registry_skips_unparseable_chunk():
    calls = {"n": 0}

    def flaky_generate(system: str, user: str) -> str:
        calls["n"] += 1
        return "GARBAGE NOT JSON" if calls["n"] == 1 else _CHUNK_RESPONSE

    lines = [f"line {i}" for i in range(300)]  # 2 chunk voi size=200/overlap=30
    reg = extract_registry(lines, "", flaky_generate)
    assert calls["n"] == 2
    assert len(reg.characters) == 2  # chunk 2 van duoc dung


def test_extract_registry_empty_transcript():
    reg = extract_registry([], "", lambda s, u: "{}")
    assert reg.characters == () and reg.relations == ()
```

- [x] **Step 2: Chạy test, xác nhận FAIL**

```bash
python -m pytest tests/test_build_registry.py -v
```

Expected: FAIL với `ImportError`

- [x] **Step 3: Implement**

`demo/CHARACTER/build_registry.py`:

```python
"""Build directed character-relationship registry tu EN transcript + VLM captions.

GPU runner: load Gemma-3-12B base 4-bit (KHONG dung adapter refine), chay
extraction theo chunk, validate + merge, ghi JSON atomic.

CLI:
    python build_registry.py --en_srt movie.srt --vlm_json captions.json \
        --output_json movie.registry.json
"""
import argparse
import json
import os
import re
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from CHARACTER.registry_prompt import (
    build_extraction_prompt,
    captions_summary,
    chunk_lines,
    number_lines,
    parse_llm_json,
)
from CHARACTER.registry_schema import (
    Registry,
    merge_registries,
    parse_registry,
    registry_to_json,
)


def extract_registry(
    en_lines: list[str],
    captions_text: str,
    generate_fn,
    chunk_size: int = 200,
    overlap: int = 30,
) -> Registry:
    """Orchestrator thuan: generate_fn(system, user) -> str duoc inject de test."""
    partials: list[Registry] = []
    chunks = chunk_lines(en_lines, chunk_size, overlap)
    for idx, (start, lines) in enumerate(chunks):
        system, user = build_extraction_prompt(
            number_lines(lines, start=start + 1), captions_text
        )
        decoded = generate_fn(system, user)
        raw = parse_llm_json(decoded)
        if raw is None:
            print(f"[registry] chunk {idx + 1}/{len(chunks)}: "
                  f"unparseable JSON, skipped", flush=True)
            continue
        partials.append(parse_registry(raw, n_lines=len(en_lines)))
        print(f"[registry] chunk {idx + 1}/{len(chunks)}: ok", flush=True)
    return merge_registries(partials)


def _load_en_lines(en_srt_path: str) -> list[str]:
    import srt
    with open(en_srt_path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))
    return [
        re.sub(r"\s+", " ",
               " ".join(l.strip() for l in s.content.splitlines())).strip()
        for s in subs
    ]


def _make_generate_fn(model_name: str, cache_dir: str,
                      max_seq_length: int, max_new_tokens: int):
    import torch
    from unsloth import FastLanguageModel

    print(f"[registry] Loading base model 4-bit: {model_name}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        cache_dir=cache_dir,
    )
    FastLanguageModel.for_inference(model)

    def generate_fn(system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            add_generation_prompt=True,
        ).to("cuda")
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True
        )

    return generate_fn


def run(
    en_srt_path: str,
    vlm_json_path: str,
    output_json_path: str,
    model_name: str = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    cache_dir: str | None = None,
    max_seq_length: int = 8192,
    max_new_tokens: int = 2048,
) -> str:
    if cache_dir is None:
        cache_dir = os.path.join(ROOT_DIR, "cache")
    en_lines = _load_en_lines(en_srt_path)
    with open(vlm_json_path, "r", encoding="utf-8") as f:
        vlm_scenes = json.load(f)
    captions_text = captions_summary(vlm_scenes)

    generate_fn = _make_generate_fn(
        model_name, cache_dir, max_seq_length, max_new_tokens
    )
    registry = extract_registry(en_lines, captions_text, generate_fn)
    print(f"[registry] {len(registry.characters)} characters, "
          f"{len(registry.relations)} directed relations", flush=True)

    tmp = output_json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry_to_json(registry), f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_json_path)
    print(f"[registry] Saved: {output_json_path}", flush=True)
    return output_json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build directed character-relationship registry."
    )
    parser.add_argument("--en_srt", type=str, required=True)
    parser.add_argument("--vlm_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default="unsloth/gemma-3-12b-it-unsloth-bnb-4bit")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()
    run(args.en_srt, args.vlm_json, args.output_json, args.model_name,
        args.cache_dir, args.max_seq_length, args.max_new_tokens)


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Chạy test local, xác nhận PASS**

```bash
python -m pytest tests/test_build_registry.py -v
```

Expected: 3 passed (torch/unsloth chỉ import bên trong `_make_generate_fn` nên test local không cần GPU)

- [x] **Step 5: Smoke test trên SERVER (cần GPU + HF_TOKEN)**

```bash
python demo/CHARACTER/build_registry.py --en_srt "demo/output/test.(Tiếng Anh).srt" --vlm_json "demo/output/test.captions.json" --output_json "demo/output/test.registry.json"
```

Kiểm tra `demo/output/test.registry.json`:
1. JSON hợp lệ, có ≥1 character.
2. Mọi relation có `vi_self`/`vi_listener` là từ xưng hô tiếng Việt và `evidence_lines` nằm trong phạm vi transcript.
3. **Case bà–cháu:** phim test có nhân vật Meemaw (bà) — registry phải có edge `grandchild->grandmother` với "cháu"/"bà" (đây chính là lỗi mẹ/con đã ghi nhận trong survey). Nếu sai → chỉnh EXTRACTION_SYSTEM (thêm hint từ caption) rồi chạy lại, KHÔNG hạ điều kiện validate.

- [x] **Step 6: Commit**

```bash
git add demo/CHARACTER/build_registry.py tests/test_build_registry.py
git commit -m "feat: GPU registry builder with injectable generate_fn and atomic output"
```

---

### Task 7: Tiêm registry vào prompt refine (refine_llm.py)

> **⚠️ THAY ĐỔI THIẾT KẾ SAU KHI CHẠY THỰC TẾ (2026-08-13):** Cách tiêm block
> `<Character Registry>` sau `</Scene Context>` như mô tả dưới đây làm adapter
> **suy biến hoàn toàn** (mọi dòng output thành 1 chuỗi rác lặp lại, vd
> `-  -Tốt.`). Đã cô lập bằng thí nghiệm A/B: thủ phạm là block XML ngoài
> Scene Context (prompt shape ngoài phân phối train của adapter), KHÔNG phải
> `max_seq_length` 2048→4096. Fix (commit `fix: inject registry as
> caption-style relationship line inside scene context`): render registry bằng
> `render_registry_context()` thành 1 dòng `Relationship (whole-film analysis):
> ...` nối VÀO TRONG `<Scene Context>`, đúng style caption VLM
> (`3. Relationship: [A & B] - [Type]`) mà adapter đã được train.
> `base_system` giữ nguyên 100% so với baseline. Hàm `render_registry_block()`
> vẫn tồn tại (đã test) nhưng không còn được refine_llm dùng.

**Files:**
- Modify: `demo/LLM/refine_llm.py` (args ~dòng 26–37, signature `refine_subtitles` ~dòng 51–61, `base_system` ~dòng 132–145, build prompt ~dòng 147–166, `main()` ~dòng 279)

**Interfaces:**
- Consumes: `load_registry` (Task 4), `render_registry_block` (Task 5).
- Produces: `refine_subtitles(..., registry_json_path: str | None = None)` + CLI arg `--registry_json`. Task 8 (run_pipeline) truyền param này.

- [x] **Step 1: Thêm CLI arg**

Trong `parse_args()` của `demo/LLM/refine_llm.py`, thêm sau `--llm_batch_size`:

```python
    parser.add_argument("--registry_json", type=str, default=None,
                        help="Optional character-relationship registry JSON "
                             "(built by CHARACTER/build_registry.py).")
```

- [x] **Step 2: Thêm param vào signature + load registry**

Sửa signature `refine_subtitles(...)`: thêm param cuối `registry_json_path=None`.

Ngay sau block đọc `vlm_scenes` (sau dòng `vlm_scenes = json.load(f)`), thêm:

```python
    # ── Optional character registry ──────────────────────────────────────────
    registry_block = ""
    if registry_json_path and os.path.exists(registry_json_path):
        import sys
        if ROOT_DIR not in sys.path:
            sys.path.insert(0, ROOT_DIR)  # ROOT_DIR = demo/, chua package CHARACTER
        from CHARACTER.registry_prompt import render_registry_block
        from CHARACTER.registry_schema import load_registry
        registry_block = render_registry_block(load_registry(registry_json_path))
        if registry_block:
            print(f"📇 Character registry loaded: {registry_json_path}", flush=True)
```

Lưu ý: `ROOT_DIR` trong refine_llm.py đã trỏ đến `demo/` (dòng 16), nên chỉ cần insert `ROOT_DIR` vào sys.path là import được package `CHARACTER`.

- [x] **Step 3: Nối block vào system prompt**

Sửa `base_system` (dòng ~132): thêm ngay trước chuỗi đóng, sau câu "Output only the corrected Vietnamese translation, line by line.":

```python
    if registry_block:
        base_system += (
            "\n    A <Character Registry> section lists the film's characters and "
            "DIRECTED relations with the Vietnamese pronouns each speaker should "
            "use. When a dialogue line matches a listed speaker->listener pair, "
            "prefer the registry's pronouns over the rough translation's."
        )
```

Sửa chỗ build `full_sys` (dòng ~151):

```python
        full_sys = (f"{base_system}\n"
                    f"    <Scene Context>\n    {item['context']}\n    </Scene Context>")
        if registry_block:
            full_sys += f"\n{registry_block}"
```

- [x] **Step 4: Nối param từ main()**

Trong `main()`, thêm vào call `refine_subtitles(...)`:

```python
        registry_json_path=args.registry_json,
```

- [x] **Step 5: Verify nhanh (local, không GPU): module compile + block rendering đúng vị trí**

```bash
python -c "
import ast, io
src = open('demo/LLM/refine_llm.py', encoding='utf-8').read()
ast.parse(src)
assert 'registry_json_path=None' in src
assert '--registry_json' in src
assert 'render_registry_block' in src
print('OK')
"
```

Expected: `OK` (không import module vì unsloth không có trên máy local)

- [x] **Step 6: Commit**

```bash
git add demo/LLM/refine_llm.py
git commit -m "feat: opt-in character-registry block in Gemma refine prompt"
```

---

### Task 8: Nối vào pipeline (step 5b) + E2E A/B + eval

**Files:**
- Modify: `demo/run_pipeline.py` (args ~dòng 35–43, thêm `step5b`, sửa `step6_run_llm` ~dòng 203–230, `main()` ~dòng 232–291)

**Interfaces:**
- Consumes: `build_registry.run(...)` (Task 6), `refine_subtitles(..., registry_json_path=...)` (Task 7), eval CLI (Task 3).
- Produces: `run_pipeline.py --use_registry` tạo `{base}.registry.json` và refined SRT có registry; mặc định (không flag) hành vi y hệt cũ.

- [x] **Step 1: Thêm flag + đường dẫn**

Trong `parse_args()` thêm:

```python
    parser.add_argument("--use_registry", action="store_true",
                        help="Build a character-relationship registry (step 5b) "
                             "and inject it into the Gemma refinement prompt.")
```

Trong `main()`, sau `output_srt_path = ...` (dòng ~249) thêm:

```python
    registry_json_path = os.path.join(args.output_dir, f"{base_name}.registry.json")
```

- [x] **Step 2: Thêm step5b**

Thêm hàm mới sau `step5_run_nmt` (sau dòng 201), cùng pattern skip/unload với các step khác:

```python
def step5b_build_registry(english_srt_path, vlm_json_path, registry_json_path, cache_dir):
    print("\n=== STEP 5b: Building Character Registry (Gemma base) ===", flush=True)
    if os.path.exists(registry_json_path):
        print(f"Registry already exists: {registry_json_path}. Skipping.", flush=True)
        return registry_json_path

    sys.path.append(ROOT_DIR)
    from CHARACTER import build_registry

    build_registry.run(
        en_srt_path=english_srt_path,
        vlm_json_path=vlm_json_path,
        output_json_path=registry_json_path,
        cache_dir=cache_dir,
    )

    print("Unloading Registry Model...", flush=True)
    for mod in ("CHARACTER.build_registry",):
        if mod in sys.modules:
            del sys.modules[mod]
    free_gpu_memory()
    return registry_json_path
```

- [x] **Step 3: Sửa step6 nhận registry**

Sửa signature `step6_run_llm(...)`: thêm param cuối `registry_json_path=None`.

Trong body, sửa call `refine_llm.refine_subtitles(...)`:

```python
    refine_llm.refine_subtitles(
        en_srt_path=english_srt_path,
        vinai_srt_path=rough_srt_path,
        vlm_json_path=vlm_json_path,
        output_srt_path=output_srt_path,
        adapter_model_name="thevan2404/best_gemma_scene_context",
        cache_dir=cache_dir,
        max_seq_length=4096 if registry_json_path else 2048,
        max_new_tokens=1024,
        llm_batch_size=llm_batch_size,
        registry_json_path=registry_json_path,
    )
```

(`max_seq_length` tăng lên 4096 khi có registry vì system prompt dài thêm ~500–800 token.)

- [x] **Step 4: Gọi step5b trong main()**

Trong `main()`, giữa Step 5 và Step 6 (sau dòng `durations["Step 5: ..."]`), thêm:

```python
    # Step 5b: Character Registry (optional)
    if args.use_registry:
        t0 = time.time()
        step5b_build_registry(english_srt_path, vlm_json_path,
                              registry_json_path, args.cache_dir)
        durations["Step 5b: Character Registry"] = time.time() - t0
```

Sửa call step6:

```python
    step6_run_llm(english_srt_path, rough_srt_path, vlm_json_path,
                  output_srt_path, args.cache_dir, args.llm_batch_size,
                  registry_json_path=registry_json_path if args.use_registry else None)
```

- [x] **Step 5: Verify cú pháp local**

```bash
python -c "
import ast
src = open('demo/run_pipeline.py', encoding='utf-8').read()
ast.parse(src)
assert '--use_registry' in src and 'step5b_build_registry' in src
print('OK')
"
```

Expected: `OK`

- [x] **Step 6: Chạy toàn bộ test suite local lần cuối**

```bash
python -m pytest tests/ -v
```

Expected: tất cả pass (≥20 test)

- [ ] **Step 7: E2E A/B trên SERVER**

Baseline đã có sẵn (`demo/output/test.(Tiếng Việt_tinh_chinh).srt`). Chạy nhánh registry với output riêng — copy output dir hoặc rename file cũ trước:

```bash
cp "demo/output/test.(Tiếng Việt_tinh_chinh).srt" "demo/output/test.baseline.srt"
rm "demo/output/test.(Tiếng Việt_tinh_chinh).srt" "demo/output/test.(Tiếng Việt_tinh_chinh).srt.json"
python demo/run_pipeline.py --video_path <test_video> --use_registry
```

Kiểm tra:
1. Pipeline chạy hết, có log `STEP 5b` và `📇 Character registry loaded`.
2. `demo/output/test.registry.json` tồn tại và hợp lệ.
3. Diff 2 bản SRT — các dòng thoại Meemaw phải chuyển mẹ/con → bà/cháu:

```bash
diff "demo/output/test.baseline.srt" "demo/output/test.(Tiếng Việt_tinh_chinh).srt" | head -50
```

- [ ] **Step 8: Đo Pronoun F1 trên eval set có reference (SERVER)**

Chọn 3–5 phim từ `data/en-vi-speaker-with-time-pronouns/` mà server còn video gốc. Với mỗi phim: chạy pipeline 2 lần (có/không `--use_registry`), rồi:

```bash
python demo/EVAL/run_eval.py --hyp_srt <refined_baseline.srt> --ref_srt <reference.srt> --report docs/eval/<movie>_baseline.json
python demo/EVAL/run_eval.py --hyp_srt <refined_registry.srt> --ref_srt <reference.srt> --report docs/eval/<movie>_registry.json
```

Nếu reference chỉ có dạng dataset JSON, dựng ref SRT từ field `vietnamese` theo timestamp `start`/`end` (thay `movie_XXX` bằng file thật):

```bash
python - <<'EOF'
import datetime
import json

import srt

records = json.load(open(
    "data/en-vi-speaker-with-time-pronouns/movie_XXX.json", encoding="utf-8"))
subs = [
    srt.Subtitle(index=i + 1,
                 start=datetime.timedelta(seconds=r["start"]),
                 end=datetime.timedelta(seconds=r["end"]),
                 content=r["vietnamese"])
    for i, r in enumerate(records) if r.get("vietnamese")
]
with open("docs/eval/movie_XXX.ref.srt", "w", encoding="utf-8") as f:
    f.write(srt.compose(subs))
print(f"wrote {len(subs)} lines")
EOF
```

Số dòng hyp (ASR cắt) sẽ khác số dòng ref — `run_eval.py` tự động align theo time overlap (đã implement + test ở Task 3), so sánh baseline/registry vẫn công bằng vì cùng một phép align.

**Acceptance criteria (gate V0):**
- Pronoun F1 (registry) ≥ Pronoun F1 (baseline) + **0.03** trung bình trên eval set.
- BLEU/chrF không giảm quá 0.5 điểm (registry không được phá nghĩa câu).
- Không dòng nào rỗng/lệch index trong SRT output (đếm dòng 2 file bằng nhau).

- [ ] **Step 9: Commit + ghi kết quả**

```bash
git add demo/run_pipeline.py docs/eval/
git commit -m "feat: wire optional character-registry step 5b into pipeline with A/B eval results"
```

Ghi kết quả A/B (con số cụ thể) vào memory `video-captioning-pipeline.md` để phiên sau có gate V1.

---

## Sau V0 — outline quyết định cho V1/V2 (KHÔNG thuộc phạm vi thực thi plan này)

Hai phase sau **phụ thuộc kết quả V0**, sẽ viết plan riêng khi có số liệu:

**Gate → V1 (Speaker Attribution, hướng Look-Listen-Recognise + DramaSR-LRM):**
- Nếu V0 đạt +0.03 nhưng error analysis cho thấy lỗi còn lại chủ yếu do *không biết ai đang nói* (registry đúng nhưng Gemma gán nhầm speaker cho dòng) → V1: CAM++ embedding per VAD-chunk (`data/speaker_verify_campp_en_full/` đã có sẵn pipeline), cluster thành speaker_id, LLM map cluster→character bằng ngữ cảnh (xử lý câu ngắn "You know."/"Me." theo Huh & Zisserman), tiêm `[SPEAKER: Sheldon]` per line vào prompt refine.
- Nếu V0 ≈ 0 cải thiện → dừng, error analysis trước khi đầu tư V1 (có thể bottleneck nằm ở NMT per-sentence, không phải thiếu quan hệ).

**Gate → V2 (Re-finetune adapter):**
- Khi V0+V1 đã có format input ổn định → re-finetune adapter Gemma trên `data/en-vi-speaker-with-time-pronouns/` (356 phim) với prompt chứa registry + speaker tag, `max_seq_length=4096`, kế thừa script finetune_gemma_v6.py trên server. Đích: Pronoun F1 ≥ 0.70.

**Rủi ro đã phòng trong V0 (mang sang V1/V2):** một edge sai đầu phim làm hỏng cả phim → đã chặn bằng evidence bắt buộc + confidence filter (`min_confidence="medium"`) + registry chỉ là *preference* trong prompt chứ không phải mệnh lệnh tuyệt đối ("prefer ... over the rough translation's").
