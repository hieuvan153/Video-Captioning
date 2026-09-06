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
    # bo sung tu calibration Task 3 (term gold xuat hien >50 lan trong
    # data/en-vi-speaker-with-time-pronouns ma lexicon ban dau thieu)
    "ai", "người", "anh ta", "mọi người", "sếp", "chúa", "cô ta", "ông ta",
    "cậu ta", "con bé", "bọn", "các", "thằng bé", "bố mẹ", "vợ", "bé",
    "con trai", "kẻ", "cả", "nhau", "đứa", "cưng", "tên", "thằng", "đồ",
    "nhóc", "lũ", "các cậu", "lão", "em bé",
    "chồng", "cô bé", "cậu bé", "bọn con", "ngươi", "mấy người",
    "các người", "anh chàng", "thằng nhóc",
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
