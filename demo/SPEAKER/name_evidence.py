"""Bang chung ve ten rut thang tu loi thoai, dung truoc va sau khi hoi LLM.

Do tren 5 phim eval (2026-08-16), mapping cluster->ten chi bang LLM sai nhieu
va sai theo kieu du doan duoc:

- Cluster noi "Adrian, I can't believe you" bi gan chinh ten Adrian — LLM bam
  vao ten XUAT HIEN trong dong, trong khi ten do la nguoi ĐƯỢC GỌI.
- Cluster co san nhan kich ban trong text ("GEORGE JR.: The first one is...")
  van bi doan lech sang nhan vat khac.

Ba loai bang chung o day khong doan gi ca, chi doc lai nhung gi loi thoai da
noi ro. Chung manh hon LLM nen duoc uu tien hon:

1. script_label_anchors — nhan kich ban ASR giu lai. Do duoc: 5/21 cluster o
   movie_045, 7/21 o movie_046, 0 o ba phim con lai. Hiem nhung gan nhu chac.
2. mention_exclusions — cluster nhac ten X nhieu lan thi hau nhu khong phai X.
   Doi dao nhat: 41-130 lan nhac ten moi phim.
3. vocative_hints — ai do goi "X," roi cluster khac noi ngay sau => cluster do
   co the la X. 7-16 luot moi phim, dung lam goi y cho LLM chu khong ep.
"""
from __future__ import annotations

import re

from SPEAKER.align import real_tag

# "MEEMAW:", "GEORGE JR.:", "ADULT SHELDON (V.O.):" — ASR giu lai nhan kich ban.
_SCRIPT_LABEL = re.compile(r"\b([A-Z][A-Z0-9 .'’-]{1,24}?)(?:\s*\([^)]{0,30}\))?:")


def _name_pattern(name: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)


def _vocative_pattern(name: str) -> re.Pattern:
    """Ten o vi tri goi dap: dau cau roi den dau phay/hoi, hoac cuoi cau sau phay."""
    n = re.escape(name)
    return re.compile(
        rf"(?:^|[.!?]\s+){n}\s*[,!?]|,\s*{n}\s*[.!?]*$",
        re.IGNORECASE,
    )


def _canonical(valid_names: list[str]) -> dict[str, str]:
    return {n.casefold(): n for n in valid_names}


def script_label_anchors(
    chunks: list[dict], valid_names: list[str]
) -> dict[str, str]:
    """cluster -> ten, tu nhan kich ban trong chinh loi thoai cua cluster do.

    Cluster co hai nhan khac nhau -> bo, khong chon bua.
    """
    canon = _canonical(valid_names)
    votes: dict[str, set[str]] = {}
    for c in chunks:
        tag = real_tag(c)
        if not tag:
            continue
        for label in _SCRIPT_LABEL.findall(c.get("english") or ""):
            real = canon.get(label.strip().casefold())
            if real:
                votes.setdefault(tag, set()).add(real)
    return {t: next(iter(v)) for t, v in votes.items() if len(v) == 1}


def mention_exclusions(
    chunks: list[dict], valid_names: list[str], min_mentions: int = 2
) -> dict[str, set[str]]:
    """cluster -> tap ten cluster do KHONG the mang.

    Nguoi ta goi ten nguoi khac, khong goi ten minh. Mot lan co the la trung
    hop (nhac lai, tu xung trong gioi thieu), nen mac dinh can >= 2 lan.

    Nhan kich ban bi XOA truoc khi dem: "ADULT SHELDON: I was thrilled" la
    bang chung nguoi noi LA Adult Sheldon (script_label_anchors dung no theo
    chieu do); dem nhan nhu mot lan "nhac ten" se loai dung cai ten dung —
    hai loai bang chung tu mau thuan nhau tren cung mot dong.
    """
    patterns = [(n, _name_pattern(n)) for n in valid_names]
    counts: dict[str, dict[str, int]] = {}
    for c in chunks:
        tag = real_tag(c)
        text = _SCRIPT_LABEL.sub(" ", c.get("english") or "")
        if not tag or not text:
            continue
        for name, pat in patterns:
            if pat.search(text):
                counts.setdefault(tag, {})
                counts[tag][name] = counts[tag].get(name, 0) + 1
    return {
        tag: {n for n, k in per.items() if k >= min_mentions}
        for tag, per in counts.items()
        if any(k >= min_mentions for k in per.values())
    }


def vocative_hints(
    chunks: list[dict], valid_names: list[str]
) -> dict[str, dict[str, int]]:
    """cluster -> {ten: so phieu}, tu luot thoai lien ke.

    "Sheldon, con co sao khong?" roi cluster khac noi ngay sau => cluster do
    nhieu kha nang la Sheldon. Chi tinh khi cluster doi khac: cung cluster noi
    tiep thi khong phai doi luot.
    """
    patterns = [(n, _vocative_pattern(n)) for n in valid_names]
    votes: dict[str, dict[str, int]] = {}
    for i in range(len(chunks) - 1):
        cur, nxt = chunks[i], chunks[i + 1]
        cur_tag, nxt_tag = real_tag(cur), real_tag(nxt)
        if not nxt_tag or cur_tag == nxt_tag:
            continue
        text = (cur.get("english") or "").strip()
        for name, pat in patterns:
            if pat.search(text):
                votes.setdefault(nxt_tag, {})
                votes[nxt_tag][name] = votes[nxt_tag].get(name, 0) + 1
    return votes


def apply_evidence(
    mapping: dict[str, str],
    anchors: dict[str, str],
    exclusions: dict[str, set[str]],
) -> tuple[dict[str, str], int, int]:
    """Ep ket qua LLM ve khop bang chung. Tra (mapping, so_sua, so_bo).

    Thu tu uu tien ro rang: anchor (nhan kich ban) > exclusion > LLM.
    Cluster co anchor thi anchor la dap an, exclusion khong duoc dung vao —
    truoc day exclusion "bo" ten trung anchor roi anchor lai lap tu lai o
    cuoi, ket qua dung nhung bao cao dem sai (dropped ao).
    """
    out: dict[str, str] = {}
    n_fixed = n_dropped = 0
    for tag, name in mapping.items():
        anchor = anchors.get(tag)
        if anchor is not None:
            out[tag] = anchor
            if anchor != name:
                n_fixed += 1
        elif name in exclusions.get(tag, ()):
            n_dropped += 1          # tu goi ten minh nhieu lan -> khong phai
        else:
            out[tag] = name
    for tag, name in anchors.items():
        out.setdefault(tag, name)
    return out, n_fixed, n_dropped
