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

# Noise mo dau truoc nhan: chi dan san khau "(school bell rings)", "[door
# slams]", not nhac, quote, dash. Nhan dung sau nhung thu nay van la nhan mo
# dau chunk (movie_045 co chunk mo dau bang "(school bell rings)").
_LEADING_NOISE = re.compile(r"^(?:\s|\([^)]*\)|\[[^\]]*\]|[\"'‘’“”]|[-–—]|♪)+")


def iter_script_labels(text: str):
    """Moi match nhan kich ban trong text — MOT nguon regex duy nhat, dung
    chung boi anchor (o day) va override cap dong (SPEAKER/label_override.py)."""
    return _SCRIPT_LABEL.finditer(text)


def label_at_text_start(text: str, match: re.Match) -> bool:
    """Nhan nam o dau text (sau noise mo dau) — bang chung cho CA chunk.

    Nhan giua text chi noi ve phan DUOI chunk (ASR chunk vat qua doi speaker):
    "Asked and answered. MISSY: Did you cry..." nghia la nua sau la Missy,
    KHONG phai ca cluster la Missy.
    """
    return _LEADING_NOISE.sub("", text[:match.start()]).strip() == ""


def strip_script_labels(text: str) -> str:
    """Xoa nhan kich ban khoi text — dung truoc khi dem mention hoac tim
    vocative, vi nhan "MOM:" nghia la nguoi noi LA me, khong phai dang goi me."""
    return _SCRIPT_LABEL.sub(" ", text)


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
    chunks: list[dict], valid_names: list[str],
    mid_anchor_votes: int | None = None,
) -> dict[str, str]:
    """cluster -> ten, tu nhan kich ban MO DAU chunk cua cluster do.

    Chi nhan o dau text (sau noise mo dau) moi anchor ca cluster. Nhan giua
    text bi loai khoi anchor: no danh dau doi speaker BEN TRONG chunk, tuc chi
    la bang chung ve phan duoi — do duoc (movie_045, 2026-08-17) mot nhan
    "MISSY:" giua chunk da dat ten sai ca cluster SPK_106 (65 chunk Sheldon
    giang vat ly), keo 60 dong SRT theo. Phan duoi do duoc xu ly o cap dong
    boi SPEAKER/label_override.py.

    mid_anchor_votes (V3.1): None = nhan giua text khong bao gio anchor (V3).
    So k = nhan giua text van anchor ca cluster neu CUNG MOT ten xuat hien o
    >= k chunk khac nhau cua cluster — mot phieu don le tren cluster lon la
    bang chung yeu (SPK_106: 1 phieu "MISSY"/65 chunk dat ten sai ca cluster),
    nhung ten lap lai nhieu chunk kho long la tinh co (V3 do duoc viec bo het
    anchor giua-chunk lam 008/046 mat ca tag DUNG: −0,0196/−0,0364 F1).

    Cluster co hai ten ung vien khac nhau -> bo, khong chon bua.
    """
    canon = _canonical(valid_names)
    votes: dict[str, set[str]] = {}
    mid_counts: dict[str, dict[str, int]] = {}
    for c in chunks:
        tag = real_tag(c)
        if not tag:
            continue
        text = c.get("english") or ""
        seen_mid: set[str] = set()   # dem theo chunk, khong theo lan xuat hien
        for m in iter_script_labels(text):
            real = canon.get(m.group(1).strip().casefold())
            if not real:
                continue
            if label_at_text_start(text, m):
                votes.setdefault(tag, set()).add(real)
            elif mid_anchor_votes is not None and real not in seen_mid:
                seen_mid.add(real)
                d = mid_counts.setdefault(tag, {})
                d[real] = d.get(real, 0) + 1
    out: dict[str, str] = {}
    for tag in set(votes) | set(mid_counts):
        cands = set(votes.get(tag, ()))
        if mid_anchor_votes is not None:
            cands |= {n for n, k in mid_counts.get(tag, {}).items()
                      if k >= mid_anchor_votes}
        if len(cands) == 1:
            out[tag] = next(iter(cands))
    return out


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
        text = strip_script_labels(c.get("english") or "")
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
