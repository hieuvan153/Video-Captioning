"""Gan speaker_tag cua CAM++ vao tung dong phu de, theo overlap thoi gian.

Nguon: data/speaker_verify_campp_en_full/.../movie_XXX.tagged.json — moi
VAD-chunk co start/end/english/speaker_tag/speaker_score. Chunk MIN hon dong
SRT (movie_045: 366 chunk / 312 dong) nhung cung truc thoi gian va cung text,
nen gan duoc bang overlap thuan, khong can GPU.

Do tren 5 phim eval (2026-08-16): gan het dong co overlap voi mot chunk nao do,
nhung sau khi loai sentinel "UNKNOWN" cua CAM++ (xem _NON_TAGS) thi ti le dong
co speaker THAT la 70,6-83,7% (008: 226/320, 009: 261/312, 015: 287/343,
045: 250/312, 046: 213/257). 5-26 dong moi phim cham tu 2 speaker tro len ->
phai chon theo TONG overlap lon nhat, khong lay chunk dau tien cham vao.

Nguyen tac chu dao (docs/superpowers/plans/2026-08-16-speaker-attribution-v1.md):
khong chac thi tra None. Tag sai con te hon khong tag — no day model chon dut
khoat mot xung ho sai, trong khi khong tag thi model roi ve generic (sai nhe
hon). Day chinh la failure mode V0 dang muon sua.
"""
from __future__ import annotations

from dataclasses import dataclass

# Duoi nguong nay coi nhu cham nhau do sai so lam tron, khong phai thoai that.
_MIN_OVERLAP_S = 0.05

# CAM++ ghi "UNKNOWN" khi chunk qua ngan / duoi nguong (db_action
# "short_below_threshold"): 75-179 chunk moi phim, tuc 20-35%. Day KHONG phai
# mot cluster — coi no la cluster thi 67-179 dong duoc gom lai va gan CHUNG
# mot ten, dung kieu tag sai hang loat ma V1 phai tranh (do duoc tren
# movie_045 ngay 2026-08-16: 67 dong "UNKNOWN" bi dat ten "Ingram").
_NON_TAGS = frozenset({"unknown", "unk", "none", "n/a", ""})

# Tag tong hop cho vung chunk duoc nhan kich ban dinh danh truc tiep
# (SPEAKER/label_override.py cat chunk tai thoi diem nhan). Dat hang so o day
# de huong import mot chieu: label_override -> align.
LABEL_TAG_PREFIX = "LABEL::"


@dataclass(frozen=True)
class AlignStats:
    n_lines: int
    n_tagged: int
    n_contested: int          # dong co >= 2 speaker khac nhau cung overlap
    n_low_score: int          # CHUNK bi bo vi speaker_score duoi nguong

    @property
    def coverage(self) -> float:
        return self.n_tagged / self.n_lines if self.n_lines else 0.0


def real_tag(chunk: dict) -> str | None:
    """speaker_tag cua chunk, hoac None neu do la sentinel "khong biet"."""
    tag = chunk.get("speaker_tag")
    if not tag or str(tag).strip().casefold() in _NON_TAGS:
        return None
    return str(tag).strip()


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(
    chunks: list[dict],
    spans: list[tuple[float, float]],
    min_score: float | None = None,
) -> tuple[list[str | None], AlignStats]:
    """Tra speaker_tag cho tung span (start, end) tinh bang giay.

    chunks: list dict co 'start', 'end', 'speaker_tag', tuy chon 'speaker_score'.
    min_score: neu dat, chunk co speaker_score thap hon bi bo qua — dung khi
        muon chi tin nhung chunk khop embedding chac (xem rui ro "cluster gop
        nham 2 nguoi giong giong" trong plan V1).

    Chon tag co TONG overlap lon nhat tren span. Hoa thi tra None: khong doan.
    """
    usable = []
    n_low = 0
    for c in chunks:
        tag = real_tag(c)
        if not tag:
            continue
        # Vung LABEL:: mien loc min_score: bang chung cua no la TEXT (nhan
        # kich ban), khong phai do khop embedding — nguong embedding khong co
        # tham quyen xoa no.
        if (min_score is not None
                and not tag.startswith(LABEL_TAG_PREFIX)
                and (c.get("speaker_score") or 0.0) < min_score):
            n_low += 1
            continue
        try:
            start = float(c["start"])
            end = float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            usable.append((start, end, tag))

    out: list[str | None] = []
    n_tagged = n_contested = 0
    for s0, s1 in spans:
        totals: dict[str, float] = {}
        for c0, c1, tag in usable:
            ov = _overlap(s0, s1, c0, c1)
            if ov > _MIN_OVERLAP_S:
                totals[tag] = totals.get(tag, 0.0) + ov
        if not totals:
            out.append(None)
            continue
        if len(totals) > 1:
            n_contested += 1
        ranked = sorted(totals.items(), key=lambda kv: -kv[1])
        # Hoa tuyet doi giua hai speaker -> khong co co so chon, tra None.
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            out.append(None)
            continue
        out.append(ranked[0][0])
        n_tagged += 1

    return out, AlignStats(len(spans), n_tagged, n_contested, n_low)


def spans_from_subs(subs) -> list[tuple[float, float]]:
    """Doi list srt.Subtitle thanh list (start, end) giay."""
    return [
        (s.start.total_seconds(), s.end.total_seconds()) for s in subs
    ]


def representative_lines(
    chunks: list[dict], per_cluster: int = 8, min_words: int = 4
) -> dict[str, list[str]]:
    """Gom vai dong tieu bieu cho moi cluster, de LLM doan ten nhan vat.

    Uu tien speaker_score cao (khop embedding chac nhat) va cau du dai. Cau
    ngan kieu "You know." / "Me." khong du ngu canh de suy ra nguoi noi — theo
    Huh & Zisserman, dua vao cac dong dai cung cluster, vi cluster da gom san.
    Neu cluster CHI co cau ngan thi van lay chung con hon khong co gi.
    """
    by_tag: dict[str, list[dict]] = {}
    for c in chunks:
        tag = real_tag(c)
        text = (c.get("english") or "").strip()
        if tag and text:
            by_tag.setdefault(tag, []).append(c)

    out: dict[str, list[str]] = {}
    for tag, items in by_tag.items():
        items.sort(key=lambda c: -(c.get("speaker_score") or 0.0))
        long_lines = [
            (c.get("english") or "").strip() for c in items
            if len((c.get("english") or "").split()) >= min_words
        ]
        chosen = long_lines[:per_cluster]
        if not chosen:
            chosen = [(c.get("english") or "").strip() for c in items[:per_cluster]]
        out[tag] = chosen
    return out
