"""Map cluster CAM++ (SPK_029) sang ten nhan vat that, bang LLM.

"SPK_029" vo nghia voi model refine — no can "Sheldon". Cluster da gom san cac
dong cua cung mot giong, nen chi can doan MOT lan cho ca cluster thay vi doan
tung dong; cau ngan ("Me.", "You know.") duoc cuu nho cac dong dai cung cluster
(Huh & Zisserman).

Chong bia ten: LLM chi duoc chon trong TAP DONG la nhan vat cua registry, hoac
tra "unknown". Ten ngoai tap do bi ep ve unknown — cung tinh than evidence-gate
cua CHARACTER/registry_schema.py, va vi tag sai con te hon khong tag (xem
docs/superpowers/plans/2026-08-16-speaker-attribution-v1.md).

Orchestrator nhan generate_fn(system, user) -> str de test khong can GPU, giong
CHARACTER/build_registry.extract_registry.
"""
from __future__ import annotations

import json
import re

UNKNOWN = "unknown"

# Prompt phai ngan: Gemma-3 qua unsloth suy giam dan khi prompt vuot ~1200
# token (xem CHARACTER/build_registry.py). Do duoc tren movie_045: prompt dai
# lam model bo qua ca bang chung hien nhien — cluster co san nhan "GEORGE JR.:"
# trong text van bi tra unknown. Moi luat duoi day deu doi mot lan mapping sai
# that, khong phai luat phong xa.
MAPPING_SYSTEM = (
    "You are an expert film-script analyst. Each cluster below holds dialogue "
    "lines spoken by the SAME voice. Decide which named character each cluster "
    "is.\n"
    "Rules:\n"
    "- Choose ONLY from the provided character list. Never invent a name.\n"
    '- A script label inside a line ("MEEMAW: There you go") names that '
    "cluster's speaker. This is the strongest evidence; trust it above all.\n"
    "- Any OTHER name inside a line is who is being addressed or discussed, "
    'NOT the speaker: a cluster saying "Adrian, I can\'t believe you" is '
    "whoever talks TO Adrian.\n"
    "- Clusters show how many subtitle lines they cover; a one-line cluster is "
    "a bit part, usually best left unknown.\n"
    '- Narration / voice-over is not dialogue: answer "unknown" unless the '
    "narrator is in the list.\n"
    "- Several clusters may map to the SAME character. That is expected.\n"
    '- No clear evidence -> "unknown". A wrong name is worse than "unknown".\n'
    "- Output STRICT JSON only, compact. No markdown fences, no commentary."
)

# Cung nguong da do o CHARACTER/build_registry.py.
_PROMPT_TOKEN_BUDGET = 1000
# Mot dong thoai dai bat thuong (ASR gop nhieu cau) an het budget cua ca lo.
_MAX_LINE_CHARS = 160


def build_mapping_prompt(
    cluster_lines: dict[str, list[str]],
    character_names: list[str],
    captions_text: str = "",
    max_lines_per_cluster: int = 8,
    line_counts: dict[str, int] | None = None,
    hints: dict[str, dict[str, int]] | None = None,
) -> tuple[str, str]:
    blocks = []
    for tag in sorted(cluster_lines):
        lines = cluster_lines[tag][:max_lines_per_cluster]
        body = "\n".join(f"  - {ln[:_MAX_LINE_CHARS]}" for ln in lines)
        n = (line_counts or {}).get(tag)
        head = f"{tag} ({n} subtitle lines)" if n else tag
        vote = (hints or {}).get(tag)
        if vote:
            best = sorted(vote.items(), key=lambda kv: -kv[1])[:2]
            called = ", ".join(f'"{n_}" (x{k})' for n_, k in best)
            body = (f"  someone called out {called} right before this cluster "
                    f"speaks\n{body}")
        blocks.append(f"{head}:\n{body}")
    names = ", ".join(f'"{n}"' for n in character_names)
    example = {"SPK_001": character_names[0] if character_names else UNKNOWN,
               "SPK_002": UNKNOWN}
    user = (
        "<Visual Scene Captions>\n"
        f"{captions_text or 'None'}\n"
        "</Visual Scene Captions>\n\n"
        f"<Characters in this film>\n{names or 'None'}\n</Characters>\n\n"
        "<Speaker clusters>\n"
        + "\n\n".join(blocks)
        + "\n</Speaker clusters>\n\n"
        'Return JSON mapping every cluster id to one character name from the '
        f'list above, or "{UNKNOWN}". Example shape:\n'
        f"{json.dumps(example, ensure_ascii=False, separators=(',', ':'))}"
    )
    return MAPPING_SYSTEM, user


def parse_mapping(text: str, valid_names: list[str]) -> dict[str, str]:
    """Doc JSON tho, ep ve tap dong ten hop le. Ten la -> bo (khong giu unknown
    trong ket qua: khong co key nghia la khong tag, don gian hon cho caller)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}

    canonical = {n.casefold(): n for n in valid_names}
    out: dict[str, str] = {}
    for tag, name in raw.items():
        if not isinstance(tag, str) or not isinstance(name, str):
            continue
        real = canonical.get(name.strip().casefold())
        if real is not None:
            out[tag.strip()] = real
    return out


def plan_batches(
    tags: list[str],
    cluster_lines: dict[str, list[str]],
    character_names: list[str],
    captions_text: str,
    count_tokens,
    line_counts: dict[str, int],
    token_budget: int = _PROMPT_TOKEN_BUDGET,
    max_batch: int = 12,
    hints: dict[str, dict[str, int]] | None = None,
) -> list[list[str]]:
    """Gom cluster thanh lo sao cho MOI prompt <= token_budget.

    Do bang tokenizer that thay vi doan theo so cluster: do dai dong thoai
    chenh nhau rat nhieu giua cac phim.
    """
    batches: list[list[str]] = []
    cur: list[str] = []
    for tag in tags:
        trial = cur + [tag]
        system, user = build_mapping_prompt(
            {t: cluster_lines[t] for t in trial}, character_names,
            captions_text, line_counts=line_counts, hints=hints,
        )
        over = count_tokens(system, user) > token_budget
        if cur and (over or len(trial) > max_batch):
            batches.append(cur)
            cur = [tag]
        else:
            cur = trial
    if cur:
        batches.append(cur)
    return batches


def map_clusters(
    cluster_lines: dict[str, list[str]],
    character_names: list[str],
    generate_fn,
    captions_text: str = "",
    batch_size: int = 12,
    line_counts: dict[str, int] | None = None,
    count_tokens=None,
    token_budget: int = _PROMPT_TOKEN_BUDGET,
    hints: dict[str, dict[str, int]] | None = None,
) -> dict[str, str]:
    """Chia cluster thanh lo nho de prompt khong qua dai, roi gop ket qua.

    count_tokens(system, user) -> int: neu co thi chia lo theo token that
    (xem plan_batches); khong co thi chia deu theo batch_size — du cho test,
    KHONG du cho GPU that, vi prompt vuot ~1200 token la Gemma-3 bo qua ca
    bang chung hien nhien (do duoc tren movie_045, xem MAPPING_SYSTEM).

    Cluster nhieu dong di truoc: chung la nhan vat chinh, co nhieu bang chung
    nhat, va la thu dang tra gia token nhat.
    """
    if not character_names or not cluster_lines:
        return {}

    counts = line_counts or {}
    tags = sorted(cluster_lines, key=lambda t: (-counts.get(t, 0), t))
    if count_tokens is not None:
        batches = plan_batches(tags, cluster_lines, character_names,
                               captions_text, count_tokens, counts,
                               token_budget, batch_size, hints)
    else:
        batches = [tags[i:i + batch_size]
                   for i in range(0, len(tags), batch_size)]

    mapping: dict[str, str] = {}
    for i, batch_tags in enumerate(batches, start=1):
        batch = {t: cluster_lines[t] for t in batch_tags}
        system, user = build_mapping_prompt(batch, character_names,
                                            captions_text, line_counts=counts,
                                            hints=hints)
        decoded = generate_fn(system, user)
        part = parse_mapping(decoded, character_names)
        # Chi nhan tag thuoc lo nay: chan LLM che them cluster khong ton tai.
        for tag, name in part.items():
            if tag in batch:
                mapping[tag] = name
        print(f"[speaker] cluster batch {i}/{len(batches)}: "
              f"{len(part)}/{len(batch)} mapped", flush=True)
    return mapping


def tags_to_names(
    line_tags: list[str | None], mapping: dict[str, str]
) -> list[str | None]:
    """Doi speaker_tag tung dong thanh ten nhan vat; cluster khong map -> None."""
    return [mapping.get(t) if t else None for t in line_tags]
