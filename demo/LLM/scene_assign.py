"""Gan dong phu de vao scene theo midpoint — GPU-free.

Tach tu LLM/refine_llm.py (import unsloth cap module nen khong import duoc
trong moi truong khong GPU) de EVAL/registry_activation.py dung CHUNG mot
logic gan scene voi refine that: du doan kich hoat offline phai khop dong log
"Per-scene registry rendered N/M" cua lan chay GPU, hai ban copy se troi dat.
"""
from __future__ import annotations


def find_best_scene(midpoint, scenes):
    """Strict scene matching by midpoint timestamp (no nearest neighbor fallback)."""
    for idx, sc in enumerate(scenes):
        s, e = sc.get('start_time'), sc.get('end_time')
        if s is None or e is None:
            continue
        if s <= midpoint <= e:
            return idx
    return -1


def scene_line_indices(vlm_scenes, en_subs) -> list[list[int]]:
    """Moi scene mot list index dong SRT co midpoint roi vao scene do."""
    out: list[list[int]] = [[] for _ in vlm_scenes]
    for i, sub in enumerate(en_subs):
        mid = (sub.start.total_seconds() + sub.end.total_seconds()) / 2.0
        si = find_best_scene(mid, vlm_scenes)
        if si != -1:
            out[si].append(i)
    return out


def split_chunk(indices: list[int], cap: int) -> list[list[int]]:
    """Chia index cua mot scene thanh cac manh lien tiep <= cap dong.

    Data train v4 toi da 32 dong/sample (p50=21); scene dai hon lam adapter
    bo cuoc (docs/eval/degeneration_e5.md). Chia can bang (ceil) de khong
    sinh manh duoi ti hon. cap <= 0: tat, giu nguyen scene.
    """
    if cap <= 0 or len(indices) <= cap:
        return [indices]
    k = -(-len(indices) // cap)
    base, extra = divmod(len(indices), k)
    pieces, pos = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        pieces.append(indices[pos:pos + size])
        pos += size
    return pieces
