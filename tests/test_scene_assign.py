from LLM.scene_assign import split_chunk


def test_short_chunk_unchanged():
    assert split_chunk([1, 2, 3], 24) == [[1, 2, 3]]


def test_cap_zero_disables_split():
    idx = list(range(50))
    assert split_chunk(idx, 0) == [idx]


def test_long_chunk_balanced_pieces():
    # scene 50 dong, cap 24 -> 3 manh can bang, khong co manh duoi ti hon
    idx = list(range(50))
    pieces = split_chunk(idx, 24)
    assert [len(p) for p in pieces] == [17, 17, 16]
    # giu thu tu va phu kin
    assert [i for p in pieces for i in p] == idx


def test_exact_cap_single_piece():
    idx = list(range(24))
    assert split_chunk(idx, 24) == [idx]
