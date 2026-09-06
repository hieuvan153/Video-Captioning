"""Kiem tra 2 cong cu moi cua Phase 0. Chay: van_env/bin/python demo/EVAL/test_context_tools.py"""
import os
import re
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import srt
from EVAL.pron_anchored import cue_prf
from EVAL.film_ref_anchored import overlap_text


def test_cue_prf_dat_dung_cho_moi_tinh_diem():
    # Cung mot multiset dai tu tren CA PHIM (2 "anh", 2 "em") nhung dat sai cho.
    ref = ["Anh di dau?", "Em o nha."]
    hyp_dung = ["Anh di dau?", "Em o nha."]
    hyp_hoan_vi = ["Em di dau?", "Anh o nha."]

    def f1(rows):
        tp = sum(r[0] for r in rows); fp = sum(r[1] for r in rows); fn = sum(r[2] for r in rows)
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    assert f1(cue_prf(ref, hyp_dung)) == 1.0
    # Metric multiset toan phim se cho 1.0 o day; metric theo cue phai cho 0.0.
    assert f1(cue_prf(ref, hyp_hoan_vi)) == 0.0


def test_cue_prf_dem_boi():
    rows = cue_prf(["anh anh em"], ["anh em em"])
    assert rows == [(2, 1, 1)]   # chung: 1 anh + 1 em = 2; thua 1 em; thieu 1 anh


def test_overlap_text_replication():
    """Khoá bản phát hiện: overlap_text nhan ban text cua cue arm dai sang nhieu cue tham chieu.

    ref = [(0-2s,"Anh à?"), (5-7s,"Em ơi"), (8-10s,"Đi")]
    arm = [(0-10s,"Anh yêu em")]
    -> hyp = ["Anh yêu em", "Anh yêu em", "Anh yêu em"] (nhan ban cho moi cue ref)
    -> cue_prf -> [(1,1,0), (1,1,0), (0,2,0)] (moi cue duoc danh gia rieng, tp va fp lap)
    """
    # Tao ref cues
    ref_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=2), content="Anh à?"),
        srt.Subtitle(index=2, start=timedelta(seconds=5), end=timedelta(seconds=7), content="Em ơi"),
        srt.Subtitle(index=3, start=timedelta(seconds=8), end=timedelta(seconds=10), content="Đi"),
    ]
    # Tao arm cue dai trum ca 3 ref cues (0-10s)
    arm_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=10), content="Anh yêu em"),
    ]

    hyp_texts = overlap_text(ref_subs, arm_subs)
    ref_texts = ["Anh à?", "Em ơi", "Đi"]

    rows = cue_prf(ref_texts, hyp_texts)
    # Du "Anh yêu em" co chat luong tuyet voi, nhung bi nhan ban -> tp va fp deu lap
    assert rows == [(1, 1, 0), (1, 1, 0), (0, 2, 0)]


from EVAL.pron_anchored import assign_scene_ids, bootstrap, bootstrap_cluster


def test_bootstrap_cluster_rong_hon_bootstrap_cue_khi_cue_tuong_quan_theo_canh():
    """Finding 1 (review tong 2026-09-06): ngu canh oracle gan theo CANH nen moi cue
    trong cung canh tuong quan voi nhau -> bootstrap theo CUE doc lap danh gia thap n
    hieu dung, CI hep hon thuc te. Du lieu gia: 10 "canh", moi canh 20 cue GIONG HET
    NHAU trong canh (tuong quan hoan hao noi bo canh), nhung 5 canh nghieng het ve arm
    a va 5 canh nghieng het ve arm b -> bat dinh THAT nam o cap CANH (n hieu dung ~10),
    khong phai o cap cue (n=200). CI cum canh phai RONG HON CI theo cue ro ret.
    """
    rows_a, rows_b, cluster_ids = [], [], []
    for scene in range(10):
        # 5 canh dau: a dung het dai tu (tp=1), b sai het (fp=1); 5 canh sau nguoc lai.
        a_row, b_row = ((1, 0, 0), (0, 1, 0)) if scene < 5 else ((0, 1, 0), (1, 0, 0))
        for _ in range(20):  # 20 cue giong het nhau trong cung 1 canh
            rows_a.append(a_row)
            rows_b.append(b_row)
            cluster_ids.append(scene)

    cue_lo, cue_hi = bootstrap(rows_a, rows_b, n=1000, seed=42)
    cl_lo, cl_hi = bootstrap_cluster(rows_a, rows_b, cluster_ids, n=1000, seed=42)

    width_cue = cue_hi - cue_lo
    width_cluster = cl_hi - cl_lo
    assert width_cluster > width_cue, (
        f"CI cum canh ({width_cluster:.4f}) phai rong hon CI theo cue ({width_cue:.4f})")
    # Bien do phai lon ro ret, khong phai chenh lech so lam tron: cluster bootstrap chi
    # co 10 "don vi" doc lap nen phai dao dong manh hon nhieu so voi 200 cue gia doc lap.
    assert width_cluster > width_cue * 2


def test_assign_scene_ids_gan_dung_canh_theo_thoi_gian_giao_nhau():
    """assign_scene_ids phai gan cue vao canh giao thoi gian NHIEU NHAT (cung quy tac
    voi scene_terms), va cue khong giao voi canh nao thi duoc gan id rieng (khong lan
    sang canh khac, khong bi loai am tham)."""
    ref_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="a"),
        srt.Subtitle(index=2, start=timedelta(seconds=11), end=timedelta(seconds=12), content="b"),
        srt.Subtitle(index=3, start=timedelta(seconds=100), end=timedelta(seconds=101), content="c"),
    ]
    scenes = [(0.0, 10.0, "canh_0"), (10.0, 20.0, "canh_1")]
    ids = assign_scene_ids(ref_subs, scenes)
    assert ids[0] == "canh_0"
    assert ids[1] == "canh_1"
    assert ids[2] not in ("canh_0", "canh_1")  # khong giao canh nao -> id rieng


from EVAL.make_oracle_context import oracle_line, replace_rel, scene_terms

CAP = ("1. Summary: A man argues with his sister.\n"
       "2. Main Characters: [Man] - [Male] - [Adult].\n"
       "3. Relationship: [None].")


def test_replace_rel_giu_nguyen_schema():
    out = replace_rel(CAP, "3. Relationship: X")
    assert out.splitlines()[0] == CAP.splitlines()[0]
    assert out.splitlines()[1] == CAP.splitlines()[1]
    assert out.splitlines()[2] == "3. Relationship: X"
    assert len(out.splitlines()) == 3


def test_replace_rel_them_moi_khi_thieu_muc_3():
    out = replace_rel("1. Summary: abc", "3. Relationship: X")
    assert out == "1. Summary: abc\n3. Relationship: X"


def test_oracle_line():
    assert oracle_line([]) == "3. Relationship: [None]."
    assert oracle_line(["anh", "em"]) == (
        "3. Relationship: [Speakers] - address terms used in this scene: anh, em.")


def test_replace_rel_khong_co_relationship():
    """Cap caption co dong 3 nhung khong co chu 'Relationship' - phai thay duoc."""
    cap = "1. A person.\n2. [ID] - [M] - [Adult].\n3. None."
    out = replace_rel(cap, "3. Relationship: X")
    lines = out.splitlines()
    assert len(lines) == 3
    assert lines[0] == "1. A person."
    assert lines[1] == "2. [ID] - [M] - [Adult]."
    assert lines[2] == "3. Relationship: X"


def test_replace_rel_format_khac():
    """Cap caption voi format dong 3 khac."""
    cap = "1. X\n2. Y\n3. [ID 1 & ID 2] - Father and Son."
    out = replace_rel(cap, "3. Relationship: Z")
    lines = out.splitlines()
    assert len(lines) == 3
    assert lines[0] == "1. X"
    assert lines[1] == "2. Y"
    assert lines[2] == "3. Relationship: Z"


def test_replace_rel_caption_rong():
    """Cap caption rong phai tra ve 3 dong voi placeholders."""
    out = replace_rel("", "3. Relationship: X")
    lines = out.splitlines()
    assert len(lines) == 3
    assert lines[0] == "1. Summary: [None]."
    assert lines[1] == "2. Main Characters: [None]."
    assert lines[2] == "3. Relationship: X"

    out2 = replace_rel("   ", "3. Relationship: Y")
    lines2 = out2.splitlines()
    assert len(lines2) == 3
    assert lines2[1] == "2. Main Characters: [None]."


def test_scene_terms_top_zero():
    """scene_terms voi top=0 tra ve tat ca terms, khong cat."""
    # Tao ref subs voi nhieu term phan biet tu lexicon
    ref_subs = [
        srt.Subtitle(index=1, start=timedelta(seconds=0), end=timedelta(seconds=10),
                     content="Tôi, anh, chị, em, bạn, cô, bà, ông, chú, bác nói chuyện"),
    ]
    # top=0 phai tra ve tat ca 10 term phan biet
    terms = scene_terms(ref_subs, 0.0, 10.0, top=0)
    assert len(terms) > 6, f"Expected more than 6 terms with top=0, got {len(terms)}"
    # top=6 phai tra ve chi 6 term
    terms_limited = scene_terms(ref_subs, 0.0, 10.0, top=6)
    assert len(terms_limited) <= 6


def test_replace_rel_khong_khop_indent_items():
    """Khop sau: indented items trong muc 2 (nhu '       3. Woman') khong bi thay."""
    cap = ("1. Summary: The video shows three individuals.\n"
           "2. Main Characters:\n"
           "   - [ID] - [Gender] - [Age Range]\n"
           "     1. Man in bathroom - Male - 30s\n"
           "     2. Man on couch - Male - 20s\n"
           "     3. Woman in kitchen - Female - 30s\n"
           "3. Relationship: None")
    out = replace_rel(cap, "3. Relationship: X")
    # Dung thay la dong '3. Relationship: None' o cuoi, khong phai dong indent '     3. Woman...'
    lines = out.splitlines()
    assert "Woman in kitchen" in out, "Content 'Woman in kitchen' phai con nguyen"
    assert "3. Relationship: X" in out, "Oracle line phai co"
    # Dung co 2 dong bat dau bang '3.' ma khong indent (1 la '3. Relationship: None', 1 la oracle)
    count_3 = sum(1 for line in lines if re.match(r"^3\s*\.", line))
    assert count_3 == 1, f"Chi co 1 dong unindented bat dau bang '3.', khong phai {count_3}"


def test_replace_rel_malformed_no_numbered_sections():
    """Caption khong co dong numbered nao (Summary: instead of 1. Summary:) - wrap toan bo noi dung."""
    cap = "Summary: A woman in a floral dress stands. Main Characters: [None] Relationship: [None]"
    out = replace_rel(cap, "3. Relationship: X")
    lines = out.splitlines()
    assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"
    assert lines[0].startswith("1. Summary:"), f"Line 0 should start with '1. Summary:', got {lines[0][:30]}"
    assert lines[1] == "2. Main Characters: [None].", f"Line 1 should be placeholder"
    assert lines[2] == "3. Relationship: X", f"Line 2 should be oracle"
    assert "woman in a floral dress" in out, "Noi dung goc phai con nguyen"


if __name__ == "__main__":
    test_cue_prf_dat_dung_cho_moi_tinh_diem()
    test_cue_prf_dem_boi()
    test_overlap_text_replication()
    test_replace_rel_giu_nguyen_schema()
    test_replace_rel_them_moi_khi_thieu_muc_3()
    test_oracle_line()
    test_replace_rel_khong_co_relationship()
    test_replace_rel_format_khac()
    test_replace_rel_caption_rong()
    test_scene_terms_top_zero()
    test_replace_rel_khong_khop_indent_items()
    test_replace_rel_malformed_no_numbered_sections()
    test_bootstrap_cluster_rong_hon_bootstrap_cue_khi_cue_tuong_quan_theo_canh()
    test_assign_scene_ids_gan_dung_canh_theo_thoi_gian_giao_nhau()
    print("OK")
