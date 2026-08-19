from LLM.output_guard import is_degenerate_line


def test_detects_repeated_bigram_collapse():
    # failure mode that that poisoned movie_015 baseline (adapter degeneration):
    # 1 n-gram lap hang tram lan trong 1 dong
    line = "Thằng nhóc, " * 50
    assert is_degenerate_line(line.strip())


def test_detects_repeated_single_token_collapse():
    # artifact NMT quan sat tren movie_009 rough: chuoi "- " lap dai
    line = "Nếu Rosa Parks... " + "- " * 60
    assert is_degenerate_line(line.strip())


def test_detects_absurdly_long_line():
    assert is_degenerate_line("xin chào " * 100)


def test_keeps_normal_subtitle_line():
    assert not is_degenerate_line("Bà ơi, cháu xin lỗi vì đã đến muộn.")


def test_keeps_short_legit_repetition():
    # lap ngan la thoai binh thuong, khong duoc coi la suy bien
    assert not is_degenerate_line("Không, không, không, không!")


def test_keeps_long_but_diverse_line():
    line = ("Hôm nay chúng ta sẽ nói về kế hoạch mở rộng cửa hàng sang "
            "khu phố mới, bao gồm ngân sách, nhân sự, và lịch trình từng "
            "giai đoạn cụ thể cho quý sau.")
    assert not is_degenerate_line(line)


def test_detects_dash_only_line():
    # scene sap nguyen khoi thanh cac dong "- -" (eval_e5 movie_336, 39.9% dong)
    for bad in ("-", "- -", "--", "."):
        assert is_degenerate_line(bad)


def test_detects_non_latin_leakage():
    # token Bengali leak o movie_312 dong [0]
    assert is_degenerate_line("SorryCode:bill_of_materials \u09aa\u09cd\u09b0")


def test_detects_more_non_latin_scripts():
    # cac he chu bo sung sau review: Thai, Hy Lap, Hebrew, fullwidth
    assert is_degenerate_line("\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35")   # Thai
    assert is_degenerate_line("\u03b1\u03b2\u03b3 test")                   # Hy Lap
    assert is_degenerate_line("\u05e9\u05dc\u05d5\u05dd")                 # Hebrew
    assert is_degenerate_line("\uff2f\uff2b nha")                           # fullwidth OK


def test_keeps_line_with_leading_dialogue_dash():
    # gach dau thoai hop le khong duoc coi la suy bien
    assert not is_degenerate_line("- Sao ông lại ở đây?")
