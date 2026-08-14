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
