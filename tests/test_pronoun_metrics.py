from EVAL.pronoun_lexicon import extract_pronouns, parse_gold_pronouns
from EVAL.pronoun_f1 import corpus_pronoun_f1


def test_extract_single_and_multiword():
    # "chúng tôi" phải match nguyên cụm, không tách thành "tôi"
    assert extract_pronouns("Chúng tôi đổi linh kiện.") == ["chúng tôi"]
    assert extract_pronouns("Ông tôi mua nó năm 1957.") == ["ông", "tôi", "nó"]


def test_extract_no_partial_word_match():
    # "bà" không được match bên trong "bàn"
    assert extract_pronouns("Cái bàn này của bà.") == ["bà"]


def test_parse_gold_pronouns_comma_separated():
    record = {"pronouns_subject": "ông, bố", "pronouns_object": "tôi"}
    assert parse_gold_pronouns(record) == ["ông", "bố", "tôi"]


def test_parse_gold_pronouns_null_fields():
    assert parse_gold_pronouns({"pronouns_subject": None, "pronouns_object": None}) == []


def test_corpus_f1_perfect_and_zero():
    perfect = corpus_pronoun_f1([(["tôi"], ["tôi"]), (["bà", "cháu"], ["bà", "cháu"])])
    assert perfect.f1 == 1.0
    zero = corpus_pronoun_f1([(["bà"], ["mẹ"])])
    assert zero.f1 == 0.0
    assert zero.fp == 1 and zero.fn == 1


def test_corpus_f1_partial_multiset():
    # gold có 2 "con", pred chỉ có 1 → tp=1, fn=1
    scores = corpus_pronoun_f1([(["con", "con", "mẹ"], ["con", "mẹ"])])
    assert scores.tp == 2 and scores.fn == 1 and scores.fp == 0
