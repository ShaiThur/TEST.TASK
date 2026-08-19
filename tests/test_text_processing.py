from tnved_ranker.text.text_processing import preprocess_text, tokenize_for_bm25


def test_preprocess_lowercases_and_collapses_whitespace():
    assert preprocess_text("  Товар   ИЗ   Стали  ") == "товар из стали"


def test_tokenize_strips_punctuation():
    tokens = tokenize_for_bm25("товар, изготовленный из стали.")
    assert tokens == ["товар", "изготовленный", "из", "стали"]
