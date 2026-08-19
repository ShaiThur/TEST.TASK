import pandas as pd
import pytest

from tnved_ranker.reranking.validation import validate_predictions


def _make_df(declaration_id="D1", n=10, duplicate_regulation=False):
    rows = []
    for rank in range(1, n + 1):
        reg_id = "R1" if duplicate_regulation and rank == 2 else f"R{rank}"
        rows.append({"declaration_id": declaration_id, "rank": rank, "regulation_id": reg_id, "score": 1.0})
    return pd.DataFrame(rows)


def test_valid_predictions_pass():
    df = _make_df()
    validate_predictions(df, ["D1"], top_n=10)


def test_duplicate_regulation_fails():
    df = _make_df(duplicate_regulation=True)
    with pytest.raises(AssertionError):
        validate_predictions(df, ["D1"], top_n=10)


def test_wrong_row_count_fails():
    df = _make_df(n=5)
    with pytest.raises(AssertionError):
        validate_predictions(df, ["D1"], top_n=10)
