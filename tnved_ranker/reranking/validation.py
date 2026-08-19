from typing import Iterable

import numpy as np
import pandas as pd


def validate_predictions(df: pd.DataFrame, declaration_ids: Iterable[str], top_n: int) -> None:
    declaration_ids = list(declaration_ids)
    expected_rows = len(declaration_ids) * top_n
    assert len(df) == expected_rows, f"Ожидалось {expected_rows} строк, получено {len(df)}"

    for decl_id, sub in df.groupby("declaration_id"):
        assert len(sub) == top_n, f"Для {decl_id} не {top_n} строк"
        assert sub["regulation_id"].nunique() == top_n, f"Повторные регуляции для {decl_id}"
        assert set(sub["rank"]) == set(range(1, top_n + 1)), f"Некорректные ранги для {decl_id}"
        assert np.isfinite(sub["score"]).all(), f"Нечисловой score для {decl_id}"
