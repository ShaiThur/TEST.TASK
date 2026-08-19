import json
from typing import Dict, List

import pandas as pd

from ..schema import PredictionRow


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def predictions_to_dataframe(rows: List[PredictionRow]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "declaration_id": r.declaration_id,
            "rank": r.rank,
            "regulation_id": r.regulation_id,
            "score": r.score,
        }
        for r in rows
    )


def write_predictions(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False)
