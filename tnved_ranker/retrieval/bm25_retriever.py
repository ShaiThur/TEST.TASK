from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from ..text.text_processing import tokenize_for_bm25


class BM25Retriever:
    def __init__(self, normalized_corpus: List[str]):
        self._index = BM25Okapi([tokenize_for_bm25(t) for t in normalized_corpus])

    def top_n(self, normalized_query: str, n: int) -> List[Tuple[int, float]]:
        scores = self._index.get_scores(tokenize_for_bm25(normalized_query))
        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_idx]
