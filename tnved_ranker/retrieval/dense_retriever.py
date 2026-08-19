from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, model_name: str, device: str, batch_size: int = 64):
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._corpus_embeddings: Optional[np.ndarray] = None

    def index_corpus(self, normalized_texts: List[str]) -> None:
        self._corpus_embeddings = self._model.encode(
            [f"passage: {t}" for t in normalized_texts],
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def similarity_matrix(self, normalized_queries: List[str]) -> np.ndarray:
        query_embeddings = self._model.encode(
            [f"query: {q}" for q in normalized_queries],
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return query_embeddings @ self._corpus_embeddings.T

    @staticmethod
    def top_n(similarity_row: np.ndarray, n: int) -> List[Tuple[int, float]]:
        top_idx = np.argsort(similarity_row)[::-1][:n]
        return [(int(i), float(similarity_row[i])) for i in top_idx]
