from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int = 32,
        max_length: int = 512,
        use_fp16: bool = False,
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(device)
        if use_fp16 and device == "cuda":
            self._model.half()
        self._model.eval()
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length

    @torch.inference_mode()
    def rerank(self, query: str, candidates: List[Tuple[int, str]]) -> List[Tuple[int, float]]:
        scores: List[float] = []
        for start in range(0, len(candidates), self._batch_size):
            chunk = candidates[start:start + self._batch_size]
            pairs = [(query, text) for _, text in chunk]
            inputs = self._tokenizer(
                pairs, padding=True, truncation=True, max_length=self._max_length, return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            logits = self._model(**inputs).logits
            chunk_scores = torch.sigmoid(logits).squeeze(-1).float().cpu().numpy()
            scores.extend(np.atleast_1d(chunk_scores).tolist())

        result = [(candidates[i][0], scores[i]) for i in range(len(candidates))]
        result.sort(key=lambda x: x[1], reverse=True)
        return result
