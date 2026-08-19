from dataclasses import dataclass, field

import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Config:
    bm25_top_n: int = 40
    dense_top_n: int = 40
    final_top_n: int = 10

    dense_model_name: str = "intfloat/multilingual-e5-base"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    dense_batch_size: int = 64
    rerank_batch_size: int = 32
    max_seq_length: int = 1024

    group_notes_char_limit: int = 400

    device: str = field(default_factory=_default_device)

    @property
    def use_fp16(self) -> bool:
        return self.device == "cuda"
