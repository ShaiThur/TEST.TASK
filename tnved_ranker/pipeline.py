import logging
import os
import time
from typing import List, Optional

from tqdm import tqdm

from .retrieval.bm25_retriever import BM25Retriever
from .retrieval.candidates import merge_candidate_indices
from .config import Config
from .retrieval.dense_retriever import DenseRetriever
from .data.io_utils import load_jsonl, predictions_to_dataframe, write_predictions
from .data.knowledge import TnvedKnowledge
from .reranking.reranker import CrossEncoderReranker
from .schema import Declaration, PredictionRow, Regulation
from .text.text_processing import (
    build_declaration_query_text,
    build_regulation_search_text,
    preprocess_text,
)
from .reranking.validation import validate_predictions

logger = logging.getLogger(__name__)


def _build_regulations(raw_regs: List[dict], knowledge: TnvedKnowledge, config: Config) -> List[Regulation]:
    regulations = []
    for i, reg in enumerate(raw_regs):
        search_text = build_regulation_search_text(reg, knowledge, config.group_notes_char_limit)
        regulations.append(
            Regulation(
                index=i,
                regulation_id=reg["regulation_id"],
                code=reg.get("code", ""),
                search_text=search_text,
                normalized_text=preprocess_text(search_text),
            )
        )
    return regulations


def _build_declarations(raw_decls: List[dict]) -> List[Declaration]:
    declarations = []
    for decl in raw_decls:
        query_text = build_declaration_query_text(decl)
        declarations.append(
            Declaration(
                declaration_id=decl["declaration_id"],
                query_text=query_text,
                normalized_query=preprocess_text(query_text),
            )
        )
    return declarations


class RankingPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.knowledge: Optional[TnvedKnowledge] = None
        self.regulations: List[Regulation] = []
        self.bm25: Optional[BM25Retriever] = None
        self.dense = DenseRetriever(config.dense_model_name, config.device, config.dense_batch_size)
        self.reranker = CrossEncoderReranker(
            config.reranker_model_name,
            config.device,
            config.rerank_batch_size,
            config.max_seq_length,
            config.use_fp16,
        )

    def prepare(self, data_dir: str, raw_decls: List[dict], raw_regs: List[dict]) -> List[Declaration]:
        logger.info("Парсинг базы знаний ТН ВЭД...")
        self.knowledge = TnvedKnowledge.from_file(os.path.join(data_dir, "tnved_knowledge.txt"))
        logger.info("Статистика базы знаний: %s", self.knowledge.stats)

        logger.info("Подготовка текстов регуляций и деклараций...")
        self.regulations = _build_regulations(raw_regs, self.knowledge, self.config)
        declarations = _build_declarations(raw_decls)

        logger.info("Построение BM25-индекса...")
        self.bm25 = BM25Retriever([r.normalized_text for r in self.regulations])

        logger.info("Индексация регуляций dense-моделью (%s)...", self.config.dense_model_name)
        self.dense.index_corpus([r.normalized_text for r in self.regulations])

        return declarations

    def rank_all(self, declarations: List[Declaration]) -> List[PredictionRow]:
        logger.info("Кодирование запросов деклараций dense-моделью...")
        sim_matrix = self.dense.similarity_matrix([d.normalized_query for d in declarations])

        predictions: List[PredictionRow] = []
        for i, decl in enumerate(tqdm(declarations, desc="Ранжирование")):
            bm25_top = self.bm25.top_n(decl.normalized_query, self.config.bm25_top_n)
            dense_top = self.dense.top_n(sim_matrix[i], self.config.dense_top_n)

            candidate_idx = merge_candidate_indices(
                bm25_top, dense_top, minimum=self.config.final_top_n, context=decl.declaration_id
            )
            candidates_for_rerank = [(idx, self.regulations[idx].search_text) for idx in candidate_idx]

            reranked = self.reranker.rerank(decl.query_text, candidates_for_rerank)
            for rank, (reg_idx, score) in enumerate(reranked[: self.config.final_top_n], start=1):
                predictions.append(
                    PredictionRow(
                        declaration_id=decl.declaration_id,
                        rank=rank,
                        regulation_id=self.regulations[reg_idx].regulation_id,
                        score=score,
                    )
                )
        return predictions

    def run(self, data_dir: str, out_dir: str) -> None:
        t0 = time.time()
        raw_decls = load_jsonl(os.path.join(data_dir, "declarations.jsonl"))
        raw_regs = load_jsonl(os.path.join(data_dir, "regulations.jsonl"))
        logger.info("Загружено %d деклараций и %d регуляций.", len(raw_decls), len(raw_regs))

        declarations = self.prepare(data_dir, raw_decls, raw_regs)

        logger.info("Ранжирование...")
        predictions = self.rank_all(declarations)

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "predictions.csv")
        df = predictions_to_dataframe(predictions)
        validate_predictions(df, [d.declaration_id for d in declarations], self.config.final_top_n)
        write_predictions(df, out_path)

        logger.info("Результат сохранён в %s", out_path)
        logger.info("Общее время выполнения: %.1f сек.", time.time() - t0)
