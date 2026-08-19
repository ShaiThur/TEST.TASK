#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Точка входа: python run.py --data ./data --out ./out"""
import argparse
import logging

from tnved_ranker.config import Config
from tnved_ranker.pipeline import RankingPipeline

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ранжирование регуляций ТН ВЭД по тексту декларации")
    parser.add_argument("--data", default="./data", help="Путь к каталогу с данными")
    parser.add_argument("--out", default="./out", help="Путь для сохранения результата")
    args = parser.parse_args()

    config = Config()
    logger.info("Устройство вычислений: %s (fp16=%s)", config.device, config.use_fp16)

    pipeline = RankingPipeline(config)
    pipeline.run(args.data, args.out)


if __name__ == "__main__":
    main()
