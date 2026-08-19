"""Пакет tnved_ranker: ранжирование регуляций ТН ВЭД по тексту декларации.

Офлайн-режим и random seed применяются здесь, при импорте пакета — до того,
как submodules импортируют transformers/sentence-transformers, которые иначе
могли бы попытаться обратиться в сеть.
"""
import logging
import os

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
