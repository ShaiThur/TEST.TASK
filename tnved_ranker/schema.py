from dataclasses import dataclass


@dataclass(frozen=True)
class Regulation:
    index: int
    regulation_id: str
    code: str
    search_text: str
    normalized_text: str


@dataclass(frozen=True)
class Declaration:
    declaration_id: str
    query_text: str
    normalized_query: str


@dataclass(frozen=True)
class PredictionRow:
    declaration_id: str
    rank: int
    regulation_id: str
    score: float
