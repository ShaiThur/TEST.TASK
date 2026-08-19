from typing import List, Set, Tuple


def merge_candidate_indices(
    *ranked_lists: List[Tuple[int, float]], minimum: int, context: str = ""
) -> Set[int]:
    indices: Set[int] = set()
    for ranked in ranked_lists:
        indices.update(idx for idx, _ in ranked)
    assert len(indices) >= minimum, (
        f"Недостаточно кандидатов{f' для {context}' if context else ''}: "
        f"{len(indices)} < {minimum}. Увеличьте top_n у ретриверов."
    )
    return indices
