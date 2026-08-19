import pytest

from tnved_ranker.retrieval.candidates import merge_candidate_indices


def test_merges_unique_indices_from_multiple_lists():
    bm25_top = [(1, 0.9), (2, 0.5)]
    dense_top = [(2, 0.8), (3, 0.7)]
    result = merge_candidate_indices(bm25_top, dense_top, minimum=2)
    assert result == {1, 2, 3}


def test_raises_when_not_enough_candidates():
    with pytest.raises(AssertionError):
        merge_candidate_indices([(1, 0.9)], minimum=5)
