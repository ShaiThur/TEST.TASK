from tnved_ranker.data.knowledge import TnvedKnowledge

SAMPLE = """
РАЗДЕЛ I
ГРУППА 01
Живые животные
01 | Живые животные
0101 | Лошади, ослы, мулы и лошаки живые:
0101210000 | – – чистопородные племенные животные [Лошади чистопородные племенные, живые]
РАЗДЕЛ II
"""


def _write_sample(tmp_path):
    path = tmp_path / "tnved_knowledge.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    return str(path)


def test_parses_codes_of_different_lengths(tmp_path):
    knowledge = TnvedKnowledge.from_file(_write_sample(tmp_path))
    hierarchy = knowledge.hierarchy_text("0101210000")
    assert "Лошади" in hierarchy or "чистопородные" in hierarchy



def test_unknown_code_returns_empty_hierarchy(tmp_path):
    knowledge = TnvedKnowledge.from_file(_write_sample(tmp_path))
    assert knowledge.hierarchy_text("9999999999") == ""
