import re
from typing import Dict, List

from ..data.knowledge import TnvedKnowledge

_TOKEN_RE = re.compile(r"[а-яё0-9a-z]+")


def preprocess_text(text: str) -> str:
    return " ".join(text.lower().split())


def tokenize_for_bm25(normalized_text: str) -> List[str]:
    return _TOKEN_RE.findall(normalized_text)


def build_regulation_search_text(
    reg: Dict, knowledge: TnvedKnowledge, group_notes_char_limit: int
) -> str:
    parts = []
    if reg.get("description"):
        parts.append(reg["description"])
    if reg.get("notes"):
        parts.append(reg["notes"])
    if reg.get("explanation"):
        parts.append(reg["explanation"])

    code = reg.get("code", "")
    if code:
        hierarchy = knowledge.hierarchy_text(code)
        if hierarchy:
            parts.append(f"Иерархия ТН ВЭД: {hierarchy}")

        group_num, notes_text = knowledge.group_notes_text(code, group_notes_char_limit)
        if notes_text:
            parts.append(f"Примечания к группе {group_num}: {notes_text}")

    return " ".join(parts)


def build_declaration_query_text(decl: Dict) -> str:
    parts = []
    if decl.get("G31_1"):
        parts.append(decl["G31_1"])
    if decl.get("desc_extention"):
        parts.append(decl["desc_extention"])
    return " ".join(parts)
