import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CODE_LINE_RE = re.compile(r'^(\d+)\s*\|\s*(?:–\s*)*([^\[]*?)(?:\s*\[([^\]]*)\])?\s*$')
_GROUP_HEADER_RE = re.compile(r'ГРУППА\s+\d+')
_GROUP_NUM_RE = re.compile(r'ГРУППА\s+(\d+)')


class TnvedKnowledge:
    def __init__(self, code_to_desc: Dict[str, str], group_notes: Dict[str, str]):
        self._code_to_desc = code_to_desc
        self._group_notes = group_notes

    @classmethod
    def from_file(cls, path: str) -> "TnvedKnowledge":
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        code_to_desc = cls._parse_codes(content)
        group_notes = cls._parse_group_notes(content)
        if not code_to_desc:
            logger.warning(
                "Из %s не извлечено ни одного кода — проверьте формат файла "
                "(регулярное выражение в _parse_codes могло не совпасть со структурой).",
                path,
            )
        return cls(code_to_desc, group_notes)

    @staticmethod
    def _parse_codes(content: str) -> Dict[str, str]:
        code_to_desc: Dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            match = _CODE_LINE_RE.match(line)
            if not match:
                continue
            code = match.group(1)
            desc_part = match.group(2).strip()
            bracket_desc = match.group(3)
            desc = bracket_desc if bracket_desc else desc_part
            if desc and code not in code_to_desc:
                code_to_desc[code] = desc
        return code_to_desc

    @staticmethod
    def _parse_group_notes(content: str) -> Dict[str, str]:
        group_notes: Dict[str, str] = {}
        sections = re.split(f"({_GROUP_HEADER_RE.pattern})", content)
        current_group: Optional[str] = None
        current_notes: List[str] = []

        for sec in sections:
            sec = sec.strip()
            if _GROUP_HEADER_RE.match(sec):
                if current_group is not None:
                    group_notes[current_group] = "\n".join(current_notes).strip()
                current_group = _GROUP_NUM_RE.search(sec).group(1)
                current_notes = []
            elif current_group is not None:
                if "РАЗДЕЛ" in sec and not sec.startswith("ГРУППА"):
                    group_notes[current_group] = "\n".join(current_notes).strip()
                    current_group = None
                    current_notes = []
                else:
                    current_notes.append(sec)
        if current_group is not None:
            group_notes[current_group] = "\n".join(current_notes).strip()
        return group_notes

    def hierarchy_text(self, code: str) -> str:
        hierarchy = []
        for length in (10, 8, 6, 4, 2):
            if len(code) >= length:
                prefix = code[:length]
                if prefix in self._code_to_desc:
                    hierarchy.append(self._code_to_desc[prefix])
        return " ".join(reversed(hierarchy))

    def group_notes_text(self, code: str, char_limit: int) -> Tuple[str, str]:
        if len(code) < 2:
            return "", ""
        group_num = code[:2]
        notes = self._group_notes.get(group_num)
        if not notes:
            return "", ""
        return group_num, notes[:char_limit]

    @property
    def stats(self) -> Dict[str, int]:
        return {"codes": len(self._code_to_desc), "groups_with_notes": len(self._group_notes)}
