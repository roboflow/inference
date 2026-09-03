import json
import re
from typing import List, Optional, Tuple, Union

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json([\s\S]*?)```", flags=re.IGNORECASE)
ANY_MARKDOWN_BLOCK_PATTERN = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\r?\n?([\s\S]*?)```")
FENCE_LINE_PATTERN = re.compile(r"^[ \t]*```[a-zA-Z0-9_+-]*[ \t]*$", flags=re.MULTILINE)
OBJECT_SEQUENCE_SEPARATOR_PATTERN = re.compile(r"^[\s,]*$")

JsonPayload = Union[dict, list]


def extract_json_payload(raw: Optional[str]) -> Tuple[bool, JsonPayload]:
    """Lenient JSON extraction from VLM output, returning (error_status, payload).

    Tries, in order: the first ```json block, the first fenced block with any
    tag, the text with stray fence lines removed, a sequence of top-level
    objects (JSON Lines / `{...}, {...}`), and the outermost [...] or {...}
    substring.
    """
    if not isinstance(raw, str):
        return True, {}
    for candidate in _candidate_texts(raw):
        payload = _try_parse_json(candidate)
        if payload is not None:
            return False, payload
    stripped = FENCE_LINE_PATTERN.sub("", raw)
    sequence = _parse_object_sequence(stripped)
    if sequence is not None:
        return False, sequence
    for opening, closing in (("[", "]"), ("{", "}")):
        payload = _parse_outermost(stripped, opening=opening, closing=closing)
        if payload is not None:
            return False, payload
    return True, {}


def _candidate_texts(raw: str) -> List[str]:
    candidates: List[str] = []
    json_blocks = JSON_MARKDOWN_BLOCK_PATTERN.findall(raw)
    if json_blocks:
        candidates.append(json_blocks[0])
    any_blocks = ANY_MARKDOWN_BLOCK_PATTERN.findall(raw)
    if any_blocks:
        candidates.append(any_blocks[0])
    candidates.append(raw)
    candidates.append(FENCE_LINE_PATTERN.sub("", raw))
    return list(dict.fromkeys(candidates))


def _try_parse_json(content: str) -> Optional[JsonPayload]:
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def _parse_object_sequence(content: str) -> Optional[list]:
    # Only succeeds when the whole text is objects separated by whitespace or
    # commas, so garbage around fragments never becomes an empty success.
    decoder = json.JSONDecoder()
    entries: List[dict] = []
    index = 0
    while index < len(content):
        start = content.find("{", index)
        if start == -1:
            break
        if not OBJECT_SEQUENCE_SEPARATOR_PATTERN.match(content[index:start]):
            return None
        try:
            entry, end = decoder.raw_decode(content, start)
        except json.JSONDecodeError:
            return None
        if not isinstance(entry, dict):
            return None
        entries.append(entry)
        index = end
    if not entries or not OBJECT_SEQUENCE_SEPARATOR_PATTERN.match(content[index:]):
        return None
    return entries


def _parse_outermost(content: str, opening: str, closing: str) -> Optional[JsonPayload]:
    start = content.find(opening)
    stop = content.rfind(closing)
    if start == -1 or stop <= start:
        return None
    return _try_parse_json(content[start : stop + 1])
