import json
import re
from typing import List, Optional, Tuple, Union

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json([\s\S]*?)```", flags=re.IGNORECASE)
ANY_MARKDOWN_BLOCK_PATTERN = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\r?\n?([\s\S]*?)```")
FENCE_LINE_PATTERN = re.compile(r"^[ \t]*```[a-zA-Z0-9_+-]*[ \t]*$", flags=re.MULTILINE)
SEQUENCE_SEPARATOR_PATTERN = re.compile(r"^[\s,]*$")
SEQUENCE_TAIL_PATTERN = re.compile(r"^[\s,]*\]?[\s,]*$")
CLASS_ENTRY_LABEL_KEYS = ("class", "class_name", "label")

JsonPayload = Union[dict, list]


def extract_json_payload(raw: Optional[str]) -> Tuple[bool, JsonPayload]:
    """Lenient JSON extraction from VLM output, returning (error_status, payload).

    Tries, in order: the first ```json block, the first fenced block with any
    tag, the text with stray fence lines removed, a sequence of top-level
    values (JSON Lines / `{...}, {...}` / `[...]\\n[...]`, tolerating one
    stray trailing `]`), and the outermost [...] or {...} substring.
    """
    if not isinstance(raw, str):
        return True, {}
    for candidate in _candidate_texts(raw):
        payload = _try_parse_json(candidate)
        if payload is not None:
            return False, payload
    stripped = FENCE_LINE_PATTERN.sub("", raw)
    sequence = _parse_value_sequence(stripped)
    if sequence is not None:
        return False, sequence
    for opening, closing in (("[", "]"), ("{", "}")):
        payload = _parse_outermost(stripped, opening=opening, closing=closing)
        if payload is not None:
            return False, payload
    return True, {}


def coerce_classification_payload(payload: JsonPayload) -> Optional[dict]:
    """Return a dict payload for the classifier, or None when the shape is unusable.

    A bare list of ``{"class": ..., "confidence": ...}`` entries (the
    ``predicted_classes`` array emitted without its wrapper) is wrapped into
    the multi-label shape; ``class_name`` / ``label`` are accepted as aliases.
    """
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, list) or not payload:
        return None
    entries = []
    for entry in payload:
        label = _get_class_entry_label(entry)
        if label is None:
            return None
        entries.append({"class": label, "confidence": entry.get("confidence", 1.0)})
    return {"predicted_classes": entries}


def _get_class_entry_label(entry: object) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    for key in CLASS_ENTRY_LABEL_KEYS:
        if entry.get(key) is not None:
            return str(entry[key])
    return None


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


def _parse_value_sequence(content: str) -> Optional[list]:
    # Only succeeds when the whole text is objects / arrays separated by
    # whitespace or commas (plus at most one stray closing bracket at the
    # end), so garbage around fragments never becomes an empty success.
    decoder = json.JSONDecoder()
    entries: list = []
    values_found = 0
    index = 0
    while index < len(content):
        start = _find_value_start(content, index)
        if start == -1:
            break
        if not SEQUENCE_SEPARATOR_PATTERN.match(content[index:start]):
            return None
        try:
            value, end = decoder.raw_decode(content, start)
        except json.JSONDecodeError:
            return None
        if isinstance(value, dict):
            entries.append(value)
        elif isinstance(value, list):
            entries.extend(value)
        else:
            return None
        values_found += 1
        index = end
    if not values_found or not SEQUENCE_TAIL_PATTERN.match(content[index:]):
        return None
    return entries


def _find_value_start(content: str, index: int) -> int:
    positions = [
        p for p in (content.find("{", index), content.find("[", index)) if p != -1
    ]
    return min(positions) if positions else -1


def _parse_outermost(content: str, opening: str, closing: str) -> Optional[JsonPayload]:
    start = content.find(opening)
    stop = content.rfind(closing)
    if start == -1 or stop <= start:
        return None
    return _try_parse_json(content[start : stop + 1])
