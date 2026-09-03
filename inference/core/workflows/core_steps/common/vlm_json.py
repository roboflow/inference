"""Lenient JSON extraction for VLM / LLM text output.

Every ``vlm_as_*`` formatter block needs to turn a model's free-text answer
into a JSON payload before its model-specific parser runs. Models drift
from the prompted format in a handful of recurring, model-agnostic ways,
and each of them used to be handled by a one-off hook keyed on
``model_type`` (or not at all). Observed in production:

* a bare JSON array followed by a lone closing fence and no opening one
  (Qwen 3.8 Max object detection, ~30% of responses);
* one JSON object per line with no enclosing array (GLM 5.3 Flash);
* ``{...}, {...}`` objects without the surrounding array (Muse Glimmer);
* a JSON array wrapped in prose (Z.ai GLM models);
* a fenced block tagged something other than ``json`` or not tagged at all.

:func:`extract_json_payload` tries a fixed chain of strategies, strictest
first, and returns the first JSON payload it can decode. The chain is
deliberately conservative: a recovery step only succeeds when the *whole*
candidate text is accounted for, so unrelated fragments in garbage output
do not turn a parse failure into an empty success.
"""

import json
import re
from typing import List, Optional, Tuple, Union

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json([\s\S]*?)```", flags=re.IGNORECASE)
ANY_MARKDOWN_BLOCK_PATTERN = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\r?\n?([\s\S]*?)```")
# A line holding nothing but a fence marker (``` or ```json), which is what a
# model leaves behind when it opens or closes a block it never balanced.
FENCE_LINE_PATTERN = re.compile(r"^[ \t]*```[a-zA-Z0-9_+-]*[ \t]*$", flags=re.MULTILINE)
# Separators tolerated between top-level objects in a sequence (JSON Lines,
# or comma-separated objects with the array brackets missing).
OBJECT_SEQUENCE_SEPARATOR_PATTERN = re.compile(r"^[\s,]*$")

JsonPayload = Union[dict, list]


def extract_json_payload(raw: str) -> Tuple[bool, JsonPayload]:
    """Decode a JSON ``dict`` or ``list`` from raw VLM text output.

    Strategies, in order; the first one that decodes wins:

    1. The first ```` ```json ```` fenced block.
    2. The first fenced block with any (or no) language tag.
    3. The whole text with stray fence lines removed.
    4. A sequence of top-level JSON objects (JSON Lines, or ``{...}, {...}``
       without array brackets), returned as a list. The whole text must be
       consumed by objects and separators.
    5. The outermost ``[...]`` or ``{...}`` substring, for JSON wrapped in prose.

    Args:
        raw: Raw text returned by the model.

    Returns:
        ``(error_status, payload)`` — ``error_status`` is ``True`` and the
        payload an empty dict when nothing decodable was found, matching the
        contract of the formatter blocks' previous ``string2json`` helpers.
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
    # Preserve order, drop exact duplicates so each text is decoded once.
    seen = set()
    unique: List[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _try_parse_json(content: str) -> Optional[JsonPayload]:
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def _parse_object_sequence(content: str) -> Optional[list]:
    """Decode ``{...}\\n{...}`` or ``{...}, {...}`` into a list of dicts.

    Succeeds only when at least one object is found and every character
    outside the objects is whitespace or a comma.
    """
    decoder = json.JSONDecoder()
    entries: List[dict] = []
    index = 0
    length = len(content)
    while index < length:
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
    if not entries:
        return None
    if not OBJECT_SEQUENCE_SEPARATOR_PATTERN.match(content[index:]):
        return None
    return entries


def _parse_outermost(content: str, opening: str, closing: str) -> Optional[JsonPayload]:
    start = content.find(opening)
    stop = content.rfind(closing)
    if start == -1 or stop <= start:
        return None
    return _try_parse_json(content[start : stop + 1])
