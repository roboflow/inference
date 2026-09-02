"""JSON extraction shared by every VLM block that decodes its own output.

VLMs answer with JSON, but not reliably: the payload may be wrapped in a
Markdown ```json fence, surrounded by prose, or (for some models) emitted as
a bare sequence of objects without the enclosing array brackets. This module
concentrates the extraction and the recovery strategies so no VLM block has
to repeat them.
"""

import json
import re
from typing import Any, List, Optional, Tuple

from inference.core.logger import logger

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json([\s\S]*?)```", flags=re.IGNORECASE)

_NAMED_BOX_FIELDS = ("x_min", "y_min", "x_max", "y_max")


def extract_json(raw: str) -> Tuple[bool, Any]:
    """Extract the JSON payload out of a raw VLM answer.

    The first ```json fenced block wins; when no fence is present the whole
    string is parsed. When that fails, two format-agnostic recovery
    strategies are attempted in order: ``extract_zai_json_array`` (prose
    wrapped around a JSON array) and ``extract_flat_object_entries`` (a bare
    ``{...}, {...}`` sequence). Both are harmless for outputs that do not
    need them.

    Args:
        raw: Raw string produced by the model.

    Returns:
        Tuple of ``(error_status, parsed)``. ``error_status`` is ``True``
        when nothing could be parsed, in which case ``parsed`` is ``{}``.
    """
    error_status, parsed = _string2json(raw_json=raw)
    if not error_status:
        return False, parsed
    recovered_array = extract_zai_json_array(raw)
    if recovered_array is not None:
        return False, recovered_array
    loose_entries = extract_flat_object_entries(raw)
    if loose_entries:
        return False, loose_entries
    return True, {}


def _string2json(raw_json: str) -> Tuple[bool, Any]:
    json_blocks_found = JSON_MARKDOWN_BLOCK_PATTERN.findall(raw_json)
    if len(json_blocks_found) == 0:
        return try_parse_json(raw_json)
    first_block = json_blocks_found[0]
    return try_parse_json(first_block)


def try_parse_json(content: str) -> Tuple[bool, Any]:
    """Parse ``content`` as JSON, requiring a dict or list root.

    Args:
        content: Candidate JSON string.

    Returns:
        Tuple of ``(error_status, parsed)``; ``parsed`` is ``{}`` on failure.
    """
    try:
        parsed = json.loads(content)
        if isinstance(parsed, (dict, list)):
            return False, parsed
        logger.warning(
            "Could not parse JSON to dict while decoding VLM output. "
            "Unexpected JSON root type: %s.",
            type(parsed).__name__,
        )
        return True, {}
    except Exception as error:
        logger.warning(
            "Could not parse JSON to dict while decoding VLM output. "
            "Error type: %s. Details: %s",
            error.__class__.__name__,
            error,
        )
        return True, {}


def extract_zai_json_array(raw: str) -> Optional[list]:
    """Recover the outermost JSON array from prose-wrapped output.

    Some models wrap the detection list in extra text that breaks
    whole-string JSON parsing. Take the substring between the first ``[``
    and the last ``]`` and parse it.

    Args:
        raw: Raw VLM output that failed regular JSON parsing.

    Returns:
        The recovered list, or ``None`` when nothing recoverable.
    """
    start = raw.find("[")
    stop = raw.rfind("]")
    if start == -1 or stop <= start:
        return None
    try:
        recovered = json.loads(raw[start : stop + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(recovered, list):
        return None
    return recovered


def extract_flat_object_entries(prediction: str) -> List[dict]:
    """Recover detection dicts from a loose, non-JSON object sequence.

    Some models emit ``{...}, {...}`` objects without the surrounding array
    brackets, which ``json.loads`` rejects. Scan the raw string and decode
    each top-level ``{...}`` object individually. Only dicts carrying all
    four named box fields are kept, so unrelated JSON fragments in garbage
    output do not turn a parse failure into an empty success.

    Args:
        prediction: Raw VLM output that failed regular JSON parsing.

    Returns:
        List of detection entry dicts; empty when nothing recoverable.
    """
    decoder = json.JSONDecoder()
    entries: List[dict] = []
    index = 0
    while True:
        start = prediction.find("{", index)
        if start == -1:
            break
        try:
            entry, end = decoder.raw_decode(prediction, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(entry, dict) and all(
            field in entry for field in _NAMED_BOX_FIELDS
        ):
            entries.append(entry)
        index = end
    return entries
