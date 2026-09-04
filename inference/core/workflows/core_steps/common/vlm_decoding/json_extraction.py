"""JSON extraction shared by every VLM block that decodes its own output.

VLMs answer with JSON, but not reliably: the payload may be wrapped in a
Markdown ```json fence, surrounded by prose, or (for some models) emitted as
a bare sequence of objects without the enclosing array brackets.
:func:`extract_json` delegates to ``common/vlm_json.extract_json_payload``,
the lenient extractor the deprecated ``vlm_as_*`` formatter blocks use, so
both paths recover exactly the same malformed shapes. The older recovery
helper below is kept for the formatter blocks that still import it.
"""

import json
from typing import Any, List, Tuple

from inference.core.logger import logger
from inference.core.workflows.core_steps.common.vlm_json import extract_json_payload

_NAMED_BOX_FIELDS = ("x_min", "y_min", "x_max", "y_max")


def extract_json(
    raw: str, salvage_truncated_detections: bool = False
) -> Tuple[bool, Any]:
    """Extract the JSON payload out of a raw VLM answer.

    Tries, in order: the first ```json block, the first fenced block with
    any tag, the text with stray fence lines removed, a sequence of
    top-level values (JSON Lines, ``{...}, {...}``, ``[...]\n[...]``,
    tolerating one stray trailing ``]``), and the outermost ``[...]`` or
    ``{...}`` substring. With ``salvage_truncated_detections`` the complete
    ``{"label", "x_min", ...}`` objects of an answer truncated at
    ``max_tokens`` are salvaged when all of that fails
    (``extract_flat_object_entries``), matching the Muse fallback of the
    deprecated ``vlm_as_detector`` block; classification must not opt in,
    because a salvaged box list is not a classification answer.

    Args:
        raw: Raw string produced by the model.
        salvage_truncated_detections: Whether to fall back to salvaging
            complete named-box objects from a truncated answer.

    Returns:
        Tuple of ``(error_status, parsed)``. ``error_status`` is ``True``
        when nothing could be parsed, in which case ``parsed`` is ``{}``.
    """
    if not isinstance(raw, str):
        return True, {}
    error_status, parsed = extract_json_payload(raw)
    if not error_status:
        return False, parsed
    if salvage_truncated_detections:
        loose_entries = extract_flat_object_entries(raw)
        if loose_entries:
            return False, loose_entries
    logger.warning("Could not parse JSON while decoding VLM output.")
    return True, {}


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
