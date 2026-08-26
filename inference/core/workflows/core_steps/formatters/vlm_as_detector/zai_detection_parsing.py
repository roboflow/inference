import json
from typing import Optional

# The Z.ai detection contracts, both "box_2d" lists normalized to 0-1000:
# GLM 5V Turbo (model_type "zai") uses xyxy order and parses with the Qwen
# parser; GLM 5.3 Flash (model_type "zai-flash") uses yxyx order and parses
# with the Gemini parser. The axis order is not recoverable from the data,
# so each model maps to its own model_type in the registry.


def extract_zai_json_array(raw: str) -> Optional[list]:
    """Recover the outermost JSON array from prose-wrapped Z.ai output.

    GLM models occasionally wrap the detection list in extra text that
    breaks whole-string JSON parsing. Mirrors the vlm-exam fallback: take
    the substring between the first ``[`` and the last ``]`` and parse it.

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
