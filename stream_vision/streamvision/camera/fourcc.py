"""Video source FOURCC codes, without OpenCV.

Kept free of cv2 so the stream manager's API models can validate a requested
``fourcc`` while staying importable without OpenCV.
"""

import math
from typing import Any, Optional

FOURCC_PROPERTY = "fourcc"


def _encode_fourcc(code: str) -> int:
    # Same packing as cv2.VideoWriter_fourcc: first character in the low byte.
    return sum((ord(char) & 0xFF) << (8 * index) for index, char in enumerate(code))


def parse_fourcc(value: Any) -> Optional[int]:
    """Return the numeric FOURCC for ``value``, or ``None`` when it is invalid.

    Args:
        value (Any): A case-sensitive four-character ASCII code (e.g. ``"MJPG"``)
            or a non-negative integral number, possibly given as a string.

    Returns:
        Optional[int]: The FOURCC as an integer, or ``None`` if ``value`` is not
            a valid code.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value >= 0 and int(value) == value:
            return int(value)
    elif isinstance(value, str):
        # FOURCC codes are case-sensitive (avc1) and may end in a space (Y16 ), so a
        # four-character value is used exactly as given; only other lengths are trimmed.
        code = value if len(value) == 4 else value.strip()
        # isdigit() alone accepts characters such as "²" that int() rejects.
        if code.isascii() and code.isdigit():
            return int(code)
        if len(code) == 4 and code.isascii():
            return _encode_fourcc(code)

    return None
