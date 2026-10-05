"""Video source FOURCC codes, without OpenCV.

Kept free of cv2 so the stream manager's API models can validate a requested
``fourcc`` while staying importable without OpenCV.
"""

import math
import numbers
from decimal import Decimal
from typing import Any, Optional

FOURCC_PROPERTY = "fourcc"
_MAX_FOURCC = 0xFFFFFFFF


def _encode_fourcc(code: str) -> int:
    # Same packing as cv2.VideoWriter_fourcc: first character in the low byte.
    return sum((ord(char) & 0xFF) << (8 * index) for index, char in enumerate(code))


def _integral(value: Any) -> Optional[int]:
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    # A FOURCC is a 32-bit unsigned code: finite, whole and within 0..0xFFFFFFFF.
    if math.isfinite(number) and number.is_integer() and 0 <= number <= _MAX_FOURCC:
        return int(number)
    return None


def parse_fourcc(value: Any) -> Optional[int]:
    """Return the numeric FOURCC for ``value``, or ``None`` when it is invalid.

    Args:
        value (Any): A case-sensitive four-character ASCII code (e.g. ``"MJPG"``)
            or a non-negative integral number, possibly given as a string in any
            form ``float()`` reads (``"1196444237"``, ``"1196444237.0"``,
            ``"1.196444237e9"``), as bytes, or as another number type such as
            ``Decimal`` or numpy's.

    Returns:
        Optional[int]: The FOURCC as an integer, or ``None`` if ``value`` is not
            a valid code.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (numbers.Real, Decimal, bytes)):
        return _integral(value)
    if isinstance(value, str):
        # FOURCC codes are case-sensitive (avc1) and may end in a space (Y16 ), so a
        # four-character value is used exactly as given; only other lengths are trimmed.
        code = value if len(value) == 4 else value.strip()
        # float() also reads non-ASCII digits such as Arabic-Indic ones; reject them.
        if not code.isascii():
            return None
        try:
            float(code)
        except ValueError:
            return _encode_fourcc(code) if len(code) == 4 else None
        return _integral(code)

    return None
