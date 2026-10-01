"""Apply ``cv2.VideoCapture`` properties in a camera-safe order.

The order in which capture properties are set matters for V4L2 / USB cameras.
Many UVC cameras (e.g. Logitech C920) open in an uncompressed pixel format
(YUYV) whose frame-rate ceiling at high resolutions is very low. Setting
``fps`` while the device is still in that format clamps the frame interval,
and a later ``fourcc`` switch (e.g. to MJPG) keeps the clamped interval.

Properties often arrive as dictionaries built from JSON payloads whose key
order is not guaranteed, so they are applied in a fixed order:

1. ``fourcc`` first - selects the pixel format,
2. every other property in the order given (e.g. frame size),
3. ``fps`` last - negotiated against the final format and frame size.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2

logger = logging.getLogger(__name__)

FOURCC_PROPERTY = "fourcc"
FPS_PROPERTY = "fps"


def apply_capture_properties(
    stream: cv2.VideoCapture,
    *,
    properties: Optional[Dict[str, Union[float, int, str]]],
) -> None:
    """Set capture properties on ``stream`` with ``fourcc`` first and ``fps`` last.

    Properties other than ``fourcc`` and ``fps`` keep their given order.
    ``fourcc`` may be a four-character code (``"MJPG"``, ``"mjpg"``) or its
    numeric value; an invalid ``fourcc`` is logged and skipped. When ``fourcc``
    or ``fps`` is requested, the effective format is read back and logged.

    Args:
        stream (cv2.VideoCapture): Opened capture to configure.
        properties (Optional[Dict[str, Union[float, int, str]]]): Mapping of
            ``cv2.CAP_PROP_*`` suffixes (case-insensitive) to values.

    Raises:
        AttributeError: If a property name has no ``cv2.CAP_PROP_*`` constant.
    """
    if not properties:
        return

    for property_id, value in _order_capture_properties(properties=properties):
        if property_id.lower() == FOURCC_PROPERTY:
            value = _parse_fourcc(value)
            if value is None:
                continue
        cv2_id = getattr(cv2, "CAP_PROP_" + property_id.upper())
        if not stream.set(cv2_id, value):
            logger.debug(
                f"Video source did not accept property {property_id}={value!r}"
            )

    requested_ids = {property_id.lower() for property_id in properties}
    if requested_ids & {FOURCC_PROPERTY, FPS_PROPERTY}:
        _log_effective_format(stream)


def _order_capture_properties(
    properties: Dict[str, Any],
) -> List[Tuple[str, Any]]:
    fourcc, others, fps = [], [], []
    for property_id, value in properties.items():
        normalised_id = property_id.lower()
        if normalised_id == FOURCC_PROPERTY:
            fourcc.append((property_id, value))
        elif normalised_id == FPS_PROPERTY:
            fps.append((property_id, value))
        else:
            others.append((property_id, value))

    ordered_properties = fourcc + others + fps

    return ordered_properties


def _parse_fourcc(value: Any) -> Optional[int]:
    """Return the numeric FOURCC for ``value``, or ``None`` when it is invalid."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value >= 0 and int(value) == value:
            return int(value)
    elif isinstance(value, str):
        # FOURCC codes are case-sensitive (avc1) and may end in a space (Y16 ), so a
        # four-character value is used exactly as given; only other lengths are trimmed.
        code = value if len(value) == 4 else value.strip()
        if code.isdigit():
            return int(code)
        if len(code) == 4 and code.isascii():
            return cv2.VideoWriter_fourcc(*code)

    logger.warning(
        f"Ignoring invalid fourcc video source property: {value!r}. Expected a "
        "four-character code (e.g. 'MJPG') or its numeric value."
    )
    return None


def _decode_fourcc(value: float) -> str:
    code = int(value)
    if code <= 0:
        return ""

    decoded = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))

    return decoded


def _log_effective_format(stream: cv2.VideoCapture) -> None:
    # Diagnostics only: surfaces drivers that silently ignore or clamp a
    # requested format / frame rate (e.g. MJPG applied but fps left at 5).
    try:
        fourcc = _decode_fourcc(stream.get(cv2.CAP_PROP_FOURCC))
        fps = stream.get(cv2.CAP_PROP_FPS)
        width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except Exception as error:  # noqa: BLE001 - diagnostics must never fail startup
        logger.debug(f"Could not read back video source format: {error!r}")
        return

    logger.info(
        "Video source format after applying properties: "
        f"fourcc={fourcc!r}, fps={fps}, size={width}x{height}"
    )
