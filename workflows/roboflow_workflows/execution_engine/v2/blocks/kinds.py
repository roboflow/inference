"""Media kinds owned by the native V2 CPU image catalogue.

The generic V2 engine never inspects payloads. It calls the hooks of these
``Kind`` objects at its boundaries: ``validate`` when a payload enters or leaves
a block, ``deserialize`` for workflow input values and ``serialize`` for
serialized output rows. Plain values (booleans, integers, floats, lists) use
the engine's built-in kinds instead, so that every catalogue shares one kind
object per name.

Image payloads are NumPy ``uint8`` arrays with shape ``(height, width, 3)`` in
RGB channel order. Serialized images use the V1 shape
``{"type": "base64", "value": <PNG bytes in base64>}``; PNG keeps pixels exact.
"""

import base64
import binascii
from typing import Any, Dict, Mapping

import cv2
import numpy as np
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.kinds import Kind

IMAGE_CHANNELS = 3
SERIALIZED_IMAGE_TYPE = "base64"


def _is_image(payload: Any) -> bool:
    if not isinstance(payload, np.ndarray):
        return False
    if payload.dtype != np.uint8 or payload.ndim != 3:
        return False

    height, width, channels = payload.shape
    valid = channels == IMAGE_CHANNELS and height > 0 and width > 0

    return valid


def _image_to_serialized(image: np.ndarray) -> Dict[str, str]:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    encoded, png = cv2.imencode(".png", bgr)
    if not encoded:
        raise ContractError(f"Cannot encode image of shape {image.shape} as PNG")

    serialized = {
        "type": SERIALIZED_IMAGE_TYPE,
        "value": base64.b64encode(png.tobytes()).decode("ascii"),
    }

    return serialized


def _image_from_value(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if not isinstance(value, Mapping) or value.get("type") != SERIALIZED_IMAGE_TYPE:
        raise ContractError(
            "An image input must be an RGB uint8 numpy array or "
            f"{{'type': '{SERIALIZED_IMAGE_TYPE}', 'value': <base64 image>}}, "
            f"got {type(value).__name__}"
        )

    try:
        encoded = base64.b64decode(value.get("value", ""), validate=True)
    except (binascii.Error, TypeError, ValueError) as error:
        raise ContractError(f"Image input holds invalid base64: {error}") from error
    bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ContractError("Image input base64 does not decode to an image")

    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return image


def _is_crop_summary(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False

    counts = [payload.get(key) for key in ("crop_count", "image_height", "image_width")]
    valid = all(
        isinstance(count, int) and not isinstance(count, bool) for count in counts
    )

    return valid


IMAGE_KIND = Kind(
    name="image",
    description="RGB uint8 numpy array of shape (height, width, 3).",
    validate=_is_image,
    deserialize=_image_from_value,
    serialize=_image_to_serialized,
)
CROP_SUMMARY_KIND = Kind(
    name="crop_summary",
    description=(
        "Mapping describing one cropped image: integer crop_count, image_height "
        "and image_width, plus kept_regions and crop_dimensions lists."
    ),
    validate=_is_crop_summary,
)
