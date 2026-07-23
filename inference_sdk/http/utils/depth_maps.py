import base64
import binascii
from typing import Any, List, Union

import cv2
import numpy as np

_MAX_VALUE_BY_DTYPE = {np.dtype(np.uint8): 255, np.dtype(np.uint16): 65535}


def decode_png_normalized_depth(payload: str) -> np.ndarray:
    """Decode a base64 grayscale PNG depth map (8- or 16-bit) to floats in [0, 1]."""
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Depth map payload is not valid base64") from error
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Depth map payload is not a decodable PNG")
    max_value = _MAX_VALUE_BY_DTYPE.get(image.dtype)
    if max_value is None:
        raise ValueError(
            f"Expected 8- or 16-bit depth map payload, got dtype {image.dtype}"
        )
    return image.astype(np.float32) / max_value


def decode_depth_estimation_result(
    result: Union[dict, List[dict]],
) -> Union[dict, List[dict]]:
    """Replace PNG-serialized `normalized_depth` payloads with numpy arrays.

    Handles both `png16` and `png8` payloads (bit depth is self-describing via
    the PNG dtype). Tolerates the legacy `json` format (nested float lists) and
    older servers that never serialize to PNG - those results pass through
    unchanged.
    """
    if isinstance(result, list):
        return [decode_depth_estimation_result(element) for element in result]
    normalized_depth: Any = result.get("normalized_depth")
    if isinstance(normalized_depth, str):
        result["normalized_depth"] = decode_png_normalized_depth(normalized_depth)
    return result
