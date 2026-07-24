import base64
import binascii
import warnings
from typing import Any, List, Union

import cv2
import numpy as np

from inference_sdk.config import InferenceSDKDeprecationWarning

_MAX_VALUE_BY_DTYPE = {np.dtype(np.uint8): 255, np.dtype(np.uint16): 65535}
_json_depth_map_deprecation_emitted = False


def warn_depth_map_json_format_deprecated() -> None:
    """Emit a one-off notice that the `"json"` depth map format is deprecated."""
    global _json_depth_map_deprecation_emitted
    if _json_depth_map_deprecation_emitted:
        return
    _json_depth_map_deprecation_emitted = True
    warnings.warn(
        "depth_estimation() is returning `normalized_depth` as a JSON float list "
        "(depth_map_format='json'). This default will change in a breaking way in "
        "one of the first `inference` releases of 2027: the client will request "
        "'png16' by default and return `normalized_depth` as a numpy.ndarray. "
        "Opt in early by passing depth_map_format='png16' (or 'png8'); to keep "
        "the current list format after the switch, pass depth_map_format='json' "
        "explicitly.",
        category=InferenceSDKDeprecationWarning,
        # user code sits 3 frames up: this helper -> client method -> wrap_errors wrapper
        stacklevel=4,
    )


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
