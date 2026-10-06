import base64
import binascii

import cv2
import numpy as np

DEPTH_MAP_FORMAT_PNG16 = "png16"
DEPTH_MAP_FORMAT_PNG8 = "png8"
DEPTH_MAP_FORMAT_JSON = "json"
_MAX_VALUE_BY_DTYPE = {np.dtype(np.uint8): 255, np.dtype(np.uint16): 65535}


def _encode_normalized_depth_to_png(
    normalized_depth: np.ndarray, dtype: np.dtype
) -> str:
    depth = np.asarray(normalized_depth, dtype=np.float32)
    max_value = _MAX_VALUE_BY_DTYPE[np.dtype(dtype)]
    quantized = np.round(np.clip(depth, 0.0, 1.0) * max_value).astype(dtype)
    success, buffer = cv2.imencode(".png", quantized)
    if not success:
        raise ValueError("Could not encode normalized depth map as PNG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def encode_normalized_depth_to_png16(normalized_depth: np.ndarray) -> str:
    """Encode a [0, 1] depth map as a base64 16-bit grayscale PNG.

    Quantization step is 1/65535, and PNG compression is lossless beyond
    quantization; smooth depth maps compress to a small fraction of their
    JSON float-list representation.
    """
    return _encode_normalized_depth_to_png(normalized_depth, np.uint16)


def encode_normalized_depth_to_png8(normalized_depth: np.ndarray) -> str:
    """Encode a [0, 1] depth map as a base64 8-bit grayscale PNG.

    Quantization step is 1/255 (256 depth levels) - roughly an order of
    magnitude smaller than PNG16, at the cost of visible banding on smooth
    gradients; adequate for visualization and thresholding, not for
    derivative-based downstream use.
    """
    return _encode_normalized_depth_to_png(normalized_depth, np.uint8)


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
