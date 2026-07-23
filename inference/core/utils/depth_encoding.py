import base64
import binascii

import cv2
import numpy as np

DEPTH_MAP_FORMAT_PNG16 = "png16"
DEPTH_MAP_FORMAT_JSON = "json"
_PNG16_MAX = 65535


def encode_normalized_depth_to_png16(normalized_depth: np.ndarray) -> str:
    """Encode a [0, 1] depth map as a base64 16-bit grayscale PNG.

    Quantization step is 1/65535, and PNG compression is lossless beyond
    quantization; smooth depth maps compress to a small fraction of their
    JSON float-list representation.
    """
    depth = np.asarray(normalized_depth, dtype=np.float32)
    quantized = np.round(np.clip(depth, 0.0, 1.0) * _PNG16_MAX).astype(np.uint16)
    success, buffer = cv2.imencode(".png", quantized)
    if not success:
        raise ValueError("Could not encode normalized depth map as PNG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_png16_normalized_depth(payload: str) -> np.ndarray:
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Depth map payload is not valid base64") from error
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Depth map payload is not a decodable PNG")
    if image.dtype != np.uint16:
        raise ValueError(f"Expected 16-bit depth map payload, got dtype {image.dtype}")
    return image.astype(np.float32) / _PNG16_MAX
