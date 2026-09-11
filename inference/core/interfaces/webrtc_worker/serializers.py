import base64
from typing import Any, Dict

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WEBRTC_PREVIEW_FRAME_JPEG_QUALITY,
)
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes

# The loader swaps the wildcard serialiser under the tensor flag; this module
# mirrors that swap, or tensor-native outputs reach orjson unserialised.
if ENABLE_TENSOR_DATA_REPRESENTATION:
    from inference.core.workflows.core_steps.common.serializers_tensor import (
        serialize_wildcard_kind,
    )
else:
    from inference.core.workflows.core_steps.common.serializers import (
        serialize_wildcard_kind,
    )
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def compress_image_for_webrtc(image: WorkflowImageData) -> Dict[str, Any]:
    """Serialize image with low JPEG quality for efficient WebRTC transmission."""
    jpeg_bytes = encode_image_to_jpeg_bytes(
        image.numpy_image, jpeg_quality=WEBRTC_PREVIEW_FRAME_JPEG_QUALITY
    )
    return {
        "type": "base64",
        "value": base64.b64encode(jpeg_bytes).decode("ascii"),
        "video_metadata": image.video_metadata.dict() if image.video_metadata else None,
    }


def serialize_for_webrtc(value: Any) -> Any:
    """Serialize for WebRTC, compressing images with low JPEG quality."""
    if isinstance(value, WorkflowImageData):
        return compress_image_for_webrtc(value)
    if isinstance(value, dict):
        return {k: serialize_for_webrtc(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_for_webrtc(v) for v in value]
    return serialize_wildcard_kind(value)
