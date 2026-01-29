import base64
from typing import Any, Dict

from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.workflows.core_steps.common.serializers import serialize_wildcard_kind
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

# WebRTC image compression quality - lower = smaller file size
WEBRTC_JPEG_QUALITY = 80


def serialise_image_for_webrtc(image: WorkflowImageData) -> Dict[str, Any]:
    """Serialize image with low JPEG quality for efficient WebRTC transmission."""
    jpeg_bytes = encode_image_to_jpeg_bytes(image.numpy_image, jpeg_quality=WEBRTC_JPEG_QUALITY)
    return {
        "type": "base64",
        "value": base64.b64encode(jpeg_bytes).decode("ascii"),
        "video_metadata": image.video_metadata.dict() if image.video_metadata else None,
    }


def serialize_for_webrtc(value: Any) -> Any:
    """Serialize for WebRTC, compressing images with low JPEG quality."""
    if isinstance(value, WorkflowImageData):
        return serialise_image_for_webrtc(value)
    if isinstance(value, dict):
        return {k: serialize_for_webrtc(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_for_webrtc(v) for v in value]
    return serialize_wildcard_kind(value)
