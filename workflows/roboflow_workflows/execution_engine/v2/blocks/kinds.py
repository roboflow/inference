"""Media kinds owned by the native V2 image catalogue.

The generic V2 engine never inspects payloads. It calls the hooks of these
``Kind`` objects at its boundaries: ``validate`` when a payload enters or leaves
a block, ``deserialize`` for workflow input values and ``serialize`` for
serialized output rows. Plain values (booleans, integers, floats, lists) use
the engine's built-in kinds instead, so that every catalogue shares one kind
object per name.

Image payloads are ``ImageData`` (see ``image_data``). A workflow image input
may be supplied as, in this order:

1. ``ImageData``: used as is.
2. ``torch.Tensor`` ``(channels, height, width)`` ``uint8``, 1 or 3 (RGB)
   channels: wrapped without copying or moving it.
3. NumPy ``uint8`` ``(height, width, 3)`` RGB or ``(height, width)``: copied
   into a CPU tensor.
4. ``{"type": "base64", "value": <PNG or JPEG>, ...}``: decoded on CPU. The
   provenance keys written by ``serialize`` are restored exactly; V1 keys
   (``parent_id``, ``parent_origin``, ``root_parent_id``,
   ``root_parent_origin``, ``video_metadata``) are read as the V1 adapter does.
5. V1 ``WorkflowImageData``: adapted by ``ImageData.from_workflow_image_data``.

Inputs without an identity get a unique ``input-...`` id. Serialization is the
explicit host boundary: pixels are copied to the host and encoded as PNG next
to the provenance::

    {"type": "base64", "value": <PNG>, "image_id": ..., "parent": {...},
     "root": {...}, "composite_sources": [...], "video_metadata": {...}}

``composite_sources`` and ``video_metadata`` are present only when set.
"""

import base64
import binascii
import uuid
from dataclasses import replace
from typing import Any, Dict, Mapping

import cv2
import numpy as np
import torch
from roboflow_workflows.execution_engine.v2.blocks.image_data import (
    CompositeSource,
    FrameMapping,
    ImageData,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.kinds import Kind

SERIALIZED_IMAGE_TYPE = "base64"
V1_IMAGE_KEYS = (
    "parent_id",
    "parent_origin",
    "root_parent_id",
    "root_parent_origin",
    "video_metadata",
)


def _is_image(payload: Any) -> bool:
    return isinstance(payload, ImageData)


def _image_from_value(value: Any) -> ImageData:
    if isinstance(value, ImageData):
        return value
    if isinstance(value, torch.Tensor):
        return ImageData.from_tensor(value)
    if isinstance(value, np.ndarray):
        return ImageData.from_numpy_rgb(value)
    if isinstance(value, Mapping) and value.get("type") == SERIALIZED_IMAGE_TYPE:
        return _image_from_serialized(value)

    # Imported lazily: the V1 entities module is slow to import.
    from roboflow_workflows.execution_engine.entities.base import WorkflowImageData

    if isinstance(value, WorkflowImageData):
        return ImageData.from_workflow_image_data(value)

    raise ContractError(
        "An image input must be ImageData, a (channels, height, width) uint8 "
        "torch.Tensor, an RGB uint8 numpy array, a V1 WorkflowImageData or "
        f"{{'type': '{SERIALIZED_IMAGE_TYPE}', 'value': <base64 image>}}, "
        f"got {type(value).__name__}"
    )


def _image_from_serialized(value: Mapping[str, Any]) -> ImageData:
    decoded = _decode_base64_image(value.get("value", ""))
    if "image_id" in value:
        image = _restore_serialized_provenance(decoded, serialized=value)
    elif any(key in value for key in V1_IMAGE_KEYS):
        image = _image_from_v1_serialized(decoded, serialized=value)
    else:
        image = ImageData.from_numpy_rgb(_to_rgb(decoded))

    return image


def _restore_serialized_provenance(
    decoded: np.ndarray, *, serialized: Mapping[str, Any]
) -> ImageData:
    video_metadata = None
    if serialized.get("video_metadata") is not None:
        video_metadata = _video_metadata_from_dict(serialized["video_metadata"])

    composite_sources = serialized.get("composite_sources")
    if composite_sources is not None:
        composite_sources = tuple(
            CompositeSource.from_dict(source) for source in composite_sources
        )

    try:
        image = replace(
            ImageData.from_numpy_rgb(_to_rgb(decoded), image_id=serialized["image_id"]),
            parent=FrameMapping.from_dict(serialized["parent"]),
            root=FrameMapping.from_dict(serialized["root"]),
            video_metadata=video_metadata,
            composite_sources=composite_sources,
        )
    except KeyError as error:
        raise ContractError(
            f"Serialized image with 'image_id' lacks the {error} mapping"
        ) from error

    return image


def _image_from_v1_serialized(
    decoded: np.ndarray, *, serialized: Mapping[str, Any]
) -> ImageData:
    # Build the V1 container the V1 deserializer would build, then adapt it,
    # so both V1 routes share one interpretation of V1 metadata.
    from roboflow_workflows.execution_engine.entities.base import (
        ImageParentMetadata,
        ParentOrigin,
        WorkflowImageData,
    )

    def parent_metadata(id_key: str, origin_key: str) -> Any:
        if serialized.get(id_key) is None:
            return None
        origin = serialized.get(origin_key)
        coordinates = (
            None
            if origin is None
            else ParentOrigin.model_validate(origin).to_origin_coordinates_system()
        )
        metadata = ImageParentMetadata(
            parent_id=serialized[id_key], origin_coordinates=coordinates
        )
        return metadata

    own_metadata = parent_metadata("parent_id", "parent_origin")
    if own_metadata is None:
        own_metadata = ImageParentMetadata(parent_id=f"input-{uuid.uuid4().hex}")

    video_metadata = None
    if serialized.get("video_metadata") is not None:
        video_metadata = _video_metadata_from_dict(serialized["video_metadata"])

    v1_image = WorkflowImageData(
        parent_metadata=own_metadata,
        workflow_root_ancestor_metadata=parent_metadata(
            "root_parent_id", "root_parent_origin"
        ),
        numpy_image=decoded,
        video_metadata=video_metadata,
    )
    image = ImageData.from_workflow_image_data(v1_image)

    return image


def _decode_base64_image(encoded: Any) -> np.ndarray:
    # Returns OpenCV order: (height, width, 3) BGR or (height, width) grayscale.
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, TypeError, ValueError) as error:
        raise ContractError(f"Image input holds invalid base64: {error}") from error

    buffer = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ContractError("Image input base64 does not decode to an image")

    is_gray = decoded.ndim == 2
    is_color = decoded.ndim == 3 and decoded.shape[2] == 3
    if decoded.dtype != np.uint8 or not (is_gray or is_color):
        # Alpha, 16-bit and other layouts: 8-bit color, as before.
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    return decoded


def _to_rgb(decoded: np.ndarray) -> np.ndarray:
    if decoded.ndim == 2:
        return decoded

    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    return rgb


def _video_metadata_from_dict(value: Mapping[str, Any]) -> Any:
    from roboflow_workflows.execution_engine.entities.base import VideoMetadata

    video_metadata = VideoMetadata.model_validate(value)

    return video_metadata


def _image_to_serialized(image: ImageData) -> Dict[str, Any]:
    # Host boundary: the only place image pixels leave their device.
    hwc = image.tensor_image.detach().permute(1, 2, 0).cpu().numpy()
    bgr = hwc[:, :, 0] if image.channels == 1 else cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
    encoded, png = cv2.imencode(".png", bgr)
    if not encoded:
        raise ContractError(f"Cannot encode image of shape {bgr.shape} as PNG")

    serialized = {
        "type": SERIALIZED_IMAGE_TYPE,
        "value": base64.b64encode(png.tobytes()).decode("ascii"),
        "image_id": image.image_id,
        "parent": image.parent.to_dict(),
        "root": image.root.to_dict(),
    }
    if image.composite_sources is not None:
        serialized["composite_sources"] = [
            source.to_dict() for source in image.composite_sources
        ]
    if image.video_metadata is not None:
        serialized["video_metadata"] = image.video_metadata.model_dump(mode="json")

    return serialized


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
    description=(
        "ImageData: (channels, height, width) uint8 torch tensor, 1 or 3 (RGB) "
        "channels, with image identity and parent/root provenance."
    ),
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
