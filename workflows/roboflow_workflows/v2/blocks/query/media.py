"""UQL operations on classification results, images, frames and zones.

Host boundaries are explicit: JPEG and base64 encoding copy the image to the
host once (OpenCV, as V1 does, so bytes match V1); the zone test uses shapely
on host coordinates. Image size and frame metadata read no pixels.
"""

import base64
from typing import Any, List, Optional, Sequence

from roboflow_workflows.v2.blocks.query.detections import (
    CLASS_NAMES_KEY,
    is_detections,
    native_of,
)
from roboflow_workflows.v2.blocks.query.errors import OperationFailure
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.rows import PredictionRows

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)

BASE64_JPEG_QUALITY = 95


def classification_property(operation, value: Any) -> Any:
    """Extract a property of one image's classification, as V1 tensor mode does."""
    name = operation.property_name
    if isinstance(value, ClassificationPrediction):
        return _single_label_property(value, name)
    if isinstance(value, MultiLabelClassificationPrediction):
        return _multi_label_property(value, name)

    raise OperationFailure(
        "expected ClassificationPrediction or MultiLabelClassificationPrediction"
    )


def _single_label_property(prediction: ClassificationPrediction, name: str) -> Any:
    if prediction.class_id.shape[0] != 1:
        raise OperationFailure(
            f"expected one image, got a batch of {prediction.class_id.shape[0]}"
        )
    if name == "all_confidences":
        return prediction.confidence[0].tolist()
    if name in ("top_class_confidence", "top_class_confidence_single"):
        class_id = int(prediction.class_id[0])
        return float(prediction.confidence[0, class_id])

    images_metadata = prediction.images_metadata
    names_by_id = _class_names(None if images_metadata is None else images_metadata[0])
    if name == "all_classes":
        return [names_by_id[class_id] for class_id in sorted(names_by_id)]

    class_names = _names_of([int(prediction.class_id[0])], names_by_id)

    return class_names[0]


def _multi_label_property(
    prediction: MultiLabelClassificationPrediction, name: str
) -> Any:
    if name == "all_confidences":
        return prediction.confidence.tolist()
    if name == "top_class_confidence":
        return prediction.confidence[prediction.class_ids].tolist()
    if name == "top_class_confidence_single":
        if prediction.class_ids.shape[0] == 0:
            return 0.0
        return float(prediction.confidence[prediction.class_ids].max())

    names_by_id = _class_names(prediction.image_metadata)
    if name == "all_classes":
        return [names_by_id[class_id] for class_id in sorted(names_by_id)]
    class_names = _names_of(prediction.class_ids.tolist(), names_by_id)

    return class_names


def _class_names(metadata: Optional[dict]) -> dict:
    names_by_id = None if metadata is None else metadata.get(CLASS_NAMES_KEY)
    if names_by_id is None:
        raise OperationFailure(
            f"image metadata has no '{CLASS_NAMES_KEY}' mapping from class id to name"
        )

    return names_by_id


def _names_of(class_ids: Sequence[int], names_by_id: dict) -> List[str]:
    names = []
    for class_id in class_ids:
        name = names_by_id.get(int(class_id))
        if name is None:
            raise OperationFailure(
                f"class id {int(class_id)} is missing from '{CLASS_NAMES_KEY}' "
                f"(keys: {sorted(names_by_id)})"
            )
        names.append(name)

    return names


def image_property(operation, value: Any) -> Any:
    """Read size properties from the tensor shape or a FrameRef; no pixels move."""
    if isinstance(value, ImageData):
        height, width = value.height, value.width
    elif isinstance(value, FrameRef):
        height, width = value.size_hw
    else:
        raise OperationFailure("expected ImageData or FrameRef")

    properties = {
        "size": height * width,
        "height": height,
        "width": width,
        "aspect_ratio": width / height if height != 0 else 0,
    }

    return properties[operation.property_name]


def image_to_jpeg(operation, value: Any) -> bytes:
    jpeg = _encode_jpeg(_require_image(value), quality=operation.compression_level)

    return jpeg


def image_to_base64(operation, value: Any) -> str:
    jpeg = _encode_jpeg(_require_image(value), quality=BASE64_JPEG_QUALITY)
    text = base64.b64encode(jpeg).decode("ascii")

    return text


def _require_image(value: Any) -> ImageData:
    if not isinstance(value, ImageData):
        raise OperationFailure("expected ImageData")

    return value


def _encode_jpeg(image: ImageData, *, quality: int) -> bytes:
    import cv2
    import numpy as np

    # Host boundary: the pixels leave their device here, once.
    hwc = image.tensor_image.detach().permute(1, 2, 0).cpu().numpy()
    if image.channels == 1:
        pixels = np.ascontiguousarray(hwc[:, :, 0])
    else:
        pixels = cv2.cvtColor(np.ascontiguousarray(hwc), cv2.COLOR_RGB2BGR)
    encoded, jpeg = cv2.imencode(
        ".jpg", pixels, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not encoded:
        raise OperationFailure(
            f"OpenCV could not encode an image of shape {pixels.shape}"
        )

    return jpeg.tobytes()


def frame_metadata(operation, value: Any) -> Any:
    """Read video metadata of an image, a FrameRef or a prediction's frame."""
    video_metadata = _video_metadata(value)
    if video_metadata is None:
        raise OperationFailure(
            "the frame has no video metadata (V1 would have invented frame 0 "
            "at the current time)"
        )
    name = operation.property_name
    if name == "frame_number":
        return video_metadata.frame_number
    if name == "frame_timestamp":
        return video_metadata.frame_timestamp
    if not video_metadata.fps:
        raise OperationFailure(
            f"seconds_since_start needs a positive fps, got {video_metadata.fps!r}"
        )

    seconds = (video_metadata.frame_number - 1) / video_metadata.fps

    return seconds


def _video_metadata(value: Any) -> Any:
    if isinstance(value, (ImageData, FrameRef)):
        return value.video_metadata
    if isinstance(value, PredictionRows):
        return value.frame.video_metadata
    if is_detections(value):
        return frame_of(native_of(value)).video_metadata

    raise OperationFailure(
        "expected ImageData, FrameRef or a detection prediction with a frame"
    )


def point_in_zone(point: Sequence[float], zone: Sequence[Sequence[float]]) -> bool:
    """V1 ``(Detection) in zone``: the point is strictly inside the polygon."""
    import shapely

    inside = shapely.geometry.Point(point[0], point[1]).within(
        shapely.geometry.Polygon([(vertex[0], vertex[1]) for vertex in zone])
    )

    return inside
