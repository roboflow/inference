"""Deterministic stand-ins for the six ``inference_models`` task-model APIs.

Each fake has the native phase API (``pre_process``, ``forward``,
``post_process``) and ``class_names``; nothing is downloaded. It is NOT model
inference: outputs are a fixed function of each image's size and mean
brightness, so tests can predict them exactly:

    brightness b = mean pixel / 255          per image, in [0, 1]
    rows         = round(4 * b)              0..4 rows: empty and ragged batches
    row i        box [i, i, w - i, h - i], class i % classes,
                 confidence 0.9 - 0.1 * i

Every fake records its calls, batch sizes and the ``post_process`` keywords it
received. Detector variants differ only in which keywords ``post_process``
names: ``NmsDetector`` names the NMS options, ``NmsFreeDetector`` names only
``confidence``, like RF-DETR. Named modes resolve as ``"best"`` -> 0.5 and
``"default"`` -> 0.35, so a test can see which mode reached the model.

Two fakes copy a backend shape from the ``inference_models`` source survey:
``IgnoringNmsDetector`` names ``class_agnostic_nms`` without using it (YOLO-NAS
TensorRT) and ``ForwardThresholdKeypointModel`` applies its thresholds inside
``forward``, so ``post_process`` reads no option (RF-DETR two-stage keypoints).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from roboflow_workflows.v2.blocks.models.cache import ModelRequest

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

CLASS_NAMES = ("person", "car", "dog")
MODE_THRESHOLDS = {"best": 0.5, "default": 0.35}
KEY_POINTS = 2


class FakeModel:
    """Shared phases: one brightness per image, sizes as metadata."""

    class_names: Sequence[str] = CLASS_NAMES

    def __init__(self, request: Optional[ModelRequest] = None):
        self.request = request
        self.calls: List[str] = []
        self.batch_sizes: List[int] = []
        self.received: List[Dict[str, Any]] = []

    def pre_process(self, images: List[torch.Tensor], input_color_format: str):
        assert input_color_format == "rgb"
        self.calls.append("pre_process")
        self.batch_sizes.append(len(images))
        batch = torch.stack([image.float().mean() / 255.0 for image in images])
        sizes = [tuple(image.shape[1:]) for image in images]

        return batch, sizes

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        self.calls.append("forward")

        return batch.clone()

    def _record(self, **keywords: Any) -> float:
        self.calls.append("post_process")
        self.received.append(keywords)
        confidence = keywords.get("confidence", 0.0)
        threshold = MODE_THRESHOLDS.get(confidence, confidence)

        return threshold

    def _note(self) -> Dict[str, Any]:
        return {"served_by": str(self.request)}


def row_count(brightness: float) -> int:
    """Rows a fake predicts for one image of this mean brightness."""
    count = round(4 * brightness)

    return count


def _rows(brightness: float, size_hw: Tuple[int, int], *, threshold: float):
    height, width = size_hw
    count = row_count(brightness)
    index = torch.arange(count)
    xyxy = torch.stack([index, index, width - index, height - index], dim=1).float()
    class_id = index % len(CLASS_NAMES)
    confidence = 0.9 - 0.1 * index.float()
    keep = confidence >= threshold

    return xyxy[keep], class_id[keep], confidence[keep]


class NmsDetector(FakeModel):
    """Object detector whose ``post_process`` names the NMS options."""

    def post_process(
        self,
        raw: torch.Tensor,
        sizes: list,
        confidence: Any = "default",
        iou_threshold: float = 0.3,
        max_detections: int = 300,
        class_agnostic_nms: bool = False,
        **kwargs: Any,
    ) -> List[Detections]:
        threshold = self._record(
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            class_agnostic_nms=class_agnostic_nms,
            **kwargs,
        )
        predictions = []
        for brightness, size in zip(raw.tolist(), sizes):
            xyxy, class_id, scores = _rows(brightness, size, threshold=threshold)
            predictions.append(
                Detections(
                    xyxy=xyxy[:max_detections],
                    class_id=class_id[:max_detections],
                    confidence=scores[:max_detections],
                    image_metadata=self._note(),
                )
            )

        return predictions


class NmsFreeDetector(FakeModel):
    """Object detector without NMS: ``post_process`` names only confidence."""

    def post_process(
        self, raw: torch.Tensor, sizes: list, confidence: Any = "default", **kwargs: Any
    ) -> List[Detections]:
        threshold = self._record(confidence=confidence, **kwargs)
        predictions = [
            Detections(*_rows(brightness, size, threshold=threshold))
            for brightness, size in zip(raw.tolist(), sizes)
        ]

        return predictions


class IgnoringNmsDetector(NmsDetector):
    """Names ``class_agnostic_nms`` but never uses it, like YOLO-NAS TensorRT.

    Its ``post_process`` carries the real backend's qualified name, the key of
    ``common.IGNORED_KEYWORDS``.
    """

    def post_process(
        self,
        raw: torch.Tensor,
        sizes: list,
        confidence: Any = "default",
        iou_threshold: float = 0.3,
        max_detections: int = 300,
        class_agnostic_nms: bool = False,
        **kwargs: Any,
    ) -> List[Detections]:
        predictions = super().post_process(
            raw,
            sizes,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            **kwargs,
        )

        return predictions


IgnoringNmsDetector.post_process.__module__ = (
    "inference_models.models.yolonas.yolonas_object_detection_trt"
)
IgnoringNmsDetector.post_process.__qualname__ = (
    "YOLONasForObjectDetectionTRT.post_process"
)


class Classifier(FakeModel):
    """Single-label classifier: one batched prediction, no metadata."""

    def pre_process(self, images: List[torch.Tensor], input_color_format: str):
        batch, _ = super().pre_process(images, input_color_format)

        return batch

    def post_process(
        self, raw: torch.Tensor, **kwargs: Any
    ) -> ClassificationPrediction:
        self._record(**kwargs)
        # Brightness picks the top class: dark -> person, mid -> car, bright -> dog.
        centres = torch.tensor([0.0, 0.5, 1.0])
        logits = -((raw[:, None] - centres[None, :]) ** 2) * 10
        confidence = logits.softmax(dim=1)
        prediction = ClassificationPrediction(
            class_id=confidence.argmax(dim=1), confidence=confidence
        )

        return prediction


class MultiLabelClassifier(Classifier):
    """Multi-label classifier: a class is predicted above the threshold."""

    def post_process(
        self, raw: torch.Tensor, confidence: Any = "default", **kwargs: Any
    ) -> List[MultiLabelClassificationPrediction]:
        threshold = self._record(confidence=confidence, **kwargs)
        predictions = []
        for brightness in raw.tolist():
            scores = torch.tensor([brightness, 1 - brightness, 0.5])
            predictions.append(
                MultiLabelClassificationPrediction(
                    class_ids=torch.nonzero(scores >= threshold).flatten(),
                    confidence=scores,
                    image_metadata=self._note(),
                )
            )

        return predictions


class KeypointModel(FakeModel):
    """Keypoint model with boxes; ``with_boxes=False`` returns ``None`` boxes."""

    key_points_classes = [["nose", "tail"]] * len(CLASS_NAMES)
    skeletons = [[(0, 1)]] * len(CLASS_NAMES)

    def __init__(self, request: Optional[ModelRequest] = None, *, with_boxes=True):
        super().__init__(request)
        self.with_boxes = with_boxes

    def post_process(
        self,
        raw: torch.Tensor,
        sizes: list,
        confidence: Any = "default",
        iou_threshold: float = 0.3,
        max_detections: int = 300,
        class_agnostic_nms: bool = False,
        key_points_threshold: float = 0.0,
        **kwargs: Any,
    ) -> Tuple[List[KeyPoints], Optional[List[Detections]]]:
        threshold = self._record(
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            class_agnostic_nms=class_agnostic_nms,
            key_points_threshold=key_points_threshold,
            **kwargs,
        )
        key_points, boxes = [], []
        for brightness, size in zip(raw.tolist(), sizes):
            xyxy, class_id, scores = _rows(brightness, size, threshold=threshold)
            key_points.append(
                KeyPoints(
                    xy=torch.stack([xyxy[:, :2], xyxy[:, 2:]], dim=1),
                    class_id=class_id,
                    confidence=scores[:, None].repeat(1, KEY_POINTS),
                    image_metadata=self._note(),
                )
            )
            boxes.append(Detections(xyxy=xyxy, class_id=class_id, confidence=scores))

        return key_points, (boxes if self.with_boxes else None)


class ForwardThresholdKeypointModel(KeypointModel):
    """Thresholds applied in ``forward``; ``post_process`` names no option."""

    def post_process(
        self, raw: torch.Tensor, sizes: list, **kwargs: Any
    ) -> Tuple[List[KeyPoints], Optional[List[Detections]]]:
        key_points, boxes = super().post_process(raw, sizes, confidence=0.0)
        # Record what the block passed, not the keywords used above.
        self.received[-1] = dict(kwargs)

        return key_points, boxes


class InstanceModel(FakeModel):
    """Instance segmentation: dense box-shaped masks, or RLE when asked."""

    supported_mask_formats = {"dense", "rle"}

    def __init__(self, request: Optional[ModelRequest] = None, *, rle_only=False):
        super().__init__(request)
        self.rle_only = rle_only

    def post_process(
        self,
        raw: torch.Tensor,
        sizes: list,
        confidence: Any = "default",
        iou_threshold: float = 0.3,
        max_detections: int = 300,
        class_agnostic_nms: bool = False,
        mask_format: str = "rle",
        **kwargs: Any,
    ) -> List[InstanceDetections]:
        threshold = self._record(
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            class_agnostic_nms=class_agnostic_nms,
            mask_format=mask_format,
            **kwargs,
        )
        predictions = []
        for brightness, (height, width) in zip(raw.tolist(), sizes):
            xyxy, class_id, scores = _rows(
                brightness, (height, width), threshold=threshold
            )
            masks = box_masks(xyxy, size_hw=(height, width))
            predictions.append(
                InstanceDetections(
                    xyxy=xyxy,
                    class_id=class_id,
                    confidence=scores,
                    mask=_as_rle(masks) if self.rle_only else masks,
                    image_metadata=self._note(),
                )
            )

        return predictions


class SemanticModel(FakeModel):
    """Semantic segmentation: left half background, right half one class."""

    class_names = ("background", "road", "car")

    def post_process(
        self, raw: torch.Tensor, sizes: list, confidence: Any = "default", **kwargs: Any
    ) -> List[SemanticSegmentationResult]:
        threshold = self._record(confidence=confidence, **kwargs)
        results = []
        for brightness, (height, width) in zip(raw.tolist(), sizes):
            labels = torch.zeros((height, width), dtype=torch.long)
            scores = torch.full((height, width), brightness)
            if brightness >= threshold:
                labels[:, width // 2 :] = 1 + int(brightness > 0.75)
            results.append(
                SemanticSegmentationResult(
                    segmentation_map=labels,
                    confidence=scores,
                    image_metadata=self._note(),
                )
            )

        return results


def box_masks(xyxy: torch.Tensor, *, size_hw: Tuple[int, int]) -> torch.Tensor:
    """``(N, H, W)`` bool masks filling each box."""
    height, width = size_hw
    masks = torch.zeros((len(xyxy), height, width), dtype=torch.bool)
    for row, (x1, y1, x2, y2) in enumerate(xyxy.long().tolist()):
        masks[row, y1:y2, x1:x2] = True

    return masks


def _as_rle(masks: torch.Tensor) -> InstancesRLEMasks:
    rle = InstancesRLEMasks.from_coco_rle_masks(
        tuple(masks.shape[1:]), [torch_mask_to_coco_rle(mask) for mask in masks]
    )

    return rle


class FakeLoader:
    """``ModelCache`` loader building one ``model_type`` per request."""

    def __init__(self, model_type: type, **model_options: Any):
        self.model_type = model_type
        self.model_options = model_options
        self.loaded: List[ModelRequest] = []
        self.models: List[FakeModel] = []

    def __call__(self, request: ModelRequest) -> FakeModel:
        self.loaded.append(request)
        model = self.model_type(request, **self.model_options)
        self.models.append(model)

        return model
