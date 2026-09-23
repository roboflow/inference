"""RF-DETR two-stage keypoint detection.

A composite registered model with two dependencies: an RF-DETR object
detector (``detector``) that finds the objects, and a stage-2 keypoint model
(``keypointHead``) that predicts each object's keypoints from its crop. The
package of the composite itself carries only a config file; the auto-loader
resolves both dependencies from the registry and passes them in as
``model_dependencies``, the way PP-OCR composes its detection and
recognition models.

Stage one runs once over the whole batch. Stage two runs once per image over
all of that image's crops, batched inside the keypoint model. Keypoint classes
follow the detector's classes by name, so a detected class the keypoint model
was not trained for is returned with no keypoints and keeps the detector's
confidence. For every other detection the confidence is the product of the
detector's box confidence and stage two's instance score, the fusion the
trainer evaluates the model with, reported on both the keypoints and the
detections.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from inference_models import Detections, KeyPoints, KeyPointsDetectionModel
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.base.object_detection import ObjectDetectionModel
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.rfdetr_two_stage.rfdetr_key_points_stage2_onnx import (
    ImagesInput,
    RFDetrKeyPointsStage2ONNX,
    images_to_numpy_rgb,
)

CONFIG_FILE = "two_stage_config.json"
DETECTOR_DEPENDENCY = "detector"
KEYPOINT_HEAD_DEPENDENCY = "keypointHead"
NO_KEYPOINTS = -1

TwoStageResult = Tuple[List[KeyPoints], List[Detections]]


class RFDetrTwoStageKeyPointsONNX(
    KeyPointsDetectionModel[List[np.ndarray], None, TwoStageResult]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        model_dependencies: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "RFDetrTwoStageKeyPointsONNX":
        get_model_package_contents(
            model_package_dir=model_name_or_path, elements=[CONFIG_FILE]
        )
        model_dependencies = model_dependencies or {}
        missing = [
            name
            for name in (DETECTOR_DEPENDENCY, KEYPOINT_HEAD_DEPENDENCY)
            if name not in model_dependencies
        ]
        if missing:
            raise CorruptedModelPackageError(
                message=f"RF-DETR two-stage keypoint model requires dependency models {missing}, "
                f"but the auto-loader provided {sorted(model_dependencies)}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        detector = model_dependencies[DETECTOR_DEPENDENCY]
        keypoint_head = model_dependencies[KEYPOINT_HEAD_DEPENDENCY]
        if not isinstance(detector, ObjectDetectionModel):
            raise CorruptedModelPackageError(
                message=f"Dependency `{DETECTOR_DEPENDENCY}` must be an object detection model, "
                f"got {type(detector).__name__}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if not isinstance(keypoint_head, RFDetrKeyPointsStage2ONNX):
            raise CorruptedModelPackageError(
                message=f"Dependency `{KEYPOINT_HEAD_DEPENDENCY}` must be a stage-2 keypoint model, "
                f"got {type(keypoint_head).__name__}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        return cls(detector=detector, keypoint_head=keypoint_head)

    def __init__(
        self, detector: ObjectDetectionModel, keypoint_head: RFDetrKeyPointsStage2ONNX
    ):
        self._detector = detector
        self._keypoint_head = keypoint_head
        head_class_index = {
            name: index for index, name in enumerate(keypoint_head.class_names)
        }
        self._head_class_for_detector_class = [
            head_class_index.get(name, NO_KEYPOINTS) for name in detector.class_names
        ]
        self._key_points_classes = [
            (
                keypoint_head.key_points_classes[head_class]
                if head_class != NO_KEYPOINTS
                else []
            )
            for head_class in self._head_class_for_detector_class
        ]
        self._skeletons = [
            keypoint_head.skeletons[head_class] if head_class != NO_KEYPOINTS else []
            for head_class in self._head_class_for_detector_class
        ]
        self._max_key_points = max(
            (len(names) for names in self._key_points_classes), default=0
        )

    @property
    def class_names(self) -> List[str]:
        return self._detector.class_names

    @property
    def key_points_classes(self) -> List[List[str]]:
        return self._key_points_classes

    @property
    def skeletons(self) -> List[List[Tuple[int, int]]]:
        return self._skeletons

    def pre_process(
        self,
        images: ImagesInput,
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[List[np.ndarray], None]:
        return (
            images_to_numpy_rgb(images=images, input_color_format=input_color_format),
            None,
        )

    def forward(
        self,
        pre_processed_images: List[np.ndarray],
        confidence: Confidence = "default",
        key_points_threshold: Optional[float] = None,
        **kwargs,
    ) -> TwoStageResult:
        images_rgb = pre_processed_images
        per_image_detections: List[Detections] = self._detector.infer(
            images_rgb, input_color_format="rgb", confidence=confidence, **kwargs
        )
        boxes, class_ids, confidences, matched_rows = [], [], [], []
        for detections in per_image_detections:
            detector_classes = (
                detections.class_id.detach().cpu().numpy().astype(np.int64)
            )
            head_classes = np.asarray(
                [self._head_class_for_detector_class[int(c)] for c in detector_classes],
                dtype=np.int64,
            )
            rows = np.flatnonzero(head_classes != NO_KEYPOINTS)
            boxes.append(
                detections.xyxy.detach().cpu().numpy().astype(np.float32)[rows]
            )
            class_ids.append(head_classes[rows])
            confidences.append(
                detections.confidence.detach().cpu().numpy().astype(np.float32)[rows]
            )
            matched_rows.append(rows)
        head_kwargs = (
            {"key_points_threshold": key_points_threshold}
            if key_points_threshold is not None
            else {}
        )
        head_key_points, _ = self._keypoint_head.infer(
            images_rgb,
            boxes=boxes,
            class_ids=class_ids,
            detection_confidences=confidences,
            input_color_format="rgb",
            **head_kwargs,
        )
        all_key_points, all_detections = [], []
        for key_points, detections, rows in zip(
            head_key_points, per_image_detections, matched_rows
        ):
            fused = self._align_to_detections(key_points, detections, rows)
            all_key_points.append(fused)
            all_detections.append(
                Detections(
                    xyxy=detections.xyxy,
                    class_id=detections.class_id,
                    confidence=fused.detection_confidence,
                    image_metadata=detections.image_metadata,
                    bboxes_metadata=detections.bboxes_metadata,
                )
            )
        return all_key_points, all_detections

    def post_process(
        self, model_results: TwoStageResult, pre_processing_meta: None, **kwargs
    ) -> Tuple[List[KeyPoints], Optional[List[Detections]]]:
        return model_results

    def _align_to_detections(
        self, key_points: KeyPoints, detections: Detections, matched_rows: np.ndarray
    ) -> KeyPoints:
        """One keypoint row per detection, in detection order; unmatched rows are empty."""
        device = detections.xyxy.device
        instances = int(detections.xyxy.shape[0])
        xy = torch.zeros(
            (instances, self._max_key_points, 2), dtype=torch.int32, device=device
        )
        confidence = torch.zeros((instances, self._max_key_points), device=device)
        covariance = torch.full(
            (instances, self._max_key_points, 2, 2), float("nan"), device=device
        )
        detection_confidence = detections.confidence.clone()
        if len(matched_rows) > 0:
            rows = torch.as_tensor(matched_rows, dtype=torch.long, device=device)
            slots = key_points.xy.shape[1]
            xy[rows, :slots] = key_points.xy.to(device)
            confidence[rows, :slots] = key_points.confidence.to(device)
            if key_points.covariance is not None:
                covariance[rows, :slots] = key_points.covariance.to(device)
            # Stage two already multiplied the detector confidence it was given
            # by its own instance score.
            detection_confidence[rows] = key_points.detection_confidence.to(
                device=device, dtype=detection_confidence.dtype
            )
        return KeyPoints(
            xy=xy,
            class_id=detections.class_id,
            confidence=confidence,
            covariance=covariance,
            detection_confidence=detection_confidence,
        )
