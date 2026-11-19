"""MediaPipe hands: hand detection followed by 21 hand landmarks, served with ONNX Runtime.

A composite registered model, composed by the auto-loader like the RF-DETR
two-stage keypoint model. Its package carries only a config file and names two
dependencies:

* ``handLandmarks``: MediaPipe's hand landmarks model,
* the stage that finds hands, either ``palmDetector`` (MediaPipe's palm
  detector, which reproduces MediaPipe's own pipeline) or ``handDetector``
  (any object detection model whose boxes are hands, for example one trained
  on gloved hands, which MediaPipe's palm detector misses).

From a palm the hand region is rotated and sized the way MediaPipe does it.
A ``handDetector`` box carries no orientation, so its first landmark pass runs
on an upright square around the box and a second pass runs on the region those
landmarks give, the same step MediaPipe uses to follow a hand between frames.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inference_models import Detections, KeyPoints, KeyPointsDetectionModel
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.base.object_detection import ObjectDetectionModel
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import PreProcessingMetadata
from inference_models.models.mediapipe_hands.common import resolve_confidence
from inference_models.models.mediapipe_hands.geometry import (
    RotatedRect,
    box_to_hand_rect,
    landmarks_to_hand_rect,
    palm_to_hand_rect,
)
from inference_models.models.mediapipe_hands.hand_landmarks_onnx import (
    DEFAULT_HAND_PRESENCE_THRESHOLD,
    HAND_CLASS_NAMES,
    HAND_LANDMARK_NAMES,
    HAND_SKELETON,
    HandPrediction,
    MediaPipeHandLandmarksONNX,
    hands_to_results,
)
from inference_models.models.mediapipe_hands.palm_detector_onnx import (
    DEFAULT_MAX_NUM_PALMS,
    DEFAULT_PALM_CONFIDENCE,
    MediaPipePalmDetectorONNX,
)
from inference_models.models.rfdetr_two_stage.rfdetr_key_points_stage2_onnx import (
    ImagesInput,
    identity_pre_processing_metadata,
    images_to_numpy_rgb,
)

CONFIG_FILE = "mediapipe_hands_config.json"
PALM_DETECTOR_DEPENDENCY = "palmDetector"
HAND_DETECTOR_DEPENDENCY = "handDetector"
HAND_LANDMARKS_DEPENDENCY = "handLandmarks"
# A hand detector's box is tight around the whole hand; the landmark model was
# trained on regions about twice the palm, which this square comes close to.
DETECTOR_BOX_SCALE = 1.3

HandsResult = List[List[Tuple[HandPrediction, float]]]


class MediaPipeHandsONNX(
    KeyPointsDetectionModel[List[np.ndarray], List[PreProcessingMetadata], HandsResult]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        model_dependencies: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "MediaPipeHandsONNX":
        get_model_package_contents(
            model_package_dir=model_name_or_path, elements=[CONFIG_FILE]
        )
        model_dependencies = model_dependencies or {}
        landmarks = model_dependencies.get(HAND_LANDMARKS_DEPENDENCY)
        palm_detector = model_dependencies.get(PALM_DETECTOR_DEPENDENCY)
        hand_detector = model_dependencies.get(HAND_DETECTOR_DEPENDENCY)
        if not isinstance(landmarks, MediaPipeHandLandmarksONNX):
            raise _corrupted(
                f"Dependency `{HAND_LANDMARKS_DEPENDENCY}` must be the MediaPipe hand landmarks model, "
                f"got {type(landmarks).__name__}."
            )
        if (palm_detector is None) == (hand_detector is None):
            raise _corrupted(
                f"MediaPipe hands needs exactly one of `{PALM_DETECTOR_DEPENDENCY}` and "
                f"`{HAND_DETECTOR_DEPENDENCY}`, the auto-loader provided {sorted(model_dependencies)}."
            )
        if palm_detector is not None and not isinstance(
            palm_detector, MediaPipePalmDetectorONNX
        ):
            raise _corrupted(
                f"Dependency `{PALM_DETECTOR_DEPENDENCY}` must be the MediaPipe palm detector, "
                f"got {type(palm_detector).__name__}."
            )
        if hand_detector is not None and not isinstance(
            hand_detector, ObjectDetectionModel
        ):
            raise _corrupted(
                f"Dependency `{HAND_DETECTOR_DEPENDENCY}` must be an object detection model, "
                f"got {type(hand_detector).__name__}."
            )
        return cls(
            landmarks=landmarks,
            palm_detector=palm_detector,
            hand_detector=hand_detector,
        )

    def __init__(
        self,
        landmarks: MediaPipeHandLandmarksONNX,
        palm_detector: Optional[MediaPipePalmDetectorONNX] = None,
        hand_detector: Optional[ObjectDetectionModel] = None,
    ):
        self._landmarks = landmarks
        self._palm_detector = palm_detector
        self._hand_detector = hand_detector

    @property
    def class_names(self) -> List[str]:
        return HAND_CLASS_NAMES

    @property
    def key_points_classes(self) -> List[List[str]]:
        return [HAND_LANDMARK_NAMES, HAND_LANDMARK_NAMES]

    @property
    def skeletons(self) -> List[List[Tuple[int, int]]]:
        return [HAND_SKELETON, HAND_SKELETON]

    def pre_process(
        self,
        images: ImagesInput,
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[List[np.ndarray], List[PreProcessingMetadata]]:
        images_rgb = images_to_numpy_rgb(
            images=images, input_color_format=input_color_format
        )
        return images_rgb, [
            identity_pre_processing_metadata(image) for image in images_rgb
        ]

    def forward(
        self,
        pre_processed_images: List[np.ndarray],
        confidence: Confidence = "default",
        max_num_hands: int = DEFAULT_MAX_NUM_PALMS,
        **kwargs,
    ) -> HandsResult:
        """Hands per image, each with its detection score (palm or detector box)."""
        images_rgb = pre_processed_images
        if self._palm_detector is not None:
            palms = self._palm_detector.detect_palms(
                images_rgb,
                confidence=resolve_confidence(confidence, DEFAULT_PALM_CONFIDENCE),
                max_num_palms=max_num_hands,
            )
            rects = [
                [palm_to_hand_rect(palm) for palm in image_palms]
                for image_palms in palms
            ]
            hands = self._landmarks.predict_hands(images_rgb, rects)
            scores = [[palm.score for palm in image_palms] for image_palms in palms]
            return _pair(hands, scores)
        rects, scores = self._detect_hand_boxes(images_rgb, confidence, max_num_hands)
        first_pass = self._landmarks.predict_hands(images_rgb, rects)
        refined_rects = [
            [landmarks_to_hand_rect(hand.landmarks[:, :2]) for hand in image_hands]
            for image_hands in first_pass
        ]
        hands = self._landmarks.predict_hands(images_rgb, refined_rects)
        return _pair(hands, scores)

    def post_process(
        self,
        model_results: HandsResult,
        pre_processing_meta: List[PreProcessingMetadata],
        hand_presence_threshold: float = DEFAULT_HAND_PRESENCE_THRESHOLD,
        image_is_mirrored: bool = False,
        **kwargs,
    ) -> Tuple[List[KeyPoints], List[Detections]]:
        """Drops regions the landmark model says hold no hand.

        The reported score is the detection score times hand presence, the
        same fusion the RF-DETR two-stage keypoint model reports.
        """
        all_key_points, all_detections = [], []
        for image_hands in model_results:
            kept = [
                (hand, score)
                for hand, score in image_hands
                if hand.presence >= hand_presence_threshold
            ]
            key_points, detections = hands_to_results(
                hands=[hand for hand, _ in kept],
                scores=[score * hand.presence for hand, score in kept],
                image_is_mirrored=image_is_mirrored,
                device=self._landmarks.device,
            )
            all_key_points.append(key_points)
            all_detections.append(detections)
        return all_key_points, all_detections

    def _detect_hand_boxes(
        self,
        images_rgb: List[np.ndarray],
        confidence: Confidence,
        max_num_hands: int,
    ) -> Tuple[List[List[RotatedRect]], List[List[float]]]:
        detections: List[Detections] = self._hand_detector.infer(
            images_rgb, input_color_format="rgb", confidence=confidence
        )
        rects, scores = [], []
        for image_detections in detections:
            boxes = image_detections.xyxy.detach().cpu().numpy().astype(np.float32)
            box_scores = (
                image_detections.confidence.detach().cpu().numpy().astype(np.float32)
            )
            order = np.argsort(-box_scores, kind="stable")
            if max_num_hands > 0:
                order = order[:max_num_hands]
            rects.append(
                [box_to_hand_rect(boxes[index], DETECTOR_BOX_SCALE) for index in order]
            )
            scores.append([float(box_scores[index]) for index in order])
        return rects, scores


def _pair(hands: List[List[HandPrediction]], scores: List[List[float]]) -> HandsResult:
    return [
        list(zip(image_hands, image_scores))
        for image_hands, image_scores in zip(hands, scores)
    ]


def _corrupted(message: str) -> CorruptedModelPackageError:
    return CorruptedModelPackageError(
        message=message,
        help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
    )
