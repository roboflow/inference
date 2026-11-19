"""MediaPipe's hand landmarks model, converted to ONNX.

Stage two of the MediaPipe hands pipeline. Given an image and a rotated hand
region, it cuts the region out as a 224 x 224 crop and predicts 21 landmarks,
a hand presence score and handedness. Called without regions, each image is
treated as one upright hand crop.

Handedness is reported for the person in a normal (not mirrored) camera view.
MediaPipe's own labels assume a mirrored selfie view, so they are swapped
unless ``image_is_mirrored=True`` is passed.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from inference_models import Detections, KeyPoints, KeyPointsDetectionModel
from inference_models.entities import ColorFormat
from inference_models.errors import ModelInputError
from inference_models.models.common.roboflow.model_packages import PreProcessingMetadata
from inference_models.models.mediapipe_hands.common import OnnxGraph, graph_kwargs
from inference_models.models.mediapipe_hands.geometry import (
    HAND_LANDMARKS,
    LANDMARKS_INPUT_SIZE,
    RotatedRect,
    crop_rect,
    landmarks_box,
    project_landmarks,
    project_world_landmarks,
)
from inference_models.models.rfdetr_two_stage.rfdetr_key_points_stage2_onnx import (
    ImagesInput,
    identity_pre_processing_metadata,
    images_to_numpy_rgb,
)

DEFAULT_HAND_PRESENCE_THRESHOLD = 0.5
LEFT_HAND, RIGHT_HAND = 0, 1
HAND_CLASS_NAMES = ["left_hand", "right_hand"]
HAND_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]  # fmt: skip

# Output order of the converted graph.
_LANDMARKS, _PRESENCE, _HANDEDNESS, _WORLD_LANDMARKS = range(4)


@dataclass(frozen=True)
class HandPrediction:
    """One hand in image pixels."""

    landmarks: np.ndarray  # (21, 3): x, y in pixels, z relative depth in pixels
    world_landmarks: np.ndarray  # (21, 3) metres, origin at the hand's centre
    presence: float
    right_hand_score: float  # MediaPipe's raw "Right" score (mirrored-view convention)
    rect: RotatedRect

    def class_id(self, image_is_mirrored: bool) -> int:
        right = self.right_hand_score > 0.5
        if not image_is_mirrored:
            right = not right
        return RIGHT_HAND if right else LEFT_HAND

    def handedness_confidence(self) -> float:
        return max(self.right_hand_score, 1.0 - self.right_hand_score)


HandsBatch = Tuple[List[np.ndarray], List[List[RotatedRect]]]


class MediaPipeHandLandmarksONNX(
    KeyPointsDetectionModel[
        HandsBatch, List[PreProcessingMetadata], List[List[HandPrediction]]
    ]
):

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "MediaPipeHandLandmarksONNX":
        return cls(
            graph=OnnxGraph.from_package(model_name_or_path, **graph_kwargs(kwargs))
        )

    def __init__(self, graph: OnnxGraph):
        self._graph = graph

    @property
    def class_names(self) -> List[str]:
        return HAND_CLASS_NAMES

    @property
    def key_points_classes(self) -> List[List[str]]:
        return [HAND_LANDMARK_NAMES, HAND_LANDMARK_NAMES]

    @property
    def skeletons(self) -> List[List[Tuple[int, int]]]:
        return [HAND_SKELETON, HAND_SKELETON]

    @property
    def device(self) -> torch.device:
        return self._graph.device

    def predict_hands(
        self, images_rgb: List[np.ndarray], hand_rects: List[List[RotatedRect]]
    ) -> List[List[HandPrediction]]:
        """Landmarks for every region of every image, all crops in one batch."""
        crops, owners = [], []
        for image_index, (image, rects) in enumerate(zip(images_rgb, hand_rects)):
            for rect in rects:
                crops.append(crop_rect(image, rect, LANDMARKS_INPUT_SIZE))
                owners.append((image_index, rect))
        results: List[List[HandPrediction]] = [[] for _ in images_rgb]
        if not crops:
            return results
        batch = np.stack(crops).astype(np.float32) / 255.0
        outputs = self._graph.run(batch)
        for crop_index, (image_index, rect) in enumerate(owners):
            raw = outputs[_LANDMARKS][crop_index].reshape(HAND_LANDMARKS, 3)
            world = outputs[_WORLD_LANDMARKS][crop_index].reshape(HAND_LANDMARKS, 3)
            results[image_index].append(
                HandPrediction(
                    landmarks=project_landmarks(raw, rect, LANDMARKS_INPUT_SIZE),
                    world_landmarks=project_world_landmarks(world, rect),
                    presence=float(outputs[_PRESENCE][crop_index].reshape(-1)[0]),
                    right_hand_score=float(
                        outputs[_HANDEDNESS][crop_index].reshape(-1)[0]
                    ),
                    rect=rect,
                )
            )
        return results

    def pre_process(
        self,
        images: ImagesInput,
        input_color_format: Optional[ColorFormat] = None,
        hand_rects: Optional[List[List[RotatedRect]]] = None,
        **kwargs,
    ) -> Tuple[HandsBatch, List[PreProcessingMetadata]]:
        images_rgb = images_to_numpy_rgb(
            images=images, input_color_format=input_color_format
        )
        if hand_rects is None:
            hand_rects = [[_whole_image_rect(image)] for image in images_rgb]
        if len(hand_rects) != len(images_rgb):
            raise ModelInputError(
                message=f"Got hand regions for {len(hand_rects)} images, but {len(images_rgb)} images.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        return (images_rgb, hand_rects), [
            identity_pre_processing_metadata(image) for image in images_rgb
        ]

    def forward(
        self, pre_processed_images: HandsBatch, **kwargs
    ) -> List[List[HandPrediction]]:
        images_rgb, hand_rects = pre_processed_images
        return self.predict_hands(images_rgb, hand_rects)

    def post_process(
        self,
        model_results: List[List[HandPrediction]],
        pre_processing_meta: List[PreProcessingMetadata],
        hand_presence_threshold: float = DEFAULT_HAND_PRESENCE_THRESHOLD,
        image_is_mirrored: bool = False,
        **kwargs,
    ) -> Tuple[List[KeyPoints], List[Detections]]:
        all_key_points, all_detections = [], []
        for hands in model_results:
            hands = [hand for hand in hands if hand.presence >= hand_presence_threshold]
            key_points, detections = hands_to_results(
                hands=hands,
                scores=[hand.presence for hand in hands],
                image_is_mirrored=image_is_mirrored,
                device=self.device,
            )
            all_key_points.append(key_points)
            all_detections.append(detections)
        return all_key_points, all_detections


def hands_to_results(
    hands: List[HandPrediction],
    scores: List[float],
    image_is_mirrored: bool,
    device: torch.device,
) -> Tuple[KeyPoints, Detections]:
    """One keypoint instance and one box per hand; the box bounds the landmarks.

    MediaPipe predicts no per-landmark visibility, so every landmark carries the
    hand's score. Depth, metric landmarks and the handedness score travel in
    the per-instance metadata.
    """
    count = len(hands)
    landmarks = np.asarray(
        [hand.landmarks for hand in hands], dtype=np.float32
    ).reshape(count, HAND_LANDMARKS, 3)
    score = torch.as_tensor(scores, dtype=torch.float32, device=device).reshape(count)
    class_id = torch.as_tensor(
        [hand.class_id(image_is_mirrored) for hand in hands],
        dtype=torch.int64,
        device=device,
    ).reshape(count)
    metadata = [
        {
            "handedness_confidence": hand.handedness_confidence(),
            "hand_presence": hand.presence,
            "landmarks_z": hand.landmarks[:, 2].tolist(),
            "world_landmarks": hand.world_landmarks.tolist(),
        }
        for hand in hands
    ]
    boxes = np.asarray(
        [landmarks_box(hand.landmarks[:, :2]) for hand in hands], dtype=np.float32
    ).reshape(count, 4)
    key_points = KeyPoints(
        xy=torch.as_tensor(landmarks[:, :, :2], device=device).round().to(torch.int32),
        class_id=class_id,
        confidence=score[:, None].expand(count, HAND_LANDMARKS).clone(),
        key_points_metadata=metadata,
        detection_confidence=score,
    )
    detections = Detections(
        xyxy=torch.as_tensor(boxes, device=device).round().to(torch.int32),
        class_id=class_id,
        confidence=score,
        bboxes_metadata=metadata,
    )
    return key_points, detections


def _whole_image_rect(image: np.ndarray) -> RotatedRect:
    height, width = image.shape[:2]
    side = float(max(height, width))
    return RotatedRect(
        center_x=width / 2, center_y=height / 2, width=side, height=side, rotation=0.0
    )
