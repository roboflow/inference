"""MediaPipe's palm detector, converted to ONNX.

Stage one of the MediaPipe hands pipeline. It finds palms rather than whole
hands (a palm is close to rigid, so a square anchor fits it), and returns
seven palm key points per palm that orient the hand region stage two crops.
Each detection's metadata carries that region as ``hand_rect``
(``[center_x, center_y, width, height, rotation]`` in pixels).
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from inference_models import Detections, KeyPoints, KeyPointsDetectionModel
from inference_models.entities import ColorFormat, Confidence
from inference_models.models.common.roboflow.model_packages import PreProcessingMetadata
from inference_models.models.mediapipe_hands.common import (
    OnnxGraph,
    graph_kwargs,
    resolve_confidence,
)
from inference_models.models.mediapipe_hands.geometry import (
    PALM_INPUT_SIZE,
    PalmDetection,
    decode_palm_detections,
    generate_palm_anchors,
    letterbox_matrix,
    palm_to_hand_rect,
    rect_to_list,
)
from inference_models.models.rfdetr_two_stage.rfdetr_key_points_stage2_onnx import (
    ImagesInput,
    identity_pre_processing_metadata,
    images_to_numpy_rgb,
)

DEFAULT_PALM_CONFIDENCE = 0.5
DEFAULT_MAX_NUM_PALMS = 2
PALM_CLASS_NAMES = ["palm"]
PALM_KEY_POINT_NAMES = [
    "wrist_center",
    "index_finger_mcp",
    "middle_finger_mcp",
    "ring_finger_mcp",
    "pinky_mcp",
    "thumb_cmc",
    "thumb_mcp",
]

PalmBatch = Tuple[np.ndarray, List[np.ndarray]]


class MediaPipePalmDetectorONNX(
    KeyPointsDetectionModel[
        PalmBatch, List[PreProcessingMetadata], List[List[PalmDetection]]
    ]
):

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "MediaPipePalmDetectorONNX":
        return cls(
            graph=OnnxGraph.from_package(model_name_or_path, **graph_kwargs(kwargs))
        )

    def __init__(self, graph: OnnxGraph):
        self._graph = graph
        self._anchors = generate_palm_anchors()

    @property
    def class_names(self) -> List[str]:
        return PALM_CLASS_NAMES

    @property
    def key_points_classes(self) -> List[List[str]]:
        return [PALM_KEY_POINT_NAMES]

    @property
    def skeletons(self) -> List[List[Tuple[int, int]]]:
        return [[]]

    def detect_palms(
        self,
        images_rgb: List[np.ndarray],
        confidence: float = DEFAULT_PALM_CONFIDENCE,
        max_num_palms: int = DEFAULT_MAX_NUM_PALMS,
    ) -> List[List[PalmDetection]]:
        """Palms per image, in image pixels. Images are uint8 RGB."""
        batch, image_to_input = self._letterbox(images_rgb)
        return self._decode(
            outputs=self._graph.run(batch) if len(batch) else [],
            image_to_input=image_to_input,
            confidence=confidence,
            max_num_palms=max_num_palms,
        )

    def pre_process(
        self,
        images: ImagesInput,
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[PalmBatch, List[PreProcessingMetadata]]:
        images_rgb = images_to_numpy_rgb(
            images=images, input_color_format=input_color_format
        )
        return self._letterbox(images_rgb), [
            identity_pre_processing_metadata(image) for image in images_rgb
        ]

    def forward(
        self,
        pre_processed_images: PalmBatch,
        confidence: Confidence = "default",
        max_num_palms: int = DEFAULT_MAX_NUM_PALMS,
        **kwargs,
    ) -> List[List[PalmDetection]]:
        batch, image_to_input = pre_processed_images
        return self._decode(
            outputs=self._graph.run(batch) if len(batch) else [],
            image_to_input=image_to_input,
            confidence=resolve_confidence(confidence, DEFAULT_PALM_CONFIDENCE),
            max_num_palms=max_num_palms,
        )

    def post_process(
        self,
        model_results: List[List[PalmDetection]],
        pre_processing_meta: List[PreProcessingMetadata],
        **kwargs,
    ) -> Tuple[List[KeyPoints], List[Detections]]:
        device = self._graph.device
        all_key_points, all_detections = [], []
        for palms in model_results:
            count = len(palms)
            xy = torch.as_tensor(
                np.asarray(
                    [palm.key_points for palm in palms], dtype=np.float32
                ).reshape(count, len(PALM_KEY_POINT_NAMES), 2),
                device=device,
            )
            scores = torch.as_tensor(
                [palm.score for palm in palms], dtype=torch.float32, device=device
            )
            class_id = torch.zeros((count,), dtype=torch.int64, device=device)
            all_key_points.append(
                KeyPoints(
                    xy=xy.round().to(torch.int32),
                    class_id=class_id,
                    confidence=scores[:, None]
                    .expand(count, len(PALM_KEY_POINT_NAMES))
                    .clone(),
                    detection_confidence=scores,
                )
            )
            all_detections.append(
                Detections(
                    xyxy=torch.as_tensor(
                        np.asarray(
                            [palm.xyxy for palm in palms], dtype=np.float32
                        ).reshape(count, 4),
                        device=device,
                    )
                    .round()
                    .to(torch.int32),
                    class_id=class_id,
                    confidence=scores,
                    bboxes_metadata=[
                        {"hand_rect": rect_to_list(palm_to_hand_rect(palm))}
                        for palm in palms
                    ],
                )
            )
        return all_key_points, all_detections

    def _letterbox(self, images_rgb: List[np.ndarray]) -> PalmBatch:
        size = PALM_INPUT_SIZE
        batch = np.zeros((len(images_rgb), size, size, 3), dtype=np.float32)
        matrices = []
        for index, image in enumerate(images_rgb):
            matrix = letterbox_matrix(image.shape[1], image.shape[0], size)
            batch[index] = (
                cv2.warpAffine(
                    image,
                    matrix,
                    (size, size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(np.float32)
                / 255.0
            )
            matrices.append(matrix)
        return batch, matrices

    def _decode(
        self,
        outputs: List[np.ndarray],
        image_to_input: List[np.ndarray],
        confidence: float,
        max_num_palms: int,
    ) -> List[List[PalmDetection]]:
        if not image_to_input:
            return []
        raw_boxes, raw_scores = outputs
        return [
            decode_palm_detections(
                raw_boxes=raw_boxes[index],
                raw_scores=raw_scores[index],
                anchors=self._anchors,
                image_to_input=matrix,
                score_threshold=confidence,
                max_detections=max_num_palms,
            )
            for index, matrix in enumerate(image_to_input)
        ]
