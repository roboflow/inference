from dataclasses import dataclass
from typing import List, Optional, Union

import cv2
import numpy as np
import torch

from inference_models.entities import ColorFormat
from inference_models.models.auto_loaders.core import AutoModel
from inference_models.models.base.object_detection import Detections
from inference_models.models.pp_ocrv6.pp_ocrv6_common import (
    images_to_numpy_bgr_for_cropping,
)

DEFAULT_DETECTION_MODEL = "pp-ocrv6-det/small"
DEFAULT_RECOGNITION_MODEL = "pp-ocrv6-rec/small"


@dataclass
class PPOCRv6PipelineResult:
    """Result of a single image passed through the pipeline.

    ``detections`` is ``None`` when the detection stage did not run
    (recognition-only pipeline); it holds zero boxes when the detector ran
    but found no text.
    """

    text: str
    line_texts: List[str]
    detections: Optional[Detections]


class PPOCRv6Pipeline:
    """Two-stage PP-OCRv6 pipeline: detect text lines, then recognize each crop.

    Detected regions are perspective-cropped via the tight quadrilateral stored
    in ``Detections.bboxes_metadata["polygon"]`` and grouped into reading order
    (top-to-bottom lines, each left-to-right) before recognition.
    """

    def __init__(self, det_model=None, rec_model=None):
        if det_model is None and rec_model is None:
            raise ValueError(
                "PPOCRv6Pipeline requires at least one of det_model or rec_model."
            )
        self._det_model = det_model
        self._rec_model = rec_model

    @classmethod
    def from_pretrained(
        cls,
        det_model_name_or_path: Optional[str] = DEFAULT_DETECTION_MODEL,
        rec_model_name_or_path: Optional[str] = DEFAULT_RECOGNITION_MODEL,
        **kwargs,
    ) -> "PPOCRv6Pipeline":
        det_model = (
            AutoModel.from_pretrained(det_model_name_or_path, **kwargs)
            if det_model_name_or_path is not None
            else None
        )
        rec_model = (
            AutoModel.from_pretrained(rec_model_name_or_path, **kwargs)
            if rec_model_name_or_path is not None
            else None
        )
        return cls(det_model=det_model, rec_model=rec_model)

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> List[PPOCRv6PipelineResult]:
        if self._det_model is None:
            line_texts = self._rec_model(
                images, input_color_format=input_color_format, **kwargs
            )
            return [
                PPOCRv6PipelineResult(
                    text=line_text,
                    line_texts=[line_text],
                    detections=None,
                )
                for line_text in line_texts
            ]
        per_image_detections = self._det_model(
            images, input_color_format=input_color_format, **kwargs
        )
        source_images = images_to_numpy_bgr_for_cropping(
            images=images, input_color_format=input_color_format
        )
        results = []
        for image, detections in zip(source_images, per_image_detections):
            results.append(self._recognize_image(image=image, detections=detections))
        return results

    def _recognize_image(
        self, image: np.ndarray, detections: Detections
    ) -> PPOCRv6PipelineResult:
        order = _reading_order(detections)
        if not order:
            return PPOCRv6PipelineResult(text="", line_texts=[], detections=detections)
        ordered_detections = _reorder_detections(detections, order)
        if self._rec_model is None:
            return PPOCRv6PipelineResult(
                text="", line_texts=[], detections=ordered_detections
            )
        crops = [
            _rotate_crop(image, meta["polygon"])
            for meta in ordered_detections.bboxes_metadata
        ]
        line_texts = self._rec_model(crops)
        return PPOCRv6PipelineResult(
            text="\n".join(line_texts),
            line_texts=line_texts,
            detections=ordered_detections,
        )

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[PPOCRv6PipelineResult]:
        return self.infer(images, **kwargs)


def _reading_order(detections: Detections) -> List[int]:
    """Order detection indices top-to-bottom by line, each line left-to-right."""
    boxes = detections.xyxy.tolist()
    items = sorted(range(len(boxes)), key=lambda index: boxes[index][1])
    lines, current, line_bottom = [], [], None
    for index in items:
        top, bottom = boxes[index][1], boxes[index][3]
        if line_bottom is None or top >= line_bottom - 0.5 * (bottom - top):
            if current:
                lines.append(current)
            current, line_bottom = [index], bottom
        else:
            current.append(index)
            line_bottom = max(line_bottom, bottom)
    if current:
        lines.append(current)
    ordered = []
    for line in lines:
        ordered.extend(sorted(line, key=lambda index: boxes[index][0]))
    return ordered


def _reorder_detections(detections: Detections, order: List[int]) -> Detections:
    index = torch.as_tensor(order, dtype=torch.long)
    return Detections(
        xyxy=detections.xyxy[index],
        class_id=detections.class_id[index],
        confidence=detections.confidence[index],
        image_metadata=detections.image_metadata,
        bboxes_metadata=[detections.bboxes_metadata[i] for i in order],
    )


def _rotate_crop(image: np.ndarray, quad: list) -> np.ndarray:
    """Perspective-crop the quadrilateral text region into an upright rectangle."""
    quad = np.array(quad, dtype="float32")
    width = int(
        max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3]))
    )
    height = int(
        max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2]))
    )
    matrix = cv2.getPerspectiveTransform(
        quad, np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    )
    crop = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if height > 0 and width > 0 and height / float(width) >= 1.5:
        crop = np.rot90(crop)
    return crop
