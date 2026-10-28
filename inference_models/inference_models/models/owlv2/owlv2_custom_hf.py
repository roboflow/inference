import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torchvision.ops import batched_nms

from inference_models import Detections, ObjectDetectionModel
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_OWLV2_DEFAULT_CLASS_AGNOSTIC_NMS,
    INFERENCE_MODELS_OWLV2_DEFAULT_IOU_THRESHOLD,
    INFERENCE_MODELS_OWLV2_DEFAULT_MAX_DETECTIONS,
)
from inference_models.entities import ImageDimensions
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.models.common.roboflow.pre_processing import images_to_pillow
from inference_models.models.owlv2.owlv2_hf import OWLv2HF


@dataclass
class OWLv2PreprocessingMetadata:
    original_size: ImageDimensions


class OWLv2CustomHF(ObjectDetectionModel):
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device=DEFAULT_DEVICE,
        local_files_only: bool = True,
        recommended_parameters=None,
        **kwargs,
    ) -> "OWLv2CustomHF":
        package = Path(model_name_or_path)
        try:
            class_names = (package / "class_names.txt").read_text().splitlines()
            prompts = json.loads((package / "owlv2_prompts.json").read_text())
            if (
                not class_names
                or len(set(class_names)) != len(class_names)
                or any(not name.strip() or name != name.strip() for name in class_names)
                or not isinstance(prompts, list)
                or len(prompts) != len(class_names)
                or any(
                    not isinstance(prompt, str) or not prompt.strip()
                    for prompt in prompts
                )
            ):
                raise ValueError(
                    "Classes and prompts must be nonempty and have matching lengths."
                )
        except (OSError, ValueError) as error:
            raise CorruptedModelPackageError(
                message=f"Invalid OWLv2 classes or prompts: {error}",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            ) from error
        model = OWLv2HF.from_pretrained(
            model_name_or_path,
            device=device,
            local_files_only=local_files_only,
            **kwargs,
        )
        return cls(model, class_names, prompts, recommended_parameters)

    def __init__(
        self,
        model: OWLv2HF,
        class_names: List[str],
        prompts: List[str],
        recommended_parameters=None,
    ):
        self._model = model
        self._class_names = class_names
        self._prompts = prompts
        self.recommended_parameters = recommended_parameters
        self.image_size = model._model.config.vision_config.image_size

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(self, images, input_color_format=None, **kwargs):
        images, dimensions = images_to_pillow(
            images, input_color_format=input_color_format
        )
        inputs = self._model._processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(self._model._device), [
            OWLv2PreprocessingMetadata(original_size=dimension)
            for dimension in dimensions
        ]

    def forward(self, pre_processed_images, **kwargs):
        return self._model.forward(pre_processed_images, classes=self._prompts)

    def post_process(
        self,
        model_results,
        pre_processing_meta,
        confidence="default",
        iou_threshold=INFERENCE_MODELS_OWLV2_DEFAULT_IOU_THRESHOLD,
        class_agnostic_nms=INFERENCE_MODELS_OWLV2_DEFAULT_CLASS_AGNOSTIC_NMS,
        max_detections=INFERENCE_MODELS_OWLV2_DEFAULT_MAX_DETECTIONS,
        **kwargs,
    ):
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        threshold = confidence_filter.get_threshold(self.class_names)
        minimum = (
            float(threshold.min().item())
            if isinstance(threshold, torch.Tensor)
            else threshold
        )
        results = self._model._processor.post_process_grounded_object_detection(
            outputs=model_results,
            target_sizes=[
                (metadata.original_size.height, metadata.original_size.width)
                for metadata in pre_processing_meta
            ],
            threshold=minimum,
        )
        detections = []
        for result in results:
            boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
            if isinstance(threshold, torch.Tensor):
                per_class = threshold.to(device=scores.device, dtype=scores.dtype)
                valid = scores >= per_class[labels.long()]
                boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
            nms_labels = torch.zeros_like(labels) if class_agnostic_nms else labels
            keep = batched_nms(boxes, scores, nms_labels, iou_threshold)[
                :max_detections
            ]
            detections.append(
                Detections(
                    xyxy=boxes[keep].int(),
                    class_id=labels[keep].int(),
                    confidence=scores[keep],
                )
            )
        return detections
