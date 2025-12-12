from typing import List, Tuple, Union

import numpy as np
import torch
import torchvision
from inference_exp import Detections, OpenVocabularyObjectDetectionModel
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ImageDimensions
from inference_exp.models.base.types import PreprocessedInputs, PreprocessingMetadata
from inference_exp.models.common.roboflow.pre_processing import (
    extract_input_images_dimensions,
)
from inference_exp.models.owlv2.reference_dataset import ReferenceExample
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers.models.owlv2.modeling_owlv2 import Owlv2ObjectDetectionOutput


class OWLv2HF(
    OpenVocabularyObjectDetectionModel[
        torch.Tensor, List[ImageDimensions], Owlv2ObjectDetectionOutput
    ]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        local_files_only: bool = True,
        **kwargs,
    ) -> "OpenVocabularyObjectDetectionModel":
        processor = Owlv2Processor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            use_fast=True,
        )
        model = Owlv2ForObjectDetection.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        ).to(device)
        return cls(
            model=model,
            processor=processor,
            device=device,
        )

    def __init__(
        self,
        model: Owlv2ForObjectDetection,
        processor: Owlv2Processor,
        device: torch.device,
    ):
        self._model = model
        self._processor = processor
        self._device = device

    def optimize_for_inference(self) -> None:
        self._model.owlv2.vision_model = torch.compile(self._model.owlv2.vision_model)
        example_image = torch.randint(
            low=0, high=255, size=(3, 128, 128), dtype=torch.uint8
        ).to(self._device)
        _ = self.infer(example_image, ["some", "other"])

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[PreprocessedInputs, PreprocessingMetadata]:
        image_dimensions = extract_input_images_dimensions(images=images)
        inputs = self._processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(self._device), image_dimensions

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        classes: List[str],
        **kwargs,
    ) -> Owlv2ObjectDetectionOutput:
        input_ids = self._processor(text=[classes], return_tensors="pt")[
            "input_ids"
        ].to(self._device)
        with torch.inference_mode():
            return self._model(input_ids=input_ids, pixel_values=pre_processed_images)

    def post_process(
        self,
        model_results: Owlv2ObjectDetectionOutput,
        pre_processing_meta: List[ImageDimensions],
        conf_thresh: float = 0.1,
        iou_thresh: float = 0.45,
        class_agnostic: bool = False,
        max_detections: int = 100,
        **kwargs,
    ) -> List[Detections]:
        target_sizes = [(dim.height, dim.width) for dim in pre_processing_meta]
        post_processed_outputs = self._processor.post_process_grounded_object_detection(
            outputs=model_results,
            target_sizes=target_sizes,
            threshold=conf_thresh,
        )
        results = []
        for i in range(len(post_processed_outputs)):
            boxes, scores, labels = (
                post_processed_outputs[i]["boxes"],
                post_processed_outputs[i]["scores"],
                post_processed_outputs[i]["labels"],
            )
            nms_class_ids = torch.zeros_like(labels) if class_agnostic else labels
            keep = torchvision.ops.batched_nms(boxes, scores, nms_class_ids, iou_thresh)
            keep = keep[:max_detections]
            results.append(
                Detections(
                    xyxy=boxes[keep].contiguous().int(),
                    class_id=labels[keep].contiguous(),
                    confidence=scores[keep].contiguous(),
                )
            )
        return results

    def infer_with_reference_examples(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        reference_examples: List[ReferenceExample],
    ) -> List[Detections]:
        pass
