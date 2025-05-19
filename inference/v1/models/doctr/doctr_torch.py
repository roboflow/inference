from typing import Tuple, List, Union, Optional, Callable

import numpy as np
import torch
from doctr.io import Document

from inference.v1 import Detections
from inference.v1.entities import ImageDimensions, ColorFormat
from inference.v1.errors import ModelRuntimeError
from inference.v1.models.base.documents_parsing import DocumentParsingModel


class DocTR(DocumentParsingModel[List[np.ndarray], ImageDimensions, Document]):

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "DocumentParsingModel":
        pass

    def __init__(
        self,
        model: Callable[[List[np.ndarray]], Document],
        device: torch.device,
    ):
        self._model = model
        self._device = device

    @property
    def class_names(self) -> List[str]:
        return ["block", "line", "word"]

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[List[np.ndarray], List[ImageDimensions]]:
        if isinstance(images, np.ndarray):
            input_color_format = input_color_format or "bgr"
            if input_color_format != "bgr":
                images = images[:, :, ::-1]
            h, w = images.shape[:2]
            return [images], [ImageDimensions(height=h, width=w)]
        if isinstance(images, torch.Tensor):
            input_color_format = input_color_format or "rgb"
            if len(images.shape) == 3:
                images = torch.unsqueeze(images, dim=0)
            if input_color_format != "bgr":
                images = images[:, [2, 1, 0], :, :]
            result = []
            dimensions = []
            for image in images:
                np_image = image.permute(1, 2, 0).cpu().numpy()
                result.append(np_image)
                dimensions.append(ImageDimensions(height=np_image.shape[0], width=np_image.shape[1]))
            return result, dimensions
        if not isinstance(images, list):
            raise ModelRuntimeError(
                "Pre-processing supports only np.array or torch.Tensor or list of above."
            )
        if not len(images):
            raise ModelRuntimeError("Detected empty input to the model")
        if isinstance(images[0], np.ndarray):
            input_color_format = input_color_format or "bgr"
            if input_color_format != "bgr":
                images = [i[:, :, ::-1] for i in images]
            dimensions = [ImageDimensions(height=i.shape[0], width=i.shape[1]) for i in images]
            return images, dimensions
        if isinstance(images[0], torch.Tensor):
            result = []
            dimensions = []
            input_color_format = input_color_format or "rgb"
            for image in images:
                if input_color_format != "bgr":
                    image = image[[2, 1, 0], :, :]
                np_image = image.permute(1, 2, 0).cpu().numpy()
                result.append(np_image)
                dimensions.append(ImageDimensions(height=np_image.shape[0], width=np_image.shape[1]))
            return images, dimensions
        raise ModelRuntimeError(f"Detected unknown input batch element: {type(images[0])}")

    def forward(
        self,
        pre_processed_images: List[np.ndarray],
        **kwargs,
    ) -> Document:
        return self._model(pre_processed_images)

    def post_process(
        self,
        model_results: Document,
        pre_processing_meta: List[ImageDimensions],
        **kwargs,
    ) -> Tuple[List[str], List[Detections]]:
        rendered_texts, all_detections = [], []
        for result_page, original_dimensions in zip(model_results.pages, pre_processing_meta):
            detections = []
            rendered_texts.append(result_page.render())
            for block in result_page.blocks:
                block_elements_probs = []
                for line in block.lines:
                    line_elements_probs = []
                    for word in line.words:
                        line_elements_probs.append(word.confidence)
                        block_elements_probs.append(word.confidence)
                        detections.append({
                            "xyxy": [word.geometry[0][0], word.geometry[0][1], word.geometry[1][0], word.geometry[1][1]],
                            "class_id": 2,
                            "confidence": word.confidence,
                            "text": word.value,
                        })
                    detections.append({
                        "xyxy": [line.geometry[0][0], line.geometry[0][1], line.geometry[1][0], line.geometry[1][1]],
                        "class_id": 1,
                        "confidence": sum(line_elements_probs) / len(line_elements_probs),
                        "text": line.render(),
                    })
                detections.append({
                    "xyxy": [block.geometry[0][0], block.geometry[0][1], block.geometry[1][0], block.geometry[1][1]],
                    "class_id": 1,
                    "confidence": sum(block_elements_probs) / len(block_elements_probs),
                    "text": block.render(),
                })
            dim_tensor = torch.tensor(
                [original_dimensions.width, original_dimensions.height, original_dimensions.width, original_dimensions.height],
                device=self._device,
            )
            xyxy = (torch.tensor([e["xyxy"] for e in detections], device=self._device) * dim_tensor).round().int()
            class_id = torch.tensor([e["class_id"] for e in detections], device=self._device)
            confidence = torch.tensor([e["confidence"] for e in detections], device=self._device)
            data = [{"text": e["text"]} for e in detections]
            all_detections.append(Detections(
                xyxy=xyxy,
                class_id=class_id,
                confidence=confidence,
                bboxes_metadata=data,
            ))
        return rendered_texts, all_detections

