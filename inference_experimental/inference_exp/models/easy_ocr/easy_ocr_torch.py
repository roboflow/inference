from typing import List, Tuple, Union, Optional

import numpy as np
import torch

from inference_exp import StructuredOCRModel, Detections
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ImageDimensions, ColorFormat
from inference_exp.errors import ModelRuntimeError


Point = Tuple[int, int]
Coordinates = Tuple[Point, Point, Point, Point]
DetectedText = str
Confidence = float
EasyOCRRawPrediction = Tuple[Coordinates, DetectedText, Confidence]


class EasyOCRTorch(StructuredOCRModel[List[np.ndarray], ImageDimensions, EasyOCRRawPrediction]):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "StructuredOCRModel":
        pass

    def __init__(
        self,
    ):
        pass

    @property
    def class_names(self) -> List[str]:
        return ["text-region"]

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs
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
                dimensions.append(
                    ImageDimensions(height=np_image.shape[0], width=np_image.shape[1])
                )
            return result, dimensions
        if not isinstance(images, list):
            raise ModelRuntimeError(
                message="Pre-processing supports only np.array or torch.Tensor or list of above.",
                help_url="https://todo",
            )
        if not len(images):
            raise ModelRuntimeError(
                message="Detected empty input to the model", help_url="https://todo"
            )
        if isinstance(images[0], np.ndarray):
            input_color_format = input_color_format or "bgr"
            if input_color_format != "bgr":
                images = [i[:, :, ::-1] for i in images]
            dimensions = [
                ImageDimensions(height=i.shape[0], width=i.shape[1]) for i in images
            ]
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
                dimensions.append(
                    ImageDimensions(height=np_image.shape[0], width=np_image.shape[1])
                )
            return result, dimensions
        raise ModelRuntimeError(
            message=f"Detected unknown input batch element: {type(images[0])}",
            help_url="https://todo",
        )

    def forward(self, pre_processed_images: List[np.ndarray], **kwargs) -> EasyOCRRawPrediction:
        pass

    def post_process(
        self,
        model_results: EasyOCRRawPrediction,
        pre_processing_meta: ImageDimensions,
        **kwargs,
    ) -> Tuple[List[str], List[Detections]]:
        pass

