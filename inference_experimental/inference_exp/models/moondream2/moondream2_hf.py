from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp import Detections
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat, ImageDimensions
from inference_exp.errors import ModelRuntimeError
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.common.roboflow.pre_processing import images_to_pillow
from inference_exp.utils.imports import import_class_from_file


@dataclass
class EncodedImage:
    moondream_encoded_image: Any
    image_dimensions: ImageDimensions


@dataclass
class Points:
    xy: torch.Tensor
    confidence: torch.Tensor
    class_id: torch.Tensor


class MoonDream2HF:

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "MoonDream2HF":
        if torch.mps.is_available():
            raise ModelRuntimeError(
                message=f"This model cannot run on Apple device with MPS unit - original implementation contains bug "
                        f"preventing proper allocation of tensors which causes runtime error. Run this model on the "
                        f"machine with Nvidia GPU or x86 CPU.",
                help_url="https://todo",
            )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["hf_moondream.py"],
        )
        model_class = import_class_from_file(
            file_path=model_package_content["hf_moondream.py"],
            class_name="HfMoondream",
        )
        model = model_class.from_pretrained(model_name_or_path).to(device)
        return cls(model=model, device=device)

    def __init__(self, model, device: torch.device):
        self._model = model
        self._device = device

    def detect(
        self,
        images: Union[
            EncodedImage,
            List[EncodedImage],
            torch.Tensor,
            List[torch.Tensor],
            np.ndarray,
            List[np.ndarray],
        ],
        classes: List[str],
        max_tokens: int = 700,
        input_color_format: Optional[ColorFormat] = None,
    ) -> List[Detections]:
        encoded_images = self.encode_images(
            images=images, input_color_format=input_color_format
        )
        results = []
        for encoded_image in encoded_images:
            image_detections = []
            for class_id, class_name in enumerate(classes):
                class_detections = self._model.detect(
                    image=encoded_image.moondream_encoded_image,
                    object=class_name,
                    settings={"max_tokens": max_tokens},
                )["objects"]
                image_detections.append((class_id, class_detections))
            image_results = post_process_detections(
                raw_detections=image_detections,
                image_dimensions=encoded_image.image_dimensions,
                device=self._device,
            )
            results.append(image_results)
        return results

    def caption(
        self,
        images: Union[
            EncodedImage,
            List[EncodedImage],
            torch.Tensor,
            List[torch.Tensor],
            np.ndarray,
            List[np.ndarray],
        ],
        length: Literal["normal", "short", "long"] = "normal",
        max_tokens: int = 700,
        input_color_format: Optional[ColorFormat] = None,
    ) -> List[str]:
        encoded_images = self.encode_images(
            images=images, input_color_format=input_color_format
        )
        results = []
        for encoded_image in encoded_images:
            result = self._model.caption(
                image=encoded_image.moondream_encoded_image,
                length=length,
                settings={"max_tokens": max_tokens},
            )
            results.append(result["caption"].strip())
        return results

    def query(
        self,
        images: Union[
            EncodedImage,
            List[EncodedImage],
            torch.Tensor,
            List[torch.Tensor],
            np.ndarray,
            List[np.ndarray],
        ],
        question: str,
        max_tokens: int = 700,
        input_color_format: Optional[ColorFormat] = None,
    ) -> List[str]:
        encoded_images = self.encode_images(
            images=images, input_color_format=input_color_format
        )
        results = []
        for encoded_image in encoded_images:
            result = self._model.query(
                image=encoded_image.moondream_encoded_image,
                question=question,
                settings={"max_tokens": max_tokens},
            )
            results.append(result["answer"].strip())
        return results

    def point(
        self,
        images: Union[
            EncodedImage,
            List[EncodedImage],
            torch.Tensor,
            List[torch.Tensor],
            np.ndarray,
            List[np.ndarray],
        ],
        classes: List[str],
        max_tokens: int = 700,
        input_color_format: Optional[ColorFormat] = None,
    ) -> List[Points]:
        encoded_images = self.encode_images(
            images=images, input_color_format=input_color_format
        )
        results = []
        for encoded_image in encoded_images:
            image_points = []
            for class_id, class_name in enumerate(classes):
                class_points = self._model.point(
                    image=encoded_image.moondream_encoded_image,
                    object=class_name,
                    settings={"max_tokens": max_tokens},
                )["points"]
                image_points.append((class_id, class_points))
            image_results = post_process_points(
                raw_points=image_points,
                image_dimensions=encoded_image.image_dimensions,
                device=self._device,
            )
            results.append(image_results)
        return results

    def encode_images(
        self,
        images: Union[
            EncodedImage,
            List[EncodedImage],
            torch.Tensor,
            List[torch.Tensor],
            np.ndarray,
            List[np.ndarray],
        ],
        input_color_format: Optional[ColorFormat] = None,
    ) -> List[EncodedImage]:
        if are_images_encoded(images=images):
            if not isinstance(images, list):
                return [images]
            return images
        pillow_images, images_dimensions = images_to_pillow(
            images=images,
            input_color_format=input_color_format,
            model_color_format="rgb",
        )
        result = []
        for image, image_dimensions in zip(pillow_images, images_dimensions):
            moondream_encoded = self._model.encode_image(image)
            result.append(
                EncodedImage(
                    moondream_encoded_image=moondream_encoded,
                    image_dimensions=image_dimensions,
                )
            )
        return result


def are_images_encoded(
    images: Union[
        EncodedImage,
        List[EncodedImage],
        torch.Tensor,
        List[torch.Tensor],
        np.ndarray,
        List[np.ndarray],
    ],
) -> bool:
    if isinstance(images, list):
        if not len(images):
            raise ModelRuntimeError(
                message="Detected empty input to the model", help_url="https://todo"
            )
        return isinstance(images[0], EncodedImage)
    return isinstance(images, EncodedImage)


def post_process_detections(
    raw_detections: List[Tuple[int, List[dict]]],
    image_dimensions: ImageDimensions,
    device: torch.device,
) -> Detections:
    xyxy, confidence, class_id = [], [], []
    for detection_class_id, raw_class_detections in raw_detections:
        for raw_detection in raw_class_detections:
            xyxy.append(
                [
                    raw_detection["x_min"] * image_dimensions.width,
                    raw_detection["y_min"] * image_dimensions.height,
                    raw_detection["x_max"] * image_dimensions.width,
                    raw_detection["y_max"] * image_dimensions.height,
                ]
            )
            class_id.append(detection_class_id)
            confidence.append(1.0)
    return Detections(
        xyxy=torch.tensor(xyxy, device=device).round().int(),
        class_id=torch.tensor(class_id, device=device).int(),
        confidence=torch.tensor(confidence, device=device),
    )


def post_process_points(
    raw_points: List[Tuple[int, List[dict]]],
    image_dimensions: ImageDimensions,
    device: torch.device,
) -> Points:
    xy, confidence, class_id = [], [], []
    for point_class_id, raw_class_points in raw_points:
        for raw_point in raw_class_points:
            xy.append(
                [
                    raw_point["x"] * image_dimensions.width,
                    raw_point["y"] * image_dimensions.height,
                ]
            )
            class_id.append(point_class_id)
            confidence.append(1.0)
    return Points(
        xy=torch.tensor(xy, device=device).round().int(),
        class_id=torch.tensor(class_id, device=device).int(),
        confidence=torch.tensor(confidence, device=device),
    )
