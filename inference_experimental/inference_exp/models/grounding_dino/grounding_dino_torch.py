import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from inference_exp import Detections
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat, ImageDimensions
from inference_exp.errors import MissingDependencyError, ModelRuntimeError
from inference_exp.models.base.object_detection import (
    OpenVocabularyObjectDetectionModel,
)
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.utils.download import download_files_to_directory
from torch import nn
from torchvision import transforms
from torchvision.ops import box_convert

try:
    from groundingdino.util.inference import load_model, predict
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import GroundingDino model - this error means that some additional dependencies "
        f"are not installed in the environment. If you run the `inference-exp` library directly in your Python "
        f"program, make sure the following extras of the package are installed: `grounding-dino`."
        f"If you see this error using Roboflow infrastructure, make sure the service you use does support the model. "
        f"You can also contact Roboflow to get support.",
        help_url="https://todo",
    ) from import_error


DEFAULT_CONFIG_URL = "https://raw.githubusercontent.com/roboflow/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_CONFIG_MD5 = "bdb07fc17b611d622633d133d2cf873a"


class GroundingDinoForObjectDetectionTorch(
    OpenVocabularyObjectDetectionModel[
        torch.Tensor,
        List[ImageDimensions],
        Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]], List[str]],
    ]
):
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "GroundingDinoForObjectDetectionTorch":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["groundingdino_swint_ogc.pth"],
        )
        config_path = os.path.join(model_name_or_path, "GroundingDINO_SwinT_OGC.py")
        if not os.path.exists(config_path):
            download_files_to_directory(
                target_dir=model_name_or_path,
                files_specs=[
                    (
                        "GroundingDINO_SwinT_OGC.py",
                        DEFAULT_CONFIG_URL,
                        DEFAULT_CONFIG_MD5,
                    )
                ],
            )
        model = load_model(
            model_config_path=config_path,
            model_checkpoint_path=model_package_content["groundingdino_swint_ogc.pth"],
        ).to(device)
        return cls(model=model, device=device)

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ):
        self._model = model
        self._device = device
        self._numpy_transformations = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([800, 800]),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self._tensors_transformations = transforms.Compose(
            [
                lambda x: x / 255.0,
                transforms.Resize([800, 800]),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[ImageDimensions]]:
        if isinstance(images, np.ndarray):
            input_color_format = input_color_format or "bgr"
            if input_color_format != "rgb":
                images = np.ascontiguousarray(images[:, :, ::-1])
            pre_processed = self._numpy_transformations(images)
            return (
                torch.unsqueeze(pre_processed, dim=0).to(self._device),
                [ImageDimensions(height=images.shape[0], width=images.shape[1])],
            )
        if isinstance(images, torch.Tensor):
            input_color_format = input_color_format or "rgb"
            if len(images.shape) == 3:
                images = torch.unsqueeze(images, dim=0)
            image_dimensions = ImageDimensions(
                height=images.shape[2], width=images.shape[3]
            )
            images = images.to(self._device)
            if input_color_format != "rgb":
                images = images[:, [2, 1, 0], :, :]
            return (
                self._tensors_transformations(images.float()),
                [image_dimensions] * images.shape[0],
            )
        if not isinstance(images, list):
            raise ModelRuntimeError(
                message="Pre-processing supports only np.array or torch.Tensor or list of above.",
                help_url="https://todo",
            )
        if not len(images):
            raise ModelRuntimeError(
                message="Detected empty input to the model",
                help_url="https://todo",
            )
        if isinstance(images[0], np.ndarray):
            input_color_format = input_color_format or "bgr"
            pre_processed, image_dimensions = [], []
            for image in images:
                if input_color_format != "rgb":
                    image = np.ascontiguousarray(image[:, :, ::-1])
                image_dimensions.append(
                    ImageDimensions(height=image.shape[0], width=image.shape[1])
                )
                pre_processed.append(self._numpy_transformations(image))
            return torch.stack(pre_processed, dim=0).to(self._device), image_dimensions
        if isinstance(images[0], torch.Tensor):
            input_color_format = input_color_format or "rgb"
            pre_processed, image_dimensions = [], []
            for image in images:
                if len(image.shape) == 3:
                    image = torch.unsqueeze(image, dim=0)
                if input_color_format != "rgb":
                    image = image[:, [2, 1, 0], :, :]
                image_dimensions.append(
                    ImageDimensions(height=image.shape[2], width=image.shape[3])
                )
                pre_processed.append(self._tensors_transformations(image.float()))
            return torch.cat(pre_processed, dim=0).to(self._device), image_dimensions
        raise ModelRuntimeError(
            message=f"Detected unknown input batch element: {type(images[0])}",
            help_url="https://todo",
        )

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        classes: List[str],
        conf_thresh: float = 0.5,
        text_threshold: Optional[float] = None,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]], List[str]]:
        if text_threshold is None:
            text_threshold = conf_thresh
        caption = ". ".join(classes)
        all_boxes, all_logits, all_phrases = [], [], []
        for image in pre_processed_images:
            boxes, logits, phrases = predict(
                model=self._model,
                image=image,
                caption=caption,
                box_threshold=conf_thresh,
                text_threshold=text_threshold,
                device=self._device,
                remove_combined=True,
            )
            all_boxes.append(boxes)
            all_logits.append(logits)
            all_phrases.append(phrases)
        return all_boxes, all_logits, all_phrases, classes

    def post_process(
        self,
        model_results: Tuple[
            List[torch.Tensor], List[torch.Tensor], List[List[str]], List[str]
        ],
        pre_processing_meta: List[ImageDimensions],
        iou_thresh: float = 0.45,
        max_detections: int = 100,
        class_agnostic: bool = False,
        **kwargs,
    ) -> List[Detections]:
        all_boxes, all_logits, all_phrases, classes = model_results
        results = []
        for boxes, logits, phrases, origin_size in zip(
            all_boxes, all_logits, all_phrases, pre_processing_meta
        ):
            boxes = boxes * torch.Tensor(
                [
                    origin_size.width,
                    origin_size.height,
                    origin_size.width,
                    origin_size.height,
                ],
                device=boxes.device,
            )
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
            class_id = map_phrases_to_classes(
                phrases=phrases,
                classes=classes,
            ).to(boxes.device)
            nms_class_ids = torch.zeros_like(class_id) if class_agnostic else class_id
            keep = torchvision.ops.batched_nms(xyxy, logits, nms_class_ids, iou_thresh)
            if keep.numel() > max_detections:
                keep = keep[:max_detections]
            results.append(
                Detections(
                    xyxy=xyxy[keep].round().int(),
                    confidence=logits[keep],
                    class_id=class_id[keep].int(),
                ),
            )
        return results


def map_phrases_to_classes(phrases: List[str], classes: List[str]) -> torch.Tensor:
    class_ids = []
    for phrase in phrases:
        for class_ in classes:
            if class_ in phrase:
                class_ids.append(classes.index(class_))
                break
        else:
            # TODO: figure out how to mark additional classes
            class_ids.append(len(classes))
    return torch.tensor(class_ids)
