from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from doctr.io import Document
from doctr.models import detection_predictor, ocr_predictor, recognition_predictor

from inference_models import Detections
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat, ImageDimensions
from inference_models.errors import CorruptedModelPackageError, ModelRuntimeError
from inference_models.models.base.documents_parsing import StructuredOCRModel
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.utils.file_system import read_json

SUPPORTED_DETECTION_MODELS = {
    "fast_base",
    "fast_small",
    "fast_tiny",
    "db_resnet50",
    "db_resnet34",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
}
SUPPORTED_RECOGNITION_MODELS = {
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "master",
    "sar_resnet31",
    "vitstr_small",
    "vitstr_base",
    "parseq",
}


class DocTR(StructuredOCRModel[List[np.ndarray], ImageDimensions, Document]):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        assume_straight_pages: bool = True,
        preserve_aspect_ratio: bool = True,
        detection_max_batch_size: int = 2,
        recognition_max_batch_size: int = 128,
        **kwargs,
    ) -> "StructuredOCRModel":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["detection_weights.pt", "recognition_weights.pt", "config.json"],
        )
        config = parse_model_config(config_path=model_package_content["config.json"])
        if config.det_model not in SUPPORTED_DETECTION_MODELS:
            raise CorruptedModelPackageError(
                message=f"{config.det_model} model denoted in configuration not supported as DocTR detection model.",
                help_url="https://todo",
            )
        if config.rec_model not in SUPPORTED_RECOGNITION_MODELS:
            raise CorruptedModelPackageError(
                message=f"{config.rec_model} model denoted in configuration not supported as DocTR recognition model.",
                help_url="https://todo",
            )
        det_model = detection_predictor(
            arch=config.det_model,
            pretrained=False,
            assume_straight_pages=assume_straight_pages,
            preserve_aspect_ratio=preserve_aspect_ratio,
            batch_size=detection_max_batch_size,
            pretrained_backbone=False,
        )
        det_model.model.to(device)
        detector_weights = torch.load(
            model_package_content["detection_weights.pt"],
            weights_only=True,
            map_location=device,
        )
        det_model.model.load_state_dict(detector_weights)
        rec_model = recognition_predictor(
            arch=config.rec_model,
            pretrained=False,
            batch_size=recognition_max_batch_size,
            pretrained_backbone=False,
        )
        rec_model.model.to(device)
        rec_weights = torch.load(
            model_package_content["recognition_weights.pt"],
            weights_only=True,
            map_location=device,
        )
        rec_model.model.load_state_dict(rec_weights)
        model = ocr_predictor(
            det_arch=det_model.model,
            reco_arch=rec_model.model,
        ).to(device=device)
        return cls(model=model, device=device)

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
        for result_page, original_dimensions in zip(
            model_results.pages, pre_processing_meta
        ):
            detections = []
            rendered_texts.append(result_page.render())
            for block in result_page.blocks:
                block_elements_probs = []
                for line in block.lines:
                    line_elements_probs = []
                    for word in line.words:
                        line_elements_probs.append(word.confidence)
                        block_elements_probs.append(word.confidence)
                        detections.append(
                            {
                                "xyxy": [
                                    word.geometry[0][0],
                                    word.geometry[0][1],
                                    word.geometry[1][0],
                                    word.geometry[1][1],
                                ],
                                "class_id": 2,
                                "confidence": word.confidence,
                                "text": word.value,
                            }
                        )
                    detections.append(
                        {
                            "xyxy": [
                                line.geometry[0][0],
                                line.geometry[0][1],
                                line.geometry[1][0],
                                line.geometry[1][1],
                            ],
                            "class_id": 1,
                            "confidence": sum(line_elements_probs)
                            / len(line_elements_probs),
                            "text": line.render(),
                        }
                    )
                detections.append(
                    {
                        "xyxy": [
                            block.geometry[0][0],
                            block.geometry[0][1],
                            block.geometry[1][0],
                            block.geometry[1][1],
                        ],
                        "class_id": 0,
                        "confidence": sum(block_elements_probs)
                        / len(block_elements_probs),
                        "text": block.render(),
                    }
                )
            dim_tensor = torch.tensor(
                [
                    original_dimensions.width,
                    original_dimensions.height,
                    original_dimensions.width,
                    original_dimensions.height,
                ],
                device=self._device,
            )
            if not detections:
                empty_detections = Detections(
                    xyxy=torch.empty(
                        (0, 4), dtype=torch.int32, device=self._device
                    ),
                    confidence=torch.empty(
                        (0,), dtype=torch.float32, device=self._device
                    ),
                    class_id=torch.empty(
                        (0,), dtype=torch.int32, device=self._device
                    ),
                    bboxes_metadata=[],
                )
                all_detections.append(empty_detections)
                continue
            xyxy = (
                (
                    torch.tensor([e["xyxy"] for e in detections], device=self._device)
                    * dim_tensor
                )
                .round()
                .int()
            )
            class_id = torch.tensor(
                [e["class_id"] for e in detections], device=self._device
            )
            confidence = torch.tensor(
                [e["confidence"] for e in detections], device=self._device
            )
            data = [{"text": e["text"]} for e in detections]
            all_detections.append(
                Detections(
                    xyxy=xyxy,
                    class_id=class_id,
                    confidence=confidence,
                    bboxes_metadata=data,
                )
            )
        return rendered_texts, all_detections


@dataclass
class DocTRConfig:
    det_model: str
    rec_model: str


def parse_model_config(config_path: str) -> DocTRConfig:
    try:
        content = read_json(path=config_path)
        if not content:
            raise ValueError("file is empty.")
        if not isinstance(content, dict):
            raise ValueError("file is malformed (not a JSON dictionary)")
        if "det_model" not in content or "rec_model" not in content:
            raise ValueError(
                "file is malformed (lack of `det_model` or `rec_model` key)"
            )
        return DocTRConfig(
            det_model=content["det_model"],
            rec_model=content["rec_model"],
        )
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            message=f"Config file located under path {config_path} is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs.",
            help_url="https://todo",
        ) from error
