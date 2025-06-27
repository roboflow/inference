import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from doctr.io import Document
from doctr.models import ocr_predictor
from inference_exp import Detections
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat, ImageDimensions
from inference_exp.errors import CorruptedModelPackageError, ModelRuntimeError
from inference_exp.models.base.documents_parsing import DocumentParsingModel
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.utils.file_system import read_json

WEIGHTS_NAMES_MAPPING = {
    "db_resnet50": "db_resnet50-79bd7d70.pt",
    "db_resnet34": "db_resnet34-cb6aed9e.pt",
    "db_mobilenet_v3_large": "db_mobilenet_v3_large-21748dd0.pt",
    "crnn_vgg16_bn": "crnn_vgg16_bn-9762b0b0.pt",
    "crnn_mobilenet_v3_small": "crnn_mobilenet_v3_small_pt-3b919a02.pt",
    "crnn_mobilenet_v3_large": "crnn_mobilenet_v3_large_pt-f5259ec2.pt",
}


class DocTR(DocumentParsingModel[List[np.ndarray], ImageDimensions, Document]):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "DocumentParsingModel":
        os.environ["DOCTR_CACHE_DIR"] = model_name_or_path
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["doctr_det", "doctr_rec", "config.json"],
        )
        config = parse_model_config(config_path=model_package_content["config.json"])
        os.makedirs(f"{model_name_or_path}/doctr_det/models/", exist_ok=True)
        os.makedirs(f"{model_name_or_path}/doctr_rec/models/", exist_ok=True)
        det_model_source_path = os.path.join(
            model_name_or_path, "doctr_det", config.det_model, "model.pt"
        )
        rec_model_source_path = os.path.join(
            model_name_or_path, "doctr_rec", config.rec_model, "model.pt"
        )
        if not os.path.exists(det_model_source_path):
            raise CorruptedModelPackageError(
                message="Could not initialize DocTR model - could not find detection model weights.",
                help_url="https://todo",
            )
        if not os.path.exists(rec_model_source_path):
            raise CorruptedModelPackageError(
                message="Could not initialize DocTR model - could not find recognition model weights.",
                help_url="https://todo",
            )
        if config.det_model not in WEIGHTS_NAMES_MAPPING:
            raise CorruptedModelPackageError(
                message=f"{config.det_model} model denoted in configuration not supported as DocTR detection model.",
                help_url="https://todo",
            )
        if config.rec_model not in WEIGHTS_NAMES_MAPPING:
            raise CorruptedModelPackageError(
                message=f"{config.det_model} model denoted in configuration not supported as DocTR recognition model.",
                help_url="https://todo",
            )
        det_model_target_path = os.path.join(
            model_name_or_path, "models", WEIGHTS_NAMES_MAPPING[config.det_model]
        )
        rec_model_target_path = os.path.join(
            model_name_or_path, "models", WEIGHTS_NAMES_MAPPING[config.rec_model]
        )
        if os.path.exists(det_model_target_path):
            os.remove(det_model_target_path)
        os.symlink(det_model_source_path, det_model_target_path)
        if os.path.exists(rec_model_target_path):
            os.remove(rec_model_target_path)
        os.symlink(rec_model_source_path, rec_model_target_path)
        model = ocr_predictor(
            det_arch=config.det_model,
            reco_arch=config.rec_model,
            pretrained=True,
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
