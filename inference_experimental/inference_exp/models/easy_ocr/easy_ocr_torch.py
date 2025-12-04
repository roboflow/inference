from typing import List, Optional, Tuple, Union

import easyocr
import numpy as np
import torch
from inference_exp import Detections, StructuredOCRModel
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat, ImageDimensions
from inference_exp.errors import CorruptedModelPackageError, ModelRuntimeError
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.utils.file_system import read_json
from pydantic import BaseModel

Point = Tuple[int, int]
Coordinates = Tuple[Point, Point, Point, Point]
DetectedText = str
Confidence = float
EasyOCRRawPrediction = Tuple[Coordinates, DetectedText, Confidence]


RECOGNIZED_DETECTORS = {"craft", "dbnet18", "dbnet50"}


class EasyOcrConfig(BaseModel):
    lang_list: List[str]
    detector_model_file_name: str
    recognition_model_file_name: str
    detect_network: str
    recognition_network: str


class EasyOCRTorch(
    StructuredOCRModel[List[np.ndarray], ImageDimensions, EasyOCRRawPrediction]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "StructuredOCRModel":
        package_contents = get_model_package_contents(
            model_package_dir=model_name_or_path, elements=["easy-ocr-config.json"]
        )
        config = parse_easy_ocr_config(
            config_path=package_contents["easy-ocr-config.json"]
        )
        device_string = device.type
        if device.type == "cuda" and device.index:
            device_string = f"{device_string}:{device.index}"
        try:
            model = easyocr.Reader(
                config.lang_list,
                download_enabled=False,
                model_storage_directory=model_name_or_path,
                user_network_directory=model_name_or_path,
                detect_network=config.detect_network,
                recog_network=config.recognition_network,
                detector=True,
                recognizer=True,
                gpu=device_string,
            )
        except Exception as error:
            raise error
            raise CorruptedModelPackageError(
                message=f"EasyOCR model package is broken - could not parse model config file. Error: {error}"
                f"If you attempt to run `inference-exp` locally - inspect the contents of local directory to check "
                f"model package - config file is corrupted. If you run the model on Roboflow platform - "
                f"contact us.",
                help_url="https://todo",
            ) from error
        return cls(model=model, device=device)

    def __init__(
        self,
        model: easyocr.Reader,
        device: torch.device,
    ):
        self._model = model
        self._device = device

    @property
    def class_names(self) -> List[str]:
        return ["text-region"]

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
        self, pre_processed_images: List[np.ndarray], **kwargs
    ) -> List[EasyOCRRawPrediction]:
        all_results = []
        for image in pre_processed_images:
            image_results_raw = self._model.readtext(image)
            image_results_parsed = [
                (
                    [
                        [x.item() if not isinstance(x, (int, float)) else x for x in c]
                        for c in res[0]
                    ],
                    res[1],
                    res[2].item() if not isinstance(res[2], (int, float)) else res[2],
                )
                for res in image_results_raw
            ]
            all_results.append(image_results_parsed)
        return all_results

    def post_process(
        self,
        model_results: List[EasyOCRRawPrediction],
        pre_processing_meta: List[ImageDimensions],
        confidence_threshold: float = 0.3,
        text_regions_separator: str = " ",
        **kwargs,
    ) -> Tuple[List[str], List[Detections]]:
        rendered_texts, all_detections = [], []
        for single_image_result, original_dimensions in zip(
            model_results, pre_processing_meta
        ):
            whole_image_text = []
            xyxy = []
            confidence = []
            class_id = []
            for box, text, text_confidence in single_image_result:
                if text_confidence < confidence_threshold:
                    continue
                whole_image_text.append(text)
                min_x = min(p[0] for p in box)
                min_y = min(p[1] for p in box)
                max_x = max(p[0] for p in box)
                max_y = max(p[1] for p in box)
                box_xyxy = [min_x, min_y, max_x, max_y]
                xyxy.append(box_xyxy)
                confidence.append(float(text_confidence))
                class_id.append(0)
            while_image_text_joined = text_regions_separator.join(whole_image_text)
            rendered_texts.append(while_image_text_joined)
            data = [{"text": text} for text in whole_image_text]
            all_detections.append(
                Detections(
                    xyxy=torch.tensor(xyxy, device=self._device),
                    class_id=torch.tensor(class_id, device=self._device),
                    confidence=torch.tensor(confidence, device=self._device),
                    bboxes_metadata=data,
                )
            )
        return rendered_texts, all_detections


def parse_easy_ocr_config(config_path: str) -> EasyOcrConfig:
    try:
        raw_config = read_json(config_path)
        return EasyOcrConfig.model_validate(raw_config)
    except Exception as error:
        raise CorruptedModelPackageError(
            message=f"EasyOCR model package is broken - could not parse model config file. Error: {error}"
            f"If you attempt to run `inference-exp` locally - inspect the contents of local directory to check "
            f"model package - config file is corrupted. If you run the model on Roboflow platform - "
            f"contact us.",
            help_url="https://todo",
        ) from error
