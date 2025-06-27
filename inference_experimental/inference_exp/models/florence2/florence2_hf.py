from typing import List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from inference_exp import Detections, InstanceDetections
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ImageDimensions
from inference_exp.errors import ModelRuntimeError
from inference_exp.models.common.roboflow.pre_processing import (
    extract_input_images_dimensions,
)
from transformers import AutoModelForCausalLM, AutoProcessor

GRANULARITY_2TASK = {
    "normal": "<CAPTION>",
    "detailed": "<DETAILED_CAPTION>",
    "very_detailed": "<MORE_DETAILED_CAPTION>",
}
LABEL_MODE2TASK = {
    "roi": "<REGION_PROPOSAL>",
    "class": "<OD>",
    "caption": "<DENSE_REGION_CAPTION>",
}
LOC_BINS = 1000


class Florence2HF:

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "Florence2HF":
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        return cls(
            model=model, processor=processor, device=device, torch_dtype=torch_dtype
        )

    def __init__(
        self,
        model: AutoModelForCausalLM,
        processor: AutoProcessor,
        device: torch.device,
        torch_dtype: torch.dtype,
    ):
        self._model = model
        self._processor = processor
        self._device = device
        self._torch_dtype = torch_dtype

    def classify_image_region(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        xyxy: Union[
            torch.Tensor,
            List[List[Union[float, int]]],
            List[Union[float, int]],
            np.ndarray,
        ],
        max_new_tokens: int = 4096,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> List[str]:
        loc_phrases = region_to_loc_phrase(images=images, xyxy=xyxy)
        prompt = [f"<REGION_TO_CATEGORY>{phrase}" for phrase in loc_phrases]
        task = "<REGION_TO_CATEGORY>"
        result = self.prompt(
            images=images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            task=task,
        )
        return [deduce_localisation(r[task]) for r in result]

    def caption_image_region(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        xyxy: Union[
            torch.Tensor,
            List[List[Union[float, int]]],
            List[Union[float, int]],
            np.ndarray,
        ],
        max_new_tokens: int = 4096,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> List[str]:
        loc_phrases = region_to_loc_phrase(images=images, xyxy=xyxy)
        prompt = [f"<REGION_TO_DESCRIPTION>{phrase}" for phrase in loc_phrases]
        task = "<REGION_TO_DESCRIPTION>"
        result = self.prompt(
            images=images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            task=task,
        )
        return [deduce_localisation(r[task]) for r in result]

    def ocr_image_region(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        xyxy: Union[
            torch.Tensor,
            List[List[Union[float, int]]],
            List[Union[float, int]],
            np.ndarray,
        ],
        max_new_tokens: int = 4096,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> List[str]:
        loc_phrases = region_to_loc_phrase(images=images, xyxy=xyxy)
        prompt = [f"<REGION_TO_OCR>{phrase}" for phrase in loc_phrases]
        task = "<REGION_TO_OCR>"
        result = self.prompt(
            images=images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            task=task,
        )
        return [deduce_localisation(r[task]) for r in result]

    def segment_region(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        xyxy: Union[
            torch.Tensor,
            List[List[Union[float, int]]],
            List[Union[float, int]],
            np.ndarray,
        ],
        max_new_tokens: int = 4096,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> List[InstanceDetections]:
        loc_phrases = region_to_loc_phrase(images=images, xyxy=xyxy)
        prompt = [f"<REGION_TO_SEGMENTATION>{phrase}" for phrase in loc_phrases]
        task = "<REGION_TO_SEGMENTATION>"
        result = self.prompt(
            images=images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            task=task,
        )
        image_dimensions = extract_input_images_dimensions(images=images)
        return [
            parse_instance_segmentation_prediction(
                prediction=r[task], input_image_dimensions=i, device=self._device
            )
            for r, i in zip(result, image_dimensions)
        ]

    def segment_phrase(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        phrase: str,
        max_new_tokens: int = 4096,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> List[InstanceDetections]:
        prompt = f"<REFERRING_EXPRESSION_SEGMENTATION>{phrase}"
        task = "<REFERRING_EXPRESSION_SEGMENTATION>"
        result = self.prompt(
            images=images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            task=task,
        )
        image_dimensions = extract_input_images_dimensions(images=images)
        return [
            parse_instance_segmentation_prediction(
                prediction=r[task], input_image_dimensions=i, device=self._device
            )
            for r, i in zip(result, image_dimensions)
        ]

    def ground_phrase(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        phrase: str,
        max_new_tokens: int = 4096,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> List[Detections]:
        prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{phrase}"
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        result = self.prompt(
            images=images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            task=task,
        )
        return [
            parse_object_detection_prediction(prediction=r[task], device=self._device)
            for r in result
        ]

    def detect_objects(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        labels_mode: Literal["classes", "captions", "rois"] = "classes",
        classes: Optional[List[str]] = None,
        max_new_tokens: int = 4096,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> List[Detections]:
        if classes:
            classes_str = "<and>".join(classes)
            # not using <OPEN_VOCABULARY_DETECTION> as it associates number of objects with phrases
            prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{classes_str}"
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
        else:
            task = LABEL_MODE2TASK[labels_mode]
            prompt = task
        result = self.prompt(
            images=images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            task=task,
        )
        return [
            parse_object_detection_prediction(
                prediction=r[task], expected_classes=classes, device=self._device
            )
            for r in result
        ]

    def caption_image(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        granularity: Literal["normal", "detailed", "very_detailed"] = "normal",
        max_new_tokens: int = 4096,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> List[str]:
        task = GRANULARITY_2TASK[granularity]
        result = self.prompt(
            images=images,
            prompt=task,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            task=task,
        )
        return [r[task] for r in result]

    def parse_document(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        max_new_tokens: int = 4096,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> List[Detections]:
        task = "<OCR_WITH_REGION>"
        result = self.prompt(
            images=images,
            prompt=task,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            task=task,
        )
        return [
            parse_dense_ocr_prediction(prediction=r[task], device=self._device)
            for r in result
        ]

    def ocr_image(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        max_new_tokens: int = 4096,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> List[str]:
        task = "<OCR>"
        result = self.prompt(
            images=images,
            prompt=task,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            task=task,
        )
        return [r[task] for r in result]

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: Union[str, List[str]],
        max_new_tokens: int = 4096,
        num_beams: int = 3,
        do_sample: bool = False,
        skip_special_tokens: bool = False,
        task: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        inputs, image_dimensions = self.pre_process_generation(
            images=images, prompt=prompt
        )
        generated_ids = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
        )
        return self.post_process_generation(
            generated_ids=generated_ids,
            skip_special_tokens=skip_special_tokens,
            image_dimensions=image_dimensions,
            task=task,
        )

    def pre_process_generation(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: Union[str, List[str]],
        **kwargs,
    ) -> Tuple[dict, List[ImageDimensions]]:
        image_dimensions = extract_input_images_dimensions(images=images)
        if isinstance(prompt, list):
            if len(prompt) != len(image_dimensions):
                raise ModelRuntimeError(
                    message="Provided prompt as list, but the number of prompt elements does not match number of input images.",
                    help_url="https://todo",
                )
        else:
            prompt = [prompt] * len(image_dimensions)
        inputs = self._processor(text=prompt, images=images, return_tensors="pt").to(
            self._device, self._torch_dtype
        )
        return inputs, image_dimensions

    def generate(
        self,
        inputs: dict,
        max_new_tokens: int = 4096,
        num_beams: int = 3,
        do_sample: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        return self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            **kwargs,
        )

    def post_process_generation(
        self,
        generated_ids: torch.Tensor,
        skip_special_tokens: bool = False,
        image_dimensions: Optional[List[ImageDimensions]] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> Union[List[dict], List[str]]:
        generated_texts = self._processor.batch_decode(
            generated_ids, skip_special_tokens=skip_special_tokens
        )
        if image_dimensions is None or task is None:
            return generated_texts
        results = []
        for single_image_text, single_image_dimensions in zip(
            generated_texts, image_dimensions
        ):
            post_processed = self._processor.post_process_generation(
                single_image_text,
                task=task,
                image_size=(
                    single_image_dimensions.width,
                    single_image_dimensions.height,
                ),
            )
            results.append(post_processed)
        return results


def region_to_loc_phrase(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    xyxy: Union[
        torch.Tensor, List[List[Union[float, int]]], List[Union[float, int]], np.ndarray
    ],
) -> List[str]:
    if isinstance(xyxy, torch.Tensor):
        xyxy = xyxy.cpu().numpy()
    if isinstance(xyxy, np.ndarray):
        xyxy = xyxy.tolist()
    image_dimensions = extract_input_images_dimensions(images=images)
    if not xyxy:
        raise ModelRuntimeError(
            message="Provided empty region grounding.", help_url="https://todo"
        )
    nested = isinstance(xyxy[0], list)
    if not nested:
        xyxy = [xyxy] * len(image_dimensions)
    if len(xyxy) != len(image_dimensions):
        raise ModelRuntimeError(
            message="Provided multiple regions - it is expected to provide a single region for each image, but number "
            "of regions does not match number of input images.",
            help_url="https://todo",
        )
    result = []
    for image_xyxy, single_image_dimensions in zip(xyxy, image_dimensions):
        if _coordinates_are_relative(xyxy=image_xyxy):
            left_top_x = _coordinate_to_loc(value=image_xyxy[0])
            left_top_y = _coordinate_to_loc(value=image_xyxy[1])
            right_bottom_x = _coordinate_to_loc(value=image_xyxy[2])
            right_bottom_y = _coordinate_to_loc(value=image_xyxy[3])
            loc_string = f"<loc_{left_top_x}><loc_{left_top_y}><loc_{right_bottom_x}><loc_{right_bottom_y}>"
            result.append(loc_string)
        else:
            left_top_x = _coordinate_to_loc(
                value=image_xyxy[0] / single_image_dimensions.width
            )
            left_top_y = _coordinate_to_loc(
                value=image_xyxy[1] / single_image_dimensions.height
            )
            right_bottom_x = _coordinate_to_loc(
                value=image_xyxy[2] / single_image_dimensions.width
            )
            right_bottom_y = _coordinate_to_loc(
                value=image_xyxy[3] / single_image_dimensions.height
            )
            loc_string = f"<loc_{left_top_x}><loc_{left_top_y}><loc_{right_bottom_x}><loc_{right_bottom_y}>"
            result.append(loc_string)
    return result


def _coordinates_are_relative(xyxy: List[Union[float, int]]) -> bool:
    return all(0 <= c <= 1 for c in xyxy)


def _coordinate_to_loc(value: float) -> int:
    loc_bin = round(_scale_value(value=value, min_value=0.0, max_value=1.0) * LOC_BINS)
    return _scale_value(  # to make sure 0-999 cutting out 1000 on 1.0
        value=loc_bin,
        min_value=0,
        max_value=LOC_BINS - 1,
    )


def _scale_value(
    value: Union[int, float],
    min_value: Union[int, float],
    max_value: Union[int, float],
) -> Union[int, float]:
    return max(min(value, max_value), min_value)


def parse_dense_ocr_prediction(
    prediction: dict,
    device: torch.device,
) -> Detections:
    bboxes = prediction["quad_boxes"]
    labels = prediction.get("labels", [""] * len(bboxes))
    class_ids = [0] * len(bboxes)
    xyxy = []
    for box in bboxes:
        np_box = np.array(box).reshape(-1, 2).round().astype(np.int32)
        min_x, min_y = np_box[:, 0].min(), np_box[:, 1].min()
        max_x, max_y = np_box[:, 0].max(), np_box[:, 1].max()
        xyxy.append([min_x, min_y, max_x, max_y])
    xyxy = torch.tensor(xyxy, device=device).round().int()
    class_ids = torch.tensor(class_ids, device=device).int()
    confidence = torch.tensor([1.0] * len(labels), device=device)
    bboxes_metadata = [{"class_name": label} for label in labels]
    return Detections(
        xyxy=xyxy,
        class_id=class_ids,
        confidence=confidence,
        bboxes_metadata=bboxes_metadata,
    )


def parse_object_detection_prediction(
    prediction: dict,
    device: torch.device,
    expected_classes: Optional[List[int]] = None,
) -> Detections:
    bboxes = prediction["bboxes"]
    labels = prediction.get(
        "labels", prediction.get("bboxes_labels", [""] * len(bboxes))
    )
    if not expected_classes:
        class_ids = [0] * len(bboxes)
    else:
        class_name2idx = {c: i for i, c in enumerate(expected_classes)}
        unknown_class_id = len(expected_classes)
        class_ids = []
        for label in labels:
            class_ids.append(class_name2idx.get(label, unknown_class_id))
    xyxy = torch.tensor(bboxes, device=device).round().int()
    class_ids = torch.tensor(class_ids, device=device).int()
    confidence = torch.tensor([1.0] * len(labels), device=device)
    bboxes_metadata = None
    if not expected_classes:
        bboxes_metadata = [{"class_name": label} for label in labels]
    return Detections(
        xyxy=xyxy,
        class_id=class_ids,
        confidence=confidence,
        bboxes_metadata=bboxes_metadata,
    )


def deduce_localisation(result: str) -> str:
    if "<loc" not in result:
        return result
    return result[: result.index("<loc")]


def parse_instance_segmentation_prediction(
    prediction: dict,
    input_image_dimensions: ImageDimensions,
    device: torch.device,
) -> InstanceDetections:
    xyxy = []
    masks = []
    for polygons in prediction["polygons"]:
        for polygon in polygons:
            mask = np.zeros(
                (input_image_dimensions.height, input_image_dimensions.width),
                dtype=np.uint8,
            )
            np_polygon = np.array(polygon).reshape(-1, 2).round().astype(np.int32)
            if len(np_polygon) < 3:
                continue
            mask = cv2.fillPoly(mask, pts=[np_polygon], color=255)
            mask = mask > 0
            masks.append(mask)
            min_x, min_y = np_polygon[:, 0].min(), np_polygon[:, 1].min()
            max_x, max_y = np_polygon[:, 0].max(), np_polygon[:, 1].max()
            xyxy.append([min_x, min_y, max_x, max_y])
    class_ids = [0] * len(xyxy)
    confidence = [1.0] * len(xyxy)
    return InstanceDetections(
        xyxy=torch.tensor(xyxy, device=device).round().int(),
        class_id=torch.tensor(class_ids, device=device).int(),
        confidence=torch.tensor(confidence),
        mask=torch.from_numpy(np.stack(masks, axis=0)).to(device),
    )
