import json
import os
import re
from typing import List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from transformers import (
    BitsAndBytesConfig,
    Florence2ForConditionalGeneration,
    Florence2Processor,
)

from inference_models import Detections, InstanceDetections
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_FLORENCE2_DEFAULT_MAX_NEW_TOKENS,
    INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
)
from inference_models.entities import ColorFormat, ImageDimensions
from inference_models.errors import (
    CorruptedModelPackageError,
    ModelInputError,
    ModelRuntimeError,
)
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    parse_inference_config,
)
from inference_models.models.common.roboflow.pre_processing import (
    extract_input_images_dimensions,
    pre_process_network_input,
)

GRANULARITY_2TASK = {
    "normal": "<CAPTION>",
    "detailed": "<DETAILED_CAPTION>",
    "very_detailed": "<MORE_DETAILED_CAPTION>",
}
LABEL_MODE2TASK = {
    "rois": "<REGION_PROPOSAL>",
    "classes": "<OD>",
    "captions": "<DENSE_REGION_CAPTION>",
}
LOC_BINS = 1000


class Florence2HF:

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        disable_quantization: bool = False,
        **kwargs,
    ) -> "Florence2HF":
        torch_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        inference_config_path = os.path.join(
            model_name_or_path, "inference_config.json"
        )
        inference_config = None
        if os.path.exists(inference_config_path):
            inference_config = parse_inference_config(
                config_path=inference_config_path,
                allowed_resize_modes={
                    ResizeMode.STRETCH_TO,
                    ResizeMode.LETTERBOX,
                    ResizeMode.CENTER_CROP,
                    ResizeMode.LETTERBOX_REFLECT_EDGES,
                    ResizeMode.FIT_LONGER_EDGE,
                },
            )

        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        is_adapter_package = os.path.exists(adapter_config_path)

        base_model_path = (
            os.path.join(model_name_or_path, "base")
            if is_adapter_package
            else model_name_or_path
        )
        if not os.path.isdir(base_model_path):
            raise ModelRuntimeError(
                message=f"Provided model path does not exist or is not a directory: {base_model_path}",
                help_url="https://todo",
            )
        if not os.path.isfile(os.path.join(base_model_path, "config.json")):
            raise ModelRuntimeError(
                message=(
                    "Provided model directory does not look like a valid HF Florence-2 checkpoint (missing config.json). "
                    "If you used the official converter, point to its output directory."
                ),
                help_url="https://todo",
            )
        if (
            quantization_config is None
            and device.type == "cuda"
            and not disable_quantization
        ):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        # Native HF Florence2 path only (require transformers >= 4.56)
        model = Florence2ForConditionalGeneration.from_pretrained(  # type: ignore[arg-type]
            pretrained_model_name_or_path=base_model_path,
            dtype=torch_dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config,
        )
        if is_adapter_package:
            # Custom LoRA attach to also cover vision modules
            adapter_cfg_path = os.path.join(model_name_or_path, "adapter_config.json")
            with open(adapter_cfg_path, "r") as f:
                adapter_cfg = json.load(f)

            requested_target_modules = adapter_cfg.get("target_modules") or []
            adapter_task_type = adapter_cfg.get("task_type") or "SEQ_2_SEQ_LM"
            lora_config = LoraConfig(
                r=adapter_cfg.get("r", 8),
                lora_alpha=adapter_cfg.get("lora_alpha", 8),
                lora_dropout=adapter_cfg.get("lora_dropout", 0.0),
                bias="none",
                target_modules=sorted(requested_target_modules),
                use_dora=bool(adapter_cfg.get("use_dora", False)),
                use_rslora=bool(adapter_cfg.get("use_rslora", False)),
                task_type=adapter_task_type,
            )

            model = get_peft_model(model, lora_config)
            # Load adapter weights
            adapter_state = load_peft_weights(model_name_or_path, device=device.type)
            adapter_state = normalize_adapter_state_dict(adapter_state)
            load_result = set_peft_model_state_dict(
                model, adapter_state, adapter_name="default"
            )
            tuner = lora_config.peft_type
            tuner_prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(tuner, "")
            adapter_missing_keys = []
            # Filter missing keys specific to the current adapter and tuner prefix.
            for key in load_result.missing_keys:
                if tuner_prefix in key and "default" in key:
                    adapter_missing_keys.append(key)
            load_result.missing_keys.clear()
            load_result.missing_keys.extend(adapter_missing_keys)
            if len(load_result.missing_keys) > 0:
                raise CorruptedModelPackageError(
                    message="Could not load LoRA weights for the model - found missing checkpoint keys "
                    f"({len(load_result.missing_keys)}): {load_result.missing_keys}",
                    help_url="https://todo",
                )
            if quantization_config is None:
                model.merge_and_unload()
            # Ensure global dtype consistency (handles CPU bfloat16 vs fp32 mismatches)
            model = model.to(dtype=torch_dtype)
            model = model.to(device)

        processor = Florence2Processor.from_pretrained(  # type: ignore[arg-type]
            pretrained_model_name_or_path=base_model_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        return cls(
            model=model,
            processor=processor,
            inference_config=inference_config,
            device=device,
            torch_dtype=torch_dtype,
        )

    def __init__(
        self,
        model: Florence2ForConditionalGeneration,
        processor: Florence2Processor,
        inference_config: Optional[InferenceConfig],
        device: torch.device,
        torch_dtype: torch.dtype,
    ):
        self._model = model
        self._processor = processor
        self._inference_config = inference_config
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
        num_beams: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
        do_sample: bool = INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
        input_color_format: Optional[ColorFormat] = None,
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
            input_color_format=input_color_format,
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
        num_beams: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
        do_sample: bool = INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
        input_color_format: Optional[ColorFormat] = None,
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
            input_color_format=input_color_format,
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
        num_beams: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
        do_sample: bool = INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
        input_color_format: Optional[ColorFormat] = None,
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
            input_color_format=input_color_format,
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
        num_beams: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
        do_sample: bool = INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
        input_color_format: Optional[ColorFormat] = None,
    ) -> List[InstanceDetections]:
        loc_phrases = region_to_loc_phrase(images=images, xyxy=xyxy)
        prompt = [f"<REGION_TO_SEGMENTATION>{phrase}" for phrase in loc_phrases]
        task = "<REGION_TO_SEGMENTATION>"
        inputs, image_dimensions, pre_processing_metadata = self.pre_process_generation(
            images=images, prompt=prompt, input_color_format=input_color_format
        )
        generated_ids = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
        )
        result = self.post_process_generation(
            generated_ids=generated_ids,
            image_dimensions=image_dimensions,
            task=task,
        )
        if pre_processing_metadata is None:
            pre_processing_metadata = [None] * len(image_dimensions)
        return [
            parse_instance_segmentation_prediction(
                prediction=r[task],
                input_image_dimensions=i,
                image_metadata=image_metadata,
                device=self._device,
            )
            for r, i, image_metadata in zip(
                result, image_dimensions, pre_processing_metadata
            )
        ]

    def segment_phrase(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        phrase: str,
        max_new_tokens: int = 4096,
        num_beams: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
        do_sample: bool = INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
        input_color_format: Optional[ColorFormat] = None,
    ) -> List[InstanceDetections]:
        prompt = f"<REFERRING_EXPRESSION_SEGMENTATION>{phrase}"
        task = "<REFERRING_EXPRESSION_SEGMENTATION>"
        inputs, image_dimensions, pre_processing_metadata = self.pre_process_generation(
            images=images, prompt=prompt, input_color_format=input_color_format
        )
        generated_ids = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
        )
        result = self.post_process_generation(
            generated_ids=generated_ids,
            image_dimensions=image_dimensions,
            task=task,
        )
        if pre_processing_metadata is None:
            pre_processing_metadata = [None] * len(image_dimensions)
        image_dimensions = extract_input_images_dimensions(images=images)
        return [
            parse_instance_segmentation_prediction(
                prediction=r[task],
                input_image_dimensions=i,
                image_metadata=image_metadata,
                device=self._device,
            )
            for r, i, image_metadata in zip(
                result, image_dimensions, pre_processing_metadata
            )
        ]

    def ground_phrase(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        phrase: str,
        max_new_tokens: int = 4096,
        num_beams: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
        do_sample: bool = INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
        input_color_format: Optional[ColorFormat] = None,
    ) -> List[Detections]:
        prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{phrase}"
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        inputs, image_dimensions, pre_processing_metadata = self.pre_process_generation(
            images=images, prompt=prompt, input_color_format=input_color_format
        )
        generated_ids = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
        )
        result = self.post_process_generation(
            generated_ids=generated_ids,
            image_dimensions=image_dimensions,
            task=task,
        )
        if pre_processing_metadata is None:
            pre_processing_metadata = [None] * len(image_dimensions)
        return [
            parse_object_detection_prediction(
                prediction=r[task],
                image_metadata=image_metadata,
                device=self._device,
            )
            for r, image_metadata in zip(result, pre_processing_metadata)
        ]

    def detect_objects(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        labels_mode: Literal["classes", "captions", "rois"] = "classes",
        classes: Optional[List[str]] = None,
        max_new_tokens: int = 4096,
        num_beams: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
        do_sample: bool = INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
        input_color_format: Optional[ColorFormat] = None,
    ) -> List[Detections]:
        if classes:
            classes_str = "<and>".join(classes)
            # not using <OPEN_VOCABULARY_DETECTION> as it associates number of objects with phrases
            prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{classes_str}"
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
        else:
            task = LABEL_MODE2TASK[labels_mode]
            prompt = task
        inputs, image_dimensions, pre_processing_metadata = self.pre_process_generation(
            images=images, prompt=prompt, input_color_format=input_color_format
        )
        generated_ids = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
        )
        result = self.post_process_generation(
            generated_ids=generated_ids,
            image_dimensions=image_dimensions,
            task=task,
        )
        if pre_processing_metadata is None:
            pre_processing_metadata = [None] * len(image_dimensions)
        return [
            parse_object_detection_prediction(
                prediction=r[task],
                image_metadata=image_metadata,
                expected_classes=classes,
                device=self._device,
            )
            for r, image_metadata in zip(result, pre_processing_metadata)
        ]

    def caption_image(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        granularity: Literal["normal", "detailed", "very_detailed"] = "normal",
        max_new_tokens: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_MAX_NEW_TOKENS,
        num_beams: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
        do_sample: bool = INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
        input_color_format: Optional[ColorFormat] = None,
    ) -> List[str]:
        task = GRANULARITY_2TASK[granularity]
        result = self.prompt(
            images=images,
            prompt=task,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            task=task,
            input_color_format=input_color_format,
        )
        return [r[task] for r in result]

    def parse_document(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        max_new_tokens: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_MAX_NEW_TOKENS,
        num_beams: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
        do_sample: bool = INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
        input_color_format: Optional[ColorFormat] = None,
    ) -> List[Detections]:
        task = "<OCR_WITH_REGION>"
        inputs, image_dimensions, pre_processing_metadata = self.pre_process_generation(
            images=images, prompt=task, input_color_format=input_color_format
        )
        generated_ids = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
        )
        result = self.post_process_generation(
            generated_ids=generated_ids,
            image_dimensions=image_dimensions,
            task=task,
        )
        if pre_processing_metadata is None:
            pre_processing_metadata = [None] * len(image_dimensions)
        return [
            parse_dense_ocr_prediction(
                prediction=r[task], image_metadata=image_metadata, device=self._device
            )
            for r, image_metadata in zip(result, pre_processing_metadata)
        ]

    def ocr_image(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        max_new_tokens: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_MAX_NEW_TOKENS,
        num_beams: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
        do_sample: bool = INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
        input_color_format: Optional[ColorFormat] = None,
    ) -> List[str]:
        task = "<OCR>"
        result = self.prompt(
            images=images,
            prompt=task,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            task=task,
            input_color_format=input_color_format,
        )
        return [r[task] for r in result]

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: Union[str, List[str]],
        max_new_tokens: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_MAX_NEW_TOKENS,
        num_beams: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
        do_sample: bool = INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = False,
        task: Optional[str] = None,
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> List[str]:
        inputs, image_dimensions, _ = self.pre_process_generation(
            images=images, prompt=prompt, input_color_format=input_color_format
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
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[dict, List[ImageDimensions], Optional[List[PreProcessingMetadata]]]:
        # # maybe don't need to convert to tensor here, since processor also accepts numpy arrays
        # # but need to handle input_color_format here and this is consistent with how we do it in other models
        def _to_tensor(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
            is_numpy = isinstance(image, np.ndarray)
            if is_numpy:
                tensor_image = torch.from_numpy(image.copy()).permute(2, 0, 1)
            else:
                tensor_image = image
            if input_color_format == "bgr" or (is_numpy and input_color_format is None):
                tensor_image = tensor_image[[2, 1, 0], :, :]
            return tensor_image

        if self._inference_config is None:
            if isinstance(images, torch.Tensor) and images.ndim > 3:
                image_list = [_to_tensor(img) for img in images]
            elif not isinstance(images, list):
                image_list = [_to_tensor(images)]
            else:
                image_list = [_to_tensor(img) for img in images]
            image_dimensions = extract_input_images_dimensions(images=image_list)
            pre_processing_metadata = None
        else:
            images, pre_processing_metadata = pre_process_network_input(
                images=images,
                image_pre_processing=self._inference_config.image_pre_processing,
                network_input=self._inference_config.network_input,
                target_device=self._device,
                input_color_format=input_color_format,
            )
            image_list = [e[0] for e in torch.split(images, 1, dim=0)]
            image_dimensions = [
                e.size_after_pre_processing for e in pre_processing_metadata
            ]

        if isinstance(prompt, list):
            if len(prompt) != len(image_dimensions):
                raise ModelInputError(
                    message="Provided prompt as list, but the number of prompt elements does not match number of input images.",
                    help_url="https://todo",
                )
        else:
            prompt = [prompt] * len(image_dimensions)

        inputs = self._processor(
            text=prompt, images=image_list, return_tensors="pt"
        ).to(self._device, self._torch_dtype)
        return inputs, image_dimensions, pre_processing_metadata

    def generate(
        self,
        inputs: dict,
        max_new_tokens: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_MAX_NEW_TOKENS,
        num_beams: int = INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
        do_sample: bool = INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
        **kwargs,
    ) -> torch.Tensor:
        return self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
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
        raise ModelInputError(
            message="Provided empty region grounding.", help_url="https://todo"
        )
    nested = isinstance(xyxy[0], list)
    if not nested:
        xyxy = [xyxy] * len(image_dimensions)
    if len(xyxy) != len(image_dimensions):
        raise ModelInputError(
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
    image_metadata: Optional[PreProcessingMetadata],
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
    if image_metadata is not None and (
        image_metadata.static_crop_offset.offset_x > 0
        or image_metadata.static_crop_offset.offset_y > 0
    ):
        static_crop_offsets = torch.as_tensor(
            [
                image_metadata.static_crop_offset.offset_x,
                image_metadata.static_crop_offset.offset_y,
                image_metadata.static_crop_offset.offset_x,
                image_metadata.static_crop_offset.offset_y,
            ],
            dtype=xyxy.dtype,
            device=xyxy.device,
        )
        xyxy.add_(static_crop_offsets).round_()
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
    image_metadata: Optional[PreProcessingMetadata],
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
    if image_metadata is not None and (
        image_metadata.static_crop_offset.offset_x > 0
        or image_metadata.static_crop_offset.offset_y > 0
    ):
        static_crop_offsets = torch.as_tensor(
            [
                image_metadata.static_crop_offset.offset_x,
                image_metadata.static_crop_offset.offset_y,
                image_metadata.static_crop_offset.offset_x,
                image_metadata.static_crop_offset.offset_y,
            ],
            dtype=xyxy.dtype,
            device=xyxy.device,
        )
        xyxy.add_(static_crop_offsets).round_()
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
    image_metadata: Optional[PreProcessingMetadata],
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
    xyxy = torch.tensor(xyxy, device=device).round().int()
    mask = torch.from_numpy(np.stack(masks, axis=0)).to(device)
    if image_metadata is not None and (
        image_metadata.static_crop_offset.offset_x > 0
        or image_metadata.static_crop_offset.offset_y > 0
    ):
        static_crop_offsets = torch.as_tensor(
            [
                image_metadata.static_crop_offset.offset_x,
                image_metadata.static_crop_offset.offset_y,
                image_metadata.static_crop_offset.offset_x,
                image_metadata.static_crop_offset.offset_y,
            ],
            dtype=xyxy.dtype,
            device=device,
        )
        xyxy.add_(static_crop_offsets).round_()
        mask_canvas = torch.zeros(
            (
                mask.shape[0],
                image_metadata.original_size.height,
                image_metadata.original_size.width,
            ),
            dtype=torch.bool,
            device=device,
        )
        mask_canvas[
            :,
            image_metadata.static_crop_offset.offset_y : image_metadata.static_crop_offset.offset_y
            + mask.shape[1],
            image_metadata.static_crop_offset.offset_x : image_metadata.static_crop_offset.offset_x
            + mask.shape[2],
        ] = mask
    return InstanceDetections(
        xyxy=xyxy,
        class_id=torch.tensor(class_ids, device=device).int(),
        confidence=torch.tensor(confidence, device=device),
        mask=mask,
    )


def normalize_adapter_state_dict(adapter_state: dict) -> dict:
    normalized = {}
    for key, value in adapter_state.items():
        new_key = key
        # Ensure Florence-2 PEFT prefix matches injected structure
        if (
            "base_model.model.vision_tower." in new_key
            and "base_model.model.model.vision_tower." not in new_key
        ):
            new_key = new_key.replace(
                "base_model.model.vision_tower.",
                "base_model.model.model.vision_tower.",
            )
        # Normalize original repo FFN path to HF-native
        if ".ffn.fn.net.fc1" in new_key:
            new_key = new_key.replace(".ffn.fn.net.fc1", ".ffn.fc1")
        if ".ffn.fn.net.fc2" in new_key:
            new_key = new_key.replace(".ffn.fn.net.fc2", ".ffn.fc2")
        # Normalize language path if needed
        if ".language_model.model." in new_key:
            new_key = new_key.replace(
                ".language_model.model.", ".model.language_model."
            )
        normalized[new_key] = value
    return normalized
