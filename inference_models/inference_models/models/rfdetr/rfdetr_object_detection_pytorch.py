import os.path
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models import Detections, ObjectDetectionModel
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat
from inference_models.errors import (
    CorruptedModelPackageError,
    InvalidModelInitParameterError,
    MissingModelInitParameterError,
    ModelRuntimeError,
)
from inference_models.logger import LOGGER
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    DivisiblePadding,
    InferenceConfig,
    NetworkInputDefinition,
    PreProcessingMetadata,
    ResizeMode,
    TrainingInputSize,
    parse_class_names_file,
    parse_inference_config,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.models.rfdetr.class_remapping import (
    ClassesReMapping,
    prepare_class_remapping,
)
from inference_models.models.rfdetr.common import parse_model_type
from inference_models.models.rfdetr.default_labels import resolve_labels
from inference_models.models.rfdetr.post_processor import PostProcess
from inference_models.models.rfdetr.rfdetr_base_pytorch import (
    LWDETR,
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    RFDETRXLargeConfig,
    RFDETR2XLargeConfig,
    build_model,
)

try:
    torch.set_float32_matmul_precision("high")
except:
    pass

CONFIG_FOR_MODEL_TYPE = {
    "rfdetr-nano": RFDETRNanoConfig,
    "rfdetr-small": RFDETRSmallConfig,
    "rfdetr-medium": RFDETRMediumConfig,
    "rfdetr-base": RFDETRBaseConfig,
    "rfdetr-large": RFDETRLargeConfig,
    "rfdetr-xlarge": RFDETRXLargeConfig,
    "rfdetr-2xlarge": RFDETR2XLargeConfig,
}

RESIZE_MODES_TO_REVERT_PADDING = {
    ResizeMode.LETTERBOX,
    ResizeMode.LETTERBOX_REFLECT_EDGES,
}


class RFDetrForObjectDetectionTorch(
    (ObjectDetectionModel[torch.Tensor, PreProcessingMetadata, dict])
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        model_type: Optional[str] = None,
        labels: Optional[Union[str, List[str]]] = None,
        resolution: Optional[int] = None,
        **kwargs,
    ) -> "RFDetrForObjectDetectionTorch":
        if os.path.isfile(model_name_or_path):
            return cls.from_checkpoint_file(
                checkpoint_path=model_name_or_path,
                model_type=model_type,
                labels=labels,
                resolution=resolution,
            )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "model_type.json",
                "weights.pth",
            ],
        )
        class_names = parse_class_names_file(
            class_names_path=model_package_content["class_names.txt"]
        )
        inference_config = parse_inference_config(
            config_path=model_package_content["inference_config.json"],
            allowed_resize_modes={
                ResizeMode.STRETCH_TO,
                ResizeMode.LETTERBOX,
                ResizeMode.CENTER_CROP,
                ResizeMode.LETTERBOX_REFLECT_EDGES,
            },
        )
        classes_re_mapping = None
        if inference_config.class_names_operations:
            class_names, classes_re_mapping = prepare_class_remapping(
                class_names=class_names,
                class_names_operations=inference_config.class_names_operations,
                device=device,
            )
        weights_dict = torch.load(
            model_package_content["weights.pth"],
            map_location=device,
            weights_only=False,
        )["model"]
        model_type = parse_model_type(
            config_path=model_package_content["model_type.json"]
        )
        if model_type not in CONFIG_FOR_MODEL_TYPE:
            raise CorruptedModelPackageError(
                message=f"Model package describes model_type as '{model_type}' which is not supported. "
                f"Supported model types: {list(CONFIG_FOR_MODEL_TYPE.keys())}.",
                help_url="https://todo",
            )
        model_config = CONFIG_FOR_MODEL_TYPE[model_type](device=device)
        checkpoint_num_classes = weights_dict["class_embed.bias"].shape[0]
        model_config.num_classes = checkpoint_num_classes - 1
        model_config.resolution = (
            inference_config.network_input.training_input_size.height
        )
        model = build_model(config=model_config)
        model.load_state_dict(weights_dict)
        model = model.eval().to(device)
        post_processor = PostProcess()
        return cls(
            model=model,
            class_names=class_names,
            classes_re_mapping=classes_re_mapping,
            device=device,
            inference_config=inference_config,
            post_processor=post_processor,
            resolution=model_config.resolution,
        )

    @classmethod
    def from_checkpoint_file(
        cls,
        checkpoint_path: str,
        model_type: Optional[str] = None,
        labels: Optional[Union[str, List[str]]] = None,
        resolution: Optional[int] = None,
        device: torch.device = DEFAULT_DEVICE,
    ):
        if model_type is None:
            raise MissingModelInitParameterError(
                message="While loading RFDetr model (using torch backend) could not determine `model_type`. "
                "If you used `RFDetrForObjectDetectionTorch` directly imported in your code, please pass "
                f"one of the value: {CONFIG_FOR_MODEL_TYPE.keys()} as the parameter. If you see this "
                f"error, while using `AutoModel.from_pretrained(...)` or thrown from managed Roboflow service, "
                f"this is a bug - raise the issue: https://github.com/roboflow/inference/issue providing "
                f"full context.",
                help_url="https://todo",
            )
        weights_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )["model"]
        if model_type not in CONFIG_FOR_MODEL_TYPE:
            raise InvalidModelInitParameterError(
                message=f"Model package describes model_type as '{model_type}' which is not supported. "
                f"Supported model types: {list(CONFIG_FOR_MODEL_TYPE.keys())}.",
                help_url="https://todo",
            )
        model_config = CONFIG_FOR_MODEL_TYPE[model_type](device=device)
        divisibility = model_config.num_windows * model_config.patch_size
        if resolution is not None:
            if resolution < 0 or resolution % divisibility != 0:
                raise InvalidModelInitParameterError(
                    message=f"Attempted to load RFDetr model (using torch backend) with `resolution` parameter which "
                    f"is invalid - the model required positive value divisible by 56. Make sure you used "
                    f"proper value, corresponding to the one used to train the model.",
                    help_url="https://todo",
                )
            model_config.resolution = resolution
        inference_config = InferenceConfig(
            network_input=NetworkInputDefinition(
                training_input_size=TrainingInputSize(
                    height=model_config.resolution,
                    width=model_config.resolution,
                ),
                dynamic_spatial_size_supported=True,
                dynamic_spatial_size_mode=DivisiblePadding(
                    type="pad-to-be-divisible",
                    value=divisibility,
                ),
                color_mode=ColorMode.BGR,
                resize_mode=ResizeMode.STRETCH_TO,
                input_channels=3,
                scaling_factor=255,
                normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            )
        )
        checkpoint_num_classes = weights_dict["class_embed.bias"].shape[0]
        model_config.num_classes = checkpoint_num_classes - 1
        model = build_model(config=model_config)
        if labels is None:
            class_names = [f"class_{i}" for i in range(checkpoint_num_classes)]
        elif isinstance(labels, str):
            class_names = resolve_labels(labels=labels)
        else:
            class_names = labels
        if checkpoint_num_classes != len(class_names):
            raise InvalidModelInitParameterError(
                message=f"Checkpoint pointed to load RFDetr defines {checkpoint_num_classes} output classes, but "
                f"loaded labels define {len(class_names)} classes - fix the value of `labels` parameter.",
                help_url="https://todo",
            )
        model.load_state_dict(weights_dict)
        model = model.eval().to(device)
        post_processor = PostProcess()
        return cls(
            model=model,
            class_names=class_names,
            classes_re_mapping=None,
            device=device,
            inference_config=inference_config,
            post_processor=post_processor,
            resolution=model_config.resolution,
        )

    def __init__(
        self,
        model: LWDETR,
        inference_config: InferenceConfig,
        class_names: List[str],
        classes_re_mapping: Optional[ClassesReMapping],
        device: torch.device,
        post_processor: PostProcess,
        resolution: int,
    ):
        self._model = model
        self._inference_config = inference_config
        self._class_names = class_names
        self._classes_re_mapping = classes_re_mapping
        self._post_processor = post_processor
        self._device = device
        self._resolution = resolution
        self._has_warned_about_not_being_optimized_for_inference = False
        self._inference_model: Optional[LWDETR] = None
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_dtype = None

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def optimize_for_inference(
        self,
        compile: bool = True,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.remove_optimized_model()
        self._inference_model = deepcopy(self._model)
        self._inference_model.eval()
        self._inference_model.export()
        self._inference_model = self._inference_model.to(dtype=dtype)
        self._optimized_dtype = dtype
        if compile:
            self._inference_model = torch.jit.trace(
                self._inference_model,
                torch.randn(
                    batch_size,
                    3,
                    self._resolution,
                    self._resolution,
                    device=self._device,
                    dtype=dtype,
                ),
            )
            self._optimized_has_been_compiled = True
            self._optimized_batch_size = batch_size

    def remove_optimized_model(self) -> None:
        self._has_warned_about_not_being_optimized_for_inference = False
        self._inference_model = None
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
            image_size_wh=image_size,
        )

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> dict:
        if (
            self._inference_model is None
            and not self._has_warned_about_not_being_optimized_for_inference
        ):
            LOGGER.warning(
                "Model is not optimized for inference. "
                "Latency may be higher than expected. "
                "You can optimize the model for inference by calling model.optimize_for_inference()."
            )
            self._has_warned_about_not_being_optimized_for_inference = True
        if self._inference_model is not None:
            if (self._resolution, self._resolution) != tuple(
                pre_processed_images.shape[2:]
            ):
                raise ModelRuntimeError(
                    message=f"Resolution mismatch. Model was optimized for resolution {self._resolution}, "
                    f"but got {tuple(pre_processed_images.shape[2:])}. "
                    "You can explicitly remove the optimized model by calling model.remove_optimized_model().",
                    help_url="https://todo",
                )
            if self._optimized_has_been_compiled:
                if self._optimized_batch_size != pre_processed_images.shape[0]:
                    raise ModelRuntimeError(
                        message="Batch size mismatch. Optimized model was compiled for batch size "
                        f"{self._optimized_batch_size}, but got {pre_processed_images.shape[0]}. "
                        "You can explicitly remove the optimized model by calling model.remove_optimized_model(). "
                        "Alternatively, you can recompile the optimized model for a different batch size "
                        "by calling model.optimize_for_inference(batch_size=<new_batch_size>).",
                        help_url="https://todo",
                    )
        with torch.inference_mode():
            if self._inference_model:
                predictions = self._inference_model(
                    pre_processed_images.to(dtype=self._optimized_dtype)
                )
            else:
                predictions = self._model(pre_processed_images)
            if isinstance(predictions, tuple):
                predictions = {
                    "pred_logits": predictions[1],
                    "pred_boxes": predictions[0],
                }
            return predictions

    def post_process(
        self,
        model_results: dict,
        pre_processing_meta: List[PreProcessingMetadata],
        threshold: float = 0.5,
        **kwargs,
    ) -> List[Detections]:
        if (
            self._inference_config.network_input.resize_mode
            in RESIZE_MODES_TO_REVERT_PADDING
        ):
            un_padding_results = []
            for out_box_tensor, image_metadata in zip(
                model_results["pred_boxes"], pre_processing_meta
            ):
                box_center_offsets = torch.as_tensor(  # bboxes in format cxcywh now, so only cx, cy to be pushed
                    [
                        image_metadata.pad_left / image_metadata.inference_size.width,
                        image_metadata.pad_top / image_metadata.inference_size.height,
                        0.0,
                        0.0,
                    ],
                    dtype=out_box_tensor.dtype,
                    device=out_box_tensor.device,
                )
                ox_padding = (
                    image_metadata.pad_left + image_metadata.pad_right
                ) / image_metadata.inference_size.width
                oy_padding = (
                    image_metadata.pad_top + image_metadata.pad_bottom
                ) / image_metadata.inference_size.height
                box_wh_offsets = torch.as_tensor(  # bboxes in format cxcywh now, so only cx, cy to be pushed
                    [
                        1.0 - ox_padding,
                        1.0 - oy_padding,
                        1.0 - ox_padding,
                        1.0 - oy_padding,
                    ],
                    dtype=out_box_tensor.dtype,
                    device=out_box_tensor.device,
                )
                out_box_tensor = (out_box_tensor - box_center_offsets) / box_wh_offsets
                un_padding_results.append(out_box_tensor)
            model_results["pred_boxes"] = torch.stack(un_padding_results, dim=0)
        if self._inference_config.network_input.resize_mode is ResizeMode.CENTER_CROP:
            orig_sizes = [
                (
                    round(e.inference_size.height / e.scale_height),
                    round(e.inference_size.width / e.scale_width),
                )
                for e in pre_processing_meta
            ]
        else:
            orig_sizes = [
                (e.size_after_pre_processing.height, e.size_after_pre_processing.width)
                for e in pre_processing_meta
            ]
        target_sizes = torch.tensor(orig_sizes, device=self._device)
        results = self._post_processor(model_results, target_sizes=target_sizes)
        detections_list = []
        for image_result, image_metadata in zip(results, pre_processing_meta):
            scores = image_result["scores"]
            labels = image_result["labels"]
            boxes = image_result["boxes"]
            if self._classes_re_mapping is not None:
                remapping_mask = torch.isin(
                    labels, self._classes_re_mapping.remaining_class_ids
                )
                scores = scores[remapping_mask]
                labels = self._classes_re_mapping.class_mapping[labels[remapping_mask]]
                boxes = boxes[remapping_mask]
            keep = scores > threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            if (
                self._inference_config.network_input.resize_mode
                is ResizeMode.CENTER_CROP
            ):
                offsets = torch.as_tensor(
                    [
                        image_metadata.pad_left,
                        image_metadata.pad_top,
                        image_metadata.pad_left,
                        image_metadata.pad_top,
                    ],
                    dtype=boxes.dtype,
                    device=boxes.device,
                )
                boxes[:, :4].sub_(offsets)
            if (
                image_metadata.static_crop_offset.offset_x != 0
                or image_metadata.static_crop_offset.offset_y != 0
            ):
                static_crop_offsets = torch.as_tensor(
                    [
                        image_metadata.static_crop_offset.offset_x,
                        image_metadata.static_crop_offset.offset_y,
                        image_metadata.static_crop_offset.offset_x,
                        image_metadata.static_crop_offset.offset_y,
                    ],
                    dtype=boxes.dtype,
                    device=boxes.device,
                )
                boxes[:, :4].add_(static_crop_offsets)
            detections = Detections(
                xyxy=boxes.round().int(),
                confidence=scores,
                class_id=labels.int(),
            )
            detections_list.append(detections)
        return detections_list
