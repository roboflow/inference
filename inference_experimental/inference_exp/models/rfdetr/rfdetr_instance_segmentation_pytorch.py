import os.path
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp import InstanceDetections, InstanceSegmentationModel
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.errors import (
    CorruptedModelPackageError,
    ModelLoadingError,
    ModelRuntimeError,
)
from inference_exp.logger import LOGGER
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.common.roboflow.model_packages import (
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
from inference_exp.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_exp.models.rfdetr.class_remapping import (
    ClassesReMapping,
    prepare_class_remapping,
)
from inference_exp.models.rfdetr.common import (
    parse_model_type,
    post_process_instance_segmentation_results,
)
from inference_exp.models.rfdetr.default_labels import resolve_labels
from inference_exp.models.rfdetr.post_processor import PostProcess
from inference_exp.models.rfdetr.rfdetr_base_pytorch import (
    LWDETR,
    RFDETRSegPreviewConfig,
    build_model,
)

try:
    torch.set_float32_matmul_precision("high")
except:
    pass

CONFIG_FOR_MODEL_TYPE = {
    "rfdetr-seg-preview": RFDETRSegPreviewConfig,
}


class RFDetrForInstanceSegmentationTorch(
    InstanceSegmentationModel[
        torch.Tensor,
        PreProcessingMetadata,
        dict,
    ]
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
    ) -> "RFDetrForInstanceSegmentationTorch":
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
        model_type: Optional[str] = "rfdetr-seg-preview",
        labels: Optional[Union[str, List[str]]] = None,
        resolution: Optional[int] = None,
        device: torch.device = DEFAULT_DEVICE,
    ):
        if model_type is None:
            raise ModelLoadingError(
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
            raise ModelLoadingError(
                message=f"Model package describes model_type as '{model_type}' which is not supported. "
                f"Supported model types: {list(CONFIG_FOR_MODEL_TYPE.keys())}.",
                help_url="https://todo",
            )
        model_config = CONFIG_FOR_MODEL_TYPE[model_type](device=device)
        divisibility = model_config.num_windows * model_config.patch_size
        if resolution is not None:
            if resolution < 0 or resolution % divisibility != 0:
                raise ModelLoadingError(
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
            raise ModelLoadingError(
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
                    "pred_masks": predictions[2],
                }
            return predictions

    def post_process(
        self,
        model_results: dict,
        pre_processing_meta: List[PreProcessingMetadata],
        threshold: float = 0.5,
        **kwargs,
    ) -> List[InstanceDetections]:
        bboxes, logits, masks = (
            model_results["pred_boxes"],
            model_results["pred_logits"],
            model_results["pred_masks"],
        )
        return post_process_instance_segmentation_results(
            bboxes=bboxes,
            logits=logits,
            masks=masks,
            pre_processing_meta=pre_processing_meta,
            threshold=threshold,
            classes_re_mapping=self._classes_re_mapping,
        )
