from threading import Lock
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import torch

from inference_models import (
    InstanceDetections,
    InstanceSegmentationMaskFormat,
    InstanceSegmentationModel,
    PreProcessingOverrides,
)
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_YOLO26_DEFAULT_CONFIDENCE,
)
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import CorruptedModelPackageError, ModelInputError
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    parse_class_names_file,
    parse_inference_config,
)
from inference_models.models.common.roboflow.post_processing import (
    ConfidenceFilter,
    post_process_nms_fused_model_output,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.models.common.torch import generate_batch_chunks
from inference_models.models.yolo26.common import prepare_dense_masks, prepare_rle_masks
from inference_models.weights_providers.entities import RecommendedParameters


class YOLO26ForInstanceSegmentationTorchScript(
    InstanceSegmentationModel[
        torch.Tensor, PreProcessingMetadata, Tuple[torch.Tensor, torch.Tensor]
    ]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        recommended_parameters: Optional[RecommendedParameters] = None,
        **kwargs,
    ) -> "YOLO26ForInstanceSegmentationTorchScript":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "weights.torchscript",
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
            implicit_resize_mode_substitutions={
                ResizeMode.FIT_LONGER_EDGE: (
                    ResizeMode.LETTERBOX,
                    127,
                    "YOLO26 Instance Segmentation model running with TorchScript backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "models with input dimensions fixed during weights export. To ensure interoperability, `letterbox` "
                    "resize mode with gray edges will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
        )
        if inference_config.forward_pass.static_batch_size is None:
            raise CorruptedModelPackageError(
                message="Expected static batch size to be registered in the inference configuration.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        model = torch.jit.load(
            model_package_content["weights.torchscript"], map_location=device
        ).eval()
        return cls(
            model=model,
            class_names=class_names,
            inference_config=inference_config,
            device=device,
            recommended_parameters=recommended_parameters,
        )

    def __init__(
        self,
        model: torch.nn.Module,
        inference_config: InferenceConfig,
        class_names: List[str],
        device: torch.device,
        recommended_parameters=None,
    ):
        self._model = model
        self._inference_config = inference_config
        self._class_names = class_names
        self._device = device
        self._lock = Lock()
        self.recommended_parameters = recommended_parameters

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def supported_mask_formats(self) -> Set[InstanceSegmentationMaskFormat]:
        return {"dense", "rle"}

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
            pre_processing_overrides=pre_processing_overrides,
        )

    def forward(
        self, pre_processed_images: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._lock, torch.inference_mode():
            if (
                pre_processed_images.shape[0]
                == self._inference_config.forward_pass.static_batch_size
            ):
                instances, protos = self._model(pre_processed_images)
                return instances.to(self._device), protos.to(self._device)
            instances, protos = [], []
            for input_tensor, padding_size in generate_batch_chunks(
                input_batch=pre_processed_images,
                chunk_size=self._inference_config.forward_pass.static_batch_size,
            ):
                instances_for_chunk, protos_for_chunk = self._model(input_tensor)
                if padding_size > 0:
                    instances_for_chunk = instances_for_chunk[:-padding_size]
                    protos_for_chunk = protos_for_chunk[:-padding_size]
                instances.append(instances_for_chunk)
                protos.append(protos_for_chunk)
            return torch.cat(instances, dim=0).to(self._device), torch.cat(
                protos, dim=0
            ).to(self._device)

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        mask_format: InstanceSegmentationMaskFormat = "dense",
        **kwargs,
    ) -> List[InstanceDetections]:
        if mask_format not in self.supported_mask_formats:
            raise ModelInputError(
                message=f"YOLO26 Instance Segmentation models support the following mask "
                f"formats: {self.supported_mask_formats}. Requested format: {mask_format} "
                f"is not supported. If you see this error while running on Roboflow platform, "
                f"contact support or raise an issue at https://github.com/roboflow/inference/issues. "
                f"When running locally - please verify your integration to make sure that appropriate "
                f"value of `mask_format` parameter is set.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_YOLO26_DEFAULT_CONFIDENCE,
        )
        confidence = confidence_filter.get_threshold(self.class_names)
        instances, protos = model_results
        filtered_results = post_process_nms_fused_model_output(
            output=instances, conf_thresh=confidence
        )
        if mask_format == "dense":
            return prepare_dense_masks(
                filtered_results=filtered_results,
                protos=protos,
                pre_processing_meta=pre_processing_meta,
            )
        return prepare_rle_masks(
            filtered_results=filtered_results,
            protos=protos,
            pre_processing_meta=pre_processing_meta,
        )
