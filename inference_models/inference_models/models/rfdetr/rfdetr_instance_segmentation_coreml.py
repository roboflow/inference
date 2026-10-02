"""RF-DETR instance segmentation running a native Core ML package (macOS)."""

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
    INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_RFDETR_DEFAULT_MAX_DETECTIONS,
)
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import ModelInputError
from inference_models.models.common.coreml import (
    CoreMLModel,
    CoreMLModelSignature,
    load_coreml_package,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    ensure_input_size_within_limit,
    parse_class_names_file,
    parse_inference_config,
)
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.models.rfdetr.class_remapping import (
    ClassesReMapping,
    prepare_class_remapping,
)
from inference_models.models.rfdetr.common import (
    post_process_instance_segmentation_results,
    post_process_instance_segmentation_results_to_rle_masks,
)
from inference_models.models.rfdetr.coreml_forward import (
    align_network_input_with_model,
    run_rfdetr_coreml,
)
from inference_models.models.rfdetr.pre_processing import pre_process_network_input
from inference_models.models.rfdetr.rfdetr_object_detection_coreml import (
    COREML_HOST_DEVICE,
)
from inference_models.weights_providers.entities import RecommendedParameters


class RFDetrForInstanceSegmentationCoreML(
    InstanceSegmentationModel[
        torch.Tensor,
        PreProcessingMetadata,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]
):
    """Load and run RF-DETR segmentation Core ML packages with the ONNX path's pre- and post-processing."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        rf_detr_max_input_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        recommended_parameters: Optional[RecommendedParameters] = None,
        **kwargs,
    ) -> "RFDetrForInstanceSegmentationCoreML":
        """Load a Core ML segmentation package.

        Args:
            model_name_or_path (str): Local model package directory.
            rf_detr_max_input_resolution (int | tuple, optional): Input size limit, checked against the
                model's own input size.
            recommended_parameters (RecommendedParameters, optional): Model defaults.
            **kwargs: Extra loader options accepted for shared API compatibility.

        Returns:
            RFDetrForInstanceSegmentationCoreML: Initialized model.

        Raises:
            ModelPackageRestrictedError: If the model's input size exceeds ``rf_detr_max_input_resolution``.
        """
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["class_names.txt", "inference_config.json"],
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
                    ResizeMode.STRETCH_TO,
                    None,
                    "RFDetr Instance Segmentation model running with Core ML backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "RFDetr models. To ensure interoperability, `stretch` "
                    "resize mode will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
        )

        def _validate_signature(signature: CoreMLModelSignature) -> None:
            nonlocal inference_config
            inference_config = align_network_input_with_model(
                inference_config=inference_config, signature=signature
            )
            # Use the model's actual dimensions, before Core ML compiles or loads it.
            ensure_input_size_within_limit(
                inference_config=inference_config,
                max_allowed_input_size=rf_detr_max_input_resolution,
            )

        coreml_model = load_coreml_package(
            model_package_dir=model_name_or_path,
            validate_signature=_validate_signature,
        )
        num_logit_classes = len(class_names) + 1
        classes_re_mapping = None
        if inference_config.class_names_operations:
            class_names, classes_re_mapping = prepare_class_remapping(
                class_names=class_names,
                class_names_operations=inference_config.class_names_operations,
                device=COREML_HOST_DEVICE,
            )
        return cls(
            coreml_model=coreml_model,
            class_names=class_names,
            num_logit_classes=num_logit_classes,
            classes_re_mapping=classes_re_mapping,
            inference_config=inference_config,
            recommended_parameters=recommended_parameters,
        )

    def __init__(
        self,
        coreml_model: CoreMLModel,
        class_names: List[str],
        num_logit_classes: int,
        classes_re_mapping: Optional[ClassesReMapping],
        inference_config: InferenceConfig,
        recommended_parameters=None,
    ):
        self._coreml_model = coreml_model
        self._inference_config = inference_config
        self._class_names = class_names
        self._num_logit_classes = num_logit_classes
        self._classes_re_mapping = classes_re_mapping
        self._device = COREML_HOST_DEVICE
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
        image_size: Optional[Tuple[int, int]] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        """Prepare images exactly as the ONNX backend does.

        Args:
            images (np.ndarray | torch.Tensor | list): Input image or batch.
            input_color_format (ColorFormat, optional): Source channel order.
            image_size (tuple[int, int], optional): Override of the network input size (width, height).
            pre_processing_overrides (PreProcessingOverrides, optional): Per-call overrides of package
                preprocessing settings.
            **kwargs: Other shared inference options, ignored by this stage.

        Returns:
            tuple: Normalized NCHW tensor and per-image preprocessing metadata.
        """
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
            image_size_wh=image_size,
            pre_processing_overrides=pre_processing_overrides,
        )

    def forward(
        self, pre_processed_images: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the Core ML package and return raw boxes, class logits and mask logits.

        Args:
            pre_processed_images (torch.Tensor): Normalized NCHW image batch.
            **kwargs: Shared inference options, ignored by this backend.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Raw boxes, class logits and mask logits.
        """
        return run_rfdetr_coreml(
            model=self._coreml_model,
            pre_processed_images=pre_processed_images,
            network_input=self._inference_config.network_input,
            num_logit_classes=self._num_logit_classes,
            with_masks=True,
        )

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        mask_format: InstanceSegmentationMaskFormat = "dense",
        max_detections: Optional[int] = INFERENCE_MODELS_RFDETR_DEFAULT_MAX_DETECTIONS,
        **kwargs,
    ) -> List[InstanceDetections]:
        """Convert raw predictions with the shared RF-DETR post-processing.

        Args:
            model_results (tuple): Raw boxes, class logits and mask logits.
            pre_processing_meta (list[PreProcessingMetadata]): Per-image transforms.
            confidence (Confidence): Detection threshold or model-default selector.
            mask_format (InstanceSegmentationMaskFormat): ``dense`` masks or ``rle`` encoded masks.
            max_detections (int, optional): Cap on detections per image, applied before masks are aligned.
            **kwargs: Other shared inference options, ignored here.

        Returns:
            list[InstanceDetections]: Filtered detections and masks in original-image coordinates.

        Raises:
            ModelInputError: If ``mask_format`` is not supported.
        """
        if mask_format not in self.supported_mask_formats:
            raise ModelInputError(
                message=f"RFDetr Instance Segmentation models support the following mask "
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
            default_confidence=INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
        )
        post_process = (
            post_process_instance_segmentation_results
            if mask_format == "dense"
            else post_process_instance_segmentation_results_to_rle_masks
        )
        bboxes, logits, masks = model_results
        return post_process(
            bboxes=bboxes,
            logits=logits,
            masks=masks,
            pre_processing_meta=pre_processing_meta,
            threshold=confidence_filter.get_threshold(self.class_names),
            num_classes=len(self.class_names),
            classes_re_mapping=self._classes_re_mapping,
            max_detections=max_detections,
        )
