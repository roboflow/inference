"""RF-DETR object detection running a native Core ML package (macOS)."""

from typing import Any, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from inference_models import Detections, ObjectDetectionModel, PreProcessingOverrides
from inference_models.configuration import INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE
from inference_models.entities import ColorFormat, Confidence
from inference_models.models.common.coreml import CoreMLModel, load_coreml_package
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    parse_class_names_file,
    parse_inference_config,
)
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.models.rfdetr.class_remapping import (
    ClassesReMapping,
    prepare_class_remapping,
)
from inference_models.models.rfdetr.common import post_process_object_detection_results
from inference_models.models.rfdetr.coreml_forward import (
    align_network_input_with_model,
    run_rfdetr_coreml,
)
from inference_models.models.rfdetr.optimization.backend_path import (
    RFDetrBackendPath,
    RFDetrBackendPlanMixin,
)
from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
    _normalize_execution_plan_argument,
)
from inference_models.weights_providers.entities import RecommendedParameters

# Core ML schedules its own compute units; tensors on the Python side stay on the CPU.
COREML_HOST_DEVICE = torch.device("cpu")


class RFDetrForObjectDetectionCoreML(
    RFDetrBackendPlanMixin,
    (
        ObjectDetectionModel[
            torch.Tensor, PreProcessingMetadata, Tuple[torch.Tensor, torch.Tensor]
        ]
    ),
):
    """Load and run RF-DETR Core ML packages with the ONNX path's pre- and post-processing."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        rf_detr_max_input_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        recommended_parameters: Optional[RecommendedParameters] = None,
        execution_plan: Optional[Union[RFDetrExecutionPlan, Mapping[str, Any]]] = None,
        **kwargs,
    ) -> "RFDetrForObjectDetectionCoreML":
        """Load a Core ML package and resolve its object-detection execution plan.

        Args:
            model_name_or_path (str): Local model package directory.
            rf_detr_max_input_resolution (int | tuple, optional): Input size limit.
            recommended_parameters (RecommendedParameters, optional): Model defaults.
            execution_plan (RFDetrExecutionPlan | Mapping, optional): Stage choices
                and fallback policies; None resolves environment/default choices.
            **kwargs: Extra loader options accepted for shared API compatibility.

        Returns:
            RFDetrForObjectDetectionCoreML: Initialized model with resolved stages.
        """
        execution_plan = _normalize_execution_plan_argument(
            execution_plan=execution_plan, kwargs=kwargs
        )
        execution_plan = RFDetrExecutionPlan.resolve(execution_plan=execution_plan)
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
                    "RFDetr Object Detection model running with Core ML backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "RFDetr models. To ensure interoperability, `stretch` "
                    "resize mode will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
            max_allowed_input_size=rf_detr_max_input_resolution,
        )
        coreml_model = load_coreml_package(model_package_dir=model_name_or_path)
        inference_config = align_network_input_with_model(
            inference_config=inference_config, signature=coreml_model.signature
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
            rfdetr_execution_plan=execution_plan,
        )

    def __init__(
        self,
        coreml_model: CoreMLModel,
        class_names: List[str],
        num_logit_classes: int,
        classes_re_mapping: Optional[ClassesReMapping],
        inference_config: InferenceConfig,
        recommended_parameters=None,
        rfdetr_execution_plan: Optional[RFDetrExecutionPlan] = None,
    ):
        self._coreml_model = coreml_model
        self._inference_config = inference_config
        self._class_names = class_names
        self._num_logit_classes = num_logit_classes
        self._classes_re_mapping = classes_re_mapping
        self._device = COREML_HOST_DEVICE
        self.recommended_parameters = recommended_parameters
        self._execution_path = RFDetrBackendPath(
            device=COREML_HOST_DEVICE,
            inference_config=inference_config,
            backend="coreml",
            execution_plan=rfdetr_execution_plan,
        )

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        independent_stage_execution: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        """Prepare images exactly as the ONNX backend does.

        Args:
            images (np.ndarray | torch.Tensor | list): Input image or batch.
            input_color_format (ColorFormat, optional): Source channel order.
            pre_processing_overrides (PreProcessingOverrides, optional): Per-call
                overrides of package preprocessing settings.
            independent_stage_execution (bool): Synchronize before returning when
                preprocessing is called outside composed inference.
            **kwargs: Other shared inference options, ignored by this stage.

        Returns:
            tuple: Normalized NCHW tensor and per-image preprocessing metadata.
        """
        return self._execution_path.preprocess(
            images,
            input_color_format=input_color_format,
            pre_processing_overrides=pre_processing_overrides,
            independent_stage_execution=independent_stage_execution,
        )

    def forward(
        self, pre_processed_images: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the Core ML package and return raw boxes and class logits.

        Args:
            pre_processed_images (torch.Tensor): Normalized NCHW image batch.
            **kwargs: Shared inference options, ignored by this backend.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Raw bounding boxes and class logits.
        """
        return self._execution_path.forward(
            pre_processed_images,
            stream=None,
            operation=lambda: run_rfdetr_coreml(
                model=self._coreml_model,
                pre_processed_images=pre_processed_images,
                network_input=self._inference_config.network_input,
                num_logit_classes=self._num_logit_classes,
                with_masks=False,
            ),
        )

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        **kwargs,
    ) -> List[Detections]:
        """Convert raw predictions with the shared RF-DETR post-processing.

        Args:
            model_results (tuple): Raw boxes and class logits.
            pre_processing_meta (list[PreProcessingMetadata]): Per-image transforms.
            confidence (Confidence): Detection threshold or model-default selector.
            **kwargs: Other shared inference options passed to postprocessing.

        Returns:
            list[Detections]: Filtered detections in original-image coordinates.
        """
        return self._execution_path.postprocess(
            lambda: self._post_process(
                model_results,
                pre_processing_meta=pre_processing_meta,
                confidence=confidence,
            )
        )

    def _post_process(self, model_results, pre_processing_meta, confidence="default"):
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
        )
        bboxes, logits = model_results
        return post_process_object_detection_results(
            bboxes=bboxes,
            logits=logits,
            pre_processing_meta=pre_processing_meta,
            threshold=confidence_filter.get_threshold(self.class_names),
            num_classes=len(self.class_names),
            classes_re_mapping=self._classes_re_mapping,
            device=COREML_HOST_DEVICE,
        )
