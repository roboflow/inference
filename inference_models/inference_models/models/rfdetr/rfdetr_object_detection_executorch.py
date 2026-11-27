"""RF-DETR object detection running an ExecuTorch program (.pte)."""

from contextlib import nullcontext
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from inference_models import Detections, ObjectDetectionModel, PreProcessingOverrides
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
)
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import (
    CorruptedModelPackageError,
    InvalidModelInitParameterError,
)
from inference_models.models.common.executorch import (
    get_executorch_input_spec,
    load_executorch_program,
    run_executorch_method_with_static_batch_size,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    align_training_input_size_with_model,
    ensure_input_size_within_limit,
    parse_class_names_file,
    parse_inference_config,
)
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.models.common.streams import get_cuda_stream, use_cuda_stream
from inference_models.models.rfdetr.class_remapping import (
    ClassesReMapping,
    prepare_class_remapping,
)
from inference_models.models.rfdetr.common import post_process_object_detection_results
from inference_models.models.rfdetr.optimization.backend_path import (
    RFDetrBackendPath,
    RFDetrBackendPlanMixin,
)
from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
    _normalize_execution_plan_argument,
)
from inference_models.weights_providers.entities import RecommendedParameters

PROGRAM_FILE_NAME = "weights.pte"
PROGRAM_METHOD_NAME = "forward"
# A program can keep data, such as CUDA backend weights, in a separate file next to it.
PROGRAM_DATA_FILE_NAME = "weights.ptd"


class RFDetrForObjectDetectionExecuTorch(
    RFDetrBackendPlanMixin,
    (
        ObjectDetectionModel[
            torch.Tensor, PreProcessingMetadata, Tuple[torch.Tensor, torch.Tensor]
        ]
    ),
):
    """Load and run RF-DETR ExecuTorch programs with the ONNX path's pre- and post-processing.

    The program takes the same normalized NCHW tensor as the ONNX export and returns
    raw boxes and class logits. The delegates it was lowered to (XNNPACK on the CPU,
    TensorRT, alone or with the CUDA backend, on NVIDIA GPUs) are chosen at export time.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        rf_detr_max_input_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        recommended_parameters: Optional[RecommendedParameters] = None,
        execution_plan: Optional[Union[RFDetrExecutionPlan, Mapping[str, Any]]] = None,
        **kwargs,
    ) -> "RFDetrForObjectDetectionExecuTorch":
        """Load an ExecuTorch package and resolve its object-detection execution plan.

        Args:
            model_name_or_path (str): Local model package directory with
                ``weights.pte``, ``class_names.txt`` and ``inference_config.json``,
                and ``weights.ptd`` when the program keeps its weights outside.
            device (torch.device): Device for pre- and post-processing. Programs
                exported with device-resident inputs must be loaded on their GPU.
            rf_detr_max_input_resolution (int | tuple, optional): Input size limit,
                checked against the program's own input size.
            recommended_parameters (RecommendedParameters, optional): Model defaults.
            execution_plan (RFDetrExecutionPlan | Mapping, optional): Stage choices
                and fallback policies; None resolves environment/default choices.
            **kwargs: Extra loader options accepted for shared API compatibility.

        Returns:
            RFDetrForObjectDetectionExecuTorch: Initialized model with resolved stages.

        Raises:
            CorruptedModelPackageError: If the program does not take one float32
                NCHW image batch, or its delegates cannot be initialized.
            InvalidModelInitParameterError: If the program expects its input on a
                GPU other than ``device``.
            MissingDependencyError: If ExecuTorch is not installed.
            ModelPackageRestrictedError: If the program's input size exceeds
                ``rf_detr_max_input_resolution``.
        """
        execution_plan = _normalize_execution_plan_argument(
            execution_plan=execution_plan, kwargs=kwargs
        )
        execution_plan = RFDetrExecutionPlan.resolve(execution_plan=execution_plan)
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["class_names.txt", "inference_config.json", PROGRAM_FILE_NAME],
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
                    "RFDetr Object Detection model running with ExecuTorch backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "RFDetr models. To ensure interoperability, `stretch` "
                    "resize mode will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
        )
        data_path = Path(model_name_or_path) / PROGRAM_DATA_FILE_NAME
        program = load_executorch_program(
            model_package_content[PROGRAM_FILE_NAME],
            data_path=str(data_path) if data_path.is_file() else None,
        )
        input_shape, input_device, input_dtype_name = get_executorch_input_spec(
            program, method_name=PROGRAM_METHOD_NAME
        )
        method_meta = program.metadata(PROGRAM_METHOD_NAME)
        input_channels = inference_config.network_input.input_channels
        if (
            method_meta.num_inputs() != 1
            or method_meta.num_outputs() != 2
            or len(input_shape) != 4
            or input_shape[1] != input_channels
            or input_dtype_name != "float32"
        ):
            raise CorruptedModelPackageError(
                message=f"RF-DETR ExecuTorch program must take one float32 NCHW image batch with "
                f"{input_channels} channels and return boxes and logits, but it takes "
                f"{method_meta.num_inputs()} input(s), the first {input_dtype_name} with shape "
                f"{input_shape}, and returns {method_meta.num_outputs()} output(s).",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        device = _resolve_device(device)
        if input_device.type != "cpu" and device != input_device:
            raise InvalidModelInitParameterError(
                message=f"RF-DETR ExecuTorch program expects its input on `{input_device}`, but the "
                f"model was requested on `{device}`. Load it on `{input_device}`, or export the "
                f"program with host inputs.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#invalidmodelinitparametererror",
            )
        inference_config = align_training_input_size_with_model(
            inference_config,
            model_input_height=input_shape[2],
            model_input_width=input_shape[3],
        )
        ensure_input_size_within_limit(
            inference_config=inference_config,
            max_allowed_input_size=rf_detr_max_input_resolution,
        )
        # Delegates are initialized only now, and create their TensorRT engines and CUDA
        # resources on the current GPU.
        device_scope = (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        )
        try:
            with device_scope:
                method = program.load_method(PROGRAM_METHOD_NAME)
        except RuntimeError as error:
            raise CorruptedModelPackageError(
                message=f"Could not initialize the RF-DETR ExecuTorch program in `{model_name_or_path}`: {error}. "
                f'A program that uses TensorRT needs `pip install "torch_tensorrt[executorch]"`, and its engines '
                f"run only on the GPU and TensorRT version they were built for. A program that keeps data "
                f"outside the `.pte` needs `{PROGRAM_DATA_FILE_NAME}` next to it.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            ) from error
        classes_re_mapping = None
        if inference_config.class_names_operations:
            class_names, classes_re_mapping = prepare_class_remapping(
                class_names=class_names,
                class_names_operations=inference_config.class_names_operations,
                device=device,
            )
        model = cls(
            method=method,
            class_names=class_names,
            classes_re_mapping=classes_re_mapping,
            inference_config=inference_config,
            device=device,
            input_batch_size=input_shape[0],
            input_device=input_device,
            recommended_parameters=recommended_parameters,
            rfdetr_execution_plan=execution_plan,
        )

        return model

    def __init__(
        self,
        method: Any,
        class_names: List[str],
        classes_re_mapping: Optional[ClassesReMapping],
        inference_config: InferenceConfig,
        device: torch.device,
        input_batch_size: int,
        input_device: torch.device,
        recommended_parameters=None,
        rfdetr_execution_plan: Optional[RFDetrExecutionPlan] = None,
    ):
        self._method = method
        self._inference_config = inference_config
        self._class_names = class_names
        self._classes_re_mapping = classes_re_mapping
        self._device = device
        self._input_batch_size = input_batch_size
        self._input_device = input_device
        self.recommended_parameters = recommended_parameters
        self._execution_path = RFDetrBackendPath(
            device=device,
            inference_config=inference_config,
            backend="executorch",
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
        result = self._execution_path.preprocess(
            images,
            input_color_format=input_color_format,
            pre_processing_overrides=pre_processing_overrides,
            independent_stage_execution=independent_stage_execution,
        )

        return result

    def forward(
        self, pre_processed_images: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the ExecuTorch program and return raw boxes and class logits.

        Args:
            pre_processed_images (torch.Tensor): Normalized NCHW image batch.
            **kwargs: Shared inference options, ignored by this backend.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Raw bounding boxes and class logits
            on the model's device.
        """
        result = self._execution_path.forward(
            pre_processed_images,
            stream=self._inference_stream,
            operation=lambda: self._forward(pre_processed_images),
        )

        return result

    def _forward(
        self, pre_processed_images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bboxes, logits = run_executorch_method_with_static_batch_size(
            self._method,
            inputs=pre_processed_images,
            batch_size=self._input_batch_size,
            input_device=self._input_device,
            stream=self._inference_stream,
        )
        bboxes = bboxes.to(device=self._device)
        logits = logits.to(device=self._device)

        return bboxes, logits

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
            **kwargs: Other shared inference options, ignored by this stage.

        Returns:
            list[Detections]: Filtered detections in original-image coordinates.
        """
        detections = self._execution_path.postprocess(
            lambda: self._post_process(
                model_results,
                pre_processing_meta=pre_processing_meta,
                confidence=confidence,
            )
        )

        return detections

    def _post_process(self, model_results, pre_processing_meta, confidence="default"):
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
        )
        post_process_stream = self._post_process_stream
        with use_cuda_stream(post_process_stream):
            if post_process_stream is not None:
                for result_element in model_results:
                    result_element.record_stream(post_process_stream)
            bboxes, logits = model_results
            detections = post_process_object_detection_results(
                bboxes=bboxes,
                logits=logits,
                pre_processing_meta=pre_processing_meta,
                threshold=confidence_filter.get_threshold(self.class_names),
                num_classes=len(self.class_names),
                classes_re_mapping=self._classes_re_mapping,
                device=self._device,
            )
        if post_process_stream is not None:
            post_process_stream.synchronize()

        return detections

    @property
    def _post_process_stream(self) -> Optional[torch.cuda.Stream]:
        return get_cuda_stream(device=self._device, purpose="post-processing")

    @property
    def _inference_stream(self) -> Optional[torch.cuda.Stream]:
        return get_cuda_stream(device=self._device, purpose="inference")


def _resolve_device(device: torch.device) -> torch.device:
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    return device
