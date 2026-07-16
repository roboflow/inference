import threading
from typing import Any, Dict, List, Optional, Tuple, Union

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
    MissingDependencyError,
    ModelRuntimeError,
)
from inference_models.logger import LOGGER
from inference_models.models.common.cuda import (
    use_cuda_context,
    use_primary_cuda_context,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    TRTConfig,
    parse_class_names_file,
    parse_inference_config,
    parse_trt_config,
)
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.models.common.trt import (
    TRTCudaGraphCache,
    establish_trt_cuda_graph_cache,
    get_trt_engine_inputs_and_outputs,
    infer_from_trt_engine,
    load_trt_model,
)
from inference_models.models.rfdetr.class_remapping import (
    ClassesReMapping,
    prepare_class_remapping,
)
from inference_models.models.rfdetr.common import post_process_object_detection_results
from inference_models.models.rfdetr.pre_processing import (
    RFDETR_PREPROCESSOR_IMPLEMENTATIONS,
    RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
    pre_process_network_input,
    resolve_rfdetr_preprocessor,
    resolve_rfdetr_preprocessor_max_workers,
)
from inference_models.models.rfdetr.triton_object_detection_postprocess import (
    RFDETR_POSTPROCESSOR_IMPLEMENTATIONS,
    RFDETR_POSTPROCESSOR_TRITON_FUSED_V1,
    FusedObjectDetectionPostprocessor,
    resolve_rfdetr_postprocessor,
)
from inference_models.models.rfdetr.triton_universal_preprocess_runtime import (
    UniversalFastPreprocessRuntime,
)
from inference_models.weights_providers.entities import RecommendedParameters

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running RFDETR model with TRT backend on GPU requires pycuda installation, which is brought with "
        "`trt-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support."
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error

try:
    import pycuda.driver as cuda
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running RFDETR with TRT backend on GPU requires pycuda installation, which is brought with "
        "`trt-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support.",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class RFDetrForObjectDetectionTRT(
    (
        ObjectDetectionModel[
            torch.Tensor, PreProcessingMetadata, Tuple[torch.Tensor, torch.Tensor]
        ]
    )
):
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        engine_host_code_allowed: bool = False,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache] = None,
        default_trt_cuda_graph_cache_size: int = 8,
        rf_detr_max_input_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        rfdetr_preprocessor: Optional[str] = None,
        rfdetr_preprocessor_max_workers: Optional[int] = None,
        rfdetr_postprocessor: Optional[str] = None,
        recommended_parameters: Optional[RecommendedParameters] = None,
        **kwargs,
    ) -> "RFDetrForObjectDetectionTRT":
        """Load an RF-DETR TensorRT model package.

        Args:
            model_name_or_path: Local model package directory.
            device: CUDA device used for inference.
            engine_host_code_allowed: Whether TensorRT may execute engine host code.
            trt_cuda_graph_cache: Optional caller-managed CUDA graph cache.
            default_trt_cuda_graph_cache_size: Default automatic graph-cache capacity.
            rf_detr_max_input_resolution: Optional maximum accepted input resolution.
            rfdetr_preprocessor: Explicit preprocessing implementation ID. When
                omitted, ``INFERENCE_MODELS_RFDETR_PREPROCESSOR`` is used.
            rfdetr_preprocessor_max_workers: Explicit threaded preprocessing worker
                limit. When omitted, the corresponding environment value is used.
            rfdetr_postprocessor: Explicit postprocessing implementation ID. When
                omitted, ``INFERENCE_MODELS_RFDETR_POSTPROCESSOR`` is used.
            recommended_parameters: Optional model-specific recommended parameters.
            **kwargs: Additional loader arguments accepted for API compatibility.

        Returns:
            Loaded RF-DETR TensorRT model.

        Raises:
            ModelRuntimeError: If the target or implementation selection is invalid.
            CorruptedModelPackageError: If required package contents are inconsistent.
        """
        if device.type != "cuda":
            raise ModelRuntimeError(
                message=f"TRT engine only runs on CUDA device - {device} device detected.",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "trt_config.json",
                "engine.plan",
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
                    ResizeMode.STRETCH_TO,
                    None,
                    "RFDetr Object Detection model running with TRT backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "RFDetr models. To ensure interoperability, `stretch` "
                    "resize mode will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
            max_allowed_input_size=rf_detr_max_input_resolution,
        )
        classes_re_mapping = None
        if inference_config.class_names_operations:
            class_names, classes_re_mapping = prepare_class_remapping(
                class_names=class_names,
                class_names_operations=inference_config.class_names_operations,
                device=device,
            )
        trt_config = parse_trt_config(
            config_path=model_package_content["trt_config.json"]
        )
        cuda.init()
        cuda_device = cuda.Device(device.index or 0)
        with use_primary_cuda_context(cuda_device=cuda_device) as cuda_context:
            engine = load_trt_model(
                model_path=model_package_content["engine.plan"],
                engine_host_code_allowed=engine_host_code_allowed,
            )
            execution_context = engine.create_execution_context()
        inputs, outputs = get_trt_engine_inputs_and_outputs(engine=engine)
        if len(inputs) != 1:
            raise CorruptedModelPackageError(
                message=f"Implementation assume single model input, found: {len(inputs)}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if len(outputs) != 2:
            raise CorruptedModelPackageError(
                message=f"Implementation assume 2 model outputs, found: {len(outputs)}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if "dets" not in outputs or "labels" not in outputs:
            raise CorruptedModelPackageError(
                message=f"Expected model outputs to be named `output0` and `output1`, but found: {outputs}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        trt_cuda_graph_cache = establish_trt_cuda_graph_cache(
            default_cuda_graph_cache_size=default_trt_cuda_graph_cache_size,
            cuda_graph_cache=trt_cuda_graph_cache,
        )
        return cls(
            engine=engine,
            input_name=inputs[0],
            output_names=["dets", "labels"],
            class_names=class_names,
            classes_re_mapping=classes_re_mapping,
            inference_config=inference_config,
            trt_config=trt_config,
            device=device,
            cuda_context=cuda_context,
            execution_context=execution_context,
            trt_cuda_graph_cache=trt_cuda_graph_cache,
            rfdetr_preprocessor=rfdetr_preprocessor,
            rfdetr_preprocessor_max_workers=rfdetr_preprocessor_max_workers,
            rfdetr_postprocessor=rfdetr_postprocessor,
            recommended_parameters=recommended_parameters,
        )

    def __init__(
        self,
        engine: trt.ICudaEngine,
        input_name: str,
        output_names: List[str],
        class_names: List[str],
        classes_re_mapping: Optional[ClassesReMapping],
        inference_config: InferenceConfig,
        trt_config: TRTConfig,
        device: torch.device,
        cuda_context: cuda.Context,
        execution_context: trt.IExecutionContext,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache],
        rfdetr_preprocessor: Optional[str] = None,
        rfdetr_preprocessor_max_workers: Optional[int] = None,
        rfdetr_postprocessor: Optional[str] = None,
        recommended_parameters=None,
    ):
        self._engine = engine
        self._input_name = input_name
        self._output_names = output_names
        self._inference_config = inference_config
        self._class_names = class_names
        self._classes_re_mapping = classes_re_mapping
        self._device = device
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._trt_config = trt_config
        self._trt_cuda_graph_cache = trt_cuda_graph_cache
        self._rfdetr_preprocessor = resolve_rfdetr_preprocessor(
            implementation_id=rfdetr_preprocessor
        )
        self._rfdetr_postprocessor = resolve_rfdetr_postprocessor(
            implementation_id=rfdetr_postprocessor
        )
        self._rfdetr_preprocessor_max_workers = resolve_rfdetr_preprocessor_max_workers(
            max_workers=rfdetr_preprocessor_max_workers
        )
        self._lock = threading.Lock()
        self._inference_stream = torch.cuda.Stream(device=self._device)
        self._universal_preprocessor = (
            UniversalFastPreprocessRuntime(device=self._device)
            if self._rfdetr_preprocessor == RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1
            else None
        )
        if self._universal_preprocessor is not None:
            LOGGER.warning(
                "Selected RF-DETR preprocessor implementation=%s",
                self._rfdetr_preprocessor,
            )
        self._fused_postprocessor = (
            FusedObjectDetectionPostprocessor(device=self._device)
            if self._rfdetr_postprocessor == RFDETR_POSTPROCESSOR_TRITON_FUSED_V1
            else None
        )
        if self._fused_postprocessor is not None:
            LOGGER.warning(
                "Selected RF-DETR postprocessor implementation=%s",
                self._rfdetr_postprocessor,
            )
        self._thread_local_storage = threading.local()
        self.recommended_parameters = recommended_parameters

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def preprocessor_implementation_id(self) -> str:
        return self._rfdetr_preprocessor

    @property
    def preprocessor_implementation_metadata(self) -> Dict[str, Any]:
        return RFDETR_PREPROCESSOR_IMPLEMENTATIONS[self._rfdetr_preprocessor]

    @property
    def postprocessor_implementation_id(self) -> str:
        return self._rfdetr_postprocessor

    @property
    def postprocessor_implementation_metadata(self) -> Dict[str, Any]:
        return RFDETR_POSTPROCESSOR_IMPLEMENTATIONS[self._rfdetr_postprocessor]

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        if self._universal_preprocessor is not None:
            result = self._universal_preprocessor.preprocess(
                images=images,
                input_color_format=input_color_format,
                image_pre_processing=self._inference_config.image_pre_processing,
                network_input=self._inference_config.network_input,
                pre_processing_overrides=pre_processing_overrides,
                stream=self._pre_process_stream,
            )
            result.tensor._rfdetr_preprocess_ready_event = (  # type: ignore[attr-defined]
                result.ready_event
            )
            result.tensor._rfdetr_preprocess_input_kind = (  # type: ignore[attr-defined]
                result.input_kind
            )
            return result.tensor, result.metadata

        with torch.cuda.stream(self._pre_process_stream):
            pre_processed_images, pre_processing_meta = pre_process_network_input(
                images=images,
                image_pre_processing=self._inference_config.image_pre_processing,
                network_input=self._inference_config.network_input,
                target_device=self._device,
                input_color_format=input_color_format,
                pre_processing_overrides=pre_processing_overrides,
                preprocessor_implementation_id=self._rfdetr_preprocessor,
                preprocessor_max_workers=self._rfdetr_preprocessor_max_workers,
            )
        self._pre_process_stream.synchronize()
        return pre_processed_images, pre_processing_meta

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        disable_cuda_graphs: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache = self._trt_cuda_graph_cache if not disable_cuda_graphs else None
        with self._lock:
            with use_cuda_context(context=self._cuda_context):
                ready_event = getattr(
                    pre_processed_images,
                    "_rfdetr_preprocess_ready_event",
                    None,
                )
                if ready_event is not None:
                    self._inference_stream.wait_event(ready_event)
                detections, labels = infer_from_trt_engine(
                    pre_processed_images=pre_processed_images,
                    trt_config=self._trt_config,
                    engine=self._engine,
                    context=self._execution_context,
                    device=self._device,
                    input_name=self._input_name,
                    outputs=self._output_names,
                    stream=self._inference_stream,
                    trt_cuda_graph_cache=cache,
                )
                return detections, labels

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        **kwargs,
    ) -> List[Detections]:
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
        )
        threshold = confidence_filter.get_threshold(self.class_names)
        with torch.cuda.stream(self._post_process_stream):
            for result_element in model_results:
                result_element.record_stream(self._post_process_stream)
            bboxes, logits = model_results
            if self._fused_postprocessor is not None:
                results = self._fused_postprocessor.postprocess(
                    bboxes=bboxes,
                    logits=logits,
                    pre_processing_meta=pre_processing_meta,
                    threshold=threshold,
                    num_classes=len(self.class_names),
                    classes_re_mapping=self._classes_re_mapping,
                    stream=self._post_process_stream,
                )
            else:
                results = post_process_object_detection_results(
                    bboxes=bboxes,
                    logits=logits,
                    pre_processing_meta=pre_processing_meta,
                    threshold=threshold,
                    num_classes=len(self.class_names),
                    classes_re_mapping=self._classes_re_mapping,
                    device=self._device,
                )
        self._post_process_stream.synchronize()
        return results

    @property
    def _pre_process_stream(self) -> torch.cuda.Stream:
        if not hasattr(self._thread_local_storage, "pre_process_stream"):
            self._thread_local_storage.pre_process_stream = torch.cuda.Stream(
                device=self._device
            )
        return self._thread_local_storage.pre_process_stream

    @property
    def _post_process_stream(self) -> torch.cuda.Stream:
        if not hasattr(self._thread_local_storage, "post_process_stream"):
            self._thread_local_storage.post_process_stream = torch.cuda.Stream(
                device=self._device
            )
        return self._thread_local_storage.post_process_stream
