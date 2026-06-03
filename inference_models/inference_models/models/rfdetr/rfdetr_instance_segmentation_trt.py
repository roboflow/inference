import threading
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
    INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED,
)
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelInputError,
    ModelRuntimeError,
)
from inference_models.models.base.async_handoff import (
    get_deferred_postprocess_done_event,
    get_trt_outputs_consumed_event,
)

# Hoisted to module scope to avoid per-call `from ... import` inside the hot
# forward_async path. Re-import inside the function added ~13µs/frame in the
# instrumented run on Jetson Orin. Import here is a no-op on every call.
from inference_models.models.base.instance_segmentation import _DirectInferenceFuture
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
from inference_models.models.rfdetr.common import (
    post_process_instance_segmentation_results,
    post_process_instance_segmentation_results_to_rle_masks,
)
from inference_models.models.rfdetr.pre_processing import pre_process_network_input
from inference_models.models.rfdetr.triton_preprocess_runtime import (
    FastPreprocessRuntime,
)
from inference_models.weights_providers.entities import RecommendedParameters

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import RFDetr model with TRT backend - this error means that some additional dependencies "
        f"are not installed in the environment.  If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support. "
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error

try:
    import pycuda.driver as cuda
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running model RFDETR with TRT backend on GPU requires pycuda installation, which is brought with "
        "`trt-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support."
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class RFDetrForInstanceSegmentationTRT(
    InstanceSegmentationModel[
        torch.Tensor,
        PreProcessingMetadata,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]
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
        recommended_parameters: Optional[RecommendedParameters] = None,
        **kwargs,
    ) -> "RFDetrForInstanceSegmentationTRT":
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
                    "RFDetr Instance Segmentation model running with TRT backend was trained with "
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
        if len(outputs) != 3:
            raise CorruptedModelPackageError(
                message=f"Implementation assume 3 model outputs, found: {len(outputs)}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        trt_cuda_graph_cache = establish_trt_cuda_graph_cache(
            default_cuda_graph_cache_size=default_trt_cuda_graph_cache_size,
            cuda_graph_cache=trt_cuda_graph_cache,
        )
        return cls(
            engine=engine,
            input_name=inputs[0],
            output_names=outputs,
            class_names=class_names,
            classes_re_mapping=classes_re_mapping,
            inference_config=inference_config,
            trt_config=trt_config,
            device=device,
            cuda_context=cuda_context,
            execution_context=execution_context,
            trt_cuda_graph_cache=trt_cuda_graph_cache,
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
        self._lock = threading.Lock()
        self._inference_stream = torch.cuda.Stream(device=self._device)
        self._pre_process_cuda_stream = torch.cuda.Stream(device=self._device)
        self._post_process_cuda_stream = torch.cuda.Stream(device=self._device)
        self._thread_local_storage = threading.local()
        self.recommended_parameters = recommended_parameters
        self._fast_preprocess_enabled = INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED
        if self._fast_preprocess_enabled:
            self._fast_preprocess_runtime = FastPreprocessRuntime(device=self._device)

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
        fast = None
        if self._fast_preprocess_enabled:
            fast = self._fast_preprocess_runtime.try_preprocess(
                images=images,
                input_color_format=input_color_format,
                image_size=image_size,
                image_pre_processing=self._inference_config.image_pre_processing,
                network_input=self._inference_config.network_input,
                stream=self._pre_process_stream,
            )
        if fast is not None:
            self._fast_preproc_event = fast.ready_event
            return fast.tensor, fast.metadata
        with torch.cuda.stream(self._pre_process_stream):
            pre_processed_images, pre_processing_meta = pre_process_network_input(
                images=images,
                image_pre_processing=self._inference_config.image_pre_processing,
                network_input=self._inference_config.network_input,
                target_device=self._device,
                input_color_format=input_color_format,
                image_size_wh=image_size,
                pre_processing_overrides=pre_processing_overrides,
            )
        self._pre_process_stream.synchronize()
        if self._fast_preprocess_enabled:
            setattr(
                pre_processed_images,
                "_pre_processing_meta",
                pre_processing_meta,
            )
        return pre_processed_images, pre_processing_meta

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        disable_cuda_graphs: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = self._trt_cuda_graph_cache if not disable_cuda_graphs else None
        preproc_event = getattr(self, "_fast_preproc_event", None)
        if preproc_event is not None:
            self._inference_stream.wait_event(preproc_event)
            self._fast_preproc_event = None
        with self._lock:
            with use_cuda_context(context=self._cuda_context):
                detections, labels, masks = infer_from_trt_engine(
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
                return detections, labels, masks

    def forward_async(
        self,
        pre_processed_images: torch.Tensor,
        pre_processing_meta,
        **kwargs,
    ):
        """Submit CUDA-graph inference without waiting for completion."""
        if self._trt_cuda_graph_cache is None:
            return super().forward_async(
                pre_processed_images, pre_processing_meta, **kwargs
            )

        preproc_event = getattr(self, "_fast_preproc_event", None)
        if preproc_event is not None:
            self._inference_stream.wait_event(preproc_event)
            self._fast_preproc_event = None
        with self._lock:
            with use_cuda_context(context=self._cuda_context):
                raw = infer_from_trt_engine(
                    pre_processed_images=pre_processed_images,
                    trt_config=self._trt_config,
                    engine=self._engine,
                    context=self._execution_context,
                    device=self._device,
                    input_name=self._input_name,
                    outputs=self._output_names,
                    stream=self._inference_stream,
                    trt_cuda_graph_cache=self._trt_cuda_graph_cache,
                    synchronize=False,
                )
        graph_state = getattr(raw[0], "_trt_graph_state", None)
        if graph_state is None:
            self._inference_stream.synchronize()
            return _DirectInferenceFuture(self, raw, pre_processing_meta, None, kwargs)
        produce_event = getattr(raw[0], "_trt_produce_event", None)
        if kwargs.get("reuse_trt_graph_outputs", False):
            future_kwargs = dict(kwargs)
            future_kwargs["defer_postprocess_sync"] = True
            return _DirectInferenceFuture(
                self, raw, pre_processing_meta, produce_event, future_kwargs
            )

        stream = graph_state.cuda_stream

        tls = self._thread_local_storage
        clone_sets = getattr(tls, "clone_sets", None)
        if clone_sets is None:
            raw0, raw1, raw2 = raw
            clone_sets = [
                (
                    torch.empty_like(raw0),
                    torch.empty_like(raw1),
                    torch.empty_like(raw2),
                )
                for _ in range(3)
            ]
            tls.clone_sets = clone_sets
            tls.clone_idx = 0
        idx = tls.clone_idx
        clones = clone_sets[idx]
        tls.clone_idx = (idx + 1) % len(clone_sets)

        prev_stream = torch.cuda.current_stream(self._device)
        torch.cuda.set_stream(stream)
        try:
            raw0, raw1, raw2 = raw
            clones[0].copy_(raw0, non_blocking=True)
            clones[1].copy_(raw1, non_blocking=True)
            clones[2].copy_(raw2, non_blocking=True)
            produce_event = torch.cuda.Event()
            produce_event.record(stream)
            consumer_done = graph_state.consumer_done_event
            if consumer_done is None:
                consumer_done = torch.cuda.Event()
                graph_state.consumer_done_event = consumer_done
            consumer_done.record(stream)
        finally:
            torch.cuda.set_stream(prev_stream)

        clones[0]._trt_produce_event = produce_event  # type: ignore[attr-defined]
        future_kwargs = dict(kwargs)
        future_kwargs["defer_postprocess_sync"] = True
        return _DirectInferenceFuture(
            self, clones, pre_processing_meta, produce_event, future_kwargs
        )

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        mask_format: InstanceSegmentationMaskFormat = "dense",
        **kwargs,
    ) -> List[InstanceDetections]:
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
        produce_event = getattr(model_results[0], "_trt_produce_event", None)
        graph_state = getattr(model_results[0], "_trt_graph_state", None)
        with torch.cuda.stream(self._post_process_stream):
            if produce_event is not None:
                self._post_process_stream.wait_event(produce_event)
            for result_element in model_results:
                result_element.record_stream(self._post_process_stream)
            bboxes, logits, masks = model_results
            if mask_format == "dense":
                results = post_process_instance_segmentation_results(
                    bboxes=bboxes,
                    logits=logits,
                    masks=masks,
                    pre_processing_meta=pre_processing_meta,
                    threshold=confidence_filter.get_threshold(self.class_names),
                    num_classes=len(self.class_names),
                    classes_re_mapping=self._classes_re_mapping,
                )
            else:
                results = post_process_instance_segmentation_results_to_rle_masks(
                    bboxes=bboxes,
                    logits=logits,
                    masks=masks,
                    pre_processing_meta=pre_processing_meta,
                    threshold=confidence_filter.get_threshold(self.class_names),
                    num_classes=len(self.class_names),
                    classes_re_mapping=self._classes_re_mapping,
                    defer_postprocess_sync=kwargs.get("defer_postprocess_sync", False),
                )
            if graph_state is not None:
                output_consumed_events = [
                    get_trt_outputs_consumed_event(result) for result in results
                ]
                if output_consumed_events and all(
                    event is not None for event in output_consumed_events
                ):
                    graph_state.consumer_done_event = output_consumed_events[-1]
                else:
                    consumer_done = graph_state.consumer_done_event
                    if consumer_done is None:
                        consumer_done = torch.cuda.Event()
                        graph_state.consumer_done_event = consumer_done
                    consumer_done.record(self._post_process_stream)
        should_sync = True
        if kwargs.get("defer_postprocess_sync", False):
            should_sync = not all(
                get_deferred_postprocess_done_event(result) is not None
                for result in results
            )
        if should_sync:
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
        return self._post_process_cuda_stream
