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
)
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelInputError,
    ModelRuntimeError,
)
from inference_models.models.common.cuda import (
    use_cuda_context,
    use_primary_cuda_context,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    StaticCropOffset,
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

try:
    from inference_models.models.rfdetr.triton_preprocess import (
        TRITON_AVAILABLE as _TRITON_AVAILABLE,
        build_resample_tables,
        triton_preprocess_rfdetr_stretch,
    )
except ImportError:
    _TRITON_AVAILABLE = False
    build_resample_tables = None
    triton_preprocess_rfdetr_stretch = None
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


class _FastPathState:
    """Per-(src_shape, target_shape) cache of GPU buffers + resample tables
    that the Triton fast path reuses across frames."""

    __slots__ = (
        "src_h",
        "src_w",
        "target_h",
        "target_w",
        "pinned_host",
        "src_gpu",
        "out_buffer",
        "tables",
    )

    def __init__(
        self,
        src_h: int,
        src_w: int,
        target_h: int,
        target_w: int,
        pinned_host: torch.Tensor,
        src_gpu: torch.Tensor,
        out_buffer: torch.Tensor,
        tables,
    ) -> None:
        self.src_h = src_h
        self.src_w = src_w
        self.target_h = target_h
        self.target_w = target_w
        self.pinned_host = pinned_host
        self.src_gpu = src_gpu
        self.out_buffer = out_buffer
        self.tables = tables

    @classmethod
    def build(
        cls,
        src_h: int,
        src_w: int,
        target_h: int,
        target_w: int,
        device: torch.device,
    ) -> "_FastPathState":
        pinned_host = torch.empty((src_h, src_w, 3), dtype=torch.uint8, pin_memory=True)
        src_gpu = torch.empty((src_h, src_w, 3), dtype=torch.uint8, device=device)
        out_buffer = torch.empty(
            (1, 3, target_h, target_w), dtype=torch.float32, device=device
        )
        tables = build_resample_tables(
            src_h=src_h,
            src_w=src_w,
            target_h=target_h,
            target_w=target_w,
            device=device,
        )
        return cls(
            src_h=src_h,
            src_w=src_w,
            target_h=target_h,
            target_w=target_w,
            pinned_host=pinned_host,
            src_gpu=src_gpu,
            out_buffer=out_buffer,
            tables=tables,
        )

    def is_stale(
        self, src_h: int, src_w: int, target_h: int, target_w: int
    ) -> bool:
        return (
            self.src_h != src_h
            or self.src_w != src_w
            or self.target_h != target_h
            or self.target_w != target_w
        )


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
        self._thread_local_storage = threading.local()
        self.recommended_parameters = recommended_parameters
        self._fast_path_state: Optional[_FastPathState] = None

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
        fast = self._try_fast_preprocess(
            images=images,
            input_color_format=input_color_format,
            image_size=image_size,
            pre_processing_overrides=pre_processing_overrides,
        )
        if fast is not None:
            return fast
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
        return pre_processed_images, pre_processing_meta

    def _try_fast_preprocess(
        self,
        images,
        input_color_format,
        image_size,
        pre_processing_overrides,
    ) -> Optional[Tuple[torch.Tensor, List[PreProcessingMetadata]]]:
        if not _TRITON_AVAILABLE:
            return None
        if image_size is not None:
            return None
        # pre_processing_overrides can only *disable* transforms; it has no
        # "enable" knob. The fast path never applies static_crop / grayscale /
        # contrast regardless, so the override flags are irrelevant — we just
        # gate on whether the image_pre_processing config itself asks for them.
        ipp = self._inference_config.image_pre_processing
        if (
            (ipp.static_crop is not None and ipp.static_crop.enabled)
            or (ipp.contrast is not None and ipp.contrast.enabled)
            or (ipp.grayscale is not None and ipp.grayscale.enabled)
        ):
            return None

        ni = self._inference_config.network_input
        if ni.dataset_version_resize_dimensions is not None:
            return None
        if ni.input_channels != 3:
            return None
        if ni.scaling_factor not in (None, 255):
            return None
        if ni.normalization is None:
            return None
        # When dataset_version_resize_dimensions is None, the prod path collapses
        # non-stretch resize modes to a single PIL stretch as well
        # (pre_processing.py:_needs_two_step_resize), so we accept all modes here.
        if ni.resize_mode not in (
            ResizeMode.STRETCH_TO,
            ResizeMode.LETTERBOX,
            ResizeMode.CENTER_CROP,
            ResizeMode.LETTERBOX_REFLECT_EDGES,
        ):
            return None

        if isinstance(images, list):
            if len(images) != 1:
                return None
            candidate = images[0]
        else:
            candidate = images
        if not isinstance(candidate, np.ndarray):
            return None
        if (
            candidate.dtype != np.uint8
            or candidate.ndim != 3
            or candidate.shape[2] != 3
        ):
            return None

        caller_mode = (
            ColorMode(input_color_format)
            if input_color_format is not None
            else ColorMode.BGR
        )
        swap_rb = caller_mode != ni.color_mode

        means, stds = ni.normalization
        means_t = (float(means[0]), float(means[1]), float(means[2]))
        stds_t = (float(stds[0]), float(stds[1]), float(stds[2]))
        target_h = ni.training_input_size.height
        target_w = ni.training_input_size.width
        orig_h, orig_w = int(candidate.shape[0]), int(candidate.shape[1])

        state = self._fast_path_state
        if state is None or state.is_stale(
            src_h=orig_h,
            src_w=orig_w,
            target_h=target_h,
            target_w=target_w,
        ):
            state = _FastPathState.build(
                src_h=orig_h,
                src_w=orig_w,
                target_h=target_h,
                target_w=target_w,
                device=self._device,
            )
            self._fast_path_state = state

        pinned_np = state.pinned_host.numpy()
        np.copyto(pinned_np, candidate, casting="no")

        with torch.cuda.stream(self._pre_process_stream):
            state.src_gpu.copy_(state.pinned_host, non_blocking=True)
            triton_preprocess_rfdetr_stretch(
                src=state.src_gpu,
                tables=state.tables,
                target_h=target_h,
                target_w=target_w,
                means=means_t,
                stds=stds_t,
                swap_rb=swap_rb,
                out=state.out_buffer,
            )
            state.out_buffer.record_stream(self._pre_process_stream)
        self._pre_process_stream.synchronize()

        meta = PreProcessingMetadata(
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            original_size=ImageDimensions(width=orig_w, height=orig_h),
            size_after_pre_processing=ImageDimensions(width=orig_w, height=orig_h),
            inference_size=ImageDimensions(width=target_w, height=target_h),
            scale_width=target_w / orig_w,
            scale_height=target_h / orig_h,
            static_crop_offset=StaticCropOffset(
                offset_x=0, offset_y=0, crop_width=orig_w, crop_height=orig_h
            ),
        )
        return state.out_buffer, [meta]

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        disable_cuda_graphs: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = self._trt_cuda_graph_cache if not disable_cuda_graphs else None
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
        with torch.cuda.stream(self._post_process_stream):
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
