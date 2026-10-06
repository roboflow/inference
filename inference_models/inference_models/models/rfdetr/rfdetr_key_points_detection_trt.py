import threading
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from inference_models import (
    Detections,
    KeyPoints,
    KeyPointsDetectionModel,
    PreProcessingOverrides,
)
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_RFDETR_DEFAULT_KEY_POINTS_THRESHOLD,
)
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelRuntimeError,
)
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
    parse_key_points_metadata,
    parse_trt_config,
)
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.models.common.streams import get_cuda_stream, use_cuda_stream
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
    post_process_keypoint_detection_results,
)
from inference_models.models.rfdetr.pre_processing import pre_process_network_input
from inference_models.weights_providers.entities import RecommendedParameters

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message="Could not import RFDetr model with TRT backend - this error means that some additional dependencies "
        "are not installed in the environment.  If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        "model, You can also contact Roboflow to get support. "
        "Additionally - if AutoModel.from_pretrained(...) "
        "automatically selects model package which does not match your environment - that's a serious problem and "
        "we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
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
        "model, You can also contact Roboflow to get support."
        "Additionally - if AutoModel.from_pretrained(...) "
        "automatically selects model package which does not match your environment - that's a serious problem and "
        "we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error

_NAMED_KEYPOINT_TRT_OUTPUTS = ("dets", "labels", "keypoints")


def _tensor_shape_dims(shape) -> Tuple[int, ...]:
    if hasattr(shape, "nbDims"):
        return tuple(int(shape[i]) for i in range(shape.nbDims))

    return tuple(int(dim) for dim in shape)


def _resolve_rfdetr_keypoint_trt_output_names(
    engine: trt.ICudaEngine,
    output_names: Sequence[str],
) -> List[str]:
    """Return TRT output names in (bboxes, logits, keypoints) order.

    RF-DETR keypoint export names outputs ``dets``, ``labels``, and ``keypoints``.
    TensorRT I/O iteration order is not guaranteed to match that contract, so this
    helper remaps by name when possible and otherwise by rank/shape.
    """
    if all(name in output_names for name in _NAMED_KEYPOINT_TRT_OUTPUTS):
        return list(_NAMED_KEYPOINT_TRT_OUTPUTS)

    ranked = {
        name: _tensor_shape_dims(engine.get_tensor_shape(name)) for name in output_names
    }
    keypoints_name = next(
        (name for name, dims in ranked.items() if len(dims) == 4),
        None,
    )
    boxes_name = next(
        (name for name, dims in ranked.items() if len(dims) == 3 and dims[-1] == 4),
        None,
    )
    used = {name for name in (keypoints_name, boxes_name) if name is not None}
    remaining = [name for name in output_names if name not in used]
    logits_name = remaining[0] if len(remaining) == 1 else None

    if boxes_name and logits_name and keypoints_name:
        return [boxes_name, logits_name, keypoints_name]

    return list(output_names)


class RFDetrForKeyPointsTRT(
    KeyPointsDetectionModel[
        torch.Tensor,
        PreProcessingMetadata,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]
):
    """RF-DETR keypoint detection model running on a TensorRT engine."""

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
    ) -> "RFDetrForKeyPointsTRT":
        """Load an RF-DETR keypoint TensorRT package from a local directory.

        Args:
            model_name_or_path: Path to a flat model package directory containing
                ``class_names.txt``, ``inference_config.json``, ``trt_config.json``,
                ``engine.plan``, and ``keypoints_metadata.json``.
            device: CUDA device used for inference. CPU is not supported.
            engine_host_code_allowed: Whether TensorRT may execute engine host code.
            trt_cuda_graph_cache: Optional CUDA graph cache to reuse across models.
            default_trt_cuda_graph_cache_size: Cache size used when no cache is
                provided.
            rf_detr_max_input_resolution: Optional cap on the parsed inference
                resolution.
            recommended_parameters: Optional confidence defaults from package
                metadata.

        Returns:
            Initialized ``RFDetrForKeyPointsTRT`` instance.

        Raises:
            ModelRuntimeError: If ``device`` is not a CUDA device.
            CorruptedModelPackageError: If required package files are missing or
                the engine does not expose exactly one input and three outputs.
            MissingDependencyError: If TensorRT or PyCUDA is not installed.
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
                "keypoints_metadata.json",
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
                    "RFDetr Keypoint Detection model running with TRT backend was trained with "
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
        parsed_key_points_metadata, skeletons = parse_key_points_metadata(
            key_points_metadata_path=model_package_content["keypoints_metadata.json"],
            classes_re_mapping=classes_re_mapping,
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
        output_names = _resolve_rfdetr_keypoint_trt_output_names(
            engine=engine,
            output_names=outputs,
        )
        trt_cuda_graph_cache = establish_trt_cuda_graph_cache(
            default_cuda_graph_cache_size=default_trt_cuda_graph_cache_size,
            cuda_graph_cache=trt_cuda_graph_cache,
        )

        return cls(
            engine=engine,
            input_name=inputs[0],
            output_names=output_names,
            class_names=class_names,
            classes_re_mapping=classes_re_mapping,
            inference_config=inference_config,
            parsed_key_points_metadata=parsed_key_points_metadata,
            skeletons=skeletons,
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
        parsed_key_points_metadata: List[List[str]],
        skeletons: List[List[Tuple[int, int]]],
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
        self._skeletons = skeletons
        self._parsed_key_points_metadata = parsed_key_points_metadata
        self._device = device
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._trt_config = trt_config
        self._trt_cuda_graph_cache = trt_cuda_graph_cache
        self._lock = threading.Lock()
        self.recommended_parameters = recommended_parameters
        self._key_points_classes_for_instances = torch.tensor(
            [len(e) for e in self._parsed_key_points_metadata], device=device
        )
        self._key_points_slots_in_prediction = max(
            len(e) for e in parsed_key_points_metadata
        )

    @property
    def class_names(self) -> List[str]:
        """Return detection class names in class-id order."""
        return self._class_names

    @property
    def key_points_classes(self) -> List[List[str]]:
        """Return per-class keypoint names."""
        return self._parsed_key_points_metadata

    @property
    def skeletons(self) -> List[List[Tuple[int, int]]]:
        """Return per-class skeleton edges as ``(from, to)`` index pairs."""
        return self._skeletons

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        """Preprocess images for the RF-DETR keypoint TensorRT engine.

        Args:
            images: Single image or batch as numpy arrays or torch tensors.
            input_color_format: Color format of ``images`` when it cannot be
                inferred (``rgb`` or ``bgr``).
            pre_processing_overrides: Optional preprocessing overrides.

        Returns:
            Preprocessed NCHW tensor on the model device and per-image metadata
            used during post-processing.
        """
        pre_process_stream = self._pre_process_stream
        with use_cuda_stream(pre_process_stream):
            pre_processed_images, pre_processing_meta = pre_process_network_input(
                images=images,
                image_pre_processing=self._inference_config.image_pre_processing,
                network_input=self._inference_config.network_input,
                target_device=self._device,
                input_color_format=input_color_format,
                pre_processing_overrides=pre_processing_overrides,
            )
        if pre_process_stream is not None:
            pre_process_stream.synchronize()

        return pre_processed_images, pre_processing_meta

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        disable_cuda_graphs: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the TensorRT engine and return boxes, logits, and keypoints.

        Args:
            pre_processed_images: NCHW tensor produced by ``pre_process``.
            disable_cuda_graphs: If True, skip CUDA graph replay for this call.

        Returns:
            Tuple of ``(bboxes, logits, keypoints)`` in the same order and
            layout as the ONNX keypoint backend.
        """
        cache = self._trt_cuda_graph_cache if not disable_cuda_graphs else None
        with self._lock:
            with use_cuda_context(context=self._cuda_context):
                bboxes, logits, keypoints = infer_from_trt_engine(
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

                return bboxes, logits, keypoints

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        key_points_threshold: float = INFERENCE_MODELS_RFDETR_DEFAULT_KEY_POINTS_THRESHOLD,
        **kwargs,
    ) -> Tuple[List[KeyPoints], Optional[List[Detections]]]:
        """Convert raw engine outputs into keypoints and optional detections.

        Args:
            model_results: ``(bboxes, logits, keypoints)`` tensors from
                ``forward``.
            pre_processing_meta: Per-image metadata from ``pre_process``.
            confidence: Instance confidence threshold, or ``default`` / ``best``.
            key_points_threshold: Minimum per-keypoint confidence to keep.

        Returns:
            Tuple of per-image ``KeyPoints`` and matching ``Detections``.
        """
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
            bboxes, logits, keypoints = model_results
            results = post_process_keypoint_detection_results(
                bboxes=bboxes,
                out_logits=logits,
                out_keypoints=keypoints,
                pre_processing_meta=pre_processing_meta,
                threshold=confidence_filter.get_threshold(self.class_names),
                key_points_threshold=key_points_threshold,
                num_classes=len(self.class_names),
                classes_re_mapping=self._classes_re_mapping,
                key_points_classes_for_instances=self._key_points_classes_for_instances,
                key_points_slots_in_prediction=self._key_points_slots_in_prediction,
                device=self._device,
            )
        if post_process_stream is not None:
            post_process_stream.synchronize()

        return results

    @property
    def _pre_process_stream(self) -> Optional[torch.cuda.Stream]:
        return get_cuda_stream(device=self._device, purpose="pre-processing")

    @property
    def _post_process_stream(self) -> Optional[torch.cuda.Stream]:
        return get_cuda_stream(device=self._device, purpose="post-processing")

    @property
    def _inference_stream(self) -> Optional[torch.cuda.Stream]:
        return get_cuda_stream(device=self._device, purpose="inference")
