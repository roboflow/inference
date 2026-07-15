from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models import Detections, ObjectDetectionModel, PreProcessingOverrides
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS,
)
from inference_models.developer_tools import align_device_with_onnx_session
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import (
    CorruptedModelPackageError,
    EnvironmentConfigurationError,
    MissingDependencyError,
    ModelInputError,
)
from inference_models.logger import LOGGER
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.onnx import (
    run_onnx_session_with_batch_size_limit,
    set_onnx_execution_provider_defaults,
)
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    parse_class_names_file,
    parse_inference_config,
)
from inference_models.models.common.roboflow.post_processing import (
    ConfidenceFilter,
    post_process_nms_fused_model_output,
    rescale_detections,
    run_nms_for_object_detection,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_images_tensor,
    pre_process_network_input,
)
from inference_models.models.common.streams import get_cuda_stream
from inference_models.utils.onnx_introspection import (
    get_selected_onnx_execution_providers,
)
from inference_models.weights_providers.entities import RecommendedParameters

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running YOLOv8 model with ONNX backend requires pycuda installation, which is brought with "
        "`onnx-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support."
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class YOLOv8ForObjectDetectionOnnx(
    ObjectDetectionModel[torch.Tensor, PreProcessingMetadata, torch.Tensor]
):
    _BASE_PREPROCESSOR = "base"
    _GPU_NUMPY_PREPROCESSOR = "torch-gpu-numpy-letterbox-v1"


    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        recommended_parameters: Optional[RecommendedParameters] = None,
        preprocess_implementation: str = _BASE_PREPROCESSOR,
        **kwargs,
    ) -> "YOLOv8ForObjectDetectionOnnx":
        if onnx_execution_providers is None:
            onnx_execution_providers = get_selected_onnx_execution_providers()
        if not onnx_execution_providers:
            raise EnvironmentConfigurationError(
                message=f"Could not initialize model - selected backend is ONNX which requires execution provider to "
                f"be specified - explicitly in `from_pretrained(...)` method or via env variable "
                f"`ONNXRUNTIME_EXECUTION_PROVIDERS`. If you run model locally - adjust your setup, otherwise "
                f"contact the platform support.",
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/#environmentconfigurationerror",
            )
        onnx_execution_providers = set_onnx_execution_provider_defaults(
            providers=onnx_execution_providers,
            model_package_path=model_name_or_path,
            device=device,
            default_onnx_trt_options=default_onnx_trt_options,
        )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "weights.onnx",
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
                    "YOLOv8 Object Detection model running with ONNX backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "models with input dimensions fixed during weights export. To ensure interoperability, `letterbox` "
                    "resize mode with gray edges will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
        )
        if inference_config.post_processing.type != "nms":
            raise CorruptedModelPackageError(
                message="Expected NMS to be the post-processing",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content["weights.onnx"],
            providers=onnx_execution_providers,
        )
        device = align_device_with_onnx_session(session=session, device=device)
        input_batch_size = session.get_inputs()[0].shape[0]
        if isinstance(input_batch_size, str):
            input_batch_size = None
        input_name = session.get_inputs()[0].name
        resolved_preprocess_implementation = cls._resolve_preprocess_implementation(
            requested_implementation=preprocess_implementation,
            device=device,
        )
        return cls(
            session=session,
            input_name=input_name,
            class_names=class_names,
            inference_config=inference_config,
            device=device,
            input_batch_size=input_batch_size,
            recommended_parameters=recommended_parameters,
            preprocess_implementation=resolved_preprocess_implementation,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_name: str,
        inference_config: InferenceConfig,
        class_names: List[str],
        device: torch.device,
        input_batch_size: Optional[int],
        recommended_parameters=None,
        preprocess_implementation: str = _BASE_PREPROCESSOR,
    ):
        self._session = session
        self._input_name = input_name
        self._inference_config = inference_config
        self._class_names = class_names
        self._device = device
        self._min_batch_size = input_batch_size
        self._max_batch_size = (
            input_batch_size
            if input_batch_size is not None
            else inference_config.forward_pass.max_dynamic_batch_size
        )
        self._session_thread_lock = Lock()
        self.recommended_parameters = recommended_parameters
        self._preprocess_implementation = preprocess_implementation
        LOGGER.info(
            "YOLOv8 ONNX selected preprocess implementation: %s",
            preprocess_implementation,
        )

    @classmethod
    def _resolve_preprocess_implementation(
        cls,
        requested_implementation: str,
        device: torch.device,
    ) -> str:
        if requested_implementation == "auto":
            return cls._BASE_PREPROCESSOR
        if requested_implementation == cls._BASE_PREPROCESSOR:
            return cls._BASE_PREPROCESSOR
        if requested_implementation != cls._GPU_NUMPY_PREPROCESSOR:
            raise EnvironmentConfigurationError(
                message=(
                    "Unknown YOLOv8 ONNX preprocess implementation "
                    f"{requested_implementation!r}. Supported values are "
                    f"{cls._BASE_PREPROCESSOR!r}, 'auto', and "
                    f"{cls._GPU_NUMPY_PREPROCESSOR!r}."
                ),
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/",
            )
        if device.type != "cuda":
            raise EnvironmentConfigurationError(
                message=(
                    f"YOLOv8 ONNX preprocess implementation "
                    f"{cls._GPU_NUMPY_PREPROCESSOR!r} requires a CUDA device; "
                    f"received {device}."
                ),
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/",
            )
        return cls._GPU_NUMPY_PREPROCESSOR

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        image_size: Optional[Union[Tuple[int, int], int]] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        pre_process_stream = self._pre_process_stream
        with torch.cuda.stream(pre_process_stream):
            if self._preprocess_implementation == self._GPU_NUMPY_PREPROCESSOR:
                pre_processed_images, pre_processing_meta = (
                    self._pre_process_numpy_on_gpu(
                        images=images,
                        input_color_format=input_color_format,
                        image_size=image_size,
                        pre_processing_overrides=pre_processing_overrides,
                    )
                )
            else:
                pre_processed_images, pre_processing_meta = pre_process_network_input(
                    images=images,
                    image_pre_processing=self._inference_config.image_pre_processing,
                    network_input=self._inference_config.network_input,
                    target_device=self._device,
                    input_color_format=input_color_format,
                    image_size_wh=image_size,
                    pre_processing_overrides=pre_processing_overrides,
                )
        if pre_process_stream is not None:
            pre_process_stream.synchronize()
        return pre_processed_images, pre_processing_meta

    def _pre_process_numpy_on_gpu(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat],
        image_size: Optional[Union[Tuple[int, int], int]],
        pre_processing_overrides: Optional[PreProcessingOverrides],
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        if isinstance(images, list):
            if len(images) != 1 or not isinstance(images[0], np.ndarray):
                raise ModelInputError(
                    message=(
                        f"YOLOv8 ONNX preprocess implementation "
                        f"{self._GPU_NUMPY_PREPROCESSOR!r} accepts one NumPy image "
                        "or a one-element list containing one. Use 'base' for "
                        "other batch forms."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/input-validation/",
                )
            images = images[0]
        if not isinstance(images, np.ndarray):
            raise ModelInputError(
                message=(
                    f"YOLOv8 ONNX preprocess implementation "
                    f"{self._GPU_NUMPY_PREPROCESSOR!r} accepts one NumPy image "
                    "or a one-element list containing one; "
                    f"received {type(images).__name__}. Use 'base' for this input."
                ),
                help_url="https://inference-models.roboflow.com/errors/input-validation/",
            )
        if images.ndim != 3 or images.shape[2] != 3 or images.dtype != np.uint8:
            raise ModelInputError(
                message=(
                    f"YOLOv8 ONNX preprocess implementation "
                    f"{self._GPU_NUMPY_PREPROCESSOR!r} requires a uint8 HWC image "
                    f"with three channels; received shape={images.shape}, dtype={images.dtype}."
                ),
                help_url="https://inference-models.roboflow.com/errors/input-validation/",
            )
        if not images.flags.c_contiguous:
            raise ModelInputError(
                message=(
                    f"YOLOv8 ONNX preprocess implementation "
                    f"{self._GPU_NUMPY_PREPROCESSOR!r} requires a C-contiguous image. "
                    "Use 'base' for non-contiguous input."
                ),
                help_url="https://inference-models.roboflow.com/errors/input-validation/",
            )
        if self._inference_config.network_input.resize_mode is not ResizeMode.LETTERBOX:
            raise ModelInputError(
                message=(
                    f"YOLOv8 ONNX preprocess implementation "
                    f"{self._GPU_NUMPY_PREPROCESSOR!r} supports only letterbox packages; "
                    f"received {self._inference_config.network_input.resize_mode.value!r}."
                ),
                help_url="https://inference-models.roboflow.com/errors/input-validation/",
            )
        return pre_process_images_tensor(
            images=torch.from_numpy(images),
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_mode=(
                ColorMode(input_color_format)
                if input_color_format is not None
                else ColorMode.BGR
            ),
            image_size_wh=image_size,
            pre_processing_overrides=pre_processing_overrides,
        )

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        with self._session_thread_lock:
            return run_onnx_session_with_batch_size_limit(
                session=self._session,
                inputs={self._input_name: pre_processed_images},
                min_batch_size=self._min_batch_size,
                max_batch_size=self._max_batch_size,
                stream=self._inference_stream,
            )[0]

    def post_process(
        self,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        iou_threshold: float = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD,
        max_detections: int = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS,
        class_agnostic_nms: bool = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS,
        **kwargs,
    ) -> List[Detections]:
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE,
        )
        confidence = confidence_filter.get_threshold(self.class_names)
        post_process_stream = self._post_process_stream
        with torch.cuda.stream(post_process_stream):
            if post_process_stream is not None:
                model_results.record_stream(post_process_stream)
            if self._inference_config.post_processing.fused:
                nms_results = post_process_nms_fused_model_output(
                    output=model_results, conf_thresh=confidence
                )
            else:
                nms_results = run_nms_for_object_detection(
                    output=model_results,
                    conf_thresh=confidence,
                    iou_thresh=iou_threshold,
                    max_detections=max_detections,
                    class_agnostic=class_agnostic_nms,
                )
            rescaled_results = rescale_detections(
                detections=nms_results,
                images_metadata=pre_processing_meta,
            )
            results = []
            for result in rescaled_results:
                results.append(
                    Detections(
                        xyxy=result[:, :4].round().int(),
                        class_id=result[:, 5].int(),
                        confidence=result[:, 4],
                    )
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
