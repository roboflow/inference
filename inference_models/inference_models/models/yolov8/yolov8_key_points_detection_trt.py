import threading
from threading import Lock
from typing import List, Optional, Tuple, Union

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
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_KEY_POINTS_THRESHOLD,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS,
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
from inference_models.models.common.roboflow.post_processing import (
    ConfidenceFilter,
    post_process_nms_fused_model_output,
    rescale_key_points_detections,
    run_nms_for_key_points_detection,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.models.common.trt import (
    TRTCudaGraphCache,
    establish_trt_cuda_graph_cache,
    get_trt_engine_inputs_and_outputs,
    infer_from_trt_engine,
    load_trt_model,
)
from inference_models.weights_providers.entities import RecommendedParameters

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running YOLOv8 model with TRT backend on GPU requires pycuda installation, which is brought with "
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
        message="Running YOLOv8 model with TRT backend on GPU requires pycuda installation, which is brought with "
        "`trt-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support.",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class YOLOv8ForKeyPointsDetectionTRT(
    KeyPointsDetectionModel[torch.Tensor, PreProcessingMetadata, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        engine_host_code_allowed: bool = False,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache] = None,
        default_trt_cuda_graph_cache_size: int = 8,
        recommended_parameters: Optional[RecommendedParameters] = None,
        **kwargs,
    ) -> "YOLOv8ForKeyPointsDetectionTRT":
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
                    ResizeMode.LETTERBOX,
                    127,
                    "YOLOv8 Key Points detection model running with TRT backend was trained with "
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
        trt_config = parse_trt_config(
            config_path=model_package_content["trt_config.json"]
        )
        parsed_key_points_metadata, skeletons = parse_key_points_metadata(
            key_points_metadata_path=model_package_content["keypoints_metadata.json"]
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
        if len(outputs) != 1:
            raise CorruptedModelPackageError(
                message=f"Implementation assume single model output, found: {len(outputs)}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        trt_cuda_graph_cache = establish_trt_cuda_graph_cache(
            default_cuda_graph_cache_size=default_trt_cuda_graph_cache_size,
            cuda_graph_cache=trt_cuda_graph_cache,
        )
        return cls(
            engine=engine,
            input_name=inputs[0],
            output_name=outputs[0],
            class_names=class_names,
            skeletons=skeletons,
            inference_config=inference_config,
            parsed_key_points_metadata=parsed_key_points_metadata,
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
        output_name: str,
        class_names: List[str],
        skeletons: List[List[Tuple[int, int]]],
        inference_config: InferenceConfig,
        parsed_key_points_metadata: List[List[str]],
        trt_config: TRTConfig,
        device: torch.device,
        cuda_context: cuda.Context,
        execution_context: trt.IExecutionContext,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache],
        recommended_parameters=None,
    ):
        self._engine = engine
        self._input_name = input_name
        self._output_names = [output_name]
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._trt_cuda_graph_cache = trt_cuda_graph_cache
        self._class_names = class_names
        self._skeletons = skeletons
        self._inference_config = inference_config
        self._parsed_key_points_metadata = parsed_key_points_metadata
        self._trt_config = trt_config
        self._device = device
        self._session_thread_lock = Lock()
        self.recommended_parameters = recommended_parameters
        self._key_points_classes_for_instances = torch.tensor(
            [len(e) for e in self._parsed_key_points_metadata], device=device
        )
        self._key_points_slots_in_prediction = max(
            len(e) for e in parsed_key_points_metadata
        )
        self._inference_stream = torch.cuda.Stream(device=self._device)
        self._thread_local_storage = threading.local()

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def key_points_classes(self) -> List[List[str]]:
        return self._parsed_key_points_metadata

    @property
    def skeletons(self) -> List[List[Tuple[int, int]]]:
        return self._skeletons

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        with torch.cuda.stream(self._pre_process_stream):
            pre_processed_images, pre_processing_meta = pre_process_network_input(
                images=images,
                image_pre_processing=self._inference_config.image_pre_processing,
                network_input=self._inference_config.network_input,
                target_device=self._device,
                input_color_format=input_color_format,
                pre_processing_overrides=pre_processing_overrides,
            )
        self._pre_process_stream.synchronize()
        return pre_processed_images, pre_processing_meta

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        disable_cuda_graphs: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        cache = self._trt_cuda_graph_cache if not disable_cuda_graphs else None
        with self._session_thread_lock:
            with use_cuda_context(context=self._cuda_context):
                return infer_from_trt_engine(
                    pre_processed_images=pre_processed_images,
                    trt_config=self._trt_config,
                    engine=self._engine,
                    context=self._execution_context,
                    device=self._device,
                    input_name=self._input_name,
                    outputs=self._output_names,
                    stream=self._inference_stream,
                    trt_cuda_graph_cache=cache,
                )[0]

    def post_process(
        self,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        iou_threshold: float = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD,
        max_detections: int = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS,
        class_agnostic_nms: bool = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS,
        key_points_threshold: float = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_KEY_POINTS_THRESHOLD,
        **kwargs,
    ) -> Tuple[List[KeyPoints], Optional[List[Detections]]]:
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE,
        )
        confidence = confidence_filter.get_threshold(self.class_names)
        with torch.cuda.stream(self._post_process_stream):
            model_results.record_stream(self._post_process_stream)
            if self._inference_config.post_processing.fused:
                nms_results = post_process_nms_fused_model_output(
                    output=model_results, conf_thresh=confidence
                )
            else:
                nms_results = run_nms_for_key_points_detection(
                    output=model_results,
                    num_classes=len(self._class_names),
                    key_points_slots_in_prediction=self._key_points_slots_in_prediction,
                    conf_thresh=confidence,
                    iou_thresh=iou_threshold,
                    max_detections=max_detections,
                    class_agnostic=class_agnostic_nms,
                )
            rescaled_results = rescale_key_points_detections(
                detections=nms_results,
                images_metadata=pre_processing_meta,
                num_classes=len(self._class_names),
                key_points_slots_in_prediction=self._key_points_slots_in_prediction,
            )
            detections, all_key_points = [], []
            for result in rescaled_results:
                class_id = result[:, 5].int()
                detections.append(
                    Detections(
                        xyxy=result[:, :4].round().int(),
                        class_id=class_id,
                        confidence=result[:, 4],
                    )
                )
                key_points_reshaped = result[:, 6:].view(
                    result.shape[0], self._key_points_slots_in_prediction, 3
                )
                xy = key_points_reshaped[:, :, :2]
                predicted_key_points_confidence = key_points_reshaped[:, :, 2]
                key_points_classes_for_instance_class = (
                    (self._key_points_classes_for_instances[class_id])
                    .unsqueeze(1)
                    .to(device=result.device)
                )
                invalid_slot_keypoints = (
                    torch.arange(
                        self._key_points_slots_in_prediction, device=result.device
                    )
                    .unsqueeze(0)
                    .repeat(result.shape[0], 1)
                    >= key_points_classes_for_instance_class
                )

                keypoints_below_threshold = (
                    predicted_key_points_confidence < key_points_threshold
                )
                mask = invalid_slot_keypoints | keypoints_below_threshold
                xy[mask] = 0.0
                predicted_key_points_confidence[mask] = 0.0
                all_key_points.append(
                    KeyPoints(
                        xy=xy.round().int(),
                        class_id=class_id,
                        confidence=predicted_key_points_confidence,
                    )
                )
        self._post_process_stream.synchronize()
        return all_key_points, detections

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
