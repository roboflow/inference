from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp import Detections, KeyPoints, KeyPointsDetectionModel
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.errors import MissingDependencyError, ModelRuntimeError
from inference_exp.models.common.cuda import use_cuda_context, use_primary_cuda_context
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.common.roboflow.model_packages import (
    PreProcessingConfig,
    PreProcessingMetadata,
    TRTConfig,
    parse_class_names_file,
    parse_key_points_metadata,
    parse_pre_processing_config,
    parse_trt_config,
)
from inference_exp.models.common.roboflow.post_processing import (
    rescale_key_points_detections,
    run_nms_for_key_points_detection,
)
from inference_exp.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_exp.models.common.trt import infer_from_trt_engine, load_model

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import YOLOv8 model with TRT backend - this error means that some additional dependencies "
        f"are not installed in the environment. If you run the `inference-exp` library directly in your Python "
        f"program, make sure the following extras of the package are installed: `trt10` - installation can only "
        f"succeed for Linux and Windows machines with Cuda 12 installed. Jetson devices, should have TRT 10.x "
        f"installed for all builds with Jetpack 6. "
        f"If you see this error using Roboflow infrastructure, make sure the service you use does support the model. "
        f"You can also contact Roboflow to get support.",
        help_url="https://todo",
    ) from import_error

try:
    import pycuda.driver as cuda
except ImportError as import_error:
    raise MissingDependencyError(
        message="TODO",
        help_url="https://todo",
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
        **kwargs,
    ) -> "YOLOv8ForKeyPointsDetectionTRT":
        if device.type != "cuda":
            raise ModelRuntimeError(
                message=f"TRT engine only runs on CUDA device - {device} device detected.",
                help_url="https://todo",
            )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "environment.json",
                "trt_config.json",
                "engine.plan",
                "keypoints_metadata.json",
            ],
        )
        class_names = parse_class_names_file(
            class_names_path=model_package_content["class_names.txt"]
        )
        pre_processing_config = parse_pre_processing_config(
            config_path=model_package_content["environment.json"],
        )
        trt_config = parse_trt_config(
            config_path=model_package_content["trt_config.json"]
        )
        parsed_key_points_metadata = parse_key_points_metadata(
            key_points_metadata_path=model_package_content["keypoints_metadata.json"]
        )
        cuda.init()
        cuda_device = cuda.Device(device.index or 0)
        with use_primary_cuda_context(cuda_device=cuda_device) as cuda_context:
            engine = load_model(
                model_path=model_package_content["engine.plan"],
                engine_host_code_allowed=engine_host_code_allowed,
            )
            execution_context = engine.create_execution_context()
        return cls(
            engine=engine,
            class_names=class_names,
            pre_processing_config=pre_processing_config,
            parsed_key_points_metadata=parsed_key_points_metadata,
            trt_config=trt_config,
            device=device,
            execution_context=execution_context,
        )

    def __init__(
        self,
        engine: trt.ICudaEngine,
        class_names: List[str],
        pre_processing_config: PreProcessingConfig,
        parsed_key_points_metadata: List[List[str]],
        trt_config: TRTConfig,
        device: torch.device,
        cuda_context: cuda.Context,
        execution_context: trt.IExecutionContext,
    ):
        self._engine = engine
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._class_names = class_names
        self._pre_processing_config = pre_processing_config
        self._parsed_key_points_metadata = parsed_key_points_metadata
        self._trt_config = trt_config
        self._device = device
        self._session_thread_lock = Lock()
        self._parsed_key_points_metadata = parsed_key_points_metadata
        self._key_points_classes_for_instances = torch.tensor(
            [len(e) for e in self._parsed_key_points_metadata], device=device
        )
        self._key_points_slots_in_prediction = max(
            len(e) for e in parsed_key_points_metadata
        )

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def key_points_classes(self) -> List[List[str]]:
        return self._parsed_key_points_metadata

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        return pre_process_network_input(
            images=images,
            pre_processing_config=self._pre_processing_config,
            expected_network_color_format="rgb",
            target_device=self._device,
            input_color_format=input_color_format,
        )

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        with self._session_thread_lock:
            with use_cuda_context(context=self._cuda_context):
                return infer_from_trt_engine(
                    pre_processed_images=pre_processed_images,
                    trt_config=self._trt_config,
                    engine=self._engine,
                    context=self._execution_context,
                    device=self._device,
                    input_name="images",
                    outputs=["output0"],
                )[0]

    def post_process(
        self,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        max_detections: int = 100,
        class_agnostic: bool = False,
        key_points_threshold: float = 0.3,
        **kwargs,
    ) -> Tuple[List[KeyPoints], Optional[List[Detections]]]:
        nms_results = run_nms_for_key_points_detection(
            output=model_results,
            num_classes=len(self._class_names),
            key_points_slots_in_prediction=self._key_points_slots_in_prediction,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            max_detections=max_detections,
            class_agnostic=class_agnostic,
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
            key_points_reshaped = result[:, 6:].view(result.shape[0], -1, 3)
            xy = key_points_reshaped[:, :, :2]
            confidence = key_points_reshaped[:, :, 2]
            key_points_classes_for_instance_class = (
                self._key_points_classes_for_instances[class_id]
            )
            instances_class_mask = (
                torch.arange(self._key_points_slots_in_prediction, device=result.device)
                .unsqueeze(0)
                .repeat(result.shape[0], 1)
                < key_points_classes_for_instance_class
            )
            confidence_mask = confidence < key_points_threshold
            mask = instances_class_mask & confidence_mask
            xy[mask] = 0.0
            confidence[mask] = 0.0
            all_key_points.append(
                KeyPoints(xy=xy.round().int(), class_id=class_id, confidence=confidence)
            )
        return all_key_points, detections
