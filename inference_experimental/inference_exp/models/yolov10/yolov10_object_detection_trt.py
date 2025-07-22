from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp import Detections, ObjectDetectionModel
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
    parse_pre_processing_config,
    parse_trt_config,
)
from inference_exp.models.common.roboflow.post_processing import (
    rescale_image_detections,
)
from inference_exp.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_exp.models.common.trt import infer_from_trt_engine, load_model

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import YOLOv10 model with TRT backend - this error means that some additional dependencies "
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


class YOLOv10ForObjectDetectionTRT(
    ObjectDetectionModel[torch.Tensor, PreProcessingMetadata, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "YOLOv10ForObjectDetectionTRT":
        if device.type != "cuda":
            raise ModelRuntimeError(
                message="TRT engine only runs on CUDA device - {device} device detected.",
                help_url="https://todo",
            )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "environment.json",
                "trt_config.json",
                "engine.plan",
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
        cuda.init()
        cuda_device = cuda.Device(device.index or 0)
        with use_primary_cuda_context(cuda_device=cuda_device) as cuda_context:
            engine = load_model(model_path=model_package_content["engine.plan"])
            execution_context = engine.create_execution_context()
        return cls(
            engine=engine,
            class_names=class_names,
            pre_processing_config=pre_processing_config,
            trt_config=trt_config,
            device=device,
            cuda_context=cuda_context,
            execution_context=execution_context,
        )

    def __init__(
        self,
        engine: trt.ICudaEngine,
        class_names: List[str],
        pre_processing_config: PreProcessingConfig,
        trt_config: TRTConfig,
        device: torch.device,
        cuda_context: cuda.Context,
        execution_context: trt.IExecutionContext,
    ):
        self._engine = engine
        self._class_names = class_names
        self._pre_processing_config = pre_processing_config
        self._trt_config = trt_config
        self._device = device
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._session_thread_lock = Lock()

    @property
    def class_names(self) -> List[str]:
        return self._class_names

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
        **kwargs,
    ) -> List[Detections]:
        results = []
        for image_result, metadata in zip(model_results, pre_processing_meta):
            mask = image_result[:, 4] > conf_thresh
            filtered = image_result[mask][:max_detections]
            rescaled = rescale_image_detections(
                image_detections=filtered,
                image_metadata=metadata,
            )
            results.append(
                Detections(
                    xyxy=rescaled[:, :4].round().int(),
                    class_id=rescaled[:, 5].int(),
                    confidence=rescaled[:, 4],
                )
            )
        return results
