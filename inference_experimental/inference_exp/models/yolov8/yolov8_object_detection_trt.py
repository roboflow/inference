from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
from inference_exp import Detections, ObjectDetectionModel
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.errors import ModelRuntimeError
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.common.post_processing import (
    rescale_detections,
    run_nms_for_object_detection,
)
from inference_exp.models.common.roboflow.model_packages import (
    PreProcessingConfig,
    PreProcessingMetadata,
    TRTConfig,
    parse_class_names_file,
    parse_pre_processing_config,
    parse_trt_config,
)
from inference_exp.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_exp.models.common.trt import infer_from_trt_engine, load_model


class YOLOv8ForObjectDetectionTRT(
    ObjectDetectionModel[torch.Tensor, PreProcessingMetadata, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "YOLOv8ForObjectDetectionTRT":
        if device.type != "cuda":
            raise ModelRuntimeError(
                f"TRT engine only runs on CUDA device - {device} device detected."
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
            environment_file_path=model_package_content["environment.json"],
        )
        trt_config = parse_trt_config(
            config_path=model_package_content["trt_config.json"]
        )
        engine = load_model(model_path=model_package_content["engine.plan"])
        context = engine.create_execution_context()
        return cls(
            engine=engine,
            context=context,
            class_names=class_names,
            pre_processing_config=pre_processing_config,
            trt_config=trt_config,
            device=device,
        )

    def __init__(
        self,
        engine: trt.ICudaEngine,
        context: trt.IExecutionContext,
        class_names: List[str],
        pre_processing_config: PreProcessingConfig,
        trt_config: TRTConfig,
        device: torch.device,
    ):
        self._engine = engine
        self._context = context
        self._class_names = class_names
        self._pre_processing_config = pre_processing_config
        self._trt_config = trt_config
        self._device = device
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
            """
            TensorRT objects are generally not thread-safe; the client must serialize access to objects
            from different threads.
            The expected runtime concurrency model is that different threads operate in different execution
            contexts. The context contains the network state (activation values and so on) during execution,
            so using a context concurrently in different threads results in undefined behavior.

            Nvidia 21, 37 (https://docs.nvidia.com/deeplearning/tensorrt/10.8.0/architecture/how-trt-works.html)
            """
            return infer_from_trt_engine(
                pre_processed_images=pre_processed_images,
                trt_config=self._trt_config,
                engine=self._engine,
                context=self._context,
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
        nms_results = run_nms_for_object_detection(
            output=model_results,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            max_detections=max_detections,
            class_agnostic=class_agnostic,
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
        return results
