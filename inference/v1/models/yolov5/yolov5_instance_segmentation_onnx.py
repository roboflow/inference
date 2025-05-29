from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import onnxruntime
import torch

from inference.v1 import InstanceDetections, InstanceSegmentationModel
from inference.v1.configuration import DEFAULT_DEVICE, ONNXRUNTIME_EXECUTION_PROVIDERS
from inference.v1.entities import ColorFormat
from inference.v1.errors import EnvironmentConfigurationError
from inference.v1.models.common.model_packages import get_model_package_contents
from inference.v1.models.common.onnx import (
    run_session_via_iobinding,
    set_execution_provider_defaults,
)
from inference.v1.models.common.post_processing import (
    align_instance_segmentation_results,
    crop_masks_to_boxes,
    preprocess_segmentation_masks,
    run_nms_for_instance_segmentation,
)
from inference.v1.models.common.roboflow.model_packages import (
    PreProcessingConfig,
    PreProcessingMetadata,
    parse_class_names_file,
    parse_pre_processing_config,
)
from inference.v1.models.common.roboflow.pre_processing import pre_process_network_input
from inference.v1.models.yolov5.nms import run_yolov5_nms_for_instance_segmentation


class YOLOv5ForInstanceSegmentationOnnx(
    InstanceSegmentationModel[
        torch.Tensor, PreProcessingMetadata, Tuple[torch.Tensor, torch.Tensor]
    ]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "YOLOv5ForInstanceSegmentationOnnx":
        if execution_providers is None:
            execution_providers = ONNXRUNTIME_EXECUTION_PROVIDERS
        if not execution_providers:
            raise EnvironmentConfigurationError(
                f"Could not initialize model - selected backend is ONNX which requires execution provider to "
                f"be specified - explicitly in `from_pretrained(...)` method or via env variable "
                f"`ONNXRUNTIME_EXECUTION_PROVIDERS`. If you run model locally - adjust your setup, otherwise "
                f"contact the platform support."
            )
        execution_providers = set_execution_provider_defaults(
            providers=execution_providers,
            model_package_path=model_name_or_path,
            device=device,
            default_trt_options=default_trt_options,
        )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "environment.json",
                "yolov5s_weights.onnx",
            ],
        )
        class_names = parse_class_names_file(
            class_names_path=model_package_content["class_names.txt"]
        )
        pre_processing_config = parse_pre_processing_config(
            environment_file_path=model_package_content["environment.json"],
        )
        session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content["yolov5s_weights.onnx"],
            providers=execution_providers,
        )
        input_batch_size = session.get_inputs()[0].shape[0]
        if isinstance(input_batch_size, str):
            input_batch_size = None
        return cls(
            session=session,
            class_names=class_names,
            pre_processing_config=pre_processing_config,
            device=device,
            input_batch_size=input_batch_size,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        pre_processing_config: PreProcessingConfig,
        class_names: List[str],
        device: torch.device,
        input_batch_size: Optional[int],
    ):
        self._session = session
        self._pre_processing_config = pre_processing_config
        self._class_names = class_names
        self._device = device
        self._input_batch_size = input_batch_size
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

    def forward(
        self, pre_processed_images: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._session_thread_lock:
            if self._input_batch_size is None:
                instances, protos = run_session_via_iobinding(
                    session=self._session,
                    input_name="images",
                    inputs=pre_processed_images,
                )
                return instances, protos
            instances, protos = [], []
            for i in range(0, pre_processed_images.shape[0], self._input_batch_size):
                batch_input = pre_processed_images[
                    i : i + self._input_batch_size
                ].contiguous()
                batch_instances, batch_protos = run_session_via_iobinding(
                    session=self._session, input_name="images", inputs=batch_input
                )
                instances.append(batch_instances)
                protos.append(batch_protos)
            return torch.cat(instances, dim=0), torch.cat(protos, dim=0)

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        max_detections: int = 100,
        mask_threshold: float = 0.5,
        class_agnostic: bool = False,
        **kwargs,
    ) -> List[InstanceDetections]:
        instances, protos = model_results
        nms_results = run_yolov5_nms_for_instance_segmentation(
            output=instances.permute(0, 2, 1),
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            max_detections=max_detections,
            class_agnostic=class_agnostic,
        )
        final_results = []
        for image_bboxes, image_protos, image_meta in zip(
            nms_results, protos, pre_processing_meta
        ):
            pre_processed_masks = preprocess_segmentation_masks(
                protos=image_protos,
                masks_in=image_bboxes[:, 6:],
                mask_threshold=mask_threshold,
            )
            cropped_masks = crop_masks_to_boxes(
                image_bboxes[:, :4], pre_processed_masks
            )
            padding = (
                image_meta.pad_left,
                image_meta.pad_top,
                image_meta.pad_right,
                image_meta.pad_bottom,
            )
            aligned_boxes, aligned_masks = align_instance_segmentation_results(
                image_bboxes=image_bboxes,
                masks=cropped_masks,
                padding=padding,
                scale_height=image_meta.scale_height,
                scale_width=image_meta.scale_width,
                original_size=image_meta.original_size,
                inference_size=image_meta.inference_size,
            )
            final_results.append(
                InstanceDetections(
                    xyxy=aligned_boxes[:, :4].round().int(),
                    class_id=aligned_boxes[:, 5].int(),
                    confidence=aligned_boxes[:, 4],
                    mask=aligned_masks,
                )
            )
        return final_results
