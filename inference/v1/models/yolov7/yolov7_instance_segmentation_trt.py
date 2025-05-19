from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorrt as trt
import torch

from inference.v1 import InstanceDetections, InstanceSegmentationModel
from inference.v1.configuration import DEFAULT_DEVICE
from inference.v1.entities import ColorFormat
from inference.v1.errors import ModelRuntimeError
from inference.v1.models.common.model_packages import get_model_package_contents
from inference.v1.models.common.post_processing import (
    align_instance_segmentation_results,
    crop_masks_to_boxes,
    preprocess_segmentation_masks,
    run_nms_for_instance_segmentation,
)
from inference.v1.models.common.roboflow.model_packages import (
    PreProcessingConfig,
    PreProcessingMetadata,
    TRTConfig,
    parse_class_names_file,
    parse_pre_processing_config,
    parse_trt_config,
)
from inference.v1.models.common.roboflow.pre_processing import pre_process_network_input
from inference.v1.models.common.trt import (
    get_output_tensor_names,
    infer_from_trt_engine,
    load_model,
)


class YOLOv7ForInstanceSegmentationTRT(
    InstanceSegmentationModel[
        torch.Tensor, PreProcessingMetadata, Tuple[torch.Tensor, torch.Tensor]
    ]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "YOLOv7ForInstanceSegmentationTRT":
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
        all_output_tensors = get_output_tensor_names(engine=engine)
        output_tensors = [all_output_tensors[0], all_output_tensors[4]]
        context = engine.create_execution_context()
        return cls(
            engine=engine,
            context=context,
            class_names=class_names,
            pre_processing_config=pre_processing_config,
            trt_config=trt_config,
            device=device,
            output_tensors=output_tensors,
        )

    def __init__(
        self,
        engine: trt.ICudaEngine,
        context: trt.IExecutionContext,
        class_names: List[str],
        pre_processing_config: PreProcessingConfig,
        trt_config: TRTConfig,
        device: torch.device,
        output_tensors: List[str],
    ):
        self._engine = engine
        self._context = context
        self._class_names = class_names
        self._pre_processing_config = pre_processing_config
        self._trt_config = trt_config
        self._device = device
        self._output_tensors = output_tensors
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
            """
            TensorRT objects are generally not thread-safe; the client must serialize access to objects
            from different threads.
            The expected runtime concurrency model is that different threads operate in different execution
            contexts. The context contains the network state (activation values and so on) during execution,
            so using a context concurrently in different threads results in undefined behavior.

            Nvidia 21, 37 (https://docs.nvidia.com/deeplearning/tensorrt/10.8.0/architecture/how-trt-works.html)
            """
            instances, protos = infer_from_trt_engine(
                pre_processed_images=pre_processed_images,
                trt_config=self._trt_config,
                engine=self._engine,
                context=self._context,
                device=self._device,
                input_name="images",
                outputs=self._output_tensors,
            )
            return instances, protos

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
        nms_results = run_nms_for_instance_segmentation(
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
