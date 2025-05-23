from typing import List, Optional, Tuple, Union

import numpy as np
import tensorrt as trt
import torch

from inference.v1 import Detections, ObjectDetectionModel
from inference.v1.configuration import DEFAULT_DEVICE
from inference.v1.entities import ColorFormat
from inference.v1.errors import ModelRuntimeError
from inference.v1.models.common.model_packages import get_model_package_contents
from inference.v1.models.common.post_processing import rescale_image_detections
from inference.v1.models.common.roboflow.model_packages import (
    PreProcessingConfig,
    PreProcessingMetadata,
    TRTConfig,
    parse_class_names_file,
    parse_pre_processing_config,
    parse_trt_config,
)
from inference.v1.models.common.roboflow.pre_processing import pre_process_network_input
from inference.v1.models.common.trt import infer_from_trt_engine, load_model


class RFDetrForObjectDetectionTRT(
    (
        ObjectDetectionModel[
            torch.Tensor, PreProcessingMetadata, Tuple[torch.Tensor, torch.Tensor]
        ]
    )
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "RFDetrForObjectDetectionTRT":
        if device.type != "cuda":
            raise ModelRuntimeError(
                f"TRT engine only runs on CUDA device - {device} device detected."
            )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "environment.json",
                "model_type.json",
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
            device=device,
            pre_processing_config=pre_processing_config,
            trt_config=trt_config,
        )

    def __init__(
        self,
        engine: trt.ICudaEngine,
        context: trt.IExecutionContext,
        pre_processing_config: PreProcessingConfig,
        class_names: List[str],
        device: torch.device,
        trt_config: TRTConfig,
    ):
        self._engine = engine
        self._context = context
        self._pre_processing_config = pre_processing_config
        self._class_names = class_names
        self._device = device
        self._trt_config = trt_config

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
            normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )

    def forward(
        self, pre_processed_images: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        detections, labels = infer_from_trt_engine(
            pre_processed_images=pre_processed_images,
            trt_config=self._trt_config,
            engine=self._engine,
            context=self._context,
            device=self._device,
            input_name="input",
            outputs=["dets", "labels"],
        )
        return detections, labels

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        threshold: float = 0.5,
        **kwargs,
    ) -> List[Detections]:
        bboxes, logits = model_results
        logits_sigmoid = torch.nn.functional.sigmoid(logits)
        results = []
        for image_bboxes, image_logits, image_meta in zip(
            bboxes, logits_sigmoid, pre_processing_meta
        ):
            confidence, top_classes = image_logits.max(dim=1)
            confidence_mask = confidence > threshold
            confidence = confidence[confidence_mask]
            top_classes = top_classes[confidence_mask]
            selected_boxes = image_bboxes[confidence_mask]
            confidence, sorted_indices = torch.sort(confidence, descending=True)
            top_classes = top_classes[sorted_indices]
            selected_boxes = selected_boxes[sorted_indices]
            cxcy = selected_boxes[:, :2]
            wh = selected_boxes[:, 2:]
            xy_min = cxcy - 0.5 * wh
            xy_max = cxcy + 0.5 * wh
            selected_boxes_xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
            inference_size_hwhw = torch.tensor(
                [
                    image_meta.inference_size.height,
                    image_meta.inference_size.width,
                    image_meta.inference_size.height,
                    image_meta.inference_size.width,
                ],
                device=self._device,
            )
            selected_boxes_xyxy = selected_boxes_xyxy_pct * inference_size_hwhw
            selected_boxes_xyxy = rescale_image_detections(
                image_detections=selected_boxes_xyxy,
                image_metadata=image_logits,
            )
            detections = Detections(
                xyxy=selected_boxes_xyxy.round().int(),
                confidence=confidence,
                class_id=top_classes.int(),
            )
            results.append(detections)
        return results
