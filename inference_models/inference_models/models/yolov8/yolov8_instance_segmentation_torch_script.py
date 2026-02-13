from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision  # DO NOT REMOVE, THIS IMPORT ENABLES NMS OPERATION

from inference_models import InstanceDetections, InstanceSegmentationModel
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS,
)
from inference_models.entities import ColorFormat
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    parse_class_names_file,
    parse_inference_config,
)
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
    crop_masks_to_boxes,
    post_process_nms_fused_model_output,
    preprocess_segmentation_masks,
    run_nms_for_instance_segmentation,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.models.common.torch import generate_batch_chunks


class YOLOv8ForInstanceSegmentationTorchScript(
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
    ) -> "YOLOv8ForInstanceSegmentationTorchScript":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "weights.torchscript",
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
                    "YOLOv8 Instance Segmentation model running with TorchScript backend was trained with "
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
                help_url="https://todo",
            )
        if inference_config.forward_pass.static_batch_size is None:
            raise CorruptedModelPackageError(
                message="Expected static batch size to be registered in the inference configuration.",
                help_url="https://todo",
            )
        model = torch.jit.load(
            model_package_content["weights.torchscript"], map_location=device
        ).eval()
        return cls(
            model=model,
            class_names=class_names,
            inference_config=inference_config,
            device=device,
        )

    def __init__(
        self,
        model: torch.nn.Module,
        inference_config: InferenceConfig,
        class_names: List[str],
        device: torch.device,
    ):
        self._model = model
        self._inference_config = inference_config
        self._class_names = class_names
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
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
        )

    def forward(
        self, pre_processed_images: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            if (
                pre_processed_images.shape[0]
                == self._inference_config.forward_pass.static_batch_size
            ):
                instances, protos = self._model(pre_processed_images)
                return instances.to(self._device), protos.to(self._device)
            instances, protos = [], []
            for input_tensor, padding_size in generate_batch_chunks(
                input_batch=pre_processed_images,
                chunk_size=self._inference_config.forward_pass.static_batch_size,
            ):
                instances_for_chunk, protos_for_chunk = self._model(input_tensor)
                if padding_size > 0:
                    instances_for_chunk = instances_for_chunk[:-padding_size]
                    protos_for_chunk = protos_for_chunk[:-padding_size]
                instances.append(instances_for_chunk)
                protos.append(protos_for_chunk)
            return torch.cat(instances, dim=0).to(self._device), torch.cat(
                protos, dim=0
            ).to(self._device)

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: float = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE,
        iou_threshold: float = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD,
        max_detections: int = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS,
        class_agnostic_nms: bool = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS,
        **kwargs,
    ) -> List[InstanceDetections]:
        instances, protos = model_results
        if self._inference_config.post_processing.fused:
            nms_results = post_process_nms_fused_model_output(
                output=instances, conf_thresh=confidence
            )
        else:
            nms_results = run_nms_for_instance_segmentation(
                output=instances,
                conf_thresh=confidence,
                iou_thresh=iou_threshold,
                max_detections=max_detections,
                class_agnostic=class_agnostic_nms,
            )
        final_results = []
        for image_bboxes, image_protos, image_meta in zip(
            nms_results, protos, pre_processing_meta
        ):
            pre_processed_masks = preprocess_segmentation_masks(
                protos=image_protos,
                masks_in=image_bboxes[:, 6:],
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
                size_after_pre_processing=image_meta.size_after_pre_processing,
                inference_size=image_meta.inference_size,
                static_crop_offset=image_meta.static_crop_offset,
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
