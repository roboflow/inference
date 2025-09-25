from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision  # DO NOT REMOVE, THIS IMPORT ENABLES NMS OPERATION
from inference_exp import Detections, KeyPoints, KeyPointsDetectionModel
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.errors import CorruptedModelPackageError
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    parse_class_names_file,
    parse_inference_config,
    parse_key_points_metadata,
)
from inference_exp.models.common.roboflow.post_processing import (
    post_process_nms_fused_model_output,
    rescale_key_points_detections,
    run_nms_for_key_points_detection,
)
from inference_exp.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_exp.models.common.torch import generate_batch_chunks


class YOLOv8ForKeyPointsDetectionTorchScript(
    KeyPointsDetectionModel[torch.Tensor, PreProcessingMetadata, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "YOLOv8ForKeyPointsDetectionTorchScript":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "weights.torchscript",
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
        parsed_key_points_metadata, skeletons = parse_key_points_metadata(
            key_points_metadata_path=model_package_content["keypoints_metadata.json"]
        )
        model = torch.jit.load(
            model_package_content["weights.torchscript"], map_location=device
        ).eval()
        return cls(
            model=model,
            class_names=class_names,
            inference_config=inference_config,
            device=device,
            parsed_key_points_metadata=parsed_key_points_metadata,
            skeletons=skeletons,
        )

    def __init__(
        self,
        model: torch.nn.Module,
        inference_config: InferenceConfig,
        class_names: List[str],
        device: torch.device,
        parsed_key_points_metadata: List[List[str]],
        skeletons: List[List[Tuple[int, int]]],
    ):
        self._model = model
        self._inference_config = inference_config
        self._class_names = class_names
        self._skeletons = skeletons
        self._device = device
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

    @property
    def skeletons(self) -> List[List[Tuple[int, int]]]:
        return self._skeletons

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

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.inference_mode():
            if (
                pre_processed_images.shape[0]
                == self._inference_config.forward_pass.static_batch_size
            ):
                return self._model(pre_processed_images).to(self._device)
            results = []
            for input_tensor, padding_size in generate_batch_chunks(
                input_batch=pre_processed_images,
                chunk_size=self._inference_config.forward_pass.static_batch_size,
            ):
                result_for_chunk = self._model(input_tensor)
                if padding_size > 0:
                    result_for_chunk = result_for_chunk[:-padding_size]
                results.append(result_for_chunk)
            return torch.cat(results, dim=0).to(self._device)

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
        if self._inference_config.post_processing.fused:
            nms_results = post_process_nms_fused_model_output(
                output=model_results, conf_thresh=conf_thresh
            )
        else:
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
                (self._key_points_classes_for_instances[class_id])
                .unsqueeze(1)
                .to(device=result.device)
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
