import json
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision

from inference.v1 import Detections
from inference.v1.entities import ColorFormat, ImageDimensions
from inference.v1.errors import CorruptedModelPackageError, ModelRuntimeError
from inference.v1.utils.file_system import read_json, stream_file_lines


def parse_class_names_file(class_names_path: str) -> List[str]:
    try:
        return list(stream_file_lines(path=class_names_path))
    except OSError as error:
        raise CorruptedModelPackageError(
            f"Could not decode file {class_names_path} which is supposed to provide list of model class names. "
            f"If you created model package manually, please verify its consistency in docs. In case that the "
            f"weights are hosted on the Roboflow platform - contact support."
        ) from error


class PreProcessingMode(Enum):
    NONE = "NONE"
    STRETCH = "STRETCH"
    LETTERBOX = "LETTERBOX"


@dataclass(frozen=True)
class PreProcessingConfig:
    mode: PreProcessingMode
    target_size: Optional[ImageDimensions] = None
    padding_value: Optional[int] = None


PADDING_VALUES_MAPPING = {
    "black edges": 0,
    "grey edges": 127,
    "white edges": 255,
}

PreProcessingMetadata = namedtuple(
    "PreProcessingMetadata",
    [
        "pad_left",
        "pad_top",
        "original_size",
        "inference_size",
        "scale_width",
        "scale_height",
    ],
)


def parse_pre_processing_config(environment_file_path: str) -> PreProcessingConfig:
    try:
        content = read_json(path=environment_file_path)
        if not content:
            raise ValueError("file is empty.")
        if not isinstance(content, dict):
            raise ValueError("file is malformed (not a JSON dictionary)")
        if "PREPROCESSING" not in content:
            raise ValueError("file is malformed (lack of `PREPROCESSING` key)")
        preprocessing_dict = json.loads(content["PREPROCESSING"])
        resize_config = preprocessing_dict["resize"]
        if not resize_config["enabled"]:
            return PreProcessingConfig(mode=PreProcessingMode.NONE)
        target_size = ImageDimensions(
            width=int(resize_config["width"]), height=int(resize_config["height"])
        )
        if resize_config["format"] == "Stretch to":
            return PreProcessingConfig(
                mode=PreProcessingMode.STRETCH, target_size=target_size
            )
        for padding_color_infix, padding_value in PADDING_VALUES_MAPPING.items():
            if padding_color_infix in resize_config["format"]:
                return PreProcessingConfig(
                    mode=PreProcessingMode.LETTERBOX,
                    target_size=target_size,
                    padding_value=padding_value,
                )
        raise ValueError("could not determine resize method or padding color")
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            f"Environment file located under path {environment_file_path} is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs."
        )


@dataclass
class ModelCharacteristics:
    task_type: str
    model_type: str


def parse_model_characteristics(config_path: str) -> ModelCharacteristics:
    try:
        with open(config_path) as f:
            parsed_config = json.load(f)
            if "project_task_type" not in parsed_config or "model_type" not in parsed_config:
                raise ValueError(
                    "could not find required entries in config - either "
                    "'project_task_type' or 'model_type' field is missing"
                )
            return ModelCharacteristics(
                task_type=parsed_config["project_task_type"],
                model_type=parsed_config["model_type"],
            )
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            f"Model type config file located under path {config_path} is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs."
        )


def pre_process_images_tensor(
    images: torch.Tensor,
    pre_processing_config: PreProcessingConfig,
    input_color_format: ColorFormat,
    target_device: torch.device,
    normalization_constant: float = 255.0,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    if (
        pre_processing_config.mode is PreProcessingMode.NONE
        or pre_processing_config.target_size is None
    ):
        raise ModelRuntimeError(
            "Could not pre-process data before model inference - pre-processing configuration "
            "does not specify input resizing."
        )
    if images.device != target_device:
        images = images.to(target_device)
    if len(images.shape) == 3:
        images = torch.unsqueeze(images, 0)
    if images.shape[1] != 3 and images.shape[3] == 3:
        images = images.permute(0, 3, 1, 2)
    original_size = ImageDimensions(height=images.shape[2], width=images.shape[3])
    if pre_processing_config.mode is PreProcessingMode.STRETCH:
        images = torch.nn.functional.interpolate(
            images,
            [
                pre_processing_config.target_size.height,
                pre_processing_config.target_size.width,
            ],
            mode="bilinear",
        )
        if input_color_format == "bgr":
            images = images[:, [2, 1, 0], :, :]
        metadata = PreProcessingMetadata(
            pad_left=0,
            pad_top=0,
            original_size=original_size,
            inference_size=pre_processing_config.target_size,
            scale_width=pre_processing_config.target_size.width / original_size.width,
            scale_height=pre_processing_config.target_size.height
            / original_size.height,
        )
        return (images / normalization_constant).contiguous(), [
            metadata
        ] * images.shape[0]
    if pre_processing_config.mode is not PreProcessingMode.LETTERBOX:
        raise ModelRuntimeError(
            f"Cannot find implementation for pre-processing operation: {pre_processing_config.mode}"
        )
    original_height, original_width = images.shape[2], images.shape[3]
    scale_w = pre_processing_config.target_size.width / original_width
    scale_h = pre_processing_config.target_size.height / original_height
    scale = min(scale_w, scale_h)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    pad_top = int((pre_processing_config.target_size.height - new_height) / 2)
    pad_left = int((pre_processing_config.target_size.width - new_width) / 2)
    images = torch.nn.functional.interpolate(
        images,
        [new_height, new_width],
        mode="bilinear",
    )
    if input_color_format == "bgr":
        images = images[:, [2, 1, 0], :, :]
    final_batch = torch.full(
        (
            images.shape[0],
            images.shape[1],
            pre_processing_config.target_size.height,
            pre_processing_config.target_size.width,
        ),
        pre_processing_config.padding_value / normalization_constant,
        dtype=torch.float32,
        device=target_device,
    )
    final_batch[
        :, :, pad_top : pad_top + new_height, pad_left : pad_left + new_width
    ] = images
    metadata = PreProcessingMetadata(
        pad_left=pad_left,
        pad_top=pad_top,
        original_size=original_size,
        inference_size=pre_processing_config.target_size,
        scale_width=new_width / original_size.width,
        scale_height=new_height / original_size.height,
    )
    return (final_batch / normalization_constant).contiguous(), [
        metadata
    ] * final_batch.shape[0]


def pre_process_images_list(
    images: List[torch.Tensor],
    pre_processing_config: PreProcessingConfig,
    input_color_format: ColorFormat,
    target_device: torch.device,
    normalization_constant: float = 255.0,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    if (
        pre_processing_config.mode is PreProcessingMode.NONE
        or pre_processing_config.target_size is None
    ):
        raise ModelRuntimeError(
            "Could not pre-process data before model inference - pre-processing configuration "
            "does not specify input resizing."
        )
    target_height = pre_processing_config.target_size.height
    target_width = pre_processing_config.target_size.width
    if pre_processing_config.mode is PreProcessingMode.STRETCH:
        processed = []
        images_metadata = []
        for img in images:
            if len(img.shape) != 3:
                raise ModelRuntimeError(
                    "When providing List[torch.Tensor] as input, model requires tensors to have 3 dimensions."
                )
            if img.device != target_device:
                img = img.to(target_device)
            if img.shape[0] != 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            original_size = ImageDimensions(height=img.shape[1], width=img.shape[2])
            if input_color_format == "bgr":
                img = img[[2, 1, 0], :, :]
            img = img.unsqueeze(0)
            img = F.interpolate(
                img, size=(target_height, target_width), mode="bilinear"
            )
            img = img / normalization_constant
            processed.append(img.contiguous())
            image_metadata = PreProcessingMetadata(
                pad_left=0,
                pad_top=0,
                original_size=original_size,
                inference_size=pre_processing_config.target_size,
                scale_width=pre_processing_config.target_size.width
                / original_size.width,
                scale_height=pre_processing_config.target_size.height
                / original_size.height,
            )
            images_metadata.append(image_metadata)
        return torch.concat(processed, dim=0).contiguous(), images_metadata
    if pre_processing_config.mode is PreProcessingMode.LETTERBOX:
        target_h, target_w = (
            pre_processing_config.target_size.height,
            pre_processing_config.target_size.width,
        )
        num_images = len(images)
        final_batch = torch.full(
            (num_images, 3, target_h, target_w),
            pre_processing_config.padding_value or 0,
            dtype=torch.float32,
            device=target_device,
        )
        original_shapes = torch.tensor(
            [[img.shape[0], img.shape[1]] for img in images], dtype=torch.float32
        )
        scale_w = target_w / original_shapes[:, 1]
        scale_h = target_h / original_shapes[:, 0]
        scales = torch.minimum(scale_w, scale_h)
        new_ws = (original_shapes[:, 1] * scales).int()
        new_hs = (original_shapes[:, 0] * scales).int()
        pad_tops = ((target_h - new_hs) / 2).int()
        pad_lefts = ((target_w - new_ws) / 2).int()
        images_metadata = []
        for i in range(num_images):
            image_hwc = images[i].to(target_device)  # Ensure on correct device
            if image_hwc.dtype != torch.float32:
                image_hwc = image_hwc.float()
            if input_color_format == "bgr":
                image_hwc = image_hwc[..., [2, 1, 0]]
            if image_hwc.shape[0] != 3 and image_hwc.shape[-1] == 3:
                image_hwc = image_hwc.permute(2, 0, 1)
            original_size = ImageDimensions(
                height=image_hwc.shape[1], width=image_hwc.shape[2]
            )
            new_h_i, new_w_i = new_hs[i].item(), new_ws[i].item()
            resized_chw = F.interpolate(
                image_hwc, size=(target_height, target_width), mode="bilinear"
            )
            pad_top_i, pad_left_i = pad_tops[i].item(), pad_lefts[i].item()
            final_batch[
                i, :, pad_top_i : pad_top_i + new_h_i, pad_left_i : pad_left_i + new_w_i
            ] = resized_chw
            image_metadata = PreProcessingMetadata(
                pad_left=pad_left_i,
                pad_top=pad_top_i,
                original_size=original_size,
                inference_size=pre_processing_config.target_size,
                scale_width=new_w_i / original_size.width,
                scale_height=new_h_i / original_size.height,
            )
            images_metadata.append(image_metadata)
        return (final_batch / normalization_constant).contiguous(), images_metadata
    raise ModelRuntimeError(
        f"Unsupported pre-processing mode: {pre_processing_config.mode}"
    )


def run_nms(
    output: torch.Tensor,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
) -> List[torch.Tensor]:
    bs = output.shape[0]
    boxes = output[:, :4, :]  # (N, 4, 8400)
    scores = output[:, 4:, :]  # (N, 80, 8400)

    results = []

    for b in range(bs):
        bboxes = boxes[b].T  # (8400, 4)
        class_scores = scores[b].T  # (8400, 80)

        class_conf, class_ids = class_scores.max(1)  # (8400,), (8400,)

        mask = class_conf > conf_thresh
        if mask.sum() == 0:
            results.append(torch.zeros((0, 6), device=output.device))
            continue

        bboxes = bboxes[mask]
        class_conf = class_conf[mask]
        class_ids = class_ids[mask]
        # Convert [x, y, w, h] -> [x1, y1, x2, y2]
        xyxy = torch.zeros_like(bboxes)
        xyxy[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x1
        xyxy[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # y1
        xyxy[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2  # x2
        xyxy[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2  # y2
        # Class-agnostic NMS -> use dummy class ids
        nms_class_ids = torch.zeros_like(class_ids) if class_agnostic else class_ids
        keep = torchvision.ops.batched_nms(xyxy, class_conf, nms_class_ids, iou_thresh)
        keep = keep[:max_detections]
        detections = torch.cat(
            [
                xyxy[keep],
                class_conf[keep].unsqueeze(1),
                class_ids[keep].unsqueeze(1).float(),
            ],
            dim=1,
        )  # [x1, y1, x2, y2, conf, cls]

        results.append(detections)
    return results


def rescale_detections(
    detections: List[torch.Tensor], images_metadata: List[PreProcessingMetadata]
) -> List[torch.Tensor]:
    for image_detections, metadata in zip(detections, images_metadata):
        offsets = torch.tensor(
            [metadata.pad_left, metadata.pad_top, metadata.pad_left, metadata.pad_top],
            dtype=image_detections.dtype,
            device=image_detections.device,
        )
        image_detections[:, :4] -= offsets
        scale = torch.tensor(
            [
                metadata.scale_width,
                metadata.scale_height,
                metadata.scale_width,
                metadata.scale_height,
            ],
            device=image_detections.device,
        )
        image_detections[:, :4] *= 1 / scale
    return detections
