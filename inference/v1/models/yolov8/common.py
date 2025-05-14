from typing import List

import torch
import torchvision

from inference.v1.models.common.roboflow.model_packages import PreProcessingMetadata


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
        # Combine transpose & max for efficiency
        class_scores = scores[b]  # (80, 8400)
        class_conf, class_ids = class_scores.max(0)  # (8400,), (8400,)
        mask = class_conf > conf_thresh
        if not torch.any(mask):
            results.append(torch.zeros((0, 6), device=output.device))
            continue
        bboxes = boxes[b][:, mask].T  # (num, 4) -- selects and then transposes
        class_conf = class_conf[mask]
        class_ids = class_ids[mask]
        # Vectorized [x, y, w, h] -> [x1, y1, x2, y2]
        xy = bboxes[:, :2]
        wh = bboxes[:, 2:]
        half_wh = wh / 2
        xyxy = torch.cat((xy - half_wh, xy + half_wh), 1)
        # Class-agnostic NMS -> use dummy class ids
        nms_class_ids = torch.zeros_like(class_ids) if class_agnostic else class_ids
        # NMS and limiting max detections
        keep = torchvision.ops.batched_nms(xyxy, class_conf, nms_class_ids, iou_thresh)
        if keep.numel() > max_detections:
            keep = keep[:max_detections]
        detections = torch.cat(
            (
                xyxy[keep],
                class_conf[keep, None],  # unsqueeze(1) is replaced with None
                class_ids[keep, None].float(),
            ),
            1,
        )  # [x1, y1, x2, y2, conf, cls]

        results.append(detections)
    return results


def rescale_detections(
    detections: List[torch.Tensor], images_metadata: List[PreProcessingMetadata]
) -> List[torch.Tensor]:
    for image_detections, metadata in zip(detections, images_metadata):
        # Extract reference dtype and device
        dtype = image_detections.dtype
        device = image_detections.device

        # Create the offsets and scaling tensors once per image
        pad_left = metadata.pad_left
        pad_top = metadata.pad_top
        scale_width = metadata.scale_width
        scale_height = metadata.scale_height

        # Use tensor construction directly, avoid as_tensor overhead
        offsets = torch.tensor(
            [pad_left, pad_top, pad_left, pad_top], dtype=dtype, device=device
        )
        scales = torch.tensor(
            [scale_width, scale_height, scale_width, scale_height],
            dtype=dtype,
            device=device,
        )
        # Use variable for detections slice to avoid repeated indexing
        image_detections_4 = image_detections[:, :4]
        image_detections_4.sub_(offsets)
        image_detections_4.div_(scales)
    return detections
