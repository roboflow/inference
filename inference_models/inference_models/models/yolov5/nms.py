from typing import List, Union

import torch
import torchvision


def run_nms_yolov5(
    output: torch.Tensor,
    conf_thresh: Union[float, torch.Tensor] = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
) -> List[torch.Tensor]:
    """
    `conf_thresh`: scalar applies to all classes; 1-D tensor of shape
    (num_classes,) indexed by class_id for per-class thresholds.
    """
    bs = output.shape[0]
    boxes = output[:, :4, :]
    top_classes_conf = output[:, 4, :]
    scores = output[:, 5:, :]
    results = []
    for b in range(bs):
        class_scores = scores[b]
        class_conf, class_ids = class_scores.max(0)
        if isinstance(conf_thresh, torch.Tensor):
            mask = top_classes_conf[b] > conf_thresh.to(output.device)[class_ids]
        else:
            mask = top_classes_conf[b] > conf_thresh
        if not torch.any(mask):
            results.append(torch.zeros((0, 6), device=output.device))
            continue
        bboxes = boxes[b][:, mask].T
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


def run_yolov5_nms_for_instance_segmentation(
    output: torch.Tensor,
    conf_thresh: Union[float, torch.Tensor] = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
) -> List[torch.Tensor]:
    """
    `conf_thresh`: scalar applies to all classes; 1-D tensor of shape
    (num_classes,) indexed by class_id for per-class thresholds.
    Objectness-dominated rows (class_id==0 in slice-space) bypass the
    per-class check and are kept.
    """
    bs = output.shape[0]
    boxes = output[:, :4, :]
    top_classes_conf = output[:, 4, :]
    scores = output[:, 4:-32, :]
    masks = output[:, -32:, :]
    results = []

    for b in range(bs):
        bboxes = boxes[b].T
        class_scores = scores[b].T
        box_masks = masks[b].T
        class_conf, class_ids = class_scores.max(1)
        if isinstance(conf_thresh, torch.Tensor):
            # class_ids are slice-indexed: 0 is objectness, k>=1 is class_{k-1}.
            # Keep objectness-dominated rows (no real class to threshold) and
            # filter class-dominated rows against the real class's threshold.
            thresh = conf_thresh.to(output.device)
            real_class = (class_ids - 1).clamp(min=0, max=thresh.shape[0] - 1)
            is_obj = class_ids == 0
            mask = is_obj | (top_classes_conf[b] > thresh[real_class])
        else:
            mask = top_classes_conf[b] > conf_thresh
        if mask.sum() == 0:
            results.append(torch.zeros((0, 38), device=output.device))
            continue
        bboxes = bboxes[mask]
        class_conf = class_conf[mask]
        class_ids = class_ids[mask]
        box_masks = box_masks[mask]
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
                box_masks[keep],
            ],
            dim=1,
        )  # [x1, y1, x2, y2, conf, cls]
        results.append(detections)
    return results
