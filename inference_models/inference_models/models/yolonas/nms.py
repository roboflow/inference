from typing import List, Union

import torch
import torchvision


def run_yolonas_nms_for_object_detection(
    output: torch.Tensor,
    conf_thresh: Union[float, torch.Tensor] = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
) -> List[torch.Tensor]:
    bs = output.shape[0]
    boxes = output[:, :, :4]
    scores = output[:, :, 4:]
    results = []
    per_class_thresh = (
        conf_thresh.to(output.device) if isinstance(conf_thresh, torch.Tensor) else None
    )
    for b in range(bs):
        class_scores = scores[b]  # (8400, cls_num)
        class_conf, class_ids = torch.max(class_scores, dim=-1)
        if per_class_thresh is not None:
            mask = class_conf > per_class_thresh[class_ids]
        else:
            mask = class_conf > conf_thresh
        if not torch.any(mask):
            results.append(torch.zeros((0, 6), device=output.device))
            continue
        bboxes = boxes[b][mask]
        class_conf = class_conf[mask]
        class_ids = class_ids[mask]
        nms_class_ids = torch.zeros_like(class_ids) if class_agnostic else class_ids
        keep = torchvision.ops.batched_nms(
            bboxes, class_conf, nms_class_ids, iou_thresh
        )
        if keep.numel() > max_detections:
            keep = keep[:max_detections]
        detections = torch.cat(
            (
                bboxes[keep],
                class_conf[keep, None],  # unsqueeze(1) is replaced with None
                class_ids[keep, None].float(),
            ),
            1,
        )  # [x1, y1, x2, y2, conf, cls]
        results.append(detections)
    return results
