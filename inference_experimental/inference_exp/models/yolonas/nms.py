from typing import List

import torch
import torchvision


def run_yolonas_nms_for_object_detection(
    output: torch.Tensor,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
) -> List[torch.Tensor]:
    bs = output.shape[0]
    boxes = output[:, :, :4]
    scores = output[:, :, 4]
    results = []
    for b in range(bs):
        # Combine transpose & max for efficiency
        class_scores = scores[b]  # (80, 8400)
        mask = class_scores > conf_thresh
        if not torch.any(mask):
            results.append(torch.zeros((0, 6), device=output.device))
            continue
        bboxes = boxes[b][mask]
        class_conf = class_scores[mask]
        # THIS IS PROBABLY BUG - Could not export class id from YOLONAS model
        class_ids = torch.zeros(size=(class_conf.shape[0],), device=output.device)
        keep = torchvision.ops.batched_nms(bboxes, class_conf, class_ids, iou_thresh)
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
