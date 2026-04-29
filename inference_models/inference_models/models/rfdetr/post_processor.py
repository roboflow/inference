from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w.clamp(min=0.0)),
        (y_c - 0.5 * h.clamp(min=0.0)),
        (x_c + 0.5 * w.clamp(min=0.0)),
        (y_c + 0.5 * h.clamp(min=0.0)),
    ]
    return torch.stack(b, dim=-1)


def select_topk_predictions(
    logits_sigmoid: torch.Tensor,
    bboxes_cxcywh: torch.Tensor,
    num_select: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Topk-flat across (queries x classes) for one image.

    Matches rf-detr-internal training-time `PostProcess` and SAB benchmarks:
    flatten the (Q, C) sigmoid score matrix, take the top `num_select` entries,
    derive the (query, class) index per pick, and gather the corresponding
    cxcywh bbox per pick. A single query can contribute multiple detections
    if more than one of its class scores ranks in the global topk.

    Args:
        logits_sigmoid: per-image sigmoid scores, shape (Q, C).
        bboxes_cxcywh: per-image normalized cxcywh boxes, shape (Q, 4).
        num_select: max number of (query, class) pairs to keep. Capped at Q*C.

    Returns:
        scores: top-K scores in descending order, shape (K,).
        top_classes: class index for each pick, shape (K,).
        gathered_bboxes_cxcywh: bboxes gathered by query index, shape (K, 4).
        query_indices: query each pick came from, shape (K,). Useful for
            downstream gather of per-query auxiliaries (e.g. masks).
    """
    num_classes = logits_sigmoid.shape[1]
    flat_scores = logits_sigmoid.reshape(-1)
    num_to_select = min(num_select, flat_scores.shape[0])
    scores, topk_indexes = torch.topk(flat_scores, num_to_select)
    query_indices = topk_indexes // num_classes
    top_classes = topk_indexes % num_classes
    gathered_bboxes = bboxes_cxcywh[query_indices]
    return scores, top_classes, gathered_bboxes, query_indices


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=300) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        out_masks = outputs.get("pred_masks", None)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        results = []
        for i in range(prob.shape[0]):
            scores, labels, gathered_bboxes_cxcywh, query_indices = (
                select_topk_predictions(
                    logits_sigmoid=prob[i],
                    bboxes_cxcywh=out_bbox[i],
                    num_select=self.num_select,
                )
            )
            boxes_xyxy = box_cxcywh_to_xyxy(gathered_bboxes_cxcywh)
            img_h, img_w = target_sizes[i].unbind(0)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h])
            boxes_xyxy = boxes_xyxy * scale_fct
            res_i = {"scores": scores, "labels": labels, "boxes": boxes_xyxy}
            if out_masks is not None:
                masks_i = out_masks[i][query_indices]  # [K, Hm, Wm]
                h, w = target_sizes[i].tolist()
                masks_i = F.interpolate(
                    masks_i.unsqueeze(1),
                    size=(int(h), int(w)),
                    mode="bilinear",
                    align_corners=False,
                )  # [K, 1, H, W]
                res_i["masks"] = masks_i > 0.0
            results.append(res_i)

        return results
