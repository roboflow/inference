# ------------------------------------------------------------------------
# Ported from RF-DETR (https://github.com/roboflow/rf-detr)
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Based on Conditional DETR, Deformable DETR, DETR
# ------------------------------------------------------------------------

"""Hungarian matching for DETR-style bipartite assignment."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from inference_models.models.rfdetr.few_shot.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)


class HungarianMatcher(nn.Module):
    """Computes an assignment between targets and predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this,
    in general, there are more predictions than targets. In this case, we do a 1-to-1
    matching of the best predictions, while the others are un-matched (and thus treated
    as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha: float = 0.25,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"
        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        flat_pred_logits = outputs["pred_logits"].flatten(0, 1)
        out_prob = flat_pred_logits.sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # GIoU cost
        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )
        cost_giou = -giou

        # Focal classification cost
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-F.logsigmoid(-flat_pred_logits))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-F.logsigmoid(flat_pred_logits))
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).float().cpu()

        # Replace NaN/Inf with large values
        max_cost = C.max() if C.numel() > 0 else 0
        C[C.isinf() | C.isnan()] = max_cost * 2

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        g_num_queries = num_queries // group_detr
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(group_detr):
            C_g = C_list[g_i]
            indices_g = [
                linear_sum_assignment(c[i])
                for i, c in enumerate(C_g.split(sizes, -1))
            ]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (
                        np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]),
                        np.concatenate([indice1[1], indice2[1]]),
                    )
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
