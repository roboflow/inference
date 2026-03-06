import torch

from pytorch3d.loss import chamfer_distance
from torch import Tensor
from torchmetrics import Metric


class ChamferDistance(Metric):
    def __init__(self, norm: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.norm = norm

        self.add_state(
            "chamfer",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_samples",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(self, pred_xyz: Tensor, gt_xyz: Tensor) -> None:
        """
        Inputs:
            pred_xyz (BxP1x3): Set of predicted XYZ points (float tensor or Pointclouds)
            gt_xyz (BxP2x3): Set of GT XYZ points (float tensor or Pointclouds)
        """
        chamfer, _ = chamfer_distance(
            pred_xyz, gt_xyz, batch_reduction="sum", norm=self.norm
        )

        self.chamfer += chamfer
        self.total_samples += gt_xyz.shape[0]

    def compute(self) -> Tensor:
        return self.chamfer.float() / self.total_samples
