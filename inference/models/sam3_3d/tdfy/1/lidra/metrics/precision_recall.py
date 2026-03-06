import torch
import warnings

from loguru import logger
from sklearn.metrics import (
    precision_recall_curve,
    auc as compute_auc,
    precision_recall_fscore_support,
)
from torch import Tensor
from torchmetrics import Metric
from dataclasses import dataclass
from lidra.model.module.tdfy.util import torch_or_numpy


def f1_beta_score(precision, recall, beta=1.0, remove_nans=True):
    num = (1 + beta**2) * (precision * recall)
    den = (beta**2) * precision + recall
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in divide")
        f1 = num / den
    f1 = f1[1:]
    if remove_nans:
        not_nan_mask = ~torch_or_numpy(f1).isnan(f1)
        f1 = f1[not_nan_mask]
        return f1, not_nan_mask
    return f1


@dataclass
class PrecisionRecallMetrics:
    precision: float
    recall: float
    auc: float
    f1: float


class PrecisionRecall(Metric):
    def __init__(self, is_train=False, **kwargs):
        super().__init__(**kwargs)

        self.is_train = is_train

        self.add_state(
            "precision",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "recall",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "f1",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "auc",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "optimal_threshold",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_samples",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        pred_occ: Tensor,
        gt_occ: Tensor,
        precomputed_optimal_threshold=None,
    ) -> None:
        """
        Inputs:
            pred_occ (BxNx3): Set of predicted occupancies (float tensor)
            gt_occ (BxNx3): Set of GT occupancies (float tensor)

        WARNING: Metric has different behavior based on is_train flag.
            is_train=True: Optimal threshold is sample dependent and calculated based
                on precision/recall of the sample
            is_train=False: Optimal threshold is constant; average optimal threshold
                is precomputed from training and used for optimal precision/recall/f1
        """
        for pred, gt in zip(pred_occ, gt_occ):
            pred, gt = pred.cpu(), gt.cpu()
            precision, recall, thresholds = precision_recall_curve(gt, pred)
            auc = compute_auc(recall, precision)

            # Optimal threshold learned during training, fixed during validation
            if self.is_train:
                f1, not_nan_mask = f1_beta_score(precision, recall, remove_nans=True)
                thresholds = thresholds[not_nan_mask]

                if len(f1) == 0:
                    logger.warning(
                        "f1 is empty. Could be values are all NaN (prec/rec all 0)"
                    )
                    continue

                optimal_threshold = thresholds[f1.argmax()].item()
                self.optimal_threshold += torch.tensor(
                    optimal_threshold, device=self.device
                )
            else:
                # Use precomputed optimal threshold to avoid unnecessary sync
                assert (
                    precomputed_optimal_threshold is not None
                ), "Must provide precomputed_optimal_threshold during evaluation"
                optimal_threshold = precomputed_optimal_threshold

            # Compute precision/recall at optimal threshold
            pred = pred > optimal_threshold
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=gt,
                y_pred=pred,
                average="binary",
                zero_division=0,
            )

            self.auc += torch.tensor(auc, device=self.device)
            self.precision += torch.tensor(precision, device=self.device)
            self.recall += torch.tensor(recall, device=self.device)
            self.f1 += torch.tensor(f1, device=self.device)
            self.total_samples += 1

    def compute_optimal_threshold(self):
        with self.sync_context():
            return self.optimal_threshold / self.total_samples

    def compute(self) -> Tensor:
        precision = self.precision.float() / self.total_samples
        recall = self.recall.float() / self.total_samples
        auc = self.auc.float() / self.total_samples
        f1 = self.f1.float() / self.total_samples
        return PrecisionRecallMetrics(precision, recall, auc, f1)
