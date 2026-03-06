from collections import namedtuple
from typing import Sequence
import random

import torch
import inspect
import random
from sklearn.metrics import (
    precision_recall_curve,
    auc as compute_auc,
    precision_recall_fscore_support,
)

from lidra.metrics.precision_recall import f1_beta_score
from lidra.model.module.base import Base, TrainableBackbone
from lidra.model.module.tdfy.util import (
    sample_uniform_box,
    grid_uniform_box,
    aug_xyz,
    binary_labels_from_point_distances,
)
from lidra.data.utils import tree_transpose_level_one
from loguru import logger

NoiseContrastiveSampleWithRGB = namedtuple(
    "NoiseContrastiveSampleWithRGB",
    ["input_xyz", "input_rgb", "query_xyz", "query_rgb", "query_occ"],
)


class OccupancyKLAutoencoder(Base):
    PREDICT_ARGS_SIGNATURE = inspect.signature(
        lambda tokenizer, generate_kwargs=None: None
    )

    def __init__(
        self,
        model: TrainableBackbone,
        box_size: float = 1.0,
        n_queries_pos: int = 550,
        n_queries_neg: int = 2000,
        n_input_points: Sequence[int] = (2048, 4096, 8192),
        granularity: float = 0.1,
        use_xyz_aug: bool = False,
        xyz_aug_random_scale_delta: float = 0.2,
        xyz_aug_origin_at_cam: bool = False,
        xyz_aug_random_rotate_degree: int = 180,
        xyz_aug_random_shift: float = 1.0,
        eval_xyz_distance_threshold: float = None,
        loss_weight_reconstruction: float = 1.0,
        loss_weight_kl: float = 0.0,
        loss_weight_color: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            model,
            **kwargs,
        )

        # Number of points for training
        self.n_input_points = n_input_points
        self.n_queries_pos = n_queries_pos
        self.n_queries_neg = n_queries_neg

        # Occupancy sampling
        self.box_size = box_size
        self.granularity = granularity
        self.eval_xyz_distance_threshold = eval_xyz_distance_threshold
        if self.eval_xyz_distance_threshold is None:
            self.eval_xyz_distance_threshold = self.granularity

        # XYZ Augmentation
        self.use_xyz_aug = use_xyz_aug
        self.xyz_aug_random_scale_delta = xyz_aug_random_scale_delta
        self.xyz_aug_origin_at_cam = xyz_aug_origin_at_cam
        self.xyz_aug_random_rotate_degree = xyz_aug_random_rotate_degree
        self.xyz_aug_random_shift = xyz_aug_random_shift

        # Losses
        self.occupancy_loss = torch.nn.BCEWithLogitsLoss()
        # self.color_loss = torch.nn.CrossEntropyLoss()
        self.loss_weight_reconstruction = loss_weight_reconstruction
        self.loss_weight_kl = loss_weight_kl
        self.loss_weight_color = loss_weight_color

        # buffer that stores validation predictions during validation epoch
        self.validation_step_outputs = []

        # buffer that stores last inputs
        self._save_last_inputs = False

    def _prepare_input(self, batch, is_train):
        B = batch.target.xyz.shape[0]
        num_input_points = self.n_input_points
        if isinstance(self.n_input_points, Sequence):
            num_input_points = random.choice(self.n_input_points)
            assert isinstance(
                num_input_points, int
            ), f"n_input_points must be an int or a sequence of ints, got {type(self.n_input_points)}"

        # gt
        gt_xyz = batch.target.xyz.reshape(B, -1, 3)
        gt_rgb = batch.target.rgb.reshape(B, -1, 3)
        device = gt_xyz.device

        # sample negatives from grid
        if is_train:
            noise_xyz = sample_uniform_box(
                B,
                self.n_queries_neg,
                self.box_size,
                device,
            )
        else:
            noise_xyz = grid_uniform_box(
                B,
                self.granularity,
                self.box_size,
                device,
            )

        # Augment XYZ
        if is_train and self.use_xyz_aug:
            gt_xyz, noise_xyz = aug_xyz(
                gt_xyz,
                noise_xyz,
                self.xyz_aug_random_scale_delta,
                self.xyz_aug_origin_at_cam,
                self.xyz_aug_random_rotate_degree,
                self.xyz_aug_random_shift,
                is_train=is_train,
            )

            # random flip
            if random.random() < 0.5:
                gt_xyz[..., 0] *= -1
                noise_xyz[..., 0] *= -1

        # Split gt_xyz and gt_rgb into n_queries_pos and n_input_points -- discarding the rest
        assert (
            self.n_queries_pos + num_input_points <= gt_xyz.shape[1]
        ), f"{self.n_queries_pos=} + {num_input_points=} > {gt_xyz.shape[1]=}"
        query_pos_xyz = gt_xyz[:, : self.n_queries_pos]
        input_xyz = gt_xyz[
            :, self.n_queries_pos : self.n_queries_pos + num_input_points
        ]

        # Combine positive and negative samples for queries
        query_xyz = self._merge_queries_pos_neg(query_pos_xyz, noise_xyz)
        query_occ = self._merge_queries_pos_neg(
            torch.ones(B, query_pos_xyz.shape[1], device=device),
            torch.zeros(B, noise_xyz.shape[1], device=device),
        )

        # RGB
        if gt_rgb is not None:
            assert self.n_queries_pos + self.n_queries_neg <= gt_rgb.shape[1]
            input_rgb = gt_rgb[
                :, self.n_queries_pos : self.n_queries_pos + num_input_points
            ]
            query_pos_rgb = gt_rgb[:, : self.n_queries_pos]
            noise_rgb = torch.ones_like(noise_xyz) / 2.0
            query_rgb = self._merge_queries_pos_neg(query_pos_rgb, noise_rgb)
        else:
            query_pos_rgb = None
            input_rgb = None

        return NoiseContrastiveSampleWithRGB(
            input_xyz=input_xyz,
            input_rgb=input_rgb,
            query_xyz=query_xyz,
            query_rgb=query_rgb,
            query_occ=query_occ,
        )

    def _split_queries_pos_neg(
        self, queries: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split queries into positive and negative samples.

        Args:
            queries: Input tensor of shape (B, N, ...)

        Returns:
            Tuple of (positive_queries, negative_queries) tensors
        """
        query_pos = queries[:, : self.n_queries_pos]
        query_neg = queries[:, self.n_queries_pos :]
        return query_pos, query_neg

    def _merge_queries_pos_neg(
        self, query_pos: torch.Tensor | None, query_neg: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Merge positive and negative query tensors along points dimension.

        Args:
            query_pos: Positive query tensor of shape (B, n_queries_pos, ...)
            query_neg: Negative query tensor of shape (B, n_queries_neg, ...)

        Returns:
            Merged tensor of shape (B, n_queries_pos + n_queries_neg, ...)
        """
        return torch.cat([query_pos, query_neg], dim=1)

    def _compute_loss(
        self,
        occ_logits,
        occ_gt,
        color_logits,
        color_gt,
        kl_loss,
    ):
        losses = {}

        # compute occupancy loss
        B = occ_logits.shape[0]
        losses["loss.reconstruction"] = self.occupancy_loss(occ_logits, occ_gt).mean()

        if kl_loss is not None:
            losses["loss.kl"] = kl_loss.mean()
        else:
            losses["loss.kl"] = 0.0

        loss = (
            self.loss_weight_reconstruction * losses["loss.reconstruction"]
            + self.loss_weight_kl * losses["loss.kl"]
        )

        # compute color loss if present
        if (color_logits is not None) and (occ_gt.sum() > 0):
            color_logits = color_logits[occ_gt.bool()].reshape((-1, 256))
            gt_rgb = torch.round(color_gt[occ_gt.bool()] * 255).long().reshape((-1,))

            losses["loss.color"] = self.color_loss(color_logits, gt_rgb).mean()
            loss += losses["loss.color"] * self.loss_weight_color

        return {"loss": loss, **losses}

    def _step(self, label, batch, batch_idx):
        # skip if batch happens to be empty
        if batch is None:
            return None

        ########### Can remove once we have more general collate fn ###########
        # TODO : Clean / Formalize
        ObjaverseSample = namedtuple(
            "ObjaverseSample",
            ["seen", "target", "extra", "metadata"],
        )
        SeenData = namedtuple("SeenData", ["xyz", "rgb", "mask"])
        TargetData = namedtuple("TargetData", ["xyz", "rgb", "normal"])

        batch = ObjaverseSample(
            seen=SeenData(batch[0][0], batch[0][1], batch[0][2]),
            target=TargetData(batch[1][0], batch[1][1], batch[1][2]),
            extra=None,
            metadata=batch[3],
        )
        ###########

        sample: NoiseContrastiveSampleWithRGB = self._prepare_input(
            batch, is_train=(label == "train")
        )

        if self._base_models is None:
            logger.error(
                f"self.base_model might be None -- did you forget to call self.configure_model()?"
            )
        output = self.base_model(sample.input_xyz, sample.query_xyz)

        occ_logits = output.get("logits", None)
        kl_loss = output.get("kl", None)
        color = output.get("color", None)

        # compute losses
        losses = self._compute_loss(
            occ_logits,
            sample.query_occ,
            color,
            sample.query_rgb,
            kl_loss,
        )

        # log losses
        for name, value in losses.items():
            self.log(
                name=f"{label}/{name}",
                value=value,
                prog_bar=(name == "loss"),
                batch_size=occ_logits.shape[0],
                sync_dist=(label == "val"),
            )

        if self._save_last_inputs:
            self.last_inputs = {
                "occ_logits": occ_logits,
                "query_xyz": sample.query_xyz,
                "query_occ": sample.query_occ,
                "gt_xyz": batch.target.xyz,
            }
        return (
            losses["loss"],
            occ_logits,
            sample.query_xyz,
            sample.query_occ,
            batch.target.xyz,
        )

    def training_step(self, batch, batch_idx):
        loss, *_ = self._step("train", batch, batch_idx)
        loss = self._aggregate_loss(loss, batch_size=batch[0][0].shape[0])
        return loss

    def _aggregate_loss(self, loss, batch_size):
        n_items = self.all_gather(batch_size).sum()
        return loss * n_items

    def validation_step(self, batch, batch_idx):
        loss, query_pred_occ, query_xyz, query_occ, gt_xyz = self._step(
            "val", batch, batch_idx
        )
        query_pos_xyz, query_neg_xyz = self._split_queries_pos_neg(query_xyz)
        query_pos_occ, query_neg_occ = self._split_queries_pos_neg(query_occ)
        query_pred_pos_occ, query_pred_neg_occ = self._split_queries_pos_neg(
            query_pred_occ
        )

        # accumulate results for eval computation later
        for bidx in range(query_pred_occ.shape[0]):
            # alternatively, we could use GT samples as inputs, and points sufficiently far away as negatives
            # this could be a more direct measure of how well the model can predict occupancy
            # since that way we avoid declaring points to be "occupied" using the distance threshold

            # On the other hand, we want to avoid breaking the grid structure that we might use for marching cubes

            # Subselect only points that are gt a certain threshold
            close_to_gt = binary_labels_from_point_distances(
                query_neg_xyz[bidx],
                gt_xyz[bidx],
                dist_thres=self.eval_xyz_distance_threshold,
            )

            easy_negs_gt = query_neg_occ[bidx].clone()
            easy_negs_gt[close_to_gt.bool()] = 1.0
            easy_negs_logits = query_pred_neg_occ[bidx]
            easy_negs_xyz = query_neg_xyz[bidx]

            occ_gt = torch.cat([query_pos_occ[bidx], easy_negs_gt], dim=0)
            preds = torch.cat([query_pred_pos_occ[bidx], easy_negs_logits], dim=0)
            query_xyz = torch.cat([query_pos_xyz[bidx], easy_negs_xyz], dim=0)
            self.validation_query_xyz = query_xyz
            self.validation_step_outputs.append((preds, occ_gt))

        return loss

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            raise ValueError(
                "validation_step_outputs is empty in on_validation_epoch_end"
            )

        # get preditions and labels
        all_occ_pred, all_occ_gt = tree_transpose_level_one(
            self.validation_step_outputs,
            map_fn=torch.stack,
        )

        # convert logits to probability
        all_occ_pred = torch.nn.functional.sigmoid(all_occ_pred)

        optimal_threshold = self._compute_optimal_threshold(all_occ_pred, all_occ_gt)
        self._compute_optimal_metrics(all_occ_pred, all_occ_gt, optimal_threshold)

        # find_optimal_thresholds()
        self.validation_step_outputs.clear()  # free memory

    def _compute_optimal_metrics(self, all_pred, all_gt, optimal_threshold):

        scores = []
        for pred, gt in zip(all_pred, all_gt):
            pred = pred > optimal_threshold
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=gt.cpu(),
                y_pred=pred.cpu(),
                average="binary",
                zero_division=0,
            )
            scores.append((precision, recall, f1))

        scores = self.all_gather(scores)
        precisions, recalls, f1s = tree_transpose_level_one(scores, map_fn=torch.stack)

        logger.info(f"local precisions: {precisions.mean().item()}")
        logger.info(f"local recalls: {recalls.mean().item()}")
        logger.info(f"local f1s: {f1s.mean().item()}")

        self.log(f"metrics/optimal.precision", precisions.mean().item(), sync_dist=True)
        self.log(f"metrics/optimal.recall", recalls.mean().item(), sync_dist=True)
        self.log(f"metrics/optimal.f1", f1s.mean().item(), sync_dist=True)

    def _compute_optimal_threshold(self, all_pred, all_gt):
        optimal_thresholds = []
        aucs = []

        # compute precision, recall and f1 score per item
        for pred, gt in zip(all_pred, all_gt):
            precision, recall, thresholds = precision_recall_curve(gt.cpu(), pred.cpu())

            f1, not_nan_mask = f1_beta_score(precision, recall, remove_nans=True)
            thresholds = thresholds[not_nan_mask]

            optimal_thresholds.append(thresholds[f1.argmax()].item())
            aucs.append(compute_auc(recall, precision))

        # gather item metrics
        optimal_thresholds = self.all_gather(optimal_thresholds)
        aucs = self.all_gather(aucs)

        optimal_thresholds = torch.cat(optimal_thresholds)
        aucs = torch.cat(aucs)

        # compute relevant stats
        auc = aucs.mean().item()
        optimal_threshold = optimal_thresholds.median().item()
        optimal_threshold_std = optimal_thresholds.std().item()

        self.log(f"metrics/auc", auc, sync_dist=True)
        self.log(f"metrics/optimal.threshold", optimal_threshold, sync_dist=True)
        self.log(
            f"metrics/optimal.threshold.std",
            optimal_threshold_std,
            sync_dist=True,
        )

        return optimal_threshold

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError
