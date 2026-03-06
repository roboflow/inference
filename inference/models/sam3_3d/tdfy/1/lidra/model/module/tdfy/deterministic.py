import torch
import random
from pytorch3d.structures.pointclouds import Pointclouds

from lidra.metrics.chamfer import ChamferDistance
from lidra.metrics.precision_recall import PrecisionRecall
from lidra.model.module.base import Base, TrainableBackbone
from lidra.model.module.tdfy.util import (
    construct_uniform_cube,
    aug_xyz,
    shrink_points_beyond_threshold,
    binary_labels_from_point_distances,
)


class Deterministic(Base):
    def __init__(
        self,
        model: TrainableBackbone,
        sampling_world_size: float = 3.0,
        n_queries: int = 550,
        train_dist_threshold: float = 0.1,
        granularity: float = 0.1,
        obj_center_norm: bool = True,
        random_scale_delta: float = 0.2,
        origin_at_cam: bool = False,
        random_rotate_degree: int = 180,
        random_shift: float = 1.0,
        shrink_threshold: float = 10.0,
        color_loss_weight: float = 0,
        eval_xyz_distance_threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            model,
            **kwargs,
        )

        assert False, """As of now (after PR https://github.com/fairinternal/lidra/pull/58), the MCC training pipeline is currently broken (poor F1 performance).
Scale change (2 in objaverse vs 6 in training) might be a prime cause for this and will need to be investigated further if need be."""

        self.sampling_world_size = sampling_world_size
        self.n_queries = n_queries
        self.train_dist_threshold = train_dist_threshold
        self.eval_xyz_distance_threshold = eval_xyz_distance_threshold
        self.granularity = granularity
        self.obj_center_norm = obj_center_norm

        self.random_scale_delta = random_scale_delta
        self.origin_at_cam = origin_at_cam
        self.random_rotate_degree = random_rotate_degree
        self.random_shift = random_shift

        self.shrink_threshold = shrink_threshold

        self.occupancy_loss = torch.nn.BCEWithLogitsLoss()
        self.color_loss = torch.nn.CrossEntropyLoss()
        self.color_loss_weights = color_loss_weight

        # Metrics
        self.opt_thresh = 0.0
        self.train_pr_metric = PrecisionRecall(is_train=True)
        self.val_pr_metric = PrecisionRecall(is_train=False)
        self.chamfer_metric = ChamferDistance()

    def _prepare_input(self, batch, is_train):
        # extract depth, image, mask and normals
        seen_xyz = batch[0][0]
        seen_rgb = batch[0][1]
        mask = batch[0][2] if len(batch[0]) > 2 else None

        B = seen_rgb.shape[0]

        # gt
        gt_xyz = batch[1][0].reshape(B, -1, 3)
        gt_rgb = batch[1][1].reshape(B, -1, 3)

        # set non-finite depth
        valid_seen_xyz = torch.isfinite(seen_xyz.sum(axis=-1))
        seen_xyz[~valid_seen_xyz] = -100

        unseen_xyz, unseen_rgb, labels = construct_uniform_cube(
            gt_xyz,
            gt_rgb,
            self.sampling_world_size,
            self.n_queries,
            self.train_dist_threshold,
            is_train,
            self.granularity,
        )

        if is_train and (not self.obj_center_norm):
            seen_xyz, unseen_xyz = aug_xyz(
                seen_xyz,
                unseen_xyz,
                self.random_scale_delta,
                self.origin_at_cam,
                self.random_rotate_degree,
                self.random_shift,
                is_train=is_train,
            )

            # random flip
            if random.random() < 0.5:
                seen_xyz[..., 0] *= -1
                unseen_xyz[..., 0] *= -1
                seen_xyz = torch.flip(seen_xyz, [2])
                valid_seen_xyz = torch.flip(valid_seen_xyz, [2])
                seen_rgb = torch.flip(seen_rgb, [3])

        unseen_xyz = shrink_points_beyond_threshold(unseen_xyz, self.shrink_threshold)

        return (
            seen_xyz,
            valid_seen_xyz,
            unseen_xyz,
            unseen_rgb,
            labels,
            seen_rgb,
            mask,
            gt_xyz,
            gt_rgb,
        )

    def _compute_occ_gt(self, occ, xyz_queries, xyz_gt):
        occ_gts = []
        for bidx in range(occ.shape[0]):
            occ_gts.append(
                binary_labels_from_point_distances(
                    xyz_queries[bidx],
                    xyz_gt[bidx],
                    dist_thres=self.eval_xyz_distance_threshold,
                )
            )
        return torch.stack(occ_gts)

    def _pred_xyz_to_pc(self, occ_pred, xyz_queries, threshold):
        pred_xyz_list = []
        for bidx in range(occ_pred.shape[0]):
            thresh_mask = occ_pred[bidx] > threshold
            pred_xyz_list.append(xyz_queries[bidx][thresh_mask])

        return Pointclouds(pred_xyz_list)

    def _compute_loss(
        self,
        occ_logits,
        occ_gt,
        color_logits,
        color_gt,
    ):
        losses = {}

        # compute occupancy loss
        loss = losses["loss.occupancy"] = self.occupancy_loss(occ_logits, occ_gt)

        # compute color loss if present
        if (color_logits is not None) and (occ_gt.sum() > 0):
            color_logits = color_logits[occ_gt.bool()].reshape((-1, 256))
            gt_rgb = torch.round(color_gt[occ_gt.bool()] * 255).long().reshape((-1,))

            losses["loss.color"] = self.color_loss(color_logits, gt_rgb)
            loss += losses["loss.color"] * self.color_loss_weights

        return {"loss": loss, **losses}

    def _step(self, label, batch, batch_idx):
        # skip if batch happens to be empty
        if batch is None:
            return None

        (
            _,
            _,
            xyz_queries,
            color_gt,
            occ_gt,
            image,
            _,
            gt_xyz,
            _,
        ) = self._prepare_input(batch, is_train=(label == "train"))

        # compute predictions
        output = self.base_model(image, xyz_queries)
        if isinstance(output, tuple) and len(output) == 2:  # color prediction
            occ, color = output
        else:
            occ = output
            color = None

        # compute losses
        losses = self._compute_loss(occ, occ_gt, color, color_gt)

        # log losses
        for name, value in losses.items():
            self.log(
                name=f"{label}/{name}",
                value=value,
                prog_bar=(name == "loss"),
                batch_size=occ.shape[0],
                sync_dist=(label == "val"),
            )

        return losses["loss"], occ, xyz_queries, gt_xyz

    def _make_effective_loss(self, loss, batch_size):
        n_items = self.all_gather(batch_size).sum()
        return loss * n_items / 512

    def training_step(self, batch, batch_idx):
        loss, occ_logits, xyz_queries, xyz_gt = self._step("train", batch, batch_idx)
        loss = self._make_effective_loss(loss, batch_size=batch[0][0].shape[0])

        # convert logits to probability
        occ = torch.nn.functional.sigmoid(occ_logits)

        # F1 eval computation
        occ_gt = self._compute_occ_gt(occ, xyz_queries, xyz_gt)
        self.train_pr_metric.update(occ, occ_gt)

        return loss

    def on_train_epoch_end(self):
        self.opt_thresh = self.train_pr_metric.compute_optimal_threshold().cpu()
        self.log(f"metrics/optimal.threshold", self.opt_thresh)

        # Reset to clear training
        self.train_pr_metric.reset()

    def validation_step(self, batch, batch_idx):
        loss, occ_logits, xyz_queries, xyz_gt = self._step("val", batch, batch_idx)

        # convert logits to probability
        occ = torch.nn.functional.sigmoid(occ_logits)

        # F1 eval computation
        occ_gt = self._compute_occ_gt(occ, xyz_queries, xyz_gt)
        self.val_pr_metric.update(occ, occ_gt, self.opt_thresh)

        # Chamfer eval computation
        pred_xyz_pc = self._pred_xyz_to_pc(occ, xyz_queries, self.opt_thresh)
        self.chamfer_metric.update(pred_xyz_pc, xyz_gt)

        return loss

    def on_validation_epoch_end(self):
        precision, recall, auc, f1 = self.val_pr_metric.compute()
        self.log(f"metrics/optimal.precision", precision)
        self.log(f"metrics/optimal.recall", recall)
        self.log(f"metrics/auc", auc)
        self.log(f"metrics/optimal.f1", f1)
        self.log(f"metrics/optimal.chamfer", self.chamfer_metric.compute())

        # Reset metrics
        self.opt_thresh = 0.0
        self.val_pr_metric.reset()
        self.chamfer_metric.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if batch is None:
            return None

        (
            _,
            _,
            xyz_queries,
            _,
            _,
            image,
            _,
            _,
            _,
        ) = self._prepare_input(batch, is_train=False)

        # compute predictions
        output = self.base_model(image, xyz_queries)
        if isinstance(output, tuple) and len(output) == 2:  # color prediction
            occ, color = output
        else:
            occ = output
            color = None

        return occ, color
