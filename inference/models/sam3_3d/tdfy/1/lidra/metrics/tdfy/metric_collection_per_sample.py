from loguru import logger
from typing import List, Optional, Dict, Any, Tuple

from sklearn.metrics import precision_recall_fscore_support

import torch
from pytorch3d.ops import iterative_closest_point
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import quaternion_to_matrix


from lidra.metrics.metric_collection_for_sample import PerSample
from lidra.metrics.tdfy.pose.rotation import (
    rotation_angle,
    generate_90_deg_rotations,
)
from lidra.metrics.tdfy.occupancy.pointcloud import (
    create_occupancy_volume,
    chamfer_distance_icp_aligned,
    occupancy_grid_to_local_points,
    normalize_pcd,
)
from lidra.metrics.tdfy.distance import (
    abs_relative_error,
    abs_log_rel,
    delta1_acc,
    delta2_acc,
    delta3_acc,
)
from loguru import logger
from lidra.data.dataset.tdfy.pose_target import (
    PoseTargetConverter,
    PoseTarget,
    InvariantPoseTarget,
    InstancePose,
)
from dataclasses import asdict


class TdfyPerSample(PerSample):
    """
    Per-sample metrics for TDFY.
    """

    SHAPE_KEY = "occupancy_volume"  # N x N x N

    # Fun fact: two float32 tensors can be multiplied together to yield a bfloat16 tensor
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    @staticmethod
    def evaluate(
        prediction: Dict[str, Any],
        target: Dict[str, Any],
        rotation_candidates: Optional[List[torch.Tensor]] = None,
        occupancy_volume_resolution: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate per-sample metrics for TDFY predictions.

        Args:
            prediction (Dict[str, Any]): Dictionary containing model predictions with keys like target
            target (Dict[str, Any]): Dictionary containing ground truth with keys:
                - SHAPE_KEY: N x N x N occupancy volume
                - ROTATION_KEY: quat: 1 x 4
                - TRANSLATION_KEY: 1 x 3
                - SCALE_KEY: 1 x 1
            rotation_candidates (Optional[List[torch.Tensor]]): List of candidate rotation matrices (3x3) to try.
                If None, uses 90-degree rotations around each axis.
            occupancy_volume_resolution (int, optional): Resolution of occupancy volume grid. Defaults to 64.

        Returns:
            Dict[str, Any]: Dictionary containing computed metrics:
        """

        pred_points, gt_points = TdfyPerSample.get_object_points(prediction, target)

        if len(pred_points) == 0 or len(gt_points) == 0:
            raise NanFoundInMetricException("pred_points or gt_points is empty")

        # Align prediction with GT: use 90-degree rotations
        pred_points_aligned, min_loss_cd, R_cand = (
            TdfyPerSample.get_min_chamfer_distance_alignment(
                pred_points, gt_points, rotation_candidates
            )
        )

        shape_metrics = TdfyPerSample.compute_shape_metrics(
            pred_points_aligned,
            gt_points,
            min_loss_cd=min_loss_cd,
            volume_gt=target.get(TdfyPerSample.SHAPE_KEY, None),
            occupancy_volume_resolution=occupancy_volume_resolution,
        )

        # Pose estimation error: compute standardized pose target convention
        pose_pred_std, pose_gt_std = TdfyPerSample.invariant_pose_targets(
            prediction, target
        )

        oriented_shape_metrics = TdfyPerSample.compute_oriented_shape_metrics(
            pred_points,
            gt_points,
            quat_pred=pose_pred_std.q if pose_pred_std is not None else None,
            quat_gt=pose_gt_std.q if pose_gt_std is not None else None,
            volume_gt=target.get(TdfyPerSample.SHAPE_KEY, None),
            occupancy_volume_resolution=occupancy_volume_resolution,
        )

        # Max rotation error = 77 degrees
        rotation_metrics = TdfyPerSample.compute_rotation_error(
            quat_pred=pose_pred_std.q if pose_pred_std is not None else None,
            quat_gt=pose_gt_std.q if pose_gt_std is not None else None,
            rotation_candidates=rotation_candidates,  # R_cand.expand(1, 3, 3),
        )

        scale_metrics = TdfyPerSample.compute_scale_error(
            pose_pred=pose_pred_std,
            pose_gt=pose_gt_std,
        )

        translation_metrics = TdfyPerSample.compute_translation_error(
            pose_pred=pose_pred_std,
            pose_gt=pose_gt_std,
        )

        return {
            **shape_metrics,
            **rotation_metrics,
            **scale_metrics,
            **translation_metrics,
            **oriented_shape_metrics,
        }

    @staticmethod
    def invariant_pose_targets(
        prediction: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Tuple[InstancePose, InstancePose]:
        pose_pred, pose_gt = PoseUtils.get_pred_gt_instance_pose(prediction, target)

        if pose_pred is None or pose_gt is None:
            return None, None
        pose_pred_std = InvariantPoseTarget.from_instance_pose(pose_pred)
        pose_gt_std = InvariantPoseTarget.from_instance_pose(pose_gt)
        return pose_pred_std, pose_gt_std

    @staticmethod
    def get_object_points(
        prediction: Dict[str, Any],
        target: Dict[str, Any],
        logit_threshold: float = 0.0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract point clouds from prediction and target dictionaries.
        Handles both direct point clouds and occupancy volumes, converting the latter to points.
        """
        if "occupancy_volume" in prediction and "occupancy_volume" in target:
            assert (
                prediction["occupancy_volume"].shape == target["occupancy_volume"].shape
            ), f"Volume shapes must match {prediction['occupancy_volume'].shape=} {target['occupancy_volume'].shape=}"

        pred_points = TdfyPerSample._get_points(prediction, logit_threshold)
        gt_points = TdfyPerSample._get_points(target, logit_threshold)
        return pred_points, gt_points

    @staticmethod
    def _get_points(
        data: Dict[str, torch.Tensor],
        logit_threshold: float = 0.0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        if "points" in data:
            return data["points"]
        elif not "occupancy_volume" in data:
            raise ValueError("Data must contain either 'occupancy_volume' or 'points'")

        volume = data["occupancy_volume"]
        n_voxels = volume.shape[-1]
        # Ensure 5D volume for consistency
        if volume.ndim == 4:
            volume = volume.unsqueeze(0)
        points = occupancy_grid_to_local_points(
            volume,
            threshold=logit_threshold,
        )
        return points

    @staticmethod
    def _default_rotation_candidates() -> torch.Tensor:
        return torch.stack([R for R in generate_90_deg_rotations()], dim=0)  #

    @staticmethod
    def get_min_chamfer_distance_alignment(
        pc_pred: torch.Tensor,
        pc_gt: torch.Tensor,
        rotation_candidates: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Get the minimum Chamfer distance alignment and the corresponding point cloud and rotation matrix.
        """
        if rotation_candidates is None:
            rotation_candidates = TdfyPerSample._default_rotation_candidates()
        rotation_candidates = rotation_candidates.to(pc_pred.device)

        # Chamfer distance
        min_loss_cd = float("inf")
        pc_pred_min_loss_cd = None
        for R_cand in rotation_candidates:
            pc_pred_cand = apply_rotation(pc_pred, R_cand)
            loss_cd, icp_solution = chamfer_distance_icp_aligned(
                pc_pred_cand, pc_gt, icp_align=True
            )
            if loss_cd < min_loss_cd:
                min_loss_cd = loss_cd
                pc_pred_min_loss_cd = icp_solution.Xt
        return pc_pred_min_loss_cd, min_loss_cd, R_cand

    @staticmethod
    def compute_shape_metrics(
        pred_points_aligned: torch.Tensor,
        gt_points: torch.Tensor,
        min_loss_cd: torch.Tensor,
        volume_gt: Optional[torch.Tensor] = None,
        occupancy_volume_resolution: Optional[int] = None,
    ) -> Dict[str, Any]:
        shape_metrics = {}
        if pred_points_aligned is None or gt_points is None:
            return {}
        # If user wants a specific resolution, don't reuse the volume_gt
        volume_gt = volume_gt if occupancy_volume_resolution is None else None
        shape_occupancy_metrics = TdfyPerSample.compute_occupancy_metrics(
            pred_points=pred_points_aligned,
            gt_points=gt_points,
            volume_gt=volume_gt,
            occupancy_volume_resolution=occupancy_volume_resolution,
        )
        chamfer_distance_metrics = TdfyPerSample.compute_chamfer_distance_metrics(
            min_loss_cd=min_loss_cd,
        )
        shape_metrics = {
            **chamfer_distance_metrics,
            **shape_occupancy_metrics,
        }
        return shape_metrics

    @staticmethod
    def compute_chamfer_distance_metrics(
        min_loss_cd: float, eps: float = 1e-10
    ) -> Tuple[float, float]:
        chamfer_distance_metrics = {
            "chamfer_distance": min_loss_cd.cpu().item(),
            "log10_chamfer_distance": torch.log10(min_loss_cd + eps).cpu().item(),
        }
        return chamfer_distance_metrics

    @staticmethod
    def compute_occupancy_metrics(
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        volume_gt: Optional[torch.Tensor] = None,
        occupancy_volume_resolution: Optional[int] = None,
        eps: float = 1e-6,
    ) -> Tuple[float, float, float]:

        if occupancy_volume_resolution is None:
            assert (
                volume_gt is not None
            ), "Must provide either occupancy_volume_resolution or volume_gt"
            occupancy_volume_resolution = volume_gt.shape[-1]

        n_voxels = occupancy_volume_resolution
        # Get occupancy volume from predicted points
        pred_points = pred_points[0].clip(min=-0.5 + eps, max=0.5 - eps)
        volume_pred = create_occupancy_volume(pred_points, n_voxels=n_voxels)
        volume_pred = volume_pred > 0

        # Get occupancy volume from ground truth points
        if volume_gt is None:
            gt_points = gt_points.clip(min=-0.5 + eps, max=0.5 - eps)
            volume_gt = create_occupancy_volume(gt_points, n_voxels=n_voxels)
        else:
            assert volume_gt.shape[-3:] == (
                n_voxels,
                n_voxels,
                n_voxels,
            ), f"{volume_gt.shape=} {volume_pred.shape=}"
        if volume_gt.dtype != torch.bool:
            volume_gt = volume_gt > 0

        # Compute precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=volume_gt.flatten().cpu(),
            y_pred=volume_pred.flatten().cpu() > 0,
            average="binary",
            zero_division=0,
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def compute_rotation_error(
        quat_pred: Optional[torch.Tensor],
        quat_gt: Optional[torch.Tensor],
        rotation_candidates: torch.Tensor,
    ) -> float:
        if quat_pred is None or quat_gt is None:
            return {}
        if rotation_candidates is None:
            rotation_candidates = TdfyPerSample._default_rotation_candidates()
            rotation_candidates = rotation_candidates.to(quat_pred.device)
        # Rotation error
        # Note: min_loss_cd and min_rot_err might not use the same R_cand
        assert quat_pred.shape == quat_gt.shape, f"{quat_pred.shape=} {quat_gt.shape=}"
        R_gt = quaternion_to_matrix(quat_gt).expand_as(rotation_candidates)
        R_pred = quaternion_to_matrix(quat_pred).expand_as(rotation_candidates)
        R_pred_cand = rotation_candidates @ R_pred
        rot_err = rotation_angle(R_pred_cand, R_gt)
        min_rot_err = rot_err.min().item()
        return {"rot_error_deg": min_rot_err}

    @staticmethod
    def compute_oriented_shape_metrics(
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        quat_pred: Optional[torch.Tensor],
        quat_gt: Optional[torch.Tensor],
        volume_gt: Optional[torch.Tensor] = None,
        occupancy_volume_resolution: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        This method evaluates the rotation and shape quality for oriented shapes.
        It applies the predicted/GT rotation on the predicted/GT shapes. It will give us some sense
            on how the final output to a user would look like.
        We compute both the F1/CD on the oriented shapes, as well as treating the ICP alignment transformation
            as the rotation error for rotation eval.
        """
        if quat_pred is None or quat_gt is None:
            return {}
        if pred_points is None or gt_points is None:
            return {}

        R_gt = quaternion_to_matrix(quat_gt).squeeze()
        R_pred = quaternion_to_matrix(quat_pred).squeeze()
        pred_points = pred_points @ R_pred
        gt_points = gt_points @ R_gt

        loss_cd, _ = chamfer_distance_icp_aligned(
            pred_points, gt_points, icp_align=False
        )
        chamfer_distance_metrics = TdfyPerSample.compute_chamfer_distance_metrics(
            min_loss_cd=loss_cd,
        )

        # compute rotation between oriented shapes
        _, icp_solution = chamfer_distance_icp_aligned(
            pred_points, gt_points, icp_align=True
        )
        Rs = icp_solution.RTs.R
        try:
            rot_error = rotation_angle(Rs, torch.eye(3, device=Rs.device)[None]).item()
        except ValueError as e:
            logger.opt(exception=e).warning(
                f"ValueError found in metrics for sample: {e}"
            )
            return {}
        rotation_metrics = {"rot_error_deg": rot_error}

        # resize the the occupancy grid to fit into [-0.5, 0.5]
        pred_points = normalize_pcd(pred_points)
        gt_points = normalize_pcd(gt_points)

        # to match the input for the compute_occupancy_metrics
        # where gt_points does not have additional axis
        pred_points = pred_points.unsqueeze(0)
        oriented_shape_rotation_metrics = {}

        if volume_gt is not None:
            occupancy_volume_resolution = volume_gt.shape[-1]
        if pred_points.isnan().any():
            raise NanFoundInMetricException("pred_points contains NaNs")
        if gt_points is not None and gt_points.isnan().any():
            raise NanFoundInMetricException("gt_points contains NaNs")

        shape_occupancy_metrics = TdfyPerSample.compute_occupancy_metrics(
            pred_points=pred_points,
            gt_points=gt_points,
            occupancy_volume_resolution=occupancy_volume_resolution,
        )
        oriented_shape_rotation_metrics = {
            **chamfer_distance_metrics,
            **shape_occupancy_metrics,
            **rotation_metrics,
        }

        # rename to append "oriented_"
        oriented_shape_rotation_metrics = {
            "oriented_" + k: v for k, v in oriented_shape_rotation_metrics.items()
        }
        return oriented_shape_rotation_metrics

    @staticmethod
    def compute_translation_error(
        pose_pred: Optional[InstancePose],
        pose_gt: Optional[InstancePose],
    ) -> float:
        if pose_pred is None or pose_gt is None:
            return {}
        _metrics = {}
        # Error of translation direction "t_unit"
        angle_metrics = compute_angle(pose_pred.t_unit, pose_gt.t_unit)
        _metrics.update(
            {
                "trans_angle_err_deg": angle_metrics["angle_deg"].abs(),
                "trans_cos_angle": angle_metrics["cos_angle"].abs(),
            }
        )

        # Error of translation norm "r"
        len_pred = pose_pred.t_rel_norm
        len_gt = pose_gt.t_rel_norm
        len_square_error = (len_pred - len_gt).square()
        norm_metrics = {
            "trans_norm_err": len_square_error,
            "trans_norm_abs_rel_error": len_square_error / len_gt,
            "trans_norm_abs_log_rel_err": abs_log_rel(len_pred, len_gt),
            "trans_norm_abs_rel_delta1_acc": delta1_acc(len_pred, len_gt),
            "trans_norm_abs_rel_delta2_acc": delta2_acc(len_pred, len_gt),
            "trans_norm_abs_rel_delta3_acc": delta3_acc(len_pred, len_gt),
        }
        _metrics.update(norm_metrics)

        # Error of combined translation "t_unit * r"
        pos_pred = pose_pred.t_unit * pose_pred.t_rel_norm
        pos_gt = pose_gt.t_unit * pose_gt.t_rel_norm
        pos_error = torch.norm(pos_pred - pos_gt, dim=-1)
        pos_norm = torch.norm(pos_gt, dim=-1)
        total_error_metrics = {
            "trans_err": pos_error.mean(),
            "trans_abs_rel_error": pos_error / pos_norm,
            "trans_abs_log_rel_err": abs_log_rel(pos_error, pos_norm),
            "trans_abs_rel_delta1_acc": delta1_acc(pos_error + pos_norm, pos_norm),
            "trans_abs_rel_delta2_acc": delta2_acc(pos_error + pos_norm, pos_norm),
            "trans_abs_rel_delta3_acc": delta3_acc(pos_error + pos_norm, pos_norm),
        }
        _metrics.update(total_error_metrics)
        return {k: v.float().mean().item() for k, v in _metrics.items()}

    @staticmethod
    def compute_scale_error(
        pose_pred: Optional[InvariantPoseTarget],
        pose_gt: Optional[InvariantPoseTarget],
    ) -> float:
        if pose_pred is None or pose_gt is None:
            return {}
        scale_pred = pose_pred.s_tilde
        scale_gt = pose_gt.s_tilde
        if not (scale_pred.min() >= 0 and scale_gt.min() >= 0):
            logger.warning(f"{scale_pred.min()=} {scale_gt.min()=}")
        _metrics = {
            "scale_abs_rel_error": abs_relative_error(scale_pred, scale_gt),
            "scale_abs_log_rel_err": abs_log_rel(scale_pred, scale_gt),
            "scale_abs_rel_delta1_acc": delta1_acc(scale_pred, scale_gt),
            "scale_abs_rel_delta2_acc": delta2_acc(scale_pred, scale_gt),
            "scale_abs_rel_delta3_acc": delta3_acc(scale_pred, scale_gt),
        }
        return {k: v.float().mean().item() for k, v in _metrics.items()}


def apply_rotation(points: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation R (3x3) to point cloud 'points' (Nx3).
    We assume points are row vectors.
    """
    return torch.matmul(points, R.T)


def compute_angle(vec_pred, vec_gt):
    # Normalize the translation unit vectors
    vec_pred_unit = vec_pred / torch.norm(vec_pred, dim=-1, keepdim=True)
    vec_gt_unit = vec_gt / torch.norm(vec_gt, dim=-1, keepdim=True)
    cos_angle = torch.sum(vec_pred_unit * vec_gt_unit, dim=-1)
    return {
        "cos_angle": cos_angle,
        "angle_deg": torch.rad2deg(torch.acos(cos_angle)),
    }


class PoseUtils:
    pose_target_keys = list(PoseTarget.__dataclass_fields__.keys())
    instance_pose_keys = list(InstancePose.__dataclass_fields__.keys())
    invariant_pose_keys = list(InvariantPoseTarget.__dataclass_fields__.keys())

    @staticmethod
    def get_pred_gt_instance_pose(
        pred_dict: Dict[str, Any], gt_dict: Dict[str, Any]
    ) -> Tuple[InstancePose, InstancePose]:
        # GT
        instance_pose_dict_gt = PoseUtils.keep_keys(
            gt_dict, PoseUtils.instance_pose_keys
        )
        if len(instance_pose_dict_gt) == 0:
            return None, None
        instance_pose_gt = InstancePose(**instance_pose_dict_gt)

        # Pred, filling missing keys from GT
        pred_pose_dict = PoseUtils.keep_keys(pred_dict, PoseUtils.pose_target_keys)
        if len(pred_pose_dict) == 0:
            return None, None
        pose_target_pred = PoseUtils.fill_missing_pose_target_keys(
            pred_pose_dict, instance_pose_gt
        )
        instance_pose_pred = PoseTargetConverter.pose_target_to_instance_pose(
            pose_target_pred, normalize=pred_dict.get("pose_normalize", False)
        )
        return instance_pose_pred, instance_pose_gt

    @staticmethod
    def keep_keys(sample: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        return {k: v for k, v in sample.items() if k in keys}

    @staticmethod
    def instance_pose_from_sample(sample: Dict[str, Any]) -> InstancePose:
        return InstancePose(**PoseUtils.keep_keys(sample, PoseUtils.instance_pose_keys))

    @staticmethod
    def fill_missing_pose_target_keys(
        dest: Dict[str, Any], source: InstancePose
    ) -> PoseTarget:
        """Fill missing keys in the prediction from the values in the ground truth"""
        pose_target_convention = dest["pose_target_convention"]
        source = PoseTargetConverter.dicts_instance_pose_to_pose_target(
            pose_target_convention=pose_target_convention, **asdict(source)
        )
        for key in PoseUtils.pose_target_keys:
            if key not in dest:
                dest[key] = source[key]
        return PoseTarget(**dest)


class NanFoundInMetricException(Exception):
    pass
