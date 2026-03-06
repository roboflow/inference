import unittest
import torch

from lidra.metrics.tdfy.metric_collection_per_sample import TdfyPerSample
from lidra.test.util import run_unittest
from lidra.data.dataset.tdfy.pose_target import (
    PoseTargetConverter,
    PoseTarget,
    InvariantPoseTarget,
    InstancePose,
)
from dataclasses import asdict


class TestTdfyPerSampleScaleTranslation(unittest.TestCase):
    def setUp(self):
        # 1. Create a example GT InstancePose
        # 2. Create a example prediction PoseTarget

        # Create example tensors for each field
        instance_scale = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(0)  # 1x3 tensor
        instance_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(
            0
        )  # 1x4 quaternion (w,x,y,z)
        instance_translation = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(0)  # 1x3 tensor
        scene_scale = torch.tensor([2.0]).unsqueeze(0)  # 1x1 tensor
        scene_shift = torch.tensor([0.0, 0.0, 1.5]).unsqueeze(0)  # 1x3 tensor
        translation_scale = torch.tensor([1.0]).unsqueeze(0)  # 1x1 tensor

        self.pose_target_convention = "ApparentSize"

        # GT
        instance_pose_gt = InstancePose(
            instance_scale_l2c=instance_scale,
            instance_quaternion_l2c=instance_rotation,
            instance_position_l2c=instance_translation,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )
        sample = {
            "extra_field1": torch.tensor([1.0]),
            "extra_field2": torch.tensor([2.0]),
            "extra_field3": torch.tensor([3.0]),
            **asdict(instance_pose_gt),
        }
        self.sample = sample

        # PREDICTION
        self.pred_dict_shape_only = {"x_shape": torch.tensor([1.0, 1.0, 1.0])}
        pred_dict = PoseTargetConverter.dicts_instance_pose_to_pose_target(
            pose_target_convention=self.pose_target_convention,
            **asdict(instance_pose_gt)
        )
        pred_dict.pop("x_scene_scale")
        pred_dict.pop("x_translation_scale")
        pred_dict.update(self.pred_dict_shape_only)
        self.pred_dict = pred_dict

        self.pose_pred_std, self.pose_gt_std = TdfyPerSample.invariant_pose_targets(
            self.pred_dict, self.sample
        )

    def test_invariant_pose_targets(self):
        self.assertTrue(
            torch.allclose(self.pose_pred_std.s_tilde, self.pose_gt_std.s_tilde)
        )

    def test_scale_error(self):
        scale_metrics = TdfyPerSample.compute_scale_error(
            pose_pred=self.pose_pred_std,
            pose_gt=self.pose_gt_std,
        )
        self.assertAlmostEqual(scale_metrics["scale_abs_rel_error"], 0.0)

    def test_translation_error(self):
        translation_metrics = TdfyPerSample.compute_translation_error(
            pose_pred=self.pose_pred_std,
            pose_gt=self.pose_gt_std,
        )
        self.assertAlmostEqual(translation_metrics["trans_err"], 0.0)


class TestTdfyPerSampleOccupancy(unittest.TestCase):
    def setUp(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Create a simple 4x4x4 ground truth occupancy volume
        self.resolution = 8
        self.gt_volume = torch.zeros(
            (1, 1, self.resolution, self.resolution, self.resolution), device=device
        )
        # Set some voxels as occupied (e.g., a simple cube in the middle)
        self.gt_volume[0, 0, 3:5, 3:5, 3:5] = 1.0
        self.rotation_candidates = torch.eye(3, device=device).unsqueeze(0)

    def _test_occupancy(self, prediction, target):
        metrics = TdfyPerSample.evaluate(
            prediction={"occupancy_volume": prediction},
            target={"occupancy_volume": target},
            occupancy_volume_resolution=self.resolution,
            rotation_candidates=self.rotation_candidates,
        )
        return metrics

    def test_perfect_prediction_occupancy(self):
        """Test that using ground truth as prediction gives perfect metrics."""
        metrics = self._test_occupancy(self.gt_volume.clone(), self.gt_volume)

        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)

    def test_empty_prediction_occupancy(self):
        """Test metrics when prediction is completely empty."""
        empty_pred = torch.zeros_like(self.gt_volume)
        metrics = self._test_occupancy(empty_pred, self.gt_volume)

        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 0.0)
        self.assertAlmostEqual(metrics["f1"], 0.0)

    def test_full_prediction_occupancy(self):
        """Test metrics when prediction is completely full."""
        full_pred = torch.ones_like(self.gt_volume)
        metrics = self._test_occupancy(full_pred, self.gt_volume)

        # Calculate expected precision
        # Precision = true positives / (true positives + false positives)
        n_occupied_gt = (self.gt_volume > 0).sum().item()
        total_voxels = self.resolution**3
        expected_precision = n_occupied_gt / total_voxels

        self.assertAlmostEqual(metrics["precision"], expected_precision)
        self.assertAlmostEqual(
            metrics["recall"], 1.0
        )  # All ground truth voxels are found

    # Add tests for pose estimation


if __name__ == "__main__":
    run_unittest(TestTdfyPerSampleScaleTranslation)
    run_unittest(TestTdfyPerSampleOccupancy)
