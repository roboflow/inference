import unittest
import torch
import logging

from lidra.data.dataset.tdfy.pose_target import (
    PoseTarget,
    InstancePose,
    InvariantPoseTarget,
    Naive,
    NormalizedSceneScale,
    NormalizedSceneScaleAndTranslation,
    ApparentSize,
    ScaleShiftInvariant,
    Identity,
    DisparitySpace,
)


def create_invariant_pose_target():
    # Use a batch size of 1 for simplicity.
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion.
    s_scene = torch.tensor([[2.0, 2.0, 2.0]])  # Scene scale set to 1.
    t_scene_center = torch.tensor([[0.0, 0.0, 1.5]])  # Scene center.
    s_rel = torch.tensor([[3.0]])  # Relative scale.
    t_rel_norm = torch.tensor([[5.0]])  # Norm of the relative translation.
    t_unit = torch.tensor([[1.0, 0.0, 0.0]])  # Unit translation along the x-axis.
    s_tilde = s_rel / t_rel_norm  # Should be 2/3.

    invariant = InvariantPoseTarget(
        q=q,
        s_scene=s_scene,
        t_scene_center=t_scene_center,
        s_rel=s_rel,
        t_rel_norm=t_rel_norm,
        t_unit=t_unit,
        s_tilde=s_tilde,
    )

    return invariant


def squeeze_instance_pose(instance_pose):
    return InstancePose(
        instance_scale_l2c=instance_pose.instance_scale_l2c.squeeze(0),
        instance_quaternion_l2c=instance_pose.instance_quaternion_l2c.squeeze(0),
        instance_position_l2c=instance_pose.instance_position_l2c.squeeze(0),
        scene_scale=instance_pose.scene_scale.squeeze(0),
        scene_shift=instance_pose.scene_shift.squeeze(0),
    )


class TestPoseTarget(unittest.TestCase):
    def setUp(self):
        # Set up a canonical invariant target and also corresponding instance pose components.
        self.invariant = create_invariant_pose_target()
        self.instance_pose = InvariantPoseTarget.to_instance_pose(self.invariant)

    def _check_invariant_equal(self, a, b, atol=1e-6):
        self.assertTrue(torch.allclose(a.q, b.q, atol=atol), f"q: {a.q} != {b.q}")
        self.assertTrue(
            torch.allclose(a.s_scene, b.s_scene, atol=atol),
            f"s_scene: {a.s_scene} != {b.s_scene}",
        )
        self.assertTrue(
            torch.allclose(a.t_scene_center, b.t_scene_center, atol=atol),
            f"t_scene_center: {a.t_scene_center} != {b.t_scene_center}",
        )
        self.assertTrue(
            torch.allclose(a.s_rel, b.s_rel, atol=atol),
            f"s_rel: {a.s_rel} != {b.s_rel}",
        )
        self.assertTrue(
            torch.allclose(a.t_rel_norm, b.t_rel_norm, atol=atol),
            f"t_rel_norm: {a.t_rel_norm} != {b.t_rel_norm}",
        )
        self.assertTrue(
            torch.allclose(a.t_unit, b.t_unit, atol=atol),
            f"t_unit: {a.t_unit} != {b.t_unit}",
        )
        self.assertTrue(
            torch.allclose(a.s_tilde, b.s_tilde, atol=atol),
            f"s_tilde: {a.s_tilde} != {b.s_tilde}",
        )

    def _check_instance_pose_equal(self, a, b, atol=1e-6):
        self.assertTrue(
            torch.allclose(a.instance_scale_l2c, b.instance_scale_l2c, atol=atol)
        )
        self.assertTrue(
            torch.allclose(
                a.instance_quaternion_l2c, b.instance_quaternion_l2c, atol=atol
            )
        )
        self.assertTrue(
            torch.allclose(a.instance_position_l2c, b.instance_position_l2c, atol=atol)
        )
        self.assertTrue(torch.allclose(a.scene_scale, b.scene_scale, atol=atol))

    def _check_pose_target_equal(self, a, b, atol=1e-6):
        self.assertTrue(
            torch.allclose(a.x_instance_scale, b.x_instance_scale, atol=atol)
        )
        self.assertTrue(
            torch.allclose(a.x_instance_rotation, b.x_instance_rotation, atol=atol)
        )
        self.assertTrue(
            torch.allclose(
                a.x_instance_translation, b.x_instance_translation, atol=atol
            )
        )
        self.assertTrue(torch.allclose(a.x_scene_scale, b.x_scene_scale, atol=atol))
        self.assertTrue(
            torch.allclose(a.x_translation_scale, b.x_translation_scale, atol=atol)
        )

    def test_invariant_pose_targets_roundtrip(self):
        invariant_pt = InvariantPoseTarget.from_instance_pose(self.instance_pose)
        instance_pose_rt = InvariantPoseTarget.to_instance_pose(invariant_pt)
        self._check_instance_pose_equal(instance_pose_rt, self.instance_pose)

        invariant_pt_rt = InvariantPoseTarget.from_instance_pose(self.instance_pose)
        self._check_invariant_equal(invariant_pt_rt, self.invariant)

    def _roundtrip_invariant(self, pose_target_convention):
        pt = pose_target_convention.from_invariant(self.invariant)
        invariant_rt = pose_target_convention.to_invariant(pt)
        self._check_invariant_equal(invariant_rt, self.invariant)

    def _roundtrip_instance_pose(self, pose_target_convention, squeeze=False):
        instance_pose = self.instance_pose
        if squeeze:
            instance_pose = squeeze_instance_pose(instance_pose)
        pt = pose_target_convention.from_instance_pose(instance_pose)
        instance_pose_rt = pose_target_convention.to_instance_pose(pt)

        self._check_instance_pose_equal(instance_pose_rt, instance_pose)

    def _test_roundtrip(self, pose_target_convention):
        self._roundtrip_invariant(pose_target_convention)
        self._roundtrip_instance_pose(pose_target_convention)
        self._roundtrip_instance_pose(pose_target_convention, squeeze=True)

    def test_pose_target_naive_roundtrip(self):
        self._test_roundtrip(Naive)

    def test_pose_target_normalized_pointmap_scale_roundtrip(self):
        self._test_roundtrip(NormalizedSceneScale)

    def test_pose_target_normalized_scale_and_translation_roundtrip(self):
        self._test_roundtrip(NormalizedSceneScaleAndTranslation)

    def test_pose_target_apparent_size_roundtrip(self):
        self._test_roundtrip(ApparentSize)

    def test_pose_target_scaleshiftinvariant_roundtrip(self):
        self._test_roundtrip(ScaleShiftInvariant)

    def test_identity_direct_mapping(self):
        """Test that Identity convention performs direct passthrough without transformation"""
        # Create a simple instance pose with known values
        scale = torch.tensor([[2.5]])
        quaternion = torch.tensor(
            [[0.7071, 0.0, 0.7071, 0.0]]
        )  # 90 degree rotation around Y
        position = torch.tensor([[1.0, 2.0, 3.0]])
        scene_scale = torch.tensor([[1.5, 1.5, 1.5]])
        scene_shift = torch.tensor([[0.5, -0.5, 1.0]])

        instance_pose = InstancePose(
            instance_scale_l2c=scale,
            instance_quaternion_l2c=quaternion,
            instance_position_l2c=position,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )

        # Convert to PoseTarget using Identity
        pose_target = Identity.from_instance_pose(instance_pose)

        # Check that all values are directly mapped (passthrough)
        self.assertTrue(torch.allclose(pose_target.x_instance_scale, scale))
        self.assertTrue(torch.allclose(pose_target.x_instance_rotation, quaternion))
        self.assertTrue(torch.allclose(pose_target.x_instance_translation, position))
        # Identity now preserves scene_scale and scene_shift
        self.assertTrue(torch.allclose(pose_target.x_scene_scale, scene_scale))
        self.assertTrue(torch.allclose(pose_target.x_scene_center, scene_shift))

        # Convert back to InstancePose
        recovered_pose = Identity.to_instance_pose(pose_target)

        # Check that ALL original values are preserved
        self.assertTrue(torch.allclose(recovered_pose.instance_scale_l2c, scale))
        self.assertTrue(
            torch.allclose(recovered_pose.instance_quaternion_l2c, quaternion)
        )
        self.assertTrue(torch.allclose(recovered_pose.instance_position_l2c, position))
        self.assertTrue(torch.allclose(recovered_pose.scene_scale, scene_scale))
        self.assertTrue(torch.allclose(recovered_pose.scene_shift, scene_shift))

    def test_identity_multiple_objects(self):
        """Test Identity convention with multiple objects (batch size > 1, K > 1)"""
        # Create instance pose with batch size 2, 3 objects each
        b, k = 2, 3
        scale = torch.randn(b, k, 1)
        quaternion = torch.randn(b, k, 4)
        # Normalize quaternions
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
        position = torch.randn(b, k, 3)
        scene_scale = torch.ones(b, 3) * 2.0
        scene_shift = torch.randn(b, 3)

        instance_pose = InstancePose(
            instance_scale_l2c=scale,
            instance_quaternion_l2c=quaternion,
            instance_position_l2c=position,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )

        # Convert through Identity
        pose_target = Identity.from_instance_pose(instance_pose)
        recovered_pose = Identity.to_instance_pose(pose_target)

        # Check shapes are preserved
        self.assertEqual(recovered_pose.instance_scale_l2c.shape, scale.shape)
        self.assertEqual(recovered_pose.instance_quaternion_l2c.shape, quaternion.shape)
        self.assertEqual(recovered_pose.instance_position_l2c.shape, position.shape)
        self.assertEqual(recovered_pose.scene_scale.shape, scene_scale.shape)
        self.assertEqual(recovered_pose.scene_shift.shape, scene_shift.shape)

        # Check ALL values are preserved (including scene parameters)
        self.assertTrue(
            torch.allclose(recovered_pose.instance_scale_l2c, scale, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(
                recovered_pose.instance_quaternion_l2c, quaternion, atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(recovered_pose.instance_position_l2c, position, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(recovered_pose.scene_scale, scene_scale, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(recovered_pose.scene_shift, scene_shift, atol=1e-5)
        )

    def test_identity_roundtrip(self):
        """Test Identity convention roundtrip - now should work perfectly as a passthrough"""
        # Test with the standard test setup
        self._test_roundtrip(Identity)

    def test_identity_preserves_scene_params(self):
        """Test Identity preserves scene parameters"""
        scale = torch.tensor([[3.0]])
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity rotation
        position = torch.tensor([[2.0, 1.0, 0.5]])
        scene_scale = torch.tensor([[1.5, 1.5, 1.5]])
        scene_shift = torch.tensor([[0.5, 0.3, 0.2]])

        instance_pose = InstancePose(
            instance_scale_l2c=scale,
            instance_quaternion_l2c=quaternion,
            instance_position_l2c=position,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )

        # Convert to PoseTarget via Identity
        pose_target = Identity.from_instance_pose(instance_pose)

        # Verify Identity convention preserves scene_scale and scene_shift
        self.assertTrue(torch.allclose(pose_target.x_scene_scale, scene_scale))
        self.assertTrue(torch.allclose(pose_target.x_scene_center, scene_shift))

    def test_identity_convention_name(self):
        """Test that Identity convention has correct name"""
        self.assertEqual(Identity.pose_target_convention, "Identity")

    def test_identity_as_noop_for_metrics(self):
        """Test Identity works as a no-op for the metric_collection_per_sample use case"""
        # This simulates what happens in PoseUtils.get_pred_gt_instance_pose

        # Create ground truth instance pose
        gt_instance_pose = InstancePose(
            instance_scale_l2c=torch.tensor([[2.0]]),
            instance_quaternion_l2c=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            instance_position_l2c=torch.tensor([[1.0, 2.0, 3.0]]),
            scene_scale=torch.tensor([[1.5, 1.5, 1.5]]),
            scene_shift=torch.tensor([[0.5, 0.5, 0.5]]),
        )

        # Convert GT to pose target using Identity (simulates fill_missing_pose_target_keys)
        from lidra.data.dataset.tdfy.pose_target import PoseTargetConverter
        from dataclasses import asdict

        gt_as_pose_target_dict = PoseTargetConverter.dicts_instance_pose_to_pose_target(
            pose_target_convention="Identity", **asdict(gt_instance_pose)
        )

        # Create a prediction that's identical to GT (to test pure passthrough)
        pred_pose_target = PoseTarget(**gt_as_pose_target_dict)

        # Convert back to instance pose (simulates pose_target_to_instance_pose)
        pred_instance_pose = PoseTargetConverter.pose_target_to_instance_pose(
            pred_pose_target, normalize=False
        )

        # Verify complete passthrough - pred should equal GT exactly
        self.assertTrue(
            torch.allclose(
                pred_instance_pose.instance_scale_l2c,
                gt_instance_pose.instance_scale_l2c,
            )
        )
        self.assertTrue(
            torch.allclose(
                pred_instance_pose.instance_quaternion_l2c,
                gt_instance_pose.instance_quaternion_l2c,
            )
        )
        self.assertTrue(
            torch.allclose(
                pred_instance_pose.instance_position_l2c,
                gt_instance_pose.instance_position_l2c,
            )
        )
        self.assertTrue(
            torch.allclose(pred_instance_pose.scene_scale, gt_instance_pose.scene_scale)
        )
        self.assertTrue(
            torch.allclose(pred_instance_pose.scene_shift, gt_instance_pose.scene_shift)
        )


class TestDisparitySpace(unittest.TestCase):
    """Separate test class for DisparitySpace to avoid setUp issues"""

    def test_disparity_space_roundtrip_simple(self):
        """Test DisparitySpace roundtrip with simple values"""
        # Create instance pose with scene_scale=1 (required for DisparitySpace)
        scale = torch.tensor([[2.0]])
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity rotation
        position = torch.tensor([[1.0, 2.0, 5.0]])
        scene_scale = torch.tensor([1.0, 1.0, 1.0])  # Must be 1 for DisparitySpace
        scene_shift = torch.tensor(
            [0.1, 0.2, torch.log(torch.tensor(2.0))]
        )  # [x_shift, y_shift, log(z_shift)]

        instance_pose = InstancePose(
            instance_scale_l2c=scale,
            instance_quaternion_l2c=quaternion,
            instance_position_l2c=position,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )

        # Convert to PoseTarget using DisparitySpace
        pose_target = DisparitySpace.from_instance_pose(instance_pose)

        # Check the convention name
        self.assertEqual(pose_target.pose_target_convention, "DisparitySpace")

        # Convert back to InstancePose
        recovered_pose = DisparitySpace.to_instance_pose(pose_target)

        # Check that all values are recovered
        self.assertTrue(
            torch.allclose(recovered_pose.instance_scale_l2c, scale, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(
                recovered_pose.instance_quaternion_l2c, quaternion, atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(recovered_pose.instance_position_l2c, position, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(recovered_pose.scene_scale, scene_scale, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(recovered_pose.scene_shift, scene_shift, atol=1e-5)
        )

    def test_disparity_space_roundtrip_multiple_objects(self):
        """Test DisparitySpace roundtrip with multiple objects"""
        # For DisparitySpace, we need proper dimension alignment
        # Using batch=1 with k=3 objects to match DisparitySpace expectations
        b, k = 1, 3
        scale = torch.abs(torch.randn(b, k, 1)) + 0.1  # Positive scales [1, 3, 1]
        quaternion = torch.randn(b, k, 4)
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)  # Normalize
        position = torch.randn(b, k, 3)
        position[..., 2] = position[..., 2].abs() + 1.0  # Ensure positive z
        scene_scale = torch.ones(b, 3)  # Shape [1, 3] - Must be 1 for DisparitySpace
        scene_shift = torch.randn(b, 3)  # Shape [1, 3]

        instance_pose = InstancePose(
            instance_scale_l2c=scale,
            instance_quaternion_l2c=quaternion,
            instance_position_l2c=position,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )

        # Convert through DisparitySpace
        pose_target = DisparitySpace.from_instance_pose(instance_pose)
        recovered_pose = DisparitySpace.to_instance_pose(pose_target)

        # Check shapes - note scale may expand from [1,3,1] to [1,3,3] due to broadcasting
        self.assertEqual(
            recovered_pose.instance_scale_l2c.shape[-2:], (k, 3)
        )  # Check last 2 dims
        self.assertEqual(recovered_pose.instance_quaternion_l2c.shape, quaternion.shape)
        self.assertEqual(recovered_pose.instance_position_l2c.shape, position.shape)
        self.assertEqual(recovered_pose.scene_scale.shape, scene_scale.shape)
        self.assertEqual(recovered_pose.scene_shift.shape, scene_shift.shape)

        # Check values are recovered (scale might be broadcast from [1,3,1] to [1,3,3])
        if scale.shape[-1] == 1:
            # If original scale was isotropic, check all dims have same value
            scale_expanded = scale.expand(b, k, 3)
            self.assertTrue(
                torch.allclose(
                    recovered_pose.instance_scale_l2c, scale_expanded, atol=1e-5
                )
            )
        else:
            self.assertTrue(
                torch.allclose(recovered_pose.instance_scale_l2c, scale, atol=1e-5)
            )
        self.assertTrue(
            torch.allclose(
                recovered_pose.instance_quaternion_l2c, quaternion, atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(recovered_pose.instance_position_l2c, position, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(recovered_pose.scene_scale, scene_scale, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(recovered_pose.scene_shift, scene_shift, atol=1e-5)
        )

    def test_disparity_space_via_invariant(self):
        """Test DisparitySpace roundtrip via InvariantPoseTarget"""
        # Create instance pose
        scale = torch.tensor(
            [[[3.0, 3.0, 3.0]]]
        )  # Shape [1, 1, 3] for InvariantPoseTarget (batch=1, objects=1, dims=3)
        quaternion = torch.tensor([[[0.7071, 0.0, 0.7071, 0.0]]])  # Shape [1, 1, 4]
        position = torch.tensor([[[2.0, 1.0, 4.0]]])  # Shape [1, 1, 3]
        scene_scale = torch.ones(1, 3)  # Must be 1 for DisparitySpace
        scene_shift = torch.tensor([[0.5, -0.3, torch.log(torch.tensor(3.0))]])

        instance_pose = InstancePose(
            instance_scale_l2c=scale,
            instance_quaternion_l2c=quaternion,
            instance_position_l2c=position,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )

        # Test via invariant
        invariant = InvariantPoseTarget.from_instance_pose(instance_pose)
        pose_target = DisparitySpace.from_invariant(invariant)
        invariant_recovered = DisparitySpace.to_invariant(pose_target)
        instance_recovered = InvariantPoseTarget.to_instance_pose(invariant_recovered)

        # Check roundtrip
        self.assertTrue(
            torch.allclose(instance_recovered.instance_scale_l2c, scale, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(
                instance_recovered.instance_quaternion_l2c, quaternion, atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(
                instance_recovered.instance_position_l2c, position, atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(instance_recovered.scene_scale, scene_scale, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(instance_recovered.scene_shift, scene_shift, atol=1e-5)
        )

    def test_disparity_space_edge_cases(self):
        """Test DisparitySpace with edge cases"""
        # Test with very small z values (should still work due to log transform)
        scale = torch.tensor([[1.0]])
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        position = torch.tensor([[0.1, 0.2, 0.01]])  # Small z
        scene_scale = torch.ones(3)
        scene_shift = torch.tensor(
            [0.0, 0.0, torch.log(torch.tensor(0.01))]
        )  # Small z shift

        instance_pose = InstancePose(
            instance_scale_l2c=scale,
            instance_quaternion_l2c=quaternion,
            instance_position_l2c=position,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )

        pose_target = DisparitySpace.from_instance_pose(instance_pose)
        recovered = DisparitySpace.to_instance_pose(pose_target)

        self.assertTrue(torch.allclose(recovered.instance_scale_l2c, scale, atol=1e-5))
        self.assertTrue(
            torch.allclose(recovered.instance_position_l2c, position, atol=1e-5)
        )

    def test_disparity_space_roundtrip_batch(self):
        """Test DisparitySpace roundtrip with batched data (no multiple objects)"""
        # Test with batch dimension but single object per batch
        b = 4  # batch size
        scale = torch.abs(torch.randn(b, 1)) + 0.1  # Positive scales [4, 1]
        quaternion = torch.randn(b, 4)
        quaternion = quaternion / quaternion.norm(
            dim=-1, keepdim=True
        )  # Normalize [4, 4]
        position = torch.randn(b, 3)
        position[:, 2] = position[:, 2].abs() + 1.0  # Ensure positive z [4, 3]
        scene_scale = torch.ones(3)  # Shape [3] - broadcasted for batch
        scene_shift = torch.randn(3)  # Shape [3] - broadcasted for batch

        # Create batch of instance poses
        instance_poses = []
        for i in range(b):
            instance_poses.append(
                InstancePose(
                    instance_scale_l2c=scale[i : i + 1],
                    instance_quaternion_l2c=quaternion[i : i + 1],
                    instance_position_l2c=position[i : i + 1],
                    scene_scale=scene_scale,
                    scene_shift=scene_shift,
                )
            )

        # Test each in batch
        for i, instance_pose in enumerate(instance_poses):
            pose_target = DisparitySpace.from_instance_pose(instance_pose)
            recovered_pose = DisparitySpace.to_instance_pose(pose_target)

            # Check roundtrip for this batch item
            self.assertTrue(
                torch.allclose(
                    recovered_pose.instance_scale_l2c, scale[i : i + 1], atol=1e-5
                ),
                f"Batch item {i} scale mismatch",
            )
            self.assertTrue(
                torch.allclose(
                    recovered_pose.instance_quaternion_l2c,
                    quaternion[i : i + 1],
                    atol=1e-5,
                ),
                f"Batch item {i} quaternion mismatch",
            )
            self.assertTrue(
                torch.allclose(
                    recovered_pose.instance_position_l2c, position[i : i + 1], atol=1e-5
                ),
                f"Batch item {i} position mismatch",
            )

    def test_disparity_space_with_converter(self):
        """Test DisparitySpace using PoseTargetConverter utility"""
        from lidra.data.dataset.tdfy.pose_target import PoseTargetConverter

        # Create instance pose
        instance_pose = InstancePose(
            instance_scale_l2c=torch.tensor([[1.5]]),
            instance_quaternion_l2c=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            instance_position_l2c=torch.tensor([[1.0, 2.0, 3.0]]),
            scene_scale=torch.ones(3),
            scene_shift=torch.tensor([0.2, 0.3, torch.log(torch.tensor(2.0))]),
        )

        # Convert using converter
        pose_target = PoseTargetConverter.instance_pose_to_pose_target(
            instance_pose, "DisparitySpace", normalize=False
        )

        # Convert back
        recovered = PoseTargetConverter.pose_target_to_instance_pose(
            pose_target, normalize=False
        )

        # Verify roundtrip
        self.assertTrue(
            torch.allclose(
                recovered.instance_scale_l2c,
                instance_pose.instance_scale_l2c,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                recovered.instance_quaternion_l2c,
                instance_pose.instance_quaternion_l2c,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                recovered.instance_position_l2c,
                instance_pose.instance_position_l2c,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(recovered.scene_scale, instance_pose.scene_scale, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(recovered.scene_shift, instance_pose.scene_shift, atol=1e-5)
        )


if __name__ == "__main__":
    # To see log warnings (for example when the invariant target's fields are inconsistent)
    unittest.main()
