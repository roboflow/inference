import unittest
import torch
from lidra.model.backbone.dit.embedder.point_remapper import PointRemapper


class TestPointRemapper(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Test with various point configurations
        self.test_points = [
            # Basic positive values
            torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32),
            # Values near zero
            torch.tensor([[[0.1, 0.01, 0.001]]], dtype=torch.float32),
            # Negative xy values
            torch.tensor([[[-1.0, -2.0, 1.0]]], dtype=torch.float32),
            # Mixed values
            torch.tensor([[[-3.5, 2.0, 0.5], [0.0, 0.0, 8.0]]], dtype=torch.float32),
            # Batch of points
            torch.tensor(
                [
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    [[-1.0, -1.0, 0.5], [0.0, 0.0, 5.0]],
                ],
                dtype=torch.float32,
            ),
        ]
        self.atol = 1e-5

    def test_all_remapping_roundtrips(self):
        """Test roundtrip (forward + inverse) for all remapping types."""
        for remap_type in PointRemapper.VALID_TYPES:
            with self.subTest(remap_type=remap_type):
                remapper = PointRemapper(remap_type)

                for points in self.test_points:
                    remapped = remapper(points)
                    recovered = remapper.inverse(remapped)

                    # Check no NaN or inf in remapped values
                    self.assertFalse(remapped.isnan().any())
                    self.assertFalse(remapped.isinf().any())

                    # Check roundtrip accuracy
                    # Note: sinh_exp might have slightly lower precision due to log clamp
                    atol = 1e-4 if remap_type == "sinh_exp" else self.atol
                    self.assertTrue(torch.allclose(points, recovered, atol=atol))

    def test_invalid_remap_type(self):
        """Test that invalid remap types raise ValueError."""
        with self.assertRaises(ValueError):
            PointRemapper("invalid_type")

    def test_gradient_flow(self):
        """Test that gradients flow through remapping."""
        for remap_type in PointRemapper.VALID_TYPES:
            with self.subTest(remap_type=remap_type):
                remapper = PointRemapper(remap_type)
                points = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)

                remapped = remapper(points)
                loss = remapped.sum()
                loss.backward()

                self.assertIsNotNone(points.grad)
                self.assertFalse(points.grad.isnan().any())

    def test_edge_cases(self):
        """Test edge cases for each remapping type."""
        edge_cases = [
            # Zero values
            torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32),
            # Very small positive z
            torch.tensor([[[1.0, 1.0, 1e-8]]], dtype=torch.float32),
            # Large values
            torch.tensor([[[100.0, -100.0, 100.0]]], dtype=torch.float32),
        ]

        for remap_type in PointRemapper.VALID_TYPES:
            remapper = PointRemapper(remap_type)
            for points in edge_cases:
                with self.subTest(remap_type=remap_type, points=points):
                    remapped = remapper(points)
                    recovered = remapper.inverse(remapped)

                    # Check no NaN or inf
                    self.assertFalse(remapped.isnan().any())
                    self.assertFalse(remapped.isinf().any())
                    self.assertFalse(recovered.isnan().any())
                    self.assertFalse(recovered.isinf().any())

                    # Check recovery (with relaxed tolerance for edge cases)
                    self.assertTrue(
                        torch.allclose(points, recovered, atol=1e-3, rtol=1e-3)
                    )

    def test_batch_processing(self):
        """Test that remapping works correctly with batched inputs."""
        batch_size = 4
        height, width = 8, 8
        points = torch.randn(batch_size, height, width, 3, dtype=torch.float32)
        # Ensure positive z values for exp remapping
        points[..., 2] = points[..., 2].abs() + 0.1

        for remap_type in PointRemapper.VALID_TYPES:
            with self.subTest(remap_type=remap_type):
                remapper = PointRemapper(remap_type)
                remapped = remapper(points)
                recovered = remapper.inverse(remapped)

                self.assertEqual(remapped.shape, points.shape)
                self.assertEqual(recovered.shape, points.shape)
                self.assertTrue(torch.allclose(points, recovered, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
