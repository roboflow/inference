import unittest
import torch
import pytorch3d.transforms
from pytorch3d.transforms import matrix_to_quaternion

from lidra.data.dataset.tdfy.transforms_3d import (
    compose_transform,
    decompose_transform,
)


class TestTransform3d(unittest.TestCase):
    def test_decompose_transform(self, device="cpu"):
        """Test the decompose_transform function with 10 random transforms"""
        n_tests = 5

        scales = torch.rand(n_tests, 3, device=device) * 2.0  # (N, 3) scale factors
        rotations = pytorch3d.transforms.random_rotations(
            n_tests, device=device
        )  # (N, 3, 3) rotation matrices
        translations = (
            torch.rand(n_tests, 3, device=device) - 0.5
        )  # (N, 3) translation vectors

        transforms = compose_transform(scales, rotations, translations).to(device)
        scales_decomposed, rotations_decomposed, translations_decomposed = (
            decompose_transform(transforms)
        )

        self.assertTrue(torch.allclose(scales, scales_decomposed))
        self.assertTrue(torch.allclose(rotations, rotations_decomposed))
        self.assertTrue(torch.allclose(translations, translations_decomposed))

    def test_decompose_transform_isotropic(self, device="cpu"):
        n_tests = 5
        scales = torch.rand(n_tests, device=device) * 2.0  # (N, 3) scale factors
        rotations = pytorch3d.transforms.random_rotations(
            n_tests, device=device
        )  # (N, 3, 3) rotation matrices
        translations = (
            torch.rand(n_tests, 3, device=device) - 0.5
        )  # (N, 3) translation vectors

        transforms = compose_transform(scales, rotations, translations).to(device)
        scales_decomposed, rotations_decomposed, translations_decomposed = (
            decompose_transform(transforms)
        )
        scales = scales.unsqueeze(-1).expand_as(scales_decomposed)

        self.assertTrue(torch.allclose(scales, scales_decomposed))
        self.assertTrue(torch.allclose(rotations, rotations_decomposed))
        self.assertTrue(torch.allclose(translations, translations_decomposed))


if __name__ == "__main__":
    unittest.main()
