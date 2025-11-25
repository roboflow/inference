import unittest
import torch
from unittest.mock import Mock, MagicMock
from typing import Tuple

from lidra.data.dataset.flexiset.transforms.trellis.image_and_mask import (
    ImageAndMaskMess,
)
from lidra.data.dataset.tdfy.trellis.dataset import PreProcessor


class TestImageAndMaskMess(unittest.TestCase):
    """Test ImageAndMaskMess class with pointmap support"""

    def setUp(self):
        """Set up test data and mocks"""
        self.device = "cpu"

        # Create test data
        self.image = torch.rand(3, 10, 10, device=self.device)
        self.mask = torch.rand(10, 10, device=self.device)
        self.pointmap = torch.rand(3, 10, 10, device=self.device)

        # Create mock transforms
        self.dual_transform_mock = Mock(side_effect=self._dual_transform)
        self.triple_transform_mock = Mock(side_effect=self._triple_transform)
        self.img_transform_mock = Mock(side_effect=lambda x: x * 2)
        self.mask_transform_mock = Mock(side_effect=lambda x: x > 0.5)
        self.pointmap_transform_mock = Mock(side_effect=lambda x: x + 1)

    def _dual_transform(self, img, mask):
        """Mock dual transform that modifies both inputs"""
        return img + 0.1, mask + 0.1

    def _triple_transform(self, img, mask, pointmap):
        """Mock triple transform that modifies all inputs"""
        return img + 0.1, mask + 0.1, pointmap + 0.1

    def _create_preprocessor(self, **kwargs):
        """Create a mock PreProcessor with specified attributes"""
        preprocessor = Mock(spec=PreProcessor)

        # Set default attributes
        preprocessor.img_mask_joint_transform = []
        preprocessor.img_mask_pointmap_joint_transform = (None,)
        preprocessor.img_transform = None
        preprocessor.mask_transform = None
        preprocessor.pointmap_transform = None

        # Override with provided kwargs
        for key, value in kwargs.items():
            setattr(preprocessor, key, value)

        return preprocessor

    def test_backward_compatibility_dual_transforms(self):
        """Test that dual transforms work without pointmap (backward compatibility)"""
        preprocessor = self._create_preprocessor(
            img_mask_joint_transform=[self.dual_transform_mock],
            img_transform=self.img_transform_mock,
            mask_transform=self.mask_transform_mock,
        )

        transform = ImageAndMaskMess(preprocessor)
        img_result, mask_result = transform._transform(self.image, self.mask)

        # Check that dual transform was called
        self.dual_transform_mock.assert_called_once()

        # Check individual transforms were applied
        expected_img = (self.image + 0.1) * 2  # dual transform + img transform
        expected_mask = (self.mask + 0.1) > 0.5  # dual transform + mask transform

        self.assertTrue(torch.allclose(img_result, expected_img, atol=1e-6))
        self.assertTrue(torch.equal(mask_result, expected_mask))

    def test_triple_transform_with_pointmap(self):
        """Test triple transforms when pointmap is provided"""
        preprocessor = self._create_preprocessor(
            img_mask_joint_transform=[self.dual_transform_mock],
            img_mask_pointmap_joint_transform=[self.triple_transform_mock],
            img_transform=self.img_transform_mock,
            mask_transform=self.mask_transform_mock,
            pointmap_transform=self.pointmap_transform_mock,
        )

        transform = ImageAndMaskMess(preprocessor)
        img_result, mask_result, pointmap_result = transform._transform(
            self.image, self.mask, self.pointmap
        )

        # Triple transform should be used, not dual
        self.triple_transform_mock.assert_called_once()
        self.dual_transform_mock.assert_not_called()

        # Check that all transforms were applied correctly
        expected_img = (self.image + 0.1) * 2
        expected_mask = (self.mask + 0.1) > 0.5
        expected_pointmap = (self.pointmap + 0.1) + 1

        self.assertTrue(torch.allclose(img_result, expected_img, atol=1e-6))
        self.assertTrue(torch.equal(mask_result, expected_mask))
        self.assertTrue(torch.allclose(pointmap_result, expected_pointmap, atol=1e-6))

    def test_fallback_to_dual_transforms(self):
        """Test fallback to dual transforms when triple transforms not available"""
        # Create a dual transform that can handle optional pointmap
        flexible_dual_transform = Mock(
            side_effect=lambda img, mask, pointmap=None: (
                (img + 0.1, mask + 0.1, pointmap + 0.1)
                if pointmap is not None
                else (img + 0.1, mask + 0.1)
            )
        )

        preprocessor = self._create_preprocessor(
            img_mask_joint_transform=[flexible_dual_transform],
            # No img_mask_pointmap_joint_transform set
        )

        transform = ImageAndMaskMess(preprocessor)
        img_result, mask_result, pointmap_result = transform._transform(
            self.image, self.mask, self.pointmap
        )

        # Dual transform should be called with pointmap
        flexible_dual_transform.assert_called_once()

        # Results should include pointmap
        self.assertEqual(img_result.shape, self.image.shape)
        self.assertEqual(mask_result.shape, self.mask.shape)
        self.assertEqual(pointmap_result.shape, self.pointmap.shape)

    def test_multiple_transforms_chain(self):
        """Test chaining multiple transforms"""
        # Create multiple transforms
        transform1 = Mock(
            side_effect=lambda img, mask, pm: (img + 0.1, mask + 0.1, pm + 0.1)
        )
        transform2 = Mock(side_effect=lambda img, mask, pm: (img * 2, mask * 2, pm * 2))

        preprocessor = self._create_preprocessor(
            img_mask_pointmap_joint_transform=[transform1, transform2]
        )

        transform = ImageAndMaskMess(preprocessor)
        img_result, mask_result, pointmap_result = transform._transform(
            self.image, self.mask, self.pointmap
        )

        # Both transforms should be called in order
        transform1.assert_called_once()
        transform2.assert_called_once()

        # Check cumulative effect
        expected_img = (self.image + 0.1) * 2
        expected_mask = (self.mask + 0.1) * 2
        expected_pointmap = (self.pointmap + 0.1) * 2

        self.assertTrue(torch.allclose(img_result, expected_img, atol=1e-6))
        self.assertTrue(torch.allclose(mask_result, expected_mask, atol=1e-6))
        self.assertTrue(torch.allclose(pointmap_result, expected_pointmap, atol=1e-6))

    def test_none_transforms_handling(self):
        """Test handling of None transforms"""
        preprocessor = self._create_preprocessor(
            img_mask_joint_transform=[],
            img_mask_pointmap_joint_transform=None,  # Explicitly None
        )

        transform = ImageAndMaskMess(preprocessor)

        # Should work without errors
        img_result, mask_result, pointmap_result = transform._transform(
            self.image, self.mask, self.pointmap
        )

        # Should return unchanged inputs
        self.assertTrue(torch.equal(img_result, self.image))
        self.assertTrue(torch.equal(mask_result, self.mask))
        self.assertTrue(torch.equal(pointmap_result, self.pointmap))

    def test_mixed_transform_returns(self):
        """Test handling of transforms that return different numbers of outputs"""
        # Transform that sometimes returns 3 values, sometimes 2
        mixed_transform = Mock(
            side_effect=lambda img, mask, pointmap=None: (
                (img + 0.1, mask + 0.1, pointmap + 0.1)
                if pointmap is not None
                else (img + 0.1, mask + 0.1)
            )
        )

        preprocessor = self._create_preprocessor(
            img_mask_joint_transform=[mixed_transform]
        )

        transform = ImageAndMaskMess(preprocessor)

        # Test with pointmap
        img_result, mask_result, pointmap_result = transform._transform(
            self.image, self.mask, self.pointmap
        )
        self.assertIsNotNone(pointmap_result)

        # Test without pointmap
        img_result, mask_result = transform._transform(self.image, self.mask)
        self.assertEqual(len([img_result, mask_result]), 2)

    def test_individual_transforms_only(self):
        """Test applying only individual transforms without joint transforms"""
        preprocessor = self._create_preprocessor(
            img_mask_joint_transform=[],
            img_transform=self.img_transform_mock,
            mask_transform=self.mask_transform_mock,
            pointmap_transform=self.pointmap_transform_mock,
        )

        transform = ImageAndMaskMess(preprocessor)
        img_result, mask_result, pointmap_result = transform._transform(
            self.image, self.mask, self.pointmap
        )

        # Check individual transforms were applied
        expected_img = self.image * 2
        expected_mask = self.mask > 0.5
        expected_pointmap = self.pointmap + 1

        self.assertTrue(torch.allclose(img_result, expected_img, atol=1e-6))
        self.assertTrue(torch.equal(mask_result, expected_mask))
        self.assertTrue(torch.allclose(pointmap_result, expected_pointmap, atol=1e-6))

    def test_missing_pointmap_transform_attribute(self):
        """Test handling when pointmap_transform attribute doesn't exist"""
        preprocessor = self._create_preprocessor(img_mask_joint_transform=[])
        # Remove pointmap_transform attribute
        delattr(preprocessor, "pointmap_transform")

        transform = ImageAndMaskMess(preprocessor)

        # Should work without errors
        img_result, mask_result, pointmap_result = transform._transform(
            self.image, self.mask, self.pointmap
        )

        # Pointmap should be unchanged
        self.assertTrue(torch.equal(pointmap_result, self.pointmap))

    def test_transform_method_wrapper(self):
        """Test the public transform method wrapper"""
        preprocessor = self._create_preprocessor(
            img_mask_joint_transform=[self.dual_transform_mock]
        )

        transform = ImageAndMaskMess(preprocessor)

        # Test with two inputs
        results = transform.transform(self.image, self.mask)
        self.assertEqual(len(results), 2)

        # Test with three inputs
        preprocessor.img_mask_pointmap_joint_transform = [self.triple_transform_mock]
        results = transform.transform(self.image, self.mask, self.pointmap)
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
