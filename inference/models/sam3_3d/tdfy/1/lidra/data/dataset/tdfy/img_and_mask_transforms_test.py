import unittest
import torch

from lidra.data.dataset.tdfy.img_and_mask_transforms import (
    rembg,
    crop_around_mask_with_padding,
    crop_around_mask_with_random_box_size_factor,
    resize_all_to_same_size,
)


class TestImgAndMaskTransformsPointmap(unittest.TestCase):
    """Test pointmap support in img_and_mask_transforms functions"""

    def setUp(self):
        """Set up test data"""
        self.device = "cpu"

        # Create test data
        self.img = torch.rand(3, 10, 10, device=self.device)
        self.mask = torch.ones(10, 10, device=self.device)
        self.mask[3:7, 3:7] = 0  # Create a hole in the mask

        # Create a test pointmap with recognizable values
        self.pointmap = torch.zeros(3, 10, 10, device=self.device)
        self.pointmap[0, :, :] = (
            torch.arange(10).unsqueeze(0).expand(10, 10)
        )  # x coordinates
        self.pointmap[1, :, :] = (
            torch.arange(10).unsqueeze(1).expand(10, 10)
        )  # y coordinates
        self.pointmap[2, :, :] = 1.0  # z coordinates

        # Create mask with object only in center
        self.object_mask = torch.zeros(10, 10, device=self.device)
        self.object_mask[3:7, 3:7] = 1.0

    def test_rembg_backward_compatibility(self):
        """Test that rembg maintains backward compatibility"""
        # Test without pointmap - returns (image, mask)
        img_result, mask_result = rembg(self.img, self.mask)
        self.assertIsInstance(img_result, torch.Tensor)
        self.assertIsInstance(mask_result, torch.Tensor)
        self.assertEqual(img_result.shape, self.img.shape)
        self.assertTrue(torch.equal(mask_result, self.mask))

    def test_rembg_with_pointmap(self):
        """Test rembg with pointmap support"""
        # Test basic pointmap support - now returns 3 values
        img_result, mask_result, pointmap_result = rembg(
            self.img, self.mask, pointmap=self.pointmap
        )

        # Check shapes match
        self.assertEqual(img_result.shape, self.img.shape)
        self.assertEqual(mask_result.shape, self.mask.shape)
        self.assertEqual(pointmap_result.shape, self.pointmap.shape)

        # Check that background is set to NaN in pointmap
        background_mask = self.mask == 0
        self.assertTrue(torch.isnan(pointmap_result[:, background_mask]).all())

        # Check that foreground values are preserved
        foreground_mask = self.mask > 0
        self.assertTrue(
            torch.allclose(
                pointmap_result[:, foreground_mask],
                self.pointmap[:, foreground_mask],
                equal_nan=True,
            )
        )

    def test_rembg_with_pointmap_returns_three_values(self):
        """Test rembg with pointmap always returns three values"""
        img_result, mask_result, pointmap_result = rembg(
            self.img, self.mask, pointmap=self.pointmap
        )

        # Check all three outputs
        self.assertEqual(img_result.shape, self.img.shape)
        self.assertTrue(torch.equal(mask_result, self.mask))
        self.assertEqual(pointmap_result.shape, self.pointmap.shape)

        # Verify NaN handling
        background_mask = self.mask == 0
        self.assertTrue(torch.isnan(pointmap_result[:, background_mask]).all())

    def test_crop_around_mask_with_padding_backward_compatibility(self):
        """Test crop_around_mask_with_padding backward compatibility"""
        # Test without pointmap
        img_result, mask_result = crop_around_mask_with_padding(
            self.img, self.object_mask, box_size_factor=1.2, padding_factor=0.1
        )

        # Should return two tensors
        self.assertIsInstance(img_result, torch.Tensor)
        self.assertIsInstance(mask_result, torch.Tensor)

        # Check that dimensions match
        self.assertEqual(img_result.shape[1:], mask_result.shape)

        # Should be square after padding
        self.assertEqual(img_result.shape[1], img_result.shape[2])

    def test_crop_around_mask_with_padding_pointmap(self):
        """Test crop_around_mask_with_padding with pointmap support"""
        img_result, mask_result, pointmap_result = crop_around_mask_with_padding(
            self.img,
            self.object_mask,
            box_size_factor=1.2,
            padding_factor=0.1,
            pointmap=self.pointmap,
        )

        # Check all three outputs
        self.assertEqual(len(img_result.shape), 3)
        self.assertEqual(len(mask_result.shape), 2)
        self.assertEqual(len(pointmap_result.shape), 3)

        # All should have same spatial dimensions
        self.assertEqual(img_result.shape[1:], mask_result.shape)
        self.assertEqual(img_result.shape[1:], pointmap_result.shape[1:])

        # Should be square
        self.assertEqual(img_result.shape[1], img_result.shape[2])
        self.assertEqual(pointmap_result.shape[1], pointmap_result.shape[2])

        # Check that the pointmap was processed (has expected shape and values)
        # The exact NaN pattern depends on the crop and padding parameters
        # For this test, we just verify the pointmap was transformed correctly
        self.assertFalse(
            torch.equal(pointmap_result, self.pointmap)
        )  # Should be different from input

        # At least verify it's not all NaN
        self.assertFalse(torch.isnan(pointmap_result).all())

    def test_crop_around_mask_with_padding_no_padding_factor(self):
        """Test crop_around_mask_with_padding with zero padding_factor"""
        img_result, mask_result, pointmap_result = crop_around_mask_with_padding(
            self.img,
            self.object_mask,
            box_size_factor=1.0,
            padding_factor=0.0,
            pointmap=self.pointmap,
        )

        # Should still work but with minimal padding
        self.assertEqual(img_result.shape[1], img_result.shape[2])
        self.assertEqual(pointmap_result.shape[1], pointmap_result.shape[2])

    def test_crop_with_mask_channel_dimension(self):
        """Test crop_around_mask_with_padding with mask having channel dimension"""
        mask_with_channel = self.object_mask.unsqueeze(0)  # Add channel dimension

        img_result, mask_result, pointmap_result = crop_around_mask_with_padding(
            self.img,
            mask_with_channel,
            box_size_factor=1.2,
            padding_factor=0.1,
            pointmap=self.pointmap,
        )

        # Mask should have channel dimension restored
        self.assertEqual(mask_result.dim(), 3)
        self.assertEqual(mask_result.shape[0], 1)

        # Other outputs should be normal
        self.assertEqual(img_result.shape[1:], mask_result.shape[1:])
        self.assertEqual(pointmap_result.shape[1:], mask_result.shape[1:])

    def test_crop_around_mask_with_random_box_size_factor(self):
        """Test crop_around_mask_with_random_box_size_factor with pointmap"""
        # Set seed for reproducibility
        torch.manual_seed(42)

        # Test without pointmap first (backward compatibility)
        img_result, mask_result = crop_around_mask_with_random_box_size_factor(
            self.img, self.object_mask, random_box_size_factor=0.5
        )
        self.assertEqual(len([img_result, mask_result]), 2)

        # Test with pointmap
        torch.manual_seed(42)  # Reset seed for consistency
        img_result, mask_result, pointmap_result = (
            crop_around_mask_with_random_box_size_factor(
                self.img,
                self.object_mask,
                random_box_size_factor=0.5,
                pointmap=self.pointmap,
            )
        )

        # Check outputs
        self.assertEqual(img_result.shape[1:], mask_result.shape)
        self.assertEqual(img_result.shape[1:], pointmap_result.shape[1:])
        self.assertEqual(img_result.shape[1], img_result.shape[2])  # Square

    def test_empty_mask_handling(self):
        """Test handling of empty masks"""
        empty_mask = torch.zeros(10, 10, device=self.device)

        # Should handle empty mask gracefully
        img_result, mask_result, pointmap_result = crop_around_mask_with_padding(
            self.img,
            empty_mask,
            box_size_factor=1.2,
            padding_factor=0.1,
            pointmap=self.pointmap,
        )

        # Should return valid tensors even with empty mask
        self.assertIsInstance(img_result, torch.Tensor)
        self.assertIsInstance(mask_result, torch.Tensor)
        self.assertIsInstance(pointmap_result, torch.Tensor)

    def test_pointmap_spatial_consistency(self):
        """Test that spatial transformations are applied consistently"""
        # Create a pointmap with a specific pattern
        test_pointmap = torch.zeros(3, 10, 10, device=self.device)
        test_pointmap[0, 5, 5] = 99.0  # Mark center position

        # Create a mask that includes the marked position
        test_mask = torch.zeros(10, 10, device=self.device)
        test_mask[3:8, 3:8] = 1.0  # Ensure marked position is within object

        # Apply transform
        _, _, pointmap_result = crop_around_mask_with_padding(
            self.img,
            test_mask,
            box_size_factor=1.5,  # Larger box to ensure marker is included
            padding_factor=0.1,
            pointmap=test_pointmap,
        )

        # The marked position should still exist in the result (not NaN)
        has_marker = (pointmap_result[0] == 99.0).any()
        self.assertTrue(
            has_marker.item() if hasattr(has_marker, "item") else has_marker
        )

    def test_different_pointmap_shapes(self):
        """Test with different pointmap channel counts"""
        # Test with 1-channel pointmap (e.g., depth only)
        pointmap_1ch = torch.rand(1, 10, 10, device=self.device)

        img_result, mask_result, pointmap_result = crop_around_mask_with_padding(
            self.img,
            self.object_mask,
            box_size_factor=1.2,
            padding_factor=0.1,
            pointmap=pointmap_1ch,
        )

        self.assertEqual(pointmap_result.shape[0], 1)
        self.assertEqual(img_result.shape[1:], pointmap_result.shape[1:])

        # Test with 6-channel pointmap (e.g., x,y,z + normals)
        pointmap_6ch = torch.rand(6, 10, 10, device=self.device)

        img_result, mask_result, pointmap_result = crop_around_mask_with_padding(
            self.img,
            self.object_mask,
            box_size_factor=1.2,
            padding_factor=0.1,
            pointmap=pointmap_6ch,
        )

        self.assertEqual(pointmap_result.shape[0], 6)
        self.assertEqual(img_result.shape[1:], pointmap_result.shape[1:])


class TestResizeAllToSameSize(unittest.TestCase):
    """Test resize_all_to_same_size function"""

    def setUp(self):
        """Set up test data with different resolutions"""
        self.device = "cpu"

        # Create RGB image at high resolution (similar to real R3 data)
        self.rgb_high_res = torch.rand(3, 1500, 2000, device=self.device)
        self.mask_high_res = torch.ones(1500, 2000, device=self.device)

        # Create pointmap at low resolution (1/4 scale, similar to MoGe output)
        self.pointmap_low_res = torch.randn(3, 375, 500, device=self.device)
        # Add some NaN values to test NaN handling
        self.pointmap_low_res[:, :50, :] = float("nan")
        self.pointmap_low_res[:, :, -50:] = float("nan")

    def test_resize_to_rgb_size_default(self):
        """Test resizing with default target size (RGB image size)"""
        rgb_result, mask_result, pointmap_result = resize_all_to_same_size(
            self.rgb_high_res, self.mask_high_res, self.pointmap_low_res
        )

        # All outputs should have the same spatial dimensions
        self.assertEqual(rgb_result.shape[1:], (1500, 2000))
        self.assertEqual(mask_result.shape, (1500, 2000))
        self.assertEqual(pointmap_result.shape[1:], (1500, 2000))

        # RGB and mask should be unchanged since they're already at target size
        self.assertTrue(torch.equal(rgb_result, self.rgb_high_res))
        self.assertTrue(torch.equal(mask_result, self.mask_high_res))

    def test_resize_with_custom_target_size(self):
        """Test resizing with custom target size"""
        target_size = (768, 1024)
        rgb_result, mask_result, pointmap_result = resize_all_to_same_size(
            self.rgb_high_res,
            self.mask_high_res,
            self.pointmap_low_res,
            target_size=target_size,
        )

        # All outputs should match target size
        self.assertEqual(rgb_result.shape[1:], target_size)
        self.assertEqual(mask_result.shape, target_size)
        self.assertEqual(pointmap_result.shape[1:], target_size)

    def test_nan_preservation(self):
        """Test that NaN regions are preserved during resize"""
        rgb_result, mask_result, pointmap_result = resize_all_to_same_size(
            self.rgb_high_res, self.mask_high_res, self.pointmap_low_res
        )

        # Check that we still have NaN values after resize
        self.assertTrue(torch.isnan(pointmap_result).any())

        # Check that NaN regions are approximately preserved
        # Top region should still have many NaNs
        top_region_nan_ratio = torch.isnan(pointmap_result[:, :200, :]).float().mean()
        self.assertGreater(top_region_nan_ratio, 0.5)

        # Middle region should have fewer NaNs
        middle_region_nan_ratio = (
            torch.isnan(pointmap_result[:, 700:800, :]).float().mean()
        )
        self.assertLess(
            middle_region_nan_ratio, 0.15
        )  # Allow some tolerance for interpolation

    def test_no_pointmap(self):
        """Test function works without pointmap (backward compatibility)"""
        rgb_result, mask_result = resize_all_to_same_size(
            self.rgb_high_res, self.mask_high_res
        )

        # Should return only two values
        self.assertEqual(rgb_result.shape, self.rgb_high_res.shape)
        self.assertEqual(mask_result.shape, self.mask_high_res.shape)

    def test_mask_with_channel_dimension(self):
        """Test handling of mask with channel dimension"""
        mask_with_channel = self.mask_high_res.unsqueeze(0)

        rgb_result, mask_result, pointmap_result = resize_all_to_same_size(
            self.rgb_high_res, mask_with_channel, self.pointmap_low_res
        )

        # Mask should have channel dimension preserved
        self.assertEqual(mask_result.shape, mask_with_channel.shape)
        self.assertEqual(mask_result.shape[0], 1)

    def test_same_size_inputs(self):
        """Test when all inputs are already the same size"""
        # Create same-size inputs
        size = (512, 512)
        rgb_same = torch.rand(3, *size, device=self.device)
        mask_same = torch.ones(*size, device=self.device)
        pointmap_same = torch.randn(3, *size, device=self.device)

        rgb_result, mask_result, pointmap_result = resize_all_to_same_size(
            rgb_same, mask_same, pointmap_same
        )

        # Should return inputs unchanged
        self.assertTrue(torch.equal(rgb_result, rgb_same))
        self.assertTrue(torch.equal(mask_result, mask_same))
        self.assertTrue(torch.equal(pointmap_result, pointmap_same))


if __name__ == "__main__":
    unittest.main()
