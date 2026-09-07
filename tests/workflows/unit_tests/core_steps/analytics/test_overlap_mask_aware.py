"""
Tests for mask-aware overlap detection (Issue #1987)

Tests that the Overlap Filter correctly uses mask information for instance
segmentation, preventing false positives from bbox-only overlap checks.
"""
import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.analytics.overlap.v1 import OverlapBlockV1


class TestMaskOverlapMethod:
    """Test the mask_overlap method directly."""

    def test_center_overlap_true_when_center_inside_mask(self):
        """Test center overlap when other's center is inside overlap mask."""
        # Create overlap mask (covers left half of 100x100 image)
        overlap_mask = np.zeros((100, 100), dtype=np.uint8)
        overlap_mask[:, :50] = 1

        # Create other mask (small region in left half, center at x=25, y=50)
        other_mask = np.zeros((100, 100), dtype=np.uint8)
        other_mask[45:55, 20:30] = 1  # Center at row=50, col=25

        # Center (50, 25) is inside overlap mask
        assert OverlapBlockV1.mask_overlap(
            overlap_mask, other_mask, "Center Overlap"
        )

    def test_center_overlap_false_when_center_outside_mask(self):
        """Test center overlap when other's center is outside overlap mask."""
        # Create overlap mask (covers left half)
        overlap_mask = np.zeros((100, 100), dtype=np.uint8)
        overlap_mask[:, :50] = 1

        # Create other mask (small region in right half, center at x=75, y=50)
        other_mask = np.zeros((100, 100), dtype=np.uint8)
        other_mask[45:55, 70:80] = 1  # Center at row=50, col=75

        # Center (50, 75) is outside overlap mask
        assert not OverlapBlockV1.mask_overlap(
            overlap_mask, other_mask, "Center Overlap"
        )

    def test_center_overlap_false_for_bbox_overlap_but_no_mask_overlap(self):
        """
        REGRESSION TEST for Issue #1987:
        Test that bbox overlap doesn't cause false positive when masks don't overlap.

        This is the key bug fix - bboxes can overlap while masks don't.
        """
        # Overlap mask: L-shaped, covers bottom-left quadrant
        overlap_mask = np.zeros((100, 100), dtype=np.uint8)
        overlap_mask[50:100, 0:50] = 1  # Bottom-left quadrant

        # Other mask: top-right region
        # Bbox: [60, 10, 90, 40] - this OVERLAPS with overlap bbox [0, 50, 50, 100]
        # But masks have NO overlap
        other_mask = np.zeros((100, 100), dtype=np.uint8)
        other_mask[10:40, 60:90] = 1  # Top-right

        # Center of other is at (25, 75) - outside overlap mask
        assert not OverlapBlockV1.mask_overlap(
            overlap_mask, other_mask, "Center Overlap"
        )

    def test_any_overlap_true_when_masks_intersect(self):
        """Test any overlap when masks have pixel intersection."""
        # Create overlap mask
        overlap_mask = np.zeros((100, 100), dtype=np.uint8)
        overlap_mask[40:60, 40:60] = 1

        # Create other mask that partially overlaps
        other_mask = np.zeros((100, 100), dtype=np.uint8)
        other_mask[50:70, 50:70] = 1

        # Masks have 10x10 pixel intersection
        assert OverlapBlockV1.mask_overlap(overlap_mask, other_mask, "Any Overlap")

    def test_any_overlap_false_when_masks_dont_intersect(self):
        """Test any overlap when masks have no pixel intersection."""
        # Create overlap mask
        overlap_mask = np.zeros((100, 100), dtype=np.uint8)
        overlap_mask[10:30, 10:30] = 1

        # Create other mask (no intersection)
        other_mask = np.zeros((100, 100), dtype=np.uint8)
        other_mask[70:90, 70:90] = 1

        # No pixel intersection
        assert not OverlapBlockV1.mask_overlap(
            overlap_mask, other_mask, "Any Overlap"
        )

    def test_any_overlap_false_for_irregular_masks_with_bbox_overlap(self):
        """
        REGRESSION TEST for Issue #1987:
        Test irregular masks where bboxes overlap but masks don't.
        """
        # Container mask: donut shape (hollow center)
        overlap_mask = np.zeros((100, 100), dtype=np.uint8)
        overlap_mask[0:100, 0:100] = 1  # Fill entire bbox
        overlap_mask[25:75, 25:75] = 0  # Hollow out center

        # Item mask: only in the hollow center
        other_mask = np.zeros((100, 100), dtype=np.uint8)
        other_mask[40:60, 40:60] = 1

        # Bboxes overlap (both are [0,0,100,100]) but masks don't intersect
        assert not OverlapBlockV1.mask_overlap(
            overlap_mask, other_mask, "Any Overlap"
        )

    def test_empty_mask_returns_false(self):
        """Test that empty masks return False."""
        overlap_mask = np.zeros((100, 100), dtype=np.uint8)
        overlap_mask[40:60, 40:60] = 1

        empty_mask = np.zeros((100, 100), dtype=np.uint8)

        assert not OverlapBlockV1.mask_overlap(
            overlap_mask, empty_mask, "Center Overlap"
        )
        assert not OverlapBlockV1.mask_overlap(
            overlap_mask, empty_mask, "Any Overlap"
        )


class TestOverlapBlockMaskAware:
    """Test the full OverlapBlockV1.run() with mask awareness."""

    def test_mask_aware_overlap_prevents_false_positive(self):
        """
        MAIN REGRESSION TEST for Issue #1987:
        Test that mask-aware overlap prevents false positives from bbox overlap.

        Setup:
        - Container (overlap class): L-shaped mask, bbox covers whole image
        - Item: top-right corner

        Expected:
        - OLD (bbox-only): Item overlaps (bbox intersection)
        - NEW (mask-aware): Item doesn't overlap (no mask intersection)
        """
        # Create container mask (L-shaped, bottom-left)
        container_mask = np.zeros((100, 100), dtype=np.uint8)
        container_mask[50:100, 0:100] = 1  # Bottom row
        container_mask[0:50, 0:50] = 1     # Top-left quadrant

        # Create item mask (top-right corner)
        item_mask = np.zeros((100, 100), dtype=np.uint8)
        item_mask[10:40, 60:90] = 1

        # Create predictions with masks
        predictions = sv.Detections(
            xyxy=np.array([[0, 0, 100, 100], [60, 10, 90, 40]]),  # Bboxes overlap!
            mask=np.array([container_mask, item_mask]),
            class_id=np.array([0, 1]),
            confidence=np.array([0.9, 0.8]),
            data={"class_name": np.array(["container", "item"])},
        )

        block = OverlapBlockV1()

        # Test "Any Overlap"
        result = block.run(
            predictions=predictions,
            overlap_type="Any Overlap",
            overlap_class_name="container",
        )

        # Should return EMPTY (no mask overlap despite bbox overlap)
        assert len(result["overlaps"]) == 0

    def test_mask_aware_overlap_detects_true_positive(self):
        """Test that mask-aware overlap correctly detects actual overlap."""
        # Create container mask
        container_mask = np.zeros((100, 100), dtype=np.uint8)
        container_mask[20:80, 20:80] = 1

        # Create item mask that DOES overlap
        item_mask = np.zeros((100, 100), dtype=np.uint8)
        item_mask[40:60, 40:60] = 1  # Inside container

        predictions = sv.Detections(
            xyxy=np.array([[20, 20, 80, 80], [40, 40, 60, 60]]),
            mask=np.array([container_mask, item_mask]),
            class_id=np.array([0, 1]),
            confidence=np.array([0.9, 0.8]),
            data={"class_name": np.array(["container", "item"])},
        )

        block = OverlapBlockV1()

        result = block.run(
            predictions=predictions,
            overlap_type="Any Overlap",
            overlap_class_name="container",
        )

        # Should return item (masks DO overlap)
        assert len(result["overlaps"]) == 1
        assert result["overlaps"].data["class_name"][0] == "item"

    def test_center_overlap_with_masks(self):
        """Test center overlap mode with mask-aware detection."""
        # Container mask (large square)
        container_mask = np.zeros((100, 100), dtype=np.uint8)
        container_mask[10:90, 10:90] = 1

        # Item mask with center at (50, 50) - inside container
        item_mask = np.zeros((100, 100), dtype=np.uint8)
        item_mask[45:55, 45:55] = 1

        predictions = sv.Detections(
            xyxy=np.array([[10, 10, 90, 90], [45, 45, 55, 55]]),
            mask=np.array([container_mask, item_mask]),
            class_id=np.array([0, 1]),
            confidence=np.array([0.9, 0.8]),
            data={"class_name": np.array(["container", "item"])},
        )

        block = OverlapBlockV1()

        result = block.run(
            predictions=predictions,
            overlap_type="Center Overlap",
            overlap_class_name="container",
        )

        # Should return item (center is inside container mask)
        assert len(result["overlaps"]) == 1

    def test_fallback_to_bbox_when_no_masks(self):
        """Test that bbox mode still works when masks are not available."""
        # Create predictions WITHOUT masks (object detection)
        predictions = sv.Detections(
            xyxy=np.array([[10, 10, 50, 50], [30, 30, 60, 60]]),
            class_id=np.array([0, 1]),
            confidence=np.array([0.9, 0.8]),
            data={"class_name": np.array(["container", "item"])},
        )

        block = OverlapBlockV1()

        result = block.run(
            predictions=predictions,
            overlap_type="Any Overlap",
            overlap_class_name="container",
        )

        # Should use bbox overlap (bboxes DO overlap)
        assert len(result["overlaps"]) == 1

    def test_multiple_containers_with_irregular_masks(self):
        """Test multiple overlap class instances with irregular masks."""
        # Container 1: left half
        container1_mask = np.zeros((100, 100), dtype=np.uint8)
        container1_mask[:, 0:50] = 1

        # Container 2: right half
        container2_mask = np.zeros((100, 100), dtype=np.uint8)
        container2_mask[:, 50:100] = 1

        # Item 1: in left half
        item1_mask = np.zeros((100, 100), dtype=np.uint8)
        item1_mask[40:60, 20:30] = 1

        # Item 2: in right half
        item2_mask = np.zeros((100, 100), dtype=np.uint8)
        item2_mask[40:60, 70:80] = 1

        # Item 3: in gap between containers (should not overlap)
        item3_mask = np.zeros((100, 100), dtype=np.uint8)
        item3_mask[40:60, 48:52] = 1  # Right on the boundary

        predictions = sv.Detections(
            xyxy=np.array([
                [0, 0, 50, 100],    # container1
                [50, 0, 100, 100],  # container2
                [20, 40, 30, 60],   # item1
                [70, 40, 80, 60],   # item2
                [48, 40, 52, 60],   # item3
            ]),
            mask=np.array([
                container1_mask,
                container2_mask,
                item1_mask,
                item2_mask,
                item3_mask,
            ]),
            class_id=np.array([0, 0, 1, 1, 1]),
            confidence=np.array([0.9, 0.9, 0.8, 0.8, 0.8]),
            data={"class_name": np.array(["container", "container", "item", "item", "item"])},
        )

        block = OverlapBlockV1()

        result = block.run(
            predictions=predictions,
            overlap_type="Any Overlap",
            overlap_class_name="container",
        )

        # Should return item1 and item2 (inside containers)
        # Should NOT return item3 (in the gap)
        assert len(result["overlaps"]) == 2
        class_names = set(result["overlaps"].data["class_name"])
        assert class_names == {"item"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
