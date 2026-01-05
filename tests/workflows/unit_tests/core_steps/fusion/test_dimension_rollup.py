"""
Comprehensive test cases for dimension_rollup_v1 block.

Tests all combinations of detection types (object detection, segmentation, keypoint)
and parameter variations (confidence strategies, overlap thresholds, keypoint merge thresholds).
"""

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.fusion.detections_list_rollup.v1 import (
    BlockManifest,
    DetectionsListRollUpBlockV1,
)

# ============================================================================
# Fixtures and Helpers
# ============================================================================


def create_parent_detections(
    xyxy_list: list,
    parent_image_shape: tuple = (480, 640),
) -> sv.Detections:
    """Create parent detections with required metadata for dimension rollup."""
    detections = sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        confidence=np.array([0.9] * len(xyxy_list), dtype=np.float32),
        class_id=np.array([0] * len(xyxy_list), dtype=int),
    )
    # Add required metadata
    detections.data["root_parent_dimensions"] = [parent_image_shape] * len(xyxy_list)
    return detections


def create_child_detections(
    xyxy_list: list,
    confidences: list = None,
    class_ids: list = None,
) -> sv.Detections:
    """Create child detections as sv.Detections object for a crop."""
    if not xyxy_list:
        # Return empty detections
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
            class_id=np.empty(0, dtype=int),
        )

    xyxy_array = np.array(xyxy_list, dtype=np.float32)
    if confidences is None:
        confidences = [0.9] * len(xyxy_list)
    if class_ids is None:
        class_ids = [0] * len(xyxy_list)

    return sv.Detections(
        xyxy=xyxy_array,
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )


# ============================================================================
# BlockManifest Tests
# ============================================================================


class TestBlockManifestValidation:
    """Test BlockManifest parsing and validation."""

    def test_manifest_with_defaults(self) -> None:
        """Test manifest parsing with default parameter values."""
        raw_manifest = {
            "type": "roboflow_core/detections_list_rollup@v1",
            "name": "rollup",
            "parent_detection": "$inputs.parent",
            "child_detections": "$inputs.children",
        }

        result = BlockManifest.model_validate(raw_manifest)

        assert result.type == "roboflow_core/detections_list_rollup@v1"
        assert result.name == "rollup"
        assert result.confidence_strategy == "max"
        assert result.overlap_threshold == 0.0
        assert result.keypoint_merge_threshold == 10

    @pytest.mark.parametrize(
        "confidence_strategy",
        ["max", "mean", "min", "$inputs.strategy"],
    )
    def test_manifest_with_confidence_strategy(self, confidence_strategy: str) -> None:
        """Test manifest with different confidence strategies."""
        raw_manifest = {
            "type": "roboflow_core/detections_list_rollup@v1",
            "name": "rollup",
            "parent_detection": "$inputs.parent",
            "child_detections": "$inputs.children",
            "confidence_strategy": confidence_strategy,
        }

        result = BlockManifest.model_validate(raw_manifest)
        assert result.confidence_strategy == confidence_strategy

    @pytest.mark.parametrize(
        "overlap_threshold",
        [0.0, 0.3, 0.5, 0.7, 1.0],
    )
    def test_manifest_with_overlap_threshold(self, overlap_threshold) -> None:
        """Test manifest with different overlap thresholds."""
        raw_manifest = {
            "type": "roboflow_core/detections_list_rollup@v1",
            "name": "rollup",
            "parent_detection": "$inputs.parent",
            "child_detections": "$inputs.children",
            "overlap_threshold": overlap_threshold,
        }

        result = BlockManifest.model_validate(raw_manifest)
        assert result.overlap_threshold == overlap_threshold

    @pytest.mark.parametrize(
        "keypoint_threshold",
        [0, 5, 10, 20, 50],
    )
    def test_manifest_with_keypoint_threshold(self, keypoint_threshold) -> None:
        """Test manifest with different keypoint thresholds."""
        raw_manifest = {
            "type": "roboflow_core/detections_list_rollup@v1",
            "name": "rollup",
            "parent_detection": "$inputs.parent",
            "child_detections": "$inputs.children",
            "keypoint_merge_threshold": keypoint_threshold,
        }

        result = BlockManifest.model_validate(raw_manifest)
        assert result.keypoint_merge_threshold == keypoint_threshold


# ============================================================================
# Block Execution Tests - Object Detection
# ============================================================================


class TestDimensionRollupObjectDetection:
    """Test dimension rollup with object detection predictions."""

    def test_simple_object_detection_no_overlap(self) -> None:
        """Test rollup with simple non-overlapping object detections."""
        block = DetectionsListRollUpBlockV1()
        parent = create_parent_detections([[0, 0, 100, 100]])

        child_detections = [
            create_child_detections(
                [[10, 10, 50, 50], [60, 60, 90, 90]],
                [0.85, 0.88],
                [0, 1],
            )
        ]

        result = block.run(
            parent_detection=parent,
            child_detections=child_detections,
            confidence_strategy="max",
            overlap_threshold=0.0,
            keypoint_merge_threshold=10,
        )

        assert "rolled_up_detections" in result
        assert "crop_zones" in result

    @pytest.mark.parametrize("strategy", ["max", "mean", "min"])
    def test_confidence_strategies(self, strategy: str) -> None:
        """Test all confidence merging strategies."""
        block = DetectionsListRollUpBlockV1()
        parent = create_parent_detections([[0, 0, 100, 100]])

        # Overlapping detections with different confidences
        child_detections = [
            create_child_detections(
                [[10, 10, 60, 60], [30, 30, 80, 80]],
                [0.7, 0.95],
                [0, 0],
            )
        ]

        result = block.run(
            parent_detection=parent,
            child_detections=child_detections,
            confidence_strategy=strategy,
            overlap_threshold=0.0,
            keypoint_merge_threshold=10,
        )

        rolled_up = result["rolled_up_detections"]
        assert len(rolled_up) >= 1

    @pytest.mark.parametrize("threshold", [0.0, 0.3, 0.5, 0.7])
    def test_overlap_thresholds(self, threshold: float) -> None:
        """Test different IoU thresholds for merging."""
        block = DetectionsListRollUpBlockV1()
        parent = create_parent_detections([[0, 0, 100, 100]])

        # Detections with varying overlaps
        child_detections = [
            create_child_detections(
                [[10, 10, 60, 60], [40, 40, 90, 90]],
                [0.8, 0.85],
                [0, 0],
            )
        ]

        result = block.run(
            parent_detection=parent,
            child_detections=child_detections,
            confidence_strategy="max",
            overlap_threshold=threshold,
            keypoint_merge_threshold=10,
        )

        rolled_up = result["rolled_up_detections"]
        assert len(rolled_up) >= 1


# ============================================================================
# Block Execution Tests - Segmentation
# ============================================================================


class TestDimensionRollupSegmentation:
    """Test dimension rollup with instance segmentation predictions."""

    def test_simple_segmentation(self) -> None:
        """Test rollup with segmentation masks."""
        block = DetectionsListRollUpBlockV1()
        parent = create_parent_detections([[0, 0, 100, 100]])

        # Create child detections with masks
        child_dets = create_child_detections(
            [[10, 10, 50, 50], [60, 60, 90, 90]],
            [0.85, 0.88],
            [0, 0],
        )

        # Add masks to the detections
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[15:45, 15:45] = True
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[60:85, 60:85] = True

        child_dets.mask = np.array([mask1, mask2])

        child_detections = [child_dets]

        result = block.run(
            parent_detection=parent,
            child_detections=child_detections,
            confidence_strategy="max",
            overlap_threshold=0.0,
            keypoint_merge_threshold=10,
        )

        assert "rolled_up_detections" in result
        rolled_up = result["rolled_up_detections"]
        assert len(rolled_up) >= 1


# ============================================================================
# Block Execution Tests - Keypoints
# ============================================================================


class TestDimensionRollupKeypoints:
    """Test dimension rollup with keypoint predictions."""

    def test_simple_keypoint_rollup(self) -> None:
        """Test rollup with keypoint detections."""
        block = DetectionsListRollUpBlockV1()
        parent = create_parent_detections([[0, 0, 100, 100]])

        keypoints1 = [[20, 25], [15, 35], [5, 35], [15, 60], [5, 60]]
        keypoints2 = [[75, 80], [70, 90], [60, 90], [70, 110], [60, 110]]

        child_dets = create_child_detections(
            [[10, 10, 50, 50], [60, 60, 90, 90]],
            [0.85, 0.88],
            [0, 0],
        )

        # Add keypoint data
        child_dets.data["keypoints_xy"] = np.array(
            [keypoints1, keypoints2], dtype=object
        )
        child_dets.data["keypoint_confidence"] = np.array(
            [[0.95] * len(keypoints1), [0.95] * len(keypoints2)], dtype=object
        )
        child_dets.data["prediction_type"] = np.array(
            ["keypoint-detection", "keypoint-detection"]
        )

        child_detections = [child_dets]

        result = block.run(
            parent_detection=parent,
            child_detections=child_detections,
            confidence_strategy="max",
            overlap_threshold=0.0,
            keypoint_merge_threshold=10,
        )

        assert "rolled_up_detections" in result
        assert len(result["rolled_up_detections"]) >= 1

    @pytest.mark.parametrize("threshold", [5, 10, 20, 50])
    def test_keypoint_merge_thresholds(self, threshold: float) -> None:
        """Test keypoint merging with different distance thresholds."""
        block = DetectionsListRollUpBlockV1()
        parent = create_parent_detections([[0, 0, 100, 100]])

        # Keypoints at varying distances
        keypoints1 = [[20, 25], [15, 35], [5, 35], [15, 60], [5, 60]]
        keypoints2 = [[25, 30], [20, 40], [10, 40], [20, 65], [10, 65]]

        child_dets = create_child_detections(
            [[10, 10, 50, 50], [15, 15, 55, 55]],
            [0.85, 0.88],
            [0, 0],
        )

        # Add keypoint data
        child_dets.data["keypoints_xy"] = np.array(
            [keypoints1, keypoints2], dtype=object
        )
        child_dets.data["keypoint_confidence"] = np.array(
            [[0.95] * len(keypoints1), [0.95] * len(keypoints2)], dtype=object
        )
        child_dets.data["prediction_type"] = np.array(
            ["keypoint-detection", "keypoint-detection"]
        )

        child_detections = [child_dets]

        result = block.run(
            parent_detection=parent,
            child_detections=child_detections,
            confidence_strategy="max",
            overlap_threshold=0.0,
            keypoint_merge_threshold=threshold,
        )

        rolled_up = result["rolled_up_detections"]
        assert len(rolled_up) >= 1


# ============================================================================
# Edge Cases and Complex Scenarios
# ============================================================================


class TestDimensionRollupEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_child_detections(self) -> None:
        """Test with empty child detections."""
        block = DetectionsListRollUpBlockV1()
        parent = create_parent_detections([[0, 0, 100, 100]])
        child_detections = [create_child_detections([])]

        result = block.run(
            parent_detection=parent,
            child_detections=child_detections,
            confidence_strategy="max",
            overlap_threshold=0.0,
            keypoint_merge_threshold=10,
        )

        assert "rolled_up_detections" in result

    def test_single_detection_per_parent(self) -> None:
        """Test with single child detection per parent."""
        block = DetectionsListRollUpBlockV1()
        parent = create_parent_detections([[0, 0, 100, 100]])
        child_detections = [create_child_detections([[10, 10, 50, 50]], [0.85], [0])]

        result = block.run(
            parent_detection=parent,
            child_detections=child_detections,
            confidence_strategy="max",
            overlap_threshold=0.0,
            keypoint_merge_threshold=10,
        )

        assert len(result["rolled_up_detections"]) == 1

    def test_multiple_parents(self) -> None:
        """Test with multiple parent detections."""
        block = DetectionsListRollUpBlockV1()
        parent = create_parent_detections([[0, 0, 100, 100], [150, 150, 250, 250]])

        child_detections = [
            create_child_detections(
                [[10, 10, 50, 50], [20, 20, 60, 60]],
                [0.85, 0.88],
                [0, 0],
            ),
            create_child_detections(
                [[160, 160, 200, 200]],
                [0.92],
                [1],
            ),
        ]

        result = block.run(
            parent_detection=parent,
            child_detections=child_detections,
            confidence_strategy="max",
            overlap_threshold=0.0,
            keypoint_merge_threshold=10,
        )

        assert len(result["rolled_up_detections"]) >= 2

    def test_many_overlapping_detections(self) -> None:
        """Test with many overlapping detections requiring merging."""
        block = DetectionsListRollUpBlockV1()
        parent = create_parent_detections([[0, 0, 100, 100]])

        # Create 10 overlapping detections
        xyxy_list = [
            [10 + i * 2, 10 + i * 2, 50 + i * 2, 50 + i * 2] for i in range(10)
        ]
        confidences = [0.7 + (i % 5) * 0.05 for i in range(10)]
        class_ids = [0] * 10

        child_detections = [create_child_detections(xyxy_list, confidences, class_ids)]

        result = block.run(
            parent_detection=parent,
            child_detections=child_detections,
            confidence_strategy="mean",
            overlap_threshold=0.3,
            keypoint_merge_threshold=10,
        )

        rolled_up = result["rolled_up_detections"]
        assert len(rolled_up) >= 1
        assert len(rolled_up) < 10  # Should be merged significantly

    def test_multi_class_detections(self) -> None:
        """Test with detections from multiple classes."""
        block = DetectionsListRollUpBlockV1()
        parent = create_parent_detections([[0, 0, 100, 100]])

        child_detections = [
            create_child_detections(
                [[10, 10, 50, 50], [30, 30, 70, 70], [50, 50, 90, 90]],
                [0.85, 0.88, 0.82],
                [0, 1, 2],
            )
        ]

        result = block.run(
            parent_detection=parent,
            child_detections=child_detections,
            confidence_strategy="max",
            overlap_threshold=0.0,
            keypoint_merge_threshold=10,
        )

        rolled_up = result["rolled_up_detections"]
        assert len(rolled_up) >= 1
