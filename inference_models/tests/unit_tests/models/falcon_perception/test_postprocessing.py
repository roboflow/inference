"""Unit tests for Falcon Perception post-processing functions.

Tests bin-to-coordinate conversion, log-scale size decoding,
confidence computation, and result-to-detections conversion.
"""

import math

import pytest
import torch

from inference_models.models.falcon_perception.config import FalconPerceptionConfig
from inference_models.models.falcon_perception.engine import (
    BatchInferenceResult,
    InstancePrediction,
    QueryResult,
    boxes_from_instances,
    decode_coord_bins,
    decode_size_bins,
    instance_confidence,
)
from inference_models.models.falcon_perception.postprocessing import (
    result_to_detections,
)
from inference_models.models.falcon_perception.preprocessing import ImageMetadata


@pytest.fixture
def config():
    return FalconPerceptionConfig()


@pytest.fixture
def image_metadata():
    return ImageMetadata(
        original_height=480,
        original_width=640,
        resized_height=480,
        resized_width=640,
        h_patches=30,
        w_patches=40,
        pad_h=0,
        pad_w=0,
    )


class TestDecodeCoordBins:
    def test_center_bin_gives_center_of_image(self):
        """Bin 511 (middle) should map approximately to image center."""
        cx, cy = decode_coord_bins(511, 511, 640, 480, num_bins=1024)
        assert abs(cx - 320) < 1.0
        assert abs(cy - 240) < 1.0

    def test_zero_bin_gives_origin(self):
        """Bin 0 should map to (0, 0)."""
        cx, cy = decode_coord_bins(0, 0, 640, 480, num_bins=1024)
        assert cx == 0.0
        assert cy == 0.0

    def test_max_bin_gives_image_extent(self):
        """Bin 1023 should map to (width, height)."""
        cx, cy = decode_coord_bins(1023, 1023, 640, 480, num_bins=1024)
        assert abs(cx - 640) < 0.001
        assert abs(cy - 480) < 0.001

    def test_quarter_bin_gives_quarter_position(self):
        """Approximately bin 256 should give ~1/4 of image dimensions."""
        cx, cy = decode_coord_bins(256, 256, 640, 480, num_bins=1024)
        expected_x = 256 / 1023 * 640
        expected_y = 256 / 1023 * 480
        assert abs(cx - expected_x) < 0.001
        assert abs(cy - expected_y) < 0.001

    def test_different_image_sizes(self):
        """Coordinate decoding should scale with image dimensions."""
        cx1, cy1 = decode_coord_bins(512, 512, 1920, 1080, num_bins=1024)
        cx2, cy2 = decode_coord_bins(512, 512, 640, 480, num_bins=1024)
        assert cx1 > cx2  # Larger image -> larger pixel coordinates
        assert cy1 > cy2


class TestDecodeSizeBins:
    def test_max_bin_gives_full_image_size(self):
        """Bin 1023 (max) should give full image size."""
        w, h = decode_size_bins(1023, 1023, 640, 480, num_bins=1024, log2_range=10.0)
        assert abs(w - 640) < 0.001
        assert abs(h - 480) < 0.001

    def test_zero_bin_gives_minimum_size(self):
        """Bin 0 should give minimum (near-zero) size."""
        w, h = decode_size_bins(0, 0, 640, 480, num_bins=1024, log2_range=10.0)
        # At bin 0: 2^0 / 2^10 = 1/1024
        expected_w = (1.0 / 1024.0) * 640
        expected_h = (1.0 / 1024.0) * 480
        assert abs(w - expected_w) < 0.001
        assert abs(h - expected_h) < 0.001

    def test_size_increases_monotonically_with_bin(self):
        """Size should increase as bin index increases."""
        sizes = []
        for b in range(0, 1024, 100):
            w, _ = decode_size_bins(b, 0, 640, 480, num_bins=1024, log2_range=10.0)
            sizes.append(w)
        assert all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1))

    def test_log_scale_produces_fine_resolution_at_small_sizes(self):
        """Log scale should give finer resolution at smaller sizes."""
        # Size difference between bins 0 and 1
        w0, _ = decode_size_bins(0, 0, 640, 480, num_bins=1024, log2_range=10.0)
        w1, _ = decode_size_bins(1, 0, 640, 480, num_bins=1024, log2_range=10.0)
        small_diff = w1 - w0

        # Size difference between bins 1022 and 1023
        w1022, _ = decode_size_bins(1022, 0, 640, 480, num_bins=1024, log2_range=10.0)
        w1023, _ = decode_size_bins(1023, 0, 640, 480, num_bins=1024, log2_range=10.0)
        large_diff = w1023 - w1022

        # Large bins should have bigger absolute differences (log scale)
        assert large_diff > small_diff


class TestInstanceConfidence:
    def test_perfect_confidence(self):
        """All confidences at 1.0 should give 1.0."""
        inst = InstancePrediction(
            center_x_bin=0,
            center_y_bin=0,
            width_bin=0,
            height_bin=0,
            coord_x_confidence=1.0,
            coord_y_confidence=1.0,
            size_w_confidence=1.0,
            size_h_confidence=1.0,
        )
        assert abs(instance_confidence(inst) - 1.0) < 1e-6

    def test_zero_confidence(self):
        """Any zero confidence should give 0.0."""
        inst = InstancePrediction(
            center_x_bin=0,
            center_y_bin=0,
            width_bin=0,
            height_bin=0,
            coord_x_confidence=0.0,
            coord_y_confidence=0.5,
            size_w_confidence=0.5,
            size_h_confidence=0.5,
        )
        assert instance_confidence(inst) == 0.0

    def test_geometric_mean(self):
        """Confidence should be geometric mean of four values."""
        inst = InstancePrediction(
            center_x_bin=0,
            center_y_bin=0,
            width_bin=0,
            height_bin=0,
            coord_x_confidence=0.8,
            coord_y_confidence=0.6,
            size_w_confidence=0.9,
            size_h_confidence=0.7,
        )
        expected = (0.8 * 0.6 * 0.9 * 0.7) ** 0.25
        assert abs(instance_confidence(inst) - expected) < 1e-6

    def test_symmetric(self):
        """Order of values shouldn't matter for geometric mean."""
        inst1 = InstancePrediction(
            center_x_bin=0, center_y_bin=0, width_bin=0, height_bin=0,
            coord_x_confidence=0.3, coord_y_confidence=0.9,
            size_w_confidence=0.5, size_h_confidence=0.7,
        )
        inst2 = InstancePrediction(
            center_x_bin=0, center_y_bin=0, width_bin=0, height_bin=0,
            coord_x_confidence=0.9, coord_y_confidence=0.7,
            size_w_confidence=0.3, size_h_confidence=0.5,
        )
        assert abs(instance_confidence(inst1) - instance_confidence(inst2)) < 1e-6


class TestBoxesFromInstances:
    def test_center_box(self, config):
        """Instance at image center with medium size."""
        inst = InstancePrediction(
            center_x_bin=511, center_y_bin=511,
            width_bin=800, height_bin=800,
            coord_x_confidence=0.9, coord_y_confidence=0.9,
            size_w_confidence=0.9, size_h_confidence=0.9,
        )
        boxes = boxes_from_instances([inst], 640, 480, config)
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]
        # Box should be roughly centered and within image bounds
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 640
        assert y2 <= 480
        assert x2 > x1
        assert y2 > y1

    def test_clipping_to_image_bounds(self, config):
        """Large boxes should be clipped to image boundaries."""
        inst = InstancePrediction(
            center_x_bin=0, center_y_bin=0,
            width_bin=1023, height_bin=1023,
            coord_x_confidence=0.9, coord_y_confidence=0.9,
            size_w_confidence=0.9, size_h_confidence=0.9,
        )
        boxes = boxes_from_instances([inst], 640, 480, config)
        x1, y1, x2, y2 = boxes[0]
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 640
        assert y2 <= 480

    def test_multiple_instances(self, config):
        """Multiple instances produce multiple boxes."""
        instances = [
            InstancePrediction(
                center_x_bin=200, center_y_bin=200,
                width_bin=500, height_bin=500,
                coord_x_confidence=0.9, coord_y_confidence=0.9,
                size_w_confidence=0.9, size_h_confidence=0.9,
            ),
            InstancePrediction(
                center_x_bin=700, center_y_bin=700,
                width_bin=500, height_bin=500,
                coord_x_confidence=0.8, coord_y_confidence=0.8,
                size_w_confidence=0.8, size_h_confidence=0.8,
            ),
        ]
        boxes = boxes_from_instances(instances, 640, 480, config)
        assert len(boxes) == 2

    def test_empty_instances(self, config):
        """No instances should return empty list."""
        boxes = boxes_from_instances([], 640, 480, config)
        assert len(boxes) == 0


class TestResultToDetections:
    def test_empty_result_no_detections(self, config, image_metadata):
        """All absent queries should produce empty detections."""
        result = BatchInferenceResult(
            query_results=[
                QueryResult(prompt="cat", present=False, presence_confidence=0.1),
                QueryResult(prompt="dog", present=False, presence_confidence=0.2),
            ]
        )
        detections = result_to_detections(
            result, image_metadata, config, ["cat", "dog"]
        )
        assert detections.xyxy.shape[0] == 0
        assert detections.class_id.shape[0] == 0
        assert detections.confidence.shape[0] == 0
        assert detections.image_metadata["class_names"] == ["cat", "dog"]

    def test_single_detection(self, config, image_metadata):
        """Single present query with one instance."""
        result = BatchInferenceResult(
            query_results=[
                QueryResult(
                    prompt="cat",
                    present=True,
                    presence_confidence=0.95,
                    instances=[
                        InstancePrediction(
                            center_x_bin=320, center_y_bin=240,
                            width_bin=600, height_bin=600,
                            coord_x_confidence=0.9, coord_y_confidence=0.85,
                            size_w_confidence=0.88, size_h_confidence=0.87,
                        ),
                    ],
                ),
            ]
        )
        detections = result_to_detections(
            result, image_metadata, config, ["cat"]
        )
        assert detections.xyxy.shape == (1, 4)
        assert detections.class_id.shape == (1,)
        assert detections.confidence.shape == (1,)
        assert detections.class_id[0] == 0  # First prompt -> class_id 0
        assert detections.confidence[0] > 0

    def test_multi_query_multi_instance(self, config, image_metadata):
        """Multiple queries, some with multiple instances."""
        result = BatchInferenceResult(
            query_results=[
                QueryResult(
                    prompt="cat",
                    present=True,
                    presence_confidence=0.95,
                    instances=[
                        InstancePrediction(
                            center_x_bin=200, center_y_bin=200,
                            width_bin=500, height_bin=500,
                            coord_x_confidence=0.9, coord_y_confidence=0.9,
                            size_w_confidence=0.9, size_h_confidence=0.9,
                        ),
                        InstancePrediction(
                            center_x_bin=600, center_y_bin=300,
                            width_bin=400, height_bin=400,
                            coord_x_confidence=0.8, coord_y_confidence=0.8,
                            size_w_confidence=0.8, size_h_confidence=0.8,
                        ),
                    ],
                ),
                QueryResult(
                    prompt="dog",
                    present=False,
                    presence_confidence=0.1,
                ),
                QueryResult(
                    prompt="person",
                    present=True,
                    presence_confidence=0.9,
                    instances=[
                        InstancePrediction(
                            center_x_bin=400, center_y_bin=300,
                            width_bin=500, height_bin=700,
                            coord_x_confidence=0.85, coord_y_confidence=0.82,
                            size_w_confidence=0.88, size_h_confidence=0.86,
                        ),
                    ],
                ),
            ]
        )
        prompts = ["cat", "dog", "person"]
        detections = result_to_detections(
            result, image_metadata, config, prompts
        )
        # 2 cats + 0 dogs + 1 person = 3 detections
        assert detections.xyxy.shape[0] == 3
        assert detections.class_id[0] == 0  # cat
        assert detections.class_id[1] == 0  # cat
        assert detections.class_id[2] == 2  # person (index 2 in prompts)
        assert detections.image_metadata["class_names"] == prompts

    def test_class_names_in_metadata(self, config, image_metadata):
        """image_metadata should include class_names for supervision compatibility."""
        result = BatchInferenceResult(
            query_results=[
                QueryResult(prompt="bicycle", present=False, presence_confidence=0.1),
            ]
        )
        detections = result_to_detections(
            result, image_metadata, config, ["bicycle"]
        )
        assert "class_names" in detections.image_metadata
        assert detections.image_metadata["class_names"] == ["bicycle"]

    def test_to_supervision_works(self, config, image_metadata):
        """Verify Detections.to_supervision() produces valid sv.Detections."""
        result = BatchInferenceResult(
            query_results=[
                QueryResult(
                    prompt="cat",
                    present=True,
                    presence_confidence=0.95,
                    instances=[
                        InstancePrediction(
                            center_x_bin=320, center_y_bin=240,
                            width_bin=600, height_bin=600,
                            coord_x_confidence=0.9, coord_y_confidence=0.9,
                            size_w_confidence=0.9, size_h_confidence=0.9,
                        ),
                    ],
                ),
            ]
        )
        detections = result_to_detections(
            result, image_metadata, config, ["cat"]
        )
        sv_detections = detections.to_supervision()
        assert sv_detections.xyxy.shape == (1, 4)
        assert sv_detections.class_id.shape == (1,)
        assert sv_detections.confidence.shape == (1,)
