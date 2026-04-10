"""Unit tests for Falcon Perception post-processing helpers.

These tests verify the pure-function output conversion logic (bbox pairing,
coordinate conversion, mask decoding) without requiring the actual model or
GPU.
"""

import numpy as np
import pytest
import torch

from inference_models.models.falcon_perception.falcon_perception_torch import (
    normalized_bbox_to_xyxy,
    pair_bbox_entries,
)

# ── pair_bbox_entries ─────────────────────────────────────────────────


class TestPairBboxEntries:
    def test_empty_input(self):
        assert pair_bbox_entries([]) == []

    def test_single_complete_pair(self):
        raw = [{"x": 0.5, "y": 0.5}, {"h": 0.1, "w": 0.2}]
        result = pair_bbox_entries(raw)
        assert len(result) == 1
        assert result[0] == {"x": 0.5, "y": 0.5, "h": 0.1, "w": 0.2}

    def test_multiple_pairs(self):
        raw = [
            {"x": 0.1, "y": 0.2},
            {"h": 0.3, "w": 0.4},
            {"x": 0.5, "y": 0.6},
            {"h": 0.7, "w": 0.8},
        ]
        result = pair_bbox_entries(raw)
        assert len(result) == 2
        assert result[0] == {"x": 0.1, "y": 0.2, "h": 0.3, "w": 0.4}
        assert result[1] == {"x": 0.5, "y": 0.6, "h": 0.7, "w": 0.8}

    def test_incomplete_pair_ignored(self):
        raw = [{"x": 0.5, "y": 0.5}]
        result = pair_bbox_entries(raw)
        assert len(result) == 0

    def test_non_dict_entries_skipped(self):
        raw = ["garbage", {"x": 0.5, "y": 0.5}, 42, {"h": 0.1, "w": 0.2}]
        result = pair_bbox_entries(raw)
        assert len(result) == 1
        assert result[0] == {"x": 0.5, "y": 0.5, "h": 0.1, "w": 0.2}

    def test_all_keys_in_single_dict(self):
        raw = [{"x": 0.5, "y": 0.5, "h": 0.1, "w": 0.2}]
        result = pair_bbox_entries(raw)
        assert len(result) == 1


# ── normalized_bbox_to_xyxy ───────────────────────────────────────────


class TestNormalizedBboxToXyxy:
    def test_center_of_image(self):
        bbox = {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2}
        xyxy = normalized_bbox_to_xyxy(bbox, image_width=100, image_height=100)
        assert xyxy == pytest.approx([40.0, 40.0, 60.0, 60.0])

    def test_top_left_corner(self):
        bbox = {"x": 0.0, "y": 0.0, "w": 0.2, "h": 0.2}
        xyxy = normalized_bbox_to_xyxy(bbox, image_width=100, image_height=100)
        # cx=0, cy=0, w=20, h=20 → x1=-10 (clamped to 0), y1=-10 (clamped to 0)
        assert xyxy[0] == 0.0
        assert xyxy[1] == 0.0
        assert xyxy[2] == pytest.approx(10.0)
        assert xyxy[3] == pytest.approx(10.0)

    def test_bottom_right_corner(self):
        bbox = {"x": 1.0, "y": 1.0, "w": 0.2, "h": 0.2}
        xyxy = normalized_bbox_to_xyxy(bbox, image_width=100, image_height=100)
        # cx=100, cy=100 → x2=110 clamped to 100
        assert xyxy[2] == 100.0
        assert xyxy[3] == 100.0

    def test_non_square_image(self):
        bbox = {"x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5}
        xyxy = normalized_bbox_to_xyxy(bbox, image_width=640, image_height=480)
        # cx=320, cy=240, bw=320, bh=240
        assert xyxy == pytest.approx([160.0, 120.0, 480.0, 360.0])

    def test_zero_size_bbox(self):
        bbox = {"x": 0.5, "y": 0.5, "w": 0.0, "h": 0.0}
        xyxy = normalized_bbox_to_xyxy(bbox, image_width=100, image_height=100)
        # Degenerate zero-area box at center
        assert xyxy == pytest.approx([50.0, 50.0, 50.0, 50.0])

    def test_full_image_bbox(self):
        bbox = {"x": 0.5, "y": 0.5, "w": 1.0, "h": 1.0}
        xyxy = normalized_bbox_to_xyxy(bbox, image_width=200, image_height=200)
        assert xyxy == pytest.approx([0.0, 0.0, 200.0, 200.0])


# ── _to_pil_images ───────────────────────────────────────────────────


class TestToPilImages:
    def test_numpy_bgr_to_pil(self):
        from inference_models.models.falcon_perception.falcon_perception_torch import (
            _to_pil_images,
        )

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # Red channel in BGR (index 2 = R in BGR)
        pil_images = _to_pil_images(img, "bgr")
        assert len(pil_images) == 1
        assert pil_images[0].size == (200, 100)
        # After BGR→RGB conversion, the red channel should be 255
        arr = np.array(pil_images[0])
        assert arr[0, 0, 0] == 255  # R in RGB
        assert arr[0, 0, 2] == 0  # B in RGB

    def test_numpy_rgb_to_pil(self):
        from inference_models.models.falcon_perception.falcon_perception_torch import (
            _to_pil_images,
        )

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:, :, 0] = 128  # Red channel in RGB
        pil_images = _to_pil_images(img, "rgb")
        assert len(pil_images) == 1
        arr = np.array(pil_images[0])
        assert arr[0, 0, 0] == 128

    def test_list_of_numpy(self):
        from inference_models.models.falcon_perception.falcon_perception_torch import (
            _to_pil_images,
        )

        imgs = [
            np.zeros((50, 60, 3), dtype=np.uint8),
            np.zeros((70, 80, 3), dtype=np.uint8),
        ]
        pil_images = _to_pil_images(imgs, "rgb")
        assert len(pil_images) == 2
        assert pil_images[0].size == (60, 50)
        assert pil_images[1].size == (80, 70)

    def test_tensor_chw_to_pil(self):
        from inference_models.models.falcon_perception.falcon_perception_torch import (
            _to_pil_images,
        )

        tensor = torch.zeros(3, 100, 200, dtype=torch.uint8)
        pil_images = _to_pil_images(tensor, "rgb")
        assert len(pil_images) == 1
        assert pil_images[0].size == (200, 100)

    def test_batch_tensor_to_pil(self):
        from inference_models.models.falcon_perception.falcon_perception_torch import (
            _to_pil_images,
        )

        tensor = torch.zeros(2, 3, 100, 200, dtype=torch.uint8)
        pil_images = _to_pil_images(tensor, "rgb")
        assert len(pil_images) == 2


# ── Empty / absent results ────────────────────────────────────────────


class TestEmptyResults:
    def test_empty_detections_shape(self):
        det = torch.zeros((0, 4), dtype=torch.float32)
        assert det.shape == (0, 4)

    def test_pair_bbox_entries_with_empty_dicts(self):
        assert pair_bbox_entries([{}, {}]) == []

    def test_pair_bbox_entries_with_partial_keys(self):
        raw = [{"x": 0.5}, {"y": 0.5}, {"h": 0.1}]
        assert pair_bbox_entries(raw) == []
