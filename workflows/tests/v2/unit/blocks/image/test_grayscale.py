"""Unit tests of ``roboflow_core/convert_grayscale@v1``, called directly.

Expected values are hand-computed from OpenCV's documented fixed-point weights,
or come from ``cv2.cvtColor`` itself on the same colors in V1's BGR layout.
"""

from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.image.convert_grayscale.v1 import ConvertGrayscaleBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import RowRef

VIDEO = VideoMetadata(
    video_identifier="cam-1",
    frame_number=7,
    frame_timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc),
)


def nested_crop(pixels: torch.Tensor) -> ImageData:
    # pixels sit at (6, 5) of a root frame, reached through two crops.
    channels, height, width = pixels.shape
    canvas = torch.zeros((channels, height + 10, width + 12), dtype=torch.uint8)
    canvas[:, 5 : 5 + height, 6 : 6 + width] = pixels
    root = ImageData.from_tensor(canvas, image_id="root", video_metadata=VIDEO)
    outer = root.crop((2, 1, width + 9, height + 7), image_id="outer")
    inner = outer.crop((4, 4, 4 + width, 4 + height), image_id="inner")
    nested = inner.with_parent_row(RowRef("detections", 3))

    return nested


def test_primary_colors_follow_opencv_fixed_point_weights() -> None:
    # given: red, green, blue, white, black and a mid gray (R=G=B=128)
    colors = torch.tensor(
        [[255, 0, 0, 255, 0, 128], [0, 255, 0, 255, 0, 128], [0, 0, 255, 255, 0, 128]],
        dtype=torch.uint8,
    ).reshape(3, 1, 6)

    # when
    gray = ConvertGrayscaleBlock().run(image=ImageData.from_tensor(colors))["image"]

    # then: (9798 R + 19235 G + 3735 B + 16384) >> 15, worked out by hand
    assert gray.tensor_image.shape == (1, 1, 6)
    assert gray.tensor_image.flatten().tolist() == [76, 150, 29, 255, 0, 128]


def test_random_image_equals_opencv_bgr2gray() -> None:
    # given
    pixels = torch.from_numpy(
        np.random.default_rng(3).integers(0, 256, (3, 37, 29), dtype=np.uint8)
    )
    bgr = np.ascontiguousarray(pixels.permute(1, 2, 0).numpy()[:, :, ::-1])

    # when
    gray = ConvertGrayscaleBlock().run(image=ImageData.from_tensor(pixels))["image"]

    # then
    expected = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    assert np.array_equal(gray.tensor_image[0].numpy(), expected)


def test_keeps_identity_geometry_and_metadata_of_a_nested_crop() -> None:
    # given
    pixels = torch.arange(3 * 4 * 5, dtype=torch.uint8).reshape(3, 4, 5) * 4
    crop = nested_crop(pixels)
    before = crop.tensor_image.clone()

    # when
    gray = ConvertGrayscaleBlock().run(image=crop)["image"]

    # then
    assert gray.image_id == "inner"
    assert gray.parent == crop.parent
    assert gray.root == crop.root
    assert gray.root.offset_xy == (6, 5)
    assert gray.video_metadata is VIDEO
    assert gray.parent_row == RowRef("detections", 3)
    assert gray.channels == 1
    assert torch.equal(crop.tensor_image, before)


def test_one_channel_image_is_returned_unchanged() -> None:
    # given
    image = ImageData.from_tensor(torch.full((1, 3, 3), 9, dtype=torch.uint8))

    # when
    result = ConvertGrayscaleBlock().run(image=image)["image"]

    # then
    assert result is image


def test_channels_last_view_gives_the_same_pixels() -> None:
    # given: a (3, H, W) view of channels-last storage, not contiguous
    hwc = torch.from_numpy(
        np.random.default_rng(4).integers(0, 256, (9, 11, 3), dtype=np.uint8)
    )
    view = hwc.permute(2, 0, 1)
    assert not view.is_contiguous()

    # when
    from_view = ConvertGrayscaleBlock().run(image=ImageData.from_tensor(view))
    from_copy = ConvertGrayscaleBlock().run(
        image=ImageData.from_tensor(view.contiguous())
    )

    # then
    assert torch.equal(from_view["image"].tensor_image, from_copy["image"].tensor_image)


def test_declares_a_pure_prunable_block() -> None:
    assert ConvertGrayscaleBlock.type == "roboflow_core/convert_grayscale@v1"
    assert ConvertGrayscaleBlock.prunable is True
    assert ConvertGrayscaleBlock.mutates == ()
