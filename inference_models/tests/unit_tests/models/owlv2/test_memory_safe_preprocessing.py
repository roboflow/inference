import pytest

pytest.importorskip("cv2")
pytest.importorskip("torchvision")
pytest.importorskip("transformers")

import numpy as np
import torch

from inference_models.entities import ImageDimensions
from inference_models.models.owlv2.owlv2_hf import OWLv2HF


class _FakeImageProcessor:
    size = {"height": 16, "width": 16}
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]


class _FakeProcessor:
    image_processor = _FakeImageProcessor()

    def __call__(self, *args, **kwargs):
        raise AssertionError("OWLv2 image preprocessing should not use HF processor")


def _empty_owlv2_model() -> OWLv2HF:
    model = OWLv2HF.__new__(OWLv2HF)
    model._processor = _FakeProcessor()
    model._device = torch.device("cpu")
    return model


def test_pre_process_resizes_extreme_aspect_ratio_numpy_image_before_padding() -> None:
    # given
    model = _empty_owlv2_model()
    image = np.ones((8, 150, 3), dtype=np.uint8) * 255

    # when
    pixel_values, image_dimensions = model.pre_process(images=image)

    # then
    assert image_dimensions == [ImageDimensions(height=8, width=150)]
    assert pixel_values.shape == (1, 3, 16, 16)
    assert torch.allclose(pixel_values[0, :, 0, :], torch.ones((3, 16)))
    assert torch.allclose(pixel_values[0, :, 1:, :], torch.zeros((3, 15, 16)))


def test_pre_process_resizes_extreme_aspect_ratio_tensor_batch_before_padding() -> None:
    # given
    model = _empty_owlv2_model()
    images = torch.ones((2, 3, 8, 150), dtype=torch.uint8) * 255

    # when
    pixel_values, image_dimensions = model.pre_process(images=images)

    # then
    assert image_dimensions == [
        ImageDimensions(height=8, width=150),
        ImageDimensions(height=8, width=150),
    ]
    assert pixel_values.shape == (2, 3, 16, 16)
    assert torch.allclose(pixel_values[:, :, 0, :], torch.ones((2, 3, 16)))
    assert torch.allclose(pixel_values[:, :, 1:, :], torch.zeros((2, 3, 15, 16)))
