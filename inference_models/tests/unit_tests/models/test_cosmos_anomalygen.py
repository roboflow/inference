from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference_models.errors import ModelInputError, ModelRuntimeError
from inference_models.models.cosmos3.cosmos_anomalygen import CosmosAnomalyGen


def _model() -> CosmosAnomalyGen:
    return CosmosAnomalyGen(runtime=MagicMock(), device=torch.device("cpu"))


def _image() -> np.ndarray:
    return np.full((8, 8, 3), 128, dtype=np.uint8)


def _mask() -> np.ndarray:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    return mask


def test_generate_passes_sdg_parameters_to_runtime() -> None:
    model = _model()
    model._runtime.generate.return_value = [np.zeros((8, 8, 3), dtype=np.uint8)]

    result = model.generate(
        image=_image(),
        mask=_mask(),
        anomaly_type="wood+crack",
        guidance=5.5,
        num_steps=20,
        seed=7,
        num_images=1,
    )

    kwargs = model._runtime.generate.call_args.kwargs
    assert kwargs["anomaly_type"] == "wood+crack"
    assert kwargs["guidance"] == 5.5
    assert kwargs["num_steps"] == 20
    assert kwargs["seed"] == 7
    assert kwargs["crop_and_paste"] is True
    assert len(result) == 1


def test_generate_defaults_match_the_production_recipe() -> None:
    model = _model()
    model._runtime.generate.return_value = []

    model.generate(image=_image(), mask=_mask(), anomaly_type="wood+crack")

    kwargs = model._runtime.generate.call_args.kwargs
    assert kwargs["guidance"] == 1.5
    assert kwargs["num_steps"] == 35
    assert kwargs["crop_ratio"] == 4.0
    assert kwargs["crop_and_paste"] is True
    assert kwargs["poisson_blend"] is False


def test_from_pretrained_rejects_package_without_runtime_module(tmp_path) -> None:
    with pytest.raises(ModelRuntimeError):
        CosmosAnomalyGen.from_pretrained(str(tmp_path), device=torch.device("cpu"))


def test_generate_binarizes_uint8_mask_with_128_threshold() -> None:
    model = _model()
    model._runtime.generate.return_value = []
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0, 0] = 127  # below threshold
    mask[1, 1] = 200  # above threshold

    model.generate(image=_image(), mask=mask, anomaly_type="wood+crack")

    sent = model._runtime.generate.call_args.kwargs["mask"]
    assert sent[0, 0] == 0
    assert sent[1, 1] == 255


def test_generate_rejects_empty_mask() -> None:
    model = _model()

    with pytest.raises(ModelInputError):
        model.generate(
            image=_image(),
            mask=np.zeros((8, 8), dtype=np.uint8),
            anomaly_type="wood+crack",
        )


def test_generate_rejects_mismatched_mask_resolution() -> None:
    model = _model()

    with pytest.raises(ModelInputError):
        model.generate(
            image=_image(),
            mask=np.full((4, 4), 255, dtype=np.uint8),
            anomaly_type="wood+crack",
        )


def test_generate_converts_colors_both_ways() -> None:
    model = _model()
    rgb_out = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb_out[..., 0] = 255  # red in RGB
    model._runtime.generate.return_value = [rgb_out]
    bgr_in = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr_in[..., 0] = 255  # blue in BGR

    result = model.generate(image=bgr_in, mask=_mask(), anomaly_type="wood+crack")

    sent = model._runtime.generate.call_args.kwargs["image"]
    assert sent[0, 0].tolist() == [0, 0, 255]  # runtime sees RGB
    assert result[0][0, 0].tolist() == [0, 0, 255]  # caller gets BGR
