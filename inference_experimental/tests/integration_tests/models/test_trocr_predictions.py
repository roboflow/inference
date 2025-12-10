import numpy as np
import pytest
import torch
from inference_exp.configuration import DEFAULT_DEVICE


@pytest.mark.slow
@pytest.mark.torch_models
def test_tr_ocr_predictions_for_numpy_image(
    tr_ocr_package: str,
    ocr_test_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.trocr.trocr_hf import TROcrHF

    model = TROcrHF.from_pretrained(tr_ocr_package, device=DEFAULT_DEVICE)

    # when
    result = model(ocr_test_image_numpy)

    # then
    assert result[0] == "THIS IS A TEST IMAGE FOR OCR."


@pytest.mark.slow
@pytest.mark.torch_models
def test_tr_ocr_predictions_for_numpy_image_list(
    tr_ocr_package: str,
    ocr_test_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.trocr.trocr_hf import TROcrHF

    model = TROcrHF.from_pretrained(tr_ocr_package, device=DEFAULT_DEVICE)

    # when
    result = model([ocr_test_image_numpy, ocr_test_image_numpy])

    # then
    assert result[0] == "THIS IS A TEST IMAGE FOR OCR."
    assert result[1] == "THIS IS A TEST IMAGE FOR OCR."


@pytest.mark.slow
@pytest.mark.torch_models
def test_tr_ocr_predictions_for_torch_image(
    tr_ocr_package: str,
    ocr_test_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.trocr.trocr_hf import TROcrHF

    model = TROcrHF.from_pretrained(tr_ocr_package, device=DEFAULT_DEVICE)

    # when
    result = model(ocr_test_image_torch)

    # then
    assert result[0] == "THIS IS A TEST IMAGE FOR OCR."


@pytest.mark.slow
@pytest.mark.torch_models
def test_tr_ocr_predictions_for_torch_batch(
    tr_ocr_package: str,
    ocr_test_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.trocr.trocr_hf import TROcrHF

    model = TROcrHF.from_pretrained(tr_ocr_package, device=DEFAULT_DEVICE)

    # when
    result = model(torch.stack([ocr_test_image_torch, ocr_test_image_torch], dim=0))

    # then
    assert result[0] == "THIS IS A TEST IMAGE FOR OCR."
    assert result[1] == "THIS IS A TEST IMAGE FOR OCR."


@pytest.mark.slow
@pytest.mark.torch_models
def test_tr_ocr_predictions_for_torch_list(
    tr_ocr_package: str,
    ocr_test_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.trocr.trocr_hf import TROcrHF

    model = TROcrHF.from_pretrained(tr_ocr_package, device=DEFAULT_DEVICE)

    # when
    result = model([ocr_test_image_torch, ocr_test_image_torch])

    # then
    assert result[0] == "THIS IS A TEST IMAGE FOR OCR."
    assert result[1] == "THIS IS A TEST IMAGE FOR OCR."
