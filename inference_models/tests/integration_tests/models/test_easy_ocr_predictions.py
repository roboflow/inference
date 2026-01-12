import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE


@pytest.mark.slow
@pytest.mark.torch_models
def test_easy_ocr_predictions_for_numpy_image(
    easy_ocr_package: str,
    ocr_test_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.easy_ocr.easy_ocr_torch import EasyOCRTorch

    model = EasyOCRTorch.from_pretrained(easy_ocr_package, device=DEFAULT_DEVICE)

    # when
    result = model(ocr_test_image_numpy)

    # then
    assert result[0][0].startswith("This is a test image for OCR")
    assert np.allclose(
        result[1][0].xyxy.cpu().numpy(), np.array([[6, 2, 786, 83]]), atol=5
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_easy_ocr_predictions_for_numpy_image_list(
    easy_ocr_package: str,
    ocr_test_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.easy_ocr.easy_ocr_torch import EasyOCRTorch

    model = EasyOCRTorch.from_pretrained(easy_ocr_package, device=DEFAULT_DEVICE)

    # when
    result = model([ocr_test_image_numpy, ocr_test_image_numpy])

    # then
    assert result[0][0].startswith("This is a test image for OCR")
    assert np.allclose(
        result[1][0].xyxy.cpu().numpy(), np.array([[6, 2, 786, 83]]), atol=5
    )
    assert result[0][1].startswith("This is a test image for OCR")
    assert np.allclose(
        result[1][1].xyxy.cpu().numpy(), np.array([[6, 2, 786, 83]]), atol=5
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_easy_ocr_predictions_for_torch_image(
    easy_ocr_package: str,
    ocr_test_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.easy_ocr.easy_ocr_torch import EasyOCRTorch

    model = EasyOCRTorch.from_pretrained(easy_ocr_package, device=DEFAULT_DEVICE)

    # when
    result = model(ocr_test_image_torch)

    # then
    assert result[0][0].startswith("This is a test image for OCR")
    assert np.allclose(
        result[1][0].xyxy.cpu().numpy(), np.array([[6, 2, 786, 83]]), atol=5
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_easy_ocr_predictions_for_torch_batch(
    easy_ocr_package: str,
    ocr_test_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.easy_ocr.easy_ocr_torch import EasyOCRTorch

    model = EasyOCRTorch.from_pretrained(easy_ocr_package, device=DEFAULT_DEVICE)

    # when
    result = model(torch.stack([ocr_test_image_torch, ocr_test_image_torch], dim=0))

    # then
    assert result[0][0].startswith("This is a test image for OCR")
    assert np.allclose(
        result[1][0].xyxy.cpu().numpy(), np.array([[6, 2, 786, 83]]), atol=5
    )
    assert result[0][1].startswith("This is a test image for OCR")
    assert np.allclose(
        result[1][1].xyxy.cpu().numpy(), np.array([[6, 2, 786, 83]]), atol=5
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_easy_ocr_predictions_for_torch_list(
    easy_ocr_package: str,
    ocr_test_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.easy_ocr.easy_ocr_torch import EasyOCRTorch

    model = EasyOCRTorch.from_pretrained(easy_ocr_package, device=DEFAULT_DEVICE)

    # when
    result = model([ocr_test_image_torch, ocr_test_image_torch])

    # then
    assert result[0][0].startswith("This is a test image for OCR")
    assert np.allclose(
        result[1][0].xyxy.cpu().numpy(), np.array([[6, 2, 786, 83]]), atol=5
    )
    assert result[0][1].startswith("This is a test image for OCR")
    assert np.allclose(
        result[1][1].xyxy.cpu().numpy(), np.array([[6, 2, 786, 83]]), atol=5
    )
