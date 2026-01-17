import numpy as np
import pytest
import torch

from inference_models import Detections
from inference_models.configuration import DEFAULT_DEVICE


@pytest.mark.slow
@pytest.mark.torch_models
def test_doctr_predictions_for_numpy_image(
    doctr_package: str, ocr_test_image_numpy: np.ndarray
) -> None:
    # given
    from inference_models.models.doctr.doctr_torch import DocTR

    model = DocTR.from_pretrained(
        doctr_package,
        device=DEFAULT_DEVICE,
    )

    # when
    result = model(ocr_test_image_numpy)

    # then
    assert len(result) == 2
    assert result[0][0] == "This is a test image for OCR."
    assert isinstance(result[1][0], Detections)
    assert np.allclose(
        result[1][0].xyxy.cpu().numpy(),
        np.array(
            [
                [
                    [11, 8, 123, 62],
                    [138, 9, 184, 61],
                    [199, 20, 235, 60],
                    [248, 12, 348, 60],
                    [359, 8, 532, 77],
                    [542, 10, 620, 60],
                    [634, 12, 780, 59],
                    [11, 8, 780, 77],
                    [11, 8, 780, 77],
                ]
            ]
        ),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_doctr_predictions_for_numpy_images_list(
    doctr_package: str, ocr_test_image_numpy: np.ndarray
) -> None:
    # given
    from inference_models.models.doctr.doctr_torch import DocTR

    model = DocTR.from_pretrained(
        doctr_package,
        device=DEFAULT_DEVICE,
    )

    # when
    result = model([ocr_test_image_numpy, ocr_test_image_numpy])

    # then
    assert len(result) == 2
    assert result[0][0] == "This is a test image for OCR."
    assert result[0][1] == "This is a test image for OCR."
    assert isinstance(result[1][0], Detections)
    assert np.allclose(
        result[1][0].xyxy.cpu().numpy(),
        np.array(
            [
                [
                    [11, 8, 123, 62],
                    [138, 9, 184, 61],
                    [199, 20, 235, 60],
                    [248, 12, 348, 60],
                    [359, 8, 532, 77],
                    [542, 10, 620, 60],
                    [634, 12, 780, 59],
                    [11, 8, 780, 77],
                    [11, 8, 780, 77],
                ]
            ]
        ),
        atol=5,
    )
    assert isinstance(result[1][1], Detections)
    assert np.allclose(
        result[1][1].xyxy.cpu().numpy(),
        np.array(
            [
                [
                    [11, 8, 123, 62],
                    [138, 9, 184, 61],
                    [199, 20, 235, 60],
                    [248, 12, 348, 60],
                    [359, 8, 532, 77],
                    [542, 10, 620, 60],
                    [634, 12, 780, 59],
                    [11, 8, 780, 77],
                    [11, 8, 780, 77],
                ]
            ]
        ),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_doctr_predictions_for_torch_tensor(
    doctr_package: str, ocr_test_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.doctr.doctr_torch import DocTR

    model = DocTR.from_pretrained(
        doctr_package,
        device=DEFAULT_DEVICE,
    )

    # when
    result = model(ocr_test_image_torch)

    # then
    assert len(result) == 2
    assert result[0][0] == "This is a test image for OCR."
    assert isinstance(result[1][0], Detections)
    assert np.allclose(
        result[1][0].xyxy.cpu().numpy(),
        np.array(
            [
                [
                    [11, 8, 123, 62],
                    [138, 9, 184, 61],
                    [199, 20, 235, 60],
                    [248, 12, 348, 60],
                    [359, 8, 532, 77],
                    [542, 10, 620, 60],
                    [634, 12, 780, 59],
                    [11, 8, 780, 77],
                    [11, 8, 780, 77],
                ]
            ]
        ),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_doctr_predictions_for_torch_batch(
    doctr_package: str, ocr_test_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.doctr.doctr_torch import DocTR

    model = DocTR.from_pretrained(
        doctr_package,
        device=DEFAULT_DEVICE,
    )

    # when
    result = model(torch.stack([ocr_test_image_torch, ocr_test_image_torch], dim=0))

    # then
    assert len(result) == 2
    assert result[0][0] == "This is a test image for OCR."
    assert result[0][1] == "This is a test image for OCR."
    assert isinstance(result[1][0], Detections)
    assert np.allclose(
        result[1][0].xyxy.cpu().numpy(),
        np.array(
            [
                [
                    [11, 8, 123, 62],
                    [138, 9, 184, 61],
                    [199, 20, 235, 60],
                    [248, 12, 348, 60],
                    [359, 8, 532, 77],
                    [542, 10, 620, 60],
                    [634, 12, 780, 59],
                    [11, 8, 780, 77],
                    [11, 8, 780, 77],
                ]
            ]
        ),
        atol=5,
    )
    assert isinstance(result[1][1], Detections)
    assert np.allclose(
        result[1][1].xyxy.cpu().numpy(),
        np.array(
            [
                [
                    [11, 8, 123, 62],
                    [138, 9, 184, 61],
                    [199, 20, 235, 60],
                    [248, 12, 348, 60],
                    [359, 8, 532, 77],
                    [542, 10, 620, 60],
                    [634, 12, 780, 59],
                    [11, 8, 780, 77],
                    [11, 8, 780, 77],
                ]
            ]
        ),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_doctr_predictions_for_torch_list(
    doctr_package: str, ocr_test_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.doctr.doctr_torch import DocTR

    model = DocTR.from_pretrained(
        doctr_package,
        device=DEFAULT_DEVICE,
    )

    # when
    result = model([ocr_test_image_torch, ocr_test_image_torch])

    # then
    assert len(result) == 2
    assert result[0][0] == "This is a test image for OCR."
    assert result[0][1] == "This is a test image for OCR."
    assert isinstance(result[1][0], Detections)
    assert np.allclose(
        result[1][0].xyxy.cpu().numpy(),
        np.array(
            [
                [
                    [11, 8, 123, 62],
                    [138, 9, 184, 61],
                    [199, 20, 235, 60],
                    [248, 12, 348, 60],
                    [359, 8, 532, 77],
                    [542, 10, 620, 60],
                    [634, 12, 780, 59],
                    [11, 8, 780, 77],
                    [11, 8, 780, 77],
                ]
            ]
        ),
        atol=5,
    )
    assert isinstance(result[1][1], Detections)
    assert np.allclose(
        result[1][1].xyxy.cpu().numpy(),
        np.array(
            [
                [
                    [11, 8, 123, 62],
                    [138, 9, 184, 61],
                    [199, 20, 235, 60],
                    [248, 12, 348, 60],
                    [359, 8, 532, 77],
                    [542, 10, 620, 60],
                    [634, 12, 780, 59],
                    [11, 8, 780, 77],
                    [11, 8, 780, 77],
                ]
            ]
        ),
        atol=5,
    )
