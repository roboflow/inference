import numpy as np
import pytest
import torch

from inference_models import AutoModel

# Same model, image and expected values as the Torch and ONNX coin-counting tests. The
# ExecuTorch package is exported from the Torch package. GPU programs differ from these
# expected values by up to 0.014 in confidence, so the tolerance is wider than theirs.
EXPECTED_CONFIDENCE = torch.tensor(
    [0.9444, 0.9384, 0.9369, 0.9279, 0.902, 0.898, 0.8813, 0.879, 0.7601, 0.5973]
)
EXPECTED_XYXY = torch.tensor(
    [
        [1743, 2295, 1919, 2465],
        [1090, 2348, 1261, 2521],
        [1460, 2301, 1629, 2468],
        [1252, 2059, 1425, 2228],
        [924, 1843, 1098, 2002],
        [1708, 2574, 1890, 2754],
        [2677, 801, 2868, 975],
        [1506, 1881, 1726, 2091],
        [1167, 2628, 1380, 2846],
        [1303, 538, 3019, 1955],
    ],
    dtype=torch.int32,
)


@pytest.mark.slow
@pytest.mark.executorch_models
def test_executorch_package_with_stretch_resize_and_contrast_stretching_numpy(
    coin_counting_rfdetr_nano_executorch_cs_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = AutoModel.from_pretrained(
        coin_counting_rfdetr_nano_executorch_cs_stretch_package
    )

    # when
    predictions = model(coins_counting_image_numpy, confidence=0.5)

    # then
    assert type(model).__name__ == "RFDetrForObjectDetectionExecuTorch"
    assert torch.allclose(
        predictions[0].confidence.cpu(), EXPECTED_CONFIDENCE, atol=0.02
    )
    assert torch.allclose(predictions[0].xyxy.cpu(), EXPECTED_XYXY, atol=2)


@pytest.mark.slow
@pytest.mark.executorch_models
def test_executorch_package_with_stretch_resize_and_contrast_stretching_batch_numpy(
    coin_counting_rfdetr_nano_executorch_cs_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = AutoModel.from_pretrained(
        coin_counting_rfdetr_nano_executorch_cs_stretch_package
    )

    # when
    predictions = model(
        [
            coins_counting_image_numpy,
            coins_counting_image_numpy,
            coins_counting_image_numpy,
        ],
        confidence=0.5,
    )

    # then
    assert len(predictions) == 3
    for prediction in predictions:
        assert torch.allclose(
            prediction.confidence.cpu(), EXPECTED_CONFIDENCE, atol=0.02
        )
        assert torch.allclose(prediction.xyxy.cpu(), EXPECTED_XYXY, atol=2)
