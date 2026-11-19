import numpy as np
import pytest
import torch

from inference_models import AutoModel
from inference_models.models.rfdetr.rfdetr_object_detection_pytorch import (
    RFDetrForObjectDetectionTorch,
)

SIZES = ["atto", "femto", "pico"]

# Expected detections (confidence >= 0.5) of the RF-DETR Atto, Femto and Pico
# coin-counting packages on the coin-counting image, sorted by x_min. The image
# is BGR as a numpy array and RGB as a torch tensor, so the two inputs give
# slightly different detections.
EXPECTED_FROM_NUMPY = {
    "atto": {
        "confidence": [
            0.8958,
            0.894,
            0.9234,
            0.9748,
            0.9263,
            0.8919,
            0.9176,
            0.8958,
            0.8268,
            0.8971,
        ],
        "class_id": [1, 1, 1, 4, 1, 1, 1, 1, 1, 1],
        "xyxy": [
            [926, 1837, 1094, 1995],
            [1089, 2354, 1251, 2525],
            [1169, 2634, 1369, 2841],
            [1229, 573, 3024, 1990],
            [1251, 2061, 1427, 2241],
            [1457, 2308, 1638, 2478],
            [1499, 1883, 1719, 2086],
            [1699, 2578, 1883, 2750],
            [1740, 2306, 1916, 2485],
            [2670, 802, 2873, 983],
        ],
    },
    "femto": {
        "confidence": [
            0.8528,
            0.8643,
            0.7505,
            0.7008,
            0.6699,
            0.7775,
            0.6357,
            0.6939,
            0.7705,
            0.621,
        ],
        "class_id": [1, 1, 1, 1, 4, 1, 1, 1, 1, 1],
        "xyxy": [
            [928, 1843, 1090, 2001],
            [1091, 2358, 1249, 2531],
            [1167, 2638, 1378, 2850],
            [1249, 2062, 1426, 2226],
            [1304, 546, 3022, 2027],
            [1464, 2312, 1625, 2467],
            [1498, 1885, 1725, 2094],
            [1709, 2592, 1893, 2763],
            [1743, 2314, 1917, 2478],
            [2676, 803, 2868, 981],
        ],
    },
    "pico": {
        "confidence": [
            0.9352,
            0.9476,
            0.9592,
            0.9463,
            0.9169,
            0.9378,
            0.9615,
            0.9233,
            0.9447,
            0.9333,
        ],
        "class_id": [1, 1, 1, 1, 4, 1, 1, 1, 1, 1],
        "xyxy": [
            [926, 1842, 1094, 1998],
            [1085, 2349, 1259, 2521],
            [1170, 2632, 1375, 2848],
            [1249, 2056, 1426, 2227],
            [1296, 557, 3012, 2011],
            [1459, 2300, 1628, 2471],
            [1501, 1881, 1722, 2087],
            [1707, 2566, 1888, 2759],
            [1743, 2292, 1914, 2464],
            [2675, 802, 2865, 975],
        ],
    },
}

EXPECTED_FROM_TORCH = {
    "atto": {
        "confidence": [
            0.8948,
            0.9063,
            0.9289,
            0.9066,
            0.9598,
            0.8915,
            0.9166,
            0.8604,
            0.7297,
            0.8568,
        ],
        "class_id": [1, 1, 1, 1, 4, 1, 1, 1, 1, 1],
        "xyxy": [
            [919, 1833, 1103, 2003],
            [1083, 2348, 1253, 2526],
            [1166, 2629, 1373, 2848],
            [1244, 2048, 1434, 2245],
            [1253, 556, 3065, 2052],
            [1455, 2298, 1642, 2479],
            [1496, 1879, 1723, 2091],
            [1694, 2577, 1891, 2760],
            [1735, 2293, 1922, 2488],
            [2665, 794, 2877, 985],
        ],
    },
    "femto": {
        "confidence": [
            0.8565,
            0.8495,
            0.7034,
            0.7087,
            0.5948,
            0.7411,
            0.5154,
            0.6371,
            0.7705,
            0.6739,
        ],
        "class_id": [1, 1, 1, 1, 4, 1, 1, 1, 1, 1],
        "xyxy": [
            [928, 1841, 1091, 2002],
            [1092, 2356, 1250, 2529],
            [1166, 2639, 1379, 2849],
            [1248, 2059, 1430, 2236],
            [1303, 554, 3026, 2072],
            [1463, 2313, 1627, 2468],
            [1498, 1883, 1725, 2096],
            [1705, 2583, 1892, 2767],
            [1741, 2305, 1918, 2482],
            [2675, 802, 2870, 980],
        ],
    },
    "pico": {
        "confidence": [
            0.9338,
            0.946,
            0.9644,
            0.9673,
            0.8797,
            0.9454,
            0.9648,
            0.9381,
            0.9639,
            0.9444,
        ],
        "class_id": [1, 1, 1, 1, 4, 1, 1, 1, 1, 1],
        "xyxy": [
            [925, 1839, 1097, 1999],
            [1084, 2346, 1260, 2524],
            [1169, 2629, 1375, 2849],
            [1247, 2056, 1427, 2231],
            [1298, 554, 3019, 2031],
            [1459, 2299, 1627, 2471],
            [1501, 1877, 1722, 2091],
            [1705, 2563, 1889, 2764],
            [1738, 2295, 1916, 2472],
            [2673, 801, 2868, 979],
        ],
    },
}


def assert_detections_match(prediction, expected: dict) -> None:
    # Detections are compared in x_min order: near-equal confidences can swap
    # places between CPU and GPU.
    order = torch.argsort(prediction.xyxy[:, 0].cpu(), stable=True)
    assert torch.allclose(
        prediction.confidence.cpu()[order],
        torch.tensor(expected["confidence"]),
        atol=0.02,
    )
    assert torch.equal(
        prediction.class_id.cpu()[order],
        torch.tensor(expected["class_id"], dtype=torch.int32),
    )
    assert torch.allclose(
        prediction.xyxy.cpu()[order],
        torch.tensor(expected["xyxy"], dtype=torch.int32),
        atol=2,
    )


def _load(request, size: str) -> RFDetrForObjectDetectionTorch:
    package = request.getfixturevalue(
        f"coin_counting_rfdetr_{size}_torch_stretch_package"
    )
    return RFDetrForObjectDetectionTorch.from_pretrained(model_name_or_path=package)


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.parametrize("size", SIZES)
def test_pe_core_t_torch_package_loads_through_auto_model(
    size: str, request, coins_counting_image_numpy: np.ndarray
) -> None:
    # given
    package = request.getfixturevalue(
        f"coin_counting_rfdetr_{size}_torch_stretch_package"
    )

    # when
    model = AutoModel.from_pretrained(package)
    predictions = model(coins_counting_image_numpy, confidence=0.5)

    # then
    assert isinstance(model, RFDetrForObjectDetectionTorch)
    assert_detections_match(predictions[0], EXPECTED_FROM_NUMPY[size])


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.parametrize("size", SIZES)
def test_pe_core_t_torch_package_numpy(
    size: str, request, coins_counting_image_numpy: np.ndarray
) -> None:
    # given
    model = _load(request, size)

    # when
    predictions = model(coins_counting_image_numpy, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert_detections_match(predictions[0], EXPECTED_FROM_NUMPY[size])


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.parametrize("size", SIZES)
def test_pe_core_t_torch_package_numpy_batch(
    size: str, request, coins_counting_image_numpy: np.ndarray
) -> None:
    # given
    model = _load(request, size)

    # when
    predictions = model(
        [coins_counting_image_numpy, coins_counting_image_numpy], confidence=0.5
    )

    # then
    assert len(predictions) == 2
    assert_detections_match(predictions[0], EXPECTED_FROM_NUMPY[size])
    assert_detections_match(predictions[1], EXPECTED_FROM_NUMPY[size])


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.parametrize("size", SIZES)
def test_pe_core_t_torch_package_torch(
    size: str, request, coins_counting_image_torch: torch.Tensor
) -> None:
    # given
    model = _load(request, size)

    # when
    predictions = model(coins_counting_image_torch, confidence=0.5)

    # then
    assert len(predictions) == 1
    assert_detections_match(predictions[0], EXPECTED_FROM_TORCH[size])


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.parametrize("size", SIZES)
def test_pe_core_t_torch_package_torch_batch(
    size: str, request, coins_counting_image_torch: torch.Tensor
) -> None:
    # given
    model = _load(request, size)

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0),
        confidence=0.5,
    )

    # then
    assert len(predictions) == 2
    assert_detections_match(predictions[0], EXPECTED_FROM_TORCH[size])
    assert_detections_match(predictions[1], EXPECTED_FROM_TORCH[size])


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.parametrize("size", SIZES)
def test_pe_core_t_torch_package_torch_list(
    size: str, request, coins_counting_image_torch: torch.Tensor
) -> None:
    # given
    model = _load(request, size)

    # when
    predictions = model(
        [coins_counting_image_torch, coins_counting_image_torch], confidence=0.5
    )

    # then
    assert len(predictions) == 2
    assert_detections_match(predictions[0], EXPECTED_FROM_TORCH[size])
    assert_detections_match(predictions[1], EXPECTED_FROM_TORCH[size])
