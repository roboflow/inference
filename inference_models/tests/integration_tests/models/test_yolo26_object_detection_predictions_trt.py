import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy(
    yolo26_object_detections_coin_counting_trt_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_object_detection_trt import (
        YOLO26ForObjectDetectionTRT,
    )

    model = YOLO26ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolo26_object_detections_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9837,
                0.9707,
                0.9202,
                0.8459,
                0.8444,
                0.8408,
                0.5737,
                0.4922,
                0.4378,
                0.4340,
                0.2636,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([2, 2, 2, 0, 1, 3, 0, 0, 1, 3, 3], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1252, 2049, 1431, 2241],
            [1741, 2285, 1921, 2480],
            [1707, 2565, 1896, 2770],
            [1459, 2296, 1633, 2476],
            [1164, 2624, 1382, 2856],
            [1502, 1867, 1728, 2096],
            [923, 1836, 1100, 2009],
            [1090, 2346, 1268, 2525],
            [1164, 2625, 1381, 2857],
            [1256, 2059, 1425, 2234],
            [2670, 792, 2875, 979],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_batch_numpy(
    yolo26_object_detections_coin_counting_trt_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_object_detection_trt import (
        YOLO26ForObjectDetectionTRT,
    )

    model = YOLO26ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolo26_object_detections_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9837,
                0.9707,
                0.9202,
                0.8459,
                0.8444,
                0.8408,
                0.5737,
                0.4922,
                0.4378,
                0.4340,
                0.2636,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([2, 2, 2, 0, 1, 3, 0, 0, 1, 3, 3], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1252, 2049, 1431, 2241],
            [1741, 2285, 1921, 2480],
            [1707, 2565, 1896, 2770],
            [1459, 2296, 1633, 2476],
            [1164, 2624, 1382, 2856],
            [1502, 1867, 1728, 2096],
            [923, 1836, 1100, 2009],
            [1090, 2346, 1268, 2525],
            [1164, 2625, 1381, 2857],
            [1256, 2059, 1425, 2234],
            [2670, 792, 2875, 979],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9837,
                0.9707,
                0.9202,
                0.8459,
                0.8444,
                0.8408,
                0.5737,
                0.4922,
                0.4378,
                0.4340,
                0.2636,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([2, 2, 2, 0, 1, 3, 0, 0, 1, 3, 3], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1252, 2049, 1431, 2241],
            [1741, 2285, 1921, 2480],
            [1707, 2565, 1896, 2770],
            [1459, 2296, 1633, 2476],
            [1164, 2624, 1382, 2856],
            [1502, 1867, 1728, 2096],
            [923, 1836, 1100, 2009],
            [1090, 2346, 1268, 2525],
            [1164, 2625, 1381, 2857],
            [1256, 2059, 1425, 2234],
            [2670, 792, 2875, 979],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch(
    yolo26_object_detections_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_object_detection_trt import (
        YOLO26ForObjectDetectionTRT,
    )

    model = YOLO26ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolo26_object_detections_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9837,
                0.9707,
                0.9196,
                0.8495,
                0.8418,
                0.8408,
                0.5737,
                0.4922,
                0.4282,
                0.4273,
                0.2606,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([2, 2, 2, 1, 3, 0, 0, 0, 3, 1, 3], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [
                [1252, 2049, 1431, 2241],
                [1741, 2286, 1921, 2480],
                [1707, 2565, 1896, 2770],
                [1164, 2624, 1382, 2856],
                [1502, 1867, 1728, 2096],
                [1459, 2296, 1633, 2476],
                [923, 1836, 1100, 2009],
                [1090, 2346, 1268, 2525],
                [1256, 2059, 1425, 2234],
                [1164, 2626, 1381, 2857],
                [2671, 792, 2875, 979],
            ]
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_multiple_predictions_in_row(
    yolo26_object_detections_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_object_detection_trt import (
        YOLO26ForObjectDetectionTRT,
    )

    model = YOLO26ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolo26_object_detections_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    for _ in range(8):
        predictions = model(coins_counting_image_torch)

        # then
        assert torch.allclose(
            predictions[0].confidence.cpu(),
            torch.tensor(
                [
                    0.9837,
                    0.9707,
                    0.9196,
                    0.8495,
                    0.8418,
                    0.8408,
                    0.5737,
                    0.4922,
                    0.4282,
                    0.4273,
                    0.2606,
                ]
            ).cpu(),
            atol=0.01,
        )
        assert torch.allclose(
            predictions[0].class_id.cpu(),
            torch.tensor([2, 2, 2, 1, 3, 0, 0, 0, 3, 1, 3], dtype=torch.int32).cpu(),
        )
        expected_xyxy = torch.tensor(
            [
                [
                    [1252, 2049, 1431, 2241],
                    [1741, 2286, 1921, 2480],
                    [1707, 2565, 1896, 2770],
                    [1164, 2624, 1382, 2856],
                    [1502, 1867, 1728, 2096],
                    [1459, 2296, 1633, 2476],
                    [923, 1836, 1100, 2009],
                    [1090, 2346, 1268, 2525],
                    [1256, 2059, 1425, 2234],
                    [1164, 2626, 1381, 2857],
                    [2671, 792, 2875, 979],
                ]
            ],
            dtype=torch.int32,
        )
        assert torch.allclose(
            predictions[0].xyxy.cpu(),
            expected_xyxy.cpu(),
            atol=5,
        )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_list(
    yolo26_object_detections_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_object_detection_trt import (
        YOLO26ForObjectDetectionTRT,
    )

    model = YOLO26ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolo26_object_detections_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9837,
                0.9707,
                0.9196,
                0.8495,
                0.8418,
                0.8408,
                0.5737,
                0.4922,
                0.4282,
                0.4273,
                0.2606,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([2, 2, 2, 1, 3, 0, 0, 0, 3, 1, 3], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [
                [1252, 2049, 1431, 2241],
                [1741, 2286, 1921, 2480],
                [1707, 2565, 1896, 2770],
                [1164, 2624, 1382, 2856],
                [1502, 1867, 1728, 2096],
                [1459, 2296, 1633, 2476],
                [923, 1836, 1100, 2009],
                [1090, 2346, 1268, 2525],
                [1256, 2059, 1425, 2234],
                [1164, 2626, 1381, 2857],
                [2671, 792, 2875, 979],
            ]
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9837,
                0.9707,
                0.9196,
                0.8495,
                0.8418,
                0.8408,
                0.5737,
                0.4922,
                0.4282,
                0.4273,
                0.2606,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([2, 2, 2, 1, 3, 0, 0, 0, 3, 1, 3], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1252, 2049, 1431, 2241],
            [1741, 2286, 1921, 2480],
            [1707, 2565, 1896, 2770],
            [1164, 2624, 1382, 2856],
            [1502, 1867, 1728, 2096],
            [1459, 2296, 1633, 2476],
            [923, 1836, 1100, 2009],
            [1090, 2346, 1268, 2525],
            [1256, 2059, 1425, 2234],
            [1164, 2626, 1381, 2857],
            [2671, 792, 2875, 979],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_batch(
    yolo26_object_detections_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_object_detection_trt import (
        YOLO26ForObjectDetectionTRT,
    )

    model = YOLO26ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolo26_object_detections_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0)
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9837,
                0.9707,
                0.9196,
                0.8495,
                0.8418,
                0.8408,
                0.5737,
                0.4922,
                0.4282,
                0.4273,
                0.2606,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([2, 2, 2, 1, 3, 0, 0, 0, 3, 1, 3], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1252, 2049, 1431, 2241],
            [1741, 2286, 1921, 2480],
            [1707, 2565, 1896, 2770],
            [1164, 2624, 1382, 2856],
            [1502, 1867, 1728, 2096],
            [1459, 2296, 1633, 2476],
            [923, 1836, 1100, 2009],
            [1090, 2346, 1268, 2525],
            [1256, 2059, 1425, 2234],
            [1164, 2626, 1381, 2857],
            [2671, 792, 2875, 979],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9837,
                0.9707,
                0.9196,
                0.8495,
                0.8418,
                0.8408,
                0.5737,
                0.4922,
                0.4282,
                0.4273,
                0.2606,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([2, 2, 2, 1, 3, 0, 0, 0, 3, 1, 3], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1252, 2049, 1431, 2241],
            [1741, 2286, 1921, 2480],
            [1707, 2565, 1896, 2770],
            [1164, 2624, 1382, 2856],
            [1502, 1867, 1728, 2096],
            [1459, 2296, 1633, 2476],
            [923, 1836, 1100, 2009],
            [1090, 2346, 1268, 2525],
            [1256, 2059, 1425, 2234],
            [1164, 2626, 1381, 2857],
            [2671, 792, 2875, 979],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_per_class_confidence_filters_detections(
    yolo26_object_detections_coin_counting_trt_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_trt import (
        YOLO26ForObjectDetectionTRT,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = YOLO26ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolo26_object_detections_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.25,
        per_class_confidence={class_names[2]: 1.01},
    )
    predictions = model(coins_counting_image_numpy, confidence="best")
    assert 2 not in predictions[0].class_id.cpu().tolist()
