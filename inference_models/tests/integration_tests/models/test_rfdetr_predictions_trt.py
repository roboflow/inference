import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy(
    rfdetr_coin_counting_trt_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
        RFDetrForObjectDetectionTRT,
    )

    model = RFDetrForObjectDetectionTRT.from_pretrained(
        model_name_or_path=rfdetr_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9697,
                0.9622,
                0.9612,
                0.9607,
                0.9602,
                0.9601,
                0.9563,
                0.9522,
                0.8563,
                0.8026,
                0.4912,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 3], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1172, 2633, 1376, 2848],
            [1091, 2356, 1259, 2523],
            [1316, 531, 3034, 1962],
            [1741, 2299, 1918, 2469],
            [1458, 2304, 1628, 2473],
            [1254, 2061, 1425, 2231],
            [1706, 2579, 1888, 2761],
            [1501, 1884, 1724, 2095],
            [922, 1842, 1095, 2007],
            [2677, 803, 2874, 978],
            [2677, 803, 2874, 978],
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
    rfdetr_coin_counting_trt_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
        RFDetrForObjectDetectionTRT,
    )

    model = RFDetrForObjectDetectionTRT.from_pretrained(
        model_name_or_path=rfdetr_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9697,
                0.9622,
                0.9612,
                0.9607,
                0.9602,
                0.9601,
                0.9563,
                0.9522,
                0.8563,
                0.8026,
                0.4912,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 3], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1172, 2633, 1376, 2848],
            [1091, 2356, 1259, 2523],
            [1316, 531, 3034, 1962],
            [1741, 2299, 1918, 2469],
            [1458, 2304, 1628, 2473],
            [1254, 2061, 1425, 2231],
            [1706, 2579, 1888, 2761],
            [1501, 1884, 1724, 2095],
            [922, 1842, 1095, 2007],
            [2677, 803, 2874, 978],
            [2677, 803, 2874, 978],
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
                0.9697,
                0.9622,
                0.9612,
                0.9607,
                0.9602,
                0.9601,
                0.9563,
                0.9522,
                0.8563,
                0.8026,
                0.4912,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 3], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1172, 2633, 1376, 2848],
            [1091, 2356, 1259, 2523],
            [1316, 531, 3034, 1962],
            [1741, 2299, 1918, 2469],
            [1458, 2304, 1628, 2473],
            [1254, 2061, 1425, 2231],
            [1706, 2579, 1888, 2761],
            [1501, 1884, 1724, 2095],
            [922, 1842, 1095, 2007],
            [2677, 803, 2874, 978],
            [2677, 803, 2874, 978],
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
    rfdetr_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
        RFDetrForObjectDetectionTRT,
    )

    model = RFDetrForObjectDetectionTRT.from_pretrained(
        model_name_or_path=rfdetr_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9686,
                0.9668,
                0.9668,
                0.9292,
                0.9239,
                0.8371,
                0.8295,
                0.7957,
                0.7114,
                0.5794,
                0.5176,
                0.4922,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 4, 1, 3, 1, 1, 3, 3, 6, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1171, 2632, 1379, 2845],
            [1458, 2305, 1631, 2471],
            [1090, 2354, 1261, 2520],
            [1315, 523, 3029, 1963],
            [1498, 1883, 1726, 2093],
            [2674, 801, 2878, 980],
            [1702, 2571, 1892, 2760],
            [1249, 2059, 1428, 2233],
            [918, 1838, 1099, 2007],
            [1737, 2289, 1922, 2472],
            [1249, 2059, 1428, 2233],
            [918, 1838, 1099, 2007],
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
    rfdetr_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
        RFDetrForObjectDetectionTRT,
    )

    model = RFDetrForObjectDetectionTRT.from_pretrained(
        model_name_or_path=rfdetr_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    for _ in range(8):
        # when
        predictions = model(coins_counting_image_torch)

        # then
        assert torch.allclose(
            predictions[0].confidence.cpu(),
            torch.tensor(
                [
                    0.9686,
                    0.9668,
                    0.9668,
                    0.9292,
                    0.9239,
                    0.8371,
                    0.8295,
                    0.7957,
                    0.7114,
                    0.5794,
                    0.5176,
                    0.4922,
                ]
            ).cpu(),
            atol=0.01,
        )
        assert torch.allclose(
            predictions[0].class_id.cpu(),
            torch.tensor([1, 1, 1, 4, 1, 3, 1, 1, 3, 3, 6, 1], dtype=torch.int32).cpu(),
        )
        expected_xyxy = torch.tensor(
            [
                [1171, 2632, 1379, 2845],
                [1458, 2305, 1631, 2471],
                [1090, 2354, 1261, 2520],
                [1315, 523, 3029, 1963],
                [1498, 1883, 1726, 2093],
                [2674, 801, 2878, 980],
                [1702, 2571, 1892, 2760],
                [1249, 2059, 1428, 2233],
                [918, 1838, 1099, 2007],
                [1737, 2289, 1922, 2472],
                [1249, 2059, 1428, 2233],
                [918, 1838, 1099, 2007],
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
    rfdetr_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
        RFDetrForObjectDetectionTRT,
    )

    model = RFDetrForObjectDetectionTRT.from_pretrained(
        model_name_or_path=rfdetr_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9686,
                0.9668,
                0.9668,
                0.9292,
                0.9239,
                0.8371,
                0.8295,
                0.7957,
                0.7114,
                0.5794,
                0.5176,
                0.4922,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 4, 1, 3, 1, 1, 3, 3, 6, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1171, 2632, 1379, 2845],
            [1458, 2305, 1631, 2471],
            [1090, 2354, 1261, 2520],
            [1315, 523, 3029, 1963],
            [1498, 1883, 1726, 2093],
            [2674, 801, 2878, 980],
            [1702, 2571, 1892, 2760],
            [1249, 2059, 1428, 2233],
            [918, 1838, 1099, 2007],
            [1737, 2289, 1922, 2472],
            [1249, 2059, 1428, 2233],
            [918, 1838, 1099, 2007],
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
                0.9686,
                0.9668,
                0.9668,
                0.9292,
                0.9239,
                0.8371,
                0.8295,
                0.7957,
                0.7114,
                0.5794,
                0.5176,
                0.4922,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1, 1, 4, 1, 3, 1, 1, 3, 3, 6, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1171, 2632, 1379, 2845],
            [1458, 2305, 1631, 2471],
            [1090, 2354, 1261, 2520],
            [1315, 523, 3029, 1963],
            [1498, 1883, 1726, 2093],
            [2674, 801, 2878, 980],
            [1702, 2571, 1892, 2760],
            [1249, 2059, 1428, 2233],
            [918, 1838, 1099, 2007],
            [1737, 2289, 1922, 2472],
            [1249, 2059, 1428, 2233],
            [918, 1838, 1099, 2007],
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
    rfdetr_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
        RFDetrForObjectDetectionTRT,
    )

    model = RFDetrForObjectDetectionTRT.from_pretrained(
        model_name_or_path=rfdetr_coin_counting_trt_package,
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
                0.9686,
                0.9668,
                0.9668,
                0.9292,
                0.9239,
                0.8371,
                0.8295,
                0.7957,
                0.7114,
                0.5794,
                0.5176,
                0.4922,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 4, 1, 3, 1, 1, 3, 3, 6, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1171, 2632, 1379, 2845],
            [1458, 2305, 1631, 2471],
            [1090, 2354, 1261, 2520],
            [1315, 523, 3029, 1963],
            [1498, 1883, 1726, 2093],
            [2674, 801, 2878, 980],
            [1702, 2571, 1892, 2760],
            [1249, 2059, 1428, 2233],
            [918, 1838, 1099, 2007],
            [1737, 2289, 1922, 2472],
            [1249, 2059, 1428, 2233],
            [918, 1838, 1099, 2007],
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
                0.9686,
                0.9668,
                0.9668,
                0.9292,
                0.9239,
                0.8371,
                0.8295,
                0.7957,
                0.7114,
                0.5794,
                0.5176,
                0.4922,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1, 1, 4, 1, 3, 1, 1, 3, 3, 6, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1171, 2632, 1379, 2845],
            [1458, 2305, 1631, 2471],
            [1090, 2354, 1261, 2520],
            [1315, 523, 3029, 1963],
            [1498, 1883, 1726, 2093],
            [2674, 801, 2878, 980],
            [1702, 2571, 1892, 2760],
            [1249, 2059, 1428, 2233],
            [918, 1838, 1099, 2007],
            [1737, 2289, 1922, 2472],
            [1249, 2059, 1428, 2233],
            [918, 1838, 1099, 2007],
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
def test_trt_cudagraph_output_matches_non_cudagraph_output(
    rfdetr_nano_t4_trt_package: str,
    dog_image_numpy: np.ndarray,
    bike_image_numpy: np.ndarray,
) -> None:
    from inference_models import AutoModel
    from inference_models.models.common.trt import TRTCudaGraphCache

    trt_cuda_graph_cache = TRTCudaGraphCache(capacity=16)
    model = AutoModel.from_pretrained(
        model_id_or_path=rfdetr_nano_t4_trt_package,
        device=torch.device("cuda:0"),
        trt_cuda_graph_cache=trt_cuda_graph_cache,
    )

    pre_processed_1, _ = model.pre_process(dog_image_numpy)
    pre_processed_2, _ = model.pre_process(bike_image_numpy)

    outputs = []
    for pre_processed in [pre_processed_1, pre_processed_2]:
        no_graph = model.forward(pre_processed, disable_cuda_graphs=True)
        capture_graph = model.forward(pre_processed)
        replay_graph = model.forward(pre_processed)

        outputs.append((no_graph, capture_graph, replay_graph))

    for image_outputs in outputs:
        no_graph, capture_graph, replay_graph = image_outputs
        for result_idx in range(2):
            assert torch.allclose(
                no_graph[result_idx],
                capture_graph[result_idx],
                atol=1e-6,
            )
            assert torch.allclose(
                no_graph[result_idx],
                replay_graph[result_idx],
                atol=1e-6,
            )

    # make sure that the allcloses aren't true because of buffer aliasing or something weird
    # outputs should be different between images and the same between execution branches.
    for execution_branch_idx in range(3):
        for result_idx in range(2):
            assert not torch.allclose(
                outputs[0][execution_branch_idx][result_idx],
                outputs[1][execution_branch_idx][result_idx],
                atol=1e-6,
            )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_outputs_match_expected_shapes(
    rfdetr_nano_t4_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    from inference_models import AutoModel
    from inference_models.models.common.trt import TRTCudaGraphCache

    trt_cuda_graph_cache = TRTCudaGraphCache(capacity=16)
    model = AutoModel.from_pretrained(
        model_id_or_path=rfdetr_nano_t4_trt_package,
        device=torch.device("cuda:0"),
        trt_cuda_graph_cache=trt_cuda_graph_cache,
    )

    pre_processed, _ = model.pre_process(dog_image_numpy)

    output = model.forward(pre_processed, disable_cuda_graphs=True)

    assert output[0].shape == (1, 300, 4)
    assert output[1].shape == (1, 300, 91)

    output = model.forward(pre_processed)  # capture

    assert output[0].shape == (1, 300, 4)
    assert output[1].shape == (1, 300, 91)

    output = model.forward(pre_processed)  # replay

    assert output[0].shape == (1, 300, 4)
    assert output[1].shape == (1, 300, 91)


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_per_class_confidence_filters_detections(
    rfdetr_coin_counting_trt_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
        RFDetrForObjectDetectionTRT,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = RFDetrForObjectDetectionTRT.from_pretrained(
        model_name_or_path=rfdetr_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.3,
        per_class_confidence={class_names[1]: 1.01},
    )
    predictions = model(coins_counting_image_numpy, confidence="best")
    assert 1 not in predictions[0].class_id.cpu().tolist()
