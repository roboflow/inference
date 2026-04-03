import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy(
    yolov8_coin_counting_trt_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_trt import (
        YOLOv8ForObjectDetectionTRT,
    )

    model = YOLOv8ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
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
    yolov8_coin_counting_trt_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_trt import (
        YOLOv8ForObjectDetectionTRT,
    )

    model = YOLOv8ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
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
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
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
    yolov8_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_trt import (
        YOLOv8ForObjectDetectionTRT,
    )

    model = YOLOv8ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
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
    yolov8_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_trt import (
        YOLOv8ForObjectDetectionTRT,
    )

    model = YOLOv8ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_coin_counting_trt_package,
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
                    0.9956,
                    0.9727,
                    0.9653,
                    0.9468,
                    0.9448,
                    0.9390,
                    0.9302,
                    0.9287,
                    0.9155,
                    0.9019,
                ]
            ).cpu(),
            atol=0.01,
        )
        assert torch.allclose(
            predictions[0].class_id.cpu(),
            torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
        )
        expected_xyxy = torch.tensor(
            [
                [1304, 614, 3024, 1918],
                [1714, 2571, 1884, 2759],
                [2678, 806, 2866, 974],
                [1744, 2294, 1914, 2469],
                [1260, 2058, 1424, 2233],
                [1469, 2302, 1624, 2467],
                [929, 1843, 1091, 1997],
                [1514, 1880, 1718, 2089],
                [1177, 2632, 1374, 2846],
                [1099, 2348, 1260, 2522],
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
    yolov8_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_trt import (
        YOLOv8ForObjectDetectionTRT,
    )

    model = YOLOv8ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
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
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
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
    yolov8_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_trt import (
        YOLOv8ForObjectDetectionTRT,
    )

    model = YOLOv8ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_coin_counting_trt_package,
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
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
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
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
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
def test_trt_cudagraph_cache_reuses_previously_seen_input_shapes(
    yolov8n_640_t4_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    from inference_models import AutoModel
    from inference_models.models.common.trt import TRTCudaGraphCache

    device = torch.device("cuda:0")
    trt_cuda_graph_cache = TRTCudaGraphCache(capacity=16)
    model = AutoModel.from_pretrained(
        model_id_or_path=yolov8n_640_t4_trt_package,
        device=device,
        trt_cuda_graph_cache=trt_cuda_graph_cache,
    )

    pre_processed_single, _ = model.pre_process(dog_image_numpy)

    seen_shapes = set()
    capture_outputs = {}
    test_sequence = [1, 2, 1, 4, 2, 1, 4, 3, 3]

    for batch_size in test_sequence:
        batch = pre_processed_single.repeat(batch_size, 1, 1, 1)
        cache_key = (tuple(batch.shape), batch.dtype, device)

        cache_size_before = trt_cuda_graph_cache.get_current_size()

        output = model.forward(batch)

        cache_size_after = trt_cuda_graph_cache.get_current_size()

        if cache_key not in seen_shapes:
            assert cache_size_after == cache_size_before + 1
            seen_shapes.add(cache_key)
            capture_outputs[cache_key] = output.clone()
            continue

        assert cache_size_after == cache_size_before
        assert torch.allclose(capture_outputs[cache_key], output, atol=1e-6)

    assert set(trt_cuda_graph_cache.list_keys()) == seen_shapes


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_cudagraph_output_matches_non_cudagraph_output(
    yolov8n_640_t4_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    from inference_models import AutoModel
    from inference_models.models.common.trt import TRTCudaGraphCache

    device = torch.device("cuda:0")
    trt_cuda_graph_cache = TRTCudaGraphCache(capacity=16)
    model = AutoModel.from_pretrained(
        model_id_or_path=yolov8n_640_t4_trt_package,
        device=device,
        trt_cuda_graph_cache=trt_cuda_graph_cache,
    )
    pre_processed_single, _ = model.pre_process(dog_image_numpy)

    for batch_size in [1, 4]:
        batch = pre_processed_single.repeat(batch_size, 1, 1, 1)

        no_graph = model.forward(batch, disable_cuda_graphs=True)

        capture_graph = model.forward(batch)
        replay_graph = model.forward(batch)

        assert torch.allclose(no_graph, capture_graph, atol=1e-6)
        assert torch.allclose(no_graph, replay_graph, atol=1e-6)


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_cudagraph_cache_eviction(
    yolov8n_640_t4_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    from inference_models import AutoModel
    from inference_models.models.common.trt import TRTCudaGraphCache

    device = torch.device("cuda:0")
    trt_cuda_graph_cache = TRTCudaGraphCache(capacity=3)
    model = AutoModel.from_pretrained(
        model_id_or_path=yolov8n_640_t4_trt_package,
        device=device,
        trt_cuda_graph_cache=trt_cuda_graph_cache,
    )

    pre_processed_single, _ = model.pre_process(dog_image_numpy)

    batch_sizes = [1, 2, 3]
    for bs in batch_sizes:
        batch = pre_processed_single.repeat(bs, 1, 1, 1)
        model.forward(batch)

    assert trt_cuda_graph_cache.get_current_size() == 3
    keys_before = list(trt_cuda_graph_cache.list_keys())

    batch_4 = pre_processed_single.repeat(4, 1, 1, 1)
    model.forward(batch_4)

    assert trt_cuda_graph_cache.get_current_size() == 3
    keys_after = trt_cuda_graph_cache.list_keys()
    assert keys_before[0] not in keys_after
    for key in keys_before[1:]:
        assert key in keys_after
    key_4 = (tuple(batch_4.shape), batch_4.dtype, device)
    assert key_4 in trt_cuda_graph_cache

    batch_2 = pre_processed_single.repeat(2, 1, 1, 1)
    model.forward(batch_2)

    batch_5 = pre_processed_single.repeat(5, 1, 1, 1)
    model.forward(batch_5)

    assert trt_cuda_graph_cache.get_current_size() == 3
    key_3 = (
        tuple(pre_processed_single.repeat(3, 1, 1, 1).shape),
        batch_2.dtype,
        device,
    )
    remaining_keys = trt_cuda_graph_cache.list_keys()
    assert key_3 not in remaining_keys

    key_2 = (tuple(batch_2.shape), batch_2.dtype, device)
    key_5 = (tuple(batch_5.shape), batch_5.dtype, device)
    assert remaining_keys == [key_4, key_2, key_5]

    no_graph = model.forward(batch_5, disable_cuda_graphs=True)
    replay = model.forward(batch_5)
    assert torch.allclose(no_graph, replay, atol=1e-6)
