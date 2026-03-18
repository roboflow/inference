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
                0.9809,
                0.9674,
                0.9638,
                0.9622,
                0.9581,
                0.9558,
                0.9555,
                0.9540,
                0.9493,
                0.9489,
                0.4168,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1324, 538, 3070, 1970],
            [1707, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2352, 1254, 2522],
            [1508, 1884, 1720, 2093],
            [2681, 802, 2867, 976],
            [929, 1843, 1091, 2004],
            [929, 1845, 1091, 2004],
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
                0.9809,
                0.9674,
                0.9638,
                0.9622,
                0.9581,
                0.9558,
                0.9555,
                0.9540,
                0.9493,
                0.9489,
                0.4168,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1324, 538, 3070, 1970],
            [1707, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2352, 1254, 2522],
            [1508, 1884, 1720, 2093],
            [2681, 802, 2867, 976],
            [929, 1843, 1091, 2004],
            [929, 1845, 1091, 2004],
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
                0.9809,
                0.9674,
                0.9638,
                0.9622,
                0.9581,
                0.9558,
                0.9555,
                0.9540,
                0.9493,
                0.9489,
                0.4168,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1324, 538, 3070, 1970],
            [1707, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2352, 1254, 2522],
            [1508, 1884, 1720, 2093],
            [2681, 802, 2867, 976],
            [929, 1843, 1091, 2004],
            [929, 1845, 1091, 2004],
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
                0.9815,
                0.9674,
                0.9638,
                0.9620,
                0.9584,
                0.9565,
                0.9560,
                0.9543,
                0.9520,
                0.9491,
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
            [1323, 533, 3071, 1970],
            [1708, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2354, 1253, 2524],
            [1508, 1884, 1721, 2093],
            [929, 1843, 1091, 2004],
            [2681, 802, 2867, 976],
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
                0.9815,
                0.9674,
                0.9638,
                0.9620,
                0.9584,
                0.9565,
                0.9560,
                0.9543,
                0.9520,
                0.9491,
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
            [1323, 533, 3071, 1970],
            [1708, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2354, 1253, 2524],
            [1508, 1884, 1721, 2093],
            [929, 1843, 1091, 2004],
            [2681, 802, 2867, 976],
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
                0.9815,
                0.9674,
                0.9638,
                0.9620,
                0.9584,
                0.9565,
                0.9560,
                0.9543,
                0.9520,
                0.9491,
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
            [1323, 533, 3071, 1970],
            [1708, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2354, 1253, 2524],
            [1508, 1884, 1721, 2093],
            [929, 1843, 1091, 2004],
            [2681, 802, 2867, 976],
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
                0.9815,
                0.9674,
                0.9638,
                0.9620,
                0.9584,
                0.9565,
                0.9560,
                0.9543,
                0.9520,
                0.9491,
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
            [1323, 533, 3071, 1970],
            [1708, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2354, 1253, 2524],
            [1508, 1884, 1721, 2093],
            [929, 1843, 1091, 2004],
            [2681, 802, 2867, 976],
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
                0.9815,
                0.9674,
                0.9638,
                0.9620,
                0.9584,
                0.9565,
                0.9560,
                0.9543,
                0.9520,
                0.9491,
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
            [1323, 533, 3071, 1970],
            [1708, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2354, 1253, 2524],
            [1508, 1884, 1721, 2093],
            [929, 1843, 1091, 2004],
            [2681, 802, 2867, 976],
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

    model = AutoModel.from_pretrained(
        model_id_or_path=rfdetr_nano_t4_trt_package,
        device=torch.device("cuda:0"),
    )

    pre_processed_1, _ = model.pre_process(dog_image_numpy)
    pre_processed_2, _ = model.pre_process(bike_image_numpy)

    outputs = []
    for pre_processed in [pre_processed_1, pre_processed_2]:
        no_graph = model.forward(pre_processed, use_cuda_graph=False)
        model._trt_cuda_graph_cache = TRTCudaGraphCache(capacity=16)
        capture_graph = model.forward(pre_processed, use_cuda_graph=True)
        replay_graph = model.forward(pre_processed, use_cuda_graph=True)

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

    model = AutoModel.from_pretrained(
        model_id_or_path=rfdetr_nano_t4_trt_package,
        device=torch.device("cuda:0"),
    )

    pre_processed, _ = model.pre_process(dog_image_numpy)

    output = model.forward(pre_processed, use_cuda_graph=False)

    assert output[0].shape == (1, 300, 4)
    assert output[1].shape == (1, 300, 91)

    output = model.forward(pre_processed, use_cuda_graph=True)  # capture

    assert output[0].shape == (1, 300, 4)
    assert output[1].shape == (1, 300, 91)

    output = model.forward(pre_processed, use_cuda_graph=True)  # replay

    assert output[0].shape == (1, 300, 4)
    assert output[1].shape == (1, 300, 91)
