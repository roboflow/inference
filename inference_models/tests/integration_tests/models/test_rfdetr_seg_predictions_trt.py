import os

import numpy as np
import pytest
import torch

from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask


def _assert_instance_segmentation_predictions_match(actual, expected) -> None:
    assert len(actual) == len(expected)
    for actual_element, expected_element in zip(actual, expected):
        torch.testing.assert_close(
            actual_element.xyxy.cpu(),
            expected_element.xyxy.cpu(),
            atol=1.0,
            rtol=0,
        )
        torch.testing.assert_close(
            actual_element.confidence.cpu(),
            expected_element.confidence.cpu(),
            atol=1e-4,
            rtol=0,
        )
        torch.testing.assert_close(
            actual_element.class_id.cpu(),
            expected_element.class_id.cpu(),
            atol=0,
            rtol=0,
        )

        actual_mask = actual_element.mask.detach().to(torch.bool).cpu()
        expected_mask = expected_element.mask.detach().to(torch.bool).cpu()
        assert tuple(actual_mask.shape) == tuple(expected_mask.shape)
        intersection = torch.logical_and(actual_mask, expected_mask).sum().item()
        union = torch.logical_or(actual_mask, expected_mask).sum().item()
        assert union > 0
        assert intersection / union >= 0.999


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy(
    rfdetr_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_trt import (
        RFDetrForInstanceSegmentationTRT,
    )

    model = RFDetrForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=rfdetr_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 172, 188, 374]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16021 <= predictions[0].mask.cpu().sum().item() <= 16071


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy_rle_variant(
    rfdetr_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_trt import (
        RFDetrForInstanceSegmentationTRT,
    )

    model = RFDetrForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=rfdetr_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(asl_image_numpy, mask_format="rle")
    predictions_ref = model(asl_image_numpy)
    decoded_mask = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[0].mask, device=torch.device("cpu")
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 172, 188, 374]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16021 <= decoded_mask.sum().item() <= 16071
    assert np.allclose(
        decoded_mask.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_batch_numpy(
    rfdetr_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_trt import (
        RFDetrForInstanceSegmentationTRT,
    )

    model = RFDetrForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=rfdetr_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 172, 188, 374]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16021 <= predictions[0].mask.cpu().sum().item() <= 16071
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 172, 188, 374]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16021 <= predictions[1].mask.cpu().sum().item() <= 16071


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_batch_numpy_rle_variant(
    rfdetr_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_trt import (
        RFDetrForInstanceSegmentationTRT,
    )

    model = RFDetrForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=rfdetr_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy], mask_format="rle")
    predictions_ref = model([asl_image_numpy, asl_image_numpy])
    decoded_mask_1 = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[0].mask, device=torch.device("cpu")
    )
    decoded_mask_2 = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[1].mask, device=torch.device("cpu")
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 172, 188, 374]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16021 <= decoded_mask_1.cpu().sum().item() <= 16071
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 172, 188, 374]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16021 <= decoded_mask_2.cpu().sum().item() <= 16071
    assert np.allclose(
        decoded_mask_1.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )
    assert np.allclose(
        decoded_mask_2.cpu().numpy(), predictions_ref[1].mask.cpu().numpy()
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch(
    rfdetr_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_trt import (
        RFDetrForInstanceSegmentationTRT,
    )

    model = RFDetrForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=rfdetr_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9548]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 173, 189, 375]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16179 <= predictions[0].mask.cpu().sum().item() <= 16229


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_multiple_predictions_in_row(
    rfdetr_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_trt import (
        RFDetrForInstanceSegmentationTRT,
    )

    model = RFDetrForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=rfdetr_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    for _ in range(8):
        # when
        predictions = model(asl_image_torch)

        # then
        assert torch.allclose(
            predictions[0].confidence.cpu(),
            torch.tensor([0.9548]).cpu(),
            atol=0.01,
        )
        assert torch.allclose(
            predictions[0].class_id.cpu(),
            torch.tensor([20], dtype=torch.int32).cpu(),
        )
        expected_xyxy = torch.tensor(
            [[63, 173, 189, 375]],
            dtype=torch.int32,
        )
        assert torch.allclose(
            predictions[0].xyxy.cpu(),
            expected_xyxy.cpu(),
            atol=5,
        )
        assert 16179 <= predictions[0].mask.cpu().sum().item() <= 16229


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_list(
    rfdetr_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_trt import (
        RFDetrForInstanceSegmentationTRT,
    )

    model = RFDetrForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=rfdetr_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9548]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 173, 189, 375]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16179 <= predictions[0].mask.cpu().sum().item() <= 16229
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9548]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 173, 189, 375]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16179 <= predictions[1].mask.cpu().sum().item() <= 16229


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_batch(
    rfdetr_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_trt import (
        RFDetrForInstanceSegmentationTRT,
    )

    model = RFDetrForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=rfdetr_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9548]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 173, 189, 375]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16179 <= predictions[0].mask.cpu().sum().item() <= 16229
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9548]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 173, 189, 375]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16179 <= predictions[1].mask.cpu().sum().item() <= 16229


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_triton_preprocess_output_matches_reference_preprocess(
    monkeypatch: pytest.MonkeyPatch,
    rfdetr_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton preprocessing parity")

    from inference_models.models.rfdetr import triton_preprocess_runtime
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_trt import (
        RFDetrForInstanceSegmentationTRT,
    )

    model_package = os.getenv(
        "RFDETR_SEG_TRT_PACKAGE_PATH",
        rfdetr_seg_asl_trt_package,
    )
    try:
        model = RFDetrForInstanceSegmentationTRT.from_pretrained(
            model_name_or_path=model_package,
            engine_host_code_allowed=True,
        )
    except CorruptedModelPackageError as error:
        if "Platform specific tag mismatch" in str(error):
            pytest.skip("TRT engine package is not compatible with this platform")
        raise

    monkeypatch.setattr(triton_preprocess_runtime, "_FAST_PATH_ENABLED", False)
    reference_predictions = model(asl_image_numpy)

    original_triton_preprocess = (
        triton_preprocess_runtime.triton_preprocess_rfdetr_stretch_two_pass_preallocated
    )
    triton_calls = {"count": 0}

    def counting_triton_preprocess(*args, **kwargs):
        triton_calls["count"] += 1
        return original_triton_preprocess(*args, **kwargs)

    monkeypatch.setattr(
        triton_preprocess_runtime,
        "triton_preprocess_rfdetr_stretch_two_pass_preallocated",
        counting_triton_preprocess,
    )
    monkeypatch.setattr(triton_preprocess_runtime, "_FAST_PATH_ENABLED", True)
    triton_predictions = model(asl_image_numpy)

    assert triton_calls["count"] == 1
    _assert_instance_segmentation_predictions_match(
        actual=triton_predictions,
        expected=reference_predictions,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_cudagraph_output_matches_non_cudagraph_output(
    rfdetr_seg_nano_t4_trt_package: str,
    snake_image_numpy: np.ndarray,
    dog_image_numpy: np.ndarray,
) -> None:
    from inference_models import AutoModel
    from inference_models.models.common.trt import TRTCudaGraphCache

    trt_cuda_graph_cache = TRTCudaGraphCache(capacity=16)
    model = AutoModel.from_pretrained(
        model_id_or_path=rfdetr_seg_nano_t4_trt_package,
        device=torch.device("cuda:0"),
        trt_cuda_graph_cache=trt_cuda_graph_cache,
    )

    pre_processed_1, _ = model.pre_process(snake_image_numpy)
    pre_processed_2, _ = model.pre_process(dog_image_numpy)

    outputs = []
    for pre_processed in [pre_processed_1, pre_processed_2]:
        no_graph = model.forward(pre_processed, disable_cuda_graphs=True)
        capture_graph = model.forward(pre_processed)
        replay_graph = model.forward(pre_processed)
        outputs.append((no_graph, capture_graph, replay_graph))

    for image_outputs in outputs:
        no_graph, capture_graph, replay_graph = image_outputs
        for result_idx in range(3):
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

    for execution_branch_idx in range(3):
        for result_idx in range(3):
            assert not torch.allclose(
                outputs[0][execution_branch_idx][result_idx],
                outputs[1][execution_branch_idx][result_idx],
                atol=1e-6,
            )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_per_class_confidence_filters_detections(
    rfdetr_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_trt import (
        RFDetrForInstanceSegmentationTRT,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = RFDetrForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=rfdetr_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.3,
        per_class_confidence={class_names[20]: 1.01},
    )
    predictions = model(asl_image_numpy, confidence="best")
    assert predictions[0].class_id.numel() == 0
