import os

import numpy as np
import pytest
import supervision as sv
import torch

from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask
from inference_models.models.common.roboflow.model_packages import StaticCrop


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

    # This test exercises the runtime fast path directly; initialize the
    # adapter-level gate/runtime even when the process default leaves it off.
    model._fast_preprocess_enabled = True
    model._fast_preprocess_runtime = triton_preprocess_runtime.FastPreprocessRuntime(
        device=model._device
    )
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


@pytest.mark.slow
@pytest.mark.trt_extras
@pytest.mark.gpu_only
@pytest.mark.parametrize("factor", [1.0, 0.5, 0.0])
@pytest.mark.parametrize(
    "crop_start", [None, 0, 25], ids=["uncropped", "origin-crop", "offset-crop"]
)
def test_mask_resolution_and_crop_round_trip(
    rfdetr_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    factor: float,
    crop_start: int | None,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_trt import (
        RFDetrForInstanceSegmentationTRT,
    )

    model = RFDetrForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=rfdetr_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )
    image = np.pad(
        asl_image_numpy,
        ((0, asl_image_numpy.shape[0] % 2), (0, asl_image_numpy.shape[1] % 2), (0, 0)),
    )
    height, width = image.shape[:2]
    scene = image
    offset_x = offset_y = 0
    crop_config = None
    if crop_start is not None:
        offset_y = 0 if crop_start == 0 else height // 2
        offset_x = 0 if crop_start == 0 else width // 2
        scene = np.zeros((2 * height, 2 * width, 3), dtype=image.dtype)
        scene[offset_y : offset_y + height, offset_x : offset_x + width] = image
        crop_config = StaticCrop(
            enabled=True,
            x_min=crop_start,
            y_min=crop_start,
            x_max=crop_start + 50,
            y_max=crop_start + 50,
        )
    monkeypatch.setattr(
        model._inference_config.image_pre_processing, "static_crop", crop_config
    )

    # when
    dense = model(
        scene, confidence=0.25, mask_format="dense", masks_resolution_factor=factor
    )[0]
    rle = model(
        scene, confidence=0.25, mask_format="rle", masks_resolution_factor=factor
    )[0]
    decoded = coco_rle_masks_to_torch_mask(
        instances_masks=rle.mask, device=torch.device("cpu")
    )
    detections = dense.to_supervision()

    # then
    offset = np.array([offset_x, offset_y, offset_x, offset_y])
    expected_boxes = np.array([[63, 172, 188, 374]]) + offset
    np.testing.assert_allclose(detections.xyxy, expected_boxes, atol=5, rtol=0)
    np.testing.assert_array_equal(detections.class_id, [20])
    np.testing.assert_allclose(detections.confidence, [0.9491], atol=0.01, rtol=0)
    assert dense.image_size == scene.shape[:2]
    assert dense.mask_frame_size == scene.shape[:2]
    assert rle.image_size == scene.shape[:2]
    assert rle.mask_frame_size == scene.shape[:2]
    if factor == 1.0:
        assert dense.mask_size == scene.shape[:2]
    else:
        assert dense.mask_size != scene.shape[:2]
    assert rle.mask_size == dense.mask_size

    grid_scale = np.array(
        [dense.mask_size[1] / scene.shape[1], dense.mask_size[0] / scene.shape[0]] * 2
    )
    np.testing.assert_allclose(
        dense.xyxy.cpu().numpy(),
        expected_boxes * grid_scale,
        atol=2 * max(grid_scale),
        rtol=0,
    )
    if dense.mask_size != scene.shape[:2]:
        assert dense.xyxy.is_floating_point()
    assert detections.mask.shape == (1, *scene.shape[:2])
    # Reuse the full-resolution area baseline in test_trt_package_numpy.
    expected_area = 16046
    assert detections.mask.sum() == pytest.approx(expected_area, rel=0.05)
    crop_masks = detections.mask[
        :, offset_y : offset_y + height, offset_x : offset_x + width
    ]
    assert crop_masks.sum() == detections.mask.sum()
    torch.testing.assert_close(dense.xyxy.cpu(), rle.xyxy.cpu())
    torch.testing.assert_close(dense.class_id.cpu(), rle.class_id.cpu())
    torch.testing.assert_close(dense.confidence.cpu(), rle.confidence.cpu())
    np.testing.assert_array_equal(decoded.numpy(), dense.mask.cpu().numpy())
    np.testing.assert_array_equal(detections.mask, rle.to_supervision().mask)
    sv.MaskAnnotator().annotate(scene.copy(), detections)
