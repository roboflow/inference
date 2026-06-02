import numpy as np
import pytest
import torch

from inference_models.entities import ImageDimensions
from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.rfdetr import common as rfdetr_common
from inference_models.models.rfdetr import triton_postprocess
from inference_models.models.rfdetr.class_remapping import ClassesReMapping
from inference_models.models.rfdetr.common import (
    _post_process_single_instance_segmentation_result_to_rle_masks_classic,
    post_process_instance_segmentation_results_to_rle_masks,
)
from inference_models.models.rfdetr.triton_postprocess import (
    _INTERPOLATION_WEIGHT_CACHE,
    _MAX_INTERPOLATION_WEIGHT_CACHE_ENTRIES,
    _get_interpolation_weights,
    _supports_triton_postprocess_path,
    _unsupported_triton_postprocess_reason,
    post_process_single_instance_segmentation_result_to_rle_masks_triton,
)


def _metadata(
    height: int = 64,
    width: int = 64,
    padding: tuple = (0, 0, 0, 0),
    static_crop_offset: tuple = (0, 0),
    size_after_pre_processing: tuple = None,
) -> PreProcessingMetadata:
    size = ImageDimensions(height=height, width=width)
    pad_left, pad_top, pad_right, pad_bottom = padding
    offset_x, offset_y = static_crop_offset
    preprocessed_height, preprocessed_width = size_after_pre_processing or (
        height,
        width,
    )
    preprocessed_size = ImageDimensions(
        height=preprocessed_height,
        width=preprocessed_width,
    )
    return PreProcessingMetadata(
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        original_size=size,
        size_after_pre_processing=preprocessed_size,
        inference_size=size,
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=offset_x,
            offset_y=offset_y,
            crop_width=preprocessed_width,
            crop_height=preprocessed_height,
        ),
        nonsquare_intermediate_size=None,
    )


def _class_mapping(device: torch.device, num_classes: int = 2) -> ClassesReMapping:
    return ClassesReMapping(
        remaining_class_ids=torch.arange(num_classes, dtype=torch.int64, device=device),
        class_mapping=torch.arange(num_classes, dtype=torch.int64, device=device),
    )


def _single_detection_inputs(device: torch.device):
    bboxes = torch.tensor(
        [
            [0.50, 0.50, 0.50, 0.50],
            [0.25, 0.25, 0.20, 0.20],
        ],
        dtype=torch.float32,
        device=device,
    )
    logits = torch.tensor(
        [
            [4.0, -4.0],
            [-4.0, -4.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    masks = torch.full((2, 8, 8), -2.0, dtype=torch.float32, device=device)
    masks[0, 2:6, 2:6] = 2.0
    return bboxes, logits, masks


def _support_kwargs(
    num_queries: int = 2,
    num_classes: int = 2,
    mask_size: tuple = (8, 8),
) -> dict:
    device = torch.device("cpu")
    return {
        "image_bboxes": torch.full(
            (num_queries, 4),
            0.5,
            dtype=torch.float32,
            device=device,
        ),
        "image_scores": torch.full(
            (num_queries, num_classes),
            0.1,
            dtype=torch.float32,
            device=device,
        ),
        "image_masks": torch.zeros(
            (num_queries, *mask_size),
            dtype=torch.float32,
            device=device,
        ),
        "image_meta": _metadata(),
        "threshold": 0.4,
        "classes_re_mapping": _class_mapping(device, num_classes=num_classes),
    }


def _assert_detections_equal(actual, expected) -> None:
    torch.testing.assert_close(actual.xyxy.cpu(), expected.xyxy.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(
        actual.confidence.cpu(), expected.confidence.cpu(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.class_id.cpu(), expected.class_id.cpu(), rtol=0, atol=0
    )
    actual_mask = coco_rle_masks_to_numpy_mask(actual.mask)
    expected_mask = coco_rle_masks_to_numpy_mask(expected.mask)
    np.testing.assert_array_equal(actual_mask, expected_mask)


def _expected_classic_result(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    metadata: PreProcessingMetadata,
    threshold,
    classes_re_mapping,
    num_classes: int = 2,
):
    return _post_process_single_instance_segmentation_result_to_rle_masks_classic(
        image_bboxes=bboxes,
        image_logits=torch.sigmoid(logits),
        image_masks=masks,
        image_meta=metadata,
        threshold=threshold,
        num_classes=num_classes,
        classes_re_mapping=classes_re_mapping,
    )


def _batched_inputs(device: torch.device):
    bboxes = torch.tensor(
        [
            [
                [0.50, 0.50, 0.50, 0.50],
                [0.25, 0.25, 0.20, 0.20],
                [0.75, 0.75, 0.15, 0.15],
            ],
            [
                [0.50, 0.50, 0.40, 0.40],
                [0.25, 0.25, 0.20, 0.20],
                [0.75, 0.75, 0.15, 0.15],
            ],
            [
                [0.25, 0.75, 0.20, 0.20],
                [0.75, 0.25, 0.20, 0.20],
                [0.50, 0.50, 0.30, 0.30],
            ],
        ],
        dtype=torch.float32,
        device=device,
    )
    logits = torch.full((3, 3, 2), -4.0, dtype=torch.float32, device=device)
    logits[0, 0, 0] = 4.0
    logits[0, 1, 1] = 3.0
    logits[2, 2, 0] = 2.0

    masks = torch.full((3, 3, 8, 8), -2.0, dtype=torch.float32, device=device)
    masks[0, 0, 2:6, 2:6] = 2.0
    masks[0, 1, 1:3, 1:3] = 2.0
    masks[1, 0, 3:5, 3:5] = 2.0
    masks[2, 2, 3:6, 3:6] = 2.0
    return bboxes, logits, masks


def _assert_batched_results_match_classic(
    actual,
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    metadata,
    threshold,
    classes_re_mapping,
    num_classes: int = 2,
) -> None:
    assert len(actual) == bboxes.shape[0]
    for image_index, actual_detections in enumerate(actual):
        expected = _expected_classic_result(
            bboxes=bboxes[image_index],
            logits=logits[image_index],
            masks=masks[image_index],
            metadata=metadata[image_index],
            threshold=threshold,
            classes_re_mapping=classes_re_mapping,
            num_classes=num_classes,
        )
        _assert_detections_equal(actual_detections, expected)


def test_rfdetr_triton_postproc_flag_false_bypasses_triton(monkeypatch) -> None:
    monkeypatch.setattr(rfdetr_common, "_TRITON_POSTPROC_ENABLED", False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Triton postproc should be disabled")

    monkeypatch.setattr(
        rfdetr_common,
        "post_process_single_instance_segmentation_result_to_rle_masks_triton",
        fail_if_called,
    )

    bboxes, logits, masks = _single_detection_inputs(torch.device("cpu"))
    results = post_process_instance_segmentation_results_to_rle_masks(
        bboxes=bboxes.unsqueeze(0),
        logits=logits.unsqueeze(0),
        masks=masks.unsqueeze(0),
        pre_processing_meta=[_metadata()],
        threshold=0.4,
        num_classes=2,
        classes_re_mapping=_class_mapping(torch.device("cpu")),
    )

    assert len(results) == 1
    assert results[0].confidence.shape == (1,)


def test_rfdetr_triton_postproc_flag_true_uses_triton_result(monkeypatch) -> None:
    monkeypatch.setattr(rfdetr_common, "_TRITON_POSTPROC_ENABLED", True)
    sentinel = object()

    def return_sentinel(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(
        rfdetr_common,
        "post_process_single_instance_segmentation_result_to_rle_masks_triton",
        return_sentinel,
    )

    bboxes, logits, masks = _single_detection_inputs(torch.device("cpu"))
    results = post_process_instance_segmentation_results_to_rle_masks(
        bboxes=bboxes.unsqueeze(0),
        logits=logits.unsqueeze(0),
        masks=masks.unsqueeze(0),
        pre_processing_meta=[_metadata()],
        threshold=0.4,
        num_classes=2,
        classes_re_mapping=_class_mapping(torch.device("cpu")),
    )

    assert results == [sentinel]


def test_rfdetr_triton_postproc_flag_true_uses_triton_per_image_for_batches(
    monkeypatch,
) -> None:
    monkeypatch.setattr(rfdetr_common, "_TRITON_POSTPROC_ENABLED", True)
    sentinels = [object(), object(), object()]
    calls = []

    def return_sentinel(**kwargs):
        calls.append(kwargs["image_bboxes"].shape)
        return sentinels[len(calls) - 1]

    monkeypatch.setattr(
        rfdetr_common,
        "post_process_single_instance_segmentation_result_to_rle_masks_triton",
        return_sentinel,
    )

    device = torch.device("cpu")
    bboxes, logits, masks = _batched_inputs(device)
    results = post_process_instance_segmentation_results_to_rle_masks(
        bboxes=bboxes,
        logits=logits,
        masks=masks,
        pre_processing_meta=[_metadata(), _metadata(), _metadata()],
        threshold=0.4,
        num_classes=2,
        classes_re_mapping=_class_mapping(device),
    )

    assert results == sentinels
    assert calls == [torch.Size([3, 4]), torch.Size([3, 4]), torch.Size([3, 4])]


def test_rfdetr_triton_postproc_reports_triton_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(triton_postprocess, "triton", None)

    reason = _unsupported_triton_postprocess_reason(**_support_kwargs())

    assert reason == "triton_unavailable"


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("no_class_mapping", "class_remapping_required"),
        ("tensor_threshold", "tensor_threshold_unsupported"),
        ("invalid_tensor_rank", "invalid_tensor_rank"),
        ("shape_mismatch", "shape_mismatch"),
        ("class_mapping_too_small", "class_mapping_too_small"),
        ("input_size_exceeds_limits", "input_size_exceeds_triton_limits"),
        ("padding", "padding_unsupported"),
        ("static_crop", "static_crop_unsupported"),
        ("resize_metadata", "resize_metadata_unsupported"),
        ("cpu_device", "cuda_device_required"),
    ],
)
def test_rfdetr_triton_postproc_unsupported_reason_matrix(
    monkeypatch,
    case: str,
    expected_reason: str,
) -> None:
    monkeypatch.setattr(triton_postprocess, "triton", object())
    kwargs = _support_kwargs()

    if case == "no_class_mapping":
        kwargs["classes_re_mapping"] = None
    elif case == "tensor_threshold":
        kwargs["threshold"] = torch.tensor([0.4, 0.4])
    elif case == "invalid_tensor_rank":
        kwargs["image_scores"] = kwargs["image_scores"][None]
    elif case == "shape_mismatch":
        kwargs["image_bboxes"] = kwargs["image_bboxes"][:1]
    elif case == "class_mapping_too_small":
        kwargs["classes_re_mapping"] = _class_mapping(torch.device("cpu"), 1)
    elif case == "input_size_exceeds_limits":
        kwargs = _support_kwargs(num_queries=129, num_classes=128)
    elif case == "padding":
        kwargs["image_meta"] = _metadata(padding=(1, 0, 0, 0))
    elif case == "static_crop":
        kwargs["image_meta"] = _metadata(static_crop_offset=(1, 0))
    elif case == "resize_metadata":
        kwargs["image_meta"] = _metadata(size_after_pre_processing=(32, 64))

    reason = _unsupported_triton_postprocess_reason(**kwargs)

    assert reason == expected_reason
    assert not _supports_triton_postprocess_path(**kwargs)


@pytest.mark.parametrize("case", ["no_class_mapping", "tensor_threshold", "padding"])
def test_rfdetr_triton_postproc_unsupported_cases_fallback_to_classic(
    monkeypatch,
    case: str,
) -> None:
    monkeypatch.setattr(rfdetr_common, "_TRITON_POSTPROC_ENABLED", True)
    calls = 0
    real_triton_postprocess = (
        rfdetr_common.post_process_single_instance_segmentation_result_to_rle_masks_triton
    )

    def spy_triton_postprocess(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_triton_postprocess(*args, **kwargs)

    monkeypatch.setattr(
        rfdetr_common,
        "post_process_single_instance_segmentation_result_to_rle_masks_triton",
        spy_triton_postprocess,
    )

    device = torch.device("cpu")
    bboxes, logits, masks = _single_detection_inputs(device)
    metadata = _metadata()
    threshold = 0.4
    classes_re_mapping = _class_mapping(device)
    if case == "no_class_mapping":
        classes_re_mapping = None
    elif case == "tensor_threshold":
        threshold = torch.tensor([0.4, 0.4], dtype=torch.float32, device=device)
    elif case == "padding":
        metadata = _metadata(padding=(1, 0, 0, 0))

    expected = _expected_classic_result(
        bboxes=bboxes,
        logits=logits,
        masks=masks,
        metadata=metadata,
        threshold=threshold,
        classes_re_mapping=classes_re_mapping,
    )
    actual = post_process_instance_segmentation_results_to_rle_masks(
        bboxes=bboxes.unsqueeze(0),
        logits=logits.unsqueeze(0),
        masks=masks.unsqueeze(0),
        pre_processing_meta=[metadata],
        threshold=threshold,
        num_classes=2,
        classes_re_mapping=classes_re_mapping,
    )[0]

    assert calls == 1
    _assert_detections_equal(actual, expected)


def test_rfdetr_batched_rle_postprocess_matches_classic_for_mixed_counts_and_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setattr(rfdetr_common, "_TRITON_POSTPROC_ENABLED", False)
    device = torch.device("cpu")
    bboxes, logits, masks = _batched_inputs(device)
    metadata = [
        _metadata(),
        _metadata(padding=(1, 0, 0, 0)),
        _metadata(),
    ]
    threshold = 0.4
    classes_re_mapping = _class_mapping(device)

    actual = post_process_instance_segmentation_results_to_rle_masks(
        bboxes=bboxes,
        logits=logits,
        masks=masks,
        pre_processing_meta=metadata,
        threshold=threshold,
        num_classes=2,
        classes_re_mapping=classes_re_mapping,
    )

    _assert_batched_results_match_classic(
        actual=actual,
        bboxes=bboxes,
        logits=logits,
        masks=masks,
        metadata=metadata,
        threshold=threshold,
        classes_re_mapping=classes_re_mapping,
    )
    assert [result.confidence.shape[0] for result in actual] == [2, 0, 1]


def test_rfdetr_batched_rle_postprocess_matches_classic_for_tensor_threshold_and_unmapped_classes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(rfdetr_common, "_TRITON_POSTPROC_ENABLED", False)
    device = torch.device("cpu")
    bboxes, logits, masks = _batched_inputs(device)
    logits[2, 0, 1] = 5.0
    metadata = [
        _metadata(),
        _metadata(padding=(1, 0, 0, 0)),
        _metadata(),
    ]
    threshold = torch.tensor([0.4, 0.4], dtype=torch.float32, device=device)
    classes_re_mapping = ClassesReMapping(
        remaining_class_ids=torch.tensor([0], dtype=torch.int64, device=device),
        class_mapping=torch.tensor([0, -1], dtype=torch.int64, device=device),
    )

    actual = post_process_instance_segmentation_results_to_rle_masks(
        bboxes=bboxes,
        logits=logits,
        masks=masks,
        pre_processing_meta=metadata,
        threshold=threshold,
        num_classes=2,
        classes_re_mapping=classes_re_mapping,
    )

    _assert_batched_results_match_classic(
        actual=actual,
        bboxes=bboxes,
        logits=logits,
        masks=masks,
        metadata=metadata,
        threshold=threshold,
        classes_re_mapping=classes_re_mapping,
    )
    assert [result.confidence.shape[0] for result in actual] == [1, 0, 1]
    assert all(result.class_id.tolist() == [0] for result in (actual[0], actual[2]))


def test_rfdetr_triton_postproc_interpolation_weight_cache_is_bounded() -> None:
    _INTERPOLATION_WEIGHT_CACHE.clear()
    try:
        for output_size in range(8, 8 + _MAX_INTERPOLATION_WEIGHT_CACHE_ENTRIES + 3):
            _get_interpolation_weights(
                src_size=8,
                output_size=output_size,
                device=torch.device("cpu"),
                axis="height",
            )

        assert (
            len(_INTERPOLATION_WEIGHT_CACHE) <= _MAX_INTERPOLATION_WEIGHT_CACHE_ENTRIES
        )
    finally:
        _INTERPOLATION_WEIGHT_CACHE.clear()


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton_postprocess.triton is None,
    reason="CUDA and Triton are required",
)
def test_rfdetr_triton_postproc_matches_classic_rle_path() -> None:
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")
    bboxes_cpu, logits_cpu, masks_cpu = _single_detection_inputs(cpu)
    scores_cpu = torch.sigmoid(logits_cpu)
    metadata = _metadata()
    expected = _post_process_single_instance_segmentation_result_to_rle_masks_classic(
        image_bboxes=bboxes_cpu,
        image_logits=scores_cpu,
        image_masks=masks_cpu,
        image_meta=metadata,
        threshold=0.4,
        num_classes=2,
        classes_re_mapping=_class_mapping(cpu),
    )
    cuda_kwargs = {
        "image_bboxes": bboxes_cpu.to(cuda),
        "image_scores": scores_cpu.to(cuda),
        "image_masks": masks_cpu.to(cuda),
        "image_meta": metadata,
        "threshold": 0.4,
        "classes_re_mapping": _class_mapping(cuda),
    }

    assert _unsupported_triton_postprocess_reason(**cuda_kwargs) is None
    assert _supports_triton_postprocess_path(**cuda_kwargs)
    actual = post_process_single_instance_segmentation_result_to_rle_masks_triton(
        **cuda_kwargs
    )

    assert actual is not None
    _assert_detections_equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton_postprocess.triton is None,
    reason="CUDA and Triton are required",
)
def test_rfdetr_triton_postproc_topk_retry_matches_classic_rle_path() -> None:
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")
    bboxes_cpu, logits_cpu, masks_cpu = _single_detection_inputs(cpu)
    logits_cpu[0, 0] = 5.0
    logits_cpu[0, 1] = 4.0
    scores_cpu = torch.sigmoid(logits_cpu)
    metadata = _metadata()
    expected = _post_process_single_instance_segmentation_result_to_rle_masks_classic(
        image_bboxes=bboxes_cpu,
        image_logits=scores_cpu,
        image_masks=masks_cpu,
        image_meta=metadata,
        threshold=0.4,
        num_classes=2,
        classes_re_mapping=_class_mapping(cpu),
    )
    cuda_kwargs = {
        "image_bboxes": bboxes_cpu.to(cuda),
        "image_scores": scores_cpu.to(cuda),
        "image_masks": masks_cpu.to(cuda),
        "image_meta": metadata,
        "threshold": 0.4,
        "classes_re_mapping": _class_mapping(cuda),
    }

    assert expected.confidence.shape == (2,)
    assert _unsupported_triton_postprocess_reason(**cuda_kwargs) is None
    actual = post_process_single_instance_segmentation_result_to_rle_masks_triton(
        **cuda_kwargs
    )

    assert actual is not None
    _assert_detections_equal(actual, expected)
