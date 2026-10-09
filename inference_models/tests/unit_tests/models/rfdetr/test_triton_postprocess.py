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
    _post_process_single_instance_segmentation_result_to_rle_masks,
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


def _expected_result(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    metadata: PreProcessingMetadata,
    threshold,
    classes_re_mapping,
    num_classes: int = 2,
    masks_resolution_factor: float = 1.0,
):
    return _post_process_single_instance_segmentation_result_to_rle_masks(
        image_bboxes=bboxes,
        image_logits=torch.sigmoid(logits),
        image_masks=masks,
        image_meta=metadata,
        threshold=threshold,
        num_classes=num_classes,
        classes_re_mapping=classes_re_mapping,
        masks_resolution_factor=masks_resolution_factor,
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


def _assert_batched_results_match_reference(
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
        expected = _expected_result(
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
        kwargs = _support_kwargs(mask_size=(193, 193))
    elif case == "padding":
        kwargs["image_meta"] = _metadata(padding=(1, 0, 0, 0))
    elif case == "static_crop":
        kwargs["image_meta"] = _metadata(static_crop_offset=(1, 0))
    elif case == "resize_metadata":
        kwargs["image_meta"] = _metadata(size_after_pre_processing=(32, 64))

    reason = _unsupported_triton_postprocess_reason(**kwargs)

    assert reason == expected_reason
    assert not _supports_triton_postprocess_path(**kwargs)


def test_rfdetr_triton_postproc_accepts_2xlarge_shape_limits(monkeypatch) -> None:
    monkeypatch.setattr(triton_postprocess, "triton", object())
    device = torch.device("cpu")
    num_queries = 300
    num_classes = 91
    kwargs = {
        "image_bboxes": torch.empty(
            (num_queries, 4),
            dtype=torch.float32,
            device=device,
        ),
        "image_scores": torch.empty(
            (num_queries, num_classes),
            dtype=torch.float32,
            device=device,
        ),
        "image_masks": torch.empty(
            (num_queries, 192, 192),
            dtype=torch.float32,
            device=device,
        ),
        "image_meta": _metadata(height=1080, width=1920),
        "threshold": 0.4,
        "classes_re_mapping": _class_mapping(device, num_classes=num_classes),
    }

    reason = _unsupported_triton_postprocess_reason(**kwargs)

    assert reason == "cuda_device_required"


@pytest.mark.parametrize("factor", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("case", ["no_class_mapping", "tensor_threshold", "padding"])
def test_rfdetr_triton_postproc_unsupported_cases_use_reference_path(
    monkeypatch,
    case: str,
    factor: float,
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

    expected = _expected_result(
        bboxes=bboxes,
        logits=logits,
        masks=masks,
        metadata=metadata,
        threshold=threshold,
        classes_re_mapping=classes_re_mapping,
        masks_resolution_factor=factor,
    )
    actual = post_process_instance_segmentation_results_to_rle_masks(
        bboxes=bboxes.unsqueeze(0),
        logits=logits.unsqueeze(0),
        masks=masks.unsqueeze(0),
        pre_processing_meta=[metadata],
        threshold=threshold,
        num_classes=2,
        classes_re_mapping=classes_re_mapping,
        masks_resolution_factor=factor,
    )[0]

    assert calls == 1
    _assert_detections_equal(actual, expected)


def test_rfdetr_batched_rle_postprocess_matches_reference_for_mixed_counts_and_metadata(
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

    _assert_batched_results_match_reference(
        actual=actual,
        bboxes=bboxes,
        logits=logits,
        masks=masks,
        metadata=metadata,
        threshold=threshold,
        classes_re_mapping=classes_re_mapping,
    )
    assert [result.confidence.shape[0] for result in actual] == [2, 0, 1]


def test_rfdetr_batched_rle_postprocess_matches_reference_for_tensor_threshold_and_unmapped_classes(
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

    _assert_batched_results_match_reference(
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
def test_rfdetr_triton_postproc_matches_reference_rle_path() -> None:
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")
    bboxes_cpu, logits_cpu, masks_cpu = _single_detection_inputs(cpu)
    scores_cpu = torch.sigmoid(logits_cpu)
    metadata = _metadata()
    expected = _post_process_single_instance_segmentation_result_to_rle_masks(
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
def test_rfdetr_triton_postproc_matches_reference_with_large_source_mask() -> None:
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")
    bboxes_cpu = torch.tensor(
        [
            [0.50, 0.50, 0.50, 0.50],
            [0.25, 0.25, 0.20, 0.20],
        ],
        dtype=torch.float32,
        device=cpu,
    )
    logits_cpu = torch.tensor(
        [
            [4.0, -4.0],
            [-4.0, -4.0],
        ],
        dtype=torch.float32,
        device=cpu,
    )
    masks_cpu = torch.full((2, 96, 96), -2.0, dtype=torch.float32, device=cpu)
    masks_cpu[0, 24:72, 24:72] = 2.0
    scores_cpu = torch.sigmoid(logits_cpu)
    metadata = _metadata(height=128, width=128)
    expected = _post_process_single_instance_segmentation_result_to_rle_masks(
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
    actual = post_process_single_instance_segmentation_result_to_rle_masks_triton(
        **cuda_kwargs
    )

    assert actual is not None
    _assert_detections_equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton_postprocess.triton is None,
    reason="CUDA and Triton are required",
)
def test_rfdetr_triton_postproc_topk_retry_matches_reference_rle_path() -> None:
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")
    bboxes_cpu, logits_cpu, masks_cpu = _single_detection_inputs(cpu)
    logits_cpu[0, 0] = 5.0
    logits_cpu[0, 1] = 4.0
    scores_cpu = torch.sigmoid(logits_cpu)
    metadata = _metadata()
    expected = _post_process_single_instance_segmentation_result_to_rle_masks(
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


@pytest.mark.parametrize("factor", [0.0, 0.25, 0.5, 1.0])
def test_reduced_mask_resolution_reaches_the_device_check(monkeypatch, factor) -> None:
    # Stand in for Triton on CPU CI to verify that every valid factor passes
    # shape/metadata validation before the CUDA requirement is checked.
    monkeypatch.setattr(triton_postprocess, "triton", object())
    reason = _unsupported_triton_postprocess_reason(
        **_support_kwargs(), masks_resolution_factor=factor
    )

    assert reason == "cuda_device_required"


@pytest.mark.parametrize("factor", [0.25, 0.5])
def test_triton_dispatcher_forwards_the_resolution_factor(monkeypatch, factor) -> None:
    # given
    # the gate only fires if the dispatcher passes the factor down. Testing the
    # gate function directly cannot detect a dispatcher that drops it.
    monkeypatch.setattr(rfdetr_common, "_TRITON_POSTPROC_ENABLED", True)
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return None  # force the eager fallback

    monkeypatch.setattr(
        rfdetr_common,
        "post_process_single_instance_segmentation_result_to_rle_masks_triton",
        spy,
    )
    device = torch.device("cpu")
    bboxes, logits, masks = _single_detection_inputs(device)

    # when
    post_process_instance_segmentation_results_to_rle_masks(
        bboxes=bboxes.unsqueeze(0),
        logits=logits.unsqueeze(0),
        masks=masks.unsqueeze(0),
        pre_processing_meta=[_metadata()],
        threshold=0.4,
        num_classes=2,
        classes_re_mapping=_class_mapping(device),
        masks_resolution_factor=factor,
    )

    # then
    assert seen.get("masks_resolution_factor") == factor


@pytest.mark.parametrize("shared_queries", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("max_detections", [None, 0, 1])
def test_sparse_assembly_preserves_independent_mask_grid(
    shared_queries, empty, max_detections
) -> None:
    metadata = np.zeros((2, triton_postprocess._HEADER_SIZE), dtype=np.float32)
    if not empty:
        metadata[:, 0] = 1
    metadata[:, 1] = [3, 8]
    metadata[:, 2] = [0.9, 0.8]
    metadata[:, 3:7] = [24, 16, 73, 49]
    metadata[:, 9] = 0  # both class rows refer to the same query
    records = np.array([[2, 0, 0], [0, 6, 8], [0, 11, 13]], dtype=np.int32)
    kwargs = dict(
        max_total_runs=2,
        height=5,
        width=7,
        image_size=(65, 97),
        max_detections=max_detections,
    )
    if shared_queries:
        result = triton_postprocess._instance_detections_from_sparse_query_records(
            class_metadata_host=metadata, records_host=records, **kwargs
        )
    else:
        result = triton_postprocess._instance_detections_from_sparse_records(
            metadata_host=metadata, records=torch.from_numpy(records), **kwargs
        )

    assert result is not None
    assert result.image_size == result.mask_frame_size == (65, 97)
    assert result.mask.image_size == (65, 97)
    assert result.mask_size == result.mask.mask_size == (5, 7)
    count = 0 if empty else min(2, max_detections if max_detections is not None else 2)
    assert len(result) == count
    if count:
        assert result.xyxy.tolist() == [[24, 16, 73, 49]] * count
        decoded = coco_rle_masks_to_numpy_mask(result.mask)
        expected = np.zeros((5, 7), dtype=bool)
        expected[1:3, 1:3] = True
        np.testing.assert_array_equal(decoded[0], expected)
        if shared_queries and count == 2:
            np.testing.assert_array_equal(decoded[1], expected)


@pytest.mark.parametrize("empty", [False, True])
def test_deferred_assembly_preserves_grid_until_finalization(
    monkeypatch, empty
) -> None:
    from unittest.mock import Mock

    from inference_models.models.base.async_handoff import (
        get_deferred_postprocess_finalizer,
    )

    metadata = torch.zeros((1, triton_postprocess._HEADER_SIZE), dtype=torch.float32)
    metadata[0, :7] = torch.tensor([not empty, 3, 0.9, 24, 16, 73, 49])
    records = torch.tensor([[1, 0, 0], [0, 6, 8]], dtype=torch.int32)
    done_event = Mock()
    release = Mock()
    monkeypatch.setattr(triton_postprocess, "_release_pinned_host_buffer", release)
    placeholder = (
        triton_postprocess._deferred_instance_detections_from_sparse_query_records(
            class_metadata_host=metadata,
            records_host=records,
            keepalive_tensors=(),
            done_event=done_event,
            outputs_consumed_event=Mock(),
            max_total_runs=1,
            height=5,
            width=7,
            image_size=(65, 97),
            max_detections=1,
        )
    )
    assert placeholder.mask_size == (5, 7)
    assert placeholder.image_size == placeholder.mask_frame_size == (65, 97)
    done_event.synchronize.assert_not_called()
    result = get_deferred_postprocess_finalizer(placeholder)()
    done_event.synchronize.assert_called_once()
    assert release.call_count == 2
    assert result.mask_size == (5, 7)
    assert result.image_size == result.mask_frame_size == (65, 97)
    assert len(result) == (0 if empty else 1)


@pytest.mark.parametrize("factor", [-0.1, 1.1, float("nan"), float("inf")])
def test_triton_rejects_invalid_mask_resolution_factor(monkeypatch, factor) -> None:
    monkeypatch.setattr(triton_postprocess, "triton", object())
    with pytest.raises(ValueError, match="finite and in"):
        _unsupported_triton_postprocess_reason(
            **_support_kwargs(), masks_resolution_factor=factor
        )


@pytest.mark.parametrize("factor", [0.0, 0.25, 0.5, 1.0])
def test_triton_downsampling_uses_the_reference_antialias_path(
    monkeypatch, factor
) -> None:
    monkeypatch.setattr(triton_postprocess, "triton", object())
    kwargs = _support_kwargs(mask_size=(8, 8))
    kwargs["image_meta"] = _metadata(height=4, width=6)
    reason = _unsupported_triton_postprocess_reason(
        **kwargs, masks_resolution_factor=factor
    )
    assert reason == (
        "cuda_device_required" if factor == 0 else "mask_downsampling_unsupported"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton_postprocess.triton is None,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("factor", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("image_size", [(64, 64), (63, 95)])
@pytest.mark.parametrize(
    "mode", ["single", "multiclass", "deferred", "empty", "deferred_empty"]
)
def test_triton_mask_grid_matches_reference_on_cuda(factor, image_size, mode) -> None:
    from inference_models.models.base.async_handoff import (
        get_deferred_postprocess_finalizer,
    )
    from inference_models.models.common.roboflow.post_processing import (
        resolve_mask_target_size,
    )

    device = torch.device("cuda")
    bboxes, logits, masks = _single_detection_inputs(device)
    # An asymmetric source and noncontiguous views expose swapped axes/strides.
    masks[0, 1:5, 2:7] = 3.0
    masks = masks.transpose(1, 2)
    if mode in ("multiclass", "deferred"):
        logits[0, 1] = 3.0
        logits[1, 0] = 2.0
    elif mode in ("empty", "deferred_empty"):
        logits.fill_(-4.0)
    scores = logits.sigmoid().T.contiguous().T
    metadata = _metadata(height=image_size[0], width=image_size[1])
    mapping = _class_mapping(device)
    mapping.class_mapping[:] = torch.tensor([3, 8], device=device)
    expected = _post_process_single_instance_segmentation_result_to_rle_masks(
        image_bboxes=bboxes,
        image_logits=scores,
        image_masks=masks,
        image_meta=metadata,
        threshold=0.4,
        num_classes=2,
        classes_re_mapping=mapping,
        masks_resolution_factor=factor,
    )
    with torch.cuda.stream(torch.cuda.Stream()):
        # Producer data were created on the default stream.
        torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
        actual = post_process_single_instance_segmentation_result_to_rle_masks_triton(
            image_bboxes=bboxes,
            image_scores=scores,
            image_masks=masks,
            image_meta=metadata,
            threshold=0.4,
            classes_re_mapping=mapping,
            masks_resolution_factor=factor,
            defer_postprocess_sync=mode.startswith("deferred"),
            max_detections=100,  # must still respect RF-DETR's query-count cap
        )
    assert actual is not None
    if mode.startswith("deferred"):
        finalize = get_deferred_postprocess_finalizer(actual)
        assert finalize is not None
        actual = finalize()
    target = resolve_mask_target_size(
        8,
        8,
        size_after_pre_processing=metadata.original_size,
        masks_resolution_factor=factor,
    )
    assert actual.image_size == actual.mask_frame_size == image_size
    assert actual.mask.image_size == image_size
    assert actual.mask_size == actual.mask.mask_size == target
    if mode not in ("empty", "deferred_empty"):
        _assert_detections_equal(actual, expected)
    else:
        assert len(actual) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton_postprocess.triton is None,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("factor", [0.0, 0.25, 0.5, 1.0])
def test_triton_native_and_large_source_grids(factor) -> None:
    from inference_models.models.common.roboflow.post_processing import (
        resolve_mask_target_size,
    )

    device = torch.device("cuda")
    bboxes, logits, _ = _single_detection_inputs(device)
    masks = torch.full((2, 192, 192), -2.0, device=device)
    masks[0, 12:185, 41:161] = 3.0
    metadata = _metadata(height=1031, width=257)
    mapping = _class_mapping(device)
    expected = _post_process_single_instance_segmentation_result_to_rle_masks(
        image_bboxes=bboxes,
        image_logits=logits.sigmoid(),
        image_masks=masks,
        image_meta=metadata,
        threshold=0.4,
        num_classes=2,
        classes_re_mapping=mapping,
        masks_resolution_factor=factor,
    )
    actual = post_process_single_instance_segmentation_result_to_rle_masks_triton(
        image_bboxes=bboxes,
        image_scores=logits.sigmoid(),
        image_masks=masks,
        image_meta=metadata,
        threshold=0.4,
        classes_re_mapping=mapping,
        masks_resolution_factor=factor,
    )
    assert actual is not None
    _assert_detections_equal(actual, expected)
    assert actual.mask_size == resolve_mask_target_size(
        192,
        192,
        size_after_pre_processing=metadata.original_size,
        masks_resolution_factor=factor,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton_postprocess.triton is None,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("factor", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("deferred", [False, True])
def test_triton_global_topk_keeps_more_than_four_classes_per_query(factor, deferred):
    from inference_models.models.base.async_handoff import (
        get_deferred_postprocess_finalizer,
    )

    device = torch.device("cuda")
    # The highest eight pairs all come from one query; an ignored class must
    # still consume a global top-k slot before remapping is applied.
    scores = torch.full((8, 10), 0.01, device=device)
    scores[0] = torch.linspace(0.99, 0.80, 10, device=device)
    bboxes = torch.tensor([[0.5, 0.5, 0.6, 0.4]], device=device).repeat(8, 1)
    masks = torch.full((8, 8, 8), -2.0, device=device)
    masks[0, 1:6, 2:7] = 3.0
    mapping = _class_mapping(device, num_classes=10)
    mapping.class_mapping[0] = -1
    meta = _metadata(height=63, width=95)
    expected = _post_process_single_instance_segmentation_result_to_rle_masks(
        image_bboxes=bboxes,
        image_logits=scores,
        image_masks=masks,
        image_meta=meta,
        threshold=0.005,
        num_classes=10,
        classes_re_mapping=mapping,
        max_detections=100,
        masks_resolution_factor=factor,
    )
    actual = post_process_single_instance_segmentation_result_to_rle_masks_triton(
        image_bboxes=bboxes,
        image_scores=scores,
        image_masks=masks,
        image_meta=meta,
        threshold=0.005,
        classes_re_mapping=mapping,
        max_detections=100,
        masks_resolution_factor=factor,
        defer_postprocess_sync=deferred,
    )
    assert actual is not None
    if deferred:
        actual = get_deferred_postprocess_finalizer(actual)()
    assert len(actual) == 7
    _assert_detections_equal(actual, expected)


@pytest.mark.parametrize("axis", ["height", "width"])
@pytest.mark.parametrize("source_size, target_size", [(1, 7), (8, 8), (8, 63)])
def test_batched_interpolation_tables_preserve_reference_weights(
    axis, source_size, target_size
):
    indices, values = _get_interpolation_weights(
        src_size=source_size,
        output_size=target_size,
        device=torch.device("cpu"),
        axis=axis,
    )
    basis_shape = (
        (source_size, 1, source_size, 1)
        if axis == "height"
        else (source_size, 1, 1, source_size)
    )
    output_shape = (target_size, 1) if axis == "height" else (1, target_size)
    resized = (
        torch.nn.functional.interpolate(
            torch.eye(source_size).reshape(basis_shape),
            size=output_shape,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        .reshape(source_size, target_size)
        .T
    )
    # Reconstruct every coefficient, including the zero-padded second tap at
    # boundaries/native grids, to catch changes to interpolation or tap order.
    reconstructed = torch.zeros_like(resized)
    reconstructed.scatter_add_(1, indices.long(), values)
    torch.testing.assert_close(reconstructed, resized, rtol=0, atol=0)
    two_taps = values[:, 1] != 0
    assert torch.all(indices[two_taps, 0] < indices[two_taps, 1])
    assert torch.all(indices[~two_taps, 1] == 0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton_postprocess.triton is None,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("factor", [0.0, 0.25, 0.5, 1.0])
def test_ignored_classes_on_inactive_query_consume_global_topk_slots(factor):
    device = torch.device("cuda")
    bboxes, _, masks = _single_detection_inputs(device)
    scores = torch.tensor([[0.01, 0.02, 0.8], [0.99, 0.9, 0.1]], device=device)
    mapping = _class_mapping(device, num_classes=3)
    mapping.class_mapping[:] = torch.tensor([-1, -1, 42], device=device)
    # Both global top-2 slots are ignored classes on query 1. Query 0's valid
    # 0.8 candidate must not be returned, although it passes the confidence cut.
    expected = _post_process_single_instance_segmentation_result_to_rle_masks(
        image_bboxes=bboxes,
        image_logits=scores,
        image_masks=masks,
        image_meta=_metadata(),
        threshold=0.4,
        num_classes=3,
        classes_re_mapping=mapping,
        masks_resolution_factor=factor,
    )
    actual = post_process_single_instance_segmentation_result_to_rle_masks_triton(
        image_bboxes=bboxes,
        image_scores=scores,
        image_masks=masks,
        image_meta=_metadata(),
        threshold=0.4,
        classes_re_mapping=mapping,
        masks_resolution_factor=factor,
    )
    assert actual is not None
    assert len(actual) == len(expected) == 0
    # The reference's empty-RLE carrier defaults to the image grid. Triton
    # explicitly retains the requested grid even when selection is empty.
    expected_side = {0.0: 8, 0.25: 22, 0.5: 36, 1.0: 64}[factor]
    assert actual.mask_size == (expected_side, expected_side)
