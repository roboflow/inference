import pytest
import torch

from inference_models.entities import ImageDimensions
from inference_models.errors import ModelRuntimeError
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.optimization.errors import RecoverableStageExecutionError
from inference_models.models.rfdetr.class_remapping import ClassesReMapping
from inference_models.models.rfdetr.common import post_process_object_detection_results
from inference_models.models.rfdetr.optimization.catalog import (
    RFDETR_POSTPROCESSOR_IMPLEMENTATIONS,
)
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_POSTPROCESSOR_TRITON_FUSED_V1,
)
from inference_models.models.rfdetr.triton_object_detection_postprocess import (
    TRITON_AVAILABLE,
    FusedObjectDetectionPostprocessor,
    _canonical_device,
    _metadata_values,
)


def _metadata(
    source_height: int = 480,
    source_width: int = 640,
    target_height: int = 512,
    target_width: int = 512,
) -> PreProcessingMetadata:
    return PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(width=source_width, height=source_height),
        size_after_pre_processing=ImageDimensions(
            width=source_width, height=source_height
        ),
        inference_size=ImageDimensions(width=target_width, height=target_height),
        scale_width=target_width / source_width,
        scale_height=target_height / source_height,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=source_width,
            crop_height=source_height,
        ),
    )


def test_fused_postprocessor_is_explicitly_selectable() -> None:
    metadata = RFDETR_POSTPROCESSOR_IMPLEMENTATIONS[
        RFDETR_POSTPROCESSOR_TRITON_FUSED_V1
    ]
    assert metadata.validation_records == ()


def test_fused_postprocessor_requires_cuda() -> None:
    with pytest.raises(ModelRuntimeError, match="requires a CUDA target"):
        FusedObjectDetectionPostprocessor(torch.device("cpu"))


def test_canonical_device_resolves_bare_cuda_to_current_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    assert _canonical_device(torch.device("cuda")) == torch.device("cuda:1")
    assert _canonical_device(torch.device("cuda:0")) == torch.device("cuda:0")
    assert _canonical_device(torch.device("cpu")) == torch.device("cpu")


def test_fused_postprocessor_canonicalizes_target_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    postprocessor = FusedObjectDetectionPostprocessor(torch.device("cuda"))

    assert postprocessor._device == torch.device("cuda:0")


def test_fused_postprocessor_reports_request_incompatibility_without_cuda() -> None:
    postprocessor = FusedObjectDetectionPostprocessor.__new__(
        FusedObjectDetectionPostprocessor
    )
    postprocessor._device = torch.device("cuda:0")

    compatibility = postprocessor.check_request_compatibility(
        bboxes=torch.zeros((1, 2, 4)),
        logits=torch.zeros((1, 2, 3)),
        pre_processing_meta=[_metadata()],
        threshold=0.5,
        num_classes=3,
        classes_re_mapping=None,
    )

    assert not compatibility.supported
    assert "boxes and logits must be CUDA tensors" in compatibility.reasons


def test_fused_postprocessor_reports_missing_triton(monkeypatch) -> None:
    monkeypatch.setattr(
        "inference_models.models.rfdetr.triton_object_detection_postprocess."
        "TRITON_AVAILABLE",
        False,
    )
    postprocessor = FusedObjectDetectionPostprocessor.__new__(
        FusedObjectDetectionPostprocessor
    )
    postprocessor._device = torch.device("cuda:0")

    compatibility = postprocessor.check_request_compatibility(
        bboxes=torch.zeros((1, 2, 4)),
        logits=torch.zeros((1, 2, 3)),
        pre_processing_meta=[_metadata()],
        threshold=0.5,
        num_classes=3,
        classes_re_mapping=None,
    )

    assert not compatibility.supported
    assert "Triton is not installed" in compatibility.reasons


def test_fused_postprocessor_direct_call_remains_strict_for_incompatibility() -> None:
    postprocessor = FusedObjectDetectionPostprocessor.__new__(
        FusedObjectDetectionPostprocessor
    )
    postprocessor._device = torch.device("cuda:0")

    with pytest.raises(ModelRuntimeError, match="cannot preserve this.*contract"):
        postprocessor._validate_inputs(
            bboxes=torch.zeros((1, 2, 4)),
            logits=torch.zeros((1, 2, 3)),
            pre_processing_meta=[_metadata()],
            threshold=0.5,
            num_classes=3,
            classes_re_mapping=None,
        )


@pytest.mark.gpu_only
@pytest.mark.trt_extras
@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)
def test_fused_postprocessor_records_recognized_runtime_failure(monkeypatch) -> None:
    class FailingKernel:
        def __getitem__(self, grid):
            del grid

            def launch(*args, **kwargs):
                del args, kwargs
                raise RuntimeError(
                    "Failed to find C compiler. "
                    "Please specify via CC environment variable."
                )

            return launch

    monkeypatch.setattr(
        "inference_models.models.rfdetr.triton_object_detection_postprocess."
        "_compact_transform_detections_kernel",
        FailingKernel(),
    )
    device = torch.device("cuda:0")
    postprocessor = FusedObjectDetectionPostprocessor(device)

    with pytest.raises(
        RecoverableStageExecutionError,
        match="Failed to find C compiler",
    ):
        postprocessor.postprocess(
            bboxes=torch.zeros((1, 2, 4), device=device),
            logits=torch.zeros((1, 2, 3), device=device),
            pre_processing_meta=[_metadata()],
            threshold=0.5,
            num_classes=3,
            classes_re_mapping=None,
            stream=torch.cuda.Stream(device=device),
        )

    compatibility = postprocessor.check_runtime_compatibility()

    assert not compatibility.supported
    assert "Failed to find C compiler" in compatibility.reason
    assert "Category: missing_compiler" in compatibility.reason
    assert "Suggested action:" in compatibility.reason


def test_metadata_values_match_reference_transform_parameters() -> None:
    values = _metadata_values(_metadata())

    assert values == (
        512.0,
        512.0,
        0.0,
        0.0,
        0.8,
        512 / 480,
        0.0,
        0.0,
        640.0,
        480.0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("per_class_threshold", [False, True])
@pytest.mark.parametrize("with_class_remapping", [False, True])
def test_fused_postprocessor_matches_reference(
    per_class_threshold: bool,
    with_class_remapping: bool,
) -> None:
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(123)
    batch_size, num_queries, logits_classes = 4, 17, 6
    bboxes = torch.rand(
        (batch_size, num_queries, 4), device=device, generator=generator
    )
    logits = torch.randn(
        (batch_size, num_queries, logits_classes),
        device=device,
        generator=generator,
    )
    metadata = [_metadata() for _ in range(batch_size)]

    if with_class_remapping:
        class_mapping = torch.tensor([0, -1, 1, 2, 3, -1], device=device)
        remapping = ClassesReMapping(
            remaining_class_ids=torch.tensor([0, 2, 3, 4], device=device),
            class_mapping=class_mapping,
        )
        num_classes = 4
    else:
        remapping = None
        num_classes = logits_classes - 1

    threshold = torch.linspace(0.1, 0.5, num_classes) if per_class_threshold else 0.25
    expected = post_process_object_detection_results(
        bboxes=bboxes,
        logits=logits,
        pre_processing_meta=metadata,
        threshold=threshold,
        num_classes=num_classes,
        classes_re_mapping=remapping,
        device=device,
    )
    stream = torch.cuda.Stream(device=device)
    actual = FusedObjectDetectionPostprocessor(device).postprocess(
        bboxes=bboxes,
        logits=logits,
        pre_processing_meta=metadata,
        threshold=threshold,
        num_classes=num_classes,
        classes_re_mapping=remapping,
        stream=stream,
    )
    stream.synchronize()

    for actual_image, expected_image in zip(actual, expected):
        assert torch.all(
            actual_image.confidence[:-1] >= actual_image.confidence[1:]
        ), f"Confidence is not decreasing: {actual_image.confidence.tolist()}"
        torch.testing.assert_close(actual_image.xyxy, expected_image.xyxy)
        torch.testing.assert_close(actual_image.class_id, expected_image.class_id)
        torch.testing.assert_close(
            actual_image.confidence,
            expected_image.confidence,
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("extent", [0.0, 0.125, 0.317, 0.7])
@pytest.mark.parametrize("with_padding_and_crop", [False, True])
def test_fused_postprocessor_preserves_reference_half_pixel_rounding(
    extent: float,
    with_padding_and_crop: bool,
) -> None:
    device = torch.device("cuda:0")
    num_queries = 300
    half_pixels = torch.arange(num_queries, dtype=torch.float32) + 100.5
    bboxes = torch.full((1, num_queries, 4), extent)
    bboxes[0, :, 0] = half_pixels / 1920
    bboxes[0, :, 1] = half_pixels / 1080
    bboxes = bboxes.to(device=device)
    logits = torch.full((1, num_queries, 91), -20.0)
    logits[0, :, 1] = torch.linspace(-0.2, 3.0, num_queries)
    logits = logits.to(device=device)
    image_meta = _metadata(source_width=1920, source_height=1080)
    if with_padding_and_crop:
        image_meta = PreProcessingMetadata(
            pad_left=13,
            pad_top=11,
            pad_right=13,
            pad_bottom=11,
            original_size=ImageDimensions(width=1984, height=1144),
            size_after_pre_processing=ImageDimensions(width=1920, height=1080),
            inference_size=ImageDimensions(width=512, height=512),
            scale_width=486 / 1920,
            scale_height=490 / 1080,
            static_crop_offset=StaticCropOffset(
                offset_x=32,
                offset_y=32,
                crop_width=1920,
                crop_height=1080,
            ),
        )
    arguments = {
        "bboxes": bboxes,
        "logits": logits,
        "pre_processing_meta": [image_meta],
        "threshold": 0.4,
        "num_classes": 91,
        "classes_re_mapping": None,
    }
    stream = torch.cuda.current_stream(device=device)

    expected = post_process_object_detection_results(**arguments, device=device)[0]
    actual = FusedObjectDetectionPostprocessor(device=device).postprocess(
        **arguments, stream=stream
    )[0]
    stream.synchronize()

    torch.testing.assert_close(actual.xyxy, expected.xyxy, rtol=0, atol=0)
    torch.testing.assert_close(actual.class_id, expected.class_id, rtol=0, atol=0)
    torch.testing.assert_close(actual.confidence, expected.confidence, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("survivor_count", [17, 33, 128])
def test_fused_postprocessor_preserves_reference_filtered_tie_order(
    survivor_count: int,
) -> None:
    device = torch.device("cuda:0")
    generator = torch.Generator().manual_seed(12345)
    bboxes = torch.rand((1, 300, 4), generator=generator).to(device=device)
    logits = torch.full((1, 300, 91), -20.0, device=device)
    logits[0, :survivor_count, 1] = 2.0
    arguments = {
        "bboxes": bboxes,
        "logits": logits,
        "pre_processing_meta": [_metadata(source_width=1920, source_height=1080)],
        "threshold": 0.4,
        "num_classes": 91,
        "classes_re_mapping": None,
    }
    stream = torch.cuda.current_stream(device=device)

    expected = post_process_object_detection_results(**arguments, device=device)[0]
    actual = FusedObjectDetectionPostprocessor(device=device).postprocess(
        **arguments, stream=stream
    )[0]
    stream.synchronize()

    assert len(actual.confidence) == survivor_count
    torch.testing.assert_close(actual.xyxy, expected.xyxy, rtol=0, atol=0)
    torch.testing.assert_close(actual.class_id, expected.class_id, rtol=0, atol=0)
    torch.testing.assert_close(actual.confidence, expected.confidence, rtol=0, atol=0)
