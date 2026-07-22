import pytest
import torch

from inference_models.entities import ImageDimensions
from inference_models.errors import ModelRuntimeError
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
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
    assert metadata.validated_environments == ()


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
        torch.testing.assert_close(actual_image.xyxy, expected_image.xyxy)
        torch.testing.assert_close(actual_image.class_id, expected_image.class_id)
        torch.testing.assert_close(
            actual_image.confidence,
            expected_image.confidence,
            rtol=0,
            atol=0,
        )
