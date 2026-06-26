import importlib
import logging
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    ImagePreProcessing,
    NetworkInputDefinition,
    PreProcessingMetadata,
    ResizeMode,
    StaticCropOffset,
    TrainingInputSize,
)
from inference_models.models.rfdetr import common as rfdetr_common
from inference_models.models.rfdetr import triton_preprocess_runtime
from inference_models.models.rfdetr.class_remapping import ClassesReMapping
from inference_models.models.rfdetr.triton_jit_fallback import (
    is_triton_jit_failure,
    warn_triton_jit_fallback,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _network_input(target_h: int = 64, target_w: int = 64) -> NetworkInputDefinition:
    return NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=target_h, width=target_w),
        dataset_version_resize_dimensions=None,
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=255,
        normalization=[list(_IMAGENET_MEAN), list(_IMAGENET_STD)],
    )


def _metadata() -> PreProcessingMetadata:
    size = ImageDimensions(height=64, width=64)
    return PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=size,
        size_after_pre_processing=size,
        inference_size=size,
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=64,
            crop_height=64,
        ),
    )


def _class_mapping(device: torch.device) -> ClassesReMapping:
    return ClassesReMapping(
        remaining_class_ids=torch.arange(2, dtype=torch.int64, device=device),
        class_mapping=torch.arange(2, dtype=torch.int64, device=device),
    )


def _single_detection_inputs(device: torch.device):
    bboxes = torch.tensor(
        [[0.50, 0.50, 0.50, 0.50], [0.25, 0.25, 0.20, 0.20]],
        dtype=torch.float32,
        device=device,
    )
    logits = torch.tensor(
        [[4.0, -4.0], [-4.0, -4.0]],
        dtype=torch.float32,
        device=device,
    )
    masks = torch.ones((2, 8, 8), dtype=torch.float32, device=device)
    return bboxes, logits, masks


def test_is_triton_jit_failure_detects_missing_c_compiler() -> None:
    exc = RuntimeError(
        "Failed to find C compiler. Please specify via CC environment variable."
    )

    assert is_triton_jit_failure(exc)


def test_is_triton_jit_failure_detects_ptxas_message() -> None:
    exc = RuntimeError(
        "PTXAS error: Internal Triton PTX codegen error\n"
        "ptxas-blackwell fatal: Value 'sm_110a' is not defined"
    )

    assert is_triton_jit_failure(exc)


def test_is_triton_jit_failure_rejects_unrelated_runtime_error() -> None:
    assert not is_triton_jit_failure(RuntimeError("CUDA out of memory"))


def test_is_triton_jit_failure_detects_out_of_resources_message() -> None:
    exc = RuntimeError(
        "out of resource: shared memory, Required: 131072, Hardware limit: 101376. "
        "Reducing block sizes or `num_stages` may help."
    )

    assert is_triton_jit_failure(exc)


def test_is_triton_jit_failure_detects_out_of_resources_type() -> None:
    pytest.importorskip("triton")
    from triton.runtime.errors import OutOfResources

    exc = OutOfResources(required=131072, limit=101376, name="shared memory")

    assert is_triton_jit_failure(exc)


def _reload_triton_jit_fallback_with_fake_errors(
    *,
    monkeypatch: pytest.MonkeyPatch,
    error_classes: dict[str, type[BaseException]],
):
    fake_errors = ModuleType("triton.runtime.errors")
    for name, cls in error_classes.items():
        setattr(fake_errors, name, cls)

    fake_runtime = ModuleType("triton.runtime")
    fake_runtime.errors = fake_errors
    fake_triton = ModuleType("triton")
    fake_triton.runtime = fake_runtime

    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "triton.runtime.errors", fake_errors)

    import inference_models.models.rfdetr.triton_jit_fallback as fallback_mod

    reloaded_fallback_mod = importlib.reload(fallback_mod)

    return reloaded_fallback_mod


@pytest.fixture
def triton_jit_fallback_module():
    import inference_models.models.rfdetr.triton_jit_fallback as fallback_mod

    yield fallback_mod

    importlib.reload(fallback_mod)


def test_triton_jit_exception_types_import_independently(
    monkeypatch: pytest.MonkeyPatch,
    triton_jit_fallback_module,
) -> None:
    class FakeOutOfResources(Exception):
        pass

    class FakePTXASError(Exception):
        pass

    fallback_mod = _reload_triton_jit_fallback_with_fake_errors(
        monkeypatch=monkeypatch,
        error_classes={"OutOfResources": FakeOutOfResources},
    )

    assert fallback_mod._TRITON_JIT_EXCEPTION_TYPES == (FakeOutOfResources,)
    assert fallback_mod.is_triton_jit_failure(FakeOutOfResources())

    fallback_mod = _reload_triton_jit_fallback_with_fake_errors(
        monkeypatch=monkeypatch,
        error_classes={"PTXASError": FakePTXASError},
    )

    assert fallback_mod._TRITON_JIT_EXCEPTION_TYPES == (FakePTXASError,)
    assert fallback_mod.is_triton_jit_failure(FakePTXASError())


def test_warn_triton_jit_fallback_logs_once(caplog: pytest.LogCaptureFixture) -> None:
    warned_reasons: set[str] = set()
    exc = RuntimeError("Failed to find C compiler")

    with caplog.at_level(logging.ERROR):
        warn_triton_jit_fallback(
            path="preprocess",
            exc=exc,
            warned_reasons=warned_reasons,
        )
        warn_triton_jit_fallback(
            path="preprocess",
            exc=exc,
            warned_reasons=warned_reasons,
        )

    matching_records = [
        record
        for record in caplog.records
        if "RF-DETR Triton preprocess JIT compilation failed" in record.message
    ]
    assert len(matching_records) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_try_preprocess_falls_back_on_triton_jit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(triton_preprocess_runtime, "_FAST_PATH_ENABLED", True)
    monkeypatch.setattr(triton_preprocess_runtime, "_TRITON_AVAILABLE", True)

    runtime = triton_preprocess_runtime.FastPreprocessRuntime(
        device=torch.device("cuda")
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    stream = torch.cuda.Stream(device=torch.device("cuda"))

    calls = {"count": 0}

    def failing_kernel(*args, **kwargs):
        calls["count"] += 1
        raise RuntimeError(
            "Failed to find C compiler. Please specify via CC environment variable."
        )

    monkeypatch.setattr(
        triton_preprocess_runtime,
        "triton_preprocess_rfdetr_stretch_two_pass_preallocated",
        failing_kernel,
    )

    result = runtime.try_preprocess(
        images=image,
        input_color_format="bgr",
        image_size=None,
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(target_h=64, target_w=64),
        stream=stream,
    )

    assert result is None
    assert calls["count"] == 1
    assert runtime._jit_disabled is True

    second_result = runtime.try_preprocess(
        images=image,
        input_color_format="bgr",
        image_size=None,
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(target_h=64, target_w=64),
        stream=stream,
    )

    assert second_result is None
    assert calls["count"] == 1


def test_postproc_falls_back_on_triton_jit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rfdetr_common, "_TRITON_POSTPROC_ENABLED", True)
    monkeypatch.setattr(rfdetr_common, "_TRITON_POSTPROC_JIT_DISABLED", False)
    rfdetr_common._TRITON_POSTPROC_JIT_WARNED_REASONS.clear()

    calls = {"count": 0}

    def failing_triton(**kwargs):
        calls["count"] += 1
        raise RuntimeError(
            "Failed to find C compiler. Please specify via CC environment variable."
        )

    monkeypatch.setattr(
        rfdetr_common,
        "post_process_single_instance_segmentation_result_to_rle_masks_triton",
        failing_triton,
    )

    device = torch.device("cpu")
    bboxes, logits, masks = _single_detection_inputs(device)
    results = rfdetr_common.post_process_instance_segmentation_results_to_rle_masks(
        bboxes=bboxes.unsqueeze(0),
        logits=logits.unsqueeze(0),
        masks=masks.unsqueeze(0),
        pre_processing_meta=[_metadata()],
        threshold=0.4,
        num_classes=2,
        classes_re_mapping=_class_mapping(device),
    )

    assert len(results) == 1
    assert results[0].confidence.shape == (1,)
    assert calls["count"] == 1
    assert rfdetr_common._TRITON_POSTPROC_JIT_DISABLED is True

    second_results = rfdetr_common.post_process_instance_segmentation_results_to_rle_masks(
        bboxes=bboxes.unsqueeze(0),
        logits=logits.unsqueeze(0),
        masks=masks.unsqueeze(0),
        pre_processing_meta=[_metadata()],
        threshold=0.4,
        num_classes=2,
        classes_re_mapping=_class_mapping(device),
    )

    assert len(second_results) == 1
    assert calls["count"] == 1