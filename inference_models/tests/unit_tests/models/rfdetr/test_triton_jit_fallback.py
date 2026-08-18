import importlib
import logging
import subprocess
import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import patch

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
from inference_models.models.optimization.triton_jit import (
    classify_triton_jit_failure,
    is_triton_jit_failure,
    warn_triton_jit_fallback,
)
from inference_models.models.rfdetr import common as rfdetr_common
from inference_models.models.rfdetr import triton_preprocess_runtime
from inference_models.models.rfdetr.class_remapping import ClassesReMapping

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


def test_is_triton_jit_failure_detects_missing_shared_library() -> None:
    exc = RuntimeError(
        "Triton JIT failed because libcuda.so cannot open shared object file"
    )

    assert is_triton_jit_failure(exc)


@pytest.mark.parametrize(
    ("message", "expected_category"),
    [
        ("Failed to find C compiler", "missing_compiler"),
        (
            "Triton JIT failed because libcuda.so cannot open shared object file",
            "missing_driver_library",
        ),
        (
            "Triton JIT failed because libnvrtc.so cannot open shared object file",
            "missing_runtime_library",
        ),
        (
            "Triton JIT compiler failed with undefined reference to cuModuleLoadData",
            "linker_failure",
        ),
        ("PTXAS fatal: target architecture is unsupported", "ptx_toolchain_mismatch"),
        (
            "out of resource: shared memory exceeds the hardware limit",
            "kernel_resource_limit",
        ),
    ],
)
def test_classify_triton_jit_failure_provides_conservative_guidance(
    message: str,
    expected_category: str,
) -> None:
    diagnostic = classify_triton_jit_failure(RuntimeError(message))

    assert diagnostic is not None
    assert diagnostic.category == expected_category
    assert diagnostic.guidance
    assert "apt " not in diagnostic.guidance


@pytest.mark.parametrize(
    "exc",
    [
        OSError(
            "libnvrtc.so.12: cannot open shared object file: "
            "No such file or directory"
        ),
        RuntimeError(
            "libcuda.so: cannot open shared object file: No such file or directory"
        ),
    ],
)
def test_is_triton_jit_failure_rejects_unscoped_loader_error(
    exc: BaseException,
) -> None:
    assert not is_triton_jit_failure(exc)


def test_is_triton_jit_failure_ignores_implicit_exception_context() -> None:
    try:
        try:
            raise OSError(
                "libcuda.so: cannot open shared object file: No such file or directory"
            )
        except OSError:
            raise ValueError("invalid kernel configuration")
    except ValueError as exc:
        assert not is_triton_jit_failure(exc)


def test_is_triton_jit_failure_follows_explicit_exception_cause() -> None:
    try:
        try:
            raise RuntimeError(
                "Failed to find C compiler. Please specify via CC environment variable."
            )
        except RuntimeError as cause:
            raise RuntimeError("Triton launch wrapper failed") from cause
    except RuntimeError as exc:
        assert is_triton_jit_failure(exc)


def test_is_triton_jit_failure_detects_failed_compiler_subprocess() -> None:
    exc = subprocess.CalledProcessError(returncode=1, cmd=["cc", "launcher.c"])

    assert is_triton_jit_failure(exc)


def test_is_triton_jit_failure_detects_failed_shell_compiler_subprocess() -> None:
    exc = subprocess.CalledProcessError(
        returncode=1,
        cmd=["/bin/sh", "-c", "/usr/bin/gcc-12 launcher.c"],
    )

    assert is_triton_jit_failure(exc)


def test_is_triton_jit_failure_detects_compiler_marker_in_subprocess_stderr() -> None:
    exc = subprocess.CalledProcessError(
        returncode=1,
        cmd=["python", "build_helper.py"],
        stderr=b"Triton JIT linker failed: ld cannot find -lcuda",
    )

    assert is_triton_jit_failure(exc)


def test_is_triton_jit_failure_rejects_unscoped_loader_subprocess_error() -> None:
    exc = subprocess.CalledProcessError(
        returncode=1,
        cmd=["python", "load_plugin.py"],
        stderr=b"libcuda.so: cannot open shared object file",
    )

    assert not is_triton_jit_failure(exc)


def test_is_triton_jit_failure_rejects_unrelated_subprocess_failure() -> None:
    exc = subprocess.CalledProcessError(
        returncode=22,
        cmd=["curl", "--fail", "https://example.com"],
        stderr=b"HTTP 404",
    )

    assert not is_triton_jit_failure(exc)


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


@contextmanager
def _triton_jit_fallback_with_fake_errors(
    error_classes: dict[str, type[BaseException]],
):
    fake_errors = ModuleType("triton.runtime.errors")
    for name, cls in error_classes.items():
        setattr(fake_errors, name, cls)

    fake_runtime = ModuleType("triton.runtime")
    fake_runtime.errors = fake_errors
    fake_compiler_errors = ModuleType("triton.compiler.errors")
    fake_compiler = ModuleType("triton.compiler")
    fake_compiler.errors = fake_compiler_errors
    fake_triton = ModuleType("triton")
    fake_triton.runtime = fake_runtime
    fake_triton.compiler = fake_compiler

    import inference_models.models.optimization.triton_jit as fallback_mod

    fake_modules = {
        "triton": fake_triton,
        "triton.runtime": fake_runtime,
        "triton.runtime.errors": fake_errors,
        "triton.compiler": fake_compiler,
        "triton.compiler.errors": fake_compiler_errors,
    }
    try:
        with patch.dict(sys.modules, fake_modules):
            yield importlib.reload(fallback_mod)
    finally:
        # Reload only after patch.dict restores the real import environment.
        importlib.reload(fallback_mod)


def test_triton_jit_exception_types_import_independently() -> None:
    import inference_models.models.optimization.triton_jit as original_fallback_mod

    original_exception_types = original_fallback_mod._TRITON_JIT_EXCEPTION_TYPES

    class FakeOutOfResources(Exception):
        pass

    class FakePTXASError(Exception):
        pass

    with _triton_jit_fallback_with_fake_errors(
        {"OutOfResources": FakeOutOfResources}
    ) as fallback_mod:
        assert fallback_mod._TRITON_JIT_EXCEPTION_TYPES == (FakeOutOfResources,)
        assert fallback_mod.is_triton_jit_failure(FakeOutOfResources())

    with _triton_jit_fallback_with_fake_errors(
        {"PTXASError": FakePTXASError}
    ) as fallback_mod:
        assert fallback_mod._TRITON_JIT_EXCEPTION_TYPES == (FakePTXASError,)
        assert fallback_mod.is_triton_jit_failure(FakePTXASError())

    assert FakeOutOfResources not in fallback_mod._TRITON_JIT_EXCEPTION_TYPES
    assert FakePTXASError not in fallback_mod._TRITON_JIT_EXCEPTION_TYPES
    assert fallback_mod._TRITON_JIT_EXCEPTION_TYPES == original_exception_types


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
    assert "Category: missing_compiler" in matching_records[0].message
    assert "Suggested action:" in matching_records[0].message


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


def test_postproc_falls_back_on_triton_jit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    second_results = (
        rfdetr_common.post_process_instance_segmentation_results_to_rle_masks(
            bboxes=bboxes.unsqueeze(0),
            logits=logits.unsqueeze(0),
            masks=masks.unsqueeze(0),
            pre_processing_meta=[_metadata()],
            threshold=0.4,
            num_classes=2,
            classes_re_mapping=_class_mapping(device),
        )
    )

    assert len(second_results) == 1
    assert calls["count"] == 1


def test_legacy_postproc_propagates_unscoped_loader_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rfdetr_common, "_TRITON_POSTPROC_ENABLED", True)
    monkeypatch.setattr(rfdetr_common, "_TRITON_POSTPROC_JIT_DISABLED", False)

    def failing_triton(**kwargs):
        del kwargs
        raise OSError(
            "libnvrtc.so.12: cannot open shared object file: "
            "No such file or directory"
        )

    monkeypatch.setattr(
        rfdetr_common,
        "post_process_single_instance_segmentation_result_to_rle_masks_triton",
        failing_triton,
    )
    device = torch.device("cpu")
    bboxes, logits, masks = _single_detection_inputs(device)

    with pytest.raises(OSError, match="libnvrtc"):
        rfdetr_common.post_process_instance_segmentation_results_to_rle_masks(
            bboxes=bboxes.unsqueeze(0),
            logits=logits.unsqueeze(0),
            masks=masks.unsqueeze(0),
            pre_processing_meta=[_metadata()],
            threshold=0.4,
            num_classes=2,
            classes_re_mapping=_class_mapping(device),
        )

    assert rfdetr_common._TRITON_POSTPROC_JIT_DISABLED is False
