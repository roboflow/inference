from unittest.mock import MagicMock

import pytest
import torch
from torchvision.transforms import functional


def _meta(
    inference_h,
    inference_w,
    pad_top=0,
    pad_bottom=0,
    pad_left=0,
    pad_right=0,
    after_h=None,
    after_w=None,
    original_h=None,
    original_w=None,
    crop_offset_x=0,
    crop_offset_y=0,
):
    after_h = after_h if after_h is not None else inference_h
    after_w = after_w if after_w is not None else inference_w
    return [
        MagicMock(
            inference_size=MagicMock(height=inference_h, width=inference_w),
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
            size_after_pre_processing=MagicMock(height=after_h, width=after_w),
            original_size=MagicMock(
                height=original_h if original_h is not None else after_h,
                width=original_w if original_w is not None else after_w,
            ),
            static_crop_offset=MagicMock(
                offset_x=crop_offset_x, offset_y=crop_offset_y
            ),
        )
    ]


def test_post_process_depth_estimation_map_preserves_values_without_padding():
    """A (B, 1, H, W) depth map with no letterbox padding comes back as a (H, W)
    map with the raw values (e.g. metric meters) untouched."""
    from inference_models.models.common.roboflow.post_processing import (
        post_process_depth_estimation_map,
    )

    h, w = 8, 8
    depth = torch.full((1, 1, h, w), 3.5)

    results = post_process_depth_estimation_map(
        model_results=depth,
        pre_processing_meta=_meta(h, w),
        device=torch.device("cpu"),
    )

    assert len(results) == 1
    assert results[0].shape == (h, w)
    assert results[0].dtype == torch.float32
    assert torch.allclose(results[0], torch.full((h, w), 3.5))


def test_post_process_depth_estimation_map_accepts_3d_input():
    """(B, H, W) outputs (no channel dim) are handled identically."""
    from inference_models.models.common.roboflow.post_processing import (
        post_process_depth_estimation_map,
    )

    h, w = 6, 6
    depth = torch.full((1, h, w), 2.0)

    results = post_process_depth_estimation_map(
        model_results=depth,
        pre_processing_meta=_meta(h, w),
        device=torch.device("cpu"),
    )

    assert results[0].shape == (h, w)
    assert torch.allclose(results[0], torch.full((h, w), 2.0))


def test_post_process_depth_estimation_map_crops_letterbox_padding():
    """Rows introduced by letterbox padding are cropped out before returning the
    map at the pre-letterbox image size."""
    from inference_models.models.common.roboflow.post_processing import (
        post_process_depth_estimation_map,
    )

    h, w = 8, 8
    depth = torch.zeros(1, 1, h, w)
    depth[0, 0, 2:6, :] = 7.0  # content region; rows 0-1 and 6-7 are padding

    results = post_process_depth_estimation_map(
        model_results=depth,
        pre_processing_meta=_meta(h, w, pad_top=2, pad_bottom=2, after_h=4, after_w=8),
        device=torch.device("cpu"),
    )

    assert results[0].shape == (4, 8)
    assert torch.allclose(results[0], torch.full((4, 8), 7.0))


def test_post_process_depth_estimation_map_resizes_to_pre_processing_size():
    """Depth maps are bilinearly resized back to the pre-letterbox image size."""
    from inference_models.models.common.roboflow.post_processing import (
        post_process_depth_estimation_map,
    )

    h, w = 8, 8
    depth = torch.full((1, 1, h, w), 4.25)

    results = post_process_depth_estimation_map(
        model_results=depth,
        pre_processing_meta=_meta(h, w, after_h=16, after_w=16),
        device=torch.device("cpu"),
    )

    assert results[0].shape == (16, 16)
    assert torch.allclose(results[0], torch.full((16, 16), 4.25))


def test_post_process_depth_estimation_map_accepts_selectable_resize():
    from inference_models.models.common.roboflow.post_processing import (
        post_process_depth_estimation_map,
    )

    resize_calls = []

    def resize(image, size):
        resize_calls.append((image.shape, size))

        return torch.full((1, *size), 6.5)

    results = post_process_depth_estimation_map(
        model_results=torch.ones((1, 1, 8, 8)),
        pre_processing_meta=_meta(8, 8, after_h=16, after_w=16),
        device=torch.device("cpu"),
        resize_function=resize,
    )

    assert resize_calls == [(torch.Size([1, 8, 8]), (16, 16))]
    assert torch.equal(results[0], torch.full((16, 16), 6.5))


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    [(6, 30), (6, 5)],
)
def test_triton_depth_resize_tables_match_torchvision_exactly(
    input_size,
    output_size,
):
    from inference_models.models.yolo26.triton_depth_postprocess import (
        _build_axis_table,
    )

    starts, sizes, weights, _ = _build_axis_table(
        input_size=input_size,
        output_size=output_size,
    )
    basis = torch.eye(input_size, dtype=torch.float32).reshape(
        input_size,
        1,
        input_size,
    )
    expected = functional.resize(
        basis,
        [1, output_size],
        interpolation=functional.InterpolationMode.BILINEAR,
    )[:, 0]
    actual = torch.zeros_like(expected)
    for output_index in range(output_size):
        start = starts[output_index]
        size = sizes[output_index]
        actual[
            start : start + size,
            output_index,
        ] = torch.from_numpy(weights[output_index, :size])

    assert torch.equal(actual, expected)


def test_yolo26_depth_execution_plan_uses_safe_stage_defaults(monkeypatch):
    from inference_models.models.optimization.ids import (
        AUTO_IMPLEMENTATION_ID,
        BASE_IMPLEMENTATION_ID,
    )
    from inference_models.models.yolo26.optimization.execution_plan import (
        YOLO26DepthExecutionPlan,
    )
    from inference_models.models.yolo26.optimization.ids import (
        YOLO26_DEPTH_POSTPROCESSOR_ENV_NAME,
        YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_V1,
        YOLO26_DEPTH_PREPROCESSOR_ENV_NAME,
        YOLO26_DEPTH_PREPROCESSOR_TRITON_CV2_RESIZE_FUSED_CONVERT_V1,
        YOLO26_DEPTH_SCHEDULER_CUDA_EVENT_HANDOFF_V1,
        YOLO26_DEPTH_SCHEDULER_ENV_NAME,
    )

    monkeypatch.delenv(YOLO26_DEPTH_PREPROCESSOR_ENV_NAME, raising=False)
    monkeypatch.delenv(YOLO26_DEPTH_SCHEDULER_ENV_NAME, raising=False)
    monkeypatch.delenv(YOLO26_DEPTH_POSTPROCESSOR_ENV_NAME, raising=False)
    default_plan = YOLO26DepthExecutionPlan.resolve()
    assert default_plan.preprocessor_id == BASE_IMPLEMENTATION_ID
    assert default_plan.scheduler_id == BASE_IMPLEMENTATION_ID
    assert default_plan.postprocessor_id == AUTO_IMPLEMENTATION_ID

    monkeypatch.setenv(
        YOLO26_DEPTH_PREPROCESSOR_ENV_NAME,
        YOLO26_DEPTH_PREPROCESSOR_TRITON_CV2_RESIZE_FUSED_CONVERT_V1,
    )
    monkeypatch.setenv(
        YOLO26_DEPTH_SCHEDULER_ENV_NAME,
        YOLO26_DEPTH_SCHEDULER_CUDA_EVENT_HANDOFF_V1,
    )
    monkeypatch.setenv(
        YOLO26_DEPTH_POSTPROCESSOR_ENV_NAME,
        YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_V1,
    )
    plan = YOLO26DepthExecutionPlan.resolve(
        allow_compatibility_fallback=False,
    )

    assert (
        plan.preprocessor_id
        == YOLO26_DEPTH_PREPROCESSOR_TRITON_CV2_RESIZE_FUSED_CONVERT_V1
    )
    assert plan.scheduler_id == YOLO26_DEPTH_SCHEDULER_CUDA_EVENT_HANDOFF_V1
    assert plan.postprocessor_id == YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_V1
    assert not plan.allow_compatibility_fallback


def test_yolo26_depth_explicit_triton_selection_never_silently_falls_back():
    from inference_models.errors import ModelRuntimeError
    from inference_models.models.optimization.contracts import (
        ExecutionContext,
        OptimizationStage,
    )
    from inference_models.models.yolo26.optimization.ids import (
        YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_V1,
    )
    from inference_models.models.yolo26.optimization.postprocessors import (
        build_yolo26_depth_implementation_registry,
    )

    registry = build_yolo26_depth_implementation_registry(
        device=torch.device("cuda:0"),
    )
    context = ExecutionContext(
        device_kind="gpu",
        device="cuda:0",
        compute_capability=(8, 7),
        runtime_components={"torch": True, "torchvision": True, "triton": False},
    )

    with pytest.raises(ModelRuntimeError, match="unavailable runtime components"):
        registry.resolve_selection(
            stage=OptimizationStage.POSTPROCESS,
            requested_id=YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_V1,
            context=context,
            allow_fallback=False,
        )


def test_yolo26_depth_explicit_triton_preprocessor_rejects_missing_runtime():
    from inference_models.errors import ModelRuntimeError
    from inference_models.models.optimization.contracts import (
        ExecutionContext,
        OptimizationStage,
    )
    from inference_models.models.yolo26.optimization.ids import (
        YOLO26_DEPTH_PREPROCESSOR_TRITON_CV2_RESIZE_FUSED_CONVERT_V1,
    )
    from inference_models.models.yolo26.optimization.postprocessors import (
        build_yolo26_depth_implementation_registry,
    )

    registry = build_yolo26_depth_implementation_registry(
        device=torch.device("cuda:0"),
    )
    context = ExecutionContext(
        device_kind="gpu",
        device="cuda:0",
        compute_capability=(8, 7),
        runtime_components={
            "opencv-python": True,
            "torch": True,
            "triton": False,
        },
    )

    with pytest.raises(ModelRuntimeError, match="unavailable runtime components"):
        registry.resolve_selection(
            stage=OptimizationStage.PREPROCESS,
            requested_id=(YOLO26_DEPTH_PREPROCESSOR_TRITON_CV2_RESIZE_FUSED_CONVERT_V1),
            context=context,
            allow_fallback=False,
        )


def test_yolo26_depth_auto_uses_base_and_retains_explicit_candidates(monkeypatch):
    from inference_models.models.optimization.contracts import (
        ExecutionContext,
        OptimizationStage,
    )
    from inference_models.models.optimization.ids import AUTO_IMPLEMENTATION_ID
    from inference_models.models.yolo26.optimization.postprocessors import (
        BaseYOLO26DepthPostprocessor,
        ExactFusedTritonAAYOLO26DepthPostprocessor,
        ExactTritonAAYOLO26DepthPostprocessor,
        TritonAAYOLO26DepthPostprocessor,
        build_yolo26_depth_implementation_registry,
    )
    from inference_models.models.yolo26.optimization.preprocessors import (
        BaseYOLO26DepthPreprocessor,
        TritonCV2ResizeFusedConvertYOLO26DepthPreprocessor,
        TritonCV2ResizePinnedFusedConvertYOLO26DepthPreprocessor,
    )
    from inference_models.models.yolo26.optimization.schedulers import (
        BaseYOLO26DepthExecutionScheduler,
        CUDAEventHandoffYOLO26DepthExecutionScheduler,
    )

    registry = build_yolo26_depth_implementation_registry(
        device=torch.device("cuda:0"),
    )
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: MagicMock(device=device))

    assert registry._auto_preferences[OptimizationStage.PREPROCESS] == ()
    preprocessor_selection = registry.resolve_selection(
        stage=OptimizationStage.PREPROCESS,
        requested_id=AUTO_IMPLEMENTATION_ID,
        context=ExecutionContext(
            device_kind="gpu",
            device="cuda:0",
            compute_capability=(8, 7),
            runtime_components={"torch": True, "torchvision": True, "triton": True},
        ),
        allow_fallback=True,
    )
    assert isinstance(
        preprocessor_selection.implementation,
        BaseYOLO26DepthPreprocessor,
    )
    assert (
        not TritonCV2ResizeFusedConvertYOLO26DepthPreprocessor.metadata.changes_numerics
    )
    assert (
        not TritonCV2ResizePinnedFusedConvertYOLO26DepthPreprocessor.metadata.changes_numerics
    )

    assert registry._auto_preferences[OptimizationStage.SCHEDULER] == ()
    scheduler_selection = registry.resolve_selection(
        stage=OptimizationStage.SCHEDULER,
        requested_id=AUTO_IMPLEMENTATION_ID,
        context=ExecutionContext(
            device_kind="gpu",
            device="cuda:0",
            compute_capability=(8, 7),
            runtime_components={"torch": True},
        ),
        allow_fallback=True,
    )
    assert isinstance(
        scheduler_selection.implementation,
        BaseYOLO26DepthExecutionScheduler,
    )
    assert not CUDAEventHandoffYOLO26DepthExecutionScheduler.metadata.changes_numerics

    assert registry._auto_preferences[OptimizationStage.POSTPROCESS] == ()
    selection = registry.resolve_selection(
        stage=OptimizationStage.POSTPROCESS,
        requested_id=AUTO_IMPLEMENTATION_ID,
        context=ExecutionContext(
            device_kind="gpu",
            device="cuda:0",
            compute_capability=(8, 7),
            runtime_components={"torch": True, "torchvision": True, "triton": True},
        ),
        allow_fallback=True,
    )
    assert isinstance(selection.implementation, BaseYOLO26DepthPostprocessor)
    assert TritonAAYOLO26DepthPostprocessor.metadata.changes_numerics
    assert not ExactTritonAAYOLO26DepthPostprocessor.metadata.changes_numerics
    assert not ExactFusedTritonAAYOLO26DepthPostprocessor.metadata.changes_numerics


def test_triton_preprocessor_uses_base_only_for_frozen_base_source_shape():
    import numpy as np

    from inference_models.models.yolo26.optimization.preprocessors import (
        _use_base_preprocess_path,
    )

    base_image = np.zeros((480, 640, 3), dtype=np.uint8)
    large_image = np.zeros((2160, 3840, 3), dtype=np.uint8)

    assert _use_base_preprocess_path(base_image)
    assert not _use_base_preprocess_path(large_image)


def test_triton_preprocessor_preserves_opencv_letterbox_image_and_metadata():
    import numpy as np

    from inference_models.models.common.roboflow.model_packages import (
        ColorMode,
        ImagePreProcessing,
        NetworkInputDefinition,
        ResizeMode,
        TrainingInputSize,
    )
    from inference_models.models.common.roboflow.pre_processing import (
        pre_process_network_input,
    )
    from inference_models.models.yolo26.optimization.preprocessors import (
        _prepare_large_numpy_image,
    )

    image = np.arange(12 * 20 * 3, dtype=np.uint8).reshape((12, 20, 3))
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(width=8, height=8),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        padding_value=127,
        input_channels=3,
        scaling_factor=255,
        normalization=None,
    )

    prepared_image, candidate_metadata = _prepare_large_numpy_image(
        image=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        input_color_mode=ColorMode.BGR,
        pre_processing_overrides=None,
    )
    pinned_buffer = np.empty_like(prepared_image)
    pinned_image, pinned_metadata = _prepare_large_numpy_image(
        image=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        input_color_mode=ColorMode.BGR,
        pre_processing_overrides=None,
        output_buffer=pinned_buffer,
    )
    base_tensor, base_metadata = pre_process_network_input(
        images=[image],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )
    candidate_tensor = (
        torch.from_numpy(prepared_image).unsqueeze(0).permute(0, 3, 1, 2)[:, [2, 1, 0]]
        / 255
    )

    assert torch.equal(candidate_tensor, base_tensor)
    assert candidate_metadata == base_metadata[0]
    assert pinned_image is pinned_buffer
    assert np.array_equal(pinned_image, prepared_image)
    assert pinned_metadata == candidate_metadata


def test_exact_fused_v3_compacts_filters_and_dispatches_small_outputs():
    from inference_models.models.yolo26.triton_depth_postprocess import (
        _maximum_axis_filter_size,
        _use_torchvision_base_path,
    )

    assert _maximum_axis_filter_size(input_size=576, output_size=480) == 3
    assert _maximum_axis_filter_size(input_size=432, output_size=2160) == 2
    assert _use_torchvision_base_path((480, 640))
    assert not _use_torchvision_base_path((2160, 3840))


def test_post_process_depth_estimation_map_scales_padding_for_low_resolution_output():
    """When the model emits a map at a lower resolution than the network input
    (e.g. eval-mode H/4 output), padding offsets are scaled to map space before
    cropping."""
    from inference_models.models.common.roboflow.post_processing import (
        post_process_depth_estimation_map,
    )

    # network input 16x16 with pad_top/bottom 4 in image space; map is 8x8, so
    # 2 rows of padding at map resolution
    depth = torch.zeros(1, 1, 8, 8)
    depth[0, 0, 2:6, :] = 9.0

    results = post_process_depth_estimation_map(
        model_results=depth,
        pre_processing_meta=_meta(
            16, 16, pad_top=4, pad_bottom=4, after_h=8, after_w=16
        ),
        device=torch.device("cpu"),
    )

    assert results[0].shape == (8, 16)
    assert torch.allclose(results[0], torch.full((8, 16), 9.0))


def test_post_process_depth_estimation_map_places_static_crop_on_zero_canvas():
    """With static-crop pre-processing, the depth map is placed at the crop
    offset on an original-size canvas; depth outside the crop is unknown and
    left at 0.0."""
    from inference_models.models.common.roboflow.post_processing import (
        post_process_depth_estimation_map,
    )

    h, w = 8, 8
    depth = torch.full((1, 1, h, w), 5.0)

    results = post_process_depth_estimation_map(
        model_results=depth,
        pre_processing_meta=_meta(
            h,
            w,
            original_h=12,
            original_w=12,
            crop_offset_x=2,
            crop_offset_y=2,
        ),
        device=torch.device("cpu"),
    )

    assert results[0].shape == (12, 12)
    assert torch.allclose(results[0][2:10, 2:10], torch.full((8, 8), 5.0))
    assert torch.all(results[0][:2, :] == 0.0)
    assert torch.all(results[0][10:, :] == 0.0)
    assert torch.all(results[0][:, :2] == 0.0)
    assert torch.all(results[0][:, 10:] == 0.0)


def test_yolo26_depth_estimation_registered():
    from inference_models.models.auto_loaders.entities import BackendType
    from inference_models.models.auto_loaders.models_registry import (
        DEPTH_ESTIMATION_TASK,
        REGISTERED_MODELS,
    )

    for backend in (BackendType.ONNX, BackendType.TORCH_SCRIPT, BackendType.TRT):
        assert (
            "yolo26",
            DEPTH_ESTIMATION_TASK,
            backend,
        ) in REGISTERED_MODELS, (
            f"Missing yolo26 depth estimation entry for backend {backend}"
        )


def test_yolo26_depth_estimation_onnx_imports():
    import pytest

    pytest.importorskip("onnxruntime")
    from inference_models.models.yolo26.yolo26_depth_estimation_onnx import (
        YOLO26ForDepthEstimationOnnx,
    )

    assert hasattr(YOLO26ForDepthEstimationOnnx, "from_pretrained")


def test_yolo26_depth_estimation_torch_script_imports():
    from inference_models.models.yolo26.yolo26_depth_estimation_torch_script import (
        YOLO26ForDepthEstimationTorchScript,
    )

    assert hasattr(YOLO26ForDepthEstimationTorchScript, "from_pretrained")
