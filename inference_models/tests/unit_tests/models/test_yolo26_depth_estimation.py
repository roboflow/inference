from unittest.mock import MagicMock

import torch


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
            static_crop_offset=MagicMock(offset_x=crop_offset_x, offset_y=crop_offset_y),
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
        pre_processing_meta=_meta(
            h, w, pad_top=2, pad_bottom=2, after_h=4, after_w=8
        ),
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
        assert ("yolo26", DEPTH_ESTIMATION_TASK, backend) in REGISTERED_MODELS, (
            f"Missing yolo26 depth estimation entry for backend {backend}"
        )


def test_yolo26_depth_estimation_onnx_imports():
    from inference_models.models.yolo26.yolo26_depth_estimation_onnx import (
        YOLO26ForDepthEstimationOnnx,
    )

    assert hasattr(YOLO26ForDepthEstimationOnnx, "from_pretrained")


def test_yolo26_depth_estimation_torch_script_imports():
    from inference_models.models.yolo26.yolo26_depth_estimation_torch_script import (
        YOLO26ForDepthEstimationTorchScript,
    )

    assert hasattr(YOLO26ForDepthEstimationTorchScript, "from_pretrained")
