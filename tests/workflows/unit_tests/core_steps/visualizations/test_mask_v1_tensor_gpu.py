import cv2
import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.workflows.core_steps.visualizations.common.base_tensor import (
    to_supervision_for_annotation,
)
from inference.core.workflows.core_steps.visualizations.mask.v1_tensor import (
    _gpu_composite_eligible,
    gpu_mask_composite,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

OPACITY = 0.5
PALETTE = sv.ColorPalette.DEFAULT
SCENE_H, SCENE_W = 540, 960


def _build_dense_detections(
    masks: np.ndarray,
    boxes: np.ndarray,
    class_id: np.ndarray,
    device: str,
) -> InstanceDetections:
    n = masks.shape[0]
    return InstanceDetections(
        xyxy=torch.tensor(boxes, dtype=torch.int32, device=device),
        class_id=torch.tensor(class_id, dtype=torch.int32, device=device),
        confidence=torch.full((n,), 0.9, device=device),
        mask=torch.from_numpy(masks).to(device),
        image_metadata={"class_names": {i: f"c{i}" for i in range(10)}},
    )


def _single_mask_inputs(h: int = 64, w: int = 64):
    masks = np.zeros((1, h, w), dtype=bool)
    masks[0, 10:30, 12:40] = True
    boxes = np.array([[12, 10, 39, 29]], dtype=np.int32)
    class_id = np.array([1], dtype=np.int32)
    return masks, boxes, class_id


def test_gpu_composite_eligible_when_mask_is_numpy_array() -> None:
    # given
    masks, boxes, class_id = _single_mask_inputs()
    predictions = InstanceDetections(
        xyxy=torch.tensor(boxes, dtype=torch.int32),
        class_id=torch.tensor(class_id, dtype=torch.int32),
        confidence=torch.full((1,), 0.9),
        mask=masks,
    )

    # when
    result = _gpu_composite_eligible(predictions, "CLASS")

    # then
    assert result is False


def test_gpu_composite_eligible_when_track_lookup_is_requested() -> None:
    # given
    masks, boxes, class_id = _single_mask_inputs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions = _build_dense_detections(masks, boxes, class_id, device=device)

    # when
    result = _gpu_composite_eligible(predictions, "TRACK")

    # then: TRACK lookup always keeps the sv path
    assert result is False


def test_gpu_composite_eligible_when_there_are_no_detections() -> None:
    # given
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions = InstanceDetections(
        xyxy=torch.zeros((0, 4), dtype=torch.int32, device=device),
        class_id=torch.zeros((0,), dtype=torch.int32, device=device),
        confidence=torch.zeros((0,), device=device),
        mask=torch.zeros((0, 8, 8), dtype=torch.bool, device=device),
    )

    # when
    result = _gpu_composite_eligible(predictions, "CLASS")

    # then
    assert result is False


def test_gpu_composite_eligible_when_predictions_are_not_instance_detections() -> None:
    # when
    result = _gpu_composite_eligible(object(), "CLASS")

    # then
    assert result is False


def test_gpu_composite_eligible_when_dense_mask_is_on_cpu() -> None:
    # given: the compositor is device-agnostic — device gating is the loader's
    # job (this block only registers on the tensor pipeline)
    masks, boxes, class_id = _single_mask_inputs()
    predictions = _build_dense_detections(masks, boxes, class_id, device="cpu")

    # when
    result = _gpu_composite_eligible(predictions, "CLASS")

    # then
    assert result is True


def test_gpu_mask_composite_pixel_parity_on_cpu_tensors() -> None:
    # given: small CPU-only parity case so the full compositor runs in CPU CI
    scene = _make_scene(404, h=128, w=192)
    masks = [
        _rect_mask(10, 10, 90, 80)[:128, :192],
        _rect_mask(50, 40, 150, 110)[:128, :192],
    ]
    _assert_pixel_parity_with_sv_annotator(masks, scene, device="cpu")


def test_gpu_composite_eligible_when_mask_carrier_is_rle() -> None:
    # given: RLE-carrier predictions take the sv fallback — the compositor
    # handles the dense (N, H, W) bool tensor carrier only
    masks, boxes, class_id = _single_mask_inputs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions = InstanceDetections(
        xyxy=torch.tensor(boxes, dtype=torch.int32, device=device),
        class_id=torch.tensor(class_id, dtype=torch.int32, device=device),
        confidence=torch.full((1,), 0.9, device=device),
        mask=InstancesRLEMasks(image_size=(64, 64), masks=[b""]),
    )

    # when
    result = _gpu_composite_eligible(predictions, "CLASS")

    # then
    assert result is False


@requires_cuda
@pytest.mark.parametrize("color_axis", ["CLASS", "INDEX"])
def test_gpu_composite_eligible_when_dense_cuda_bool_mask_is_given(
    color_axis: str,
) -> None:
    # given
    masks, boxes, class_id = _single_mask_inputs()
    predictions = _build_dense_detections(masks, boxes, class_id, device="cuda")

    # when
    result = _gpu_composite_eligible(predictions, color_axis)

    # then
    assert result is True


@requires_cuda
def test_gpu_composite_eligible_when_dense_cuda_mask_has_wrong_dtype() -> None:
    # given
    masks, boxes, class_id = _single_mask_inputs()
    predictions = _build_dense_detections(masks, boxes, class_id, device="cuda")
    predictions.mask = predictions.mask.float()

    # when
    result = _gpu_composite_eligible(predictions, "CLASS")

    # then
    assert result is False


def _make_scene(seed: int, h: int = SCENE_H, w: int = SCENE_W) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    b = 180.0 * xx + 40.0 * yy
    g = 150.0 * yy + 50.0 * (1.0 - xx)
    r = 160.0 * (1.0 - xx) * (1.0 - yy) + 60.0
    grad = np.stack([b, g, r], axis=2)
    noise = rng.integers(0, 255, (h, w, 3)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=9)
    scene = grad + 0.45 * (noise - 127.0)
    return np.clip(scene, 0, 255).astype(np.uint8)


def _ellipse_mask(
    cx: float, cy: float, ax: float, ay: float, angle: float = 0.0
) -> np.ndarray:
    mask = np.zeros((SCENE_H, SCENE_W), dtype=np.uint8)
    cv2.ellipse(
        mask, (int(cx), int(cy)), (int(ax), int(ay)), float(angle), 0, 360, 1, -1
    )
    return mask.astype(bool)


def _rect_mask(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros((SCENE_H, SCENE_W), dtype=bool)
    mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def _tight_xyxy(mask: np.ndarray) -> list:
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _scenario_partial_overlap() -> list:
    return [
        _ellipse_mask(360, 270, 240, 180),
        _ellipse_mask(600, 270, 240, 180),
    ]


def _scenario_three_level_nesting() -> list:
    return [
        _ellipse_mask(480, 270, 320, 220),
        _ellipse_mask(480, 270, 180, 130),
        _ellipse_mask(480, 270, 80, 55),
    ]


def _scenario_three_way_chain() -> list:
    masks = [
        _ellipse_mask(350, 220, 200, 150, angle=15),
        _ellipse_mask(510, 300, 200, 150, angle=-10),
        _ellipse_mask(650, 210, 200, 150, angle=25),
    ]
    assert (masks[0] & masks[1] & masks[2]).sum() > 0
    return masks


def _scenario_equal_area_tie() -> list:
    big = _ellipse_mask(480, 280, 330, 210)
    tie_1 = _rect_mask(340, 190, 489, 339)
    tie_2 = _rect_mask(430, 250, 579, 399)
    assert tie_1.sum() == tie_2.sum()
    return [big, tie_1, tie_2]


def _scenario_twelve_mask_cluster() -> list:
    masks = []
    center_x, center_y = 480, 270
    for i in range(12):
        angle = 2 * np.pi * i / 12
        cx = center_x + 150 * np.cos(angle)
        cy = center_y + 100 * np.sin(angle)
        masks.append(
            _ellipse_mask(cx, cy, 170 + 7 * i, 90 + 4 * i, angle=np.degrees(angle))
        )
    common = masks[0].copy()
    for mask in masks[1:]:
        common &= mask
    assert common.sum() > 0
    return masks


def _scenario_edge_touching() -> list:
    return [
        _rect_mask(0, 0, 199, 159),
        _rect_mask(SCENE_W - 240, SCENE_H - 180, SCENE_W - 1, SCENE_H - 1),
        _ellipse_mask(0, SCENE_H // 2, 160, 120),
        _ellipse_mask(SCENE_W - 1, 100, 180, 140),
    ]


def _assert_pixel_parity_with_sv_annotator(
    masks: list, scene: np.ndarray, device: str = "cuda"
) -> None:
    dense = np.stack(masks, axis=0)
    boxes = np.asarray([_tight_xyxy(mask) for mask in masks], dtype=np.int32)
    class_id = (np.arange(len(masks)) % 10).astype(np.int32)
    detections = _build_dense_detections(dense, boxes, class_id, device=device)
    assert _gpu_composite_eligible(detections, "CLASS") is True

    sv_detections = to_supervision_for_annotation(detections)
    annotator = sv.MaskAnnotator(
        color=PALETTE, color_lookup=sv.ColorLookup.CLASS, opacity=OPACITY
    )
    expected = annotator.annotate(scene=scene.copy(), detections=sv_detections)

    colors_bgr = np.asarray(
        [PALETTE.by_idx(int(c)).as_bgr() for c in class_id], dtype=np.uint8
    )
    actual = gpu_mask_composite(scene, detections, colors_bgr, OPACITY)

    exact_match_rate = float((expected == actual).all(axis=2).mean())
    max_channel_diff = int(
        np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max()
    )
    assert exact_match_rate == 1.0
    assert max_channel_diff == 0


@requires_cuda
@pytest.mark.parametrize(
    "scenario",
    [
        _scenario_partial_overlap,
        _scenario_three_level_nesting,
        _scenario_three_way_chain,
        _scenario_equal_area_tie,
        _scenario_twelve_mask_cluster,
        _scenario_edge_touching,
    ],
    ids=[
        "partial_overlap",
        "three_level_nesting",
        "three_way_chain",
        "equal_area_tie",
        "twelve_mask_cluster",
        "edge_touching",
    ],
)
def test_gpu_mask_composite_pixel_parity_with_sv_annotator(scenario) -> None:
    # given
    scene = _make_scene(101)
    masks = scenario()

    # when / then
    _assert_pixel_parity_with_sv_annotator(masks, scene)


@requires_cuda
def test_gpu_mask_composite_pixel_parity_on_random_masks() -> None:
    # given: 15 random box-bounded blobs, including a degenerate all-False mask
    rng = np.random.default_rng(7)
    masks = []
    for i in range(15):
        crop_w = int(rng.integers(90, 360))
        crop_h = int(rng.integers(90, 360))
        x1 = int(rng.integers(0, SCENE_W - crop_w - 1))
        y1 = int(rng.integers(0, SCENE_H - crop_h - 1))
        mask = np.zeros((SCENE_H, SCENE_W), dtype=bool)
        if i != 7:  # detection 7 stays empty -> both paths paint nothing
            low = rng.random((max(2, crop_h // 24), max(2, crop_w // 24)))
            up = torch.nn.functional.interpolate(
                torch.from_numpy(low)[None, None].float(),
                size=(crop_h, crop_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
            blob = up > 0.45
            if not blob.any():
                blob[crop_h // 2, crop_w // 2] = True
            mask[y1 : y1 + crop_h, x1 : x1 + crop_w] = blob
        else:
            mask[y1, x1] = True  # single-pixel mask keeps a valid tight box
        masks.append(mask)

    scene = _make_scene(202)

    # when / then
    _assert_pixel_parity_with_sv_annotator(masks, scene)


@requires_cuda
def test_gpu_mask_composite_chw_rgb_tensor_scene_matches_sv_annotator() -> None:
    # given: the WorkflowImageData.tensor_image contract - CHW uint8 RGB on device
    scene = _make_scene(303)
    masks = _scenario_three_way_chain()
    dense = np.stack(masks, axis=0)
    boxes = np.asarray([_tight_xyxy(mask) for mask in masks], dtype=np.int32)
    class_id = (np.arange(len(masks)) % 10).astype(np.int32)
    detections = _build_dense_detections(dense, boxes, class_id, device="cuda")
    scene_chw_rgb = (
        torch.from_numpy(scene[:, :, ::-1].copy()).permute(2, 0, 1).contiguous().cuda()
    )

    sv_detections = to_supervision_for_annotation(detections)
    annotator = sv.MaskAnnotator(
        color=PALETTE, color_lookup=sv.ColorLookup.CLASS, opacity=OPACITY
    )
    expected = annotator.annotate(scene=scene.copy(), detections=sv_detections)
    colors_bgr = np.asarray(
        [PALETTE.by_idx(int(c)).as_bgr() for c in class_id], dtype=np.uint8
    )

    # when
    annotated_tensor = gpu_mask_composite(
        scene_chw_rgb,
        detections,
        colors_bgr,
        OPACITY,
        return_tensor=True,
        scene_layout="chw_rgb",
    )

    # then: result stays on device; converting back to HWC BGR matches sv exactly
    assert annotated_tensor.is_cuda
    assert annotated_tensor.data_ptr() == scene_chw_rgb.data_ptr()  # in-place
    actual = annotated_tensor.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    assert float((expected == actual).all(axis=2).mean()) == 1.0
    assert int(np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max()) == 0
