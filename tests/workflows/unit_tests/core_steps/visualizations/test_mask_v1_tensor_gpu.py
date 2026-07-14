import cv2
import numpy as np
import pytest
import supervision as sv
import torch
from pycocotools import mask as mask_utils

from inference.core.workflows.core_steps.visualizations.common.base_tensor import (
    to_supervision_for_annotation,
)
from inference.core.workflows.core_steps.visualizations.mask.v1_tensor import (
    _coco_rle_counts_to_runs,
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

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


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


def _build_rle_detections(
    masks: list,
    boxes: np.ndarray,
    class_id: np.ndarray,
    device: str,
) -> InstanceDetections:
    n = len(masks)
    payloads = [
        mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"]
        for mask in masks
    ]
    return InstanceDetections(
        xyxy=torch.tensor(boxes, dtype=torch.int32, device=device),
        class_id=torch.tensor(class_id, dtype=torch.int32, device=device),
        confidence=torch.full((n,), 0.9, device=device),
        mask=InstancesRLEMasks(image_size=tuple(masks[0].shape), masks=payloads),
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


def test_gpu_composite_eligible_when_class_id_is_missing_for_class_lookup() -> None:
    # given
    masks, boxes, class_id = _single_mask_inputs()
    predictions = _build_dense_detections(masks, boxes, class_id, device="cpu")
    predictions.class_id = None

    # when / then: CLASS lookup needs class ids; INDEX does not
    assert _gpu_composite_eligible(predictions, "CLASS") is False
    assert _gpu_composite_eligible(predictions, "INDEX") is True


def test_gpu_composite_eligible_when_dense_mask_is_on_cpu() -> None:
    # given: the compositor is device-agnostic — device gating is the loader's
    # job (this block only registers on the tensor pipeline)
    masks, boxes, class_id = _single_mask_inputs()
    predictions = _build_dense_detections(masks, boxes, class_id, device="cpu")

    # when
    result = _gpu_composite_eligible(predictions, "CLASS")

    # then
    assert result is True


def test_gpu_composite_eligible_when_mask_carrier_is_rle() -> None:
    # given: the compositor decodes the RLE carrier natively
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
    assert result is True


def test_gpu_composite_eligible_when_rle_mask_count_mismatches_boxes() -> None:
    # given: 1 box but 2 RLE payloads
    masks, boxes, class_id = _single_mask_inputs()
    predictions = InstanceDetections(
        xyxy=torch.tensor(boxes, dtype=torch.int32),
        class_id=torch.tensor(class_id, dtype=torch.int32),
        confidence=torch.full((1,), 0.9),
        mask=InstancesRLEMasks(image_size=(64, 64), masks=[b"", b""]),
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


def _scenario_disjoint_masks() -> list:
    return [
        _ellipse_mask(180, 140, 120, 90),
        _rect_mask(430, 60, 640, 240),
        _ellipse_mask(760, 400, 140, 100, angle=30),
    ]


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


def _random_blob_masks(seed: int = 7, n: int = 15) -> list:
    # 15 random box-bounded blobs, including a single-pixel near-empty mask
    rng = np.random.default_rng(seed)
    masks = []
    for i in range(n):
        crop_w = int(rng.integers(90, 360))
        crop_h = int(rng.integers(90, 360))
        x1 = int(rng.integers(0, SCENE_W - crop_w - 1))
        y1 = int(rng.integers(0, SCENE_H - crop_h - 1))
        mask = np.zeros((SCENE_H, SCENE_W), dtype=bool)
        if i != 7:
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
    return masks


OVERLAP_SCENARIOS = [
    _scenario_partial_overlap,
    _scenario_three_level_nesting,
    _scenario_three_way_chain,
    _scenario_twelve_mask_cluster,
    _scenario_edge_touching,
    _random_blob_masks,
]
OVERLAP_SCENARIO_IDS = [
    "partial_overlap",
    "three_level_nesting",
    "three_way_chain",
    "twelve_mask_cluster",
    "edge_touching",
    "random_blobs",
]


def _reference_blend_all(
    scene: np.ndarray, masks: list, colors_bgr: np.ndarray, opacity: float
) -> np.ndarray:
    """Order-independent blend-all reference: the overlay color of a pixel is
    the mean of the covering masks' colors, alpha-composited once with the
    scene (np.round is round-half-to-even, like the compositor)."""
    stack = np.stack(masks).astype(np.float64)  # (N, H, W)
    count = stack.sum(axis=0)  # (H, W)
    premul = np.einsum("nhw,nc->hwc", stack, colors_bgr.astype(np.float64) * opacity)
    hit = count > 0
    out = scene.astype(np.float64)
    out[hit] = np.round(premul[hit] / count[hit][:, None] + (1.0 - opacity) * out[hit])
    return out.astype(np.uint8)


def _detections_and_colors(masks: list, device: str, rle: bool = False):
    boxes = np.asarray([_tight_xyxy(mask) for mask in masks], dtype=np.int32)
    class_id = (np.arange(len(masks)) % 10).astype(np.int32)
    if rle:
        detections = _build_rle_detections(masks, boxes, class_id, device=device)
    else:
        detections = _build_dense_detections(
            np.stack(masks, axis=0), boxes, class_id, device=device
        )
    colors_bgr = np.asarray(
        [PALETTE.by_idx(int(c)).as_bgr() for c in class_id], dtype=np.uint8
    )
    return detections, colors_bgr


def _composite_bgr(
    scene_bgr: np.ndarray,
    detections: InstanceDetections,
    colors_bgr: np.ndarray,
    opacity: float,
    device: str = "cpu",
) -> np.ndarray:
    """Test adapter: the production compositor is tensor-only (CHW RGB uint8
    in, the same tensor out, mutated in place) — wrap numpy HWC BGR scenes and
    colors so parity checks against sv / the numpy reference stay convenient."""
    scene_t = (
        torch.from_numpy(scene_bgr[:, :, ::-1].copy())
        .permute(2, 0, 1)
        .contiguous()
        .to(device)
    )
    out = gpu_mask_composite(
        scene_t, detections, np.ascontiguousarray(colors_bgr[:, ::-1]), opacity
    )
    return out.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]


def _runs_to_dense(runs: np.ndarray, h: int, w: int) -> np.ndarray:
    flat = np.zeros(h * w, dtype=bool)
    position, value = 0, False
    for run in runs:
        flat[position : position + run] = value
        position += int(run)
        value = not value
    return flat.reshape((h, w), order="F")


@pytest.mark.parametrize("seed", [3, 11, 42])
def test_coco_rle_counts_decoder_matches_pycocotools(seed: int) -> None:
    # given: blobs with long background runs (multi-char varints) and jagged
    # boundaries (negative deltas)
    masks = _random_blob_masks(seed=seed, n=6)

    for mask in masks:
        encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))

        # when
        runs = _coco_rle_counts_to_runs(encoded["counts"])
        rebuilt = _runs_to_dense(runs, SCENE_H, SCENE_W)

        # then
        assert np.array_equal(rebuilt, mask_utils.decode(encoded).astype(bool))


def test_coco_rle_counts_decoder_accepts_uncompressed_lists() -> None:
    assert np.array_equal(
        _coco_rle_counts_to_runs([3, 2, 5]), np.array([3, 2, 5], dtype=np.int64)
    )
    assert _coco_rle_counts_to_runs(b"").size == 0


@pytest.mark.parametrize("device", DEVICES)
def test_gpu_mask_composite_matches_sv_annotator_on_disjoint_masks(
    device: str,
) -> None:
    # given: no overlaps — single-covered pixels keep bit-exact sv parity
    scene = _make_scene(101)
    masks = _scenario_disjoint_masks()
    detections, colors_bgr = _detections_and_colors(masks, device=device)
    annotator = sv.MaskAnnotator(
        color=PALETTE, color_lookup=sv.ColorLookup.CLASS, opacity=OPACITY
    )
    expected = annotator.annotate(
        scene=scene.copy(), detections=to_supervision_for_annotation(detections)
    )

    # when
    actual = _composite_bgr(scene, detections, colors_bgr, OPACITY, device=device)

    # then
    assert float((expected == actual).all(axis=2).mean()) == 1.0
    assert int(np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max()) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rle", [False, True], ids=["dense", "rle"])
@pytest.mark.parametrize("scenario", OVERLAP_SCENARIOS, ids=OVERLAP_SCENARIO_IDS)
def test_gpu_mask_composite_matches_blend_all_reference(
    scenario, rle: bool, device: str
) -> None:
    # given
    scene = _make_scene(101)
    masks = scenario()
    detections, colors_bgr = _detections_and_colors(masks, device=device, rle=rle)
    expected = _reference_blend_all(scene, masks, colors_bgr, OPACITY)

    # when
    actual = _composite_bgr(scene, detections, colors_bgr, OPACITY, device=device)

    # then
    assert float((expected == actual).all(axis=2).mean()) == 1.0
    assert int(np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max()) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_gpu_mask_composite_rle_carrier_matches_dense_carrier(device: str) -> None:
    # given: the same masks through both carriers
    scene = _make_scene(505)
    masks = _scenario_twelve_mask_cluster()
    dense_detections, colors_bgr = _detections_and_colors(masks, device=device)
    rle_detections, _ = _detections_and_colors(masks, device=device, rle=True)

    # when
    from_dense = _composite_bgr(
        scene, dense_detections, colors_bgr, OPACITY, device=device
    )
    from_rle = _composite_bgr(scene, rle_detections, colors_bgr, OPACITY, device=device)

    # then
    assert np.array_equal(from_dense, from_rle)


@pytest.mark.parametrize("device", DEVICES)
def test_gpu_mask_composite_is_order_independent(device: str) -> None:
    # given: blend-all semantics must not depend on detection order
    scene = _make_scene(606)
    masks = _scenario_three_way_chain()
    detections, colors_bgr = _detections_and_colors(masks, device=device)
    permutation = [2, 0, 1]
    permuted, permuted_colors = _detections_and_colors(
        [masks[i] for i in permutation], device=device
    )
    # keep each mask's color stable under the permutation
    permuted.class_id = detections.class_id[permutation]
    permuted_colors = colors_bgr[permutation]

    # when
    original = _composite_bgr(scene, detections, colors_bgr, OPACITY, device=device)
    shuffled = _composite_bgr(scene, permuted, permuted_colors, OPACITY, device=device)

    # then
    assert np.array_equal(original, shuffled)


def test_gpu_mask_composite_rejects_mismatched_mask_canvas() -> None:
    # given: mask canvas smaller than the scene — silent slicing would paint
    # misaligned masks; the block's try/except routes this to the sv fallback
    scene = _make_scene(707, h=256, w=256)
    masks, boxes, class_id = _single_mask_inputs(h=64, w=64)
    detections = _build_dense_detections(masks, boxes, class_id, device="cpu")
    colors_bgr = np.asarray([[255, 0, 0]], dtype=np.uint8)

    # when / then
    with pytest.raises(ValueError, match="does not match scene"):
        _composite_bgr(scene, detections, colors_bgr, OPACITY)


def test_gpu_mask_composite_with_fully_off_frame_boxes_leaves_scene_unchanged() -> None:
    # given: union ROI collapses to nothing
    scene = _make_scene(808, h=128, w=128)
    masks = np.zeros((1, 128, 128), dtype=bool)
    boxes = np.array([[-50, -50, -10, -10]], dtype=np.int32)
    detections = _build_dense_detections(
        masks, boxes, np.array([0], dtype=np.int32), device="cpu"
    )
    colors_bgr = np.asarray([[255, 0, 0]], dtype=np.uint8)

    # when
    actual = _composite_bgr(scene, detections, colors_bgr, OPACITY)

    # then
    assert np.array_equal(actual, scene)


@requires_cuda
@pytest.mark.parametrize("rle", [False, True], ids=["dense", "rle"])
def test_gpu_mask_composite_chw_rgb_tensor_scene_matches_reference(
    rle: bool,
) -> None:
    # given: the WorkflowImageData.tensor_image contract - CHW uint8 RGB on device
    scene = _make_scene(303)
    masks = _scenario_three_way_chain()
    detections, colors_bgr = _detections_and_colors(masks, device="cuda", rle=rle)
    scene_chw_rgb = (
        torch.from_numpy(scene[:, :, ::-1].copy()).permute(2, 0, 1).contiguous().cuda()
    )
    expected = _reference_blend_all(scene, masks, colors_bgr, OPACITY)

    # when
    annotated_tensor = gpu_mask_composite(
        scene_chw_rgb,
        detections,
        np.ascontiguousarray(colors_bgr[:, ::-1]),
        OPACITY,
    )

    # then: result stays on device; converting back to HWC BGR matches the
    # reference exactly
    assert annotated_tensor.is_cuda
    assert annotated_tensor.data_ptr() == scene_chw_rgb.data_ptr()  # in-place
    actual = annotated_tensor.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    assert float((expected == actual).all(axis=2).mean()) == 1.0
    assert int(np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max()) == 0
