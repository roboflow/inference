import cv2
import numpy as np
import pytest
import supervision as sv
import torch
from pycocotools import mask as mask_utils
from torch.utils._python_dispatch import TorchDispatchMode

from inference.core.workflows.core_steps.visualizations.common.base_tensor import (
    to_supervision_for_annotation,
)
from inference.core.workflows.core_steps.visualizations.mask.v1_tensor import (
    MaskVisualizationBlockV1,
    _coco_rle_counts_to_runs,
    _resolve_color_ids,
    _rle_to_dense_masks,
    gpu_mask_composite,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
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


def test_resolve_color_ids_matches_sv_semantics() -> None:
    # given
    masks, boxes, class_id = _single_mask_inputs()
    predictions = _build_dense_detections(masks, boxes, class_id, device="cpu")
    predictions.bboxes_metadata = [{"tracker_id": 7}]

    # when / then: same palette indices sv's resolve_color_idx would use,
    # returned as device tensors (never a device->host read)
    for axis, expected in (("CLASS", class_id), ("INDEX", [0]), ("TRACK", [7])):
        ids = _resolve_color_ids(predictions, axis, torch.device("cpu"))
        assert isinstance(ids, torch.Tensor) and ids.dtype == torch.int64
        assert np.array_equal(ids.numpy(), expected)


def test_resolve_color_ids_raises_when_ids_are_missing() -> None:
    # given: missing class_id / tracker_id crash with a clear ValueError,
    # raised before any mask work
    masks, boxes, class_id = _single_mask_inputs()
    predictions = _build_dense_detections(masks, boxes, class_id, device="cpu")
    predictions.class_id = None

    # when / then
    with pytest.raises(ValueError, match="resolve color by class"):
        _resolve_color_ids(predictions, "CLASS", torch.device("cpu"))
    with pytest.raises(ValueError, match="resolve color by track"):
        _resolve_color_ids(predictions, "TRACK", torch.device("cpu"))


def test_resolve_color_ids_passes_pending_track_sentinel_through() -> None:
    # given: sv's pending-track id (-1) must reach the caller unmapped so it
    # can be painted with sv's gray
    masks, boxes, class_id = _single_mask_inputs()
    predictions = _build_dense_detections(masks, boxes, class_id, device="cpu")
    predictions.bboxes_metadata = [{"tracker_id": -1}]

    # when
    ids = _resolve_color_ids(predictions, "TRACK", torch.device("cpu"))

    # then
    assert ids.tolist() == [-1]


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
    # masks clipped by every frame edge
    return [
        _rect_mask(0, 0, 199, 159),
        _rect_mask(SCENE_W - 240, SCENE_H - 180, SCENE_W - 1, SCENE_H - 1),
        _ellipse_mask(0, SCENE_H // 2, 160, 120),
        _ellipse_mask(SCENE_W - 1, 100, 180, 140),
    ]


def _scenario_full_frame() -> list:
    # one mask covering every pixel plus a nested one
    return [
        _rect_mask(0, 0, SCENE_W - 1, SCENE_H - 1),
        _ellipse_mask(480, 270, 180, 130),
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
    _scenario_full_frame,
    _random_blob_masks,
]
OVERLAP_SCENARIO_IDS = [
    "partial_overlap",
    "three_level_nesting",
    "three_way_chain",
    "twelve_mask_cluster",
    "edge_touching",
    "full_frame",
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
    colors_rgb_t = torch.from_numpy(np.ascontiguousarray(colors_bgr[:, ::-1])).to(
        device
    )
    out = gpu_mask_composite(scene_t, detections.mask, colors_rgb_t, opacity)
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
def test_rle_to_dense_masks_matches_pycocotools(device: str) -> None:
    # given
    masks = _random_blob_masks(seed=5, n=4)
    payloads = [
        mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"]
        for mask in masks
    ]
    rle = InstancesRLEMasks(image_size=(SCENE_H, SCENE_W), masks=payloads)

    # when
    dense = _rle_to_dense_masks(rle, torch.device(device))

    # then
    assert dense.shape == (len(masks), SCENE_H, SCENE_W)
    assert dense.dtype == torch.bool
    assert np.array_equal(dense.cpu().numpy(), np.stack(masks))


@pytest.mark.parametrize("device", DEVICES)
def test_gpu_mask_composite_matches_sv_annotator_on_disjoint_masks(
    device: str,
) -> None:
    # given: no overlaps — single-covered pixels use the exact same
    # premultiplied blend as sv (round-half-even like cvRound). Observed max
    # abs diff on this scenario: 0 (bit-exact); the tolerance of 1 covers
    # last-ulp division differences on other data.
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
    max_diff = int(np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max())
    assert max_diff <= 1
    mismatched_share = 1.0 - float((expected == actual).all(axis=2).mean())
    assert mismatched_share < 0.001


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
def test_gpu_mask_composite_leaves_unmasked_pixels_untouched(device: str) -> None:
    # given
    scene = _make_scene(404)
    masks = _scenario_disjoint_masks()
    detections, colors_bgr = _detections_and_colors(masks, device=device)
    covered = np.stack(masks).any(axis=0)

    # when
    actual = _composite_bgr(scene, detections, colors_bgr, OPACITY, device=device)

    # then: sv's addWeighted re-blends the whole frame; the compositor must
    # leave uncovered pixels BITWISE untouched
    assert np.array_equal(actual[~covered], scene[~covered])
    assert not np.array_equal(actual[covered], scene[covered])


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


def test_gpu_mask_composite_accepts_numpy_mask_stack() -> None:
    # given: a numpy-carried dense mask stack is uploaded and painted like the
    # torch carrier (there is no sv fallback to route it to any more)
    scene = _make_scene(111, h=128, w=128)
    masks = np.zeros((1, 128, 128), dtype=bool)
    masks[0, 20:60, 30:90] = True
    colors_bgr = np.asarray([PALETTE.by_idx(1).as_bgr()], dtype=np.uint8)
    expected = _reference_blend_all(scene, list(masks), colors_bgr, OPACITY)

    scene_t = torch.from_numpy(scene[:, :, ::-1].copy()).permute(2, 0, 1).contiguous()
    out = gpu_mask_composite(
        scene_t,
        masks,
        torch.from_numpy(np.ascontiguousarray(colors_bgr[:, ::-1])),
        OPACITY,
    )

    # then
    actual = out.permute(1, 2, 0).numpy()[:, :, ::-1]
    assert np.array_equal(actual, expected)


def test_gpu_mask_composite_casts_non_bool_masks() -> None:
    # given: float dense masks (previously served by the sv fallback's
    # astype(bool)) are cast on device — nonzero values paint
    scene = _make_scene(222, h=96, w=96)
    masks = np.zeros((1, 96, 96), dtype=bool)
    masks[0, 10:40, 10:40] = True
    colors_bgr = np.asarray([PALETTE.by_idx(2).as_bgr()], dtype=np.uint8)
    expected = _reference_blend_all(scene, list(masks), colors_bgr, OPACITY)

    scene_t = torch.from_numpy(scene[:, :, ::-1].copy()).permute(2, 0, 1).contiguous()
    out = gpu_mask_composite(
        scene_t,
        torch.from_numpy(masks).float(),
        torch.from_numpy(np.ascontiguousarray(colors_bgr[:, ::-1])),
        OPACITY,
    )

    # then
    actual = out.permute(1, 2, 0).numpy()[:, :, ::-1]
    assert np.array_equal(actual, expected)


def test_gpu_mask_composite_rejects_mismatched_mask_canvas() -> None:
    # given: mask canvas smaller than the scene — silent slicing would paint
    # misaligned masks; with the sv fallback gone this must raise loudly
    scene = _make_scene(707, h=256, w=256)
    masks, boxes, class_id = _single_mask_inputs(h=64, w=64)
    detections = _build_dense_detections(masks, boxes, class_id, device="cpu")
    colors_bgr = np.asarray([[255, 0, 0]], dtype=np.uint8)

    # when / then
    with pytest.raises(ValueError, match="does not match scene"):
        _composite_bgr(scene, detections, colors_bgr, OPACITY)


def test_gpu_mask_composite_with_all_false_masks_leaves_scene_unchanged() -> None:
    # given: no foreground pixels at all (boxes play no role in the composite)
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
        detections.mask,
        torch.from_numpy(np.ascontiguousarray(colors_bgr[:, ::-1])).cuda(),
        OPACITY,
    )

    # then: result stays on device; converting back to HWC BGR matches the
    # reference exactly
    assert annotated_tensor.is_cuda
    assert annotated_tensor.data_ptr() == scene_chw_rgb.data_ptr()  # in-place
    actual = annotated_tensor.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    assert float((expected == actual).all(axis=2).mean()) == 1.0
    assert int(np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max()) == 0


# --------------------------------------------------------------------------
# Block-level tests (MaskVisualizationBlockV1.run)
# --------------------------------------------------------------------------


def _tensor_backed_image(
    scene_bgr: np.ndarray, device: str = "cpu"
) -> WorkflowImageData:
    tensor = (
        torch.from_numpy(scene_bgr[:, :, ::-1].copy())
        .permute(2, 0, 1)
        .contiguous()
        .to(device)
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"), tensor_image=tensor
    )


def _numpy_backed_image(scene_bgr: np.ndarray) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"),
        numpy_image=scene_bgr,
    )


def _run_block(
    image: WorkflowImageData,
    detections,
    copy_image: bool = True,
    color_axis: str = "CLASS",
    opacity: float = OPACITY,
) -> WorkflowImageData:
    return MaskVisualizationBlockV1().run(
        image=image,
        predictions=detections,
        copy_image=copy_image,
        color_palette="DEFAULT",
        palette_size=10,
        custom_colors=None,
        color_axis=color_axis,
        opacity=opacity,
    )["image"]


def _empty_detections(device: str = "cpu") -> InstanceDetections:
    return InstanceDetections(
        xyxy=torch.zeros((0, 4), dtype=torch.int32, device=device),
        class_id=torch.zeros((0,), dtype=torch.int32, device=device),
        confidence=torch.zeros((0,), device=device),
        mask=torch.zeros((0, 8, 8), dtype=torch.bool, device=device),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rle", [False, True], ids=["dense", "rle"])
def test_block_tensor_path_matches_reference(rle: bool, device: str) -> None:
    # given
    scene = _make_scene(120)
    masks = _scenario_three_way_chain()
    detections, colors_bgr = _detections_and_colors(masks, device=device, rle=rle)
    image = _tensor_backed_image(scene, device=device)
    expected = _reference_blend_all(scene, masks, colors_bgr, OPACITY)

    # when
    out = _run_block(image, detections, copy_image=True)

    # then: tensor in -> tensor out, pixels match the reference
    assert out._tensor_image is not None and out._numpy_image is None
    actual = out._tensor_image.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    assert np.array_equal(actual, expected)


def test_block_copy_image_true_preserves_input_tensor() -> None:
    # given
    scene = _make_scene(121, h=128, w=128)
    masks = np.zeros((1, 128, 128), dtype=bool)
    masks[0, 20:60, 30:90] = True
    detections = _build_dense_detections(
        masks,
        np.array([[30, 20, 89, 59]], dtype=np.int32),
        np.array([1], dtype=np.int32),
        device="cpu",
    )
    image = _tensor_backed_image(scene)
    before = image.tensor_image.clone()

    # when
    out = _run_block(image, detections, copy_image=True)

    # then: independent storage, input untouched
    assert out._tensor_image.data_ptr() != image.tensor_image.data_ptr()
    assert torch.equal(image.tensor_image, before)
    assert not torch.equal(out._tensor_image, before)


def test_block_copy_image_false_mutates_input_tensor_in_place() -> None:
    # given
    scene = _make_scene(122, h=128, w=128)
    masks = np.zeros((1, 128, 128), dtype=bool)
    masks[0, 20:60, 30:90] = True
    detections = _build_dense_detections(
        masks,
        np.array([[30, 20, 89, 59]], dtype=np.int32),
        np.array([1], dtype=np.int32),
        device="cpu",
    )
    image = _tensor_backed_image(scene)
    input_tensor = image.tensor_image
    before = input_tensor.clone()

    # when
    out = _run_block(image, detections, copy_image=False)

    # then: same storage, annotated in place, numpy/base64 caches invalidated
    assert out._tensor_image.data_ptr() == input_tensor.data_ptr()
    assert not torch.equal(input_tensor, before)
    assert image._numpy_image is None and image._base64_image is None


@pytest.mark.parametrize("copy_image", [True, False])
def test_block_numpy_path_matches_sv_annotator_reference(copy_image: bool) -> None:
    # given: numpy-sourced images take the pre-rewrite sv.MaskAnnotator path
    # unchanged — bit-exact vs a directly-constructed annotator (painter's
    # algorithm on overlaps included, which the torch compositor diverges from)
    scene = _make_scene(123)
    masks = _scenario_three_way_chain()
    detections, _ = _detections_and_colors(masks, device="cpu")
    annotator = sv.MaskAnnotator(
        color=PALETTE, color_lookup=sv.ColorLookup.CLASS, opacity=OPACITY
    )
    expected = annotator.annotate(
        scene=scene.copy(), detections=to_supervision_for_annotation(detections)
    )
    image = _numpy_backed_image(scene.copy())

    # when
    out = _run_block(image, detections, copy_image=copy_image)

    # then: numpy in -> numpy out, sv's exact pixels
    assert out._numpy_image is not None and out._tensor_image is None
    assert np.array_equal(out._numpy_image, expected)


def test_block_numpy_path_copy_semantics() -> None:
    # given
    scene = _make_scene(124, h=128, w=128)
    masks = np.zeros((1, 128, 128), dtype=bool)
    masks[0, 20:60, 30:90] = True
    detections = _build_dense_detections(
        masks,
        np.array([[30, 20, 89, 59]], dtype=np.int32),
        np.array([1], dtype=np.int32),
        device="cpu",
    )

    # when: copy_image=True leaves the caller's buffer untouched
    image = _numpy_backed_image(scene.copy())
    out = _run_block(image, detections, copy_image=True)
    assert not np.shares_memory(out._numpy_image, image.numpy_image)
    assert np.array_equal(image.numpy_image, scene)

    # and: copy_image=False mutates the caller's buffer in place
    image = _numpy_backed_image(scene.copy())
    buffer = image.numpy_image
    out = _run_block(image, detections, copy_image=False)
    assert np.shares_memory(out._numpy_image, buffer)
    assert not np.array_equal(buffer, scene)


def test_block_numpy_path_wraps_negative_class_id() -> None:
    scene = np.zeros((64, 64, 3), dtype=np.uint8)
    masks, boxes, _ = _single_mask_inputs()
    detections = _build_dense_detections(
        masks,
        boxes,
        np.array([-1], dtype=np.int32),
        device="cpu",
    )

    out = _run_block(_numpy_backed_image(scene), detections)

    assert out._numpy_image is not None and out._tensor_image is None
    assert np.any(out._numpy_image)
    assert detections.class_id.tolist() == [-1]


def test_block_empty_predictions_take_the_tensor_passthrough() -> None:
    # given
    scene = _make_scene(125, h=64, w=64)
    image = _tensor_backed_image(scene)

    # when
    out_copy = _run_block(image, _empty_detections(), copy_image=True)
    out_share = _run_block(image, _empty_detections(), copy_image=False)

    # then: stays on-device; independent storage iff copy_image
    assert out_copy._tensor_image is not None and out_copy._numpy_image is None
    assert out_copy._tensor_image.data_ptr() != image.tensor_image.data_ptr()
    assert torch.equal(out_copy._tensor_image, image.tensor_image)
    assert out_share._tensor_image.data_ptr() == image.tensor_image.data_ptr()


def test_block_empty_predictions_on_numpy_sourced_image_stay_numpy() -> None:
    # given
    scene = _make_scene(126, h=64, w=64)
    image = _numpy_backed_image(scene)

    # when
    out = _run_block(image, _empty_detections(), copy_image=True)

    # then
    assert out._numpy_image is not None and out._tensor_image is None
    assert not np.shares_memory(out._numpy_image, image.numpy_image)
    assert np.array_equal(out._numpy_image, image.numpy_image)


@pytest.mark.parametrize("device", DEVICES)
def test_block_track_colors_match_sv_annotator(device: str) -> None:
    # given: disjoint masks colored by tracker_id (non-contiguous ids to
    # catch any id-vs-index mixup)
    scene = _make_scene(909)
    masks = _scenario_disjoint_masks()
    detections, _ = _detections_and_colors(masks, device=device)
    tracker_ids = [12, 3, 27]
    detections.bboxes_metadata = [{"tracker_id": tid} for tid in tracker_ids]
    annotator = sv.MaskAnnotator(
        color=PALETTE, color_lookup=sv.ColorLookup.TRACK, opacity=OPACITY
    )
    expected = annotator.annotate(
        scene=scene.copy(), detections=to_supervision_for_annotation(detections)
    )
    image = _tensor_backed_image(scene, device=device)

    # when
    out = _run_block(image, detections, color_axis="TRACK")

    # then
    actual = out._tensor_image.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    max_diff = int(np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max())
    assert max_diff <= 1
    mismatched_share = 1.0 - float((expected == actual).all(axis=2).mean())
    assert mismatched_share < 0.001


def test_block_pending_track_id_paints_sv_gray() -> None:
    # given: sv's pending-track sentinel (-1) maps to Color.GREY (128,128,128)
    scene = np.full((128, 128, 3), 200, dtype=np.uint8)
    masks = np.zeros((1, 128, 128), dtype=bool)
    masks[0, 20:60, 30:90] = True
    detections = _build_dense_detections(
        masks,
        np.array([[30, 20, 89, 59]], dtype=np.int32),
        np.array([1], dtype=np.int32),
        device="cpu",
    )
    detections.bboxes_metadata = [{"tracker_id": -1}]
    image = _tensor_backed_image(scene)

    # when
    out = _run_block(image, detections, color_axis="TRACK")

    # then: masked pixels = round(128*0.5 + 200*0.5) = 164 on every channel
    actual = out._tensor_image.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    assert np.array_equal(
        actual[masks[0]], np.full((int(masks[0].sum()), 3), 164, dtype=np.uint8)
    )
    assert np.array_equal(actual[~masks[0]], scene[~masks[0]])


def test_block_missing_tracker_ids_raise_value_error() -> None:
    # given
    scene = _make_scene(127, h=64, w=64)
    masks, boxes, class_id = _single_mask_inputs()
    detections = _build_dense_detections(masks, boxes, class_id, device="cpu")
    image = _tensor_backed_image(scene)

    # when / then: raised BEFORE any painting, input untouched
    before = image.tensor_image.clone()
    with pytest.raises(ValueError, match="resolve color by track"):
        _run_block(image, detections, color_axis="TRACK", copy_image=False)
    assert torch.equal(image.tensor_image, before)


def test_block_raises_for_unsupported_color_axis() -> None:
    scene = _make_scene(128, h=64, w=64)
    masks, boxes, class_id = _single_mask_inputs()
    detections = _build_dense_detections(masks, boxes, class_id, device="cpu")

    with pytest.raises(ValueError, match="color_axis"):
        _run_block(_tensor_backed_image(scene), detections, color_axis="SOMETHING")


def test_block_raises_for_non_instance_detections() -> None:
    scene = _make_scene(129, h=64, w=64)

    class _NotDetections:
        xyxy = torch.ones((1, 4))

    with pytest.raises(ValueError, match="instance segmentation"):
        _run_block(_tensor_backed_image(scene), _NotDetections())


def test_block_raises_for_mask_count_mismatch() -> None:
    # given: 1 box but 2 RLE payloads (previously silently sv-fallback-routed)
    scene = _make_scene(130, h=64, w=64)
    masks, boxes, class_id = _single_mask_inputs()
    detections = InstanceDetections(
        xyxy=torch.tensor(boxes, dtype=torch.int32),
        class_id=torch.tensor(class_id, dtype=torch.int32),
        confidence=torch.full((1,), 0.9),
        mask=InstancesRLEMasks(image_size=(64, 64), masks=[b"", b""]),
    )

    with pytest.raises(ValueError, match="RLE masks"):
        _run_block(_tensor_backed_image(scene), detections)


def test_block_raises_for_missing_mask_carrier() -> None:
    scene = _make_scene(131, h=64, w=64)
    masks, boxes, class_id = _single_mask_inputs()
    detections = _build_dense_detections(masks, boxes, class_id, device="cpu")
    detections.mask = None

    with pytest.raises(ValueError, match="no usable mask"):
        _run_block(_tensor_backed_image(scene), detections)


def test_block_canvas_mismatch_raises_without_partial_mutation() -> None:
    # given: the composite validates before its single staged write, so a
    # raising run must leave a copy_image=False input bitwise untouched
    scene = _make_scene(132, h=256, w=256)
    masks, boxes, class_id = _single_mask_inputs(h=64, w=64)
    detections = _build_dense_detections(masks, boxes, class_id, device="cpu")
    image = _tensor_backed_image(scene)
    before = image.tensor_image.clone()

    with pytest.raises(ValueError, match="does not match scene"):
        _run_block(image, detections, copy_image=False)
    assert torch.equal(image.tensor_image, before)


# --------------------------------------------------------------------------
# Sync audit: the dense path must enqueue a fixed, N-independent op sequence
# with no device->host reads.
# --------------------------------------------------------------------------


class _DispatchRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.calls.append((str(func), kwargs))
        return func(*args, **kwargs)


def _dense_composite_trace(n: int) -> list:
    scene = (
        torch.from_numpy(_make_scene(999, h=96, w=128)[:, :, ::-1].copy())
        .permute(2, 0, 1)
        .contiguous()
    )
    rng = np.random.default_rng(n)
    masks = torch.from_numpy(rng.random((n, 96, 128)) > 0.6)
    colors = torch.randint(0, 255, (n, 3), dtype=torch.uint8)
    recorder = _DispatchRecorder()
    with recorder:
        gpu_mask_composite(scene, masks, colors, OPACITY)
    return recorder.calls


def test_dense_composite_dispatches_no_sync_ops() -> None:
    # The structural zero-sync invariant (asserted on op names so it holds on
    # CPU-only runners too): no data-dependent indexing (`nonzero`,
    # `masked_select`), no host readback (`_local_scalar_dense` is `.item()`,
    # `aten.item` its alias), and no op asked to change a tensor's device
    # (`_to_copy`/`to`/`copy_` are dtype-only on this path — a `device=` kwarg
    # would be the D2H/H2D tell).
    calls = _dense_composite_trace(8)
    op_names = [name for name, _ in calls]
    assert calls, "dispatch trace is empty - the mode did not record"
    forbidden_fragments = ("nonzero", "_local_scalar_dense", "masked_select")
    for name in op_names:
        for fragment in forbidden_fragments:
            assert fragment not in name, f"sync-inducing op in dense path: {name}"
        assert not name.startswith("aten.item"), f"host readback in dense path: {name}"
    for name, kwargs in calls:
        assert kwargs.get("device") is None, (
            f"{name} was asked to change device ({kwargs['device']}) inside "
            "the dense composite - that is a cross-device copy"
        )
    # the single staged write into the caller's storage is present
    assert any(name.startswith("aten.copy_") for name in op_names)


def test_dense_composite_dispatch_count_is_n_independent() -> None:
    # Fixed-shape contract: the op SEQUENCE (not just the count) must be
    # identical for N=1/8/32 - nothing in the path branches on data or N.
    traces = {n: [name for name, _ in _dense_composite_trace(n)] for n in (1, 8, 32)}
    assert traces[1] == traces[8] == traces[32]
    assert len(traces[1]) < 40  # bounded: a handful of fixed ops per frame
