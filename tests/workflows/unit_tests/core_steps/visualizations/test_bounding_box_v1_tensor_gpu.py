import itertools

import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.workflows.core_steps.visualizations.bounding_box.v1_tensor import (
    _get_geometry,
    _gpu_box_draw_eligible,
    gpu_draw_boxes,
)
from inference_models.models.base.object_detection import Detections

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

PALETTE = sv.ColorPalette.DEFAULT
SCENE_H, SCENE_W = 240, 320


def _build_detections(boxes: np.ndarray, class_id: np.ndarray, device: str) -> Detections:
    n = boxes.shape[0]
    return Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32, device=device),
        class_id=torch.tensor(class_id, dtype=torch.int32, device=device),
        confidence=torch.full((n,), 0.9, device=device),
        image_metadata={"class_names": {i: f"c{i}" for i in range(10)}},
    )


def _assert_parity(
    xyxy: np.ndarray,
    class_id: np.ndarray,
    thickness: int,
    color_axis: str,
    device: str,
) -> None:
    rng = np.random.default_rng(7)
    base_bgr = rng.integers(0, 256, (SCENE_H, SCENE_W, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=xyxy.astype(np.float32), class_id=class_id.astype(int)
    )
    annotator = sv.BoxAnnotator(
        color=PALETTE,
        color_lookup=getattr(sv.ColorLookup, color_axis),
        thickness=thickness,
    )
    reference = annotator.annotate(scene=base_bgr.copy(), detections=detections)

    scene_chw_rgb = (
        torch.from_numpy(base_bgr[:, :, ::-1].copy())
        .permute(2, 0, 1)
        .contiguous()
        .to(device)
    )
    ids = class_id if color_axis == "CLASS" else np.arange(xyxy.shape[0])
    colors_rgb = np.asarray(
        [PALETTE.by_idx(int(idx)).as_rgb() for idx in ids], dtype=np.uint8
    )
    annotated = gpu_draw_boxes(
        scene_chw_rgb,
        xyxy.astype(np.float32).astype(int),
        colors_rgb,
        thickness,
    )
    got_bgr = annotated.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]

    exact = np.all(got_bgr == reference, axis=-1).mean()
    assert exact == 1.0, f"exact match rate {exact}"
    assert int(np.abs(got_bgr.astype(int) - reference.astype(int)).max()) == 0


_SCENARIOS = {
    "plain": np.array([[30, 40, 120, 100], [150, 60, 290, 200]], dtype=float),
    "overlap_chain": np.array(
        [[30, 40, 160, 160], [80, 90, 200, 210], [100, 20, 140, 230]], dtype=float
    ),
    "off_image": np.array(
        [[-20, -30, 60, 50], [250, 180, 400, 400], [-50, 100, 380, 140]], dtype=float
    ),
    "tiny_and_degenerate": np.array(
        [[10, 10, 12, 12], [50, 50, 50, 60], [70, 70, 70, 70], [90, 5, 95, 9]],
        dtype=float,
    ),
    "subpixel_coords": np.array(
        [[30.7, 40.2, 120.9, 100.5], [-0.4, -0.6, 33.3, 44.9]], dtype=float
    ),
    "edge_touching": np.array(
        [[0, 0, SCENE_W - 1, SCENE_H - 1], [5, 5, 15, SCENE_H - 10]], dtype=float
    ),
}


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
@pytest.mark.parametrize("thickness", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("color_axis", ["CLASS", "INDEX"])
def test_gpu_box_parity_on_cpu(scenario: str, thickness: int, color_axis: str) -> None:
    # The painter is device-agnostic; CPU run gives full parity coverage in
    # plain CI (identical kernels run on CUDA).
    xyxy = _SCENARIOS[scenario]
    class_id = np.arange(xyxy.shape[0]) % 3
    _assert_parity(xyxy, class_id, thickness, color_axis, device="cpu")


@requires_cuda
@pytest.mark.parametrize("thickness", [1, 2, 4])
def test_gpu_box_parity_on_cuda(thickness: int) -> None:
    rng = np.random.default_rng(11)
    xs = np.sort(rng.uniform(-30, SCENE_W + 30, (12, 2)), axis=1)
    ys = np.sort(rng.uniform(-30, SCENE_H + 30, (12, 2)), axis=1)
    xyxy = np.stack([xs[:, 0], ys[:, 0], xs[:, 1], ys[:, 1]], axis=1)
    class_id = np.arange(12) % 4
    _assert_parity(xyxy, class_id, thickness, "CLASS", device="cuda")


def test_gpu_box_paint_is_in_place() -> None:
    scene = torch.zeros((3, 64, 64), dtype=torch.uint8)
    annotated = gpu_draw_boxes(
        scene,
        np.array([[10, 10, 40, 40]], dtype=int),
        np.array([[255, 0, 0]], dtype=np.uint8),
        2,
    )
    assert annotated.data_ptr() == scene.data_ptr()


@pytest.mark.parametrize(
    "thickness", [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
)
def test_border_geometry_reconstruction_holds(thickness: int) -> None:
    # _BorderGeometry asserts pixel-perfect reconstruction against
    # cv2.rectangle on two reference sizes at construction time.
    geometry = _get_geometry(thickness)
    assert geometry.corner_masks.shape[0] == 4


def _eligible_detections(device: str = "cpu") -> Detections:
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    return _build_detections(boxes, np.array([1]), device=device)


def test_gpu_box_draw_eligible_happy_path() -> None:
    assert _gpu_box_draw_eligible(_eligible_detections(), "CLASS", 0.0, 2) is True


def test_gpu_box_draw_not_eligible_for_roundness() -> None:
    assert _gpu_box_draw_eligible(_eligible_detections(), "CLASS", 0.4, 2) is False


def test_gpu_box_draw_not_eligible_for_track_lookup() -> None:
    assert _gpu_box_draw_eligible(_eligible_detections(), "TRACK", 0.0, 2) is False


def test_gpu_box_draw_not_eligible_for_empty_detections() -> None:
    empty = _build_detections(
        np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=int), device="cpu"
    )
    assert _gpu_box_draw_eligible(empty, "CLASS", 0.0, 2) is False


def test_gpu_box_draw_not_eligible_for_non_int_thickness() -> None:
    assert _gpu_box_draw_eligible(_eligible_detections(), "CLASS", 0.0, "2") is False
