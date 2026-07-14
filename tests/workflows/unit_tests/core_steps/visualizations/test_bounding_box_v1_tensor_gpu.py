import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.workflows.core_steps.visualizations.bounding_box.v1_tensor import (
    _gpu_box_draw_eligible,
    gpu_draw_boxes,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference_models.models.base.object_detection import Detections

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

PALETTE = sv.ColorPalette.DEFAULT
SCENE_H, SCENE_W = 240, 320


def _paint(
    xyxy: np.ndarray, colors_rgb: np.ndarray, thickness: int, device: str = "cpu"
) -> tuple:
    """Run the painter on a zero scene; return (chw tensor, painted-pixel mask)."""
    scene = torch.zeros((3, SCENE_H, SCENE_W), dtype=torch.uint8, device=device)
    annotated = gpu_draw_boxes(scene, xyxy.astype(int), colors_rgb, thickness)
    painted = (annotated != 0).any(dim=0).cpu().numpy()
    return annotated, painted


def _sv_painted_mask(xyxy: np.ndarray, thickness: int) -> np.ndarray:
    scene = np.zeros((SCENE_H, SCENE_W, 3), dtype=np.uint8)
    annotator = sv.BoxAnnotator(
        color=PALETTE, color_lookup=sv.ColorLookup.INDEX, thickness=thickness
    )
    out = annotator.annotate(
        scene, sv.Detections(xyxy=xyxy.astype(np.float32))
    )
    return (out != 0).any(axis=2)


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
    "edge_touching": np.array(
        [[0, 0, SCENE_W - 1, SCENE_H - 1], [5, 5, 15, SCENE_H - 10]], dtype=float
    ),
}


def _index_colors(n: int) -> np.ndarray:
    return np.asarray(
        [PALETTE.by_idx(i).as_rgb() for i in range(n)], dtype=np.uint8
    )


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
@pytest.mark.parametrize("thickness", [1, 2, 3, 5, 8])
def test_gpu_boxes_visually_match_sv(scenario: str, thickness: int) -> None:
    # Approximate renderer: borders are square-cornered `thickness`-wide
    # bands, cv2 draws round joins / slightly wider bands. Painted regions
    # must still substantially agree (IoU), and t=1 is pixel-identical.
    xyxy = _SCENARIOS[scenario]
    _, painted = _paint(xyxy, _index_colors(xyxy.shape[0]), thickness)
    reference = _sv_painted_mask(xyxy, thickness)
    union = (painted | reference).sum()
    if union == 0:
        return
    iou = (painted & reference).sum() / union
    threshold = 1.0 if thickness == 1 else 0.5
    assert iou >= threshold, f"painted-region IoU {iou:.3f}"


def test_border_band_geometry_exact() -> None:
    # One box, thickness 3: band spans [edge - 1, edge + 1] (outer = t // 2),
    # interior and background stay untouched.
    x1, y1, x2, y2 = 50, 60, 150, 140
    annotated, painted = _paint(
        np.array([[x1, y1, x2, y2]], dtype=float),
        np.array([[255, 0, 0]], np.uint8),
        thickness=3,
    )
    assert painted[y1 - 1 : y1 + 2, x1 - 1 : x2 + 2].all()  # top band
    assert painted[y2 - 1 : y2 + 2, x1 - 1 : x2 + 2].all()  # bottom band
    assert painted[y1 - 1 : y2 + 2, x1 - 1 : x1 + 2].all()  # left band
    assert painted[y1 - 1 : y2 + 2, x2 - 1 : x2 + 2].all()  # right band
    assert not painted[y1 + 2 : y2 - 1, x1 + 2 : x2 - 1].any()  # interior clean
    assert not painted[: y1 - 1].any() and not painted[y2 + 2 :].any()  # outside
    assert (annotated[0][painted] == 255).all()  # right channel, right color


def test_overlapping_boxes_later_box_wins() -> None:
    # sv paints sequentially: the higher detection index owns contested pixels.
    xyxy = np.array([[20, 20, 100, 100], [60, 20, 140, 100]], dtype=float)
    colors = np.array([[255, 0, 0], [0, 255, 0]], np.uint8)
    annotated, _ = _paint(xyxy, colors, thickness=2)
    # box 1's left band crosses box 0's interior row: contested column region
    # around x=60 on box 0's top band must be box 1's color where box 1 paints.
    top = annotated[:, 20, 60].cpu().numpy()  # (3,) at shared corner pixel
    assert tuple(top) == (0, 255, 0)


def test_fully_off_frame_box_is_noop() -> None:
    annotated, painted = _paint(
        np.array([[SCENE_W + 10, SCENE_H + 10, SCENE_W + 50, SCENE_H + 50]], float),
        np.array([[255, 0, 0]], np.uint8),
        thickness=2,
    )
    assert not painted.any()


def test_gpu_box_paint_is_in_place() -> None:
    scene = torch.zeros((3, 64, 64), dtype=torch.uint8)
    annotated = gpu_draw_boxes(
        scene,
        np.array([[10, 10, 40, 40]], dtype=int),
        np.array([[255, 0, 0]], dtype=np.uint8),
        2,
    )
    assert annotated.data_ptr() == scene.data_ptr()


def test_non_contiguous_scene_raises() -> None:
    # .view must fail (caught by the block's sv fallback) rather than write
    # into a silent copy. Transposed spatial dims are not viewable as (3, -1).
    scene = torch.zeros((64, 64, 3), dtype=torch.uint8).permute(2, 1, 0)
    with pytest.raises(RuntimeError):
        gpu_draw_boxes(
            scene,
            np.array([[10, 10, 40, 40]], dtype=int),
            np.array([[255, 0, 0]], dtype=np.uint8),
            2,
        )


@requires_cuda
def test_gpu_boxes_on_cuda_match_cpu() -> None:
    xyxy = _SCENARIOS["overlap_chain"]
    colors = _index_colors(xyxy.shape[0])
    cpu_out, _ = _paint(xyxy, colors, thickness=2, device="cpu")
    cuda_out, _ = _paint(xyxy, colors, thickness=2, device="cuda")
    assert np.array_equal(cpu_out.numpy(), cuda_out.cpu().numpy())


def _build_detections(boxes: np.ndarray, class_id: np.ndarray, device: str) -> Detections:
    n = boxes.shape[0]
    return Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32, device=device),
        class_id=torch.tensor(class_id, dtype=torch.int32, device=device),
        confidence=torch.full((n,), 0.9, device=device),
        image_metadata={"class_names": {i: f"c{i}" for i in range(10)}},
    )


def _eligible_detections(device: str = "cpu") -> Detections:
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    return _build_detections(boxes, np.array([1]), device=device)


def _tensor_backed_image() -> WorkflowImageData:
    tensor = torch.zeros((3, 64, 64), dtype=torch.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"), tensor_image=tensor
    )


def test_gpu_box_draw_eligible_happy_path() -> None:
    assert (
        _gpu_box_draw_eligible(
            _eligible_detections(), "CLASS", 0.0, 2, _tensor_backed_image()
        )
        is True
    )


def test_gpu_box_draw_not_eligible_for_roundness() -> None:
    assert (
        _gpu_box_draw_eligible(
            _eligible_detections(), "CLASS", 0.4, 2, _tensor_backed_image()
        )
        is False
    )


def test_gpu_box_draw_not_eligible_for_track_lookup() -> None:
    assert (
        _gpu_box_draw_eligible(
            _eligible_detections(), "TRACK", 0.0, 2, _tensor_backed_image()
        )
        is False
    )


def test_gpu_box_draw_not_eligible_for_empty_detections() -> None:
    empty = _build_detections(
        np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=int), device="cpu"
    )
    assert (
        _gpu_box_draw_eligible(empty, "CLASS", 0.0, 2, _tensor_backed_image())
        is False
    )


def test_gpu_box_draw_not_eligible_for_non_int_thickness() -> None:
    assert (
        _gpu_box_draw_eligible(
            _eligible_detections(), "CLASS", 0.0, "2", _tensor_backed_image()
        )
        is False
    )


def test_gpu_box_draw_not_eligible_for_numpy_sourced_image() -> None:
    numpy_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"),
        numpy_image=np.zeros((64, 64, 3), dtype=np.uint8),
    )
    assert (
        _gpu_box_draw_eligible(_eligible_detections(), "CLASS", 0.0, 2, numpy_image)
        is False
    )
