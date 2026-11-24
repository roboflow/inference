"""Label patches: linear planning work, bounded transfers, V1 pixels.

``paint_labels`` exchanges label-sized patches with the device. Its patch plan
is one walk over the label regions, with P extra pixels per extra patch
(P = 0 on CPU, ``_CUDA_PATCH_PIXELS`` on CUDA)::

    separate = summed region area + (R - 1) * P
    separate <= enclosing area   one patch per label (may overlap)
    separate >  enclosing area   one patch: the enclosing rectangle

The plan must never change pixels, so either plan is checked against the
other on CPU.

Three layouts on a 1920x1080 frame exercise it at 30, 300 and 1000 labels::

    grid     separate labels on a lattice     per-label patches
    random   seeded anchors; labels overlap   per-label patches
    pile     every label within a small area  one enclosing patch

Work is counted as executed lines of ``text.py`` (deterministic, unlike
time). Transfers are counted by spying on the one device-to-host and one
host-to-device copy. Pixels are compared with supervision 0.29 annotators,
the CPU library V1 draws labels with, on overlapping labels that cross every
frame border.
"""

import random
import sys
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pytest
import supervision as sv
import torch
from roboflow_workflows.v2.blocks.visualization.annotations import text
from roboflow_workflows.v2.blocks.visualization.annotations.anchors import (
    box_anchors,
)
from roboflow_workflows.v2.blocks.visualization.annotations.text import (
    Label,
    TextStyle,
    detection_labels,
    paint_labels,
    vertex_labels,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
FRAME_HW = (1080, 1920)
COUNTS = (30, 300, 1000)
LAYOUTS = ("grid", "random", "pile")
STYLE = TextStyle(scale=0.5, thickness=1, padding=4, border_radius=0)
Rect = Tuple[int, int, int, int]


def chw(pixels: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))

    return tensor


def hwc(tensor: torch.Tensor) -> np.ndarray:
    pixels = tensor.cpu().numpy().transpose(1, 2, 0)

    return pixels


def random_image(seed: int, height: int, width: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    pixels = generator.integers(0, 256, (height, width, 3), dtype=np.uint8)

    return pixels


def layout_labels(layout: str, count: int) -> List[Label]:
    """``count`` TOP_LEFT labels "car"; a label region is 35x25 pixels."""
    if layout == "grid":
        anchors = [(index % 48 * 40, 30 + index // 48 * 28) for index in range(count)]
    elif layout == "random":
        generator = random.Random(0)
        anchors = [
            (generator.randrange(0, 1860), generator.randrange(20, 1080))
            for _ in range(count)
        ]
    else:
        generator = random.Random(1)
        anchors = [
            (900 + generator.randrange(0, 20), 500 + generator.randrange(0, 10))
            for _ in range(count)
        ]
    labels = detection_labels(
        ["car"] * count,
        np.array(anchors),
        position="TOP_LEFT",
        style=STYLE,
        background_rgb=[(index % 256, 40, 200) for index in range(count)],
        text_rgb=(255, 255, 255),
    )

    return labels


def frame_regions(labels: Sequence[Label], *, size_hw: Tuple[int, int]) -> List[Rect]:
    height, width = size_hw
    clipped = [
        text._clipped(label.region(), height=height, width=width) for label in labels
    ]
    regions = [region for region in clipped if region is not None]

    return regions


def area(region: Rect) -> int:
    x0, y0, x1, y1 = region

    return (x1 - x0) * (y1 - y0)


def enclosing(regions: Sequence[Rect]) -> Rect:
    rectangle = (
        min(region[0] for region in regions),
        min(region[1] for region in regions),
        max(region[2] for region in regions),
        max(region[3] for region in regions),
    )

    return rectangle


def overlapping_pairs(regions: Sequence[Rect]) -> int:
    # Test-side check of the layouts only; the product never compares pairs.
    pairs = sum(
        first[0] < second[2]
        and second[0] < first[2]
        and first[1] < second[3]
        and second[1] < first[3]
        for index, first in enumerate(regions)
        for second in regions[index + 1 :]
    )

    return pairs


def text_lines_executed(function: Callable[[], None]) -> int:
    """Count executed lines of ``text.py`` while ``function`` runs."""
    executed = 0

    def trace(frame, event, arg):
        nonlocal executed
        if frame.f_code.co_filename != text.__file__:
            return None
        if event == "line":
            executed += 1

        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        function()
    finally:
        sys.settrace(previous)

    return executed


class TransferSpy:
    """Record the bytes of every ``Tensor.cpu`` and ``Tensor.to`` call."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.to_host: List[int] = []
        self.to_device: List[int] = []
        original_cpu, original_to = torch.Tensor.cpu, torch.Tensor.to

        def cpu(tensor, *args, **kwargs):
            self.to_host.append(tensor.nbytes)

            return original_cpu(tensor, *args, **kwargs)

        def to(tensor, *args, **kwargs):
            self.to_device.append(tensor.nbytes)

            return original_to(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "cpu", cpu)
        monkeypatch.setattr(torch.Tensor, "to", to)


# ---------------------------------------------------------------- work


@pytest.mark.parametrize("layout", LAYOUTS)
def test_patch_work_per_label_does_not_grow_with_label_count(layout: str) -> None:
    # given: the same layout at 30, 300 and 1000 labels
    lines_per_label: Dict[int, float] = {}

    for count in COUNTS:
        labels = layout_labels(layout, count)
        scene = torch.zeros((3, *FRAME_HW), dtype=torch.uint8)

        # when
        executed = text_lines_executed(lambda: paint_labels(scene, labels))

        lines_per_label[count] = executed / count

    # then: linear work. A pairwise merge runs R(R-1)/2 overlap checks on
    # separate labels; r3 ran 158 lines per label at 30 grid labels, 3094 at 1000.
    assert lines_per_label[1000] <= lines_per_label[300] <= lines_per_label[30]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("count", COUNTS)
def test_one_transfer_each_way_moves_only_the_planned_patches(
    layout: str, count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    labels = layout_labels(layout, count)
    regions = frame_regions(labels, size_hw=FRAME_HW)
    region_bytes = 3 * sum(area(region) for region in regions)
    enclosing_bytes = 3 * area(enclosing(regions))
    scene = torch.zeros((3, *FRAME_HW), dtype=torch.uint8)
    spy = TransferSpy(monkeypatch)

    # when
    paint_labels(scene, labels)

    # then: one copy each way, the same bytes, never above either bound
    assert len(spy.to_host) == 1 and len(spy.to_device) == 1
    assert spy.to_host == spy.to_device
    assert spy.to_host[0] == min(region_bytes, enclosing_bytes)
    if layout == "pile":
        assert spy.to_host[0] == enclosing_bytes < region_bytes
    else:
        assert spy.to_host[0] == region_bytes <= enclosing_bytes
    assert spy.to_host[0] < 3 * FRAME_HW[0] * FRAME_HW[1]


def test_layouts_have_the_overlap_they_are_named_for() -> None:
    grid = frame_regions(layout_labels("grid", 300), size_hw=FRAME_HW)
    scattered = frame_regions(layout_labels("random", 300), size_hw=FRAME_HW)
    pile = frame_regions(layout_labels("pile", 30), size_hw=FRAME_HW)

    assert overlapping_pairs(grid) == 0
    assert overlapping_pairs(scattered) > 0
    assert overlapping_pairs(pile) == 30 * 29 // 2


# ---------------------------------------------------------------- plan cost


def test_patch_cost_sends_enclosing_rectangle_for_many_small_patches() -> None:
    # given: two separate 1x1 regions; their enclosing row is P + 2 pixels long
    # (separate costs exactly as much) or P + 1 pixels long (separate costs more)
    cost = text._CUDA_PATCH_PIXELS
    tie = [(0, 0, 1, 1), (cost + 1, 0, cost + 2, 1)]
    dearer = [(0, 0, 1, 1), (cost, 0, cost + 1, 1)]

    # when
    plans = {
        "cpu tie": text._patch_plan(tie, patch_pixels=0),
        "cpu dearer": text._patch_plan(dearer, patch_pixels=0),
        "cuda tie": text._patch_plan(tie, patch_pixels=cost),
        "cuda dearer": text._patch_plan(dearer, patch_pixels=cost),
        "cuda one": text._patch_plan(tie[:1], patch_pixels=cost),
    }

    # then: CPU keeps small separate patches; CUDA switches strictly above the tie
    assert area(enclosing(tie)) == 2 + cost
    assert plans["cpu tie"] == tie
    assert plans["cpu dearer"] == dearer
    assert plans["cuda tie"] == tie
    assert plans["cuda dearer"] == [enclosing(dearer)]
    assert plans["cuda one"] == tie[:1]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("count", COUNTS)
def test_patch_plan_choice_does_not_change_pixels(
    layout: str, count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given: the CPU plan, and the other plan forced. Pile labels overlap, so
    # forcing separate patches also checks overlapping write-back.
    labels = layout_labels(layout, count)
    pixels = random_image(count, *FRAME_HW)
    regions = frame_regions(labels, size_hw=FRAME_HW)
    planned = text._patch_plan(regions, patch_pixels=0)
    forced = regions if planned == [enclosing(regions)] else [enclosing(regions)]
    assert forced != planned
    expected = chw(pixels)
    paint_labels(expected, labels)
    scene = chw(pixels)
    monkeypatch.setattr(text, "_patch_plan", lambda regions, **_: forced)

    # when
    paint_labels(scene, labels)

    # then
    assert torch.equal(scene, expected)
    assert not torch.equal(scene, chw(pixels))


# ---------------------------------------------------------------- pixels


def sv_detection_labels(
    pixels: np.ndarray, xyxy: np.ndarray, texts: Sequence[str], *, style: TextStyle
) -> np.ndarray:
    annotator = sv.LabelAnnotator(
        color=sv.ColorPalette([sv.Color(*background_rgb(i)) for i in range(len(xyxy))]),
        color_lookup=sv.ColorLookup.INDEX,
        text_color=sv.Color.WHITE,
        text_scale=style.scale,
        text_thickness=style.thickness,
        text_padding=style.padding,
        text_position=sv.Position.TOP_LEFT,
        border_radius=style.border_radius,
    )
    detections = sv.Detections(xyxy=xyxy, class_id=np.zeros(len(xyxy), dtype=int))
    annotated = annotator.annotate(
        pixels[:, :, ::-1].copy(), detections, labels=list(texts)
    )

    return annotated[:, :, ::-1]


def background_rgb(index: int) -> Tuple[int, int, int]:
    color = (37 * index % 256, 90 + 53 * index % 166, 200 - 29 * index % 200)

    return color


def v2_detection_labels(
    pixels: np.ndarray, xyxy: np.ndarray, texts: Sequence[str], *, style: TextStyle
) -> np.ndarray:
    labels = detection_labels(
        texts,
        box_anchors(xyxy, "TOP_LEFT"),
        position="TOP_LEFT",
        style=style,
        background_rgb=[background_rgb(i) for i in range(len(xyxy))],
        text_rgb=(255, 255, 255),
    )
    scene = chw(pixels)
    paint_labels(scene, labels)

    return hwc(scene)


def crossing_borders(regions: Sequence[Rect], *, size_hw: Tuple[int, int]) -> set:
    height, width = size_hw
    borders = set()
    for x0, y0, x1, y1 in regions:
        borders |= {"left"} if x0 < 0 else set()
        borders |= {"top"} if y0 < 0 else set()
        borders |= {"right"} if x1 > width else set()
        borders |= {"bottom"} if y1 > height else set()

    return borders


# Box corners; TOP_LEFT labels sit above each corner. A diagonal chain of
# partly overlapping labels crosses all four borders; a pile of labels on top
# of each other crosses the right border.
CHAIN = [(-20 + 20 * i, 4 + 13 * i) for i in range(10)]
PILE = [(150 + 3 * (i % 4), 60 + 2 * i) for i in range(10)]
OVERLAP_CASES = {"chain": (CHAIN, 1), "pile": (PILE, 0)}


@pytest.mark.parametrize("case", sorted(OVERLAP_CASES))
def test_overlapping_labels_across_borders_equal_sv_label_annotator(
    case: str,
) -> None:
    # given: overlapping labels whose regions cross frame borders
    corners, seed = OVERLAP_CASES[case]
    size_hw = (120, 170)
    pixels = random_image(seed, *size_hw)
    xyxy = np.array([(x, y, x + 30, y + 20) for x, y in corners], dtype=np.float32)
    texts = [f"object {index}" for index in range(len(corners))]
    style = TextStyle(scale=0.5, thickness=1, padding=3, border_radius=4)
    labels = detection_labels(
        texts,
        box_anchors(xyxy, "TOP_LEFT"),
        position="TOP_LEFT",
        style=style,
        background_rgb=[(0, 0, 0)] * len(texts),
        text_rgb=(255, 255, 255),
    )
    unclipped = [label.region() for label in labels]
    regions = frame_regions(labels, size_hw=size_hw)
    plan = text._patch_plan(regions, patch_pixels=0)

    # when
    actual = v2_detection_labels(pixels, xyxy, texts, style=style)

    # then: the case reaches its plan branch, and pixels equal V1's
    assert overlapping_pairs(regions) >= len(corners) - 1
    if case == "chain":
        assert plan == regions
        assert crossing_borders(unclipped, size_hw=size_hw) == {
            "left",
            "top",
            "right",
            "bottom",
        }
    else:
        assert plan == [enclosing(regions)]
        assert "right" in crossing_borders(unclipped, size_hw=size_hw)
    expected = sv_detection_labels(pixels, xyxy, texts, style=style)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("seed", range(24))
def test_random_dense_labels_equal_sv_label_annotator(seed: int) -> None:
    # given: 8-40 labels on a small frame, so most overlap and many are clipped
    generator = np.random.default_rng(500 + seed)
    height, width = int(generator.integers(40, 120)), int(generator.integers(50, 160))
    pixels = random_image(seed, height, width)
    count = int(generator.integers(8, 41))
    corner = generator.uniform(-20, [width + 10, height + 10], (count, 2))
    xyxy = np.concatenate([corner, corner + 10], axis=1).astype(np.float32)
    texts = [["a", "car 0.9", "Tg", "a\nbc"][i % 4] for i in range(count)]
    style = TextStyle(
        scale=float(generator.choice([0.3, 0.5])),
        thickness=int(generator.integers(1, 3)),
        padding=int(generator.integers(0, 5)),
        border_radius=int(generator.integers(0, 6)),
    )

    # when
    actual = v2_detection_labels(pixels, xyxy, texts, style=style)

    # then
    expected = sv_detection_labels(pixels, xyxy, texts, style=style)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("spread", [4, 40])
def test_overlapping_vertex_labels_equal_sv_vertex_label_annotator(spread: int) -> None:
    # given: 12 vertex labels within `spread` pixels (pile) or along a line
    pixels = random_image(spread, 50, 80)
    points = np.array(
        [(2 + 6 * i % 78, 1 + (i * spread) % 52) for i in range(12)], dtype=np.int64
    )
    style = TextStyle(scale=0.4, thickness=1, padding=3, border_radius=3)
    texts = [str(index) for index in range(12)]
    labels = vertex_labels(
        texts,
        [(int(x), int(y)) for x, y in points],
        style=style,
        background_rgb=(10, 200, 30),
        text_rgb=(0, 0, 0),
    )
    scene = chw(pixels)
    assert overlapping_pairs(frame_regions(labels, size_hw=(50, 80))) > 0

    # when
    paint_labels(scene, labels)

    # then
    annotator = sv.VertexLabelAnnotator(
        color=sv.Color(10, 200, 30),
        text_color=sv.Color.BLACK,
        text_scale=style.scale,
        text_thickness=1,
        text_padding=style.padding,
        border_radius=style.border_radius,
    )
    key_points = sv.KeyPoints(xy=points[None].astype(np.float32))
    expected = annotator.annotate(pixels[:, :, ::-1].copy(), key_points, labels=texts)
    np.testing.assert_array_equal(hwc(scene), expected[:, :, ::-1])


# ---------------------------------------------------------------- CUDA (authored, unrun locally)


@CUDA
@pytest.mark.parametrize("layout", LAYOUTS)
def test_cuda_labels_paint_like_cpu_with_one_transfer_each_way(
    layout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given: at 1000 labels the CUDA plan is one enclosing patch; for grid and
    # random it differs from the CPU plan that paints the expected pixels
    labels = layout_labels(layout, 1000)
    regions = frame_regions(labels, size_hw=FRAME_HW)
    cpu_plan = text._patch_plan(regions, patch_pixels=0)
    cuda_plan = text._patch_plan(regions, patch_pixels=text._CUDA_PATCH_PIXELS)
    assert cuda_plan == [enclosing(regions)]
    assert (cuda_plan == cpu_plan) == (layout == "pile")
    pixels = random_image(7, *FRAME_HW)
    expected = chw(pixels)
    paint_labels(expected, labels)
    scene = chw(pixels).cuda()
    spy = TransferSpy(monkeypatch)

    # when
    paint_labels(scene, labels)
    torch.cuda.synchronize()

    # then: no readback per label, only the CUDA plan's bytes; the CPU result
    assert len(spy.to_host) == 1 and len(spy.to_device) == 1
    assert spy.to_host == spy.to_device
    assert spy.to_host[0] == 3 * sum(area(patch) for patch in cuda_plan)
    assert spy.to_host[0] <= 3 * FRAME_HW[0] * FRAME_HW[1]
    monkeypatch.undo()
    assert torch.equal(scene.cpu(), expected)
