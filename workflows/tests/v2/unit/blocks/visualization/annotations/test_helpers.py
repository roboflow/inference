"""Drawing helpers of the annotation blocks against OpenCV and supervision.

Oracles are the CPU libraries V1 draws with: ``cv2`` for strokes and raster
operations, supervision 0.29 annotators for labels, masks and anchors. V2
images are RGB ``(3, H, W)``; supervision draws BGR, so oracles run on the
channel-reversed image and compare after reversing back.

=====================  ===================================  ====================
helper                 oracle                               equality
=====================  ===================================  ====================
Strokes                cv2 drawing on the frame, in order   exact
paint_labels           sv.LabelAnnotator / VertexLabel...   exact
box_blur               cv2.blur                             exact
grow_mask              cv2.dilate, elliptic kernel          exact
resize_nearest         cv2.resize INTER_NEAREST             exact
resize_linear          cv2.resize INTER_LINEAR              <= 1 level; exact on
                                                            integer-valued cases
box_anchors            sv.Detections.get_anchors_coord...   exact
mass_centres           same, CENTER_OF_MASS                 exact
paint_instances+blend  sv.MaskAnnotator / cv2.addWeighted   exact
=====================  ===================================  ====================

Random cases are bounded, seeded samples of small images.
"""

from typing import List, Sequence, Tuple

import cv2
import numpy as np
import pytest
import supervision as sv
import torch
from roboflow_workflows.v2.blocks.visualization.annotations.anchors import (
    box_anchors,
    mass_centres,
)
from roboflow_workflows.v2.blocks.visualization.annotations.blend import (
    blend,
    paint_instances,
)
from roboflow_workflows.v2.blocks.visualization.annotations.raster import (
    box_blur,
    grow_mask,
    resize_linear,
    resize_nearest,
)
from roboflow_workflows.v2.blocks.visualization.annotations.strokes import (
    Strokes,
    bounds_scan,
    rectangle_scan,
    stroke_reach,
)
from roboflow_workflows.v2.blocks.visualization.annotations.text import (
    POSITIONS,
    Label,
    TextStyle,
    detection_labels,
    paint_labels,
    vertex_labels,
)

MPS = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="local MPS device not available"
)
BOX_POSITIONS = list(POSITIONS)
LABEL_TEXTS = ["car", "person 0.50", "", "a\nbc", "Tg", "x" * 12, "two\n\nlines"]


def chw(pixels: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))

    return tensor


def hwc(tensor: torch.Tensor) -> np.ndarray:
    pixels = tensor.cpu().numpy().transpose(1, 2, 0)

    return pixels


def random_image(generator: np.random.Generator, height: int, width: int) -> np.ndarray:
    pixels = generator.integers(0, 256, (height, width, 3), dtype=np.uint8)

    return pixels


def max_level_difference(first: np.ndarray, second: np.ndarray) -> int:
    difference = int(np.abs(first.astype(np.int64) - second.astype(np.int64)).max())

    return difference


# ---------------------------------------------------------------- strokes


def test_rectangle_outline_hand_case() -> None:
    strokes = Strokes((6, 6))

    pixels = strokes.add(
        lambda mask: cv2.rectangle(mask, (1, 1), (4, 3), 1, 1),
        color=(9, 8, 7),
        scan=rectangle_scan((1, 1, 4, 3), reach=stroke_reach(1)),
    )
    scene = torch.zeros((3, 6, 6), dtype=torch.uint8)
    strokes.paint(scene)

    # Rows 1 and 3 from x=1..4, plus x=1 and x=4 on row 2 (flat y * 6 + x).
    np.testing.assert_array_equal(pixels, [7, 8, 9, 10, 13, 16, 19, 20, 21, 22])
    painted = scene.view(3, -1)[:, pixels]
    assert (painted == torch.tensor([[9], [8], [7]], dtype=torch.uint8)).all()
    assert int((scene.sum(dim=0) > 0).sum()) == len(pixels)


def test_strokes_clip_at_the_frame_edge() -> None:
    strokes = Strokes((6, 6))

    inside = strokes.add(
        lambda mask: cv2.line(mask, (-5, 2), (3, 2), 1, 1),
        color=(1, 1, 1),
        scan=bounds_scan(np.array([[-5, 2], [3, 2]]), reach=stroke_reach(1)),
    )
    outside = strokes.add(
        lambda mask: cv2.line(mask, (-9, -9), (-2, -9), 1, 3),
        color=(2, 2, 2),
        scan=bounds_scan(np.array([[-9, -9], [-2, -9]]), reach=stroke_reach(3)),
    )

    np.testing.assert_array_equal(inside, [12, 13, 14, 15])
    assert outside.size == 0 and len(strokes) == 2


def test_later_strokes_cover_earlier_ones() -> None:
    def filled(corner_a: Tuple[int, int], corner_b: Tuple[int, int]):
        return lambda mask: cv2.rectangle(mask, corner_a, corner_b, 1, -1)

    for first, second in [((255, 0, 0), (0, 0, 255)), ((0, 0, 255), (255, 0, 0))]:
        strokes = Strokes((5, 5))
        strokes.add(filled((0, 0), (2, 2)), color=first, scan=[(0, 0, 5, 5)])
        strokes.add(filled((2, 2), (4, 4)), color=second, scan=[(0, 0, 5, 5)])
        scene = torch.zeros((3, 5, 5), dtype=torch.uint8)
        strokes.paint(scene)

        assert tuple(scene[:, 2, 2].tolist()) == second
        assert tuple(scene[:, 1, 1].tolist()) == first


def test_empty_strokes_leave_the_scene_unchanged() -> None:
    scene = torch.arange(3 * 4 * 5, dtype=torch.uint8).reshape(3, 4, 5)
    before = scene.clone()
    strokes = Strokes((4, 5))

    strokes.paint(scene)
    no_scan = strokes.add(lambda mask: None, color=(1, 2, 3), scan=[])
    off_frame = strokes.add(lambda mask: None, color=(1, 2, 3), scan=[(9, 9, 12, 12)])
    strokes.paint(scene)
    strokes.upload(scene.device).paint_one(scene, 1)
    Strokes((4, 5)).upload(scene.device)

    assert no_scan.size == 0 and off_frame.size == 0
    assert torch.equal(scene, before)
    assert bounds_scan(np.zeros((0, 2), dtype=np.int64), reach=3) == []


def _random_stroke(
    generator: np.random.Generator, height: int, width: int
) -> Tuple[object, List[Tuple[int, int, int, int]]]:
    """One stroke as the blocks build it: (cv2 draw function, scan)."""
    kind = int(generator.integers(0, 5))
    thickness = int(generator.integers(1, 8))
    reach = stroke_reach(thickness)
    a = (
        int(generator.integers(-8, width + 8)),
        int(generator.integers(-8, height + 8)),
    )
    b = (
        int(generator.integers(-8, width + 8)),
        int(generator.integers(-8, height + 8)),
    )
    if kind == 0:  # bounding box / crop outline
        return (
            lambda mask: cv2.rectangle(mask, a, b, 1, thickness),
            rectangle_scan((*a, *b), reach=reach),
        )
    if kind == 1:  # keypoint edge
        return (
            lambda mask: cv2.line(mask, a, b, 1, thickness),
            bounds_scan(np.array([a, b]), reach=reach),
        )
    if kind == 2:  # keypoint vertex
        radius = int(generator.integers(1, 9))
        return (
            lambda mask: cv2.circle(mask, a, radius, 1, -1),
            bounds_scan(np.array([a]), reach=radius + 2),
        )
    if kind == 3:  # rounded corner arc
        radius = int(generator.integers(1, 9))
        return (
            lambda mask: cv2.ellipse(
                mask, a, (radius, radius), 180, 0, 90, 1, thickness
            ),
            bounds_scan(np.array([a]), reach=radius + reach),
        )
    points = np.stack(
        [generator.integers(-8, width + 8, 4), generator.integers(-8, height + 8, 4)], 1
    ).astype(np.int32)
    return (  # polygon contour
        lambda mask: cv2.polylines(mask, [points], True, 1, thickness),
        bounds_scan(points, reach=reach),
    )


@pytest.mark.parametrize("seed", range(30))
def test_random_strokes_equal_direct_cv2_drawing(seed: int) -> None:
    generator = np.random.default_rng(seed)
    height, width = int(generator.integers(1, 40)), int(generator.integers(1, 40))
    pixels = random_image(generator, height, width)
    expected = pixels.copy()
    strokes = Strokes((height, width))

    for _ in range(int(generator.integers(1, 7))):
        draw, scan = _random_stroke(generator, height, width)
        color = tuple(int(value) for value in generator.integers(0, 256, 3))
        found = strokes.add(draw, color=color, scan=scan)
        alone = np.zeros((height, width), dtype=np.uint8)
        draw(alone)
        expected[alone > 0] = color
        np.testing.assert_array_equal(found, np.flatnonzero(alone))

    painted = chw(pixels)
    strokes.paint(painted)
    one_by_one = chw(pixels)
    uploaded = strokes.upload(one_by_one.device)
    for index in range(len(strokes)):
        uploaded.paint_one(one_by_one, index)

    np.testing.assert_array_equal(hwc(painted), expected)
    assert torch.equal(painted, one_by_one)


# ---------------------------------------------------------------- labels


def sv_detection_labels(
    pixels: np.ndarray,
    xyxy: np.ndarray,
    texts: Sequence[str],
    *,
    position: str,
    style: TextStyle,
    backgrounds: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    annotator = sv.LabelAnnotator(
        color=sv.ColorPalette([sv.Color(*rgb) for rgb in backgrounds]),
        color_lookup=sv.ColorLookup.INDEX,
        text_color=sv.Color.WHITE,
        text_scale=style.scale,
        text_thickness=style.thickness,
        text_padding=style.padding,
        text_position=sv.Position[position],
        border_radius=style.border_radius,
    )
    detections = sv.Detections(xyxy=xyxy, class_id=np.zeros(len(xyxy), dtype=int))
    annotated = annotator.annotate(
        pixels[:, :, ::-1].copy(), detections, labels=list(texts)
    )

    return annotated[:, :, ::-1]


def v2_labels(pixels: np.ndarray, labels: Sequence[Label]) -> np.ndarray:
    scene = chw(pixels)
    paint_labels(scene, labels)

    return hwc(scene)


@pytest.mark.parametrize("seed", range(40))
def test_random_detection_labels_equal_sv_label_annotator(seed: int) -> None:
    generator = np.random.default_rng(seed)
    height, width = int(generator.integers(20, 70)), int(generator.integers(20, 90))
    pixels = random_image(generator, height, width)
    count = int(generator.integers(1, 5))
    corner = generator.uniform(-15, [width, height], (count, 2))
    sizes = generator.uniform(1, 30, (count, 2))
    xyxy = np.concatenate([corner, corner + sizes], axis=1).astype(np.float32)
    texts = [LABEL_TEXTS[i] for i in generator.integers(0, len(LABEL_TEXTS), count)]
    backgrounds = [tuple(int(v) for v in generator.integers(0, 256, 3)) for _ in texts]
    position = BOX_POSITIONS[seed % len(BOX_POSITIONS)]
    style = TextStyle(
        scale=float(generator.choice([0.3, 0.5, 0.8])),
        thickness=int(generator.integers(1, 3)),
        padding=int(generator.integers(0, 6)),
        border_radius=int(generator.integers(0, 8)),
    )

    labels = detection_labels(
        texts,
        box_anchors(xyxy, position),
        position=position,
        style=style,
        background_rgb=backgrounds,
        text_rgb=(255, 255, 255),
    )

    expected = sv_detection_labels(
        pixels, xyxy, texts, position=position, style=style, backgrounds=backgrounds
    )
    np.testing.assert_array_equal(v2_labels(pixels, labels), expected)


@pytest.mark.parametrize("seed", range(20))
def test_random_vertex_labels_equal_sv_vertex_label_annotator(seed: int) -> None:
    generator = np.random.default_rng(100 + seed)
    height, width = int(generator.integers(20, 70)), int(generator.integers(20, 90))
    pixels = random_image(generator, height, width)
    count = int(generator.integers(1, 6))
    # Points avoid (0, 0): the keypoint block skips it before labelling.
    points = np.stack(
        [
            generator.integers(1, width + 5, count),
            generator.integers(1, height + 5, count),
        ],
        1,
    )
    style = TextStyle(
        scale=float(generator.choice([0.3, 0.5])),
        thickness=1,
        padding=int(generator.integers(0, 6)),
        border_radius=int(generator.integers(0, 6)),
    )
    texts = [str(index) for index in range(count)]

    labels = vertex_labels(
        texts,
        [(int(x), int(y)) for x, y in points],
        style=style,
        background_rgb=(10, 200, 30),
        text_rgb=(0, 0, 0),
    )

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
    np.testing.assert_array_equal(v2_labels(pixels, labels), expected[:, :, ::-1])


@pytest.mark.parametrize("seed", range(10))
def test_a_label_changes_pixels_only_inside_its_region(seed: int) -> None:
    generator = np.random.default_rng(200 + seed)
    pixels = random_image(generator, 40, 50)
    anchor = generator.uniform(-10, 55, (1, 2))
    style = TextStyle(
        scale=0.5, thickness=int(generator.integers(1, 4)), padding=3, border_radius=4
    )
    (label,) = detection_labels(
        ["Tg\nq"],
        anchor,
        position=BOX_POSITIONS[seed % len(BOX_POSITIONS)],
        style=style,
        background_rgb=[(1, 2, 3)],
        text_rgb=(250, 250, 250),
    )

    changed = (v2_labels(pixels, [label]) != pixels).any(axis=2)

    x0, y0, x1, y1 = label.region()
    outside = changed.copy()
    outside[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)] = False
    assert not outside.any()


def test_rounded_background_keeps_its_corner_and_text_is_antialiased() -> None:
    pixels = np.zeros((40, 80, 3), dtype=np.uint8)
    anchors = np.array([[20.0, 30.0]])

    def drawn(border_radius: int) -> Tuple[np.ndarray, Label]:
        style = TextStyle(
            scale=0.5, thickness=1, padding=4, border_radius=border_radius
        )
        (label,) = detection_labels(
            ["dog"],
            anchors,
            position="TOP_LEFT",
            style=style,
            background_rgb=[(0, 0, 200)],
            text_rgb=(255, 255, 255),
        )
        return v2_labels(pixels, [label]), label

    square, label = drawn(0)
    rounded, _ = drawn(5)

    x1, y1, x2, y2 = label.background
    assert tuple(square[y1, x1]) == (0, 0, 200)
    assert tuple(rounded[y1, x1]) == (0, 0, 0)
    assert tuple(rounded[(y1 + y2) // 2, x1]) == (0, 0, 200)
    inside = square[y1 : y2 + 1, x1 : x2 + 1].reshape(-1, 3)
    pure = (inside == (0, 0, 200)).all(axis=1) | (inside == 255).all(axis=1)
    assert (~pure).any(), "LINE_AA text blends partial pixels into the background"


def test_later_labels_cover_earlier_ones() -> None:
    pixels = np.zeros((40, 60, 3), dtype=np.uint8)
    style = TextStyle(scale=0.5, thickness=1, padding=5, border_radius=0)
    anchors = np.array([[10.0, 30.0], [14.0, 32.0]])

    labels = detection_labels(
        ["first", "second"],
        anchors,
        position="TOP_LEFT",
        style=style,
        background_rgb=[(200, 0, 0), (0, 200, 0)],
        text_rgb=(255, 255, 255),
    )
    reversed_labels = labels[::-1]

    x1, y1 = labels[1].background[:2]
    assert tuple(v2_labels(pixels, labels)[y1 + 1, x1 + 1]) == (0, 200, 0)
    assert tuple(v2_labels(pixels, reversed_labels)[y1 + 1, x1 + 1]) == (200, 0, 0)


def test_no_labels_or_labels_off_the_frame_change_nothing() -> None:
    pixels = random_image(np.random.default_rng(7), 20, 30)
    style = TextStyle(scale=0.5, thickness=1, padding=2, border_radius=0)
    far_away = detection_labels(
        ["x"],
        np.array([[500.0, 500.0]]),
        position="TOP_LEFT",
        style=style,
        background_rgb=[(1, 1, 1)],
        text_rgb=(2, 2, 2),
    )

    no_rows = detection_labels(
        [],
        np.zeros((0, 2)),
        position="CENTER",
        style=style,
        background_rgb=[],
        text_rgb=(2, 2, 2),
    )

    assert no_rows == []
    np.testing.assert_array_equal(v2_labels(pixels, []), pixels)
    np.testing.assert_array_equal(v2_labels(pixels, far_away), pixels)


def test_center_of_mass_labels_are_laid_out_as_center() -> None:
    style = TextStyle(scale=0.5, thickness=1, padding=3, border_radius=0)
    common = dict(style=style, background_rgb=[(1, 1, 1)], text_rgb=(2, 2, 2))

    centre = detection_labels(["ab"], np.array([[20, 20]]), position="CENTER", **common)
    mass = detection_labels(
        ["ab"], np.array([[20, 20]]), position="CENTER_OF_MASS", **common
    )

    assert centre == mass
    with pytest.raises(ValueError, match="unknown text position"):
        detection_labels(["ab"], np.array([[20, 20]]), position="MIDDLE", **common)


# ---------------------------------------------------------------- raster


def window_sum_tiles(
    sums: np.ndarray, kernel_size: int
) -> Tuple[np.ndarray, int, np.ndarray]:
    """One-channel image holding one isolated ``k x k`` tile per window sum.

    A tile is 255s, one remainder pixel and zeros. Returns the image and the
    row and columns of the pixels whose window is exactly one tile.
    """
    k, count = kernel_size, len(sums)
    flat = np.arange(k * k)
    full = sums[:, None] // 255
    tiles = np.where(flat < full, 255, np.where(flat == full, sums[:, None] % 255, 0))
    image = np.zeros((3 * k, k + 2 * k * count), dtype=np.uint8)
    for index, tile in enumerate(tiles.reshape(count, k, k)):
        image[k : 2 * k, k + 2 * k * index : 2 * k + 2 * k * index] = tile
    row = k + k // 2
    columns = row + 2 * k * np.arange(count)

    return image, row, columns


SMALL_SHAPES = [(1, 1), (1, 7), (2, 2), (5, 3), (9, 13), (3, 20), (17, 19)]
# Odd and even; areas <= 256 (16-bit sums) and above (float32 division).
BLUR_KERNELS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 17, 18, 22, 40, 62]


@pytest.mark.parametrize("kernel_size", BLUR_KERNELS)
@pytest.mark.parametrize("shape", SMALL_SHAPES)
def test_box_blur_equals_cv2_blur(kernel_size: int, shape: Tuple[int, int]) -> None:
    pixels = random_image(np.random.default_rng(kernel_size), *shape)
    region = chw(pixels)

    blurred = box_blur(region, kernel_size)

    np.testing.assert_array_equal(
        hwc(blurred), cv2.blur(pixels, (kernel_size, kernel_size))
    )
    assert torch.equal(region, chw(pixels)), "the input region stays unchanged"


@pytest.mark.parametrize("kernel_size", [2, 4, 6, 8, 16, 22, 34, 40, 44, 62])
def test_box_blur_rounds_means_near_a_half_as_cv2_blur(kernel_size: int) -> None:
    # Where the rounding rules differ: every sum of kernel 2 (1/4 rounds up),
    # and for all kernels every sum within 1 of a .5 mean.
    area = kernel_size * kernel_size
    halves = np.arange(255)[:, None] * area + area // 2 + np.array([-1, 0, 1])
    sums = np.arange(255 * area + 1) if kernel_size == 2 else halves.ravel()
    image, row, columns = window_sum_tiles(sums, kernel_size)

    blurred = box_blur(torch.from_numpy(image)[None], kernel_size)[0].numpy()

    expected = cv2.blur(image, (kernel_size, kernel_size))
    np.testing.assert_array_equal(blurred[row, columns], expected[row, columns])


def test_box_blur_rounds_dim_means_up_like_cv2_blur() -> None:
    # A single 1 in a 2x2 window: mean 1/4 is 1 in cv2.blur, not 0.
    pixels = np.zeros((6, 6, 1), dtype=np.uint8)
    pixels[2, 2] = 1

    blurred = hwc(box_blur(chw(pixels), 2))

    np.testing.assert_array_equal(blurred, cv2.blur(pixels, (2, 2))[:, :, None])
    assert np.argwhere(blurred[:, :, 0]).tolist() == [[2, 2], [2, 3], [3, 2], [3, 3]]


def test_box_blur_keeps_flat_regions_and_never_aliases_its_input() -> None:
    flat = torch.full((3, 6, 5), 77, dtype=torch.uint8)

    for kernel_size in (1, 2, 4, 9):
        blurred = box_blur(flat, kernel_size)
        assert torch.equal(blurred, flat)
        assert blurred.data_ptr() != flat.data_ptr()


@pytest.mark.parametrize("padding", [1, 2, 3, 5])
@pytest.mark.parametrize("seed", range(4))
def test_grow_mask_equals_cv2_dilate_with_an_elliptic_kernel(
    padding: int, seed: int
) -> None:
    generator = np.random.default_rng(seed)
    height, width = int(generator.integers(1, 25)), int(generator.integers(1, 25))
    mask = generator.random((height, width)) < 0.08
    mask[0, 0] = bool(seed % 2)  # sometimes a pixel on the raster edge

    grown = grow_mask(torch.from_numpy(mask), padding).numpy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * padding + 1,) * 2)
    expected = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    np.testing.assert_array_equal(grown, expected)
    assert grown.dtype == np.bool_ and not (mask & ~grown).any()


def test_grow_mask_of_an_empty_mask_is_empty() -> None:
    grown = grow_mask(torch.zeros((7, 9), dtype=torch.bool), 3)

    assert grown.shape == (7, 9) and not grown.any()


RESIZE_SOURCES = [(1, 1), (3, 5), (7, 9), (20, 15), (13, 2)]
RESIZE_TARGETS = [(1, 1), (2, 3), (6, 10), (40, 30), (5, 17)]


@pytest.mark.parametrize("source", RESIZE_SOURCES)
@pytest.mark.parametrize("target", RESIZE_TARGETS)
def test_resize_nearest_equals_cv2(
    source: Tuple[int, int], target: Tuple[int, int]
) -> None:
    pixels = random_image(np.random.default_rng(sum(source + target)), *source)

    resized = hwc(resize_nearest(chw(pixels), target))

    expected = cv2.resize(pixels, target[::-1], interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(resized, expected)


@pytest.mark.parametrize("source, target", [((6, 2), (34, 98)), ((9, 4), (51, 98))])
def test_resize_nearest_follows_cv2_double_scale_rounding(
    source: Tuple[int, int], target: Tuple[int, int]
) -> None:
    # Here floor(i * (1 / (out / in))) in double, as cv2 computes it, differs
    # from the exact floor(i * in / out) for some output pixels.
    pixels = random_image(np.random.default_rng(0), *source)

    resized = hwc(resize_nearest(chw(pixels), target))

    expected = cv2.resize(pixels, target[::-1], interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(resized, expected)


@pytest.mark.parametrize("source", RESIZE_SOURCES)
@pytest.mark.parametrize("target", RESIZE_TARGETS)
def test_resize_linear_to_a_size_is_within_one_level_of_cv2(
    source: Tuple[int, int], target: Tuple[int, int]
) -> None:
    pixels = random_image(np.random.default_rng(sum(source + target)), *source)
    scale_xy = (source[1] / target[1], source[0] / target[0])

    resized = hwc(resize_linear(chw(pixels), target, scale_xy=scale_xy))

    expected = cv2.resize(pixels, target[::-1], interpolation=cv2.INTER_LINEAR)
    assert resized.shape == expected.shape
    assert max_level_difference(resized, expected) <= 1


@pytest.mark.parametrize("factor", [1 / 2, 1 / 3, 1 / 6, 0.25, 0.7, 2.0])
@pytest.mark.parametrize("source", [(7, 9), (20, 15), (40, 33), (6, 6)])
def test_resize_linear_by_a_factor_is_within_one_level_of_cv2(
    factor: float, source: Tuple[int, int]
) -> None:
    pixels = random_image(np.random.default_rng(int(factor * 100)), *source)
    expected = cv2.resize(
        pixels, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR
    )

    resized = hwc(
        resize_linear(
            chw(pixels), expected.shape[:2], scale_xy=(1 / factor, 1 / factor)
        )
    )

    assert max_level_difference(resized, expected) <= 1


@pytest.mark.parametrize("factor", [2.0, 0.5, 4.0])
def test_resize_linear_is_exact_when_every_sample_is_an_integer(factor: float) -> None:
    # Per-axis weights are k/8 at most here, so 2-D weights are k/64; values
    # that are multiples of 64 leave no rounding to disagree on. This pins
    # cv2's sample positions and edge clamping.
    pixels = (random_image(np.random.default_rng(3), 8, 12) // 64) * 64
    expected = cv2.resize(
        pixels, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR
    )

    resized = hwc(
        resize_linear(
            chw(pixels), expected.shape[:2], scale_xy=(1 / factor, 1 / factor)
        )
    )

    np.testing.assert_array_equal(resized, expected)


# ---------------------------------------------------------------- anchors


@pytest.mark.parametrize("position", BOX_POSITIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_box_anchors_equal_sv_anchors(position: str, dtype: type) -> None:
    xyxy = np.array([[4.6, 6.2, 30.4, 30.9], [-3.0, 2.5, 7.0, 8.0]], dtype=dtype)

    anchors = box_anchors(xyxy, position)

    expected = sv.Detections(xyxy=xyxy).get_anchors_coordinates(sv.Position[position])
    np.testing.assert_array_equal(anchors, expected)
    assert anchors.dtype == dtype


def test_box_anchors_of_no_rows_and_unknown_positions() -> None:
    anchors = box_anchors(np.zeros((0, 4), dtype=np.float32), "CENTER")

    assert anchors.shape == (0, 2)
    with pytest.raises(ValueError, match="unknown anchor position"):
        box_anchors(np.zeros((1, 4)), "MIDDLE")


def test_mass_centres_equal_sv_center_of_mass() -> None:
    generator = np.random.default_rng(5)
    masks = generator.random((6, 17, 23)) < 0.3
    masks[1] = False  # empty mask: V1 gives (0, 0)
    masks[2] = False
    masks[2, 16, 22] = True  # one pixel in the far corner
    xyxy = np.zeros((len(masks), 4), dtype=np.float32)

    centres = mass_centres(torch.from_numpy(masks))

    detections = sv.Detections(xyxy=xyxy, mask=masks)
    expected = detections.get_anchors_coordinates(sv.Position.CENTER_OF_MASS)
    np.testing.assert_array_equal(centres, expected.astype(np.int64))
    assert centres[1].tolist() == [0, 0] and centres[2].tolist() == [22, 16]
    assert mass_centres(torch.zeros((0, 4, 4), dtype=torch.bool)).shape == (0, 2)


# ---------------------------------------------------------------- blend


def sv_masks(
    pixels: np.ndarray, masks: np.ndarray, colors: np.ndarray, opacity: float
) -> np.ndarray:
    annotator = sv.MaskAnnotator(
        color=sv.ColorPalette([sv.Color(*map(int, rgb)) for rgb in colors]),
        opacity=opacity,
        color_lookup=sv.ColorLookup.INDEX,
    )
    xyxy = np.zeros((len(masks), 4), dtype=np.float32)
    detections = sv.Detections(xyxy=xyxy, mask=masks)
    annotated = annotator.annotate(pixels[:, :, ::-1].copy(), detections)

    return annotated[:, :, ::-1]


@pytest.mark.parametrize("seed", range(12))
def test_painted_and_blended_masks_equal_sv_mask_annotator(seed: int) -> None:
    generator = np.random.default_rng(300 + seed)
    height, width = int(generator.integers(4, 30)), int(generator.integers(4, 30))
    pixels = random_image(generator, height, width)
    count = int(generator.integers(1, 9))
    masks = generator.random((count, height, width)) < generator.uniform(
        0.05, 0.6, (count, 1, 1)
    )
    colors = generator.integers(0, 256, (count, 3), dtype=np.uint8)
    opacity = float(generator.choice([0.3, 0.5, 0.75]))
    canvas = chw(pixels)

    colored = paint_instances(canvas, torch.from_numpy(masks), torch.from_numpy(colors))
    blended = hwc(blend(canvas, colored, opacity=opacity))

    expected = sv_masks(pixels, masks, colors, opacity)
    covered = masks.any(axis=0)
    np.testing.assert_array_equal(blended[~covered], pixels[~covered])
    if opacity == 0.3:  # not dyadic: <= 1 level at half ties, see blend tests
        assert max_level_difference(blended, expected) <= 1
    else:
        np.testing.assert_array_equal(blended, expected)
    assert torch.equal(canvas, chw(pixels)), "inputs stay unchanged"


def test_equal_areas_paint_the_first_row_on_top() -> None:
    # V1: np.flip(np.argsort(areas)) paints the later of two equal areas first.
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 1:3, 1:3] = True
    colors = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    canvas = torch.zeros((3, 4, 4), dtype=torch.uint8)

    colored = paint_instances(canvas, torch.from_numpy(masks), torch.from_numpy(colors))

    assert colored[:, 1, 1].tolist() == [255, 0, 0]
    np.testing.assert_array_equal(
        hwc(blend(canvas, colored, opacity=1.0)),
        sv_masks(np.zeros((4, 4, 3), np.uint8), masks, colors, 1.0),
    )


def every_value_pair() -> Tuple[np.ndarray, np.ndarray]:
    """``(256, 256, 3)`` image and layer holding every uint8 value pair once."""
    values = np.arange(256, dtype=np.uint8)
    layer, image = np.meshgrid(values, values)
    pair = (np.repeat(image[..., None], 3, 2), np.repeat(layer[..., None], 3, 2))

    return pair


@pytest.mark.parametrize("opacity", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_blend_equals_cv2_add_weighted_for_dyadic_opacities(opacity: float) -> None:
    image, layer = every_value_pair()

    blended = blend(chw(image), chw(layer), opacity=opacity)

    expected = cv2.addWeighted(layer, opacity, image, 1 - opacity, 0)
    np.testing.assert_array_equal(hwc(blended), expected)


@pytest.mark.parametrize("opacity", [0.1, 0.3, 0.6, 0.7, 0.9])
def test_blend_differs_from_cv2_only_by_one_level_at_half_ties(opacity: float) -> None:
    # Documented approximation: float32 rounding may land a true .5 tie on
    # the other side than cv2.addWeighted does.
    image, layer = every_value_pair()

    blended = hwc(blend(chw(image), chw(layer), opacity=opacity))

    expected = cv2.addWeighted(layer, opacity, image, 1 - opacity, 0)
    true_value = layer.astype(np.float64) * opacity + image * (1 - opacity)
    at_tie = np.isclose(true_value - np.floor(true_value), 0.5, atol=1e-6)
    assert max_level_difference(blended, expected) <= 1
    assert not ((blended != expected) & ~at_tie).any()


def test_blend_changes_only_pixels_inside_its_region() -> None:
    generator = np.random.default_rng(11)
    image, layer = random_image(generator, 9, 7), random_image(generator, 9, 7)
    region = generator.random((9, 7)) < 0.4

    blended = hwc(
        blend(chw(image), chw(layer), opacity=0.4, region=torch.from_numpy(region))
    )

    everywhere = cv2.addWeighted(layer, 0.4, image, 0.6, 0)
    np.testing.assert_array_equal(blended[region], everywhere[region])
    np.testing.assert_array_equal(blended[~region], image[~region])


def test_no_masks_give_an_unchanged_copy() -> None:
    canvas = torch.arange(3 * 5 * 4, dtype=torch.uint8).reshape(3, 5, 4)

    colored = paint_instances(
        canvas,
        torch.zeros((0, 5, 4), dtype=torch.bool),
        torch.zeros((0, 3), dtype=torch.uint8),
    )

    assert torch.equal(colored, canvas) and colored.data_ptr() != canvas.data_ptr()


# ---------------------------------------------------------------- local MPS


@MPS
def test_device_helpers_give_cpu_pixels_on_mps() -> None:
    generator = np.random.default_rng(21)
    pixels = random_image(generator, 11, 13)
    masks = torch.from_numpy(generator.random((3, 11, 13)) < 0.3)
    colors = torch.from_numpy(generator.integers(0, 256, (3, 3), dtype=np.uint8))
    mps = torch.device("mps")
    cases = {
        "box_blur odd": lambda region: box_blur(region, 3),
        "box_blur even": lambda region: box_blur(region, 4),
        "resize_nearest": lambda region: resize_nearest(region, (5, 17)),
        "resize_linear": lambda region: resize_linear(
            region, (5, 6), scale_xy=(13 / 6, 11 / 5)
        ),
        "mask blend": lambda region: blend(
            region,
            paint_instances(region, masks.to(region.device), colors.to(region.device)),
            opacity=0.5,
        ),
    }
    failures = {}

    for name, run in cases.items():
        try:
            on_mps = run(chw(pixels).to(mps))
        except (RuntimeError, TypeError) as error:
            failures[name] = f"{type(error).__name__}: {error}"[:160]
            continue
        if on_mps.device.type != "mps" or not torch.equal(
            on_mps.cpu(), run(chw(pixels))
        ):
            failures[name] = "different pixels or device"

    assert failures == {}


@MPS
def test_grow_mask_and_mass_centres_on_mps_equal_cpu() -> None:
    masks = torch.from_numpy(np.random.default_rng(22).random((3, 11, 13)) < 0.2)
    mps = torch.device("mps")
    failures = {}

    try:
        grown = grow_mask(masks[0].to(mps), 2)
        if not torch.equal(grown.cpu(), grow_mask(masks[0], 2)):
            failures["grow_mask"] = "different pixels"
    except (RuntimeError, TypeError) as error:
        failures["grow_mask"] = f"{type(error).__name__}: {error}"[:160]
    try:
        centres = mass_centres(masks.to(mps))
        if not np.array_equal(centres, mass_centres(masks)):
            failures["mass_centres"] = "different centres"
    except (RuntimeError, TypeError) as error:
        failures["mass_centres"] = f"{type(error).__name__}: {error}"[:160]

    assert failures == {}


@MPS
def test_strokes_and_labels_paint_mps_scenes_like_cpu() -> None:
    pixels = random_image(np.random.default_rng(23), 30, 40)
    strokes = Strokes((30, 40))
    strokes.add(
        lambda mask: cv2.line(mask, (-3, 4), (35, 20), 1, 3),
        color=(5, 6, 7),
        scan=bounds_scan(np.array([[-3, 4], [35, 20]]), reach=stroke_reach(3)),
    )
    labels = vertex_labels(
        ["7"],
        [(20, 15)],
        style=TextStyle(scale=0.5, thickness=1, padding=2, border_radius=2),
        background_rgb=(0, 0, 0),
        text_rgb=(255, 255, 255),
    )
    results = []

    for device in ("cpu", "mps"):
        scene = chw(pixels).to(device)
        strokes.paint(scene)
        paint_labels(scene, labels)
        results.append(scene.cpu())

    assert torch.equal(results[0], results[1])
