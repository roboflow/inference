import logging

import numpy as np
import pytest
import supervision as sv
import torch

import inference.core.workflows.core_steps.visualizations.label.v1_tensor as label_v1_tensor
from inference.core.workflows.core_steps.visualizations.common.base_tensor import (
    to_supervision_for_annotation,
)
from inference.core.workflows.core_steps.visualizations.common.label_text import (
    build_detection_labels,
)
from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color
from inference.core.workflows.core_steps.visualizations.label.v1_tensor import (
    LabelVisualizationBlockV1,
    _gpu_label_paste_eligible,
    _measure_label,
    _render_label_sprite,
    _SceneDependentLabelError,
    gpu_paste_label_sprites,
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
SCENE_H, SCENE_W = 480, 640
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

ALL_ANCHORS = [
    "TOP_LEFT",
    "TOP_RIGHT",
    "TOP_CENTER",
    "CENTER",
    "CENTER_LEFT",
    "CENTER_RIGHT",
    "BOTTOM_LEFT",
    "BOTTOM_CENTER",
    "BOTTOM_RIGHT",
]

_DEFAULT_RUN_KWARGS = dict(
    copy_image=True,
    color_palette="DEFAULT",
    palette_size=10,
    custom_colors=None,
    color_axis="CLASS",
    text="Class and Confidence",
    text_position="TOP_LEFT",
    text_color="WHITE",
    text_scale=1.0,
    text_thickness=1,
    text_padding=10,
    border_radius=0,
)


def _make_scene(seed: int, h: int = SCENE_H, w: int = SCENE_W) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3)).astype(np.uint8)


def _build_detections(
    boxes: np.ndarray,
    class_id: np.ndarray,
    device: str = "cpu",
    confidence: np.ndarray = None,
    bboxes_metadata=None,
) -> Detections:
    n = boxes.shape[0]
    if confidence is None:
        confidence = np.linspace(0.42, 0.99, max(n, 1))[:n]
    return Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32, device=device),
        class_id=torch.tensor(class_id, dtype=torch.int32, device=device),
        confidence=torch.tensor(confidence, dtype=torch.float32, device=device),
        image_metadata={"class_names": {0: "goggles", 1: "cat", 2: "dog"}},
        bboxes_metadata=bboxes_metadata,
    )


def _default_detections(device: str = "cpu") -> Detections:
    boxes = np.array(
        [
            [40, 60, 200, 200],
            [250, 90, 420, 260],
            [90, 260, 300, 430],
        ],
        dtype=np.float32,
    )
    return _build_detections(boxes, np.array([0, 1, 2]), device=device)


def _tensor_image_from_bgr(scene_bgr: np.ndarray, device: str = "cpu"):
    tensor = (
        torch.from_numpy(scene_bgr[:, :, ::-1].copy())
        .permute(2, 0, 1)
        .contiguous()
        .to(device)
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"), tensor_image=tensor
    )


def _run_block(image, detections, block=None, **overrides):
    block = block or LabelVisualizationBlockV1()
    kwargs = {**_DEFAULT_RUN_KWARGS, **overrides}
    return block.run(image=image, predictions=detections, **kwargs)["image"]


def _sv_reference(scene_bgr: np.ndarray, detections: Detections, **overrides):
    """What the sv path (and the flag-off block) draws for the same inputs."""
    kwargs = {**_DEFAULT_RUN_KWARGS, **overrides}
    sv_view = to_supervision_for_annotation(detections)
    labels = build_detection_labels(sv_view, kwargs["text"])
    annotator = sv.LabelAnnotator(
        color=PALETTE,
        color_lookup=getattr(sv.ColorLookup, kwargs["color_axis"]),
        text_position=getattr(sv.Position, kwargs["text_position"]),
        text_color=str_to_color(kwargs["text_color"]),
        text_scale=kwargs["text_scale"],
        text_thickness=kwargs["text_thickness"],
        text_padding=kwargs["text_padding"],
        border_radius=kwargs["border_radius"],
    )
    return annotator.annotate(scene_bgr.copy(), sv_view, labels=labels)


def _to_bgr(tensor_chw: torch.Tensor) -> np.ndarray:
    return tensor_chw.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]


def _assert_gpu_bit_exact(out, expected_bgr: np.ndarray) -> None:
    # the GPU path returns a tensor image and never materialises numpy
    assert out._tensor_image is not None and out._numpy_image is None
    actual = _to_bgr(out._tensor_image)
    assert np.array_equal(actual, expected_bgr)


def test_class_axis_paints_unknown_class_gray() -> None:
    scene = np.zeros((64, 64, 3), dtype=np.uint8)
    detections = _build_detections(
        boxes=np.array([[10, 20, 50, 54]], dtype=np.float32),
        class_id=np.array([-1]),
    )

    out = _run_block(
        _tensor_image_from_bgr(scene),
        detections,
        text="Confidence",
        text_scale=0.5,
        text_padding=2,
    )

    assert out._tensor_image is not None and out._numpy_image is None
    annotated = _to_bgr(out._tensor_image)
    assert np.any(np.all(annotated == (128, 128, 128), axis=-1))
    assert detections.class_id.tolist() == [-1]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("text_position", ALL_ANCHORS)
def test_gpu_labels_match_sv_bit_exact_for_every_anchor(
    device: str, text_position: str
) -> None:
    # given: interior boxes plus boxes that push the label over each frame
    # edge (the label patch of an edge box lands partially off-frame for most
    # anchors, exercising the frame-clipped sprite variants)
    scene = _make_scene(11)
    boxes = np.array(
        [
            [40, 60, 200, 200],
            [250, 90, 420, 260],
            [90, 260, 300, 430],
            [5, 5, 60, 60],
            [560, 400, 635, 475],
        ],
        dtype=np.float32,
    )
    detections = _build_detections(boxes, np.array([0, 1, 2, 1, 0]), device=device)
    expected = _sv_reference(scene, detections, text_position=text_position)

    # when
    out = _run_block(
        _tensor_image_from_bgr(scene, device=device),
        detections,
        text_position=text_position,
    )

    # then
    assert out._tensor_image is not None and out._numpy_image is None
    assert np.array_equal(_to_bgr(out._tensor_image), expected)


@pytest.mark.parametrize(
    "text", ["Class", "Confidence", "Class and Confidence", "Index", "Dimensions"]
)
def test_gpu_labels_match_sv_for_text_options(text: str) -> None:
    # given
    scene = _make_scene(23)
    detections = _default_detections()
    expected = _sv_reference(scene, detections, text=text)

    # when
    out = _run_block(_tensor_image_from_bgr(scene), detections, text=text)

    # then
    _assert_gpu_bit_exact(out, expected)


def test_gpu_labels_match_sv_with_overlapping_labels() -> None:
    # given: staggered boxes whose label patches overlap each other — sv draws
    # sequentially, so a later label must overwrite an earlier one; the last
    # two detections share the same box, so their labels coincide entirely
    scene = _make_scene(31)
    boxes = np.array(
        [
            [100, 100, 300, 300],
            [130, 115, 330, 300],
            [160, 130, 360, 300],
            [160, 130, 360, 300],
        ],
        dtype=np.float32,
    )
    detections = _build_detections(boxes, np.array([0, 1, 2, 1]))
    expected = _sv_reference(scene, detections)

    # when
    out = _run_block(_tensor_image_from_bgr(scene), detections)

    # then: paste order matches sv's draw order bit-for-bit
    _assert_gpu_bit_exact(out, expected)


@pytest.mark.parametrize("text_position", ["TOP_LEFT", "CENTER", "BOTTOM_RIGHT"])
def test_gpu_labels_match_sv_when_clipped_at_every_frame_edge(
    text_position: str,
) -> None:
    # given: boxes hugging each frame edge and corner so label patches are
    # clipped left/top/right/bottom and across two edges at once — cv2's AA
    # rasterisation changes where strokes are cut by the frame border, which
    # the frame-clipped sprite variants must reproduce exactly
    scene = _make_scene(47)
    boxes = np.array(
        [
            [2, 2, 60, 40],  # top-left corner
            [580, 3, 637, 50],  # top-right corner
            [3, 430, 70, 477],  # bottom-left corner
            [590, 440, 636, 476],  # bottom-right corner
            [2, 200, 40, 260],  # left edge
            [600, 200, 638, 260],  # right edge
        ],
        dtype=np.float32,
    )
    detections = _build_detections(boxes, np.array([0, 1, 2, 0, 1, 2]))
    expected = _sv_reference(scene, detections, text_position=text_position)

    # when
    out = _run_block(
        _tensor_image_from_bgr(scene), detections, text_position=text_position
    )

    # then
    _assert_gpu_bit_exact(out, expected)


def test_gpu_labels_match_sv_when_label_is_wider_than_frame() -> None:
    # given: a label so long it overflows both vertical frame edges at once
    scene = _make_scene(53, h=240, w=320)
    boxes = np.array([[100, 100, 220, 200]], dtype=np.float32)
    detections = _build_detections(boxes, np.array([0]))
    detections.bboxes_metadata = [
        {"note": "a very long annotation that cannot possibly fit this frame"}
    ]
    expected = _sv_reference(scene, detections, text="note", text_position="CENTER")

    # when
    out = _run_block(
        _tensor_image_from_bgr(scene), detections, text="note", text_position="CENTER"
    )

    # then
    _assert_gpu_bit_exact(out, expected)


def test_gpu_labels_match_sv_for_fully_off_frame_labels() -> None:
    # given: box far outside the frame — sv draws nothing visible
    scene = _make_scene(59, h=240, w=320)
    boxes = np.array([[400, 400, 500, 500]], dtype=np.float32)
    detections = _build_detections(boxes, np.array([0]))

    # when
    out = _run_block(_tensor_image_from_bgr(scene), detections)

    # then: scene unchanged, still on the tensor path
    _assert_gpu_bit_exact(out, scene)


@pytest.mark.parametrize("border_radius", [3, 8, 15])
def test_gpu_labels_match_sv_with_border_radius(border_radius: int) -> None:
    # given: rounded corners leave the scene visible through the corner cuts
    scene = _make_scene(61)
    detections = _default_detections()
    expected = _sv_reference(scene, detections, border_radius=border_radius)

    # when
    out = _run_block(
        _tensor_image_from_bgr(scene), detections, border_radius=border_radius
    )

    # then
    _assert_gpu_bit_exact(out, expected)


@pytest.mark.parametrize(
    "text_scale,text_thickness,text_padding",
    [(0.5, 1, 10), (0.5, 2, 8), (2.0, 2, 25), (1.0, 1, 15)],
)
def test_gpu_labels_match_sv_with_custom_typography(
    text_scale: float, text_thickness: int, text_padding: int
) -> None:
    # given
    scene = _make_scene(67)
    detections = _default_detections()
    expected = _sv_reference(
        scene,
        detections,
        text_scale=text_scale,
        text_thickness=text_thickness,
        text_padding=text_padding,
    )

    # when
    out = _run_block(
        _tensor_image_from_bgr(scene),
        detections,
        text_scale=text_scale,
        text_thickness=text_thickness,
        text_padding=text_padding,
    )

    # then
    _assert_gpu_bit_exact(out, expected)


def test_gpu_labels_match_sv_for_multiline_custom_text() -> None:
    # given: sv's wrap_text splits custom labels on newlines
    scene = _make_scene(71)
    boxes = np.array([[80, 120, 300, 300], [320, 120, 560, 300]], dtype=np.float32)
    detections = _build_detections(boxes, np.array([0, 1]))
    detections.bboxes_metadata = [
        {"note": "first line\nsecond line"},
        {"note": "single"},
    ]
    expected = _sv_reference(scene, detections, text="note")

    # when
    out = _run_block(_tensor_image_from_bgr(scene), detections, text="note")

    # then
    _assert_gpu_bit_exact(out, expected)


@pytest.mark.parametrize("color_axis", ["INDEX", "TRACK"])
def test_gpu_labels_match_sv_for_other_color_axes(color_axis: str) -> None:
    # given: non-contiguous tracker ids incl. sv's pending track (-1), whose
    # resolve_color turns BOTH the background and the text gray
    scene = _make_scene(73)
    boxes = np.array(
        [[40, 60, 200, 200], [250, 90, 420, 260], [90, 260, 300, 430]],
        dtype=np.float32,
    )
    detections = _build_detections(
        boxes,
        np.array([0, 1, 2]),
        bboxes_metadata=[{"tracker_id": 12}, {"tracker_id": -1}, {"tracker_id": 3}],
    )
    expected = _sv_reference(scene, detections, color_axis=color_axis)

    # when
    out = _run_block(_tensor_image_from_bgr(scene), detections, color_axis=color_axis)

    # then
    _assert_gpu_bit_exact(out, expected)


def test_track_axis_without_tracker_ids_raises_sv_error() -> None:
    # given: TRACK lookup but no tracker ids — the sv annotator's exact
    # ValueError must surface (via the sv fallback)
    scene = _make_scene(79)
    detections = _default_detections()

    # when / then
    with pytest.raises(ValueError, match="resolve color by track"):
        _run_block(_tensor_image_from_bgr(scene), detections, color_axis="TRACK")


def test_sprite_cache_reuses_sprites_across_runs(monkeypatch) -> None:
    # given: a render-call counter around the real renderer
    calls = {"n": 0}
    real_render = label_v1_tensor._render_label_sprite

    def counting_render(*args, **kwargs):
        calls["n"] += 1
        return real_render(*args, **kwargs)

    monkeypatch.setattr(label_v1_tensor, "_render_label_sprite", counting_render)
    scene = _make_scene(83)
    detections = _default_detections()  # 3 distinct labels
    block = LabelVisualizationBlockV1()

    # when: the same frame twice
    _run_block(_tensor_image_from_bgr(scene), detections, block=block)
    first_run_renders = calls["n"]
    _run_block(_tensor_image_from_bgr(scene), detections, block=block)

    # then: one sprite per distinct label, and the second run renders nothing
    assert first_run_renders == 3
    assert len(block._sprite_cache) == 3
    assert calls["n"] == first_run_renders


@pytest.mark.parametrize("device", DEVICES)
def test_cached_sprites_are_device_resident_tensors(device: str) -> None:
    # given
    scene = _make_scene(89)
    detections = _default_detections(device=device)
    block = LabelVisualizationBlockV1()

    # when
    _run_block(_tensor_image_from_bgr(scene, device=device), detections, block=block)

    # then: sprite pixel payloads live on the scene device — a cache-hit frame
    # uploads no pixel data
    assert len(block._sprite_cache) > 0
    for sprite in block._sprite_cache.values():
        assert isinstance(sprite.colors_dev, torch.Tensor)
        assert sprite.colors_dev.device.type == device
        for flat in sprite._flat_by_width.values():
            assert flat.device.type == device


def test_sprite_cache_is_bounded(monkeypatch) -> None:
    # given
    monkeypatch.setattr(label_v1_tensor, "_SPRITE_CACHE_MAX_ENTRIES", 2)
    scene = _make_scene(97)
    boxes = np.array(
        [
            [40, 60, 200, 200],
            [250, 90, 420, 260],
            [90, 260, 300, 430],
            [330, 280, 500, 430],
        ],
        dtype=np.float32,
    )
    detections = _build_detections(boxes, np.array([0, 1, 2, 0]))
    block = LabelVisualizationBlockV1()

    # when: 4 distinct labels through a cache capped at 2
    _run_block(_tensor_image_from_bgr(scene), detections, block=block)

    # then
    assert len(block._sprite_cache) == 2


def test_copy_image_true_leaves_input_untouched() -> None:
    # given
    scene = _make_scene(101)
    image = _tensor_image_from_bgr(scene)
    original = image.tensor_image.clone()

    # when
    out = _run_block(image, _default_detections(), copy_image=True)

    # then: independent storage, input pixels intact, output annotated
    assert out._tensor_image is not None
    assert out._tensor_image.data_ptr() != image.tensor_image.data_ptr()
    assert torch.equal(image.tensor_image, original)
    assert not torch.equal(out._tensor_image, original)


def test_copy_image_false_mutates_input_in_place() -> None:
    # given
    scene = _make_scene(103)
    image = _tensor_image_from_bgr(scene)
    input_tensor = image.tensor_image

    # when
    out = _run_block(image, _default_detections(), copy_image=False)

    # then: same storage annotated in place, and numpy is never materialised
    assert out._tensor_image is not None
    assert out._tensor_image.data_ptr() == input_tensor.data_ptr()
    assert image._numpy_image is None


def _empty_detections(device: str = "cpu") -> Detections:
    return _build_detections(
        np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=int), device=device
    )


def test_empty_predictions_take_the_tensor_passthrough_with_copy() -> None:
    # given
    scene = _make_scene(107)
    image = _tensor_image_from_bgr(scene)

    # when
    out = _run_block(image, _empty_detections(), copy_image=True)

    # then: output stays on-device (an empty annotate must never pay the
    # full-resolution numpy materialisation) with independent storage
    assert out._tensor_image is not None and out._numpy_image is None
    assert out._tensor_image.data_ptr() != image.tensor_image.data_ptr()
    assert torch.equal(out._tensor_image, image.tensor_image)
    assert image._numpy_image is None


def test_empty_predictions_passthrough_shares_backing_without_copy() -> None:
    # given
    image = _tensor_image_from_bgr(_make_scene(109))

    # when
    out = _run_block(image, _empty_detections(), copy_image=False)

    # then
    assert out._tensor_image is not None and out._numpy_image is None
    assert out._tensor_image.data_ptr() == image.tensor_image.data_ptr()


def test_empty_predictions_on_numpy_sourced_image_stay_numpy() -> None:
    # given
    scene = _make_scene(113)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"), numpy_image=scene
    )

    # when
    out = _run_block(image, _empty_detections(), copy_image=True)

    # then
    assert out._numpy_image is not None and out._tensor_image is None
    assert not np.shares_memory(out._numpy_image, scene)
    assert np.array_equal(out._numpy_image, scene)


def test_numpy_sourced_image_takes_the_sv_path_unchanged() -> None:
    # given: a numpy-backed image must behave exactly as before the GPU path
    # existed — same sv annotator, numpy output, no tensor materialisation
    scene = _make_scene(127)
    detections = _default_detections()
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"), numpy_image=scene
    )
    expected = _sv_reference(scene, detections)

    # when
    out = _run_block(image, detections)

    # then
    assert out._numpy_image is not None and out._tensor_image is None
    assert image._tensor_image is None  # never materialised as a side effect
    assert np.array_equal(out._numpy_image, expected)


def test_mask_dependent_area_text_keeps_the_sv_path() -> None:
    # given: `Area` reads masks in sv (falling back to box area for OD input)
    # — the sprite compositor must not take over that configuration
    scene = _make_scene(131)
    detections = _default_detections()
    expected = _sv_reference(scene, detections, text="Area")

    # when
    out = _run_block(_tensor_image_from_bgr(scene), detections, text="Area")

    # then: sv fallback produced a numpy image identical to flag-off behavior
    assert out._numpy_image is not None
    assert np.array_equal(out._numpy_image, expected)


def test_center_of_mass_on_od_input_raises_the_sv_error() -> None:
    # given: CENTER_OF_MASS anchors on the mask centroid; without masks sv's
    # get_anchors_coordinates raises — the block must surface the same error
    # it always has (via the sv path), not swallow it in the GPU branch
    scene = _make_scene(131)
    detections = _default_detections()

    # when / then
    with pytest.raises(ValueError, match="CENTER_OF_MASS"):
        _run_block(
            _tensor_image_from_bgr(scene),
            detections,
            text_position="CENTER_OF_MASS",
        )


def test_scene_dependent_sprite_falls_back_to_sv_path() -> None:
    # given: text_padding smaller than the font's descender extent — the AA
    # descender ink of "goggles" escapes the opaque background, so sv blends
    # it with the scene; the sprite path must refuse and fall back, keeping
    # the output bit-identical to sv
    scene = _make_scene(137)
    detections = _default_detections()
    expected = _sv_reference(scene, detections, text="Class", text_padding=2)

    # when
    out = _run_block(
        _tensor_image_from_bgr(scene), detections, text="Class", text_padding=2
    )

    # then
    assert out._numpy_image is not None  # sv fallback was taken
    assert np.array_equal(out._numpy_image, expected)


def test_render_label_sprite_raises_when_text_ink_escapes_background() -> None:
    # given
    measurement = _measure_label("goggles", 1.0, 1, 2)
    margin = measurement.margin

    # when / then
    with pytest.raises(_SceneDependentLabelError):
        _render_label_sprite(
            measurement=measurement,
            text_color_bgr=(255, 255, 255),
            background_color_bgr=(0, 0, 255),
            text_scale=1.0,
            text_thickness=1,
            text_padding=2,
            border_radius=0,
            device=torch.device("cpu"),
            box_in_canvas=(
                margin,
                margin,
                margin + measurement.width_padded,
                margin + measurement.height_padded,
            ),
            canvas_hw=(
                measurement.height_padded + 1 + 2 * margin,
                measurement.width_padded + 1 + 2 * margin,
            ),
            frame_edge_sides=(False, False, False, False),
        )


def _interior_sprite(label: str = "ab", device: str = "cpu"):
    measurement = _measure_label(label, 1.0, 1, 10)
    margin = measurement.margin
    return _render_label_sprite(
        measurement=measurement,
        text_color_bgr=(255, 255, 255),
        background_color_bgr=(0, 0, 255),
        text_scale=1.0,
        text_thickness=1,
        text_padding=10,
        border_radius=0,
        device=torch.device(device),
        box_in_canvas=(
            margin,
            margin,
            margin + measurement.width_padded,
            margin + measurement.height_padded,
        ),
        canvas_hw=(
            measurement.height_padded + 1 + 2 * margin,
            measurement.width_padded + 1 + 2 * margin,
        ),
        frame_edge_sides=(False, False, False, False),
    )


def test_gpu_paste_is_in_place() -> None:
    # given
    sprite = _interior_sprite()
    scene = torch.zeros((3, 200, 200), dtype=torch.uint8)

    # when
    out = gpu_paste_label_sprites(scene, [sprite], [(5, 5)])

    # then
    assert out.data_ptr() == scene.data_ptr()
    assert int((out != 0).any(dim=0).sum()) > 0


def test_gpu_paste_rejects_non_contiguous_scene() -> None:
    # given: .view must fail (caught by the block's sv fallback) rather than
    # write into a silent copy
    sprite = _interior_sprite()
    scene = torch.zeros((200, 200, 3), dtype=torch.uint8).permute(2, 1, 0)

    # when / then
    with pytest.raises(RuntimeError):
        gpu_paste_label_sprites(scene, [sprite], [(5, 5)])


def test_gpu_paste_rejects_out_of_frame_pixels() -> None:
    # given: an origin that pushes sprite pixels off-frame — the block should
    # have picked a frame-clipped variant, so this is a hard error, raised
    # before any scene write
    sprite = _interior_sprite()
    scene = torch.zeros((3, 200, 200), dtype=torch.uint8)

    # when / then
    with pytest.raises(ValueError, match="outside the frame"):
        gpu_paste_label_sprites(scene, [sprite], [(150, 150)])
    assert int(scene.sum()) == 0  # untouched


def test_gpu_label_paste_eligible_semantics() -> None:
    # given
    tensor_image = _tensor_image_from_bgr(_make_scene(139))
    numpy_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"),
        numpy_image=_make_scene(139),
    )
    detections = _default_detections()

    # when / then
    for axis in ("CLASS", "INDEX", "TRACK"):
        assert _gpu_label_paste_eligible(detections, axis, tensor_image) is True
    assert _gpu_label_paste_eligible(detections, "SOMETHING", tensor_image) is False
    assert _gpu_label_paste_eligible(detections, "CLASS", numpy_image) is False
    assert (
        _gpu_label_paste_eligible(_empty_detections(), "CLASS", tensor_image) is False
    )
    assert _gpu_label_paste_eligible(object(), "CLASS", tensor_image) is False


@requires_cuda
def test_gpu_labels_on_cuda_match_cpu() -> None:
    # given
    scene = _make_scene(149)
    expected_out = _run_block(_tensor_image_from_bgr(scene), _default_detections())

    # when
    cuda_out = _run_block(
        _tensor_image_from_bgr(scene, device="cuda"),
        _default_detections(device="cuda"),
    )

    # then
    assert cuda_out._tensor_image.is_cuda
    assert np.array_equal(
        _to_bgr(cuda_out._tensor_image), _to_bgr(expected_out._tensor_image)
    )


def test_gpu_fallback_warns_once_then_stays_quiet(monkeypatch, caplog) -> None:
    # given: a permanently broken GPU fast path. It must be visible in
    # production logs (WARNING) on the first fallback, but must not emit one
    # warning per frame afterwards.
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated GPU compositor failure")

    monkeypatch.setattr(label_v1_tensor, "gpu_paste_label_sprites", _boom)
    scene = _make_scene(151)
    detections = _default_detections()
    block = LabelVisualizationBlockV1()

    def _warnings():
        return [r for r in caplog.records if r.levelno == logging.WARNING]

    # The top-level `inference` logger is configured with propagate=False
    # (see inference/core/logger.py), so its records never reach the root
    # logger pytest's caplog handler is attached to. Attach the caplog
    # handler directly to the module logger.
    label_v1_tensor.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=label_v1_tensor.logger.name):
            # when: two frames both hit the broken fast path
            first_out = _run_block(
                _tensor_image_from_bgr(scene), detections, block=block
            )
            after_first = _warnings()
            second_out = _run_block(
                _tensor_image_from_bgr(scene), detections, block=block
            )
            after_second = _warnings()
    finally:
        label_v1_tensor.logger.removeHandler(caplog.handler)

    # then: both frames still rendered through the sv fallback
    assert first_out._numpy_image is not None
    assert second_out._numpy_image is not None
    # ...and exactly one warning was emitted, naming the block and the error
    assert len(after_first) == 1
    message = after_first[0].getMessage()
    assert "Label Visualization" in message
    assert "simulated GPU compositor failure" in message
    assert len(after_second) == 1, "the second fallback must not warn again"
