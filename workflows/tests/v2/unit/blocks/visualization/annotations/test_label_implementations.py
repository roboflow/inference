"""Canonical label implementations preserve pixels, ownership and style changes."""

from contextlib import contextmanager

import pytest
import torch
from roboflow_workflows.v2.blocks.visualization.annotations.label_visualization import (
    _sprites,
)
from roboflow_workflows.v2.blocks.visualization.annotations.label_visualization.v2 import (
    AccurateLabels,
    FastLabels,
    LabelVisualizationBlock,
)
from roboflow_workflows.v2.engine.declaration import spec_of

from tests.v2.unit.blocks.visualization.annotations.support import (
    detections,
    image_of,
    instance_detections,
    scene,
)


def _run(implementation, image, predictions, **options):
    parameters = spec_of(LabelVisualizationBlock).validate_params(
        {"image": "$inputs.image", "predictions": "$inputs.predictions"}
    )
    arguments = {
        **dict(parameters),
        **options,
        "image": image,
        "predictions": predictions,
    }
    result = implementation.run(**arguments)["image"]

    return result


def test_accurate_is_default_and_fast_is_an_explicit_choice():
    assert LabelVisualizationBlock.implementations == (AccurateLabels, FastLabels)
    assert AccurateLabels.name == "accurate"
    assert AccurateLabels.quality == ("accurate",)
    assert FastLabels.name == "fast"
    assert FastLabels.quality == ("fast",)


@pytest.mark.parametrize(
    "position",
    [
        "TOP_LEFT",
        "TOP_CENTER",
        "TOP_RIGHT",
        "CENTER_LEFT",
        "CENTER",
        "CENTER_RIGHT",
        "BOTTOM_LEFT",
        "BOTTOM_CENTER",
        "BOTTOM_RIGHT",
        "CENTER_OF_MASS",
    ],
)
@pytest.mark.parametrize(
    "style",
    [
        {},
        {"text_scale": 0.35, "text_padding": 3, "text_color": "#b4349a"},
        {"text_thickness": 3, "border_radius": 20, "text_padding": 0},
        {"text_size_mode": "Automatic", "text_scale": 2.1, "border_radius": 6},
    ],
)
def test_fast_matches_accurate_at_clipped_overlapping_anchors(position, style):
    image = image_of(scene())
    source = image.tensor_image.clone()
    predictions = instance_detections(image)
    options = {"text_position": position, **style}

    accurate = _run(AccurateLabels(), image, predictions, **options)
    fast = _run(FastLabels(), image, predictions, **options)

    assert torch.equal(fast.tensor_image, accurate.tensor_image)
    assert torch.equal(image.tensor_image, source)
    assert fast.tensor_image.data_ptr() != image.tensor_image.data_ptr()
    assert fast.image_id == image.image_id
    assert fast.parent == image.parent
    assert fast.root == image.root


@pytest.mark.parametrize(
    "text",
    [
        "Class",
        "Confidence",
        "Class and Confidence",
        "Index",
        "Dimensions",
        "Area",
        "Tracker Id",
        "Area (mask)",
        "Area (converted)",
        "Time In Zone",
    ],
)
def test_fast_preserves_every_canonical_text_option(text):
    image = image_of(scene(180, 240))
    predictions = instance_detections(image)
    accurate = _run(AccurateLabels(), image, predictions, text=text)
    fast = _run(FastLabels(), image, predictions, text=text)

    assert torch.equal(fast.tensor_image, accurate.tensor_image)


def test_cached_sprite_is_reused_and_dynamic_styles_do_not_reuse_wrong_pixels(
    monkeypatch,
):
    image = image_of(scene(200, 300))
    predictions = detections(image, [(70, 90, 120, 130, 0, 0.9)])
    renderer = _sprites._render_label_sprite
    calls = []

    def render(**kwargs):
        calls.append(kwargs)
        return renderer(**kwargs)

    monkeypatch.setattr(_sprites, "_render_label_sprite", render)
    fast = FastLabels()
    first = _run(fast, image, predictions)
    second = _run(fast, image, predictions)
    assert len(calls) == 1
    assert torch.equal(first.tensor_image, second.tensor_image)

    for options in [
        {"text_color": "BLACK"},
        {"color_palette": "CUSTOM", "custom_colors": ["#995512"]},
        {"text_scale": 0.7, "text_padding": 15, "text_thickness": 2},
        {"text_position": "CENTER"},
        {"border_radius": 8},
    ]:
        actual = _run(fast, image, predictions, **options)
        expected = _run(AccurateLabels(), image, predictions, **options)
        assert torch.equal(actual.tensor_image, expected.tensor_image)


@pytest.mark.parametrize(
    "label,padding,radius",
    [("one\ntwo", 10, 0), ("", 10, 0), ("gyp", 0, 0), ("gyp", 0, 30)],
)
def test_scene_dependent_and_multiline_styles_use_reference_fallback(
    monkeypatch, label, padding, radius
):
    image = image_of(scene(160, 260))
    predictions = detections(
        image, [(70, 90, 120, 130, 0, 0.9)], rows=[{"label": label}]
    )
    calls = []
    reference = _sprites.paint_labels

    def paint(canvas, labels):
        calls.append(len(labels))
        reference(canvas, labels)

    monkeypatch.setattr(_sprites, "paint_labels", paint)
    options = {"text": "label", "text_padding": padding, "border_radius": radius}
    actual = _run(FastLabels(), image, predictions, **options)
    expected = _run(AccurateLabels(), image, predictions, **options)

    assert calls == [1]
    assert torch.equal(actual.tensor_image, expected.tensor_image)


def test_empty_detections_return_a_copy_without_painting(monkeypatch):
    image = image_of(scene())
    predictions = detections(image, [])

    def unexpected(*args, **kwargs):
        pytest.fail("empty predictions must not paint")

    monkeypatch.setattr(_sprites, "gpu_paste_label_sprites", unexpected)
    result = _run(FastLabels(), image, predictions)

    assert torch.equal(result.tensor_image, image.tensor_image)
    assert result.tensor_image.data_ptr() != image.tensor_image.data_ptr()


def test_cache_eviction_is_bounded_and_later_overlapping_labels_win(monkeypatch):
    monkeypatch.setattr(_sprites, "_MAX_SPRITES", 2)
    image = image_of(scene(200, 300))
    predictions = detections(image, [(70, 90, 120, 130, n, 0.9) for n in range(3)])
    fast = FastLabels()
    actual = _run(fast, image, predictions)
    expected = _run(AccurateLabels(), image, predictions)

    assert torch.equal(actual.tensor_image, expected.tensor_image)
    assert len(fast._sprites._sprites) == 2
    assert fast._sprites._retired == []


def test_clipping_changes_cache_variant_and_scene_colors_are_not_cached():
    fast = FastLabels()
    for anchor in [(-8, 5), (0, 30), (280, 15), (290, 190), (500, 400), (70, 90)]:
        for seed in [1, 5]:
            image = image_of(scene(200, 300, seed=seed))
            x, y = anchor
            predictions = detections(image, [(x, y, x + 30, y + 20, 0, 0.9)])
            actual = _run(fast, image, predictions)
            expected = _run(AccurateLabels(), image, predictions)
            assert torch.equal(actual.tensor_image, expected.tensor_image)


def test_evicted_sprites_survive_the_completion_boundary_even_on_fallback(monkeypatch):
    monkeypatch.setattr(_sprites, "_MAX_SPRITES", 1)
    image = image_of(scene(240, 320))
    predictions = detections(
        image,
        [(70, 110, 120, 150, n, 0.9) for n in range(3)],
        rows=[{"label": "car"}, {"label": "dog"}, {"label": "two\nlines"}],
    )
    fast = FastLabels()
    boundaries = []

    @contextmanager
    def work(device):
        assert device == image.device
        boundaries.append("entered")
        try:
            yield
        finally:
            assert fast._sprites._retired
            boundaries.append("completed")

    reference = _sprites.paint_labels

    def fallback(canvas, labels):
        assert boundaries == ["entered"]
        # Sprite preparation never partly paints before a later fallback.
        assert torch.equal(canvas, image.tensor_image)
        reference(canvas, labels)

    monkeypatch.setattr(fast._sprites._scope, "work", work)
    monkeypatch.setattr(_sprites, "paint_labels", fallback)
    result = _run(fast, image, predictions, text="label")
    expected = _run(AccurateLabels(), image, predictions, text="label")

    assert torch.equal(result.tensor_image, expected.tensor_image)
    assert boundaries == ["entered", "completed"]
    assert fast._sprites._retired == []


def test_frame_width_changes_do_not_reuse_flat_indices_for_the_wrong_image():
    fast = FastLabels()
    for width in [180, 300, 180]:
        image = image_of(scene(180, width))
        predictions = detections(image, [(70, 90, 120, 130, 0, 0.9)])
        actual = _run(fast, image, predictions)
        expected = _run(AccurateLabels(), image, predictions)
        assert torch.equal(actual.tensor_image, expected.tensor_image)
