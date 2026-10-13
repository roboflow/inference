"""The development YOLOv8 view adapter reproduces the model's dense results.

The adapter lives in ``development/workflows-2.0/24-segmentation-views``; these
tests skip when that directory is absent (for example in an installed wheel).
They use synthetic raw outputs. The cached-model case is
``integration/external/models/test_segmentation_view_model_adapter.py``.
"""

from types import SimpleNamespace

import pytest
import torch
from roboflow_workflows.v2.data.masks import mask_provider_of

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from tests.v2.support.data.segmentation_adapter import (
    SelectionSettings,
    YoloSegmentationViewAdapter,
    check_selection,
    dense_from_rows,
)

CLASSES = 3
ANCHORS = 64
GRID = 16
INPUT = 64


class _FakeYolo:
    """Model-shaped stand-in: class names, no recommended parameters, NMS."""

    class_names = ["a", "b", "c"]
    recommended_parameters = None
    _inference_config = SimpleNamespace(post_processing=SimpleNamespace(fused=False))


def _raw_output(seed: int):
    generator = torch.Generator().manual_seed(seed)
    centres = torch.rand((2, ANCHORS), generator=generator) * INPUT
    sizes = 4 + torch.rand((2, ANCHORS), generator=generator) * 24
    class_scores = torch.rand((CLASSES, ANCHORS), generator=generator)
    coefficients = torch.randn((32, ANCHORS), generator=generator)
    instances = torch.cat([centres, sizes, class_scores, coefficients])[None]
    protos = torch.randn((1, 32, GRID, GRID), generator=generator)

    return instances, protos


def _letterbox(original_hw, *, crop_xywh=None):
    height, width = original_hw
    crop_xywh = crop_xywh or (0, 0, width, height)
    crop_w, crop_h = crop_xywh[2:]
    scale = min(INPUT / crop_w, INPUT / crop_h)
    new_w, new_h = round(crop_w * scale), round(crop_h * scale)
    pad_x, pad_y = INPUT - new_w, INPUT - new_h
    metadata = PreProcessingMetadata(
        pad_left=pad_x // 2,
        pad_top=pad_y // 2,
        pad_right=pad_x - pad_x // 2,
        pad_bottom=pad_y - pad_y // 2,
        original_size=ImageDimensions(height, width),
        size_after_pre_processing=ImageDimensions(crop_h, crop_w),
        inference_size=ImageDimensions(INPUT, INPUT),
        scale_width=scale,
        scale_height=scale,
        static_crop_offset=StaticCropOffset(*crop_xywh),
    )

    return metadata


CASES = {
    "hd_letterbox": ((72, 128), None),
    "portrait_odd_pad": ((131, 67), None),
    "static_crop": ((90, 160), (20, 10, 120, 70)),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("smoothing", [True, False])
@pytest.mark.parametrize("seed", [0, 1])
def test_view_materializes_exactly_the_model_family_reference(
    case, smoothing, seed
) -> None:
    original_hw, crop = CASES[case]
    settings = SelectionSettings(confidence=0.5, masks_smoothing_enabled=smoothing)
    adapter = YoloSegmentationViewAdapter(_FakeYolo(), settings=settings)
    raw = _raw_output(seed)
    metadata = [_letterbox(original_hw, crop_xywh=crop)]

    selected = adapter.select(raw, metadata)
    view = adapter.view(selected)
    reference = dense_from_rows(selected, protos=raw[1], settings=settings)

    assert len(view) > 0
    assert view.score_type == ("probabilities" if smoothing else "logits")
    assert check_selection(view, reference) == {
        "rows_view": len(reference),
        "rows_reference": len(reference),
        "boxes_equal": True,
        "classes_equal": True,
        "confidence_equal": True,
        "masks_equal": True,
    }
    assert view.materialization_count == 1


def test_no_selected_rows_give_an_empty_view() -> None:
    settings = SelectionSettings(confidence=0.5)
    adapter = YoloSegmentationViewAdapter(_FakeYolo(), settings=settings)
    instances, protos = _raw_output(0)
    instances[:, 4 : 4 + CLASSES] = 0.0  # no class scores: nothing passes
    raw = (instances, protos)
    metadata = [_letterbox((72, 128))]

    selected = adapter.select(raw, metadata)
    view = adapter.view(selected)

    assert len(view) == 0
    assert tuple(view.full_res().mask.shape) == (0, 72, 128)
    assert check_selection(
        view, dense_from_rows(selected, protos=raw[1], settings=settings)
    )["masks_equal"]


def test_view_does_not_alias_the_raw_output() -> None:
    adapter = YoloSegmentationViewAdapter(
        _FakeYolo(), settings=SelectionSettings(confidence=0.5)
    )
    raw = _raw_output(0)
    view = adapter.view(adapter.select(raw, [_letterbox((72, 128))]))
    scores_before = view.scores.clone()

    raw[1].fill_(1000.0)  # a reusable output buffer refilled by the next frame

    assert torch.equal(view.scores, scores_before)


@pytest.mark.parametrize("case", sorted(CASES))
def test_grid_provider_selection_matches_the_reference_rows(case) -> None:
    original_hw, crop = CASES[case]
    settings = SelectionSettings(confidence=0.5)
    adapter = YoloSegmentationViewAdapter(_FakeYolo(), settings=settings)
    raw = _raw_output(0)
    selected = adapter.select(raw, [_letterbox(original_hw, crop_xywh=crop)])
    reference = dense_from_rows(selected, protos=raw[1], settings=settings)
    rows = [len(reference) - 1, 0, len(reference) - 1]

    provider = mask_provider_of(adapter.view(selected)).select(indices=rows)
    masks = provider.binary(fidelity="reference")

    assert masks.fidelity == "reference"
    assert masks.recipe.score_domain == provider.view.score_type == "probabilities"
    assert masks.recipe.threshold == provider.view.threshold
    assert torch.equal(masks.predictions.mask, reference.mask[rows])
    assert torch.equal(masks.predictions.class_id, reference.class_id[rows])
    assert provider.view.materialization_count == 1
