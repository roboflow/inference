"""Unit tests for the SAM3 interactive segmentation adapter (PVS serving path)."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


def _install_sam3_package_mocks():
    names = [
        "sam3",
        "sam3.eval",
        "sam3.eval.postprocessors",
        "sam3.model",
        "sam3.model.sam3_image_processor",
        "sam3.model.utils",
        "sam3.model.utils.misc",
        "sam3.train",
        "sam3.train.data",
        "sam3.train.data.collator",
        "sam3.train.data.sam3_image_dataset",
        "sam3.train.transforms",
        "sam3.train.transforms.basic_for_api",
    ]
    for name in names:
        sys.modules.setdefault(name, MagicMock())


_install_sam3_package_mocks()

import inference.models.sam3.visual_segmentation_inference_models as vs_module


def _build_adapter(prediction) -> "vs_module.InferenceModelsSAM3InteractiveAdapter":
    adapter = vs_module.InferenceModelsSAM3InteractiveAdapter.__new__(
        vs_module.InferenceModelsSAM3InteractiveAdapter
    )
    adapter._model = MagicMock()
    adapter._model.segment_with_visual_prompts.return_value = [prediction]
    return adapter


def test_segment_image_returns_one_prediction_per_prompt():
    """Regression test: a request with N prompts must return N masks - the adapter
    must not reduce across prompts (SAM3Torch already selects the best of the
    multimask proposals for each prompt)."""
    # given
    prediction = SimpleNamespace(
        masks=torch.rand(2, 8, 8),
        scores=torch.tensor([0.9, 0.8]),
        logits=torch.rand(2, 16, 16),
    )
    adapter = _build_adapter(prediction)

    # when
    masks, scores, logits = adapter.segment_image(
        image=None,
        prompts={
            "prompts": [
                {"points": [{"x": 1, "y": 1, "positive": True}]},
                {"points": [{"x": 5, "y": 5, "positive": True}]},
            ]
        },
    )

    # then
    assert masks.shape == (2, 8, 8)
    assert scores.shape == (2,)
    assert logits.shape == (2, 16, 16)
    np.testing.assert_allclose(scores, [0.9, 0.8], rtol=1e-6)


def test_segment_image_with_single_prompt_keeps_prompt_dimension():
    # given
    prediction = SimpleNamespace(
        masks=torch.rand(1, 8, 8),
        scores=torch.tensor([0.7]),
        logits=torch.rand(1, 16, 16),
    )
    adapter = _build_adapter(prediction)

    # when
    masks, scores, logits = adapter.segment_image(
        image=None,
        prompts={"prompts": [{"points": [{"x": 1, "y": 1, "positive": True}]}]},
    )

    # then
    assert masks.shape == (1, 8, 8)
    assert scores.shape == (1,)
    assert logits.shape == (1, 16, 16)


def test_segment_image_forwards_all_prompts_to_model():
    # given
    prediction = SimpleNamespace(
        masks=torch.rand(2, 8, 8),
        scores=torch.tensor([0.9, 0.8]),
        logits=torch.rand(2, 16, 16),
    )
    adapter = _build_adapter(prediction)

    # when
    _ = adapter.segment_image(
        image=None,
        prompts={
            "prompts": [
                {"box": {"x": 10, "y": 10, "width": 4, "height": 4}},
                {"box": {"x": 30, "y": 30, "width": 8, "height": 8}},
            ]
        },
    )

    # then
    call_kwargs = adapter._model.segment_with_visual_prompts.call_args.kwargs
    boxes = call_kwargs["boxes"]
    assert boxes.shape == (2, 4)
    np.testing.assert_allclose(boxes, [[8, 8, 12, 12], [26, 26, 34, 34]])
