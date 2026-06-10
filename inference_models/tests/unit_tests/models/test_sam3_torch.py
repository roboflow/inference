"""Unit tests for SAM3Torch interactive (PVS) prompt handling."""

import sys
from threading import RLock
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

from inference_models.models.sam3.entities import SAM3ImageEmbeddings
from inference_models.models.sam3.sam3_torch import SAM3Torch


def _build_model_with_mocked_predictor(num_prompts: int) -> SAM3Torch:
    model = SAM3Torch.__new__(SAM3Torch)
    model._predict_inst_lock = RLock()
    model._model = MagicMock()
    if num_prompts == 1:
        # the underlying predictor squeezes the prompt dimension for a single prompt
        predict_inst_result = (
            np.random.rand(3, 8, 8),
            np.array([0.5, 0.9, 0.7]),
            np.random.rand(3, 16, 16),
        )
    else:
        predict_inst_result = (
            np.random.rand(num_prompts, 3, 8, 8),
            np.tile([0.5, 0.9, 0.7], (num_prompts, 1)),
            np.random.rand(num_prompts, 3, 16, 16),
        )
    model._model.predict_inst.return_value = predict_inst_result
    return model


def _example_embeddings() -> SAM3ImageEmbeddings:
    return SAM3ImageEmbeddings(
        image_hash="image-hash",
        image_size_hw=(8, 8),
        embeddings={"state": MagicMock()},
    )


def test_predict_for_single_image_forwards_all_boxes():
    """Regression test: multiple box prompts must all reach the predictor -
    previously only the first box was forwarded."""
    # given
    model = _build_model_with_mocked_predictor(num_prompts=2)
    boxes = np.array([[0, 0, 4, 4], [2, 2, 6, 6]])

    # when
    prediction = model._predict_for_single_image(
        embeddings=_example_embeddings(),
        original_image_size=(8, 8),
        boxes=boxes,
        return_logits=True,
    )

    # then
    call_kwargs = model._model.predict_inst.call_args.kwargs
    assert call_kwargs["box"] == [[0, 0, 4, 4], [2, 2, 6, 6]]
    assert prediction.masks.shape == (2, 8, 8)
    assert prediction.scores.shape == (2,)


def test_predict_for_single_image_accepts_flat_single_box():
    # given
    model = _build_model_with_mocked_predictor(num_prompts=1)

    # when
    prediction = model._predict_for_single_image(
        embeddings=_example_embeddings(),
        original_image_size=(8, 8),
        boxes=[0, 0, 4, 4],
        return_logits=True,
    )

    # then
    call_kwargs = model._model.predict_inst.call_args.kwargs
    assert call_kwargs["box"] == [0, 0, 4, 4]
    assert prediction.masks.shape == (1, 8, 8)
    assert prediction.scores.shape == (1,)


def test_predict_for_single_image_selects_best_proposal_per_prompt():
    # given
    model = _build_model_with_mocked_predictor(num_prompts=2)

    # when
    prediction = model._predict_for_single_image(
        embeddings=_example_embeddings(),
        original_image_size=(8, 8),
        boxes=np.array([[0, 0, 4, 4], [2, 2, 6, 6]]),
        return_logits=True,
    )

    # then - proposal scores per prompt are [0.5, 0.9, 0.7], the best is kept
    assert torch.allclose(prediction.scores, torch.tensor([0.9, 0.9]).double())
