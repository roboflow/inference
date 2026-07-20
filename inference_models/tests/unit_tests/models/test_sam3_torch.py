"""Unit tests for SAM3Torch interactive (PVS) prompt handling."""

import importlib
import sys
from threading import RLock
from types import ModuleType
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from inference_models.models.sam3.entities import SAM3ImageEmbeddings

# Meta's `sam3` package ships only with the CUDA extras (torch-cu*), so it is
# not installed in the unit-test environment and the module under test cannot
# be imported without stubbing it.
_SAM3_PACKAGE_MODULES = [
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

SAM3_TORCH_MODULE = "inference_models.models.sam3.sam3_torch"


@pytest.fixture()
def sam3_torch_module() -> Generator[ModuleType, None, None]:
    """Imports the module under test with `sam3` stubbed, per test.

    `patch.dict` restores `sys.modules` to its pre-test state on exit, which
    drops both the stubs and the module imported against them - no mocked
    modules leak into other tests.
    """
    stubs = {name: MagicMock() for name in _SAM3_PACKAGE_MODULES}
    with patch.dict(sys.modules, stubs):
        sys.modules.pop(SAM3_TORCH_MODULE, None)
        yield importlib.import_module(SAM3_TORCH_MODULE)


def _build_model_with_mocked_predictor(sam3_torch_module: ModuleType, num_prompts: int):
    model = sam3_torch_module.SAM3Torch.__new__(sam3_torch_module.SAM3Torch)
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


def test_predict_for_single_image_forwards_all_boxes(
    sam3_torch_module: ModuleType,
) -> None:
    """Regression test: multiple box prompts must all reach the predictor -
    previously only the first box was forwarded."""
    # given
    model = _build_model_with_mocked_predictor(sam3_torch_module, num_prompts=2)
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


def test_predict_for_single_image_accepts_flat_single_box(
    sam3_torch_module: ModuleType,
) -> None:
    # given
    model = _build_model_with_mocked_predictor(sam3_torch_module, num_prompts=1)

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


def test_predict_for_single_image_selects_best_proposal_per_prompt(
    sam3_torch_module: ModuleType,
) -> None:
    # given
    model = _build_model_with_mocked_predictor(sam3_torch_module, num_prompts=2)

    # when
    prediction = model._predict_for_single_image(
        embeddings=_example_embeddings(),
        original_image_size=(8, 8),
        boxes=np.array([[0, 0, 4, 4], [2, 2, 6, 6]]),
        return_logits=True,
    )

    # then - proposal scores per prompt are [0.5, 0.9, 0.7], the best is kept
    assert torch.allclose(prediction.scores, torch.tensor([0.9, 0.9]).double())


def _example_batch_state(batch_size: int = 2) -> dict:
    return {
        "original_heights": [8, 16][:batch_size],
        "original_widths": [12, 20][:batch_size],
        "backbone_out": {
            "vision_features": torch.randn(batch_size, 3, 4, 4),
            "vision_pos_enc": [torch.randn(batch_size, 3, 4, 4)],
            "backbone_fpn": [
                torch.randn(batch_size, 3, 8, 8),
                torch.randn(batch_size, 3, 4, 4),
            ],
            "sam2_backbone_out": {
                "vision_features": torch.randn(batch_size, 3, 4, 4),
                "vision_pos_enc": [torch.randn(batch_size, 3, 4, 4)],
                "backbone_fpn": [torch.randn(batch_size, 3, 8, 8)],
            },
        },
    }


def test_split_batch_state_slices_nested_tensors_per_image(
    sam3_torch_module: ModuleType,
) -> None:
    # given
    batch_state = _example_batch_state(batch_size=2)

    # when
    states = sam3_torch_module.split_batch_state(batch_state)

    # then
    assert len(states) == 2
    assert states[0]["original_height"] == 8
    assert states[0]["original_width"] == 12
    assert states[1]["original_height"] == 16
    assert states[1]["original_width"] == 20
    backbone = batch_state["backbone_out"]
    assert torch.equal(
        states[1]["backbone_out"]["vision_features"],
        backbone["vision_features"][1:2],
    )
    assert torch.equal(
        states[0]["backbone_out"]["backbone_fpn"][1],
        backbone["backbone_fpn"][1][0:1],
    )
    assert torch.equal(
        states[1]["backbone_out"]["sam2_backbone_out"]["vision_pos_enc"][0],
        backbone["sam2_backbone_out"]["vision_pos_enc"][0][1:2],
    )


def test_split_batch_state_preserves_none_sam2_backbone(
    sam3_torch_module: ModuleType,
) -> None:
    # given
    batch_state = _example_batch_state(batch_size=2)
    batch_state["backbone_out"]["sam2_backbone_out"] = None

    # when
    states = sam3_torch_module.split_batch_state(batch_state)

    # then
    assert states[0]["backbone_out"]["sam2_backbone_out"] is None
    assert states[1]["backbone_out"]["sam2_backbone_out"] is None


def test_forward_image_embeddings_runs_single_batched_encoder_forward(
    sam3_torch_module: ModuleType,
) -> None:
    # given
    model = sam3_torch_module.SAM3Torch.__new__(sam3_torch_module.SAM3Torch)
    model._model = MagicMock()
    model._device = torch.device("cpu")
    model._model.backbone.forward_image.return_value = _example_batch_state(
        batch_size=2
    )["backbone_out"]
    decoder = model._model.inst_interactive_predictor.model.sam_mask_decoder
    decoder.conv_s0.side_effect = lambda t: t
    decoder.conv_s1.side_effect = lambda t: t
    processor = MagicMock()
    processor.transform.side_effect = lambda t: torch.zeros(3, 8, 8)

    # when
    with patch.object(sam3_torch_module, "Sam3Processor", return_value=processor):
        result = model._forward_image_embeddings(
            images=[
                np.zeros((8, 12, 3), dtype=np.uint8),
                np.zeros((16, 20, 3), dtype=np.uint8),
            ],
            image_hashes=["hash-1", "hash-2"],
            original_sizes=[(8, 12), (16, 20)],
        )

    # then - one stacked backbone forward, no per-image processor calls
    model._model.backbone.forward_image.assert_called_once()
    stacked = model._model.backbone.forward_image.call_args.args[0]
    assert stacked.shape[0] == 2
    processor.set_image.assert_not_called()
    processor.set_image_batch.assert_not_called()
    decoder.conv_s0.assert_called_once()
    decoder.conv_s1.assert_called_once()
    assert [r.image_hash for r in result] == ["hash-1", "hash-2"]
    assert result[0].image_size_hw == (8, 12)
    assert result[1].image_size_hw == (16, 20)
    assert result[0].embeddings["original_height"] == 8
    assert result[1].embeddings["original_width"] == 20
    assert result[0].embeddings["backbone_out"]["vision_features"].shape[0] == 1
