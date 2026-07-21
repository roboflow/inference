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
    "sam3.model.data_misc",
    "sam3.model.sam3_image_processor",
    "sam3.model.utils",
    "sam3.model.utils.misc",
    "sam3.train",
    "sam3.train.masks_ops",
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


def test_from_pretrained_loads_package_without_sam_configuration(
    sam3_torch_module: ModuleType, tmp_path
) -> None:
    # given - fine-tuned packages ship without sam_configuration.json
    (tmp_path / "weights.pt").touch()
    (tmp_path / "bpe_simple_vocab_16e6.txt.gz").touch()

    # when
    model = sam3_torch_module.SAM3Torch.from_pretrained(
        model_name_or_path=str(tmp_path),
        device=torch.device("cpu"),
    )

    # then
    assert isinstance(model, sam3_torch_module.SAM3Torch)


def test_from_pretrained_accepts_supported_sam_configuration_version(
    sam3_torch_module: ModuleType, tmp_path
) -> None:
    # given
    (tmp_path / "weights.pt").touch()
    (tmp_path / "bpe_simple_vocab_16e6.txt.gz").touch()
    (tmp_path / "sam_configuration.json").write_text('{"version": "base"}')

    # when
    model = sam3_torch_module.SAM3Torch.from_pretrained(
        model_name_or_path=str(tmp_path),
        device=torch.device("cpu"),
    )

    # then
    assert isinstance(model, sam3_torch_module.SAM3Torch)


def test_from_pretrained_rejects_unsupported_sam_configuration_version(
    sam3_torch_module: ModuleType, tmp_path
) -> None:
    # given
    from inference_models.errors import CorruptedModelPackageError

    (tmp_path / "weights.pt").touch()
    (tmp_path / "bpe_simple_vocab_16e6.txt.gz").touch()
    (tmp_path / "sam_configuration.json").write_text('{"version": "unsupported"}')

    # when / then
    with pytest.raises(CorruptedModelPackageError):
        sam3_torch_module.SAM3Torch.from_pretrained(
            model_name_or_path=str(tmp_path),
            device=torch.device("cpu"),
        )
