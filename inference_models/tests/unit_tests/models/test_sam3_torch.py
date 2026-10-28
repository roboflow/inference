"""Unit tests for SAM3Torch interactive (PVS) prompt handling."""

import importlib
import inspect
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


def test_predict_for_single_image_binarizes_strictly_positive_logits_by_default(
    sam3_torch_module: ModuleType,
) -> None:
    # given
    model = _build_model_with_mocked_predictor(sam3_torch_module, num_prompts=1)
    logits = np.full((3, 8, 8), -1.0)
    logits[:, 0, 0] = 0.0
    logits[:, 1, 1] = 0.5
    model._model.predict_inst.return_value = (
        logits,
        np.array([0.5, 0.9, 0.7]),
        np.random.rand(3, 16, 16),
    )

    # when
    prediction = model._predict_for_single_image(
        embeddings=_example_embeddings(),
        original_image_size=(8, 8),
        boxes=[0, 0, 4, 4],
    )

    # then
    assert prediction.masks.dtype == torch.bool
    assert prediction.masks[0, 0, 0].item() is False
    assert prediction.masks[0, 1, 1].item() is True


def test_segmentation_methods_return_dense_masks_by_default(
    sam3_torch_module: ModuleType,
) -> None:
    visual = inspect.signature(sam3_torch_module.SAM3Torch.segment_with_visual_prompts)
    text = inspect.signature(sam3_torch_module.SAM3Torch.segment_with_text_prompts)

    assert visual.parameters["mask_format"].default == "dense"
    assert visual.parameters["mask_format"].kind is inspect.Parameter.KEYWORD_ONLY
    assert text.parameters["mask_format"].default == "dense"


def test_new_sam3_parameters_are_keyword_only(sam3_torch_module: ModuleType) -> None:
    embed = inspect.signature(sam3_torch_module.SAM3Torch.embed_images)

    assert embed.parameters["return_embeddings"].kind is inspect.Parameter.KEYWORD_ONLY
    assert embed.parameters["return_embeddings"].default is True


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


def test_from_pretrained_rejects_malformed_sam_configuration(
    sam3_torch_module: ModuleType, tmp_path
) -> None:
    # given - config present but missing the "version" key
    from inference_models.errors import CorruptedModelPackageError

    (tmp_path / "weights.pt").touch()
    (tmp_path / "bpe_simple_vocab_16e6.txt.gz").touch()
    (tmp_path / "sam_configuration.json").write_text('{"unexpected": "content"}')

    # when / then
    with pytest.raises(CorruptedModelPackageError):
        sam3_torch_module.SAM3Torch.from_pretrained(
            model_name_or_path=str(tmp_path),
            device=torch.device("cpu"),
        )


def _bgr_image() -> np.ndarray:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    return image


def _model_for_embedding(sam3_torch_module: ModuleType):
    model = sam3_torch_module.SAM3Torch.__new__(sam3_torch_module.SAM3Torch)
    model._sam3_allow_client_generated_hash_ids = False
    model._max_batch_size = 1
    model._model = MagicMock()
    return model


@pytest.mark.parametrize(
    "input_color_format, expected_red", [("bgr", 0), ("rgb", 255), (None, 255)]
)
def test_segment_with_text_prompts_honours_the_input_colour_order(
    sam3_torch_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    input_color_format,
    expected_red: int,
) -> None:
    model = sam3_torch_module.SAM3Torch.__new__(sam3_torch_module.SAM3Torch)
    model._model = MagicMock()
    model._transform = MagicMock()
    model._device = "cpu"
    image_datapoint = MagicMock()
    monkeypatch.setattr(sam3_torch_module, "Sam3ImageDP", image_datapoint)

    model.segment_with_text_prompts(
        images=_bgr_image(), prompts=[], input_color_format=input_color_format
    )

    pil_image = image_datapoint.call_args.kwargs["data"]
    assert np.asarray(pil_image)[0, 0, 0] == expected_red


@pytest.mark.parametrize(
    "input_color_format, expected_red", [("bgr", 0), ("rgb", 255), (None, 255)]
)
def test_segment_with_visual_prompts_embeds_the_image_in_its_colour_order(
    sam3_torch_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    input_color_format,
    expected_red: int,
) -> None:
    model = _model_for_embedding(sam3_torch_module)
    processor = MagicMock()
    monkeypatch.setattr(sam3_torch_module, "Sam3Processor", processor)
    monkeypatch.setattr(model, "_predict_for_single_image", MagicMock())

    model.segment_with_visual_prompts(
        images=_bgr_image(),
        use_embeddings_cache=False,
        input_color_format=input_color_format,
    )

    chw_image = processor.return_value.set_image.call_args.args[0]
    assert chw_image[0, 0, 0].item() == expected_red


def test_embeddings_cache_does_not_mix_colour_orders(
    sam3_torch_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference_models.models.sam3.cache import Sam3ImageEmbeddingsInMemoryCache

    model = _model_for_embedding(sam3_torch_module)
    model._device = "cpu"
    model._sam3_image_embeddings_cache = Sam3ImageEmbeddingsInMemoryCache.init(
        size_limit=10, send_to_cpu=False
    )
    processor = MagicMock()
    monkeypatch.setattr(sam3_torch_module, "Sam3Processor", processor)

    model.embed_images(images=_bgr_image(), input_color_format="rgb")
    model.embed_images(images=_bgr_image(), input_color_format="bgr")

    set_image_calls = processor.return_value.set_image.call_args_list
    assert len(set_image_calls) == 2
    assert set_image_calls[1].args[0][0, 0, 0].item() == 0


def test_bgr_embedding_normalizes_the_image_once(
    sam3_torch_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model_for_embedding(sam3_torch_module)
    processor = MagicMock()
    monkeypatch.setattr(sam3_torch_module, "Sam3Processor", processor)
    image = torch.zeros((3, 3, 5))
    image[0] = 0.004

    model.embed_images(
        images=image, use_embeddings_cache=False, input_color_format="bgr"
    )

    chw_image = processor.return_value.set_image.call_args.args[0]
    assert tuple(chw_image.shape) == (3, 3, 5)
    assert chw_image[2, 0, 0].item() == 1
    assert chw_image[0, 0, 0].item() == 0
