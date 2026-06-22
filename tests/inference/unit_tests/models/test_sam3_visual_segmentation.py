"""Unit tests for the SAM3 interactive segmentation adapter (PVS serving path)."""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Meta's `sam3` package is not installed in the unit-test environment, so the
# adapter module (which imports it transitively via inference_models) cannot
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

ADAPTER_MODULE = "inference.models.sam3.visual_segmentation_inference_models"
SAM3_TORCH_MODULE = "inference_models.models.sam3.sam3_torch"


@pytest.fixture()
def adapter_module() -> Generator[ModuleType, None, None]:
    """Imports the module under test with `sam3` stubbed, per test.

    `patch.dict` restores `sys.modules` to its pre-test state on exit, which
    drops both the stubs and the modules imported against them - no mocked
    modules leak into other tests.
    """
    stubs = {name: MagicMock() for name in _SAM3_PACKAGE_MODULES}
    with patch.dict(sys.modules, stubs):
        sys.modules.pop(ADAPTER_MODULE, None)
        sys.modules.pop(SAM3_TORCH_MODULE, None)
        yield importlib.import_module(ADAPTER_MODULE)


def _build_adapter(adapter_module: ModuleType, prediction):
    adapter = adapter_module.InferenceModelsSAM3InteractiveAdapter.__new__(
        adapter_module.InferenceModelsSAM3InteractiveAdapter
    )
    adapter._model = MagicMock()
    adapter._model.segment_with_visual_prompts.return_value = [prediction]
    return adapter


def test_segment_image_returns_one_prediction_per_prompt(
    adapter_module: ModuleType,
) -> None:
    """Regression test: a request with N prompts must return N masks - the adapter
    must not reduce across prompts (SAM3Torch already selects the best of the
    multimask proposals for each prompt)."""
    # given
    prediction = SimpleNamespace(
        masks=torch.rand(2, 8, 8),
        scores=torch.tensor([0.9, 0.8]),
        logits=torch.rand(2, 16, 16),
    )
    adapter = _build_adapter(adapter_module, prediction)

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


def test_segment_image_with_single_prompt_keeps_prompt_dimension(
    adapter_module: ModuleType,
) -> None:
    # given
    prediction = SimpleNamespace(
        masks=torch.rand(1, 8, 8),
        scores=torch.tensor([0.7]),
        logits=torch.rand(1, 16, 16),
    )
    adapter = _build_adapter(adapter_module, prediction)

    # when
    masks, scores, logits = adapter.segment_image(
        image=None,
        prompts={"prompts": [{"points": [{"x": 1, "y": 1, "positive": True}]}]},
    )

    # then
    assert masks.shape == (1, 8, 8)
    assert scores.shape == (1,)
    assert logits.shape == (1, 16, 16)


def test_segment_image_forwards_all_prompts_to_model(
    adapter_module: ModuleType,
) -> None:
    # given
    prediction = SimpleNamespace(
        masks=torch.rand(2, 8, 8),
        scores=torch.tensor([0.9, 0.8]),
        logits=torch.rand(2, 16, 16),
    )
    adapter = _build_adapter(adapter_module, prediction)

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


def _single_prediction() -> SimpleNamespace:
    return SimpleNamespace(
        masks=torch.rand(1, 8, 8),
        scores=torch.tensor([0.7]),
        logits=torch.rand(1, 16, 16),
    )


_CACHE_MISS_MESSAGE = (
    "Attempted to use SAM3 model segment_with_visual_prompts(...) method providing "
    "`image_hashes` for which no embeddings were found in the cache."
)


def test_segment_image_with_cached_image_id_skips_preprocessing(
    adapter_module: ModuleType,
) -> None:
    # given
    adapter = _build_adapter(adapter_module, _single_prediction())
    adapter.preproc_image = MagicMock()

    # when
    adapter.segment_image(image=object(), image_id="abc")

    # then
    adapter.preproc_image.assert_not_called()
    adapter._model.segment_with_visual_prompts.assert_called_once()
    call_kwargs = adapter._model.segment_with_visual_prompts.call_args.kwargs
    assert call_kwargs["images"] is None
    assert call_kwargs["image_hashes"] == "abc"


def test_segment_image_falls_back_to_preprocessing_on_cache_miss(
    adapter_module: ModuleType,
) -> None:
    # given
    adapter = _build_adapter(adapter_module, _single_prediction())
    adapter._model.segment_with_visual_prompts.side_effect = [
        adapter_module.ModelInputError(message=_CACHE_MISS_MESSAGE),
        [_single_prediction()],
    ]
    loaded_image = object()
    adapter.preproc_image = MagicMock(return_value=loaded_image)

    # when
    adapter.segment_image(image=object(), image_id="abc")

    # then
    adapter.preproc_image.assert_called_once()
    assert adapter._model.segment_with_visual_prompts.call_count == 2
    call_kwargs = adapter._model.segment_with_visual_prompts.call_args.kwargs
    assert call_kwargs["images"] is loaded_image
    assert call_kwargs["image_hashes"] == "abc"


def test_segment_image_propagates_non_cache_miss_input_error(
    adapter_module: ModuleType,
) -> None:
    # given
    adapter = _build_adapter(adapter_module, _single_prediction())
    adapter._model.segment_with_visual_prompts.side_effect = (
        adapter_module.ModelInputError(message="invalid point shape")
    )
    adapter.preproc_image = MagicMock()

    # when / then
    with pytest.raises(adapter_module.ModelInputError):
        adapter.segment_image(image=object(), image_id="abc")
    adapter.preproc_image.assert_not_called()


@pytest.mark.parametrize("send_to_cpu", [True, False])
def test_adapter_init_honors_cache_device_setting(
    adapter_module: ModuleType, send_to_cpu: bool
) -> None:
    # when
    with patch.object(
        adapter_module.Sam3ImageEmbeddingsInMemoryCache, "init"
    ) as emb_init, patch.object(
        adapter_module.Sam3LowResolutionMasksInMemoryCache, "init"
    ) as masks_init, patch.object(
        adapter_module.AutoModel, "from_pretrained"
    ), patch.object(
        adapter_module, "get_extra_weights_provider_headers"
    ), patch.object(
        adapter_module, "SAM3_INTERACTIVE_CACHE_SEND_TO_CPU", send_to_cpu
    ):
        adapter_module.InferenceModelsSAM3InteractiveAdapter(api_key="k")

    # then
    assert emb_init.call_args.kwargs["send_to_cpu"] is send_to_cpu
    assert masks_init.call_args.kwargs["send_to_cpu"] is send_to_cpu
