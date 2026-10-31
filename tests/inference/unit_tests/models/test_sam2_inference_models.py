"""Unit tests for the inference_models-backed SAM2 adapter."""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import torch

ADAPTER_MODULE = "inference.models.sam2.segment_anything2_inference_models"


def _build_sam2_stubs() -> dict:
    sam2 = ModuleType("sam2")
    sam2.__path__ = []
    build_sam = ModuleType("sam2.build_sam")
    build_sam.build_sam2 = MagicMock()
    sam2_image_predictor = ModuleType("sam2.sam2_image_predictor")
    sam2_image_predictor.SAM2ImagePredictor = MagicMock()

    modeling = ModuleType("sam2.modeling")
    modeling.__path__ = []
    sam2_base = ModuleType("sam2.modeling.sam2_base")
    sam2_base.SAM2Base = object

    utils = ModuleType("sam2.utils")
    utils.__path__ = []
    misc = ModuleType("sam2.utils.misc")
    transforms = ModuleType("sam2.utils.transforms")
    transforms.SAM2Transforms = MagicMock()

    sam2.build_sam = build_sam
    sam2.sam2_image_predictor = sam2_image_predictor
    sam2.modeling = modeling
    sam2.utils = utils
    modeling.sam2_base = sam2_base
    utils.misc = misc
    utils.transforms = transforms

    return {
        "sam2": sam2,
        "sam2.build_sam": build_sam,
        "sam2.sam2_image_predictor": sam2_image_predictor,
        "sam2.modeling": modeling,
        "sam2.modeling.sam2_base": sam2_base,
        "sam2.utils": utils,
        "sam2.utils.misc": misc,
        "sam2.utils.transforms": transforms,
    }


@pytest.fixture()
def adapter_module() -> Generator[ModuleType, None, None]:
    with patch.dict(sys.modules, _build_sam2_stubs()):
        sys.modules.pop(ADAPTER_MODULE, None)
        yield importlib.import_module(ADAPTER_MODULE)


def _single_prediction() -> SimpleNamespace:
    return SimpleNamespace(
        masks=torch.rand(1, 8, 8),
        scores=torch.tensor([0.7]),
        logits=torch.rand(1, 16, 16),
    )


def _build_adapter(adapter_module: ModuleType, prediction):
    adapter = adapter_module.InferenceModelsSAM2Adapter.__new__(
        adapter_module.InferenceModelsSAM2Adapter
    )
    adapter._model = MagicMock()
    adapter._model.segment_images.return_value = [prediction]
    return adapter


_CACHE_MISS_MESSAGE = (
    "Attempted to use SAM model segment_images(...) method providing "
    "`image_hashes` for which no embeddings were found in the cache."
)


def test_segment_image_with_cached_image_id_skips_preprocessing(
    adapter_module: ModuleType,
) -> None:
    adapter = _build_adapter(adapter_module, _single_prediction())
    adapter.preproc_image = MagicMock()

    adapter.segment_image(image=object(), image_id="abc")

    adapter.preproc_image.assert_not_called()
    adapter._model.segment_images.assert_called_once()
    call_kwargs = adapter._model.segment_images.call_args.kwargs
    assert call_kwargs["images"] is None
    assert call_kwargs["image_hashes"] == "abc"


def test_segment_image_falls_back_to_preprocessing_on_cache_miss(
    adapter_module: ModuleType,
) -> None:
    adapter = _build_adapter(adapter_module, _single_prediction())
    adapter._model.segment_images.side_effect = [
        adapter_module.ModelInputError(message=_CACHE_MISS_MESSAGE),
        [_single_prediction()],
    ]
    loaded_image = object()
    adapter.preproc_image = MagicMock(return_value=loaded_image)

    adapter.segment_image(image=object(), image_id="abc")

    adapter.preproc_image.assert_called_once()
    assert adapter._model.segment_images.call_count == 2
    call_kwargs = adapter._model.segment_images.call_args.kwargs
    assert call_kwargs["images"] is loaded_image
    assert call_kwargs["image_hashes"] == "abc"


def test_segment_image_propagates_non_cache_miss_input_error(
    adapter_module: ModuleType,
) -> None:
    adapter = _build_adapter(adapter_module, _single_prediction())
    adapter._model.segment_images.side_effect = adapter_module.ModelInputError(
        message="invalid point shape"
    )
    adapter.preproc_image = MagicMock()

    with pytest.raises(adapter_module.ModelInputError):
        adapter.segment_image(image=object(), image_id="abc")
    adapter.preproc_image.assert_not_called()
