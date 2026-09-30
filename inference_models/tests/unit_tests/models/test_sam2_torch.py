"""Unit tests for SAM2Torch cache handling."""

import importlib
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

SAM2_TORCH_MODULE = "inference_models.models.sam2.sam2_torch"


def _build_sam2_stubs() -> dict:
    sam2 = ModuleType("sam2")
    sam2.__path__ = []
    build_sam = ModuleType("sam2.build_sam")
    build_sam.build_sam2 = MagicMock()

    modeling = ModuleType("sam2.modeling")
    modeling.__path__ = []
    sam2_base = ModuleType("sam2.modeling.sam2_base")
    sam2_base.SAM2Base = object

    utils = ModuleType("sam2.utils")
    utils.__path__ = []
    transforms = ModuleType("sam2.utils.transforms")
    transforms.SAM2Transforms = MagicMock()

    sam2.build_sam = build_sam
    sam2.modeling = modeling
    sam2.utils = utils
    modeling.sam2_base = sam2_base
    utils.transforms = transforms

    return {
        "sam2": sam2,
        "sam2.build_sam": build_sam,
        "sam2.modeling": modeling,
        "sam2.modeling.sam2_base": sam2_base,
        "sam2.utils": utils,
        "sam2.utils.transforms": transforms,
    }


@pytest.fixture(scope="module")
def sam2_torch_module() -> Generator[ModuleType, None, None]:
    with patch.dict(sys.modules, _build_sam2_stubs()):
        sys.modules.pop(SAM2_TORCH_MODULE, None)
        yield importlib.import_module(SAM2_TORCH_MODULE)


def test_find_prior_prompt_in_cache_returns_none_when_no_prompt_matches(
    sam2_torch_module: ModuleType,
) -> None:
    cached_mask = torch.ones(1, 4, 4)
    cache_entry = sam2_torch_module.SAM2MaskCacheEntry(
        prompt_hash="cached",
        serialized_prompt=[
            {"points": [{"x": 5, "y": 5, "positive": True}], "box": None}
        ],
        mask=cached_mask,
    )

    result = sam2_torch_module.find_prior_prompt_in_cache(
        serialized_prompt_hash="current",
        serialized_prompt=[
            {"points": [{"x": 10, "y": 10, "positive": True}], "box": None}
        ],
        matching_cache_entries=[cache_entry],
        device=torch.device("cpu"),
    )

    assert result is None


def test_find_prior_prompt_in_cache_returns_nearest_matching_prior_prompt(
    sam2_torch_module: ModuleType,
) -> None:
    cached_mask = torch.ones(1, 4, 4)
    cache_entry = sam2_torch_module.SAM2MaskCacheEntry(
        prompt_hash="cached",
        serialized_prompt=[
            {"points": [{"x": 5, "y": 5, "positive": True}], "box": None}
        ],
        mask=cached_mask,
    )

    result = sam2_torch_module.find_prior_prompt_in_cache(
        serialized_prompt_hash="current",
        serialized_prompt=[
            {
                "points": [
                    {"x": 5, "y": 5, "positive": True},
                    {"x": 10, "y": 10, "positive": False},
                ],
                "box": None,
            }
        ],
        matching_cache_entries=[cache_entry],
        device=torch.device("cpu"),
    )

    np.testing.assert_array_equal(result.numpy(), cached_mask.numpy())


def test_segment_images_moves_cached_embedding_to_model_device(
    sam2_torch_module: ModuleType,
) -> None:
    model = sam2_torch_module.SAM2Torch.__new__(sam2_torch_module.SAM2Torch)
    model._device = torch.device("cpu")
    model._lock = nullcontext()
    model._model = object()
    model._transform = object()
    model._sam2_allow_client_generated_hash_ids = True
    cached_embedding = MagicMock()
    moved_embedding = SimpleNamespace(image_hash="abc", image_size_hw=(8, 8))
    cached_embedding.to.return_value = moved_embedding
    model._sam2_image_embeddings_cache = SimpleNamespace(
        retrieve_embeddings=MagicMock(return_value=cached_embedding)
    )

    with patch.object(
        sam2_torch_module,
        "equalize_batch_size",
        return_value=(None, None, None, None),
    ), patch.object(
        sam2_torch_module,
        "pre_process_prompts",
        return_value=(None, None, None, None),
    ), patch.object(
        sam2_torch_module,
        "generate_model_inputs",
        return_value=[
            (moved_embedding, "abc", (8, 8), None, None, None, None),
        ],
    ), patch.object(
        sam2_torch_module,
        "predict_for_single_image",
        return_value=(
            torch.zeros(1, 8, 8),
            torch.tensor([1.0]),
            torch.zeros(1, 16, 16),
        ),
    ):
        predictions = model.segment_images(images=None, image_hashes="abc")

    cached_embedding.to.assert_called_once_with(device=torch.device("cpu"))
    assert len(predictions) == 1
