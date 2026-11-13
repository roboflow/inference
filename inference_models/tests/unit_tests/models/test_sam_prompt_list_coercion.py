"""Unit tests for SAM/SAM2 prompt coercion from plain nested lists."""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

SAM_TORCH_MODULE = "inference_models.models.sam.sam_torch"
SAM2_TORCH_MODULE = "inference_models.models.sam2.sam2_torch"


def _build_segment_anything_stubs() -> dict:
    segment_anything = ModuleType("segment_anything")
    segment_anything.__path__ = []
    segment_anything.sam_model_registry = {}

    modeling = ModuleType("segment_anything.modeling")
    modeling.Sam = object

    utils = ModuleType("segment_anything.utils")
    utils.__path__ = []
    transforms = ModuleType("segment_anything.utils.transforms")
    transforms.ResizeLongestSide = MagicMock()

    segment_anything.modeling = modeling
    segment_anything.utils = utils
    utils.transforms = transforms

    return {
        "segment_anything": segment_anything,
        "segment_anything.modeling": modeling,
        "segment_anything.utils": utils,
        "segment_anything.utils.transforms": transforms,
    }


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
def sam_torch_module() -> Generator[ModuleType, None, None]:
    with patch.dict(sys.modules, _build_segment_anything_stubs()):
        sys.modules.pop(SAM_TORCH_MODULE, None)
        yield importlib.import_module(SAM_TORCH_MODULE)


@pytest.fixture(scope="module")
def sam2_torch_module() -> Generator[ModuleType, None, None]:
    with patch.dict(sys.modules, _build_sam2_stubs()):
        sys.modules.pop(SAM2_TORCH_MODULE, None)
        yield importlib.import_module(SAM2_TORCH_MODULE)


def test_sam_equalize_batch_size_coerces_list_point_prompts(
    sam_torch_module: ModuleType,
) -> None:
    coords, labels, boxes, mask_input = sam_torch_module.equalize_batch_size(
        embeddings_batch_size=1,
        point_coordinates=[[[0.0, 0.0]]],
        point_labels=[[-1]],
        boxes=None,
        mask_input=None,
    )

    assert isinstance(coords[0], np.ndarray)
    assert coords[0].shape == (1, 2)
    assert isinstance(labels[0], np.ndarray)
    assert labels[0].shape == (1,)
    assert boxes is None
    assert mask_input is None


def test_sam_equalize_batch_size_broadcasts_single_list_prompt(
    sam_torch_module: ModuleType,
) -> None:
    coords, labels, _, _ = sam_torch_module.equalize_batch_size(
        embeddings_batch_size=2,
        point_coordinates=[[[10.0, 20.0]]],
        point_labels=[[1]],
        boxes=None,
        mask_input=None,
    )

    assert len(coords) == 2
    assert len(labels) == 2
    assert all(isinstance(c, np.ndarray) for c in coords)
    np.testing.assert_array_equal(coords[0], coords[1])


def test_sam_equalize_batch_size_coerces_list_mask_input(
    sam_torch_module: ModuleType,
) -> None:
    _, _, _, mask_input = sam_torch_module.equalize_batch_size(
        embeddings_batch_size=1,
        point_coordinates=None,
        point_labels=None,
        boxes=None,
        mask_input=[[[0.0, 1.0], [1.0, 0.0]]],
    )

    assert isinstance(mask_input[0], np.ndarray)
    assert mask_input[0].shape == (1, 2, 2)


def test_sam_equalize_batch_size_keeps_arrays_untouched(
    sam_torch_module: ModuleType,
) -> None:
    coords_array = np.array([[5.0, 6.0]])
    labels_array = np.array([1])

    coords, labels, _, _ = sam_torch_module.equalize_batch_size(
        embeddings_batch_size=1,
        point_coordinates=[coords_array],
        point_labels=[labels_array],
        boxes=None,
        mask_input=None,
    )

    assert coords[0] is coords_array
    assert labels[0] is labels_array


def test_sam_pre_process_prompts_accepts_coerced_list_prompts(
    sam_torch_module: ModuleType,
) -> None:
    transform = SimpleNamespace(
        apply_coords_torch=lambda point_coords, image_shape: point_coords
    )
    coords, labels, boxes, mask_input = sam_torch_module.equalize_batch_size(
        embeddings_batch_size=1,
        point_coordinates=[[[0.0, 0.0]]],
        point_labels=[[-1]],
        boxes=None,
        mask_input=None,
    )

    coords, labels, boxes, mask_input = sam_torch_module.pre_process_prompts(
        point_coordinates=coords,
        point_labels=labels,
        boxes=boxes,
        mask_input=mask_input,
        device=torch.device("cpu"),
        transform=transform,
        original_image_sizes=[(100, 200)],
    )

    assert isinstance(coords[0], torch.Tensor)
    assert tuple(coords[0].shape) == (1, 1, 2)
    assert isinstance(labels[0], torch.Tensor)
    assert tuple(labels[0].shape) == (1, 1)


class _PromptEncoderReached(Exception):
    pass


def test_sam_predict_receives_float32_mask_from_list_input(
    sam_torch_module: ModuleType,
) -> None:
    captured = {}

    class _FakeModel:
        def prompt_encoder(self, points, boxes, masks):
            captured["masks"] = masks
            raise _PromptEncoderReached()

    _, _, _, mask_input = sam_torch_module.equalize_batch_size(
        embeddings_batch_size=1,
        point_coordinates=None,
        point_labels=None,
        boxes=None,
        mask_input=[[[0, 1], [1, 0]]],
    )
    _, _, _, mask_input = sam_torch_module.pre_process_prompts(
        point_coordinates=None,
        point_labels=None,
        boxes=None,
        mask_input=mask_input,
        device=torch.device("cpu"),
        transform=SimpleNamespace(),
        original_image_sizes=[(100, 200)],
    )

    with pytest.raises(_PromptEncoderReached):
        sam_torch_module.predict_for_single_image(
            model=_FakeModel(),
            transform=SimpleNamespace(),
            embeddings=torch.zeros(4, 2, 2),
            original_image_size=(100, 200),
            point_coordinates=None,
            point_labels=None,
            boxes=None,
            mask_input=mask_input[0],
        )

    assert captured["masks"].dtype == torch.float32
    assert tuple(captured["masks"].shape) == (1, 1, 2, 2)


def test_sam2_pre_process_prompts_casts_list_mask_input_to_float32(
    sam2_torch_module: ModuleType,
) -> None:
    _, _, _, mask_input = sam2_torch_module.equalize_batch_size(
        embeddings_batch_size=1,
        point_coordinates=None,
        point_labels=None,
        boxes=None,
        mask_input=[[[0, 1], [1, 0]]],
    )

    _, _, _, mask_input = sam2_torch_module.pre_process_prompts(
        point_coordinates=None,
        point_labels=None,
        boxes=None,
        mask_input=mask_input,
        device=torch.device("cpu"),
        transform=SimpleNamespace(),
        original_image_sizes=[(100, 200)],
    )

    assert mask_input[0].dtype == torch.float32


def test_sam2_embed_images_serves_second_call_from_cache_by_client_hash(
    sam2_torch_module: ModuleType,
) -> None:
    cache_module = importlib.import_module("inference_models.models.sam2.cache")
    model = sam2_torch_module.SAM2Torch.__new__(sam2_torch_module.SAM2Torch)
    model._device = torch.device("cpu")
    model._sam2_allow_client_generated_hash_ids = True
    model._sam2_image_embeddings_cache = (
        cache_module.Sam2ImageEmbeddingsInMemoryCache.init(size_limit=10)
    )
    forward_calls = []

    def fake_forward(model_input_images, image_hashes, original_image_sizes, **kwargs):
        forward_calls.append(list(image_hashes))
        return [
            sam2_torch_module.SAM2ImageEmbeddings(
                image_hash=image_hash,
                image_size_hw=image_size,
                embeddings=torch.zeros(1, 4, 2, 2),
                high_resolution_features=[torch.zeros(1, 2, 4, 4)],
            )
            for image_hash, image_size in zip(image_hashes, original_image_sizes)
        ]

    model.forward_image_embeddings = fake_forward
    with patch.object(
        sam2_torch_module.SAM2Torch,
        "pre_process_images",
        return_value=([torch.zeros(3, 8, 8)], ["content-sha"], [(8, 8)]),
    ):
        first = model.embed_images(
            images=np.zeros((8, 8, 3), dtype=np.uint8), image_hashes=["client-id"]
        )
        second = model.embed_images(
            images=np.zeros((8, 8, 3), dtype=np.uint8), image_hashes=["client-id"]
        )

    assert forward_calls == [["client-id"]]
    assert first[0].image_hash == "client-id"
    assert second[0].image_hash == "client-id"


def test_sam2_equalize_batch_size_coerces_sentinel_list_prompts(
    sam2_torch_module: ModuleType,
) -> None:
    coords, labels, boxes, mask_input = sam2_torch_module.equalize_batch_size(
        embeddings_batch_size=1,
        point_coordinates=[[[0, 0]]],
        point_labels=[[-1]],
        boxes=None,
        mask_input=None,
    )

    assert isinstance(coords[0], np.ndarray)
    assert coords[0].shape == (1, 2)
    assert isinstance(labels[0], np.ndarray)
    assert labels[0].shape == (1,)
    assert boxes is None
    assert mask_input is None


def test_sam2_equalize_batch_size_coerces_multi_prompt_lists(
    sam2_torch_module: ModuleType,
) -> None:
    coords, labels, _, _ = sam2_torch_module.equalize_batch_size(
        embeddings_batch_size=1,
        point_coordinates=[[[[10.0, 20.0], [30.0, 40.0]]]],
        point_labels=[[[1, 0]]],
        boxes=None,
        mask_input=None,
    )

    assert isinstance(coords[0], np.ndarray)
    assert coords[0].shape == (1, 2, 2)
    assert isinstance(labels[0], np.ndarray)
    assert labels[0].shape == (1, 2)


def test_sam2_equalize_batch_size_coerces_list_boxes(
    sam2_torch_module: ModuleType,
) -> None:
    _, _, boxes, _ = sam2_torch_module.equalize_batch_size(
        embeddings_batch_size=1,
        point_coordinates=None,
        point_labels=None,
        boxes=[[[1.0, 2.0, 3.0, 4.0]]],
        mask_input=None,
    )

    assert isinstance(boxes[0], np.ndarray)
    assert boxes[0].shape == (1, 4)


def test_sam2_equalize_batch_size_keeps_tensors_untouched(
    sam2_torch_module: ModuleType,
) -> None:
    coords_tensor = torch.tensor([[5.0, 6.0]])
    labels_tensor = torch.tensor([1])

    coords, labels, _, _ = sam2_torch_module.equalize_batch_size(
        embeddings_batch_size=1,
        point_coordinates=[coords_tensor],
        point_labels=[labels_tensor],
        boxes=None,
        mask_input=None,
    )

    assert coords[0] is coords_tensor
    assert labels[0] is labels_tensor


def test_sam2_equalize_batch_size_rejects_inconsistent_list_prompts(
    sam2_torch_module: ModuleType,
) -> None:
    errors_module = importlib.import_module("inference_models.errors")

    with pytest.raises(errors_module.ModelInputError):
        sam2_torch_module.equalize_batch_size(
            embeddings_batch_size=1,
            point_coordinates=[[[0, 0]]],
            point_labels=[[1, 0]],
            boxes=None,
            mask_input=None,
        )


def test_sam2_pre_process_prompts_accepts_coerced_list_prompts(
    sam2_torch_module: ModuleType,
) -> None:
    transform = SimpleNamespace(
        transform_coords=lambda coords, normalize, orig_hw: coords,
        transform_boxes=lambda boxes, normalize, orig_hw: boxes.reshape(-1, 2, 2),
    )
    coords, labels, boxes, mask_input = sam2_torch_module.equalize_batch_size(
        embeddings_batch_size=1,
        point_coordinates=[[[0, 0]]],
        point_labels=[[-1]],
        boxes=[[[1.0, 2.0, 3.0, 4.0]]],
        mask_input=None,
    )

    coords, labels, boxes, mask_input = sam2_torch_module.pre_process_prompts(
        point_coordinates=coords,
        point_labels=labels,
        boxes=boxes,
        mask_input=mask_input,
        device=torch.device("cpu"),
        transform=transform,
        original_image_sizes=[(100, 200)],
    )

    assert isinstance(coords[0], torch.Tensor)
    assert tuple(coords[0].shape) == (1, 1, 2)
    assert isinstance(labels[0], torch.Tensor)
    assert tuple(labels[0].shape) == (1, 1)
    assert isinstance(boxes[0], torch.Tensor)
    assert tuple(boxes[0].shape) == (1, 2, 2)
