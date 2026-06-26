from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference_models.errors import ModelInputError
from inference_models.models.sam3.cache import (
    Sam3ImageEmbeddingsCacheNullObject,
    Sam3LowResolutionMasksCacheNullObject,
)

pytestmark = [pytest.mark.torch_models, pytest.mark.gpu_only]


def _make_model(allow_client_hashes: bool):
    # sam3 is GPU-only and absent from CPU/vino builds, so import the model class
    # inside the helper (these tests are gpu_only and run where sam3 is installed).
    from inference_models.models.sam3.sam3_torch import SAM3Torch

    return SAM3Torch(
        model=MagicMock(),
        transform=MagicMock(),
        device=torch.device("cpu"),
        max_batch_size=8,
        image_size=1008,
        sam3_image_embeddings_cache=Sam3ImageEmbeddingsCacheNullObject(),
        sam3_low_resolution_masks_cache=Sam3LowResolutionMasksCacheNullObject(),
        enable_inst_interactivity=True,
        sam3_allow_client_generated_hash_ids=allow_client_hashes,
    )


def _dummy_image() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_embed_images_with_hashes_raises_when_flag_disabled() -> None:
    model = _make_model(allow_client_hashes=False)
    with pytest.raises(ModelInputError):
        model.embed_images(images=_dummy_image(), image_hashes="some-hash")


def test_segment_with_visual_prompts_hash_only_raises_when_flag_disabled() -> None:
    model = _make_model(allow_client_hashes=False)
    with pytest.raises(ModelInputError):
        model.segment_with_visual_prompts(
            image_hashes="some-hash",
            point_coordinates=np.array([[[1, 1]]]),
            point_labels=np.array([[1]]),
        )


def test_embed_images_raises_when_hash_count_mismatches_images() -> None:
    model = _make_model(allow_client_hashes=True)
    with pytest.raises(ModelInputError):
        model.embed_images(
            images=[_dummy_image(), _dummy_image()],
            image_hashes=["only-one-hash"],
        )


def test_segment_with_visual_prompts_no_input_raises() -> None:
    model = _make_model(allow_client_hashes=True)
    with pytest.raises(ModelInputError):
        model.segment_with_visual_prompts(
            point_coordinates=np.array([[[1, 1]]]),
            point_labels=np.array([[1]]),
        )


def test_segment_with_visual_prompts_hash_with_cache_disabled_raises() -> None:
    model = _make_model(allow_client_hashes=True)
    with pytest.raises(ModelInputError):
        model.segment_with_visual_prompts(
            image_hashes="some-hash",
            use_embeddings_cache=False,
            point_coordinates=np.array([[[1, 1]]]),
            point_labels=np.array([[1]]),
        )


def test_serialize_prompt_single_point_does_not_raise() -> None:
    # Regression: a single-point visual prompt arrives shaped
    # (num_prompts, num_points, 2) = (1, 1, 2). serialize_prompt previously iterated
    # the leading prompt axis as if it were the point axis, so the lone "coordinate"
    # was [[x, y]] (length 1) and coord[1] raised IndexError, 500-ing SAM3 Smart Select
    # on a single click / tiny-box conversion.
    from inference_models.models.sam3.sam3_torch import serialize_prompt

    serialized = serialize_prompt(
        point_coordinates=np.array([[[10, 20]]]),
        point_labels=np.array([[1]]),
        boxes=None,
    )

    assert serialized == [
        {"points": [{"x": 10, "y": 20, "positive": True}], "box": None}
    ]


def test_serialize_prompt_multiple_points_flattens_in_order() -> None:
    from inference_models.models.sam3.sam3_torch import serialize_prompt

    serialized = serialize_prompt(
        point_coordinates=np.array([[[1, 2], [3, 4]]]),
        point_labels=np.array([[1, 0]]),
        boxes=None,
    )

    assert serialized == [
        {
            "points": [
                {"x": 1, "y": 2, "positive": True},
                {"x": 3, "y": 4, "positive": False},
            ],
            "box": None,
        }
    ]
