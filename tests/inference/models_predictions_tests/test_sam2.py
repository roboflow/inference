import numpy as np
import pytest
import torch

from inference.core.entities.requests.sam2 import Sam2PromptSet
from inference.models.sam2.segment_anything2 import (
    hash_prompt_set,
    maybe_load_low_res_logits_from_cache,
)
from inference.models.sam2 import SegmentAnything2


@pytest.mark.slow
def test_sam2_single_image_embedding(
    sam2_small_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    model = SegmentAnything2(model_id=sam2_small_model)

    # when
    embedding, img_shape, id_ = model.embed_image(example_image)

    # then
    drift = np.abs(
        torch.sum(embedding["image_embed"]).cpu().detach().numpy() - 25155.9336
    )
    assert drift < 1, "embedding sum doesnt drift/change"
    assert img_shape == (427, 640), "Image shape must match the expected shape"


@pytest.mark.slow
def test_sam2_single_prompted_image_segmentation(
    sam2_small_model: str,
    truck_image: np.ndarray,
    sam2_small_truck_logits: np.ndarray,
) -> None:
    # given
    model = SegmentAnything2(model_id=sam2_small_model)

    prompt = Sam2PromptSet(
        prompts=[{"points": [{"x": 500, "y": 375, "positive": True}]}]
    )

    # when
    masks, scores, low_res_logits = model.segment_image(
        truck_image,
        prompts=prompt,
    )

    # then
    score_drift = np.abs(scores[0] - 0.9426716566085815)
    assert np.allclose(
        low_res_logits, sam2_small_truck_logits, atol=0.01
    ), "logits mask is as expected"
    assert score_drift < 0.01, "score doesnt drift/change"


@pytest.mark.slow
def test_sam2_single_prompted_image_segmentation_uses_cache(
    sam2_small_model: str, truck_image: np.ndarray
) -> None:
    # given
    model = SegmentAnything2(model_id=sam2_small_model)

    prompt = Sam2PromptSet(
        prompts=[{"points": [{"x": 500, "y": 375, "positive": True}]}]
    )

    # when
    embedding, img_shape, id_ = model.embed_image(truck_image)
    masks, scores, low_res_logits = model.segment_image(truck_image, prompts=prompt)

    # then
    assert id_ in model.embedding_cache, "embedding is cached"


@pytest.mark.slow
def test_sam2_single_prompted_image_segmentation_mask_cache_works(
    sam2_small_model: str, truck_image: np.ndarray
) -> None:
    # given
    model = SegmentAnything2(model_id=sam2_small_model)

    prompt = Sam2PromptSet(
        prompts=[{"points": [{"x": 1235, "y": 530, "positive": True}]}]
    )

    # when
    image_id = "truck"
    masks, scores, low_res_logits = model.segment_image(
        truck_image, image_id=image_id, prompts=prompt
    )
    assert hash_prompt_set(image_id, prompt) in model.low_res_logits_cache

    prompt = Sam2PromptSet(
        prompts=[
            {
                "points": [
                    {"x": 1235, "y": 530, "positive": True},
                    {"x": 10, "y": 500, "positive": False},
                ]
            }
        ]
    )
    assert (
        maybe_load_low_res_logits_from_cache(
            image_id, prompt, model.low_res_logits_cache
        )
        is not None
    )
    masks2, scores2, low_res_logits2 = model.segment_image(
        truck_image, prompts=prompt, mask_input=low_res_logits
    )

    # then
    assert True, "doesnt crash when passing mask_input"
