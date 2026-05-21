import gc

import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import ModelInputError
from inference_models.models.sam3.cache import (
    Sam3ImageEmbeddingsInMemoryCache,
    Sam3LowResolutionMasksInMemoryCache,
)
from inference_models.models.sam3.sam3_torch import SAM3Torch


@pytest.fixture(scope="module")
def sam3_model(sam3_package: str) -> SAM3Torch:
    model = SAM3Torch.from_pretrained(
        sam3_package,
        device=DEFAULT_DEVICE,
        sam3_image_embeddings_cache=Sam3ImageEmbeddingsInMemoryCache.init(
            size_limit=None, send_to_cpu=True
        ),
        sam3_low_resolution_masks_cache=Sam3LowResolutionMasksInMemoryCache.init(
            size_limit=None, send_to_cpu=True
        ),
        sam3_allow_client_generated_hash_ids=True,
    )
    yield model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def _free_gpu_after_test():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_embeddings_numpy(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray
) -> None:
    # given
    model = sam3_model

    # when
    results = model.embed_images(truck_image_numpy)

    # then
    assert len(results) == 1
    assert results[0].embeddings is not None
    assert results[0].image_size_hw == truck_image_numpy.shape[:2]


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_embeddings_torch(
    sam3_model: SAM3Torch,
    truck_image_torch: torch.Tensor
) -> None:
    # given
    model = sam3_model

    # when
    results = model.embed_images(truck_image_torch)

    # then
    assert len(results) == 1
    assert results[0].embeddings is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_embeddings_batch_numpy(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray
) -> None:
    # given
    model = sam3_model

    # when
    results = model.embed_images([truck_image_numpy, truck_image_numpy])

    # then
    assert len(results) == 2
    assert results[0].embeddings is not None
    assert results[1].embeddings is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_embeddings_caching(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray
) -> None:
    # given
    model = sam3_model

    # when - first call computes embeddings
    results1 = model.embed_images(truck_image_numpy)
    # second call should retrieve from cache
    results2 = model.embed_images(truck_image_numpy)

    # then - hashes should match (same image)
    assert results1[0].image_hash == results2[0].image_hash


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_images_without_prompting_numpy(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray
) -> None:
    # given
    model = sam3_model

    # when
    results = model.segment_with_visual_prompts(truck_image_numpy)

    # then
    assert len(results) == 1
    assert results[0].masks is not None
    assert results[0].scores is not None
    assert results[0].logits is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_images_without_prompting_batch_numpy(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray
) -> None:
    # given
    model = sam3_model

    # when
    results = model.segment_with_visual_prompts([truck_image_numpy, truck_image_numpy])

    # then
    assert len(results) == 2
    assert results[0].masks is not None
    assert results[1].masks is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_images_with_point_prompting(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = sam3_model
    input_point = np.array([[[500, 375]]])
    input_label = np.array([[1]])

    # when
    results = model.segment_with_visual_prompts(
        truck_image_numpy,
        point_coordinates=input_point,
        point_labels=input_label,
    )

    # then
    assert len(results) == 1
    assert results[0].masks is not None
    assert results[0].scores is not None
    # With a positive point prompt, we should get a mask with some area
    mask_sum = (
        results[0].masks.sum()
        if isinstance(results[0].masks, torch.Tensor)
        else results[0].masks.sum()
    )
    assert mask_sum > 0


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_images_with_multiple_points(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = sam3_model
    input_points = np.array([[[500, 375], [600, 400], [450, 350]]])
    input_labels = np.array([[1, 1, 1]])

    # when
    results = model.segment_with_visual_prompts(
        truck_image_numpy,
        point_coordinates=input_points,
        point_labels=input_labels,
    )

    # then
    assert len(results) == 1
    assert results[0].masks is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_images_with_embeddings(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray
) -> None:
    # given
    model = sam3_model
    input_point = np.array([[[500, 375]]])
    input_label = np.array([[1]])

    # when
    embeddings = model.embed_images(truck_image_numpy)
    results = model.segment_with_visual_prompts(
        embeddings=[embeddings[0], embeddings[0]],
        point_coordinates=input_point,
        point_labels=input_label,
    )

    # then
    assert len(results) == 2
    assert results[0].masks is not None
    assert results[1].masks is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_images_with_box_prompting(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = sam3_model
    input_box = np.array([425, 600, 700, 875])

    # when
    results = model.segment_with_visual_prompts(
        truck_image_numpy,
        boxes=input_box,
    )

    # then
    assert len(results) == 1
    assert results[0].masks is not None
    assert results[0].scores is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_images_with_box_prompting_and_embeddings(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = sam3_model
    input_box = np.array([425, 600, 700, 875])

    # when
    embeddings = model.embed_images(truck_image_numpy)
    results = model.segment_with_visual_prompts(
        embeddings=[embeddings[0], embeddings[0]],
        boxes=input_box,
    )

    # then
    assert len(results) == 2
    assert results[0].masks is not None
    assert results[1].masks is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_images_with_combined_prompting(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray
) -> None:
    # given
    model = sam3_model
    input_box = np.array([425, 600, 700, 875])
    input_point = np.array([[[575, 750]]])
    input_label = np.array([[0]])  # negative point

    # when
    results = model.segment_with_visual_prompts(
        truck_image_numpy,
        point_coordinates=input_point,
        point_labels=input_label,
        boxes=input_box,
    )

    # then
    assert len(results) == 1
    assert results[0].masks is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_images_with_mask_prompting(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray
) -> None:
    # given
    model = sam3_model
    input_point = np.array([[[500, 375]]])
    input_label = np.array([[1]])

    # when - first pass to get a mask
    first_results = model.segment_with_visual_prompts(
        images=truck_image_numpy,
        point_coordinates=input_point,
        point_labels=input_label,
    )
    # second pass using the logits as mask input
    second_results = model.segment_with_visual_prompts(
        images=truck_image_numpy,
        mask_input=first_results[0].logits,
        point_coordinates=input_point,
        point_labels=input_label,
    )

    # then
    assert len(second_results) == 1
    assert second_results[0].masks is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_images_raises_on_missing_input(
    sam3_model: SAM3Torch,
    ) -> None:
    # given
    model = sam3_model

    # when / then
    with pytest.raises(ModelInputError):
        _ = model.segment_with_visual_prompts()


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_images_with_misaligned_batch_sizes(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray
) -> None:
    # given
    model = sam3_model
    input_point = np.array([[[575, 750]]])
    input_label = np.array([[0]])

    # when / then - misaligned outer batch (1 image of coords vs 2 of labels)
    with pytest.raises(ModelInputError):
        _ = model.segment_with_visual_prompts(
            truck_image_numpy,
            point_coordinates=[input_point],
            point_labels=[input_label, input_label],
        )


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_with_text_single_prompt(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = sam3_model
    prompts = [{"text": "truck"}]

    # when
    results = model.segment_with_text_prompts(
        images=truck_image_numpy,
        prompts=prompts,
        output_prob_thresh=0.3,
    )

    # then
    assert len(results) == 1
    assert len(results[0]) == 1  # one prompt result
    assert results[0][0]["prompt_index"] == 0
    assert "masks" in results[0][0]
    assert "scores" in results[0][0]


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_with_text_multiple_prompts(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = sam3_model
    prompts = [
        {"text": "truck"},
        {"text": "wheel"},
        {"text": "sky"},
    ]

    # when
    results = model.segment_with_text_prompts(
        images=truck_image_numpy,
        prompts=prompts,
        output_prob_thresh=0.3,
    )

    # then
    assert len(results) == 1
    assert len(results[0]) == 3  # three prompt results
    for i, prompt_result in enumerate(results[0]):
        assert prompt_result["prompt_index"] == i


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_with_text_visual_prompt(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = sam3_model
    prompts = [
        {
            "text": "vehicle",
            "boxes": [[425, 600, 700, 875]],  # XYXY format
            "box_labels": [1],
        }
    ]

    # when
    results = model.segment_with_text_prompts(
        images=truck_image_numpy,
        prompts=prompts,
        output_prob_thresh=0.3,
    )

    # then
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0]["masks"] is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_with_text_batch_images(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = sam3_model
    prompts = [{"text": "truck"}]

    # when
    results = model.segment_with_text_prompts(
        images=[truck_image_numpy, truck_image_numpy],
        prompts=prompts,
        output_prob_thresh=0.3,
    )

    # then
    assert len(results) == 2  # two images
    assert len(results[0]) == 1  # one prompt per image
    assert len(results[1]) == 1


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_images_multi_mask_output(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = sam3_model
    input_point = np.array([[[500, 375]]])
    input_label = np.array([[1]])

    # when - with multi_mask_output=True (default)
    results_multi = model.segment_with_visual_prompts(
        truck_image_numpy,
        point_coordinates=input_point,
        point_labels=input_label,
        multi_mask_output=True,
    )

    # when - with multi_mask_output=False
    results_single = model.segment_with_visual_prompts(
        truck_image_numpy,
        point_coordinates=input_point,
        point_labels=input_label,
        multi_mask_output=False,
    )

    # then - both should return results
    assert len(results_multi) == 1
    assert len(results_single) == 1
    assert results_multi[0].masks is not None
    assert results_single[0].masks is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_images_return_logits(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = sam3_model
    input_point = np.array([[[500, 375]]])
    input_label = np.array([[1]])

    # when - with return_logits=True
    results_logits = model.segment_with_visual_prompts(
        truck_image_numpy,
        point_coordinates=input_point,
        point_labels=input_label,
        return_logits=True,
    )

    # when - with return_logits=False (default)
    results_binary = model.segment_with_visual_prompts(
        truck_image_numpy,
        point_coordinates=input_point,
        point_labels=input_label,
        return_logits=False,
    )

    # then
    assert len(results_logits) == 1
    assert len(results_binary) == 1
    # Logits should have floating point values, binary should be 0/1 or True/False
    assert results_logits[0].masks is not None
    assert results_binary[0].masks is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_caching_disabled(
    sam3_model: SAM3Torch,
    truck_image_numpy: np.ndarray
) -> None:
    # given
    model = sam3_model

    # when - with caching disabled
    results1 = model.embed_images(truck_image_numpy, use_embeddings_cache=False)
    results2 = model.embed_images(truck_image_numpy, use_embeddings_cache=False)

    # then - both should succeed
    assert len(results1) == 1
    assert len(results2) == 1
    assert results1[0].embeddings is not None
    assert results2[0].embeddings is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_embed_then_segment_with_client_hash(
    sam3_model: SAM3Torch, truck_image_numpy: np.ndarray
) -> None:
    """SAM2-style embed-then-segment flow: caller provides image_id at embed time
    and reuses it at segment time without re-sending image bytes."""
    # given
    model = sam3_model
    client_hash = "client-provided-hash-segment-flow"
    input_point = np.array([[[500, 375]]])
    input_label = np.array([[1]])

    # when - embed with client-provided hash
    embed_result = model.embed_images(
        truck_image_numpy, image_hashes=client_hash
    )

    # then - returned embedding carries the client hash
    assert len(embed_result) == 1
    assert embed_result[0].image_hash == client_hash

    # when - segment using only the hash (no image bytes)
    seg_result = model.segment_with_visual_prompts(
        image_hashes=client_hash,
        point_coordinates=input_point,
        point_labels=input_label,
    )

    # then - segmentation succeeds via cache lookup
    assert len(seg_result) == 1
    assert seg_result[0].masks is not None


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.gpu_only
def test_sam3_segment_with_unknown_hash_raises(
    sam3_model: SAM3Torch,
) -> None:
    """segment_with_visual_prompts with a hash never seen by the cache must raise."""
    # given
    model = sam3_model
    input_point = np.array([[[500, 375]]])
    input_label = np.array([[1]])

    # when / then
    with pytest.raises(ModelInputError):
        model.segment_with_visual_prompts(
            image_hashes="never-embedded-hash",
            point_coordinates=input_point,
            point_labels=input_label,
        )
