import json
from copy import deepcopy
from io import BytesIO
from time import perf_counter
from typing import Dict

import numpy as np
import pytest
import requests
import torch
from PIL import Image

from inference.core.entities.requests.sam2 import Sam2PromptSet, Sam2SegmentationRequest
from inference.core.entities.responses.sam2 import Sam2SegmentationPrediction
from inference.core.workflows.core_steps.common.utils import (
    convert_inference_detections_batch_to_sv_detections,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything2.v1 import (
    convert_sam2_segmentation_response_to_inference_instances_seg_response,
)

try:
    from inference.models.sam2 import SegmentAnything2
    from inference.models.sam2.segment_anything2 import (
        hash_prompt_set,
        maybe_load_low_res_logits_from_cache,
        turn_segmentation_results_into_rle_response,
    )
except ModuleNotFoundError:
    # SAM2 is not installed
    pass


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
        low_res_logits, sam2_small_truck_logits, atol=0.02
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
        truck_image, image_id=image_id, prompts=prompt, save_logits_to_cache=True
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


@pytest.mark.slow
def test_sam2_single_prompted_image_segmentation_mask_cache_changes_behavior(
    sam2_small_model: str,
    truck_image: np.ndarray,
    sam2_small_truck_mask_from_cached_logits: np.ndarray,
) -> None:
    # given
    model = SegmentAnything2(model_id=sam2_small_model)

    prompt = Sam2PromptSet(
        prompts=[{"points": [{"x": 1235, "y": 530, "positive": True}]}]
    )

    # when
    image_id = "truck"
    masks, scores, low_res_logits = model.segment_image(
        truck_image, image_id=image_id, prompts=prompt, save_logits_to_cache=True
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
        truck_image,
        prompts=prompt,
        mask_input=low_res_logits,
        load_logits_from_cache=True,
    )
    assert np.allclose(sam2_small_truck_mask_from_cached_logits, masks2, atol=0.01)


def convert_response_dict_to_sv_detections(image: Image, response_dict: Dict):
    class DummyImage:
        def __init__(self, image_array):
            self.numpy_image = image_array

    image_object = DummyImage(np.asarray(image))
    preds = convert_sam2_segmentation_response_to_inference_instances_seg_response(
        [Sam2SegmentationPrediction(**p) for p in response_dict["predictions"]],
        image_object,
        [],
        [],
        [],
        0,
    )
    preds = preds.model_dump(by_alias=True, exclude_none=True)
    return convert_inference_detections_batch_to_sv_detections([preds])[0]


def test_sam2_multi_poly(sam2_tiny_model: str, sam2_multipolygon_response: Dict):
    image_url = "https://media.roboflow.com/inference/seawithdock.jpeg"
    payload = {
        "image": {
            "type": "url",
            "value": image_url,
        },
        "image_id": "test",
    }
    payload["prompts"] = {
        "prompts": [{"points": [{"x": 58, "y": 379, "positive": True}]}]
    }
    payload["image_id"] = "test_seawithdock"
    model = SegmentAnything2(model_id=sam2_tiny_model)
    request = Sam2SegmentationRequest(**payload)
    response = model.infer_from_request(request)
    try:
        sam2_multipolygon_response = deepcopy(sam2_multipolygon_response)
        data = response.model_dump(by_alias=True, exclude_none=True)
        with open("test_multi.json", "w") as f:
            json.dump(data, f)
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        preds = convert_response_dict_to_sv_detections(image, data)
        ground_truth = convert_response_dict_to_sv_detections(
            image, sam2_multipolygon_response
        )
        preds_bool_mask = np.logical_or.reduce(preds.mask, axis=0)
        ground_truth_bool_mask = np.logical_or.reduce(ground_truth.mask, axis=0)
        iou = (
            np.logical_and(preds_bool_mask, ground_truth_bool_mask).sum()
            / np.logical_or(preds_bool_mask, ground_truth_bool_mask).sum()
        )
        assert iou > 0.99
        try:
            assert "predictions" in data
        except:
            print(f"Invalid response: {data}, expected 'predictions' in response")
            raise
    except Exception as e:
        raise e


def test_model_clears_cache_properly(sam2_small_model, truck_image):
    cache_size = 2
    model = SegmentAnything2(
        model_id=sam2_small_model,
        low_res_logits_cache_size=cache_size,
        embedding_cache_size=cache_size,
    )

    prompt = Sam2PromptSet(
        prompts=[{"points": [{"x": 1235, "y": 530, "positive": True}]}]
    )
    for i in range(5):
        masks, scores, low_res_logits = model.segment_image(
            truck_image,
            image_id=f"truck_{i}",
            prompts=prompt,
            save_logits_to_cache=True,
            load_logits_from_cache=True,
        )
        assert masks is not None
        assert scores is not None
        assert low_res_logits is not None


@pytest.mark.slow
def test_sam2_segment_with_rle_format(sam2_small_model: str, truck_image: np.ndarray):
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

    # Convert to RLE response
    t1 = perf_counter()
    resp = turn_segmentation_results_into_rle_response(
        masks=masks,
        scores=scores,
        mask_threshold=0.0,
        inference_start_timestamp=t1,
    )

    # then
    assert resp.time >= 0
    assert hasattr(resp, "predictions")
    assert len(resp.predictions) > 0

    pred = resp.predictions[0]
    assert pred.format == "rle"
    assert isinstance(pred.masks, dict)
    assert "size" in pred.masks
    assert "counts" in pred.masks
    assert isinstance(pred.masks["counts"], str)
