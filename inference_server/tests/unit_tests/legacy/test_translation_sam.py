from types import SimpleNamespace

import numpy as np
import pytest

from inference_model_manager.hash_namespacing import namespace_client_hash_id
from inference_server.legacy.entities import (
    Sam2EmbeddingRequest,
    Sam2PromptSet,
    Sam2SegmentationRequest,
    Sam3Prompt,
    Sam3SegmentationRequest,
    SamSegmentationRequest,
)
from inference_server.legacy.errors import LegacyHTTPError
from inference_server.legacy.translation import (
    build_interactive_segmentation_params,
    repack_interactive_segmentation_response,
)

IMG = {"type": "base64", "value": "x"}


def test_embed_params_namespace_client_hash():
    req = Sam2EmbeddingRequest(image=IMG, image_id="img-1")
    assert build_interactive_segmentation_params("embed", req, "key") == {
        "image_hashes": [namespace_client_hash_id("img-1", "key")]
    }
    assert (
        build_interactive_segmentation_params("embed_images", req, "key")[
            "return_embeddings"
        ]
        is False
    )


def test_sam1_segment_params():
    req = SamSegmentationRequest(
        image=IMG, point_coords=[[1, 2]], point_labels=[1], format="json"
    )
    assert build_interactive_segmentation_params("segment", req, None) == {
        "multi_mask_output": False,
        "point_coordinates": [[[1, 2]]],
        "point_labels": [[1]],
    }
    with pytest.raises(LegacyHTTPError):
        build_interactive_segmentation_params(
            "segment", SamSegmentationRequest(image=IMG, format="binary"), None
        )


def test_sam2_visual_prompt_params_pad_points():
    prompts = Sam2PromptSet(
        prompts=[
            {"points": [{"x": 1, "y": 1, "positive": True}]},
            {
                "points": [
                    {"x": 2, "y": 2, "positive": True},
                    {"x": 3, "y": 3, "positive": False},
                ]
            },
        ]
    )
    req = Sam2SegmentationRequest(image=IMG, prompts=prompts, multimask_output=False)
    params = build_interactive_segmentation_params(
        "segment_with_visual_prompts", req, None
    )
    assert params["multi_mask_output"] is False
    assert [len(p) for p in params["point_coordinates"][0]] == [2, 2]
    assert params["point_labels"][0][0][1] == -1


def test_sam3_text_prompt_params_take_min_threshold():
    req = Sam3SegmentationRequest(
        image=IMG,
        prompts=[
            Sam3Prompt(text="cat", output_prob_thresh=0.2),
            Sam3Prompt(text="dog"),
        ],
        output_prob_thresh=0.6,
    )
    params = build_interactive_segmentation_params(
        "segment_with_text_prompts", req, None
    )
    assert params["output_prob_thresh"] == 0.2
    assert [p["text"] for p in params["prompts"]] == ["cat", "dog"]


def test_embed_repack_strips_namespace():
    req = Sam2EmbeddingRequest(image=IMG)
    pred = SimpleNamespace(image_hash=namespace_client_hash_id("srv-hash", "key"))
    response = repack_interactive_segmentation_response("embed", [pred], req, "key")
    assert response.image_id == "srv-hash"


def test_sam2_segment_repack_picks_most_confident_mask():
    masks = np.zeros((1, 2, 4, 4))
    masks[0, 1, 1:3, 1:3] = 1.0
    pred = SimpleNamespace(masks=masks, scores=np.array([[0.2, 0.9]]))
    req = Sam2SegmentationRequest(
        image=IMG,
        prompts=Sam2PromptSet(
            prompts=[{"points": [{"x": 1, "y": 1, "positive": True}]}]
        ),
    )
    resp = repack_interactive_segmentation_response(
        "segment_with_visual_prompts", [pred], req, None
    )
    assert len(resp.predictions) == 1
    assert resp.predictions[0].confidence == 0.9
    assert resp.predictions[0].format == "polygon"


def test_sam3_text_repack_groups_by_prompt_and_applies_nms():
    a = np.zeros((4, 4), dtype=np.uint8)
    a[0:3, 0:3] = 1
    b = a.copy()
    pred = [
        {"prompt_index": 0, "masks": np.stack([a]), "scores": [0.9]},
        {"prompt_index": 1, "masks": np.stack([b]), "scores": [0.8]},
    ]
    req = Sam3SegmentationRequest(
        image=IMG,
        prompts=[Sam3Prompt(text="cat"), Sam3Prompt(text="dog")],
        nms_iou_threshold=0.5,
    )
    resp = repack_interactive_segmentation_response(
        "segment_with_text_prompts", pred, req, None
    )
    assert [len(r.predictions) for r in resp.prompt_results] == [1, 0]
    assert resp.prompt_results[1].echo.text == "dog"
