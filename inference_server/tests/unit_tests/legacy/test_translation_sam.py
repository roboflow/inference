from types import SimpleNamespace

import numpy as np
import pytest

from inference_model_manager.hash_namespacing import (
    namespace_client_hash_id,
    tenant_namespace,
)
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


def test_embed_params_generate_namespaced_id_when_missing():
    req = Sam2EmbeddingRequest(image=IMG)
    params = build_interactive_segmentation_params("embed", req, "key")
    assert len(params["image_hashes"]) == 1
    generated = params["image_hashes"][0]
    assert generated.startswith(f"{tenant_namespace('key')}:")


def test_embed_generated_id_round_trips_to_segment():
    embed_req = Sam2EmbeddingRequest(image=IMG)
    embed_params = build_interactive_segmentation_params("embed", embed_req, "key")
    generated_hash = embed_params["image_hashes"][0]

    embeddings_obj = SimpleNamespace(image_hash=generated_hash)
    repacked = repack_interactive_segmentation_response(
        "embed", [embeddings_obj], embed_req, "key"
    )

    segment_req = Sam2SegmentationRequest(image=IMG, image_id=repacked.image_id)
    segment_params = build_interactive_segmentation_params(
        "segment", segment_req, "key"
    )

    assert segment_params["image_hashes"] == [generated_hash]


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


def _cache_request():
    return Sam2SegmentationRequest(
        image=IMG,
        prompts=Sam2PromptSet(
            prompts=[{"points": [{"x": 1, "y": 1, "positive": True}]}]
        ),
        load_logits_from_cache=True,
        save_logits_to_cache=True,
    )


@pytest.mark.parametrize(
    "flag,model_id,disabled",
    [
        ("DISABLE_SAM2_LOGITS_CACHE", "sam2/hiera_large", True),
        ("DISABLE_SAM2_LOGITS_CACHE", "sam2/hiera_large", False),
        ("DISABLE_SAM3_LOGITS_CACHE", "sam3/sam3_interactive", True),
        ("DISABLE_SAM3_LOGITS_CACHE", "sam3/sam3_interactive", False),
    ],
)
def test_logits_cache_is_gated_by_the_flag_of_the_model_family(
    monkeypatch, flag, model_id, disabled
):
    monkeypatch.setattr(f"inference_server.configuration.{flag}", disabled)
    for action in ("segment", "segment_with_visual_prompts"):
        params = build_interactive_segmentation_params(
            action, _cache_request(), None, model_id=model_id
        )
        assert params["load_from_mask_input_cache"] is (not disabled)
        assert params["save_to_mask_input_cache"] is (not disabled)


def test_sam2_logits_cache_ignores_the_sam3_flag(monkeypatch):
    monkeypatch.setattr(
        "inference_server.configuration.DISABLE_SAM2_LOGITS_CACHE", False
    )
    monkeypatch.setattr(
        "inference_server.configuration.DISABLE_SAM3_LOGITS_CACHE", True
    )
    params = build_interactive_segmentation_params(
        "segment", _cache_request(), None, model_id="sam2/hiera_large"
    )
    assert params["load_from_mask_input_cache"] is True
    assert params["save_to_mask_input_cache"] is True


def test_sam3_text_prompt_params_carry_the_max_detections_setting(monkeypatch):
    monkeypatch.setattr("inference_server.configuration.SAM3_MAX_DETECTIONS", 7)
    req = Sam3SegmentationRequest(image=IMG, prompts=[Sam3Prompt(text="cat")])
    params = build_interactive_segmentation_params(
        "segment_with_text_prompts", req, None
    )
    assert params["max_detections"] == 7


def test_sam3_max_detections_defaults_to_unlimited():
    req = Sam3SegmentationRequest(image=IMG, prompts=[Sam3Prompt(text="cat")])
    params = build_interactive_segmentation_params(
        "segment_with_text_prompts", req, None
    )
    assert params["max_detections"] == -1


@pytest.mark.parametrize("threshold", [None, 0.0])
def test_sam3_request_threshold_falls_back_to_default_for_none_and_zero(threshold):
    req = Sam3SegmentationRequest(
        image=IMG, prompts=[Sam3Prompt(text="cat")], output_prob_thresh=threshold
    )
    params = build_interactive_segmentation_params(
        "segment_with_text_prompts", req, None
    )
    assert params["output_prob_thresh"] == 0.5


@pytest.mark.parametrize("threshold", [None, 0.0])
def test_sam3_nms_default_threshold_falls_back_for_none_and_zero(threshold):
    masks = np.zeros((1, 4, 4), dtype=np.uint8)
    masks[0, 0:3, 0:3] = 1
    pred = [{"prompt_index": 0, "masks": masks, "scores": [0.3]}]
    req = Sam3SegmentationRequest(
        image=IMG,
        prompts=[Sam3Prompt(text="cat")],
        nms_iou_threshold=0.5,
        output_prob_thresh=threshold,
    )
    resp = repack_interactive_segmentation_response(
        "segment_with_text_prompts", pred, req, None
    )
    assert [len(r.predictions) for r in resp.prompt_results] == [0]


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
