"""Tenant namespacing of client-provided hash ids on the MMP adapter path."""

import base64
import hashlib
from types import SimpleNamespace

from inference.core.entities.requests.sam import (
    SamEmbeddingRequest,
    SamSegmentationRequest,
)
from inference.core.entities.requests.sam2 import (
    Sam2EmbeddingRequest,
    Sam2SegmentationRequest,
)
from inference.core.managers.mmp_translation import (
    _build_interactive_segmentation_params,
    _repack_sam_embeddings,
)
from inference_model_manager.hash_namespacing import (
    namespace_client_hash_id,
    tenant_namespace,
)

IMAGE = {"type": "base64", "value": base64.b64encode(b"fake").decode()}


def test_namespace_scheme_is_sha256_digest_prefix():
    expected_digest = hashlib.sha256(b"key-a").hexdigest()[:16]

    assert namespace_client_hash_id("img-1", "key-a") == f"{expected_digest}:img-1"


def test_missing_api_key_namespaces_under_anonymous():
    expected_digest = hashlib.sha256(b"anonymous").hexdigest()[:16]

    assert namespace_client_hash_id("img-1", None) == f"{expected_digest}:img-1"


def test_sam2_embed_and_segment_share_the_same_tenant_namespaced_hash():
    embed_request = Sam2EmbeddingRequest(
        api_key="key-a", image=IMAGE, image_id="img-1"
    )
    segment_request = Sam2SegmentationRequest(
        api_key="key-a", image=IMAGE, image_id="img-1"
    )

    embed_params = _build_interactive_segmentation_params("embed", embed_request)
    segment_params = _build_interactive_segmentation_params(
        "segment", segment_request
    )

    expected = [namespace_client_hash_id("img-1", "key-a")]
    assert embed_params["image_hashes"] == expected
    assert segment_params["image_hashes"] == expected
    assert expected != ["img-1"]


def test_sam1_embed_and_segment_share_the_same_tenant_namespaced_hash():
    embed_request = SamEmbeddingRequest(
        api_key="key-a", image=IMAGE, image_id="img-1", format="json"
    )
    segment_request = SamSegmentationRequest(
        api_key="key-a",
        image=None,
        image_id="img-1",
        point_coords=[[1, 2]],
        point_labels=[1],
    )

    embed_params = _build_interactive_segmentation_params("embed", embed_request)
    segment_params = _build_interactive_segmentation_params(
        "segment", segment_request
    )

    expected = [namespace_client_hash_id("img-1", "key-a")]
    assert embed_params["image_hashes"] == expected
    assert segment_params["image_hashes"] == expected


def test_different_api_keys_namespace_the_same_id_to_disjoint_hashes():
    request_a = Sam2EmbeddingRequest(api_key="key-a", image=IMAGE, image_id="img-1")
    request_b = Sam2EmbeddingRequest(api_key="key-b", image=IMAGE, image_id="img-1")

    params_a = _build_interactive_segmentation_params("embed", request_a)
    params_b = _build_interactive_segmentation_params("embed", request_b)

    assert params_a["image_hashes"] != params_b["image_hashes"]
    assert params_a["image_hashes"] == [f"{tenant_namespace('key-a')}:img-1"]
    assert params_b["image_hashes"] == [f"{tenant_namespace('key-b')}:img-1"]


def test_sam3_visual_segment_hash_matches_embed_namespace():
    embed_request = Sam2EmbeddingRequest(
        api_key="key-a", image=IMAGE, image_id="img-1"
    )
    segment_request = Sam2SegmentationRequest(
        api_key="key-a", image=IMAGE, image_id="img-1"
    )

    embed_params = _build_interactive_segmentation_params(
        "embed_images", embed_request
    )
    segment_params = _build_interactive_segmentation_params(
        "segment_with_visual_prompts", segment_request
    )

    expected = [namespace_client_hash_id("img-1", "key-a")]
    assert embed_params["image_hashes"] == expected
    assert segment_params["image_hashes"] == expected


def test_embed_response_echoes_raw_client_id():
    request = Sam2EmbeddingRequest(api_key="key-a", image=IMAGE, image_id="img-1")
    prediction = [
        SimpleNamespace(image_hash=namespace_client_hash_id("img-1", "key-a"))
    ]

    response = _repack_sam_embeddings("embed", prediction, request)

    assert response.image_id == "img-1"


def test_embed_response_strips_namespace_from_worker_returned_hash():
    request = Sam2EmbeddingRequest(api_key="key-a", image=IMAGE, image_id=None)
    prediction = [
        SimpleNamespace(image_hash=namespace_client_hash_id("img-1", "key-a"))
    ]

    response = _repack_sam_embeddings("embed", prediction, request)

    assert response.image_id == "img-1"


def test_embed_response_passes_worker_content_hash_through():
    request = Sam2EmbeddingRequest(api_key="key-a", image=IMAGE, image_id=None)
    prediction = [SimpleNamespace(image_hash="cafe1234")]

    response = _repack_sam_embeddings("embed", prediction, request)

    assert response.image_id == "cafe1234"
