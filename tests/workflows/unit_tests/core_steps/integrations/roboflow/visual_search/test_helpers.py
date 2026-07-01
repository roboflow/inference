from unittest import mock

import numpy as np

from inference.core.workflows.core_steps.integrations.roboflow.visual_search.helpers import (
    build_visual_search_candidate_image,
    format_visual_search_candidate,
)
from inference.core.workflows.execution_engine.entities import base


def test_format_visual_search_candidate_keeps_common_fields_only_by_default() -> None:
    candidate = {
        "id": "img-1",
        "name": "Widget A",
        "filename": "widget-a.jpg",
        "url": "https://example.com/widget-a.jpg",
        "user_metadata": {"sku": "A-123"},
        "tags": ["reference"],
        "width": 640,
        "height": 480,
        "aspectRatio": 1.3333,
        "score": 0.87,
        "classification": {"class": "widget"},
        "owner": "workspace-id",
    }

    result = format_visual_search_candidate(candidate=candidate)

    assert result == {
        "image_id": "img-1",
        "image_url": "https://example.com/widget-a.jpg",
        "name": "Widget A",
        "filename": "widget-a.jpg",
        "metadata": {"sku": "A-123"},
        "tags": ["reference"],
        "width": 640,
        "height": 480,
        "aspect_ratio": 1.3333,
    }


def test_format_visual_search_candidate_can_include_requested_extra_fields() -> None:
    candidate = {
        "id": "img-1",
        "name": "Widget A",
        "score": 0.87,
        "classification": {"class": "widget"},
    }

    result = format_visual_search_candidate(
        candidate=candidate,
        extra_fields=["score", "classification"],
    )

    assert result["score"] == 0.87
    assert result["classification"] == {"class": "widget"}


@mock.patch.object(base, "load_image_from_url")
def test_build_visual_search_candidate_image_uses_candidate_identity(
    load_image_from_url_mock: mock.MagicMock,
) -> None:
    load_image_from_url_mock.return_value = np.zeros((4, 6, 3), dtype=np.uint8)
    candidate = {
        "image_id": "img-1",
        "filename": "widget-a.jpg",
        "image_url": "https://example.com/widget-a.jpg",
    }

    result = build_visual_search_candidate_image(
        candidate=candidate,
        fallback_parent_id="fallback",
    )

    assert result.parent_metadata.parent_id == "img-1"
    assert result.to_inference_format() == {
        "type": "url",
        "value": "https://example.com/widget-a.jpg",
    }


def test_build_visual_search_candidate_image_returns_none_without_url() -> None:
    result = build_visual_search_candidate_image(
        candidate={"image_id": "img-1"},
        fallback_parent_id="fallback",
    )

    assert result is None
