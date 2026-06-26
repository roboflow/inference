from unittest import mock

import numpy as np
import pytest

from inference.core.workflows.core_steps.common.serializers import (
    serialise_image,
    serialize_wildcard_kind,
)
from inference.core.workflows.core_steps.integrations.roboflow.visual_search import v1
from inference.core.workflows.core_steps.integrations.roboflow.visual_search.v1 import (
    BlockManifest,
    RoboflowVisualSearchBlockV1,
)
from inference.core.workflows.execution_engine.entities import base
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import IMAGE_KIND


def make_image(base64_image: str = "base64-image") -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="query-image"),
        base64_image=base64_image,
    )


def make_batch() -> Batch:
    return Batch(
        content=[make_image("image-1"), make_image("image-2")],
        indices=[(0,), (1,)],
    )


def test_manifest_parsing_valid() -> None:
    raw_manifest = {
        "type": "roboflow_core/visual_search@v1",
        "name": "visual_search",
        "image": "$inputs.image",
        "workspace": "my-workspace",
        "target_project": "reference-images",
        "top_k": 3,
    }

    manifest = BlockManifest.model_validate(raw_manifest)

    assert manifest.image == "$inputs.image"
    assert manifest.workspace == "my-workspace"
    assert manifest.target_project == "reference-images"
    assert manifest.top_k == 3
    assert BlockManifest.get_parameters_accepting_batches() == ["image"]
    outputs = {output.name: output.kind for output in BlockManifest.describe_outputs()}
    assert outputs["best_candidate_image"] == [IMAGE_KIND]


@mock.patch.object(base, "load_image_from_url")
def test_run_calls_project_search_and_returns_best_candidate(
    load_image_from_url_mock: mock.MagicMock,
) -> None:
    load_image_from_url_mock.return_value = np.zeros((4, 6, 3), dtype=np.uint8)
    block = RoboflowVisualSearchBlockV1(api_key="api-key")

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {
            "results": [
                {
                    "id": "img-1",
                    "name": "Widget A",
                    "filename": "widget-a.jpg",
                    "score": 1.82,
                    "url": "https://example.com/widget-a.jpg",
                    "user_metadata": {"sku": "A-123"},
                    "tags": ["reference"],
                    "owner": "workspace-id",
                },
                {
                    "id": "img-2",
                    "name": "Widget B",
                    "user_metadata": {"sku": "B-123"},
                },
            ]
        }

        result = block.run(
            image=make_image("query-base64"),
            workspace="my-workspace",
            target_project="reference-images",
            top_k=2,
        )

    search_mock.assert_called_once_with(
        api_key="api-key",
        workspace="my-workspace",
        project="reference-images",
        image_base64="query-base64",
        limit=2,
    )
    assert result["candidate_found"] is True
    assert result["error_status"] is False
    assert result["best_candidate"]["image_id"] == "img-1"
    assert result["best_candidate"]["metadata"] == {"sku": "A-123"}
    assert "score" not in result["best_candidate"]
    assert "owner" not in result["best_candidate"]
    assert "raw" not in result["best_candidate"]
    assert result["candidates"][1]["image_id"] == "img-2"
    assert result["best_candidate_metadata"] == {"sku": "A-123"}
    assert result["best_candidate_tags"] == ["reference"]
    assert result["best_candidate_image"].to_inference_format() == {
        "type": "url",
        "value": "https://example.com/widget-a.jpg",
    }
    assert result["best_candidate_image"].parent_metadata.parent_id == "img-1"
    serialized_image = serialise_image(result["best_candidate_image"])
    load_image_from_url_mock.assert_called_once_with(
        value="https://example.com/widget-a.jpg"
    )
    assert serialized_image["type"] == "base64"
    assert serialized_image["value"]
    assert serialized_image["video_metadata"]["video_identifier"] == "img-1"
    serialized_result = serialize_wildcard_kind(result)
    assert serialized_result["best_candidate_image"]["type"] == "base64"
    assert (
        serialized_result["best_candidate_image"]["value"] == serialized_image["value"]
    )
    assert (
        serialized_result["best_candidate_image"]["video_metadata"]["video_identifier"]
        == "img-1"
    )


def test_run_returns_unmatched_when_api_returns_no_results() -> None:
    block = RoboflowVisualSearchBlockV1(api_key="api-key")

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {"results": []}

        result = block.run(
            image=make_image(),
            workspace="my-workspace",
            target_project="empty-reference-images",
        )

    assert result == {
        "candidate_found": False,
        "best_candidate": {},
        "candidates": [],
        "error_status": False,
        "message": "No visually similar images found.",
        "best_candidate_image": None,
        "best_candidate_metadata": {},
        "best_candidate_tags": [],
    }


def test_run_batch_returns_one_result_per_image() -> None:
    block = RoboflowVisualSearchBlockV1(api_key="api-key")

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.side_effect = [
            {"results": [{"id": "img-1", "user_metadata": {"sku": "A"}}]},
            {"results": [{"id": "img-2", "user_metadata": {"sku": "B"}}]},
        ]

        result = block.run(
            image=make_batch(),
            workspace="my-workspace",
            target_project="reference-images",
        )

    assert [row["best_candidate"]["image_id"] for row in result] == [
        "img-1",
        "img-2",
    ]
    assert search_mock.call_args_list[0].kwargs["image_base64"] == "image-1"
    assert search_mock.call_args_list[1].kwargs["image_base64"] == "image-2"


def test_run_without_api_key_raises() -> None:
    block = RoboflowVisualSearchBlockV1(api_key=None)

    with pytest.raises(ValueError, match="without a Roboflow API key"):
        block.run(
            image=make_image(),
            workspace="my-workspace",
            target_project="reference-images",
        )
