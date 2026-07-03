import math
import threading
import time
from unittest import mock

import numpy as np
import pytest

from inference.core.utils.image_utils import load_image_base64
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    execute_operations,
)
from inference.core.workflows.core_steps.integrations.roboflow.visual_search_classifier import (
    v1,
)
from inference.core.workflows.core_steps.integrations.roboflow.visual_search_classifier.v1 import (
    BlockManifest,
    RoboflowVisualSearchClassifierBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    INFERENCE_ID_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)


def make_image(base64_image: str = "query-base64") -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="query-image"),
        base64_image=base64_image,
        numpy_image=np.zeros((4, 6, 3), dtype=np.uint8),
    )


def make_sized_image(height: int, width: int) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="query-image"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def image_size_from_base64(value: str) -> tuple[int, int]:
    image = load_image_base64(value=value)
    height, width = image.shape[:2]
    return height, width


def make_batch() -> Batch:
    return Batch(
        content=[make_image("image-1"), make_image("image-2")],
        indices=[(0,), (1,)],
    )


def make_candidate(
    image_id: str = "img-1",
    class_name: str = "widget-a",
    class_id: int = 7,
    score: object = 1.74,
) -> dict:
    return {
        "id": image_id,
        "name": "Widget A",
        "filename": "widget-a.jpg",
        "url": "https://example.com/widget-a.jpg",
        "user_metadata": {"sku": "A-123"},
        "tags": ["reference"],
        "width": 640,
        "height": 480,
        "aspectRatio": 1.3333,
        "score": score,
        "labels": [{"class": class_name, "class_id": class_id}],
    }


def test_manifest_parsing_valid() -> None:
    raw_manifest = {
        "type": "roboflow_core/visual_search_classifier@v1",
        "name": "visual_search_classifier",
        "image": "$inputs.image",
        "target_project": "reference-images",
        "top_k": 3,
    }

    manifest = BlockManifest.model_validate(raw_manifest)

    assert manifest.image == "$inputs.image"
    assert manifest.target_project == "reference-images"
    assert manifest.workspace is None
    assert manifest.top_k == 3
    assert manifest.max_image_size == 224
    assert BlockManifest.get_parameters_accepting_batches() == ["image"]
    outputs = {output.name: output.kind for output in BlockManifest.describe_outputs()}
    assert outputs == {
        "predictions": [CLASSIFICATION_PREDICTION_KIND],
        "inference_id": [INFERENCE_ID_KIND],
        "candidate_found": [BOOLEAN_KIND],
        "class_found": [BOOLEAN_KIND],
        "best_candidate": [DICTIONARY_KIND],
        "candidates": [LIST_OF_VALUES_KIND],
        "best_candidate_image": [IMAGE_KIND],
        "visual_search_score": [FLOAT_KIND],
        "error_status": [BOOLEAN_KIND],
        "message": [STRING_KIND],
    }
    assert "classification_predictions" not in outputs


def test_manifest_describes_visual_search_latency_and_resize_default() -> None:
    schema_extra = BlockManifest.model_config["json_schema_extra"]
    long_description = " ".join(schema_extra["long_description"].split())

    assert "not intended for real-time" in long_description
    assert "224" in long_description
    assert "latency" in BlockManifest.model_fields["max_image_size"].description


def test_run_calls_project_search_and_returns_predictions() -> None:
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {"results": [make_candidate()]}

        result = block.run(
            image=make_image(),
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
        fields=[
            "id",
            "name",
            "filename",
            "url",
            "user_metadata",
            "tags",
            "width",
            "height",
            "aspectRatio",
            "score",
            "labels",
            "annotations",
        ],
    )
    assert result["candidate_found"] is True
    assert result["class_found"] is True
    assert result["error_status"] is False
    assert result["message"] == "Visual search classification completed."
    assert result["visual_search_score"] == 1.74
    assert result["best_candidate"]["image_id"] == "img-1"
    assert result["best_candidate"]["labels"] == [{"class": "widget-a", "class_id": 7}]
    assert result["best_candidate_image"].to_inference_format() == {
        "type": "url",
        "value": "https://example.com/widget-a.jpg",
    }
    assert "classification_predictions" not in result
    predictions = result["predictions"]
    assert predictions == {
        "image": {"width": 6, "height": 4},
        "predictions": [{"class": "widget-a", "class_id": 7, "confidence": 0.87}],
        "top": "widget-a",
        "confidence": 0.87,
        "prediction_type": "classification",
        "inference_id": result["inference_id"],
        "parent_id": "query-image",
        "root_parent_id": "query-image",
    }


def test_run_downscales_query_image_to_default_max_size() -> None:
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {"results": [make_candidate()]}

        result = block.run(
            image=make_sized_image(height=800, width=1200),
            workspace="my-workspace",
            target_project="reference-images",
        )

    query_height, query_width = image_size_from_base64(
        search_mock.call_args.kwargs["image_base64"]
    )
    assert (query_height, query_width) == (149, 224)
    assert result["predictions"]["image"] == {"width": 1200, "height": 800}


def test_run_honors_custom_query_image_max_size() -> None:
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {"results": [make_candidate()]}

        block.run(
            image=make_sized_image(height=800, width=1200),
            workspace="my-workspace",
            target_project="reference-images",
            max_image_size=640,
        )

    query_height, query_width = image_size_from_base64(
        search_mock.call_args.kwargs["image_base64"]
    )
    assert (query_height, query_width) == (426, 640)


def test_run_does_not_reencode_image_when_it_is_smaller_than_resize_cap() -> None:
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {"results": [make_candidate()]}

        block.run(
            image=make_image(base64_image="small-query-base64"),
            workspace="my-workspace",
            target_project="reference-images",
        )

    assert search_mock.call_args.kwargs["image_base64"] == "small-query-base64"


def test_run_returns_multi_label_predictions_when_candidate_has_multiple_classes() -> (
    None
):
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")
    candidate = make_candidate()
    candidate["labels"] = [
        {"class": "widget-a", "class_id": 7},
        {"class": "fragile", "class_id": "9"},
    ]

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {"results": [candidate]}

        result = block.run(
            image=make_image(),
            workspace="my-workspace",
            target_project="reference-images",
        )

    assert result["candidate_found"] is True
    assert result["class_found"] is True
    assert result["error_status"] is False
    assert "classification_predictions" not in result
    predictions = result["predictions"]
    assert predictions == {
        "image": {"width": 6, "height": 4},
        "predictions": {
            "widget-a": {"class_id": 7, "confidence": 0.87},
            "fragile": {"class_id": 9, "confidence": 0.87},
        },
        "predicted_classes": ["widget-a", "fragile"],
        "prediction_type": "classification",
        "inference_id": result["inference_id"],
        "parent_id": "query-image",
        "root_parent_id": "query-image",
    }


def test_run_returns_multi_label_predictions_compatible_with_all_classes_extraction() -> (
    None
):
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")
    candidate = make_candidate()
    candidate["labels"] = ["widget-a", "fragile"]

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {"results": [candidate]}

        result = block.run(
            image=make_image(),
            workspace="my-workspace",
            target_project="reference-images",
        )

    extracted_classes = execute_operations(
        value=result["predictions"],
        operations=[
            {
                "type": "ClassificationPropertyExtract",
                "property_name": "all_classes",
            }
        ],
    )

    assert extracted_classes == ["widget-a", "fragile"]


def test_run_returns_prediction_with_zero_confidence_when_candidate_has_no_score() -> (
    None
):
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")
    candidate = make_candidate()
    del candidate["score"]

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {"results": [candidate]}

        result = block.run(
            image=make_image(),
            workspace="my-workspace",
            target_project="reference-images",
        )

    assert result["candidate_found"] is True
    assert result["class_found"] is True
    assert result["error_status"] is False
    assert result["visual_search_score"] is None
    assert result["predictions"]["confidence"] == 0.0
    assert result["predictions"]["predictions"][0]["confidence"] == 0.0


def test_run_ignores_raw_search_engine_score_when_candidate_score_is_missing() -> None:
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")
    candidate = make_candidate()
    del candidate["score"]
    candidate["_score"] = "1.64"

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {"results": [candidate]}

        result = block.run(
            image=make_image(),
            workspace="my-workspace",
            target_project="reference-images",
        )

    assert result["candidate_found"] is True
    assert result["class_found"] is True
    assert result["error_status"] is False
    assert result["visual_search_score"] is None
    assert result["best_candidate"]["score"] is None
    assert result["predictions"]["confidence"] == 0.0


def test_run_resolves_workspace_from_api_key_when_workspace_is_not_provided() -> None:
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")

    with mock.patch.object(
        v1, "get_roboflow_workspace", return_value="resolved-workspace"
    ) as workspace_mock, mock.patch.object(
        v1, "search_project_images_at_roboflow"
    ) as search_mock:
        search_mock.return_value = {"results": [make_candidate()]}

        result = block.run(
            image=make_image(),
            target_project="reference-images",
        )

    workspace_mock.assert_called_once_with(api_key="api-key")
    assert search_mock.call_args.kwargs["workspace"] == "resolved-workspace"
    assert result["candidate_found"] is True


def test_run_returns_numeric_visual_search_score_when_candidate_score_is_string() -> (
    None
):
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {"results": [make_candidate(score="1.74")]}

        result = block.run(
            image=make_image(),
            workspace="my-workspace",
            target_project="reference-images",
        )

    assert result["candidate_found"] is True
    assert result["class_found"] is True
    assert result["error_status"] is False
    assert result["visual_search_score"] == 1.74
    assert result["best_candidate"]["score"] == 1.74
    assert result["candidates"][0]["score"] == 1.74
    assert result["predictions"]["confidence"] == 0.87


def test_run_returns_zero_confidence_when_candidate_score_is_not_finite() -> None:
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {"results": [make_candidate(score=math.nan)]}

        result = block.run(
            image=make_image(),
            workspace="my-workspace",
            target_project="reference-images",
        )

    assert result["candidate_found"] is True
    assert result["class_found"] is True
    assert result["error_status"] is False
    assert result["visual_search_score"] is None
    assert result["best_candidate"]["score"] is None
    assert result["candidates"][0]["score"] is None
    assert result["predictions"]["confidence"] == 0.0
    assert result["predictions"]["predictions"][0]["confidence"] == 0.0


def test_run_returns_no_prediction_when_api_returns_no_results() -> None:
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {"results": []}

        result = block.run(
            image=make_image(),
            workspace="my-workspace",
            target_project="empty-reference-images",
        )

    assert result["candidate_found"] is False
    assert result["class_found"] is False
    assert result["predictions"] is None
    assert result["visual_search_score"] is None
    assert result["best_candidate"] == {}
    assert result["candidates"] == []
    assert result["error_status"] is False
    assert result["message"] == "No visually similar images found."
    assert result["inference_id"]


def test_run_returns_error_when_best_candidate_has_no_class_annotation() -> None:
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")
    candidate = make_candidate()
    del candidate["labels"]

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {"results": [candidate]}

        result = block.run(
            image=make_image(),
            workspace="my-workspace",
            target_project="reference-images",
        )

    assert result["candidate_found"] is True
    assert result["class_found"] is False
    assert result["predictions"] is None
    assert result["visual_search_score"] == 1.74
    assert result["error_status"] is True
    assert (
        result["message"]
        == "Best visual search candidate does not include classification labels "
        "or annotations."
    )


def test_run_returns_missing_class_error_when_candidate_has_no_score_or_class() -> None:
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")
    candidate = make_candidate()
    del candidate["score"]
    del candidate["labels"]

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.return_value = {"results": [candidate]}

        result = block.run(
            image=make_image(),
            workspace="my-workspace",
            target_project="reference-images",
        )

    assert result["candidate_found"] is True
    assert result["class_found"] is False
    assert result["predictions"] is None
    assert result["visual_search_score"] is None
    assert result["error_status"] is True
    assert (
        result["message"]
        == "Best visual search candidate does not include classification labels "
        "or annotations."
    )


def test_run_returns_error_when_project_search_fails() -> None:
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.side_effect = RuntimeError("boom")

        result = block.run(
            image=make_image(),
            workspace="my-workspace",
            target_project="reference-images",
        )

    assert result["candidate_found"] is False
    assert result["class_found"] is False
    assert result["predictions"] is None
    assert result["error_status"] is True
    assert result["message"] == "Visual search classification failed: boom"
    assert result["inference_id"]


def test_run_batch_returns_one_classification_per_image() -> None:
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.side_effect = [
            {"results": [make_candidate(image_id="img-1", class_name="a", class_id=0)]},
            {"results": [make_candidate(image_id="img-2", class_name="b", class_id=1)]},
        ]

        result = block.run(
            image=make_batch(),
            workspace="my-workspace",
            target_project="reference-images",
        )

    assert [row["predictions"]["top"] for row in result] == ["a", "b"]
    assert [row["inference_id"] for row in result]
    assert search_mock.call_args_list[0].kwargs["image_base64"] == "image-1"
    assert search_mock.call_args_list[1].kwargs["image_base64"] == "image-2"


def test_run_batch_searches_images_in_parallel_and_preserves_output_order() -> None:
    block = RoboflowVisualSearchClassifierBlockV1(api_key="api-key")
    active_requests = 0
    max_active_requests = 0
    active_requests_lock = threading.Lock()

    def search_side_effect(**kwargs: object) -> dict:
        nonlocal active_requests, max_active_requests
        with active_requests_lock:
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
        try:
            image_base64 = kwargs["image_base64"]
            if image_base64 == "image-1":
                time.sleep(0.08)
                return {
                    "results": [
                        make_candidate(image_id="img-1", class_name="a", class_id=0)
                    ]
                }
            time.sleep(0.01)
            return {
                "results": [
                    make_candidate(image_id="img-2", class_name="b", class_id=1)
                ]
            }
        finally:
            with active_requests_lock:
                active_requests -= 1

    with mock.patch.object(v1, "search_project_images_at_roboflow") as search_mock:
        search_mock.side_effect = search_side_effect

        result = block.run(
            image=make_batch(),
            workspace="my-workspace",
            target_project="reference-images",
        )

    assert max_active_requests == 2
    assert [row["predictions"]["top"] for row in result] == ["a", "b"]


def test_run_without_api_key_raises() -> None:
    block = RoboflowVisualSearchClassifierBlockV1(api_key=None)

    with pytest.raises(ValueError, match="without a Roboflow API key"):
        block.run(
            image=make_image(),
            workspace="my-workspace",
            target_project="reference-images",
        )
