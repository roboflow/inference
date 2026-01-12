import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.common.deserializers import (
    deserialize_detections_kind,
)
from inference.core.workflows.core_steps.common.utils import (
    convert_inference_detections_batch_to_sv_detections,
    post_process_ocr_result,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def test_convert_inference_detections_batch_to_sv_detections_with_invalid_polygons() -> (
    None
):
    # given
    # supervision skips polygons with < 3 points.
    # If we have 2 predictions, one valid and one invalid,
    # sv.Detections.from_inference returns a Detections object of length 1.
    predictions = [
        {
            "image": {"height": 200, "width": 100},
            "predictions": [
                {
                    "width": 50,
                    "height": 100,
                    "x": 50,
                    "y": 100,
                    "confidence": 0.1,
                    "class_id": 1,
                    "points": [
                        {"x": 30, "y": 80},
                        {"x": 30, "y": 120},
                        {"x": 70, "y": 120},
                    ],
                    "class": "dog",
                    "detection_id": "valid",
                    "parent_id": "image",
                },
                {
                    "width": 50,
                    "height": 100,
                    "x": 75,
                    "y": 175,
                    "confidence": 0.2,
                    "class_id": 0,
                    "points": [
                        {"x": 90, "y": 170},
                        {"x": 90, "y": 190},
                    ],  # ONLY 2 POINTS - will be skipped by supervision
                    "class": "cat",
                    "detection_id": "invalid",
                    "parent_id": "image",
                },
            ],
        }
    ]

    # when
    result = convert_inference_detections_batch_to_sv_detections(
        predictions=predictions,
    )

    # then
    assert len(result) == 1
    detections = result[0]

    # Core fields length
    assert len(detections.xyxy) == 1
    assert len(detections.confidence) == 1
    assert len(detections.class_id) == 1

    # Metadata fields length (THIS WAS THE BUG - they used to be length 2)
    assert len(detections.data["detection_id"]) == 1
    assert len(detections.data["parent_id"]) == 1
    assert len(detections.data["image_dimensions"]) == 1

    # Ensure they contain the right data (the valid one)
    assert detections.data["detection_id"][0] == "valid"


def test_deserialize_detections_kind_with_invalid_polygons() -> None:
    # given
    detections_input = {
        "image": {"height": 200, "width": 100},
        "predictions": [
            {
                "width": 10,
                "height": 10,
                "x": 5,
                "y": 5,
                "confidence": 0.5,
                "class_id": 1,
                "points": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 5, "y": 10}],
                "class": "valid",
            },
            {
                "width": 10,
                "height": 10,
                "x": 15,
                "y": 15,
                "confidence": 0.6,
                "class_id": 2,
                "points": [{"x": 10, "y": 10}, {"x": 20, "y": 20}],  # INVALID
                "class": "invalid",
            },
        ],
    }

    # when
    result = deserialize_detections_kind(
        parameter="test_param",
        detections=detections_input,
    )

    # then
    assert len(result) == 1
    assert len(result.data["detection_id"]) == 1
    assert len(result.data["parent_id"]) == 1
    assert len(result.data["image_dimensions"]) == 1
    assert result.data["parent_id"][0] == "test_param"


def test_post_process_ocr_result_with_invalid_polygons() -> None:
    # given
    predictions = [
        {
            "image": {"height": 200, "width": 100},
            "predictions": [
                {
                    "width": 10,
                    "height": 10,
                    "x": 5,
                    "y": 5,
                    "confidence": 0.5,
                    "class_id": 1,
                    "points": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 5, "y": 10}],
                    "class": "valid",
                    "detection_id": "valid_ocr",
                },
                {
                    "width": 10,
                    "height": 10,
                    "x": 15,
                    "y": 15,
                    "confidence": 0.6,
                    "class_id": 2,
                    "points": [{"x": 10, "y": 10}, {"x": 20, "y": 20}],  # INVALID
                    "class": "invalid",
                    "detection_id": "invalid_ocr",
                },
            ],
        }
    ]
    images = [
        WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="image_root"),
            workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="image_root"),
            numpy_image=np.zeros((200, 100, 3), dtype=np.uint8),
        )
    ]
    expected_output_keys = {
        "result",
        "parent_id",
        "root_parent_id",
        "prediction_type",
        "predictions",
    }

    # when
    result = post_process_ocr_result(
        images=images,
        predictions=predictions,
        expected_output_keys=expected_output_keys,
    )

    # then
    assert len(result) == 1
    detections = result[0]["predictions"]
    assert len(detections) == 1
    assert len(detections.data["detection_id"]) == 1
    assert detections.data["detection_id"][0] == "valid_ocr"
