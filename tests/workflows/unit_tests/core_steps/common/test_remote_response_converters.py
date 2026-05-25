import torch

from inference.core.workflows.core_steps.common.remote_response_converters import (
    class_id_to_name_from_responses,
    dict_response_to_object_detections,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    INFERENCE_ID_KEY,
    PARENT_ID_KEY,
    WIDTH_KEY,
)


def test_dict_response_to_object_detections_converts_center_xywh_to_xyxy() -> None:
    # given
    response = {
        "image": {"width": 640, "height": 480},
        "predictions": [
            {
                "x": 100.0,
                "y": 200.0,
                "width": 40.0,
                "height": 60.0,
                "class": "cat",
                "class_id": 5,
                "confidence": 0.9,
            },
        ],
    }

    # when
    detections = dict_response_to_object_detections(response)

    # then
    assert detections.xyxy.shape == (1, 4)
    assert detections.xyxy.dtype == torch.float32
    # cx=100, cy=200, w=40, h=60  -> xyxy = (80, 170, 120, 230)
    assert torch.allclose(detections.xyxy[0], torch.tensor([80.0, 170.0, 120.0, 230.0]))
    assert detections.class_id[0].item() == 5
    assert detections.class_id.dtype == torch.int64
    assert detections.confidence[0].item() == 0.9
    assert detections.confidence.dtype == torch.float32


def test_dict_response_to_object_detections_handles_empty_predictions() -> None:
    # given
    response = {
        "image": {"width": 640, "height": 480},
        "predictions": [],
    }

    # when
    detections = dict_response_to_object_detections(response)

    # then
    assert detections.xyxy.shape == (0, 4)
    assert detections.class_id.shape == (0,)
    assert detections.confidence.shape == (0,)
    assert detections.bboxes_metadata is None


def test_dict_response_to_object_detections_handles_missing_predictions_key() -> None:
    # given
    response = {"image": {"width": 100, "height": 50}}

    # when
    detections = dict_response_to_object_detections(response)

    # then
    assert detections.xyxy.shape == (0, 4)
    assert detections.bboxes_metadata is None


def test_dict_response_to_object_detections_preserves_top_level_inference_id() -> None:
    # given
    response = {
        "image": {"width": 100, "height": 100},
        "predictions": [],
        INFERENCE_ID_KEY: "inf-123",
    }

    # when
    detections = dict_response_to_object_detections(response)

    # then
    assert detections.image_metadata is not None
    assert detections.image_metadata[INFERENCE_ID_KEY] == "inf-123"


def test_dict_response_to_object_detections_writes_image_dimensions_into_metadata() -> None:
    # given
    response = {
        "image": {"width": 640, "height": 480},
        "predictions": [],
    }

    # when
    detections = dict_response_to_object_detections(response)

    # then
    assert detections.image_metadata is not None
    assert detections.image_metadata[WIDTH_KEY] == 640
    assert detections.image_metadata[HEIGHT_KEY] == 480


def test_dict_response_to_object_detections_returns_none_metadata_when_no_image_or_id() -> None:
    # given
    response = {"predictions": []}

    # when
    detections = dict_response_to_object_detections(response)

    # then
    assert detections.image_metadata is None


def test_dict_response_to_object_detections_preserves_detection_and_parent_ids() -> None:
    # given
    response = {
        "image": {"width": 100, "height": 100},
        "predictions": [
            {
                "x": 10.0,
                "y": 10.0,
                "width": 4.0,
                "height": 4.0,
                "class": "cat",
                "class_id": 0,
                "confidence": 0.5,
                DETECTION_ID_KEY: "det-1",
                PARENT_ID_KEY: "parent-1",
            },
        ],
    }

    # when
    detections = dict_response_to_object_detections(response)

    # then
    assert detections.bboxes_metadata is not None
    assert detections.bboxes_metadata[0][DETECTION_ID_KEY] == "det-1"
    assert detections.bboxes_metadata[0][PARENT_ID_KEY] == "parent-1"


def test_dict_response_to_object_detections_mints_detection_id_when_absent() -> None:
    # given
    response = {
        "image": {"width": 100, "height": 100},
        "predictions": [
            {
                "x": 10.0,
                "y": 10.0,
                "width": 4.0,
                "height": 4.0,
                "class": "cat",
                "class_id": 0,
                "confidence": 0.5,
            },
        ],
    }

    # when
    detections = dict_response_to_object_detections(response)

    # then
    assert detections.bboxes_metadata is not None
    assigned = detections.bboxes_metadata[0][DETECTION_ID_KEY]
    assert isinstance(assigned, str)
    assert len(assigned) > 0


# ---------------------------------------------------------------------------
# class_id_to_name_from_responses
# ---------------------------------------------------------------------------


def test_class_id_to_name_from_responses_collects_pairs_across_batch() -> None:
    # given
    responses = [
        {
            "predictions": [
                {"class_id": 0, "class": "cat"},
                {"class_id": 1, "class": "dog"},
            ],
        },
        {
            "predictions": [
                {"class_id": 2, "class": "bird"},
            ],
        },
    ]

    # when
    mapping = class_id_to_name_from_responses(responses)

    # then
    assert mapping == {0: "cat", 1: "dog", 2: "bird"}


def test_class_id_to_name_from_responses_first_seen_wins_on_duplicate_id() -> None:
    # given
    responses = [
        {"predictions": [{"class_id": 0, "class": "cat"}]},
        {"predictions": [{"class_id": 0, "class": "kitten"}]},
    ]

    # when
    mapping = class_id_to_name_from_responses(responses)

    # then
    assert mapping == {0: "cat"}


def test_class_id_to_name_from_responses_skips_entries_missing_id_or_name() -> None:
    # given
    responses = [
        {
            "predictions": [
                {"class_id": 0, "class": "cat"},
                {"class_id": 1},  # no class name
                {"class": "dog"},  # no class id
            ],
        },
    ]

    # when
    mapping = class_id_to_name_from_responses(responses)

    # then
    assert mapping == {0: "cat"}


def test_class_id_to_name_from_responses_empty_batch_returns_empty_dict() -> None:
    # when
    mapping = class_id_to_name_from_responses([])

    # then
    assert mapping == {}


def test_class_id_to_name_from_responses_handles_missing_predictions_key() -> None:
    # given
    responses = [{"image": {"width": 1, "height": 1}}]

    # when
    mapping = class_id_to_name_from_responses(responses)

    # then
    assert mapping == {}


def test_class_id_to_name_from_responses_coerces_class_id_to_int() -> None:
    # given — some APIs return class_id as a numpy int or string number
    responses = [
        {"predictions": [{"class_id": "5", "class": "person"}]},
    ]

    # when
    mapping = class_id_to_name_from_responses(responses)

    # then
    assert mapping == {5: "person"}
    assert isinstance(list(mapping.keys())[0], int)
