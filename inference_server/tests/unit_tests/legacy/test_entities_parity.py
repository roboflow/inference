import pytest

from inference_server.legacy import entities, prompts

pytest.importorskip("roboflow_workflows")

from roboflow_workflows.core_steps.common import (  # noqa: E402
    inference_response_entities as workflows_responses,
)
from roboflow_workflows.core_steps.common import (  # noqa: E402
    segmentation_entities as workflows_segmentation,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything_common import (  # noqa: E402
    prompts as workflows_prompts,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition import (  # noqa: E402
    entities as workflows_action_recognition,
)

_RESPONSE_CLASSES = [
    "InferenceResponseImage",
    "ResolvedModel",
    "InferenceResponse",
    "CvInferenceResponse",
    "WithVisualizationResponse",
    "InstanceSegmentationInferenceResponse",
]
_SEGMENTATION_CLASSES = [
    "Point",
    "InstanceSegmentationBasePrediction",
    "InstanceSegmentationPrediction",
    "InstanceSegmentationRLEPrediction",
    "Sam2SegmentationPrediction",
]
_PROMPT_CLASSES = ["Point", "Box", "Sam2Prompt", "Sam2PromptSet", "Sam3Prompt"]


@pytest.mark.parametrize("name", _RESPONSE_CLASSES)
def test_response_entities_match_workflows_schema(name):
    copied = getattr(entities, name)
    source = getattr(workflows_responses, name)
    assert copied.model_json_schema() == source.model_json_schema()


@pytest.mark.parametrize("name", _SEGMENTATION_CLASSES)
def test_segmentation_entities_match_workflows_schema(name):
    copied = getattr(entities, name)
    source = getattr(workflows_segmentation, name)
    assert copied.model_json_schema() == source.model_json_schema()


@pytest.mark.parametrize("name", _PROMPT_CLASSES)
def test_prompt_entities_match_workflows_schema(name):
    copied = getattr(prompts, name)
    source = getattr(workflows_prompts, name)
    assert copied.model_json_schema() == source.model_json_schema()


def test_prompt_set_round_trip_keeps_positive_and_negative_points():
    prompt_set = prompts.Sam2PromptSet(
        prompts=[
            prompts.Sam2Prompt(
                box=prompts.Box(x=10.0, y=20.0, width=4.0, height=6.0),
                points=[
                    prompts.Point(x=1.0, y=2.0, positive=True),
                    prompts.Point(x=3.0, y=4.0, positive=False),
                ],
            )
        ]
    )
    assert prompt_set.num_points() == 2
    assert prompt_set.to_sam2_inputs() == {
        "point_coords": [[[1.0, 2.0], [3.0, 4.0]]],
        "point_labels": [[1, 0]],
        "box": [[8.0, 17.0, 12.0, 23.0]],
    }


_LEGACY_BASE_REQUEST_FIELDS = {
    "id",
    "api_key",
    "usage_billable",
    "start",
    "source",
    "source_info",
    "stream_pipeline_context_id",
    "disable_model_monitoring",
}
_LEGACY_INFERENCE_RESPONSE_FIELDS = {
    "inference_id",
    "frame_id",
    "time",
    "resolved_model",
}


def test_action_recognition_prediction_matches_workflows_schema():
    copied = entities.ActionRecognitionPrediction
    source = workflows_action_recognition.ActionRecognitionPrediction
    assert copied.model_json_schema() == source.model_json_schema()
    assert copied.model_config.get("populate_by_name") is True


def test_action_recognition_prediction_serialises_class_name_under_class():
    prediction = entities.ActionRecognitionPrediction(
        start_frame_idx=1, end_frame_idx=4, class_name="wave", class_id=0
    )
    assert prediction.model_dump(by_alias=True) == {
        "start_frame_idx": 1,
        "end_frame_idx": 4,
        "class": "wave",
        "class_id": 0,
    }
    by_alias = entities.ActionRecognitionPrediction(
        **{"start_frame_idx": 1, "end_frame_idx": 4, "class": "wave", "class_id": 0}
    )
    assert by_alias == prediction


def test_action_recognition_response_carries_legacy_fields_and_workflows_timeline():
    schema = entities.ActionRecognitionInferenceResponse.model_json_schema()
    source = workflows_action_recognition.ActionRecognitionPrediction
    assert set(entities.ActionRecognitionInferenceResponse.model_fields) == (
        _LEGACY_INFERENCE_RESPONSE_FIELDS
        | {"timeline", "source_fps", "frame_count", "windows_classified"}
    )
    assert set(schema["required"]) == {
        "timeline",
        "source_fps",
        "frame_count",
        "windows_classified",
    }
    assert schema["$defs"]["ActionRecognitionPrediction"] == {
        key: value
        for key, value in source.model_json_schema().items()
        if key != "$defs"
    }


def test_action_recognition_request_carries_legacy_fields():
    request_fields = entities.ActionRecognitionInferenceRequest.model_fields
    assert set(request_fields) == _LEGACY_BASE_REQUEST_FIELDS | {
        "model_id",
        "video",
        "class_filter",
    }
    assert request_fields["model_id"].is_required()
    assert request_fields["video"].is_required()
    assert request_fields["class_filter"].default is None
    assert set(entities.InferenceRequestVideo.model_fields) == {"type", "value"}
    assert entities.InferenceRequestVideo.model_fields["type"].is_required()
    assert entities.InferenceRequestVideo.model_fields["value"].default is None
    assert issubclass(entities.ActionRecognitionInferenceRequest, entities.BaseRequest)
