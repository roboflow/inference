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
