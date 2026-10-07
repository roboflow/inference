"""Per-step resource completeness through REAL registered blocks.

A legacy `discover_dependent_resources()` list says which resources a step
uses; it does not say every identity is known. These tests compile real
definitions through `describe_workflow_workload` and check that:

* a selector-fed or blank identity field keeps the declared item but makes the
  step's resources incomplete, with a structured problem;
* a project-only identity problem does not make a fully known MODEL inventory
  incomplete;
* the streaming video blocks report their real model, with the internal
  `preloadable` aid never reaching the wire.

Nothing here loads a model, initialises a block or contacts a model API.
"""

import json

from roboflow_workflows.execution_engine.entities.workload import (
    ModelMetadataLookup,
    unresolved_selector_problem,
)
from roboflow_workflows.execution_engine.introspection.workload import (
    describe_workflow_workload,
)

OBJECT_DETECTION_MODEL = "roboflow_core/roboflow_object_detection_model@v3"
SAM2_VIDEO = "roboflow_core/segment_anything_2_video@v1"
SAM3_VIDEO = "roboflow_core/sam3_video@v1"
ACTION_RECOGNITION = "roboflow_core/roboflow_action_recognition_model@v1"


class RecordingProvider:
    def __init__(self) -> None:
        self.calls = []

    def resolve_model_metadata(self, provider: str, model_id: str):
        self.calls.append((provider, model_id))
        return ModelMetadataLookup(status="unavailable")


def _definition(steps: list, extra_inputs: list = ()) -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}]
        + [{"type": "WorkflowParameter", "name": name} for name in extra_inputs],
        "steps": steps,
        "outputs": [],
    }


def _step(introspection, node_id: str):
    return next(step for step in introspection.steps if step.node_id == node_id)


def test_selector_fed_project_is_step_incomplete_but_model_inventory_complete() -> None:
    # given - active learning literally enabled, target project from an input
    definition = _definition(
        steps=[
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "detection",
                "images": "$inputs.image",
                "model_id": "my-project/3",
                "disable_active_learning": False,
                "active_learning_target_dataset": "$inputs.al_project",
            }
        ],
        extra_inputs=["al_project"],
    )
    provider = RecordingProvider()

    # when
    introspection = describe_workflow_workload(
        definition, model_metadata_provider=provider
    )

    # then - both items are kept; only the project identity is unknown
    detection = _step(introspection, "$steps.detection")
    assert detection.resources.complete is False
    assert detection.resources.unknown_reasons == [
        unresolved_selector_problem(
            node_id="$steps.detection",
            declaration="resources",
            field="project_url",
            selector="$inputs.al_project",
            resource_type="roboflow_platform_project",
        )
    ]
    assert {
        (item.resource_type.value, json.dumps(item.to_dict()["metadata"]))
        for item in detection.resources.items
    } == {
        (
            "roboflow_platform_model",
            json.dumps(
                {
                    "model_id": "my-project/3",
                    "required_action": "execution",
                    "execution_location": "environment_defined",
                }
            ),
        ),
        (
            "roboflow_platform_project",
            json.dumps({"project_url": "$inputs.al_project"}),
        ),
    }
    models = introspection.summary.models
    assert models.complete is True
    assert [(m.provider, m.model_id) for m in models.items] == [
        ("roboflow", "my-project/3")
    ]
    assert provider.calls == [("roboflow", "my-project/3")]


def test_selector_fed_model_is_step_and_inventory_incomplete_without_lookup() -> None:
    # given
    definition = _definition(
        steps=[
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "detection",
                "images": "$inputs.image",
                "model_id": "$inputs.model",
            }
        ],
        extra_inputs=["model"],
    )
    provider = RecordingProvider()

    # when
    introspection = describe_workflow_workload(
        definition, model_metadata_provider=provider
    )

    # then
    expected = [
        unresolved_selector_problem(
            node_id="$steps.detection",
            declaration="resources",
            field="model_id",
            selector="$inputs.model",
            resource_type="roboflow_platform_model",
        )
    ]
    detection = _step(introspection, "$steps.detection")
    assert detection.resources.complete is False
    assert detection.resources.unknown_reasons == expected
    assert [item.metadata.model_id for item in detection.resources.items] == [
        "$inputs.model"
    ]
    assert introspection.summary.models.complete is False
    assert introspection.summary.models.unknown_reasons == expected
    assert introspection.summary.models.items == []
    assert provider.calls == []


def test_streaming_video_models_are_visible_and_preloadable_stays_internal() -> None:
    # given
    definition = _definition(
        steps=[
            {
                "type": SAM2_VIDEO,
                "name": "sam2",
                "images": "$inputs.image",
            },
            {
                "type": SAM3_VIDEO,
                "name": "sam3",
                "images": "$inputs.image",
                "class_names": ["person"],
            },
            {
                "type": ACTION_RECOGNITION,
                "name": "actions",
                "images": "$inputs.image",
                "model_id": "my-actions/2",
            },
        ]
    )

    # when
    introspection = describe_workflow_workload(definition)

    # then - each step fully knows its one LOCAL model ...
    for node_id in ("$steps.sam2", "$steps.sam3", "$steps.actions"):
        step = _step(introspection, node_id)
        assert step.resources.complete is True, node_id
        assert len(step.resources.items) == 1
        assert step.resources.items[0].to_dict()["metadata"]["execution_location"] == (
            "local"
        )
    assert {
        (m.model_id, tuple(m.used_by_steps)) for m in introspection.summary.models.items
    } == {
        ("sam2video/small", ("$steps.sam2",)),
        ("sam3video", ("$steps.sam3",)),
        ("my-actions/2", ("$steps.actions",)),
    }
    assert introspection.summary.models.complete is True
    # ... and the in-process loader aid never reaches the wire
    assert "preloadable" not in introspection.model_dump_json()


def test_selector_fed_sam3_visual_model_is_kept_and_incomplete() -> None:
    # given
    definition = _definition(
        steps=[
            {
                "type": SAM3_VIDEO,
                "name": "sam3",
                "images": "$inputs.image",
                "tracking_mode": "visual",
                "points": "$inputs.points",
                "visual_model_id": "$inputs.visual_model",
            }
        ],
        extra_inputs=["points", "visual_model"],
    )

    # when
    introspection = describe_workflow_workload(definition)

    # then - only the visual model is declared, verbatim
    sam3 = _step(introspection, "$steps.sam3")
    assert [item.metadata.model_id for item in sam3.resources.items] == [
        "$inputs.visual_model"
    ]
    assert sam3.resources.complete is False
    assert sam3.resources.unknown_reasons == [
        unresolved_selector_problem(
            node_id="$steps.sam3",
            declaration="resources",
            field="model_id",
            selector="$inputs.visual_model",
            resource_type="roboflow_platform_model",
        )
    ]
    assert introspection.summary.models.items == []
    assert introspection.summary.models.complete is False
