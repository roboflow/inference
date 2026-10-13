"""Compiled workflows with both event blocks from an explicit local catalogue."""

import torch
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.roboflow_platform.vision_events import (
    EventBundle,
    RoboflowVisionEventsBlock,
    VisionEventBundleBlock,
)
from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.compilation import compile_workflow


class Ports:
    """Both fake ports in one recorder, in call order."""

    def __init__(self):
        self.calls = []

    def submit(self, request, **kwargs):
        self.calls.append(("event", request))
        return f"event-{len(self.calls)}"

    def prepare_destination(self, **kwargs):
        pass

    def create_bundle(self, request, **kwargs):
        self.calls.append(("bundle", request))
        return EventBundle(
            event_id=f"bundle-{len(self.calls)}",
            bundle_path=f"/bundles/{kwargs['file_name']}.tar.gz",
        )

    def requests(self, kind):
        return [request for called, request in self.calls if called == kind]


def compile_plan(*, bundle=True, fire_and_forget=False):
    steps = [
        {
            "type": RoboflowVisionEventsBlock.type,
            "name": "event",
            "event_type": "custom",
            "solution": "use-case",
            "input_image": "$inputs.image",
            "custom_value": "$inputs.value",
            "custom_metadata": {"resolved": "$inputs.value", "fixed": 3},
            "cooldown_seconds": 0,
            "fire_and_forget": fire_and_forget,
        }
    ]
    outputs = [
        {"type": "JsonField", "name": "event_id", "selector": "$steps.event.event_id"},
        {"type": "JsonField", "name": "message", "selector": "$steps.event.message"},
    ]
    if bundle:
        steps.append(
            {
                "type": VisionEventBundleBlock.type,
                "name": "bundle",
                "event_type": "operator_feedback",
                "related_event_id": "$steps.event.event_id",
                "feedback": "correct",
                "output_image": "$inputs.image",
                "target_directory": "/bundles",
                "file_name": "$inputs.value",
                "cooldown_seconds": 0,
                "fire_and_forget": False,
            }
        )
        outputs.append(
            {
                "type": "JsonField",
                "name": "bundle_path",
                "selector": "$steps.bundle.bundle_path",
            }
        )
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "value"},
        ],
        "steps": steps,
        "outputs": outputs,
    }
    catalogue = create_catalogue().with_blocks(
        [RoboflowVisionEventsBlock, VisionEventBundleBlock]
    )
    plan = compile_workflow(definition, catalogue=catalogue)

    return plan


def images(count):
    return [
        ImageData.from_tensor(
            torch.full((3, 4, 5), index, dtype=torch.uint8), image_id=str(index)
        )
        for index in range(count)
    ]


def test_batched_images_keep_order_and_feed_the_downstream_bundle():
    ports = Ports()
    with compile_plan().create_session(
        resources={
            "api_key": "invented",
            "vision_events_port": ports,
            "vision_event_bundle_port": ports,
        }
    ) as session:
        rows = session.run({"image": images(2), "value": "checked"}).rows()

    events, bundles = ports.requests("event"), ports.requests("bundle")
    assert [request.input_image.image_id for request in events] == ["0", "1"]
    assert all(request.event_data == {"value": "checked"} for request in events)
    assert all(
        request.custom_metadata == {"resolved": "checked", "fixed": 3}
        for request in events
    )
    # The bundle step reviews the event its own row created.
    assert [request.event_data["relatedEventId"] for request in bundles] == [
        row["event_id"] for row in rows
    ]
    assert [request.output_image.image_id for request in bundles] == ["0", "1"]
    assert [row["bundle_path"] for row in rows] == ["/bundles/checked.tar.gz"] * 2


def test_empty_image_batch_makes_no_host_call():
    ports = Ports()
    with compile_plan(bundle=False).create_session(
        resources={"api_key": "invented", "vision_events_port": ports}
    ) as session:
        rows = session.run({"image": [], "value": "empty"}).rows()

    assert rows == [] and not ports.calls


def test_policy_disabled_runs_need_no_credentials_or_ports():
    with compile_plan().create_session(resources={"disable_sinks": True}) as session:
        rows = session.run({"image": images(1), "value": "disabled"}).rows()

    assert rows == [
        {
            "event_id": "",
            "message": "Sink was disabled by workflow execution policy",
            "bundle_path": "",
        }
    ]


def test_scheduled_events_report_admission_and_deliver_after_drain():
    ports = Ports()
    with BackgroundEffects(max_workers=1) as effects:
        with compile_plan(bundle=False, fire_and_forget=True).create_session(
            resources={
                "api_key": "invented",
                "vision_events_port": ports,
                "sink_effects": effects,
            }
        ) as session:
            rows = session.run({"image": images(2), "value": "later"}).rows()

    assert (
        rows
        == [{"event_id": "", "message": "Vision event sent in background task"}] * 2
    )
    assert effects.stats().completed == 2
    assert sorted(r.input_image.image_id for r in ports.requests("event")) == [
        "0",
        "1",
    ]
