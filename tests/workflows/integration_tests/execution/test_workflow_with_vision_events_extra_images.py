from unittest import mock

import numpy as np
import requests

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

# The slicer raises dimensionality the way a crop step does, so `extra_images` holds
# a batch of crops one level below the workflow input, next to `input_image`.
WORKFLOW_WITH_CROPS_AS_EXTRA_IMAGES = {
    "version": "1.4.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/image_slicer@v1",
            "name": "crop_to_part",
            "image": "$inputs.image",
            "slice_width": 64,
            "slice_height": 64,
            "overlap_ratio_width": 0.0,
            "overlap_ratio_height": 0.0,
        },
        {
            "type": "roboflow_core/roboflow_vision_events@v1",
            "name": "write_part_event",
            "input_image": "$steps.crop_to_part.slices",
            "extra_images": {"raw_crop": "$steps.crop_to_part.slices"},
            "event_type": "quality_check",
            "qc_result": "pass",
            "solution": "part-inspection",
            "fire_and_forget": False,
            "cooldown_seconds": 0,
            "write_to_event_store": True,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "event_id",
            "selector": "$steps.write_part_event.event_id",
        }
    ],
}


def _local_event_response(_url: str, **_kwargs) -> mock.MagicMock:
    response = mock.MagicMock()
    response.status_code = 201
    response.json.return_value = {"id": "evt-local"}
    response.raise_for_status.return_value = None
    return response


def test_workflow_sends_each_crop_as_its_own_event_with_raw_crop(
    model_manager: ModelManager,
) -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CROPS_AS_EXTRA_IMAGES,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=1,
    )
    image = np.zeros((128, 128, 3), dtype=np.uint8)

    # when
    with mock.patch.object(requests, "post", side_effect=_local_event_response) as post:
        result = execution_engine.run(runtime_parameters={"image": image})

    # then
    assert result == [{"event_id": ["evt-local"] * 4}]
    payloads = [call.kwargs["json"] for call in post.call_args_list]
    assert len(payloads) == 4
    for payload in payloads:
        labels = [entry["label"] for entry in payload["images"]]
        assert labels == ["workflow", "raw_crop"]
        workflow_entry, crop_entry = payload["images"]
        assert crop_entry["base64Image"] == workflow_entry["inputBase64Image"]


def test_workflow_skips_only_the_frame_with_a_missing_extra_image(
    model_manager: ModelManager,
) -> None:
    crop = {
        "type": "roboflow_core/absolute_static_crop@v1",
        "name": "crop_to_part",
        "images": "$inputs.image",
        "x_center": 80,
        "y_center": 80,
        "width": 16,
        "height": 16,
    }
    sink = {
        **WORKFLOW_WITH_CROPS_AS_EXTRA_IMAGES["steps"][1],
        "input_image": "$inputs.image",
        "extra_images": {"raw_crop": "$steps.crop_to_part.crops"},
    }
    engine = ExecutionEngine.init(
        workflow_definition={
            **WORKFLOW_WITH_CROPS_AS_EXTRA_IMAGES,
            "steps": [crop, sink],
        },
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=1,
    )
    # Both frames have a primary image. The extra crop fits only the first.
    frames = [np.zeros((size, size, 3), dtype=np.uint8) for size in (128, 32)]

    with mock.patch.object(requests, "post", side_effect=_local_event_response) as post:
        result = engine.run(runtime_parameters={"image": frames})

    assert result == [{"event_id": "evt-local"}, {"event_id": None}]
    post.assert_called_once()
    assert [image["label"] for image in post.call_args.kwargs["json"]["images"]] == [
        "workflow",
        "raw_crop",
    ]
