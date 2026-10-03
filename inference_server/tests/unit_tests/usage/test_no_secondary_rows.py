import base64
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

from tests.unit_tests.legacy.conftest import FakeGateway

SLOW_SLEEP_S = 0.05
QUICK_SLEEP_S = 0.02


def _jpeg_b64():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue()).decode()


def _detection_gateway(fake_stat, *model_ids):
    detections = SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    for model_id in model_ids:
        fake_stat[model_id] = ("object-detection", "infer", "yolov8", "yolov8-n")
    gateway = FakeGateway(
        predictions={(model_id, "infer"): detections for model_id in model_ids},
        model_info={
            model_id: {
                "class_names": ["cat"],
                "actions": {"infer": {}},
                "input_height": 640,
                "input_width": 640,
            }
            for model_id in model_ids
        },
    )

    return gateway


def _detection_workflow(steps):
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": name,
                "image": "$inputs.image",
                "model_id": model_id,
            }
            for name, model_id in steps
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": name,
                "selector": f"$steps.{name}.predictions",
            }
            for name, _ in steps
        ],
    }


def _python_block(block_type, run_function_code):
    return {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": block_type,
            "inputs": {
                "value": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["input_parameter"],
                },
            },
            "outputs": {"value": {"type": "DynamicOutputDefinition", "kind": []}},
        },
        "code": {
            "type": "PythonCode",
            "run_function_code": run_function_code,
            "run_function_name": "run",
        },
    }


def _sleeping_block(block_type, seconds):
    code = (
        "def run(self, value):\n"
        "    import time\n"
        f"    time.sleep({seconds})\n"
        "    return {'value': value}\n"
    )

    return _python_block(block_type, code)


def _python_workflow(blocks):
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowParameter", "name": "x"}],
        "dynamic_blocks_definitions": blocks,
        "steps": [
            {
                "type": block["manifest"]["block_type"],
                "name": block["manifest"]["block_type"].lower(),
                "value": "$inputs.x",
            }
            for block in blocks
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": block["manifest"]["block_type"].lower(),
                "selector": f"$steps.{block['manifest']['block_type'].lower()}.value",
            }
            for block in blocks
        ],
    }


def _only_request_row(usage_collector):
    assert [row["category"] for row in usage_collector.rows] == ["request"]

    return usage_collector.rows[0]


def _run_workflow(client, specification, inputs):
    response = client.post(
        "/workflows/run",
        json={"specification": specification, "inputs": inputs, "api_key": "k"},
    )
    assert response.status_code == 200, response.text

    return response


def test_object_detection_route_records_only_a_request_row(
    usage_client, usage_collector, fake_stat
):
    client = usage_client(_detection_gateway(fake_stat, "ds/1"))

    response = client.post(
        "/infer/object_detection",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": {"type": "base64", "value": _jpeg_b64()},
        },
    )

    assert response.status_code == 200, response.text
    row = _only_request_row(usage_collector)
    assert [entry["model_id"] for entry in row["resource_details"]["models"]] == [
        "ds/1"
    ]


def test_two_model_steps_give_one_row_with_two_models_entries(
    usage_client, usage_collector, fake_stat
):
    client = usage_client(_detection_gateway(fake_stat, "ds/1", "ds/2"))

    _run_workflow(
        client,
        _detection_workflow([("first", "ds/1"), ("second", "ds/2")]),
        {"image": [{"type": "base64", "value": _jpeg_b64()}] * 3},
    )

    row = _only_request_row(usage_collector)
    entries = sorted(row["resource_details"]["models"], key=lambda e: e["model_id"])
    assert entries == [
        {
            "model_id": "ds/1",
            "model_architecture": "yolov8",
            "model_variant": "yolov8-n",
            "task_type": "object-detection",
            "model_input_height": 640,
            "model_input_width": 640,
            "execution_duration": entries[0]["execution_duration"],
            "frames": 3,
        },
        {
            "model_id": "ds/2",
            "model_architecture": "yolov8",
            "model_variant": "yolov8-n",
            "task_type": "object-detection",
            "model_input_height": 640,
            "model_input_width": 640,
            "execution_duration": entries[1]["execution_duration"],
            "frames": 3,
        },
    ]
    assert row["resource_details"]["custom_python"] == []


def test_two_steps_running_one_model_merge_into_one_entry(
    usage_client, usage_collector, fake_stat
):
    client = usage_client(_detection_gateway(fake_stat, "ds/1"))

    _run_workflow(
        client,
        _detection_workflow([("first", "ds/1"), ("second", "ds/1")]),
        {"image": {"type": "base64", "value": _jpeg_b64()}},
    )

    row = _only_request_row(usage_collector)
    assert [
        (entry["model_id"], entry["frames"])
        for entry in row["resource_details"]["models"]
    ] == [("ds/1", 2)]


def test_parallel_custom_python_blocks_give_one_row_with_their_own_durations(
    usage_client, usage_collector
):
    client = usage_client(FakeGateway())
    specification = _python_workflow(
        [_sleeping_block("Slow", SLOW_SLEEP_S), _sleeping_block("Quick", QUICK_SLEEP_S)]
    )

    response = _run_workflow(client, specification, {"x": 3})

    assert response.json()["outputs"] == [{"slow": 3, "quick": 3}]
    row = _only_request_row(usage_collector)
    entries = {
        entry["step_name"]: entry for entry in row["resource_details"]["custom_python"]
    }
    assert set(entries) == {"slow", "quick"}
    assert entries["slow"]["block_type"] == "Slow"
    assert entries["quick"]["block_type"] == "Quick"
    assert entries["slow"]["execution_duration"] >= SLOW_SLEEP_S
    assert entries["quick"]["execution_duration"] >= QUICK_SLEEP_S
    assert (
        entries["quick"]["execution_duration"] < entries["slow"]["execution_duration"]
    )
    assert row["resource_details"]["models"] == []


def test_block_owned_model_run_gives_a_models_entry(usage_client, usage_collector):
    client = usage_client(FakeGateway())
    code = (
        "def run(self, value):\n"
        "    return self._execution_observer.observe_model_run(\n"
        "        block=self,\n"
        "        model_id='sam2/hiera_small',\n"
        "        images=[value, value],\n"
        "        run=lambda: {'value': value},\n"
        "    )\n"
    )
    specification = _python_workflow([_python_block("Video", code)])

    response = _run_workflow(client, specification, {"x": 3})

    assert response.json()["outputs"] == [{"video": 3}]
    row = _only_request_row(usage_collector)
    assert row["resource_details"]["models"] == [
        {
            "model_id": "sam2/hiera_small",
            "frames": 2,
            "execution_duration": row["resource_details"]["models"][0][
                "execution_duration"
            ],
        }
    ]
    assert [
        entry["step_name"] for entry in row["resource_details"]["custom_python"]
    ] == ["video"]
