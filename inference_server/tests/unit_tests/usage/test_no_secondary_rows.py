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


def _keys(usage_collector):
    return [(row["category"], row["resource_id"]) for row in usage_collector.rows]


def _rows_of(usage_collector, category):
    return [row for row in usage_collector.rows if row["category"] == category]


def _run_workflow(client, specification, inputs):
    response = client.post(
        "/workflows/run",
        json={"specification": specification, "inputs": inputs, "api_key": "k"},
    )
    assert response.status_code == 200, response.text

    return response


def test_object_detection_route_records_a_model_row_and_a_request_row(
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
    assert _keys(usage_collector) == [("model", "ds/1"), ("request", "ds/1")]
    for row in usage_collector.rows:
        assert "models" not in row["resource_details"]
        assert "custom_python" not in row["resource_details"]


def test_two_model_steps_give_a_model_row_each_plus_the_workflow_rows(
    usage_client, usage_collector, fake_stat
):
    client = usage_client(_detection_gateway(fake_stat, "ds/1", "ds/2"))

    _run_workflow(
        client,
        _detection_workflow([("first", "ds/1"), ("second", "ds/2")]),
        {"image": [{"type": "base64", "value": _jpeg_b64()}] * 3},
    )

    keys = _keys(usage_collector)
    assert sorted(keys[:2]) == [("model", "ds/1"), ("model", "ds/2")]
    assert keys[2][0] == "workflows"
    assert keys[3][0] == "request"
    assert len(keys) == 4
    models = sorted(_rows_of(usage_collector, "model"), key=lambda r: r["resource_id"])
    assert models == [
        {
            "api_key": "k",
            "category": "model",
            "resource_id": model_id,
            "resource_details": {
                "model_architecture": "yolov8",
                "model_variant": "yolov8-n",
                "task_type": "object-detection",
                "model_input_height": 640,
                "model_input_width": 640,
            },
            "frames": 3,
            "execution_duration": model["execution_duration"],
            "fps": 0.0,
            "source_duration": 0.0,
            "billable": True,
            "is_preview": False,
            "error_type": None,
            "error_status_code": None,
            "roboflow_service_name": None,
            "roboflow_internal_secret": None,
            "megapixel_buckets": {
                "0.25-0.5": {
                    "processed_frames": 3,
                    "execution_duration": model["execution_duration"],
                }
            },
        }
        for model, model_id in zip(models, ["ds/1", "ds/2"])
    ]
    for row in usage_collector.rows:
        assert "models" not in row["resource_details"]
        assert "custom_python" not in row["resource_details"]


def test_two_steps_running_one_model_record_one_model_row_increment_each(
    usage_client, usage_collector, fake_stat
):
    client = usage_client(_detection_gateway(fake_stat, "ds/1"))

    _run_workflow(
        client,
        _detection_workflow([("first", "ds/1"), ("second", "ds/1")]),
        {"image": {"type": "base64", "value": _jpeg_b64()}},
    )

    keys = _keys(usage_collector)
    assert keys[:2] == [("model", "ds/1"), ("model", "ds/1")]
    assert [key[0] for key in keys[2:]] == ["workflows", "request"]
    assert [row["frames"] for row in _rows_of(usage_collector, "model")] == [1, 1]


def test_parallel_custom_python_blocks_give_a_workflow_block_row_each(
    usage_client, usage_collector
):
    client = usage_client(FakeGateway())
    specification = _python_workflow(
        [_sleeping_block("Slow", SLOW_SLEEP_S), _sleeping_block("Quick", QUICK_SLEEP_S)]
    )

    response = _run_workflow(client, specification, {"x": 3})

    assert response.json()["outputs"] == [{"slow": 3, "quick": 3}]
    keys = _keys(usage_collector)
    assert [key[0] for key in keys] == [
        "workflow_block",
        "workflow_block",
        "workflows",
        "request",
    ]
    blocks = {
        row["resource_details"]["step_name"]: row
        for row in _rows_of(usage_collector, "workflow_block")
    }
    assert set(blocks) == {"slow", "quick"}
    assert blocks["slow"]["resource_details"]["block_type"] == "Slow"
    assert blocks["quick"]["resource_details"]["block_type"] == "Quick"
    assert blocks["slow"]["execution_duration"] >= SLOW_SLEEP_S
    assert blocks["quick"]["execution_duration"] >= QUICK_SLEEP_S
    assert blocks["quick"]["execution_duration"] < blocks["slow"]["execution_duration"]
    for row in blocks.values():
        assert row["resource_id"].startswith("custom_python/")
        assert row["api_key"] == "k"
        assert row["frames"] == 1
        assert row["resource_details"]["block_kind"] == "custom_python"
        assert row["resource_details"]["duration_source"] == "local_runtime"
        assert row["resource_details"]["execution_mode"] == "local"
        assert row["resource_details"]["is_preview"] is False
    assert blocks["slow"]["resource_id"] != blocks["quick"]["resource_id"]


def test_block_owned_model_run_gives_a_model_row(usage_client, usage_collector):
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
    keys = _keys(usage_collector)
    assert keys[0] == ("model", "sam2/hiera_small")
    assert keys[1][0] == "workflow_block"
    assert [key[0] for key in keys[2:]] == ["workflows", "request"]
    (model,) = _rows_of(usage_collector, "model")
    assert model["frames"] == 2
    assert model["api_key"] == "k"
    assert model["resource_details"] == {}
    assert model["megapixel_buckets"] == {
        "unknown": {
            "processed_frames": 2,
            "execution_duration": model["execution_duration"],
        }
    }
