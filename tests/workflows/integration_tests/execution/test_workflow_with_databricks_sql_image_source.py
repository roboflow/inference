"""Databricks SQL Image Source chained into Roboflow Dataset Upload.

Reproduces the "upload images from Databricks to Roboflow" flow as a Workflow:
rows come out of a (faked) SQL warehouse, every row becomes an image and is
registered in a Roboflow project with the row key as the image name.
"""

import sys
import types
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List
from unittest import mock
from unittest.mock import MagicMock

import cv2
import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

DATABRICKS_TO_ROBOFLOW_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "databricks_token"},
    ],
    "steps": [
        {
            "type": "roboflow_core/databricks_sql_image_source@v1",
            "name": "db",
            "server_hostname": "dbc-a1b2c3d4-e5f6.cloud.databricks.com",
            "http_path": "/sql/1.0/warehouses/abcdef1234567890",
            "access_token": "$inputs.databricks_token",
            "catalog": "main",
            "schema_name": "vision",
            "table": "images",
            "metadata_columns": ["id"],
            "image_name_column": "id",
        },
        {
            "type": "roboflow_core/roboflow_dataset_upload@v2",
            "name": "upload",
            "images": "$steps.db.images",
            "image_name": "$steps.db.image_names",
            "target_project": "raccoons",
            "usage_quota_name": "databricks-import",
            "registration_tags": ["databricks"],
            "labeling_batch_prefix": "data-bricks-upload",
            "fire_and_forget": False,
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "upload", "selector": "$steps.upload.*"},
    ],
}


class FakeRow:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def asDict(self) -> Dict[str, Any]:
        return dict(self._data)


class FakeCursor:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
        self._position = 0

    def execute(self, statement: str, parameters=None) -> None:
        pass

    def fetchmany(self, size: int) -> List[FakeRow]:
        batch = self._rows[self._position : self._position + size]
        self._position += size
        return [FakeRow(row) for row in batch]

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class FakeConnection:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def cursor(self, **kwargs: Any) -> FakeCursor:
        return FakeCursor(self._rows)

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


@contextmanager
def fake_databricks(rows: List[Dict[str, Any]]) -> Iterator[None]:
    from roboflow_workflows.core_steps.integrations.databricks.sql_image_source.v1 import (
        DatabricksSQLImageSourceBlockV1,
    )

    sql_module = types.ModuleType("databricks.sql")
    sql_module.connect = lambda **kwargs: FakeConnection(rows)
    databricks_module = types.ModuleType("databricks")
    databricks_module.sql = sql_module
    with mock.patch.dict(
        sys.modules, {"databricks": databricks_module, "databricks.sql": sql_module}
    ), mock.patch(
        "roboflow_workflows.core_steps.integrations.databricks.sql_image_source.v1"
        ".resolve_and_validate_ips",
        return_value=["52.1.2.3"],
    ), mock.patch(
        "roboflow_workflows.core_steps.integrations.databricks.sql_image_source.v1"
        ".install_pinned_transport"
    ), mock.patch(
        "roboflow_workflows.core_steps.integrations.databricks.sql_image_source.v1"
        ".connector_available",
        return_value=True,
    ), mock.patch.object(
        DatabricksSQLImageSourceBlockV1,
        "run",
        DatabricksSQLImageSourceBlockV1._run_in_process,
    ):
        yield


def jpeg_bytes() -> bytes:
    image = np.full((32, 48, 3), 200, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    return buffer.tobytes()


def _upload_registration_targets() -> List[str]:
    # The loader binds either the plain or the tensor-native module depending on
    # the process configuration; patch the registration helper on both so the
    # test is independent of ENABLE_TENSOR_DATA_REPRESENTATION.
    targets = []
    for module_name in (
        "roboflow_workflows.core_steps.sinks.roboflow.dataset_upload.v2",
        "roboflow_workflows.core_steps.sinks.roboflow.dataset_upload.v2_tensor",
    ):
        try:
            __import__(module_name)
        except ImportError:
            continue
        if hasattr(sys.modules[module_name], "register_datapoint_at_roboflow"):
            targets.append(f"{module_name}.register_datapoint_at_roboflow")
    assert targets, "no dataset upload module exposes register_datapoint_at_roboflow"
    return targets


def test_databricks_rows_are_uploaded_to_roboflow_with_row_keys_as_names(
    model_manager: ModelManager,
) -> None:
    # given
    rows = [{"id": 1, "bytes": jpeg_bytes()}, {"id": 2, "bytes": jpeg_bytes()}]
    register_mock = MagicMock(return_value=(False, "registered"))
    patches = [
        mock.patch(target, register_mock) for target in _upload_registration_targets()
    ]
    execution_engine = ExecutionEngine.init(
        workflow_definition=DATABRICKS_TO_ROBOFLOW_WORKFLOW,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": "rf-test-key",
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    for patcher in patches:
        patcher.start()
    try:
        with fake_databricks(rows):
            result = execution_engine.run(
                runtime_parameters={"databricks_token": "dapi-xyz"}
            )
    finally:
        for patcher in patches:
            patcher.stop()

    # then
    assert register_mock.call_count == 2
    names = sorted(call.kwargs["image_name"] for call in register_mock.call_args_list)
    assert names == ["1", "2"]
    projects = {call.kwargs["target_project"] for call in register_mock.call_args_list}
    assert projects == {"raccoons"}
    assert result[0]["upload"] == [
        {"error_status": False, "message": "registered"},
        {"error_status": False, "message": "registered"},
    ]
