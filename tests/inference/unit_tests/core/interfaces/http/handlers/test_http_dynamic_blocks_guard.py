from unittest import mock

import pytest
from fastapi import HTTPException

from inference.core.interfaces.http.handlers import workflows as workflows_handlers


def test_ensure_http_dynamic_python_blocks_allowed_when_gate_enabled() -> None:
    with mock.patch.object(
        workflows_handlers,
        "ALLOW_HTTP_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS",
        True,
    ):
        workflows_handlers.ensure_http_dynamic_python_blocks_allowed(
            workflow_definition={"dynamic_blocks_definitions": [{"manifest": {}}]},
            dynamic_blocks_definitions=[{"manifest": {}}],
        )


def test_ensure_http_dynamic_python_blocks_allowed_rejects_explicit_definitions() -> (
    None
):
    with mock.patch.object(
        workflows_handlers,
        "ALLOW_HTTP_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS",
        False,
    ):
        with pytest.raises(HTTPException) as error:
            workflows_handlers.ensure_http_dynamic_python_blocks_allowed(
                dynamic_blocks_definitions=[{"manifest": {"block_type": "MyBlock"}}],
            )

    assert error.value.status_code == 403


def test_ensure_http_dynamic_python_blocks_allowed_rejects_workflow_definition() -> (
    None
):
    with mock.patch.object(
        workflows_handlers,
        "ALLOW_HTTP_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS",
        False,
    ):
        with pytest.raises(HTTPException) as error:
            workflows_handlers.ensure_http_dynamic_python_blocks_allowed(
                workflow_definition={
                    "steps": [
                        {
                            "type": "roboflow_core/inner_workflow@v1",
                            "workflow_definition": {
                                "dynamic_blocks_definitions": [
                                    {
                                        "manifest": {"block_type": "NestedBlock"},
                                    }
                                ],
                            },
                        }
                    ],
                },
            )

    assert error.value.status_code == 403


def test_ensure_http_dynamic_python_blocks_allowed_allows_empty_payload() -> None:
    with mock.patch.object(
        workflows_handlers,
        "ALLOW_HTTP_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS",
        False,
    ):
        workflows_handlers.ensure_http_dynamic_python_blocks_allowed(
            workflow_definition={"steps": []},
            dynamic_blocks_definitions=[],
        )
