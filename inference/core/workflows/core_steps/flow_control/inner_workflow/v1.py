import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import requests
from fastapi import BackgroundTasks
from pydantic import BaseModel, ConfigDict, Field, model_validator

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_INNER_WORKFLOW_REMOTE_DISPATCH_REQUEST_TIMEOUT,
)

if ENABLE_TENSOR_DATA_REPRESENTATION:
    from inference.core.workflows.core_steps.common.serializers_tensor import (
        serialize_wildcard_kind,
    )
else:
    from inference.core.workflows.core_steps.common.serializers import (
        serialize_wildcard_kind,
    )
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    WILDCARD_KIND,
    Selector,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    INNER_WORKFLOW_EXECUTION_MODE_EMBEDDED,
    INNER_WORKFLOW_EXECUTION_MODE_REMOTE_DISPATCH,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.errors import (
    InnerWorkflowRunNotSupportedError,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

logger = logging.getLogger(__name__)

SHORT_DESCRIPTION = (
    "Run a nested workflow definition with parameters mapped from the parent workflow."
)

LONG_DESCRIPTION = """
Execute or dispatch a **nested workflow** while mapping parent data into the child's inputs via
`parameter_bindings`.

Provide either a full inline definition in `workflow_definition`, or resolve a saved workflow using
`workflow_workspace_id` and `workflow_id` (optional `workflow_version_id`).
Reference fields are expanded at compile time via `workflows_core.inner_workflow_spec_resolver`
(default: Roboflow API using `workflows_core.api_key`, or local definitions when workspace is
`"local"`).

With `execution_mode="embedded"` (the default), the engine validates composition and
`parameter_bindings`, then **inlines** the child's steps into the parent workflow graph.

With `execution_mode="remote_dispatch"`, the block is kept as an outputless runtime sink.
It serializes the bound child inputs and submits the child workflow to the configured inference
server in a background task. Set `remote_target` on the block to point at a dedicated deployment or
local inference server. When omitted, the target defaults to `https://serverless.roboflow.com` and
can be changed by the runtime with `WORKFLOWS_INNER_WORKFLOW_REMOTE_TARGET`.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Inner Workflow",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "flow_control",
            "ui_manifest": {
                "section": "flow_control",
                "icon": "fak fa-diagram-nested",
                "blockPriority": 2,
            },
        }
    )
    type: Literal["roboflow_core/inner_workflow@v1"]
    execution_mode: Literal["embedded", "remote_dispatch"] = Field(
        default=INNER_WORKFLOW_EXECUTION_MODE_EMBEDDED,
        description=(
            "`embedded` preserves the current compile-time inlining behavior. "
            "`remote_dispatch` serializes the bound inputs and submits the child "
            "workflow in the background without exposing child outputs."
        ),
        examples=[
            INNER_WORKFLOW_EXECUTION_MODE_EMBEDDED,
            INNER_WORKFLOW_EXECUTION_MODE_REMOTE_DISPATCH,
        ],
    )
    remote_target: Optional[str] = Field(
        default=None,
        description=(
            "Base URL of the inference server that will execute the workflow in "
            "`remote_dispatch` mode. When omitted, the runtime-configured default is used."
        ),
        examples=[
            "https://serverless.roboflow.com",
            "http://127.0.0.1:9001",
        ],
    )
    workflow_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Full nested workflow definition (same JSON shape as a root workflow: version, inputs, "
            "steps, outputs). Required unless `workflow_workspace_id` and `workflow_id` are set; "
            "mutually exclusive with those reference fields."
        ),
    )
    workflow_workspace_id: Optional[str] = Field(
        default=None,
        description=(
            'Workspace id for a saved workflow to load (Roboflow slug or `"local"` for on-disk '
            "definitions). Use with `workflow_id`; mutually exclusive with a non-empty "
            "`workflow_definition`."
        ),
    )
    workflow_id: Optional[str] = Field(
        default=None,
        description="Saved workflow id to fetch. Use with `workflow_workspace_id`.",
    )
    workflow_version_id: Optional[str] = Field(
        default=None,
        description="Optional pinned workflow version when resolving by id.",
    )
    parameter_bindings: Dict[str, Selector()] = Field(
        description=(
            "Maps **child** workflow input names to a selector (or literal coerced by the engine) "
            "from the parent. Required for every child input except `WorkflowParameter` / "
            "`InferenceParameter` entries that declare a non-null `default_value` (those may be "
            "omitted and the child's default is used during compilation inlining)."
        ),
        json_schema_extra={
            "keys_bound_in": "parameter_bindings",
        },
    )

    @model_validator(mode="after")
    def validate_workflow_or_reference(self) -> "BlockManifest":
        if self.remote_target is not None and not self.remote_target.strip():
            raise ValueError("`remote_target` must be a non-empty URL when provided.")

        has_inline = (
            isinstance(self.workflow_definition, dict)
            and len(self.workflow_definition) > 0
        )

        workspace_id = (self.workflow_workspace_id or "").strip()
        workflow_id = (self.workflow_id or "").strip()
        has_ref = bool(workspace_id and workflow_id)

        if has_inline and has_ref:
            raise ValueError(
                "Provide either `workflow_definition` or workflow reference fields "
                "(`workflow_workspace_id` and `workflow_id`), not both."
            )

        if has_inline or has_ref:
            return self

        raise ValueError(
            "inner_workflow requires a non-empty `workflow_definition` object or both "
            "`workflow_workspace_id` and `workflow_id`."
        )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="*", kind=[WILDCARD_KIND])]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        if self.execution_mode == INNER_WORKFLOW_EXECUTION_MODE_REMOTE_DISPATCH:
            return []
        return self.describe_outputs()

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return False

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class InnerWorkflowBlockV1(WorkflowBlock):
    """Dispatch block; embedded inner workflows are still removed during compilation."""

    def __init__(
        self,
        api_key: Optional[str],
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
        inner_workflow_remote_target: str,
        disable_sinks: bool = False,
    ):
        self._api_key = api_key
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._remote_target = inner_workflow_remote_target
        self._disable_sinks = disable_sinks

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return [
            "api_key",
            "background_tasks",
            "thread_pool_executor",
            "inner_workflow_remote_target",
            "disable_sinks",
        ]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        execution_mode: str,
        remote_target: Optional[str],
        parameter_bindings: Dict[str, Any],
        workflow_definition: Optional[Dict[str, Any]],
        workflow_workspace_id: Optional[str],
        workflow_id: Optional[str],
        workflow_version_id: Optional[str],
    ) -> BlockResult:
        if execution_mode != INNER_WORKFLOW_EXECUTION_MODE_REMOTE_DISPATCH:
            raise InnerWorkflowRunNotSupportedError(
                "Embedded inner_workflow steps must be compiled away before execution."
            )
        if self._disable_sinks:
            return {}

        target_url = (remote_target or self._remote_target).strip()
        if not target_url:
            raise ValueError(
                "inner_workflow dispatch requires a non-empty dispatch target URL."
            )
        url, payload = prepare_workflow_dispatch_request(
            remote_target=target_url,
            api_key=self._api_key,
            parameter_bindings=parameter_bindings,
            workflow_definition=workflow_definition,
            workflow_workspace_id=workflow_workspace_id,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
        )
        request_handler = partial(
            execute_workflow_dispatch_request,
            url=url,
            payload=payload,
        )
        if self._background_tasks:
            self._background_tasks.add_task(request_handler)
        elif self._thread_pool_executor:
            self._thread_pool_executor.submit(request_handler)
        else:
            # Match existing fire-and-forget sink behavior: environments which do
            # not provide a safe background worker execute synchronously rather
            # than lose work.
            request_handler()
        return {}


def prepare_workflow_dispatch_request(
    *,
    remote_target: str,
    api_key: Optional[str],
    parameter_bindings: Dict[str, Any],
    workflow_definition: Optional[Dict[str, Any]],
    workflow_workspace_id: Optional[str],
    workflow_id: Optional[str],
    workflow_version_id: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    base_url = remote_target.rstrip("/")
    payload: Dict[str, Any] = {
        "api_key": api_key,
        "inputs": serialize_workflow_dispatch_inputs(parameter_bindings),
    }
    if workflow_definition is not None:
        payload["specification"] = workflow_definition
        return f"{base_url}/workflows/run", payload

    if not workflow_workspace_id or not workflow_id:
        raise ValueError(
            "inner_workflow dispatch requires either a workflow definition or both "
            "workflow_workspace_id and workflow_id."
        )
    payload["use_cache"] = True
    if workflow_version_id is not None:
        payload["workflow_version_id"] = workflow_version_id
    return (
        f"{base_url}/{workflow_workspace_id}/workflows/{workflow_id}",
        payload,
    )


def serialize_workflow_dispatch_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    serialized = serialize_wildcard_kind(inputs)
    return _make_json_serializable(serialized)


def _make_json_serializable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _make_json_serializable(value.model_dump(mode="json"))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, Enum):
        return _make_json_serializable(value.value)
    if isinstance(value, dict):
        return {key: _make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_serializable(item) for item in value]
    return value


def execute_workflow_dispatch_request(url: str, payload: Dict[str, Any]) -> None:
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=WORKFLOWS_INNER_WORKFLOW_REMOTE_DISPATCH_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except Exception as error:
        logger.warning("Could not dispatch inner workflow to %s. Error: %s", url, error)
