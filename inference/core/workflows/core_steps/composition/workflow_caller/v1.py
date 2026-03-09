import logging
import re
from functools import lru_cache
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import requests
from pydantic import ConfigDict, Field, field_validator

from inference.core.env import LOCAL_INFERENCE_API_URL
from inference.core.roboflow_api import get_roboflow_workspace, get_workflow_specification
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    INTEGER_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    Kind,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
The **Workflow Caller** block enables calling another Roboflow workflow from within a workflow.
This is useful for composing complex pipelines from smaller, reusable workflow components.

### How it works

The block takes a workflow ID (from the same workspace), passes inputs (including images)
to the target workflow, executes it via HTTP, and returns the results.

### Configuration

Specify the target workflow using `workflow_id` (just the workflow slug, not a full URL). The
workspace is resolved automatically from the API key used to run the parent workflow.

```
workflow_id = "my-detection-pipeline"
```

### Inputs

Pass all inputs the target workflow expects via `inputs`, including images:

```
inputs = {
    "image": "$inputs.image",
    "confidence_threshold": 0.5,
    "model_id": "$inputs.model_id",
}
```

Use `input_definitions` to declare the expected input types so the UI can render
proper connectors.

### Circular Dependency Protection

To prevent infinite recursion (e.g. Workflow A calls B, B calls A), the block tracks the chain
of workflow calls using a header. If a circular reference is detected, the block returns an error
instead of making the call.

### Dynamic Outputs

When the target workflow_id is a static string, the block resolves the target workflow's
outputs at compile time. Each output from the target workflow becomes a separate output on
this block with proper Kind types. When the workflow_id is dynamic (a selector), outputs
fall back to a single ``result`` dict.

### Limitations

* Only workflows within the same workspace are supported (v1).
* Execution is done via HTTP call, which adds network latency.
* Subject to the same timeout constraints as the hosting environment.
"""

WORKFLOW_CALL_CHAIN_HEADER = "X-Roboflow-Workflow-Call-Chain"
MAX_WORKFLOW_CALL_DEPTH = 10
WORKFLOW_CALLER_BLOCK_TYPE = "roboflow_core/workflow_caller@v1"

# Module-level cache mapping (workflow_id, version_id) -> {output_name: [Kind]}.
# Populated during compile-time validation, read by get_actual_outputs() and run().
_RESOLVED_WORKFLOW_OUTPUTS: Dict[Tuple[str, Optional[str]], Dict[str, List[Kind]]] = {}

# Module-level cache mapping (workflow_id, version_id) -> list of parsed input dicts
# from spec.  Each dict has: name, type, kind, has_default.
_RESOLVED_WORKFLOW_INPUTS: Dict[Tuple[str, Optional[str]], List[Dict[str, Any]]] = {}


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Workflow Caller",
            "version": "v1",
            "short_description": "Call another Roboflow workflow from within a workflow.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-diagram-subtask",
                "blockPriority": 2,
            },
        }
    )
    type: Literal["roboflow_core/workflow_caller@v1"]
    workflow_id: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="The workflow ID to call (must be in the same workspace).",
        examples=["my-workflow-id", "$inputs.workflow_id"],
    )
    workflow_version_id: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Optional version ID of the target workflow. When provided, "
        "the block calls the specific version instead of the latest. "
        "Use this to pin a dependency to a known-good version.",
        examples=["1709234567890", "$inputs.workflow_version"],
    )
    inputs: Dict[
        str,
        Union[
            Selector(),
            str,
            float,
            bool,
            int,
            dict,
            list,
        ],
    ] = Field(
        default_factory=dict,
        description="Inputs to pass to the target workflow, keyed by input name. "
        "All inputs including images should be listed here.",
        examples=[
            {"image": "$inputs.image", "confidence_threshold": 0.5},
            {"image": "$inputs.image", "model_id": "$inputs.model_id"},
        ],
    )
    request_timeout: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=30,
        description="HTTP request timeout in seconds.",
        examples=["$inputs.request_timeout", 30],
    )
    input_definitions: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Mapping of expected input names to their kind types. "
            "Each key is an input name in the target workflow, and the value "
            "is a list of kind name strings (e.g. ['float'], ['roboflow_model_id']). "
            "This tells the parent workflow what inputs the target expects, "
            "enabling proper UI connectors. Actual values are provided via "
            "'inputs'."
        ),
        examples=[
            {"image": ["image"], "confidence_threshold": ["float"]},
            {"image": ["image"], "class_filter": ["string"], "iou_threshold": ["float"]},
        ],
    )
    output_definitions: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Mapping of expected output names to their kind types. "
            "Each key is an output name from the target workflow, and the value "
            "is a list of kind name strings (e.g. ['object_detection_prediction']). "
            "This tells the parent workflow what types to expect, enabling proper "
            "connections to downstream blocks like visualizers."
        ),
        examples=[
            {"predictions": ["object_detection_prediction"]},
            {"result_image": ["image"], "count": ["integer"]},
        ],
    )

    @field_validator("workflow_id")
    @classmethod
    def validate_workflow_id_slug(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.startswith("$"):
            if not re.fullmatch(r"[\w\-]+", value):
                raise ValueError(
                    f"`workflow_id` must be a valid slug (letters, numbers, "
                    f"hyphens, and underscores only). Got: '{value}'"
                )
        return value

    @field_validator("request_timeout")
    @classmethod
    def ensure_request_timeout_is_positive(cls, value: Any) -> Any:
        if isinstance(value, int) and value <= 0:
            raise ValueError("`request_timeout` must be a positive integer.")
        return value

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return []

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="*"),
        ]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        base: List[OutputDefinition] = []
        # 1. User-defined output definitions take precedence
        if self.output_definitions:
            kind_name_map = _build_kind_name_map()
            for name, kind_names in self.output_definitions.items():
                kinds = [
                    kind_name_map[k] for k in kind_names if k in kind_name_map
                ]
                base.append(
                    OutputDefinition(name=name, kind=kinds or [WILDCARD_KIND])
                )
            return base
        # 2. Fall back to auto-discovery from compile-time resolution
        if isinstance(self.workflow_id, str) and not self.workflow_id.startswith("$"):
            version_id = self.workflow_version_id
            if isinstance(version_id, str) and version_id.startswith("$"):
                version_id = None
            cache_key = _make_cache_key(self.workflow_id, version_id)
            resolved = _RESOLVED_WORKFLOW_OUTPUTS.get(cache_key)
            if resolved is not None:
                for name, kinds in resolved.items():
                    base.append(OutputDefinition(name=name, kind=kinds))
                return base
        # 3. Fall back to generic result dict
        base.append(OutputDefinition(name="result", kind=[DICTIONARY_KIND]))
        return base

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class WorkflowCallerBlockV1(WorkflowBlock):

    def __init__(
        self,
        api_key: Optional[str],
        workflow_call_chain: Optional[str] = None,
    ):
        self._api_key = api_key
        self._workspace_name: Optional[str] = None
        self._workflow_call_chain = workflow_call_chain or ""

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "workflow_call_chain"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        workflow_id: str,
        inputs: Dict[str, Any],
        request_timeout: int,
        input_definitions: Dict[str, List[str]],
        output_definitions: Dict[str, List[str]],
        workflow_version_id: Optional[str] = None,
    ) -> BlockResult:
        workspace_name = self._resolve_workspace_name()
        if workspace_name is None:
            raise RuntimeError(
                f"Could not resolve workspace for workflow '{workflow_id}'. "
                f"Ensure a valid API key is configured."
            )
        error_status, message, raw_outputs = call_workflow(
            api_key=self._api_key,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            inputs=inputs,
            request_timeout=request_timeout,
            call_chain=self._workflow_call_chain,
            workflow_version_id=workflow_version_id,
        )
        if error_status:
            raise RuntimeError(
                f"Workflow '{workflow_id}' execution failed: {message}"
            )
        result: Dict[str, Any] = {}
        # Determine which outputs to extract and their kinds for deserialization
        output_kinds_map = _resolve_output_kinds_for_run(
            output_definitions=output_definitions,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
        )
        if output_kinds_map is not None:
            deserializers = _build_kinds_deserializers_map()
            for output_name, output_kinds in output_kinds_map.items():
                raw_value = raw_outputs.get(output_name)
                result[output_name] = _deserialize_output_value(
                    output_name=output_name,
                    raw_value=raw_value,
                    output_kinds=output_kinds,
                    deserializers=deserializers,
                )
        else:
            result["result"] = raw_outputs
        return result

    def _resolve_workspace_name(self) -> Optional[str]:
        if self._workspace_name is not None:
            return self._workspace_name
        if not self._api_key:
            return None
        try:
            self._workspace_name = get_roboflow_workspace(api_key=self._api_key)
            return self._workspace_name
        except Exception as error:
            logging.warning(
                f"Could not resolve workspace from API key. Error: {str(error)}"
            )
            return None


def _make_cache_key(
    workflow_id: str,
    workflow_version_id: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Build a cache key that accounts for both workflow_id and version_id."""
    return (workflow_id, workflow_version_id)


def call_workflow(
    api_key: Optional[str],
    workspace_name: str,
    workflow_id: str,
    inputs: Dict[str, Any],
    request_timeout: int,
    call_chain: Optional[str] = None,
    workflow_version_id: Optional[str] = None,
) -> Tuple[bool, str, dict]:
    """Execute a workflow call and return (error_status, message, result).

    This function is separated from the block's run() to facilitate testing.
    """
    current_chain = call_chain or ""
    chain_entries = [entry for entry in current_chain.split(",") if entry]
    if workflow_id in chain_entries:
        logging.warning(
            f"Circular workflow call detected: "
            f"'{workflow_id}' is already in call chain [{current_chain}]."
        )
        return (
            True,
            f"Circular workflow call detected: '{workflow_id}' is already "
            f"in the call chain [{current_chain}]. Aborting to prevent "
            f"infinite recursion.",
            {},
        )
    if len(chain_entries) >= MAX_WORKFLOW_CALL_DEPTH:
        logging.warning(
            f"Workflow call depth limit ({MAX_WORKFLOW_CALL_DEPTH}) reached. "
            f"Call chain: [{current_chain}]."
        )
        return (
            True,
            f"Workflow call depth limit ({MAX_WORKFLOW_CALL_DEPTH}) reached. "
            f"Call chain: [{current_chain}].",
            {},
        )
    updated_chain = ",".join([*chain_entries, workflow_id])
    inputs = build_workflow_inputs(inputs=inputs)
    return execute_workflow_request(
        api_key=api_key,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        inputs=inputs,
        request_timeout=request_timeout,
        call_chain=updated_chain,
        workflow_version_id=workflow_version_id,
    )


def build_workflow_inputs(
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the inputs dictionary for the target workflow.

    Handles ``WorkflowImageData`` objects found in ``inputs`` by
    automatically serialising them to the ``{type: base64, value: ...}``
    format expected by the workflow HTTP API.
    """
    resolved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, WorkflowImageData):
            resolved[key] = {
                "type": "base64",
                "value": value.base64_image,
            }
        else:
            resolved[key] = value
    return resolved


def execute_workflow_request(
    api_key: Optional[str],
    workspace_name: str,
    workflow_id: str,
    inputs: Dict[str, Any],
    request_timeout: int,
    call_chain: str,
    workflow_version_id: Optional[str] = None,
) -> Tuple[bool, str, dict]:
    """Make the HTTP request to the target workflow and return
    (error_status, message, result)."""
    url = build_workflow_url(
        workspace_name=workspace_name,
        workflow_id=workflow_id,
    )
    payload: Dict[str, Any] = {"inputs": inputs}
    if api_key is not None:
        payload["api_key"] = api_key
    if workflow_version_id is not None:
        payload["workflow_version_id"] = workflow_version_id
    headers = {WORKFLOW_CALL_CHAIN_HEADER: call_chain}
    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=request_timeout,
        )
        response.raise_for_status()
        response_data = response.json()
        outputs = response_data.get("outputs", [])
        result = outputs[0] if outputs else {}
        return (
            False,
            "Workflow executed successfully",
            result,
        )
    except Exception as error:
        logging.warning(
            f"Could not execute workflow '{workflow_id}'. Error: {str(error)}"
        )
        error_type = type(error).__name__
        return (
            True,
            f"Failed to execute workflow '{workflow_id}': {error_type}",
            {},
        )


def build_workflow_url(workspace_name: str, workflow_id: str) -> str:
    """Build the URL for calling a workflow on the local inference server.

    Uses LOCAL_INFERENCE_API_URL to call the workflow on the same inference
    server instance that is running the parent workflow.
    """
    base_url = LOCAL_INFERENCE_API_URL.rstrip("/")
    return f"{base_url}/{workspace_name}/workflows/{workflow_id}"


# ---------------------------------------------------------------------------
# Compile-time validation
# ---------------------------------------------------------------------------


def validate_workflow_caller_no_circular_references(
    steps: List[WorkflowBlockManifest],
    api_key: Optional[str],
) -> None:
    """Compile-time validation: check that workflow_caller blocks with static
    workflow_id values don't create circular dependencies, resolve the target
    workflow's input and output interfaces, and validate that required inputs
    are provided.

    Only validates when workflow_id is a literal string (not a selector like
    ``$inputs.workflow_id``). Errors during spec fetching are logged as warnings
    and do not block compilation — the runtime circular dependency check via
    the call chain header provides an additional safety net.
    """
    caller_steps = _extract_workflow_caller_steps(steps=steps)
    if not caller_steps:
        return
    workspace_name = _resolve_workspace_for_validation(api_key=api_key)
    if workspace_name is None:
        return
    for workflow_id, version_id, step_manifest in caller_steps:
        spec = _fetch_workflow_spec_for_validation(
            api_key=api_key,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            workflow_version_id=version_id,
        )
        if spec is None:
            continue
        _check_workflow_for_circular_references(
            api_key=api_key,
            workspace_name=workspace_name,
            target_workflow_id=workflow_id,
            target_workflow_version_id=version_id,
            visited=set(),
            prefetched_spec=spec,
        )
        cache_key = _make_cache_key(workflow_id, version_id)
        _resolve_and_cache_workflow_outputs(
            cache_key=cache_key,
            spec=spec,
        )
        _resolve_and_cache_workflow_inputs(
            cache_key=cache_key,
            spec=spec,
        )
        _validate_required_inputs(
            workflow_id=workflow_id,
            workflow_version_id=version_id,
            step_manifest=step_manifest,
        )


def _extract_workflow_caller_steps(
    steps: List[WorkflowBlockManifest],
) -> List[Tuple[str, Optional[str], WorkflowBlockManifest]]:
    """Extract (workflow_id, workflow_version_id, manifest) tuples from
    workflow_caller steps that have literal (non-selector) workflow_id values."""
    result = []
    for step in steps:
        if not isinstance(step, BlockManifest):
            continue
        workflow_id = step.workflow_id
        if isinstance(workflow_id, str) and not workflow_id.startswith("$"):
            version_id = step.workflow_version_id
            # Only use version_id if it's a literal string (not a selector)
            if isinstance(version_id, str) and version_id.startswith("$"):
                version_id = None
            result.append((workflow_id, version_id, step))
    return result


def _fetch_workflow_spec_for_validation(
    api_key: Optional[str],
    workspace_name: str,
    workflow_id: str,
    workflow_version_id: Optional[str] = None,
) -> Optional[dict]:
    """Fetch a workflow spec, returning None on any error."""
    try:
        return get_workflow_specification(
            api_key=api_key,
            workspace_id=workspace_name,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
        )
    except Exception as error:
        logging.debug(
            "Could not fetch workflow '%s' for compile-time validation (error type: %s)",
            workflow_id,
            type(error).__name__,
        )
        return None


def _resolve_workspace_for_validation(
    api_key: Optional[str],
) -> Optional[str]:
    if not api_key:
        return None
    try:
        return get_roboflow_workspace(api_key=api_key)
    except Exception as error:
        logging.debug(
            "Could not resolve workspace for compile-time workflow validation (error type: %s)",
            type(error).__name__,
        )
        return None


def _check_workflow_for_circular_references(
    api_key: Optional[str],
    workspace_name: str,
    target_workflow_id: str,
    visited: set,
    target_workflow_version_id: Optional[str] = None,
    prefetched_spec: Optional[dict] = None,
) -> None:
    """Recursively check if a target workflow creates a circular call chain.

    Raises ``ExecutionGraphStructureError`` if a cycle is found. Silently
    returns on any fetch error — the runtime check will catch those cases.

    When ``prefetched_spec`` is provided it is used directly instead of
    fetching the spec again.  The caller in
    ``validate_workflow_caller_no_circular_references`` already fetches the
    spec for the top-level workflow, so passing it here avoids a duplicate
    API call.
    """
    from inference.core.workflows.errors import ExecutionGraphStructureError

    if target_workflow_id in visited:
        chain = " -> ".join([*visited, target_workflow_id])
        raise ExecutionGraphStructureError(
            public_message=(
                f"Circular workflow call detected at compile time: {chain}. "
                f"This would cause infinite recursion at runtime."
            ),
            context="workflow_compilation | workflow_caller_validation",
        )
    if prefetched_spec is not None:
        spec = prefetched_spec
    else:
        try:
            spec = get_workflow_specification(
                api_key=api_key,
                workspace_id=workspace_name,
                workflow_id=target_workflow_id,
                workflow_version_id=target_workflow_version_id,
            )
        except Exception as error:
            logging.debug(
                "Could not fetch workflow '%s' for circular dependency check (error type: %s)",
                target_workflow_id,
                type(error).__name__,
            )
            return
    nested_workflows = _extract_workflow_caller_ids_from_spec(spec=spec)
    if not nested_workflows:
        return
    visited_with_current = visited | {target_workflow_id}
    for nested_id, nested_version_id in nested_workflows:
        _check_workflow_for_circular_references(
            api_key=api_key,
            workspace_name=workspace_name,
            target_workflow_id=nested_id,
            target_workflow_version_id=nested_version_id,
            visited=visited_with_current,
        )


def _extract_workflow_caller_ids_from_spec(
    spec: dict,
) -> List[Tuple[str, Optional[str]]]:
    """Extract static (workflow_id, workflow_version_id) tuples from a raw
    workflow specification.  Only includes entries where ``workflow_id`` is a
    literal string (not a selector).  ``workflow_version_id`` is included when
    it is a literal string, otherwise ``None``."""
    steps = spec.get("steps", [])
    result: List[Tuple[str, Optional[str]]] = []
    for step in steps:
        step_type = step.get("type", "")
        if step_type != WORKFLOW_CALLER_BLOCK_TYPE:
            continue
        workflow_id = step.get("workflow_id", "")
        if not isinstance(workflow_id, str) or workflow_id.startswith("$"):
            continue
        version_id = step.get("workflow_version_id")
        if isinstance(version_id, str) and version_id.startswith("$"):
            version_id = None
        result.append((workflow_id, version_id))
    return result


# ---------------------------------------------------------------------------
# Output interface resolution
# ---------------------------------------------------------------------------


def _resolve_and_cache_workflow_outputs(
    cache_key: Tuple[str, Optional[str]],
    spec: dict,
) -> None:
    """Resolve the target workflow's output interface from its spec and cache
    the result in ``_RESOLVED_WORKFLOW_OUTPUTS``.

    Tries ``describe_workflow_outputs()`` for proper Kind resolution. Falls
    back to extracting output names with ``WILDCARD_KIND`` on any error.
    """
    try:
        outputs_description = _describe_outputs_from_spec(spec=spec)
    except Exception as error:
        logging.debug(
            f"Could not resolve output kinds for workflow '{cache_key}' "
            f"via describe_workflow_outputs: {error}. "
            f"Falling back to output names with WILDCARD_KIND."
        )
        outputs_description = None

    if outputs_description is not None:
        resolved = _convert_output_descriptions_to_kinds(
            outputs_description=outputs_description,
        )
    else:
        resolved = _extract_output_names_with_wildcard(spec=spec)

    if resolved:
        _RESOLVED_WORKFLOW_OUTPUTS[cache_key] = resolved


def _describe_outputs_from_spec(
    spec: dict,
) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
    """Call the standard output discovery to resolve output kinds."""
    from inference.core.workflows.execution_engine.v1.introspection.outputs_discovery import (
        describe_workflow_outputs,
    )

    return describe_workflow_outputs(definition=spec)


def _convert_output_descriptions_to_kinds(
    outputs_description: Dict[str, Union[List[str], Dict[str, List[str]]]],
) -> Dict[str, List[Kind]]:
    """Convert ``describe_workflow_outputs()`` result (kind name strings) to
    ``Kind`` objects."""
    kind_name_map = _build_kind_name_map()
    result: Dict[str, List[Kind]] = {}
    for name, kind_info in outputs_description.items():
        if isinstance(kind_info, dict):
            # Wildcard selector ($steps.step.*) — nested structure.
            # Represent as DICTIONARY_KIND since we can't express nested
            # typed properties in a single OutputDefinition.
            result[name] = [DICTIONARY_KIND]
        else:
            kinds = [
                kind_name_map[k]
                for k in kind_info
                if k in kind_name_map
            ]
            result[name] = kinds if kinds else [WILDCARD_KIND]
    return result


def _extract_output_names_with_wildcard(
    spec: dict,
) -> Dict[str, List[Kind]]:
    """Fallback: extract just the output names and assign WILDCARD_KIND."""
    outputs = spec.get("outputs", [])
    result: Dict[str, List[Kind]] = {}
    for output in outputs:
        name = output.get("name")
        if name:
            result[name] = [WILDCARD_KIND]
    return result


@lru_cache(maxsize=1)
def _build_kind_name_map() -> Dict[str, Kind]:
    """Build a mapping from kind name to Kind object using the loader.

    Result is cached since the set of registered kinds does not change
    within a process.
    """
    from inference.core.workflows.core_steps.loader import load_kinds

    return {kind.name: kind for kind in load_kinds()}


# ---------------------------------------------------------------------------
# Input interface resolution
# ---------------------------------------------------------------------------


def _resolve_and_cache_workflow_inputs(
    cache_key: Tuple[str, Optional[str]],
    spec: dict,
) -> None:
    """Parse the target workflow's input interface from its spec and cache
    the result in ``_RESOLVED_WORKFLOW_INPUTS``.

    Each input is stored as a dict with keys: name, type, has_default.
    """
    raw_inputs = spec.get("inputs", [])
    parsed: List[Dict[str, Any]] = []
    for inp in raw_inputs:
        name = inp.get("name")
        if not name:
            continue
        inp_type = inp.get("type", "")
        parsed.append({
            "name": name,
            "type": inp_type,
            "has_default": "default_value" in inp,
        })
    _RESOLVED_WORKFLOW_INPUTS[cache_key] = parsed


def _validate_required_inputs(
    workflow_id: str,
    step_manifest: BlockManifest,
    workflow_version_id: Optional[str] = None,
) -> None:
    """Validate that the workflow caller step provides all required inputs
    that the target workflow expects.

    Raises ``ExecutionGraphStructureError`` if:
    - The target workflow has required parameters (no default) not covered by
      ``inputs``.
    - The user declared ``input_definitions`` but ``inputs`` does not
      cover all declared input names.
    """
    from inference.core.workflows.errors import ExecutionGraphStructureError

    input_keys = set(step_manifest.inputs.keys())

    # Validate user-declared input_definitions: every declared input must be
    # present in inputs.
    if step_manifest.input_definitions:
        missing_declared = [
            name for name in step_manifest.input_definitions
            if name not in input_keys
        ]
        if missing_declared:
            raise ExecutionGraphStructureError(
                public_message=(
                    f"Workflow caller step '{step_manifest.name}' declares "
                    f"input_definitions for {list(step_manifest.input_definitions.keys())}, "
                    f"but input(s) {missing_declared} are not provided in "
                    f"'inputs'."
                ),
                context="workflow_compilation | workflow_caller_validation",
            )

    cache_key = _make_cache_key(workflow_id, workflow_version_id)
    resolved_inputs = _RESOLVED_WORKFLOW_INPUTS.get(cache_key)
    if resolved_inputs is None:
        return

    missing = []
    for inp in resolved_inputs:
        if inp["has_default"]:
            continue
        if inp["name"] not in input_keys:
            missing.append(inp["name"])
    if missing:
        raise ExecutionGraphStructureError(
            public_message=(
                f"Target workflow '{workflow_id}' requires input(s) "
                f"{missing} (no default values), but they are not provided "
                f"in 'inputs' on workflow_caller step "
                f"'{step_manifest.name}'."
            ),
            context="workflow_compilation | workflow_caller_validation",
        )


# ---------------------------------------------------------------------------
# Output resolution for runtime
# ---------------------------------------------------------------------------


def _resolve_output_kinds_for_run(
    output_definitions: Dict[str, List[str]],
    workflow_id: str,
    workflow_version_id: Optional[str] = None,
) -> Optional[Dict[str, List[Kind]]]:
    """Determine which outputs to extract and their Kind objects.

    Priority:
    1. User-declared ``output_definitions`` (explicit kind names)
    2. Auto-discovered outputs from ``_RESOLVED_WORKFLOW_OUTPUTS`` cache
    3. Returns ``None`` → caller should fall back to generic ``result`` dict
    """
    if output_definitions:
        kind_name_map = _build_kind_name_map()
        result: Dict[str, List[Kind]] = {}
        for name, kind_names in output_definitions.items():
            kinds = [
                kind_name_map[k] for k in kind_names if k in kind_name_map
            ]
            result[name] = kinds or [WILDCARD_KIND]
        return result
    cache_key = _make_cache_key(workflow_id, workflow_version_id)
    resolved = _RESOLVED_WORKFLOW_OUTPUTS.get(cache_key)
    if resolved is not None:
        return resolved
    return None


# ---------------------------------------------------------------------------
# Output deserialization
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_kinds_deserializers_map() -> Dict[str, Callable[[str, Any], Any]]:
    """Build the map of kind name -> deserializer function.

    Result is cached since the set of registered deserializers does not
    change within a process.
    """
    from inference.core.workflows.execution_engine.introspection.blocks_loader import (
        load_kinds_deserializers,
    )

    return load_kinds_deserializers()


def _deserialize_output_value(
    output_name: str,
    raw_value: Any,
    output_kinds: List[Kind],
    deserializers: Dict[str, Callable[[str, Any], Any]],
) -> Any:
    """Deserialize a single output value using the deserializer matching its
    resolved kind.

    For outputs with known kinds, deserialization is required — failure raises
    an error so downstream blocks don't receive unexpected raw data. For
    WILDCARD_KIND outputs the raw value is returned as-is since we don't know
    the expected type.
    """
    if raw_value is None:
        return None
    if all(kind == WILDCARD_KIND for kind in output_kinds):
        return raw_value
    errors: List[Tuple[str, Exception]] = []
    for kind in output_kinds:
        if kind == WILDCARD_KIND:
            continue
        if kind.name not in deserializers:
            continue
        try:
            return deserializers[kind.name](output_name, raw_value)
        except Exception as error:
            errors.append((kind.name, error))
            continue
    if errors:
        error_details = "; ".join(
            f"{kind}: {error}" for kind, error in errors
        )
        raise RuntimeError(
            f"Failed to deserialize output '{output_name}' from target "
            f"workflow. Tried deserializers for kinds "
            f"{[k for k, _ in errors]} but all failed: {error_details}. "
            f"This likely means the target workflow's output format doesn't "
            f"match the expected kind."
        )
    return raw_value
