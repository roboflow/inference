"""Blocks and one compiled workflow shared by the introspection tests.

Every block refuses construction: introspection must answer from class
declarations and the plan alone.

The workflow::

    values ──▶ split (expands parts) ──▶ collect (Group of parts) ──▶ total
                        │                        │
                        └──────▶ detect ◀────────┘ confidence (parent level)
                        │          model_id = $inputs.model (root selector)
                        ├──▶ positive ──▶ gate ──controls──▶ child (nested)
                        └──────────────────────────────────▶ child/detect
                                                    model_id bound to literal
    pick: outputs named by its literal ``keys`` configuration
    sentinel: dynamic custom Python whose module top writes a file
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field
from roboflow_workflows.execution_engine.entities.workload import (
    RuntimeRestriction,
    Severity,
    WorkOperation,
)
from roboflow_workflows.execution_engine.v2 import (
    Batch,
    Block,
    BlockParams,
    Catalogue,
    CompiledWorkflow,
    DependentResource,
    Group,
    Kind,
    Output,
    Ref,
    compile_workflow,
)
from roboflow_workflows.execution_engine.v2.blocks.control import ContinueIfBlock
from roboflow_workflows.execution_engine.v2.kinds import (
    BOOLEAN_KIND,
    BUILTIN_KINDS,
    DICTIONARY_KIND,
    FLOAT_KIND,
    STRING_KIND,
)

PREDICTION_KIND = Kind(name="prediction")


class NotConstructible(Block):
    """Base whose construction fails, to prove inspection never constructs."""

    def __init__(self):
        raise AssertionError(f"{type(self).__name__} was constructed")


class Split(NotConstructible):
    """Expand each value into parts."""

    type = "demo/split@v1"
    outputs = {"part": Output(FLOAT_KIND, expand="parts")}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value: float) -> dict:
        return {"part": Batch.of([value, value])}


class Collect(NotConstructible):
    """Collapse the parts of each value."""

    type = "demo/collect@v1"
    outputs = {"total": Output(FLOAT_KIND)}

    class Params(BlockParams):
        parts: Group(FLOAT_KIND)

    def run(self, *, parts: Batch) -> dict:
        return {"total": sum(parts)}


class Positive(NotConstructible):
    """Report whether a value is positive."""

    type = "demo/positive@v1"
    outputs = {"ok": Output(BOOLEAN_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value: float) -> dict:
        return {"ok": value > 0}


class Detect(NotConstructible):
    """Model-like block declaring its model as a resource."""

    type = "demo/detect@v1"
    aliases = ("demo/detect",)
    outputs = {"predictions": Output(PREDICTION_KIND)}

    class Params(BlockParams):
        image: Ref(FLOAT_KIND)
        model_id: str | Ref(STRING_KIND) = "default-model"
        confidence: float | Ref(FLOAT_KIND) = 0.5
        classes: List[str] = Field(default_factory=list)
        note: Optional[str] = None

    def run(self, **kwargs: Any) -> dict:
        return {"predictions": []}

    @classmethod
    def discover_dependent_resources(cls, params: BlockParams) -> list:
        return [
            DependentResource(
                resource_type="roboflow_platform_model", identifier=params.model_id
            )
        ]

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> list:
        return [WorkOperation.MODEL_INFERENCE]

    @classmethod
    def discover_restrictions(cls, params: BlockParams) -> list:
        return []


class Undeclared(NotConstructible):
    """Declares no workload at all."""

    type = "demo/undeclared@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value: Any) -> dict:
        return {"value": value}


class Faulty(NotConstructible):
    """Workload hooks that fail in three different ways."""

    type = "demo/faulty@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value: Any) -> dict:
        return {"value": value}

    @classmethod
    def discover_dependent_resources(cls, params: BlockParams) -> list:
        return [DependentResource(resource_type="storage_bucket", identifier="  ")]

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> list:
        raise RuntimeError("secret token abc123 must not leak")

    @classmethod
    def discover_restrictions(cls, params: BlockParams) -> list:
        return ["not a restriction"]


class Restricted(NotConstructible):
    """Declares one portable restriction."""

    type = "demo/restricted@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value: Any) -> dict:
        return {"value": value}

    @classmethod
    def discover_restrictions(cls, params: BlockParams) -> list:
        return [
            RuntimeRestriction(
                severity=Severity.SOFT,
                code="needs_gpu",
                note="Slow on CPU.",
                applies_to_configuration={"device": "cpu"},
            )
        ]


class Pick(NotConstructible):
    """Outputs named by configuration: one per requested key."""

    type = "demo/pick@v1"
    output_fields = ("keys",)

    class Params(BlockParams):
        record: Ref(DICTIONARY_KIND)
        keys: List[str]

    @classmethod
    def describe_outputs(cls, params: BlockParams) -> Dict[str, Output]:
        return {key: Output() for key in params.keys}

    def run(self, *, record: dict, keys: List[str]) -> dict:
        return {key: record.get(key) for key in keys}


CATALOGUE = Catalogue(
    [
        Split,
        Collect,
        Positive,
        Detect,
        Undeclared,
        Faulty,
        Restricted,
        Pick,
        ContinueIfBlock,
    ],
    kinds=BUILTIN_KINDS,
    namespace="demo",
)


def sentinel_definition(sentinel: Path) -> Dict[str, Any]:
    """Dynamic block whose module top writes ``sentinel`` if ever executed."""
    run_code = (
        f"open({str(sentinel)!r}, 'w').write('executed')\n"
        "def run(self, value):\n"
        "    return {'value': value}\n"
    )
    definition = {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": "Sentinel",
            "description": "Custom Python that must never run during inspection.",
            "inputs": {
                "value": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["step_output", "input_parameter"],
                }
            },
            "outputs": {"value": {"type": "DynamicOutputDefinition", "kind": []}},
        },
        "code": {
            "type": "PythonCode",
            "run_function_code": run_code,
            "imports": ["import numpy as np", "from json import dumps"],
        },
    }

    return definition


CHILD = {
    "version": "2.0",
    "inputs": [
        {"type": "WorkflowBatchInput", "name": "image", "kind": ["float"]},
        {"type": "WorkflowParameter", "name": "model", "kind": ["string"]},
    ],
    "steps": [
        {
            "type": "demo/detect",
            "name": "detect",
            "image": "$inputs.image",
            "model_id": "$inputs.model",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.detect.predictions",
        }
    ],
}


def compile_fixture(sentinel: Path) -> CompiledWorkflow:
    """Compile the module's workflow; local code stays disallowed."""
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]},
            {
                "type": "WorkflowParameter",
                "name": "model",
                "kind": ["string"],
                "default_value": "root-default-model",
            },
            {"type": "WorkflowParameter", "name": "record", "kind": ["dictionary"]},
        ],
        "dynamic_blocks_definitions": [sentinel_definition(sentinel)],
        "steps": [
            {"type": "demo/split@v1", "name": "split", "value": "$inputs.values"},
            {
                "type": "demo/collect@v1",
                "name": "collect",
                "parts": "$steps.split.part",
            },
            {
                "type": "demo/detect@v1",
                "name": "detect",
                "image": "$steps.split.part",
                "model_id": "$inputs.model",
                "confidence": "$steps.collect.total",
                "note": None,
            },
            {"type": "demo/positive@v1", "name": "positive", "value": "$inputs.values"},
            {
                "type": "v2/continue_if",
                "name": "gate",
                "condition": "$steps.positive.ok",
                "next_steps": ["$steps.child"],
            },
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "child",
                "workflow_definition": CHILD,
                "parameter_bindings": {
                    "image": "$steps.split.part",
                    "model": "child-model",
                },
            },
            {
                "type": "demo/undeclared@v1",
                "name": "undeclared",
                "value": "$inputs.values",
            },
            {"type": "demo/faulty@v1", "name": "faulty", "value": "$inputs.values"},
            {
                "type": "demo/restricted@v1",
                "name": "restricted",
                "value": "$inputs.values",
            },
            {"type": "Sentinel", "name": "custom", "value": "$inputs.values"},
            {
                "type": "demo/pick@v1",
                "name": "pick",
                "record": "$inputs.record",
                "keys": ["left", "right"],
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "total", "selector": "$steps.collect.total"},
            {
                "type": "JsonField",
                "name": "child",
                "selector": "$steps.child.predictions",
            },
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


MIDDLE = {
    "version": "2.0",
    "inputs": [
        {"type": "WorkflowBatchInput", "name": "image", "kind": ["float"]},
        {
            "type": "WorkflowParameter",
            "name": "model",
            "kind": ["string"],
            "default_value": "middle-default",
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/inner_workflow@v1",
            "name": "inner",
            "workflow_definition": CHILD,
            "parameter_bindings": {"image": "$inputs.image", "model": "$inputs.model"},
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.inner.predictions",
        },
        {"type": "JsonField", "name": "image", "selector": "$inputs.image"},
    ],
}


def compile_deep_fixture() -> CompiledWorkflow:
    """Two nesting levels; the middle ``model`` input is bound three ways.

    ``selected`` binds it to a root input, ``literal`` to a literal and
    ``defaulted`` leaves the middle workflow's default. ``echo`` forwards the
    middle workflow's own input directly as a workflow output.
    """

    def middle(name: str, **bindings: str) -> Dict[str, Any]:
        return {
            "type": "roboflow_core/inner_workflow@v1",
            "name": name,
            "workflow_definition": MIDDLE,
            "parameter_bindings": {"image": "$inputs.values", **bindings},
        }

    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]},
            {"type": "WorkflowParameter", "name": "model", "kind": ["string"]},
        ],
        "steps": [
            middle("selected", model="$inputs.model"),
            middle("literal", model="deep-literal"),
            middle("defaulted"),
        ],
        "outputs": [
            {"type": "JsonField", "name": "echo", "selector": "$steps.literal.image"},
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan
