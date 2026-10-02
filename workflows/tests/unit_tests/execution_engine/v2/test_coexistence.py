"""Coexistence evidence: the explicit V2 entry point leaves the legacy V1
engine, its default selection and its block discovery unchanged, in both
import orders, while a real legacy static-crop workflow still executes.

The subprocess cases import the full legacy catalogue and need its
dependencies (NumPy, OpenCV, ...) installed; they take a few seconds each.
"""

import json
import os
import subprocess
import sys

import pytest
from packaging.version import Version
from roboflow_workflows.execution_engine.v2 import (
    Batch,
    Catalogue,
    WorkflowCompileError,
    compile_workflow,
)

V2_PACKAGE = "roboflow_workflows.execution_engine.v2"

CHILD = r"""
import json
import sys

ORDER = sys.argv[1]


def observe_v1():
    import numpy as np
    from roboflow_workflows.core_steps.loader import load_blocks
    from roboflow_workflows.execution_engine.core import (
        REGISTERED_ENGINES,
        ExecutionEngine,
        get_available_versions,
        retrieve_requested_execution_engine_version,
    )
    from roboflow_workflows.execution_engine.introspection import blocks_loader

    definition = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/absolute_static_crop@v1",
                "name": "crop",
                "images": "$inputs.image",
                "x_center": 3,
                "y_center": 3,
                "width": 4,
                "height": 4,
            }
        ],
        "outputs": [{"type": "JsonField", "name": "crop", "selector": "$steps.crop.crops"}],
    }
    image = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3)
    engine = ExecutionEngine.init(workflow_definition=definition)
    results = engine.run(runtime_parameters={"image": image})
    actual = results[0]["crop"].numpy_image
    blocks_loader.clear_caches()
    discovered = blocks_loader.describe_available_blocks(dynamic_blocks=[])
    identifiers = sorted(block.manifest_type_identifier for block in discovered.blocks)
    return {
        "registered": sorted(str(version) for version in REGISTERED_ENGINES),
        "versions": get_available_versions(),
        "default": str(retrieve_requested_execution_engine_version({})),
        "block_count": len(load_blocks()),
        "discovered_count": len(identifiers),
        "discovered_digest": identifiers[:3] + identifiers[-3:],
        "has_v2_identifiers": any(identifier.startswith("v2/") for identifier in identifiers),
        "crop_shape": list(actual.shape),
        "pixel_equality": bool(np.array_equal(actual, image[1:5, 1:5])),
    }


def observe_v2():
    from pydantic import Field
    from roboflow_workflows.execution_engine.v2 import (
        Block,
        BlockParams,
        Catalogue,
        Kind,
        Output,
        Ref,
        compile_workflow,
    )

    number = Kind("number", validate=lambda value: isinstance(value, int))

    class Double(Block):
        type = "demo/double"
        outputs = {"out": Output(number)}

        class Params(BlockParams):
            value: Ref(number) = Field(description="Number to double.")

        def run(self, *, value):
            return {"out": value * 2}

    plan = compile_workflow(
        {
            "version": "2.0",
            "inputs": [{"type": "WorkflowBatchInput", "name": "values", "kind": ["number"]}],
            "steps": [{"name": "d", "type": "demo/double", "value": "$inputs.values"}],
            "outputs": [{"name": "out", "selector": "$steps.d.out"}],
        },
        catalogue=Catalogue([Double]),
    )
    result = plan.create_session().run({"values": [1, 2, 3]})
    return {
        "doubled": [row["out"] for row in result.rows()],
        "complete": all(status == "complete" for status in result.statuses.values()),
        "v2_blocks_loaded": "roboflow_workflows.execution_engine.v2.blocks" in sys.modules,
    }


report = {"order": ORDER}
if ORDER == "v1-then-v2":
    report["v1_before"] = observe_v1()
    report["v2"] = observe_v2()
    report["v1_after"] = observe_v1()
else:
    report["v2"] = observe_v2()
    legacy_modules = (
        "roboflow_workflows.execution_engine.v1",
        "roboflow_workflows.execution_engine.core",
        "roboflow_workflows.core_steps.loader",
    )
    report["legacy_loaded_by_v2"] = sorted(
        name for name in legacy_modules if name in sys.modules
    )
    report["v1_after"] = observe_v1()
print(json.dumps(report))
"""


def _run_child(order: str) -> dict:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(sys.path),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", CHILD, order],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    return report


def _expected_v1(report_entry: dict) -> None:
    from roboflow_workflows.execution_engine.v1.core import EXECUTION_ENGINE_V1_VERSION

    assert report_entry["registered"] == [str(EXECUTION_ENGINE_V1_VERSION)]
    assert report_entry["versions"] == [str(EXECUTION_ENGINE_V1_VERSION)]
    assert report_entry["default"] == str(EXECUTION_ENGINE_V1_VERSION)
    assert report_entry["block_count"] > 0
    assert report_entry["discovered_count"] > 0
    assert report_entry["has_v2_identifiers"] is False
    assert report_entry["crop_shape"] == [4, 4, 3]
    assert report_entry["pixel_equality"] is True


def test_v1_then_v2_import_order_leaves_legacy_engine_unchanged() -> None:
    report = _run_child("v1-then-v2")

    assert report["v1_before"] == report["v1_after"]
    _expected_v1(report["v1_after"])
    assert report["v2"] == {
        "doubled": [2, 4, 6],
        "complete": True,
        "v2_blocks_loaded": False,
    }


def test_v2_then_v1_import_order_matches_v1_first_observation() -> None:
    v2_first = _run_child("v2-then-v1")
    v1_first = _run_child("v1-then-v2")

    assert v2_first["v1_after"] == v1_first["v1_after"]
    _expected_v1(v2_first["v1_after"])
    assert v2_first["v2"]["doubled"] == [2, 4, 6]
    assert (
        v2_first["v2"]["v2_blocks_loaded"] is False
    ), "generic core loads no native blocks"
    assert v2_first["legacy_loaded_by_v2"] == [], "V2 must not import the V1 engine"


def test_in_process_v2_import_does_not_register_with_legacy_dispatcher() -> None:
    from roboflow_workflows.errors import NotSupportedExecutionEngineError
    from roboflow_workflows.execution_engine import core
    from roboflow_workflows.execution_engine.v1.core import EXECUTION_ENGINE_V1_VERSION

    # Whether the generic core loads native blocks is asserted in the fresh
    # subprocesses above; this pytest process may already have imported them
    # through test_blocks.py.
    assert V2_PACKAGE in sys.modules
    assert set(core.REGISTERED_ENGINES) == {EXECUTION_ENGINE_V1_VERSION}
    assert (
        core.retrieve_requested_execution_engine_version({})
        == EXECUTION_ENGINE_V1_VERSION
    )
    assert core.get_available_versions() == [str(EXECUTION_ENGINE_V1_VERSION)]
    with pytest.raises(NotSupportedExecutionEngineError):
        core._select_execution_engine(requested_engine_version=Version("2.0.0"))


def test_v2_compiler_rejects_v1_definitions_instead_of_falling_back() -> None:
    legacy = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [],
        "outputs": [],
    }

    with pytest.raises(WorkflowCompileError, match="must be '2.0' for the V2 engine"):
        compile_workflow(legacy, catalogue=Catalogue())
    assert Batch.empty().indices == ()
