"""Hosts use the same explicit V2 session boundary in every execution mode."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import roboflow_workflows
from pydantic import Field
from roboflow_workflows.v2 import engine as v2
from roboflow_workflows.v2.data.kinds import INTEGER_KIND
from roboflow_workflows.v2.engine import (
    Block,
    BlockParams,
    Catalogue,
    ContractError,
    Output,
    Ref,
    SessionClosedError,
    compile_workflow,
)


class _Double(Block):
    type = "test/host_double@v1"
    outputs = {"value": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref(INTEGER_KIND) = Field(description="The integer to double.")

    def run(self, *, value: int) -> dict:
        return {"value": value * 2}


def _plan():
    plan = compile_workflow(
        {
            "version": "2.0",
            "inputs": [
                {"type": "WorkflowParameter", "name": "value", "kind": ["integer"]}
            ],
            "steps": [
                {"type": _Double.type, "name": "double", "value": "$inputs.value"}
            ],
            "outputs": [{"name": "value", "selector": "$steps.double.value"}],
        },
        catalogue=Catalogue([_Double]),
    )

    return plan


def test_session_context_closes_after_a_successful_run():
    with _plan().create_session() as session:
        assert session.run({"value": 7}).rows() == [{"value": 14}]

    with pytest.raises(SessionClosedError):
        session.run({"value": 8})
    with pytest.raises(SessionClosedError):
        with session:
            pytest.fail("A closed session must not enter another context")


def test_session_context_closes_without_swallowing_body_failure():
    session = _plan().create_session()
    with pytest.raises(ValueError, match="host failed"):
        with session:
            session.run({"value": 1})
            raise ValueError("host failed")

    assert session.closed


def test_pipeline_finishes_before_enclosing_session_closes():
    with _plan().create_session() as session:
        with session.pipeline() as pipeline:
            future = pipeline.submit({"value": 9})
            assert future.result(timeout=5).rows() == [{"value": 18}]

    assert session.closed


def test_context_exit_does_not_cancel_an_unfinished_pipeline():
    session = _plan().create_session()
    with session.pipeline() as pipeline:
        with pytest.raises(ContractError, match="pipeline"):
            with session:
                pass
        assert not session.closed
        assert pipeline.submit({"value": 4}).result(timeout=5).rows() == [{"value": 8}]

    session.close()
    assert session.closed


def test_public_host_errors_and_receipt_keep_generic_import_light(tmp_path):
    script = """
import importlib.abc
import sys

forbidden = ('inference', 'inference_models', 'inference_sdk', 'torch',
             'numpy', 'cv2', 'redis', 'streamvision')
attempted = []
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            attempted.append(fullname)
            raise RuntimeError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Guard())
from roboflow_workflows.v2 import engine as v2
assert not any(n.startswith(v2.__name__ + '.updates') for n in sys.modules)
from roboflow_workflows.v2.engine import (
    GraphUpdateError, ResourceError, SessionClosedError, StepExecutionError,
    UpdateReceipt,
)
assert issubclass(SessionClosedError, v2.ContractError)
assert issubclass(GraphUpdateError, v2.ContractError)
assert issubclass(ResourceError, v2.ContractError)
assert issubclass(StepExecutionError, v2.WorkflowExecutionError)
assert UpdateReceipt.__name__ == 'UpdateReceipt'
assert not attempted, attempted
"""
    environment = dict(os.environ)
    # The directory holding the ``roboflow_workflows`` package under test.
    environment["PYTHONPATH"] = str(
        Path(roboflow_workflows.__file__).resolve().parent.parent
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
