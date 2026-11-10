"""Public compilation surfaces real warnings once and keeps strict rejection."""

import logging

import pytest
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    MutationConflictError,
    WorkflowCompileError,
)
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.support.compilation.fixture_blocks import (
    CATALOGUE,
    nested,
    parameter,
    step,
    workflow,
)

# Operator-visible logger name: the historic ``inference.*`` form of the module.
LOGGER = "inference.core.workflows.v2.engine.compilation"


def _conflict():
    definition = workflow(
        [
            step("increment", "change", value="$inputs.payload"),
            step("reader", "read", value="$inputs.payload"),
        ],
        inputs=[parameter("payload", kind=["dictionary"])],
    )

    return definition


def _records(caplog):
    records = [record for record in caplog.records if record.name == LOGGER]

    return records


@pytest.mark.parametrize("nested_child", [False, True])
def test_public_compilation_logs_each_mutation_warning_once(caplog, nested_child):
    """Keep inspectable warnings and expose nested diagnostics without duplicates.

    Args:
        caplog: Captured library logging.
        nested_child: Put the conflicting steps inside a nested workflow.
    """
    definition = _conflict()
    if nested_child:
        definition = workflow(
            [
                nested(
                    "child",
                    workflow_definition=definition,
                    parameter_bindings={"payload": "$inputs.payload"},
                )
            ],
            inputs=[parameter("payload", kind=["dictionary"])],
        )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        plan = compile_workflow(definition, catalogue=CATALOGUE)

    records = _records(caplog)
    assert len(records) == len(plan.warnings) == 1
    assert plan.warnings[0] in records[0].getMessage()
    assert (
        "$steps.child/change" in records[0].getMessage()
        if nested_child
        else "$steps.change" in records[0].getMessage()
    )
    assert records[0].levelno == logging.WARNING
    # A second public compilation is a separate diagnostic occurrence.
    caplog.clear()
    compile_workflow(definition, catalogue=CATALOGUE)
    assert len(_records(caplog)) == 1


def test_strict_mutation_error_does_not_emit_success_warning(caplog):
    """Reject strict conflicts without also reporting a successful compilation.

    Args:
        caplog: Captured library logging.
    """
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        with pytest.raises(MutationConflictError):
            compile_workflow(
                _conflict(),
                catalogue=CATALOGUE,
                options=CompileOptions(mutation_conflicts="error"),
            )

    assert not _records(caplog)


def test_later_compilation_failure_does_not_log_partial_plan(caplog):
    """Log only after all public compilation stages succeeded.

    Args:
        caplog: Captured library logging.
    """
    definition = _conflict()
    definition["controls"] = {"bad": {"type": "input", "input": "$inputs.missing"}}
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        with pytest.raises(WorkflowCompileError):
            compile_workflow(definition, catalogue=CATALOGUE)

    assert not _records(caplog)


def test_separate_handler_diagnostics_have_their_own_scope(caplog):
    """Surface separately compiled handler plans at the root success boundary.

    Args:
        caplog: Captured library logging.
    """
    definition = workflow([], inputs=[])
    definition["signals"] = [{"name": "sample", "fields": {"payload": "dictionary"}}]
    handler_workflow = _conflict()
    handler_workflow.pop("version")
    handler_workflow["inputs"] = [{"name": "payload", "kind": ["dictionary"]}]
    definition["handlers"] = [
        {
            "name": name,
            "on": "$signals.sample",
            "bindings": {"payload": "$event.payload"},
            "workflow": handler_workflow,
        }
        for name in ("first", "second")
    ]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        plan = compile_workflow(definition, catalogue=CATALOGUE)

    assert plan.warnings == ()
    records = _records(caplog)
    assert len(records) == 2
    messages = "\n".join(record.getMessage() for record in records)
    assert "handler 'first'" in messages and "handler 'second'" in messages
    assert all("is mutated in place" in record.getMessage() for record in records)
