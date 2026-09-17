"""ContextVars must flow from the calling thread into step workers - and stop there.

The HTTP interface owns one process-wide `ThreadPoolExecutor` handed to every
`ExecutionEngine.init(...)`, so step worker threads are reused across requests.
Two invariants, exercised through the real `ExecutionEngine.run(...)` rather
than the executor helper in isolation:

- a ContextVar bound in the calling thread is visible inside a step,
- a ContextVar set inside a step does not survive into the next run on the
  same executor.

`usage_billing_suppressed` is the ContextVar billing depends on, so it doubles
as the probe here.
"""

from concurrent.futures import ThreadPoolExecutor

from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.usage_tracking.decorator_helpers import usage_billing_suppressed

PROBE_BLOCK_CODE = """
def run(self, value) -> BlockResult:
    from inference.usage_tracking.decorator_helpers import usage_billing_suppressed

    seen = usage_billing_suppressed.get()
    # Poison the worker thread; the next run must not observe it.
    usage_billing_suppressed.set(True)
    return {"result": seen}
"""

WORKFLOW_WITH_CONTEXT_PROBES = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "ContextProbe",
                "inputs": {
                    "value": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["input_parameter"],
                    },
                },
                "outputs": {"result": {"type": "DynamicOutputDefinition", "kind": []}},
            },
            "code": {"type": "PythonCode", "run_function_code": PROBE_BLOCK_CODE},
        },
    ],
    "steps": [
        {"type": "ContextProbe", "name": "first_probe", "value": "$inputs.value"},
        {"type": "ContextProbe", "name": "second_probe", "value": "$inputs.value"},
    ],
    "outputs": [
        {"type": "JsonField", "name": "first", "selector": "$steps.first_probe.result"},
        {
            "type": "JsonField",
            "name": "second",
            "selector": "$steps.second_probe.result",
        },
    ],
}


def test_contextvars_reach_steps_and_do_not_leak_across_runs_on_shared_executor():
    # given: a single-worker external pool, so the second run provably reuses
    # the exact thread the first run's steps poisoned
    with ThreadPoolExecutor(max_workers=1) as shared_executor:
        execution_engine = ExecutionEngine.init(
            workflow_definition=WORKFLOW_WITH_CONTEXT_PROBES,
            init_parameters={"workflows_core.api_key": None},
            max_concurrent_steps=2,
            executor=shared_executor,
        )

        # when: the first run executes with the var bound in the calling thread,
        # the second with the calling thread back at the default
        token = usage_billing_suppressed.set(True)
        try:
            suppressed_run = execution_engine.run(runtime_parameters={"value": 1})
        finally:
            usage_billing_suppressed.reset(token)
        plain_run = execution_engine.run(runtime_parameters={"value": 1})

    # then: workers saw the caller's value, and neither the first run's caller
    # binding nor the in-step writes reached the second run
    assert suppressed_run == [{"first": True, "second": True}]
    assert plain_run == [{"first": False, "second": False}]
