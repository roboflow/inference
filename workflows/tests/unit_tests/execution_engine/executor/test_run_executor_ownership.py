"""Run-scoped executor ownership regression tests.

Before this change, `executor=None` hosts (standalone `roboflow-workflows`,
SDK, embedded usage) constructed a fresh `ThreadPoolExecutor` for EVERY DAG
wave of EVERY run. These tests pin the new contract:

* `executor=None` + multi-wave run -> exactly ONE pool is constructed,
  and it is shut down before the call returns (also on the exception path).
* a host-provided executor -> NO pool is constructed by the engine.
* results are identical with and without a host-provided executor.
* both supported early-termination paths (exception abort, conditional
  branch termination via `FlowControl`) tear the pool down exactly once
  and recover on the next run; the engine has no cancellation primitive.
* a failing task in a flat-dispatched wave is joined with its in-flight
  siblings before the exception leaves `run_steps_in_parallel` (the
  historical per-wave `with` block did the same), and the caller-owned
  pool is not shut down there.

Pool construction is observed at the `concurrent.futures.ThreadPoolExecutor`
class level so the assertion also catches the historical per-wave pool built
inside `run_steps_in_parallel` (the pre-fix construction site) - the test
fails with 3 pools on a 3-wave chain before the fix.
"""

import contextvars
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, List, Literal, Type

import networkx as nx
import pytest
from roboflow_workflows.errors import StepExecutionError
from roboflow_workflows.execution_engine.constants import (
    NODE_COMPILATION_OUTPUT_PROPERTY,
)
from roboflow_workflows.execution_engine.entities.base import (
    JsonField,
    OutputDefinition,
)
from roboflow_workflows.execution_engine.entities.types import WILDCARD_KIND
from roboflow_workflows.execution_engine.profiling.core import NullWorkflowsProfiler
from roboflow_workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
    CompiledWorkflow,
    DynamicStepInputDefinition,
    InitialisedStep,
    NodeCategory,
    NodeInputCategory,
    OutputNode,
    ParameterSpecification,
    ParsedWorkflowDefinition,
    StepNode,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.debug_logs import (
    current_debug_collector,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.workflow_debug import (
    current_debug_step_name,
    current_debug_trace,
)
from roboflow_workflows.execution_engine.v1.entities import FlowControl
from roboflow_workflows.execution_engine.v1.executor.core import (
    _run_workflow,
    flush_stream_pipeline_workflow,
)
from roboflow_workflows.execution_engine.v1.executor.utils import run_steps_in_parallel
from roboflow_workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class SeedManifest(WorkflowBlockManifest):
    type: Literal["test/seed@v1"]
    name: str

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="out")]


class SeedBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SeedManifest

    def run(self) -> BlockResult:
        return {"out": "seed"}


class RelayManifest(WorkflowBlockManifest):
    type: Literal["test/relay@v1"]
    name: str

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="consumed")]


class RelayBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return RelayManifest

    def run(self, predictions: Any) -> BlockResult:
        return {"consumed": predictions}


class ExplodingBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return RelayManifest

    def run(self, predictions: Any) -> BlockResult:
        raise RuntimeError("boom")


def _step_node(name: str, manifest: Any, input_data: Any = None) -> StepNode:
    return StepNode(
        node_category=NodeCategory.STEP_NODE,
        name=name,
        selector=f"$steps.{name}",
        data_lineage=[],
        step_manifest=manifest,
        **({"input_data": input_data} if input_data else {}),
    )


def _relay_input(previous_step: str, previous_output: str) -> Any:
    return {
        "predictions": DynamicStepInputDefinition(
            parameter_specification=ParameterSpecification(
                parameter_name="predictions"
            ),
            category=NodeInputCategory.NON_BATCH_STEP_OUTPUT,
            data_lineage=[],
            selector=f"$steps.{previous_step}.{previous_output}",
        )
    }


def _three_wave_workflow(second_relay_class: Type[WorkflowBlock] = RelayBlock):
    """seed -> relay_1 -> relay_2: exactly three DAG waves."""
    seed_manifest = SeedManifest(type="test/seed@v1", name="seed")
    relay_1_manifest = RelayManifest(type="test/relay@v1", name="relay_1")
    relay_2_manifest = RelayManifest(type="test/relay@v1", name="relay_2")
    workflow_output = JsonField(
        type="JsonField",
        name="result",
        selector="$steps.relay_2.consumed",
    )

    graph = nx.DiGraph()
    graph.add_node(
        "$steps.seed",
        **{
            NODE_COMPILATION_OUTPUT_PROPERTY: _step_node("seed", seed_manifest),
        },
    )
    graph.add_node(
        "$steps.relay_1",
        **{
            NODE_COMPILATION_OUTPUT_PROPERTY: _step_node(
                "relay_1", relay_1_manifest, _relay_input("seed", "out")
            ),
        },
    )
    graph.add_node(
        "$steps.relay_2",
        **{
            NODE_COMPILATION_OUTPUT_PROPERTY: _step_node(
                "relay_2", relay_2_manifest, _relay_input("relay_1", "consumed")
            ),
        },
    )
    graph.add_node(
        "$outputs.result",
        **{
            NODE_COMPILATION_OUTPUT_PROPERTY: OutputNode(
                node_category=NodeCategory.OUTPUT_NODE,
                name="result",
                selector="$outputs.result",
                data_lineage=[],
                output_manifest=workflow_output,
                kind=[WILDCARD_KIND],
            )
        },
    )
    graph.add_edge("$steps.seed", "$steps.relay_1")
    graph.add_edge("$steps.relay_1", "$steps.relay_2")
    graph.add_edge("$steps.relay_2", "$outputs.result")

    def spec(identifier: str, block_class: Type[WorkflowBlock]) -> BlockSpecification:
        manifest_class = block_class.get_manifest()
        return BlockSpecification(
            block_source="test",
            identifier=identifier,
            block_class=block_class,
            manifest_class=manifest_class,
        )

    return CompiledWorkflow(
        workflow_definition=ParsedWorkflowDefinition(
            version="1.0",
            inputs=[],
            steps=[seed_manifest, relay_1_manifest, relay_2_manifest],
            outputs=[workflow_output],
        ),
        execution_graph=graph,
        steps={
            "seed": InitialisedStep(
                block_specification=spec("test/seed@v1", SeedBlock),
                manifest=seed_manifest,
                step=SeedBlock(),
            ),
            "relay_1": InitialisedStep(
                block_specification=spec("test/relay@v1", RelayBlock),
                manifest=relay_1_manifest,
                step=RelayBlock(),
            ),
            "relay_2": InitialisedStep(
                block_specification=spec("test/relay@v1", second_relay_class),
                manifest=relay_2_manifest,
                step=second_relay_class(),
            ),
        },
        input_substitutions=[],
        workflow_json={},
        init_parameters={},
        kinds_serializers={},
        kinds_deserializers={},
    )


@pytest.fixture()
def created_pools(monkeypatch: pytest.MonkeyPatch) -> List[ThreadPoolExecutor]:
    created: List[ThreadPoolExecutor] = []
    original_init = ThreadPoolExecutor.__init__

    def counting_init(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)
        created.append(self)
        return result

    monkeypatch.setattr(ThreadPoolExecutor, "__init__", counting_init)
    return created


@pytest.fixture()
def wave_dispatch_sizes(monkeypatch: pytest.MonkeyPatch) -> List[int]:
    """Record the size of every dispatched batch, mechanism-agnostic.

    Consecutive `ThreadPoolExecutor.submit` calls form one batch, flushed at
    the first collected `Future.result()`. Flat (run-owned) dispatch submits
    a whole wave before consuming it -> one entry per wave. Batched host
    dispatch submits a chunk, collects it, then submits the next chunk
    (both via `map` internals) -> one entry per chunk. A regression that
    chunked the owned path would show [1, 2, 2, ...] instead of [1, 4].
    """
    sizes: List[int] = []
    pending: List[int] = []
    original_submit = ThreadPoolExecutor.submit
    original_result = Future.result

    def recording_submit(self, fn, *args, **kwargs):
        pending.append(1)
        return original_submit(self, fn, *args, **kwargs)

    def recording_result(self, *args, **kwargs):
        if pending:
            sizes.append(len(pending))
            pending.clear()
        return original_result(self, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "submit", recording_submit)
    monkeypatch.setattr(Future, "result", recording_result)
    return sizes


def _run(workflow: CompiledWorkflow, executor: Any = None):
    return _run_workflow(
        workflow=workflow,
        runtime_parameters={},
        max_concurrent_steps=2,
        kinds_serializers={},
        serialize_results=False,
        profiler=NullWorkflowsProfiler.init(),
        executor=executor,
    )


def test_multi_wave_run_without_executor_builds_single_run_scoped_pool(
    created_pools: List[ThreadPoolExecutor],
) -> None:
    workflow = _three_wave_workflow()

    result = _run(workflow, executor=None)

    assert result == [{"result": "seed"}]
    # Three DAG waves used to mean three ThreadPoolExecutor constructions.
    assert (
        len(created_pools) == 1
    ), f"expected exactly one run-scoped pool, observed {len(created_pools)}"
    owned = created_pools[0]
    assert owned._shutdown is True, "run-scoped pool must be shut down before return"
    assert not [
        t for t in owned._threads if t.is_alive()
    ], "run-scoped pool threads must be joined before return"


def test_multi_wave_run_with_host_executor_builds_no_pool(
    created_pools: List[ThreadPoolExecutor],
) -> None:
    workflow = _three_wave_workflow()
    with ThreadPoolExecutor(max_workers=3) as host_pool:
        # the counter already observed this pool being built by the TEST;
        # the engine must not add any pool of its own on top of it.
        assert created_pools == [host_pool]
        result = _run(workflow, executor=host_pool)
        assert result == [{"result": "seed"}]
        assert host_pool._shutdown is False, "host-owned executor must stay open"
        assert created_pools == [host_pool], (
            "engine must not construct pools when executor is provided: "
            f"{created_pools}"
        )


def test_results_identical_with_and_without_host_executor(
    created_pools: List[ThreadPoolExecutor],
) -> None:
    without = _run(_three_wave_workflow(), executor=None)
    with ThreadPoolExecutor(max_workers=3) as host_pool:
        with_host = _run(_three_wave_workflow(), executor=host_pool)
    assert without == with_host == [{"result": "seed"}]


def test_run_scoped_pool_is_torn_down_when_a_step_raises(
    created_pools: List[ThreadPoolExecutor],
) -> None:
    workflow = _three_wave_workflow(second_relay_class=ExplodingBlock)

    with pytest.raises(StepExecutionError):
        _run(workflow, executor=None)

    assert (
        len(created_pools) == 1
    ), f"expected exactly one run-scoped pool, observed {len(created_pools)}"
    owned = created_pools[0]
    assert owned._shutdown is True, "exception path must still shut the pool down"
    assert not [
        t for t in owned._threads if t.is_alive()
    ], "exception path must join pool threads before returning"


def test_flush_path_uses_single_run_scoped_pool(
    created_pools: List[ThreadPoolExecutor],
) -> None:
    workflow = _three_wave_workflow()
    # The flush path only re-runs steps downstream of flushed producers; the
    # seed block has no flush, so no waves run and no pool must be built.
    result = flush_stream_pipeline_workflow(
        workflow=workflow,
        runtime_parameters={},
        max_concurrent_steps=2,
        kinds_serializers={},
        serialize_results=False,
        profiler=NullWorkflowsProfiler.init(),
        executor=None,
    )
    assert (
        created_pools == []
    ), f"flush with no runnable downstream steps must not build a pool: {created_pools}"
    assert result == [{"result": None}]


def test_flush_path_downstream_wave_builds_single_pool(
    created_pools: List[ThreadPoolExecutor],
) -> None:
    """Use a flushing producer so the flush path runs downstream waves."""

    class FlushingSeedBlock(SeedBlock):
        def flush_stream_pipeline_outputs(self):
            return [([], [{"out": "tail"}])]

    workflow = _three_wave_workflow()
    workflow.steps["seed"] = InitialisedStep(
        block_specification=BlockSpecification(
            block_source="test",
            identifier="test/seed@v1",
            block_class=FlushingSeedBlock,
            manifest_class=SeedManifest,
        ),
        manifest=workflow.steps["seed"].manifest,
        step=FlushingSeedBlock(),
    )

    result = flush_stream_pipeline_workflow(
        workflow=workflow,
        runtime_parameters={},
        max_concurrent_steps=2,
        kinds_serializers={},
        serialize_results=False,
        profiler=NullWorkflowsProfiler.init(),
        executor=None,
    )

    # waves: relay_1, relay_2 -> historically two pools
    assert (
        len(created_pools) == 1
    ), f"expected exactly one run-scoped pool, observed {len(created_pools)}"
    assert created_pools[0]._shutdown is True
    assert result == [{"result": "tail"}]


def _wide_wave_workflow() -> CompiledWorkflow:
    """seed fans out to four parallel relays: a wave of size 4."""
    seed_manifest = SeedManifest(type="test/seed@v1", name="seed")
    relay_manifests = [
        RelayManifest(type="test/relay@v1", name=f"relay_{i}") for i in range(4)
    ]
    workflow_output = JsonField(
        type="JsonField",
        name="result",
        selector="$steps.relay_0.consumed",
    )

    graph = nx.DiGraph()
    graph.add_node(
        "$steps.seed",
        **{NODE_COMPILATION_OUTPUT_PROPERTY: _step_node("seed", seed_manifest)},
    )
    for manifest in relay_manifests:
        graph.add_node(
            f"$steps.{manifest.name}",
            **{
                NODE_COMPILATION_OUTPUT_PROPERTY: _step_node(
                    manifest.name, manifest, _relay_input("seed", "out")
                )
            },
        )
        graph.add_edge("$steps.seed", f"$steps.{manifest.name}")
    graph.add_node(
        "$outputs.result",
        **{
            NODE_COMPILATION_OUTPUT_PROPERTY: OutputNode(
                node_category=NodeCategory.OUTPUT_NODE,
                name="result",
                selector="$outputs.result",
                data_lineage=[],
                output_manifest=workflow_output,
                kind=[WILDCARD_KIND],
            )
        },
    )
    graph.add_edge("$steps.relay_0", "$outputs.result")

    def spec() -> BlockSpecification:
        return BlockSpecification(
            block_source="test",
            identifier="test/relay@v1",
            block_class=RelayBlock,
            manifest_class=RelayManifest,
        )

    steps = {
        "seed": InitialisedStep(
            block_specification=BlockSpecification(
                block_source="test",
                identifier="test/seed@v1",
                block_class=SeedBlock,
                manifest_class=SeedManifest,
            ),
            manifest=seed_manifest,
            step=SeedBlock(),
        )
    }
    for manifest in relay_manifests:
        steps[manifest.name] = InitialisedStep(
            block_specification=spec(),
            manifest=manifest,
            step=RelayBlock(),
        )

    return CompiledWorkflow(
        workflow_definition=ParsedWorkflowDefinition(
            version="1.0",
            inputs=[],
            steps=[seed_manifest, *relay_manifests],
            outputs=[workflow_output],
        ),
        execution_graph=graph,
        steps=steps,
        input_substitutions=[],
        workflow_json={},
        init_parameters={},
        kinds_serializers={},
        kinds_deserializers={},
    )


def test_owned_pool_dispatches_wide_wave_flat_like_the_old_per_wave_pool(
    created_pools: List[ThreadPoolExecutor],
    wave_dispatch_sizes: List[int],
) -> None:
    """executor=None historical scheduling: one flat dispatch per wave.

    Host-provided pools keep batched dispatch (server semantics):
    max_concurrent_steps=2 over the 4-wide wave -> chunks of 2.
    """
    result = _run(_wide_wave_workflow(), executor=None)
    assert result == [{"result": "seed"}]
    assert len(created_pools) == 1
    # waves: [seed] (size 1), [relay_0..relay_3] (size 4) - never chunked
    assert wave_dispatch_sizes == [1, 4], wave_dispatch_sizes

    wave_dispatch_sizes.clear()
    with ThreadPoolExecutor(max_workers=2) as host_pool:
        result = _run(_wide_wave_workflow(), executor=host_pool)
    assert result == [{"result": "seed"}]
    assert wave_dispatch_sizes == [1, 2, 2], wave_dispatch_sizes


def test_flat_dispatch_wave_joins_inflight_siblings_before_raising() -> None:
    """A step error may not escape while a sibling wave task is still running.

    The historical per-wave `with` block ran `shutdown(wait=True)` before the
    exception left `run_steps_in_parallel`, and `_run_workflow`'s outer
    `finally` joins too late for callers in between (e.g. `@execution_phase`
    closing `group_of_steps_execution` while a sibling still appends to
    `BaseWorkflowsProfiler._current_run_events`). Asserting only inside
    `_run_workflow` cannot catch this - its `finally` already joins - so this
    pins `run_steps_in_parallel(..., flat_dispatch=True)` directly.
    """
    sibling_started = threading.Event()
    sibling_finished = threading.Event()

    def fail_after_sibling_started() -> str:
        # Guarantees the sibling is RUNNING (so cancellation cannot silently
        # drop it) at the moment this step raises.
        assert sibling_started.wait(timeout=5), "sibling wave task never started"
        raise ValueError("boom")

    def slow_sibling() -> str:
        sibling_started.set()
        time.sleep(0.3)
        sibling_finished.set()
        return "ok"

    owned_pool = ThreadPoolExecutor(max_workers=2)
    try:
        with pytest.raises(ValueError, match="boom"):
            run_steps_in_parallel(
                steps=[fail_after_sibling_started, slow_sibling],
                max_workers=2,
                executor=owned_pool,
                flat_dispatch=True,
            )
        assert sibling_finished.is_set(), (
            "wave raised while an in-flight sibling was still running: "
            "run_steps_in_parallel must drain the wave before propagating"
        )
        # The drain joins tasks; it must not tear down the caller-owned pool.
        assert (
            owned_pool._shutdown is False
        ), "flat dispatch must not shut the caller-owned pool down"
    finally:
        owned_pool.shutdown(wait=True)


def test_deferred_flush_run_still_shuts_the_owned_pool_down(
    created_pools: List[ThreadPoolExecutor],
) -> None:
    workflow = _three_wave_workflow()
    result = _run_workflow(
        workflow=workflow,
        runtime_parameters={},
        max_concurrent_steps=2,
        kinds_serializers={},
        serialize_results=False,
        profiler=NullWorkflowsProfiler.init(),
        executor=None,
        defer_stream_pipeline_flush=True,
    )
    assert result == [{"result": "seed"}]
    assert len(created_pools) == 1
    assert created_pools[0]._shutdown is True
    assert not [t for t in created_pools[0]._threads if t.is_alive()]


def test_two_sequential_runs_use_two_run_scoped_pools(
    created_pools: List[ThreadPoolExecutor],
) -> None:
    first = _run(_three_wave_workflow(), executor=None)
    second = _run(_three_wave_workflow(), executor=None)
    assert first == second == [{"result": "seed"}]
    # pools are run-scoped, never cached on the engine or module
    assert (
        len(created_pools) == 2
    ), f"expected one fresh pool per run, observed {len(created_pools)}"
    assert all(pool._shutdown for pool in created_pools)


def test_exception_path_shuts_pool_before_closing_stream_pipelines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup ordering pin: every pool shutdown happens BEFORE
    close_stream_pipelines, and on the run-scoped path there is exactly one
    such shutdown. On unfixed base code the wave3 failure records THREE
    shutdowns (one per per-wave pool), so the count assertion fails there -
    the ordering half is the invariant both sides must keep.
    """
    events: List[str] = []

    original_shutdown = ThreadPoolExecutor.shutdown

    def recording_shutdown(self, *args, **kwargs):
        events.append("shutdown")
        return original_shutdown(self, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "shutdown", recording_shutdown)

    class ClosingExplodingBlock(ExplodingBlock):
        def close_stream_pipeline(self) -> None:
            events.append("close")

    workflow = _three_wave_workflow(second_relay_class=ClosingExplodingBlock)

    with pytest.raises(StepExecutionError):
        _run_workflow(
            workflow=workflow,
            runtime_parameters={},
            max_concurrent_steps=2,
            kinds_serializers={},
            serialize_results=False,
            profiler=NullWorkflowsProfiler.init(),
            executor=None,
            defer_stream_pipeline_flush=False,
        )

    # invariant: stream pipelines close LAST, after every pool shutdown
    assert events and events[-1] == "close", events
    assert all(e == "shutdown" for e in events[:-1]), events
    # run-scoped contract: exactly one pool is built and drained for the run
    assert (
        events.count("shutdown") == 1
    ), f"expected one owned-pool shutdown before pipeline close, got {events}"


# ------------------------------------------------------------------
# Worker-reuse semantics: run-scoped pools reuse OS threads across the
# waves of ONE run and destroy them at run end. The contract that must
# hold is isolation of FRAMEWORK-owned state, not thread freshness:
#   - every task runs in its own contextvars snapshot, and
#   - safe_execute_step unconditionally rebinds framework contextvars.
# Third-party threading.local set inside one step MAY be observed by a
# later step of the SAME run (same reuse profile hosted pools have had
# since #1717; the block contract never promised wave-fresh workers -
# state belongs to block instances / InstanceCache). Cross-run leakage
# of raw thread-locals is impossible on this path and IS pinned below.
#
# On early termination: the v1 executor has NO cancellation primitive
# (no cooperative cancel / future cancellation anywhere in the executor),
# so the supported early exits are (a) exception abort - a failing wave
# raises and later waves are never scheduled - and (b) conditional
# branch termination - a flow-control step returns FlowControl() and its
# downstream block bodies are discarded inside still-scheduled waves.
# Both unwind through the run-scoped pool's `finally` and are pinned at
# the bottom of this file; arbitrary third-party thread-local state
# cannot be cleared by the framework on either path (documented
# boundary above), so neither test asserts it - only framework-owned
# context, pool lifetime, and block non-execution are asserted.
# ------------------------------------------------------------------

_WAVE_PROBE: contextvars.ContextVar = contextvars.ContextVar(
    "wf_test_wave_probe", default="request-value"
)


class WaveSetterManifest(WorkflowBlockManifest):
    type: Literal["test/wave_setter@v1"]
    name: str

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="out")]


class WaveSetterBlock(WorkflowBlock):
    """Runs in wave 1: records the clean state it observes, then poisons
    its own task-local view plus (optionally) a raw thread-local."""

    def __init__(self) -> None:
        self.observed_probe: List[Any] = []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return WaveSetterManifest

    def run(self) -> BlockResult:
        self.observed_probe.append(_WAVE_PROBE.get())
        _WAVE_PROBE.set("wave1-poison")  # poisons only wave 1's own snapshot
        return {"out": "v"}


class WaveCheckerManifest(WorkflowBlockManifest):
    type: Literal["test/wave_checker@v1"]
    name: str

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="consumed")]


class WaveCheckerBlock(WorkflowBlock):
    """Runs in wave 2 (may share the wave-1 OS thread): records what a
    later wave observes of framework and probe state."""

    def __init__(self) -> None:
        self.observed_probe: List[Any] = []
        self.observed_step_name: List[Any] = []
        self.observed_collector: List[Any] = []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return WaveCheckerManifest

    def run(self, value: Any) -> BlockResult:
        self.observed_probe.append(_WAVE_PROBE.get())
        self.observed_step_name.append(current_debug_step_name.get())
        self.observed_collector.append(current_debug_collector.get())
        return {"consumed": value}


def _two_wave_workflow(
    setter: WorkflowBlock, checker: WorkflowBlock
) -> CompiledWorkflow:
    setter_manifest = WaveSetterManifest(type="test/wave_setter@v1", name="setter")
    checker_manifest = WaveCheckerManifest(type="test/wave_checker@v1", name="checker")
    workflow_output = JsonField(
        type="JsonField",
        name="result",
        selector="$steps.checker.consumed",
    )
    graph = nx.DiGraph()
    graph.add_node(
        "$steps.setter",
        **{NODE_COMPILATION_OUTPUT_PROPERTY: _step_node("setter", setter_manifest)},
    )
    graph.add_node(
        "$steps.checker",
        **{
            NODE_COMPILATION_OUTPUT_PROPERTY: _step_node(
                "checker",
                checker_manifest,
                {
                    "value": DynamicStepInputDefinition(
                        parameter_specification=ParameterSpecification(
                            parameter_name="value"
                        ),
                        category=NodeInputCategory.NON_BATCH_STEP_OUTPUT,
                        data_lineage=[],
                        selector="$steps.setter.out",
                    )
                },
            ),
        },
    )
    graph.add_node(
        "$outputs.result",
        **{
            NODE_COMPILATION_OUTPUT_PROPERTY: OutputNode(
                node_category=NodeCategory.OUTPUT_NODE,
                name="result",
                selector="$outputs.result",
                data_lineage=[],
                output_manifest=workflow_output,
                kind=[WILDCARD_KIND],
            )
        },
    )
    graph.add_edge("$steps.setter", "$steps.checker")
    graph.add_edge("$steps.checker", "$outputs.result")
    return CompiledWorkflow(
        workflow_definition=ParsedWorkflowDefinition(
            version="1.0",
            inputs=[],
            steps=[setter_manifest, checker_manifest],
            outputs=[workflow_output],
        ),
        execution_graph=graph,
        steps={
            "setter": InitialisedStep(
                block_specification=BlockSpecification(
                    block_source="test",
                    identifier="test/wave_setter@v1",
                    block_class=WaveSetterBlock,
                    manifest_class=WaveSetterManifest,
                ),
                manifest=setter_manifest,
                step=setter,
            ),
            "checker": InitialisedStep(
                block_specification=BlockSpecification(
                    block_source="test",
                    identifier="test/wave_checker@v1",
                    block_class=WaveCheckerBlock,
                    manifest_class=WaveCheckerManifest,
                ),
                manifest=checker_manifest,
                step=checker,
            ),
        },
        input_substitutions=[],
        workflow_json={},
        init_parameters={},
        kinds_serializers={},
        kinds_deserializers={},
    )


def test_task_local_state_does_not_leak_from_wave1_to_wave2() -> None:
    """Wave 1 poisons a task-local ContextVar; wave 2 - which may execute
    on the SAME reused OS thread - must observe the request-level value,
    because every task runs inside its own contextvars snapshot."""
    setter, checker = WaveSetterBlock(), WaveCheckerBlock()
    workflow = _two_wave_workflow(setter, checker)

    result = _run(workflow, executor=None)

    assert result == [{"result": "v"}]
    # wave 1 saw the clean request state before poisoning its own snapshot
    assert setter.observed_probe == ["request-value"]
    # wave 2 must NOT see wave 1's poison even if the thread was reused
    assert checker.observed_probe == ["request-value"], checker.observed_probe


def test_framework_contextvars_rebound_per_step_and_request_thread_untouched() -> None:
    """Framework-owned contextvars are unconditionally rebound inside each
    task (safe_execute_step) and never escape into the request thread,
    on the success path. Sentinels on the REQUEST thread model an ambient
    host context that must survive engine runs."""
    tokens = [
        current_debug_collector.set("req-collector-sentinel"),
        current_debug_trace.set("req-trace-sentinel"),
        current_debug_step_name.set("req-step-sentinel"),
        _WAVE_PROBE.set("req-probe-sentinel"),
    ]
    try:
        setter, checker = WaveSetterBlock(), WaveCheckerBlock()
        workflow = _two_wave_workflow(setter, checker)
        result = _run(workflow, executor=None)
        assert result == [{"result": "v"}]

        # request thread: all sentinels intact after the run
        assert current_debug_collector.get() == "req-collector-sentinel"
        assert current_debug_trace.get() == "req-trace-sentinel"
        assert current_debug_step_name.get() == "req-step-sentinel"
        assert _WAVE_PROBE.get() == "req-probe-sentinel"
        # wave 2 observed the REBOUND framework values, not the sentinels
        assert checker.observed_step_name == ["checker"], checker.observed_step_name
        # safe_execute_step re-establishes the REQUEST's host context inside
        # every step: the step sees the request collector, never a worker
        # leftover from an earlier wave or run
        assert checker.observed_collector == [
            "req-collector-sentinel"
        ], checker.observed_collector
        # ...and wave 2's ContextVar view was the REQUEST copy (sentinel),
        # proving snapshots carry request context, not worker leftovers
        assert checker.observed_probe == ["req-probe-sentinel"]
        assert setter.observed_probe == ["req-probe-sentinel"]
    finally:
        current_debug_step_name.reset(tokens[2])
        current_debug_trace.reset(tokens[1])
        current_debug_collector.reset(tokens[0])
        _WAVE_PROBE.reset(tokens[3])


def test_framework_context_survives_block_exception_and_next_run_recovers(
    created_pools: List[ThreadPoolExecutor],
) -> None:
    """Failure path: a step raising mid-run must not leak framework state
    into the request thread, and the NEXT run on a fresh engine must
    produce correct output (recovery after abort)."""
    tok_a = current_debug_collector.set("req-collector-sentinel")
    tok_b = current_debug_step_name.set("req-step-sentinel")
    try:
        exploding = _three_wave_workflow(second_relay_class=ExplodingBlock)
        with pytest.raises(StepExecutionError):
            _run(exploding, executor=None)
        assert current_debug_collector.get() == "req-collector-sentinel"
        assert current_debug_step_name.get() == "req-step-sentinel"

        # consecutive run on a NEW engine recovers cleanly
        healthy = _three_wave_workflow()
        result = _run(healthy, executor=None)
        assert result == [{"result": "seed"}]
        assert current_debug_collector.get() == "req-collector-sentinel"
        assert current_debug_step_name.get() == "req-step-sentinel"
        # each run owns its pool: two runs -> two pools, all shut down
        assert len(created_pools) == 2
        assert all(p._shutdown for p in created_pools)
    finally:
        current_debug_collector.reset(tok_a)
        current_debug_step_name.reset(tok_b)


def test_raw_thread_local_state_cannot_cross_runs_on_executor_none_path() -> None:
    """The hard boundary this implementation guarantees: raw
    threading.local set inside a step cannot outlive the run, because the
    owned pool's threads are destroyed at run end. Within a single run,
    later steps MAY observe an earlier step's thread-local on a reused
    worker - that is the documented boundary (block state belongs to
    instances / InstanceCache, not threads), so it is deliberately NOT
    asserted here; the framework-owned isolation is pinned by the tests
    above instead."""

    class TlsWritingManifest(WorkflowBlockManifest):
        type: Literal["test/tls_writer@v1"]
        name: str

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [OutputDefinition(name="out")]

    class TlsWritingBlock(WorkflowBlock):
        _tls = threading.local()  # class-level: shared storage per THREAD

        def __init__(self) -> None:
            self.entry_markers: List[Any] = []

        @classmethod
        def get_manifest(cls) -> Type[WorkflowBlockManifest]:
            return TlsWritingManifest

        def run(self) -> BlockResult:
            self.entry_markers.append(getattr(self._tls, "marker", None))
            self._tls.marker = "written-by-a-step"
            return {"out": "v"}

    def one_step_workflow(block) -> CompiledWorkflow:
        manifest = TlsWritingManifest(type="test/tls_writer@v1", name="writer")
        workflow_output = JsonField(
            type="JsonField", name="result", selector="$steps.writer.out"
        )
        graph = nx.DiGraph()
        graph.add_node(
            "$steps.writer",
            **{
                NODE_COMPILATION_OUTPUT_PROPERTY: _step_node("writer", manifest),
            },
        )
        graph.add_node(
            "$outputs.result",
            **{
                NODE_COMPILATION_OUTPUT_PROPERTY: OutputNode(
                    node_category=NodeCategory.OUTPUT_NODE,
                    name="result",
                    selector="$outputs.result",
                    data_lineage=[],
                    output_manifest=workflow_output,
                    kind=[WILDCARD_KIND],
                )
            },
        )
        graph.add_edge("$steps.writer", "$outputs.result")
        return CompiledWorkflow(
            workflow_definition=ParsedWorkflowDefinition(
                version="1.0",
                inputs=[],
                steps=[manifest],
                outputs=[workflow_output],
            ),
            execution_graph=graph,
            steps={
                "writer": InitialisedStep(
                    block_specification=BlockSpecification(
                        block_source="test",
                        identifier="test/tls_writer@v1",
                        block_class=TlsWritingBlock,
                        manifest_class=TlsWritingManifest,
                    ),
                    manifest=manifest,
                    step=block,
                )
            },
            input_substitutions=[],
            workflow_json={},
            init_parameters={},
            kinds_serializers={},
            kinds_deserializers={},
        )

    run1_block = TlsWritingBlock()
    result1 = _run(one_step_workflow(run1_block), executor=None)
    assert result1 == [{"result": "v"}]
    assert run1_block.entry_markers == [None]  # fresh threads in run 1

    # run 2: brand-new engine, brand-new owned pool, brand-new threads -
    # the class-level thread-local storage must be EMPTY on entry even
    # though the previous run wrote it on its workers
    run2_block = TlsWritingBlock()
    result2 = _run(one_step_workflow(run2_block), executor=None)
    assert result2 == [{"result": "v"}]
    assert run2_block.entry_markers == [None], (
        "thread-local state from run 1 leaked into run 2 workers: "
        f"{run2_block.entry_markers}"
    )


# ------------------------------------------------------------------
# Early-termination paths (the engine's actual "cancellation" surface).
# See the boundary note above: no cancellation primitive exists, so these
# two tests pin exception abort and conditional branch termination.
# ------------------------------------------------------------------


class CountingRelayBlock(RelayBlock):
    """Relay that records how many times its block body executed."""

    def __init__(self) -> None:
        self.run_count = 0

    def run(self, predictions: Any) -> BlockResult:
        self.run_count += 1
        return {"consumed": predictions}


def _replace_step(workflow: CompiledWorkflow, name: str, block: WorkflowBlock) -> None:
    manifest = workflow.steps[name].manifest
    workflow.steps[name] = InitialisedStep(
        block_specification=BlockSpecification(
            block_source="test",
            identifier=manifest.type,
            block_class=type(block),
            manifest_class=type(manifest),
        ),
        manifest=manifest,
        step=block,
    )


def test_exception_aborts_later_waves_and_next_run_completes(
    created_pools: List[ThreadPoolExecutor],
) -> None:
    """Early termination by EXCEPTION: wave 2 (relay_1) raises, so wave 3
    (relay_2) must never execute its block body. The run-scoped pool must
    still be built once, drained, and joined before the error escapes, and
    framework context on the request thread must be untouched. The next
    run must execute every wave again on a fresh pool - no sticky state
    from the aborted run."""
    tok_a = current_debug_collector.set("req-collector-sentinel")
    tok_b = current_debug_step_name.set("req-step-sentinel")
    try:
        workflow = _three_wave_workflow(second_relay_class=CountingRelayBlock)
        aborted_wave3 = workflow.steps["relay_2"].step
        _replace_step(workflow, "relay_1", ExplodingBlock())

        with pytest.raises(StepExecutionError):
            _run(workflow, executor=None)

        # abort semantics: the wave after the failure never ran its block
        assert aborted_wave3.run_count == 0, (
            "wave 3 executed after wave 2 aborted: "
            f"run_count={aborted_wave3.run_count}"
        )
        # framework-owned context never escaped to the request thread
        assert current_debug_collector.get() == "req-collector-sentinel"
        assert current_debug_step_name.get() == "req-step-sentinel"
        # one pool for the run, fully drained before the raise surfaced
        assert len(created_pools) == 1, f"pools: {len(created_pools)}"
        assert created_pools[0]._shutdown is True
        assert not [t for t in created_pools[0]._threads if t.is_alive()]

        healthy = _three_wave_workflow(second_relay_class=CountingRelayBlock)
        healthy_wave3 = healthy.steps["relay_2"].step
        result = _run(healthy, executor=None)
        assert result == [{"result": "seed"}]
        assert healthy_wave3.run_count == 1, (
            "post-abort run did not execute wave 3: "
            f"run_count={healthy_wave3.run_count}"
        )
        assert len(created_pools) == 2
        assert all(pool._shutdown for pool in created_pools)
        assert current_debug_collector.get() == "req-collector-sentinel"
        assert current_debug_step_name.get() == "req-step-sentinel"
    finally:
        current_debug_collector.reset(tok_a)
        current_debug_step_name.reset(tok_b)


def test_branch_termination_discards_downstream_block_and_shuts_pool(
    created_pools: List[ThreadPoolExecutor],
) -> None:
    """Early termination by CONDITIONAL BRANCH TERMINATION: relay_1 acts
    as a flow-control step and returns FlowControl() with no context,
    which masks its downstream execution branch. The wave containing
    relay_2 is still scheduled (the coordinator is static), but the
    block body must be discarded without running - the supported "skip"
    semantics. The run completes normally, the run-scoped pool shuts
    down exactly once with its threads joined, request-thread framework
    context is untouched, and a following healthy run executes the
    skipped block again."""
    tok = current_debug_collector.set("req-collector-sentinel")
    try:
        workflow = _three_wave_workflow(second_relay_class=CountingRelayBlock)
        skipped_wave3 = workflow.steps["relay_2"].step

        # Wire the same branch metadata the compiler would produce for a
        # flow-control connection relay_1 -> relay_2.
        branch = "wf_ownership_test_branch"
        relay_1_node = workflow.execution_graph.nodes["$steps.relay_1"][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        relay_1_node.child_execution_branches = {"$steps.relay_2": branch}
        relay_2_node = workflow.execution_graph.nodes["$steps.relay_2"][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        relay_2_node.execution_branches_impacting_inputs = {branch}

        class TerminatingRelayBlock(RelayBlock):
            def run(self, predictions: Any) -> BlockResult:
                return FlowControl()  # context=None -> no target selected

        _replace_step(workflow, "relay_1", TerminatingRelayBlock())

        result = _run(workflow, executor=None)

        # downstream block body discarded, output stays unset -> None
        assert result == [{"result": None}]
        assert skipped_wave3.run_count == 0, (
            "terminated branch still executed its downstream block: "
            f"run_count={skipped_wave3.run_count}"
        )
        assert len(created_pools) == 1, f"pools: {len(created_pools)}"
        assert created_pools[0]._shutdown is True
        assert not [t for t in created_pools[0]._threads if t.is_alive()]
        assert current_debug_collector.get() == "req-collector-sentinel"

        healthy = _three_wave_workflow(second_relay_class=CountingRelayBlock)
        healthy_wave3 = healthy.steps["relay_2"].step
        healthy_result = _run(healthy, executor=None)
        assert healthy_result == [{"result": "seed"}]
        assert healthy_wave3.run_count == 1, (
            "run after branch termination did not execute wave 3: "
            f"run_count={healthy_wave3.run_count}"
        )
        assert len(created_pools) == 2
        assert all(pool._shutdown for pool in created_pools)
    finally:
        current_debug_collector.reset(tok)
