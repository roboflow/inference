from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, get_args
from unittest.mock import MagicMock

import networkx as nx
import pytest
from pydantic import BaseModel

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything3 import (
    v3 as segment_anything3_v3,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v3 import (
    SegmentAnything3BlockV3,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v3 import (
    BlockManifest as ObjectDetectionModelV3Manifest,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v3 import (
    RoboflowObjectDetectionModelBlockV3,
)
from inference.core.workflows.core_steps.visualizations.common.base import (
    VisualizationManifest,
)
from inference.core.workflows.core_steps.visualizations.heatmap.v1 import (
    HeatmapManifest,
)
from inference.core.workflows.core_steps.visualizations.trace.v1 import TraceManifest
from inference.core.workflows.execution_engine.constants import (
    NODE_COMPILATION_OUTPUT_PROPERTY,
)
from inference.core.workflows.execution_engine.entities.base import (
    JsonField,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import WILDCARD_KIND
from inference.core.workflows.execution_engine.profiling.core import (
    NullWorkflowsProfiler,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
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
from inference.core.workflows.execution_engine.v1.executor.core import (
    compute_async_stream_step_selectors,
    compute_stream_lookahead_frontier,
    launch_async_simd_step,
    resume_stream_lookahead_workflow,
    run_stream_lookahead_workflow,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class StatelessModelManifest(WorkflowBlockManifest):
    type: Literal["test/async_model@v1"]
    name: str

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="predictions"),
            OutputDefinition(name="inference_id"),
        ]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def is_stateful_for_video_processing(cls) -> bool:
        return False


class ScalarModelManifest(StatelessModelManifest):
    type: Literal["test/scalar_model@v1"]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return []


class WildcardOutputModelManifest(StatelessModelManifest):
    type: Literal["test/wildcard_model@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="*")]


class DimensionalityOffsetModelManifest(StatelessModelManifest):
    type: Literal["test/dim_offset_model@v1"]

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1


class StatefulConsumerManifest(WorkflowBlockManifest):
    type: Literal["test/consumer@v1"]
    name: str

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="consumed")]


class BareManifest(WorkflowBlockManifest):
    type: Literal["test/bare@v1"]
    name: str


class DynamicBlockManifest(BaseModel):
    # Dynamic Custom Python block manifests are plain pydantic models built
    # via create_model, not WorkflowBlockManifest subclasses - they have no
    # is_stateful_for_video_processing method at all.
    type: Literal["test/dynamic_block@v1"]
    name: str


class AsyncModelBlock(WorkflowBlock):
    def __init__(self, run_fn: Optional[Callable[..., BlockResult]] = None) -> None:
        self.run_calls: List[dict] = []
        self._run_fn = run_fn

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return StatelessModelManifest

    def is_async_stream_step(self) -> bool:
        return True

    def run(self, **kwargs) -> BlockResult:
        self.run_calls.append(kwargs)
        if self._run_fn is not None:
            return self._run_fn(**kwargs)
        return [
            {"predictions": f"prediction-{image}", "inference_id": f"id-{image}"}
            for image in kwargs["images"]
        ]


class PlainStepBlock(WorkflowBlock):
    # Stateless, non-async step (e.g. a static crop, or an undeclared model).
    def __init__(self) -> None:
        self.run_calls = 0

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return StatelessModelManifest

    def run(self, **kwargs) -> BlockResult:
        self.run_calls += 1
        return {"predictions": f"plain-output-{self.run_calls}", "inference_id": None}


class ConsumerBlock(WorkflowBlock):
    def __init__(self) -> None:
        self.seen_predictions: List[Any] = []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return StatefulConsumerManifest

    def run(self, predictions: Any) -> BlockResult:
        self.seen_predictions.append(predictions)
        return {"consumed": predictions}


@pytest.mark.parametrize(
    "manifest_class, block_factory, expected_selectors",
    [
        (StatelessModelManifest, AsyncModelBlock, {"$steps.model"}),
        (ScalarModelManifest, AsyncModelBlock, set()),
        (WildcardOutputModelManifest, AsyncModelBlock, set()),
        (DimensionalityOffsetModelManifest, AsyncModelBlock, set()),
        (StatelessModelManifest, PlainStepBlock, set()),
    ],
    ids=["declared_and_eligible", "scalar", "wildcard_output", "dim_offset", "undeclared"],
)
def test_compute_async_stream_step_selectors_eligibility(
    manifest_class: Type[WorkflowBlockManifest],
    block_factory: Type[WorkflowBlock],
    expected_selectors: set,
) -> None:
    # given
    workflow = _lookahead_compiled_workflow(
        steps={"model": _step_spec(block_factory(), manifest_class)},
        edges=[],
    )

    # when / then
    assert compute_async_stream_step_selectors(workflow=workflow) == expected_selectors


_FRONTIER_CASES = {
    "linear_model_into_tracker": lambda: (
        {
            "model": _step_spec(AsyncModelBlock(), StatelessModelManifest),
            "tracker": _step_spec(ConsumerBlock(), StatefulConsumerManifest),
        },
        [("$steps.model", "$steps.tracker")],
        {"$steps.model"},
    ),
    "model_fed_by_stateful_step": lambda: (
        {
            "stateful_pre": _step_spec(ConsumerBlock(), StatefulConsumerManifest),
            "model": _step_spec(AsyncModelBlock(), StatelessModelManifest),
        },
        [("$steps.stateful_pre", "$steps.model")],
        set(),
    ),
    "model_chained_after_model": lambda: (
        {
            "model_a": _step_spec(AsyncModelBlock(), StatelessModelManifest),
            "model_b": _step_spec(AsyncModelBlock(), StatelessModelManifest),
        },
        [("$steps.model_a", "$steps.model_b")],
        {"$steps.model_a"},
    ),
    "stateless_side_branch": lambda: (
        {
            "crop": _step_spec(PlainStepBlock(), StatelessModelManifest),
            "model": _step_spec(AsyncModelBlock(), StatelessModelManifest),
            "side": _step_spec(PlainStepBlock(), StatelessModelManifest),
            "tracker": _step_spec(ConsumerBlock(), StatefulConsumerManifest),
        },
        [
            ("$steps.crop", "$steps.model"),
            ("$steps.crop", "$steps.side"),
            ("$steps.model", "$steps.tracker"),
        ],
        {"$steps.crop", "$steps.model", "$steps.side"},
    ),
    # Dynamic Custom Python manifests lack the statefulness declaration
    # entirely and must be treated as stateful, not crash.
    "dynamic_manifest_without_statefulness_method": lambda: (
        {"custom": _step_spec(PlainStepBlock(), DynamicBlockManifest)},
        [],
        set(),
    ),
}


@pytest.mark.parametrize("case_name", list(_FRONTIER_CASES))
def test_compute_stream_lookahead_frontier_shapes(case_name: str) -> None:
    # given
    steps, edges, expected_frontier = _FRONTIER_CASES[case_name]()
    workflow = _lookahead_compiled_workflow(steps=steps, edges=edges)

    # when / then
    assert compute_stream_lookahead_frontier(workflow=workflow) == expected_frontier


def test_launch_async_simd_step_registers_per_output_chained_futures() -> None:
    # given
    block = AsyncModelBlock()
    workflow = _lookahead_compiled_workflow(
        steps={"model": _step_spec(block, StatelessModelManifest)},
        edges=[],
    )
    execution_data_manager = MagicMock()
    execution_data_manager.get_simd_step_input.return_value = SimpleNamespace(
        indices=[(0,), (1,)],
        parameters={"images": ["a", "b"]},
    )

    # when
    with ThreadPoolExecutor(max_workers=1) as lookahead_executor:
        launch_async_simd_step(
            step_selector="$steps.model",
            workflow=workflow,
            execution_data_manager=execution_data_manager,
            lookahead_executor=lookahead_executor,
        )

        # then - one future per declared output per batch element, resolving
        # to that element's slice of the offloaded run() result
        registered = execution_data_manager.register_simd_step_output.call_args
        assert registered.kwargs["indices"] == [(0,), (1,)]
        outputs = registered.kwargs["outputs"]
        assert len(outputs) == 2
        assert outputs[0]["predictions"].result(timeout=5) == "prediction-a"
        assert outputs[0]["inference_id"].result(timeout=5) == "id-a"
        assert outputs[1]["predictions"].result(timeout=5) == "prediction-b"
        assert outputs[1]["inference_id"].result(timeout=5) == "id-b"
        assert block.run_calls == [{"images": ["a", "b"]}]


def test_launch_async_simd_step_with_empty_indices_registers_empty_output() -> None:
    # given
    block = AsyncModelBlock()
    workflow = _lookahead_compiled_workflow(
        steps={"model": _step_spec(block, StatelessModelManifest)},
        edges=[],
    )
    execution_data_manager = MagicMock()
    execution_data_manager.get_simd_step_input.return_value = SimpleNamespace(
        indices=[],
        parameters={"images": []},
    )
    lookahead_executor = MagicMock()

    # when
    launch_async_simd_step(
        step_selector="$steps.model",
        workflow=workflow,
        execution_data_manager=execution_data_manager,
        lookahead_executor=lookahead_executor,
    )

    # then
    execution_data_manager.register_simd_step_output.assert_called_once_with(
        step_selector="$steps.model",
        indices=[],
        outputs=[],
    )
    lookahead_executor.submit.assert_not_called()


@pytest.mark.parametrize(
    "run_fn, expected_error",
    [
        (lambda **kwargs: (_ for _ in ()).throw(RuntimeError("remote failed")), RuntimeError),
        (lambda **kwargs: [{}], KeyError),
    ],
    ids=["run_raises", "malformed_payload"],
)
def test_launch_async_simd_step_surfaces_errors_through_output_futures(
    run_fn: Callable[..., BlockResult],
    expected_error: Type[Exception],
) -> None:
    # given - the offloaded run() either raises or returns a payload the
    # chained futures cannot select from; either way consumers must get the
    # error instead of hanging
    block = AsyncModelBlock(run_fn=run_fn)
    workflow = _lookahead_compiled_workflow(
        steps={"model": _step_spec(block, StatelessModelManifest)},
        edges=[],
    )
    execution_data_manager = MagicMock()
    execution_data_manager.get_simd_step_input.return_value = SimpleNamespace(
        indices=[(0,)],
        parameters={"images": ["a"]},
    )

    # when
    with ThreadPoolExecutor(max_workers=1) as lookahead_executor:
        launch_async_simd_step(
            step_selector="$steps.model",
            workflow=workflow,
            execution_data_manager=execution_data_manager,
            lookahead_executor=lookahead_executor,
        )

        # then
        outputs = execution_data_manager.register_simd_step_output.call_args.kwargs[
            "outputs"
        ]
        with pytest.raises(expected_error):
            outputs[0]["predictions"].result(timeout=5)


def test_stream_lookahead_run_and_resume_split_execution_between_passes() -> None:
    # given
    model = PlainStepBlock()
    consumer = ConsumerBlock()
    workflow = _lookahead_compiled_workflow(
        steps={
            "model": _step_spec(model, StatelessModelManifest),
            "consumer": _step_spec(
                consumer,
                StatefulConsumerManifest,
                inputs={"predictions": "$steps.model.predictions"},
            ),
        },
        edges=[("$steps.model", "$steps.consumer")],
        output_selector="$steps.consumer.consumed",
    )
    frontier = compute_stream_lookahead_frontier(workflow=workflow)
    assert frontier == {"$steps.model"}

    # when - the deferred pass runs only the frontier
    with ThreadPoolExecutor(max_workers=1) as lookahead_executor:
        execution_data_manager = run_stream_lookahead_workflow(
            workflow=workflow,
            runtime_parameters={},
            max_concurrent_steps=1,
            frontier_step_selectors=frontier,
            async_step_selectors=set(),
            lookahead_executor=lookahead_executor,
            profiler=NullWorkflowsProfiler.init(),
        )

    # then
    assert model.run_calls == 1
    assert consumer.seen_predictions == []

    # when - the resume pass runs only the remainder on the same state
    result = resume_stream_lookahead_workflow(
        workflow=workflow,
        execution_data_manager=execution_data_manager,
        max_concurrent_steps=1,
        kinds_serializers={},
        frontier_step_selectors=frontier,
        serialize_results=False,
        profiler=NullWorkflowsProfiler.init(),
    )

    # then - no step executed twice, outputs correct
    assert model.run_calls == 1
    assert consumer.seen_predictions == ["plain-output-1"]
    assert result == [{"result": "plain-output-1"}]


@pytest.mark.parametrize(
    "manifest_class, expected_stateful",
    [
        (BareManifest, True),
        (VisualizationManifest, False),
        (TraceManifest, True),
        (HeatmapManifest, True),
        (ObjectDetectionModelV3Manifest, False),
    ],
    ids=["default", "visualization_base", "trace", "heatmap", "object_detection_v3"],
)
def test_is_stateful_for_video_processing_metadata(
    manifest_class: Type[WorkflowBlockManifest],
    expected_stateful: bool,
) -> None:
    assert manifest_class.is_stateful_for_video_processing() is expected_stateful


@pytest.mark.parametrize(
    "block_class, sam3_exec_mode, execution_mode, expected",
    [
        (RoboflowObjectDetectionModelBlockV3, None, StepExecutionMode.REMOTE, True),
        (RoboflowObjectDetectionModelBlockV3, None, StepExecutionMode.LOCAL, False),
        (SegmentAnything3BlockV3, "local", StepExecutionMode.REMOTE, True),
        (SegmentAnything3BlockV3, "remote", StepExecutionMode.REMOTE, False),
        (SegmentAnything3BlockV3, "local", StepExecutionMode.LOCAL, False),
    ],
    ids=["od_remote", "od_local", "sam3_remote", "sam3_proxy_mode", "sam3_local"],
)
def test_model_blocks_declare_async_stream_step_only_for_reentrant_remote_execution(
    monkeypatch,
    block_class: Type[WorkflowBlock],
    sam3_exec_mode: Optional[str],
    execution_mode: StepExecutionMode,
    expected: bool,
) -> None:
    # given
    if sam3_exec_mode is not None:
        monkeypatch.setattr(segment_anything3_v3, "SAM3_EXEC_MODE", sam3_exec_mode)
    block = block_class(
        model_manager=MagicMock(),
        api_key="key",
        step_execution_mode=execution_mode,
    )

    # when / then
    assert block.is_async_stream_step() is expected


def _step_spec(
    block: WorkflowBlock,
    manifest_class: Type[WorkflowBlockManifest],
    inputs: Optional[Dict[str, str]] = None,
) -> dict:
    return {"block": block, "manifest_class": manifest_class, "inputs": inputs or {}}


def _lookahead_compiled_workflow(
    steps: Dict[str, dict],
    edges: List[Tuple[str, str]],
    output_selector: Optional[str] = None,
) -> CompiledWorkflow:
    manifests = {
        step_name: spec["manifest_class"](
            type=get_args(spec["manifest_class"].model_fields["type"].annotation)[0],
            name=step_name,
        )
        for step_name, spec in steps.items()
    }
    graph = nx.DiGraph()
    for step_name, spec in steps.items():
        graph.add_node(
            f"$steps.{step_name}",
            **{
                NODE_COMPILATION_OUTPUT_PROPERTY: StepNode(
                    node_category=NodeCategory.STEP_NODE,
                    name=step_name,
                    selector=f"$steps.{step_name}",
                    data_lineage=[],
                    step_manifest=manifests[step_name],
                    input_data={
                        parameter_name: DynamicStepInputDefinition(
                            parameter_specification=ParameterSpecification(
                                parameter_name=parameter_name
                            ),
                            category=NodeInputCategory.NON_BATCH_STEP_OUTPUT,
                            data_lineage=[],
                            selector=selector,
                        )
                        for parameter_name, selector in spec["inputs"].items()
                    },
                )
            },
        )
    for edge_start, edge_end in edges:
        graph.add_edge(edge_start, edge_end)
    workflow_outputs = []
    if output_selector is not None:
        workflow_output = JsonField(
            type="JsonField",
            name="result",
            selector=output_selector,
        )
        workflow_outputs.append(workflow_output)
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
        output_step_selector = ".".join(output_selector.split(".")[:2])
        graph.add_edge(output_step_selector, "$outputs.result")

    return CompiledWorkflow(
        workflow_definition=ParsedWorkflowDefinition(
            version="1.0",
            inputs=[],
            steps=list(manifests.values()),
            outputs=workflow_outputs,
        ),
        execution_graph=graph,
        steps={
            step_name: InitialisedStep(
                block_specification=BlockSpecification(
                    block_source="test",
                    identifier=manifests[step_name].type,
                    block_class=type(spec["block"]),
                    manifest_class=spec["manifest_class"],
                ),
                manifest=manifests[step_name],
                step=spec["block"],
            )
            for step_name, spec in steps.items()
        },
        input_substitutions=[],
        workflow_json={},
        init_parameters={},
        kinds_serializers={},
        kinds_deserializers={},
    )
