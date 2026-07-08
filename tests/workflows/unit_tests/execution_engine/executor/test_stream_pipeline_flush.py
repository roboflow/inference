from typing import Any, Dict, List, Literal, Optional, Tuple, Type, get_args

import networkx as nx

from inference.core.workflows.core_steps.models.roboflow.object_detection.v3 import (
    BlockManifest as ObjectDetectionModelV3Manifest,
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
    compute_stream_lookahead_frontier,
    flush_stream_pipeline_workflow,
    resume_stream_lookahead_workflow,
    run_stream_lookahead_workflow,
    run_workflow,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class DeferredProducerManifest(WorkflowBlockManifest):
    type: Literal["test/deferred_producer@v1"]
    name: str

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="predictions")]


class DeferredProducerBlock(WorkflowBlock):
    def __init__(self) -> None:
        self.flush_calls = 0
        self.closed = False

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return DeferredProducerManifest

    def run(self) -> BlockResult:
        return {"predictions": "warmup-placeholder"}

    def flush_stream_pipeline_outputs(
        self,
    ) -> List[Tuple[List[Tuple[int, ...]], BlockResult]]:
        self.flush_calls += 1
        if self.flush_calls > 1:
            return []
        return [([], [{"predictions": "flushed-tail"}])]

    def close_stream_pipeline(self) -> None:
        self.closed = True


class ConsumerManifest(WorkflowBlockManifest):
    type: Literal["test/consumer@v1"]
    name: str

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="consumed")]


class ConsumerBlock(WorkflowBlock):
    def __init__(self) -> None:
        self.seen_predictions: List[Any] = []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ConsumerManifest

    def run(self, predictions: Any) -> BlockResult:
        self.seen_predictions.append(predictions)
        return {"consumed": predictions}


class StatelessModelManifest(WorkflowBlockManifest):
    type: Literal["test/stateless_model@v1"]
    name: str

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="predictions")]

    @classmethod
    def is_stateful_for_video_processing(cls) -> bool:
        return False


class AsyncModelBlock(WorkflowBlock):
    # Stand-in for a stream-pipelined remote model step: stateless manifest,
    # async-launching at runtime.
    def __init__(self) -> None:
        self.run_calls = 0

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return StatelessModelManifest

    def is_stream_pipelined(self) -> bool:
        return True

    def run(self, **kwargs) -> BlockResult:
        self.run_calls += 1
        return {"predictions": f"model-output-{self.run_calls}"}


class StatelessTransformBlock(WorkflowBlock):
    # Stand-in for a stateless, non-async step (e.g. a static crop).
    def __init__(self) -> None:
        self.run_calls = 0

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return StatelessModelManifest

    def run(self, **kwargs) -> BlockResult:
        self.run_calls += 1
        return {"predictions": "transformed"}


def test_stream_pipeline_flush_runs_tail_output_through_downstream_steps() -> None:
    producer = DeferredProducerBlock()
    consumer = ConsumerBlock()
    workflow = _compiled_workflow(
        producer=producer,
        consumer=consumer,
        producer_manifest=DeferredProducerManifest(
            type="test/deferred_producer@v1",
            name="segmentation",
        ),
        consumer_manifest=ConsumerManifest(
            type="test/consumer@v1",
            name="consumer",
        ),
    )

    result = flush_stream_pipeline_workflow(
        workflow=workflow,
        runtime_parameters={},
        max_concurrent_steps=1,
        kinds_serializers={},
        serialize_results=False,
        profiler=NullWorkflowsProfiler.init(),
    )

    assert consumer.seen_predictions == ["flushed-tail"]
    assert result == [{"result": "flushed-tail"}]
    assert producer.flush_calls == 1
    assert producer.closed is False


def test_deferred_pass_runs_all_steps() -> None:
    producer = DeferredProducerBlock()
    consumer = ConsumerBlock()
    workflow = _compiled_workflow(
        producer=producer,
        consumer=consumer,
        producer_manifest=DeferredProducerManifest(
            type="test/deferred_producer@v1",
            name="segmentation",
        ),
        consumer_manifest=ConsumerManifest(
            type="test/consumer@v1",
            name="consumer",
        ),
    )

    result = run_workflow(
        workflow=workflow,
        runtime_parameters={},
        max_concurrent_steps=1,
        kinds_serializers={},
        serialize_results=False,
        profiler=NullWorkflowsProfiler.init(),
        defer_stream_pipeline_flush=True,
        resolve_output_futures=False,
    )

    # The RF-DETR-style deferred pass runs every step; deferring the flush
    # only skips the pipeline flush and close.
    assert consumer.seen_predictions == ["warmup-placeholder"]
    assert result == [{"result": "warmup-placeholder"}]
    assert producer.flush_calls == 0
    assert producer.closed is False


def test_is_stateful_for_video_processing_defaults_to_true_and_is_overridden() -> None:
    class BareManifest(WorkflowBlockManifest):
        type: Literal["test/bare@v1"]
        name: str

    assert BareManifest.is_stateful_for_video_processing() is True
    assert VisualizationManifest.is_stateful_for_video_processing() is False
    assert ObjectDetectionModelV3Manifest.is_stateful_for_video_processing() is False
    # Trace and heatmap render from accumulated cross-frame state, so they
    # re-declare themselves stateful despite being visualizations.
    assert TraceManifest.is_stateful_for_video_processing() is True
    assert HeatmapManifest.is_stateful_for_video_processing() is True


def test_frontier_of_linear_model_to_tracker_workflow() -> None:
    workflow = _lookahead_compiled_workflow(
        steps={
            "model": _step_spec(AsyncModelBlock(), StatelessModelManifest),
            "tracker": _step_spec(
                ConsumerBlock(),
                ConsumerManifest,
                inputs={"predictions": "$steps.model.predictions"},
            ),
        },
        edges=[("$steps.model", "$steps.tracker")],
    )

    frontier = compute_stream_lookahead_frontier(workflow=workflow)

    assert frontier == {"$steps.model"}


def test_frontier_of_two_models_with_consensus_workflow() -> None:
    # consensus is stateless but fed by async steps, so it must resolve their
    # futures at resume time - async steps are frontier sinks
    workflow = _lookahead_compiled_workflow(
        steps={
            "model_a": _step_spec(AsyncModelBlock(), StatelessModelManifest),
            "model_b": _step_spec(AsyncModelBlock(), StatelessModelManifest),
            "consensus": _step_spec(StatelessTransformBlock(), StatelessModelManifest),
        },
        edges=[
            ("$steps.model_a", "$steps.consensus"),
            ("$steps.model_b", "$steps.consensus"),
        ],
    )

    frontier = compute_stream_lookahead_frontier(workflow=workflow)

    assert frontier == {"$steps.model_a", "$steps.model_b"}


def test_frontier_excludes_model_chained_after_another_model() -> None:
    workflow = _lookahead_compiled_workflow(
        steps={
            "model_a": _step_spec(AsyncModelBlock(), StatelessModelManifest),
            "model_b": _step_spec(AsyncModelBlock(), StatelessModelManifest),
        },
        edges=[("$steps.model_a", "$steps.model_b")],
    )

    frontier = compute_stream_lookahead_frontier(workflow=workflow)

    assert frontier == {"$steps.model_a"}


def test_frontier_excludes_model_fed_by_stateful_step() -> None:
    workflow = _lookahead_compiled_workflow(
        steps={
            "stateful_pre": _step_spec(ConsumerBlock(), ConsumerManifest),
            "model": _step_spec(AsyncModelBlock(), StatelessModelManifest),
        },
        edges=[("$steps.stateful_pre", "$steps.model")],
    )

    frontier = compute_stream_lookahead_frontier(workflow=workflow)

    assert frontier == set()


def test_frontier_includes_stateless_side_branch_and_upstream_crop() -> None:
    workflow = _lookahead_compiled_workflow(
        steps={
            "crop": _step_spec(StatelessTransformBlock(), StatelessModelManifest),
            "model": _step_spec(AsyncModelBlock(), StatelessModelManifest),
            "side": _step_spec(StatelessTransformBlock(), StatelessModelManifest),
            "tracker": _step_spec(ConsumerBlock(), ConsumerManifest),
        },
        edges=[
            ("$steps.crop", "$steps.model"),
            ("$steps.crop", "$steps.side"),
            ("$steps.model", "$steps.tracker"),
        ],
    )

    frontier = compute_stream_lookahead_frontier(workflow=workflow)

    assert frontier == {"$steps.crop", "$steps.model", "$steps.side"}


def test_stream_lookahead_run_and_resume_split_execution_between_passes() -> None:
    model = AsyncModelBlock()
    consumer = ConsumerBlock()
    workflow = _lookahead_compiled_workflow(
        steps={
            "model": _step_spec(model, StatelessModelManifest),
            "consumer": _step_spec(
                consumer,
                ConsumerManifest,
                inputs={"predictions": "$steps.model.predictions"},
            ),
        },
        edges=[("$steps.model", "$steps.consumer")],
        output_selector="$steps.consumer.consumed",
    )
    frontier = compute_stream_lookahead_frontier(workflow=workflow)
    assert frontier == {"$steps.model"}

    execution_data_manager = run_stream_lookahead_workflow(
        workflow=workflow,
        runtime_parameters={},
        max_concurrent_steps=1,
        frontier_step_selectors=frontier,
        profiler=NullWorkflowsProfiler.init(),
    )

    # The lookahead pass runs only the frontier steps.
    assert model.run_calls == 1
    assert consumer.seen_predictions == []

    result = resume_stream_lookahead_workflow(
        workflow=workflow,
        execution_data_manager=execution_data_manager,
        max_concurrent_steps=1,
        kinds_serializers={},
        frontier_step_selectors=frontier,
        serialize_results=False,
        profiler=NullWorkflowsProfiler.init(),
    )

    # The resume pass runs only the remainder, on the same execution state,
    # without re-executing any frontier step.
    assert model.run_calls == 1
    assert consumer.seen_predictions == ["model-output-1"]
    assert result == [{"result": "model-output-1"}]


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


def _compiled_workflow(
    producer: DeferredProducerBlock,
    consumer: ConsumerBlock,
    producer_manifest: DeferredProducerManifest,
    consumer_manifest: ConsumerManifest,
) -> CompiledWorkflow:
    workflow_output = JsonField(
        type="JsonField",
        name="result",
        selector="$steps.consumer.consumed",
    )
    graph = nx.DiGraph()
    graph.add_node(
        "$steps.segmentation",
        **{
            NODE_COMPILATION_OUTPUT_PROPERTY: StepNode(
                node_category=NodeCategory.STEP_NODE,
                name="segmentation",
                selector="$steps.segmentation",
                data_lineage=[],
                step_manifest=producer_manifest,
            )
        },
    )
    graph.add_node(
        "$steps.consumer",
        **{
            NODE_COMPILATION_OUTPUT_PROPERTY: StepNode(
                node_category=NodeCategory.STEP_NODE,
                name="consumer",
                selector="$steps.consumer",
                data_lineage=[],
                step_manifest=consumer_manifest,
                input_data={
                    "predictions": DynamicStepInputDefinition(
                        parameter_specification=ParameterSpecification(
                            parameter_name="predictions"
                        ),
                        category=NodeInputCategory.NON_BATCH_STEP_OUTPUT,
                        data_lineage=[],
                        selector="$steps.segmentation.predictions",
                    )
                },
            )
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
    graph.add_edge("$steps.segmentation", "$steps.consumer")
    graph.add_edge("$steps.consumer", "$outputs.result")

    return CompiledWorkflow(
        workflow_definition=ParsedWorkflowDefinition(
            version="1.0",
            inputs=[],
            steps=[producer_manifest, consumer_manifest],
            outputs=[workflow_output],
        ),
        execution_graph=graph,
        steps={
            "segmentation": InitialisedStep(
                block_specification=BlockSpecification(
                    block_source="test",
                    identifier="test/deferred_producer@v1",
                    block_class=DeferredProducerBlock,
                    manifest_class=DeferredProducerManifest,
                ),
                manifest=producer_manifest,
                step=producer,
            ),
            "consumer": InitialisedStep(
                block_specification=BlockSpecification(
                    block_source="test",
                    identifier="test/consumer@v1",
                    block_class=ConsumerBlock,
                    manifest_class=ConsumerManifest,
                ),
                manifest=consumer_manifest,
                step=consumer,
            ),
        },
        input_substitutions=[],
        workflow_json={},
        init_parameters={},
        kinds_serializers={},
        kinds_deserializers={},
    )
