from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import networkx as nx

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
    flush_stream_pipeline_workflow,
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


class DeferringProducerBlock(DeferredProducerBlock):
    # A pipelined producer that defers downstream execution: the deferred pass
    # must skip its downstream consumers, which run later via the flush path.
    def is_stream_pipelined(self) -> bool:
        return True

    def defers_downstream_execution(self) -> bool:
        return True


class TaggedDeferringProducerBlock(DeferringProducerBlock):
    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag

    def flush_stream_pipeline_outputs(
        self,
    ) -> List[Tuple[List[Tuple[int, ...]], BlockResult]]:
        self.flush_calls += 1
        if self.flush_calls > 1:
            return []
        return [([], [{"predictions": f"flushed-{self.tag}"}])]


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


class FusionConsumerManifest(WorkflowBlockManifest):
    type: Literal["test/fusion_consumer@v1"]
    name: str

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="consumed")]


class FusionConsumerBlock(WorkflowBlock):
    def __init__(self) -> None:
        self.seen_predictions: List[Any] = []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return FusionConsumerManifest

    def run(self, predictions_a: Any, predictions_b: Any) -> BlockResult:
        self.seen_predictions.append((predictions_a, predictions_b))
        return {"consumed": f"{predictions_a}+{predictions_b}"}


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


def test_deferred_pass_skips_downstream_of_deferring_step_until_flush() -> None:
    producer = DeferringProducerBlock()
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

    deferred_result = run_workflow(
        workflow=workflow,
        runtime_parameters={},
        max_concurrent_steps=1,
        kinds_serializers={},
        serialize_results=False,
        profiler=NullWorkflowsProfiler.init(),
        defer_stream_pipeline_flush=True,
        resolve_output_futures=False,
    )

    # The deferred pass must not run the downstream consumer.
    assert consumer.seen_predictions == []
    assert producer.flush_calls == 0
    assert producer.closed is False

    flushed_result = flush_stream_pipeline_workflow(
        workflow=workflow,
        runtime_parameters={},
        max_concurrent_steps=1,
        kinds_serializers={},
        serialize_results=False,
        profiler=NullWorkflowsProfiler.init(),
    )

    # The flush pass runs the consumer with the producer's flushed tail output.
    assert consumer.seen_predictions == ["flushed-tail"]
    assert flushed_result == [{"result": "flushed-tail"}]
    assert producer.flush_calls == 1


def test_deferred_pass_runs_downstream_when_producer_does_not_defer() -> None:
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

    # Without the defer marker the consumer runs in the deferred pass, on the
    # producer's warmup output — the pre-existing behavior.
    assert consumer.seen_predictions == ["warmup-placeholder"]
    assert result == [{"result": "warmup-placeholder"}]
    assert producer.flush_calls == 0


def test_flush_runs_consumer_once_with_outputs_of_both_deferring_producers() -> None:
    producer_a = TaggedDeferringProducerBlock(tag="model-a")
    producer_b = TaggedDeferringProducerBlock(tag="model-b")
    consumer = FusionConsumerBlock()
    workflow = _two_producer_compiled_workflow(
        producer_a=producer_a,
        producer_b=producer_b,
        consumer=consumer,
    )

    run_workflow(
        workflow=workflow,
        runtime_parameters={},
        max_concurrent_steps=1,
        kinds_serializers={},
        serialize_results=False,
        profiler=NullWorkflowsProfiler.init(),
        defer_stream_pipeline_flush=True,
        resolve_output_futures=False,
    )

    # The deferred pass must not run the consumer of either producer.
    assert consumer.seen_predictions == []
    assert producer_a.flush_calls == 0
    assert producer_b.flush_calls == 0

    flushed_result = flush_stream_pipeline_workflow(
        workflow=workflow,
        runtime_parameters={},
        max_concurrent_steps=1,
        kinds_serializers={},
        serialize_results=False,
        profiler=NullWorkflowsProfiler.init(),
    )

    # The flush pass drains both producers and runs the consumer exactly once
    # with both flushed values.
    assert consumer.seen_predictions == [("flushed-model-a", "flushed-model-b")]
    assert flushed_result == [{"result": "flushed-model-a+flushed-model-b"}]
    assert producer_a.flush_calls == 1
    assert producer_b.flush_calls == 1


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


def _two_producer_compiled_workflow(
    producer_a: TaggedDeferringProducerBlock,
    producer_b: TaggedDeferringProducerBlock,
    consumer: FusionConsumerBlock,
) -> CompiledWorkflow:
    producer_a_manifest = DeferredProducerManifest(
        type="test/deferred_producer@v1",
        name="model_a",
    )
    producer_b_manifest = DeferredProducerManifest(
        type="test/deferred_producer@v1",
        name="model_b",
    )
    consumer_manifest = FusionConsumerManifest(
        type="test/fusion_consumer@v1",
        name="consumer",
    )
    workflow_output = JsonField(
        type="JsonField",
        name="result",
        selector="$steps.consumer.consumed",
    )
    graph = nx.DiGraph()
    for producer_manifest in (producer_a_manifest, producer_b_manifest):
        graph.add_node(
            f"$steps.{producer_manifest.name}",
            **{
                NODE_COMPILATION_OUTPUT_PROPERTY: StepNode(
                    node_category=NodeCategory.STEP_NODE,
                    name=producer_manifest.name,
                    selector=f"$steps.{producer_manifest.name}",
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
                    "predictions_a": DynamicStepInputDefinition(
                        parameter_specification=ParameterSpecification(
                            parameter_name="predictions_a"
                        ),
                        category=NodeInputCategory.NON_BATCH_STEP_OUTPUT,
                        data_lineage=[],
                        selector="$steps.model_a.predictions",
                    ),
                    "predictions_b": DynamicStepInputDefinition(
                        parameter_specification=ParameterSpecification(
                            parameter_name="predictions_b"
                        ),
                        category=NodeInputCategory.NON_BATCH_STEP_OUTPUT,
                        data_lineage=[],
                        selector="$steps.model_b.predictions",
                    ),
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
    graph.add_edge("$steps.model_a", "$steps.consumer")
    graph.add_edge("$steps.model_b", "$steps.consumer")
    graph.add_edge("$steps.consumer", "$outputs.result")

    return CompiledWorkflow(
        workflow_definition=ParsedWorkflowDefinition(
            version="1.0",
            inputs=[],
            steps=[producer_a_manifest, producer_b_manifest, consumer_manifest],
            outputs=[workflow_output],
        ),
        execution_graph=graph,
        steps={
            "model_a": InitialisedStep(
                block_specification=BlockSpecification(
                    block_source="test",
                    identifier="test/deferred_producer@v1",
                    block_class=TaggedDeferringProducerBlock,
                    manifest_class=DeferredProducerManifest,
                ),
                manifest=producer_a_manifest,
                step=producer_a,
            ),
            "model_b": InitialisedStep(
                block_specification=BlockSpecification(
                    block_source="test",
                    identifier="test/deferred_producer@v1",
                    block_class=TaggedDeferringProducerBlock,
                    manifest_class=DeferredProducerManifest,
                ),
                manifest=producer_b_manifest,
                step=producer_b,
            ),
            "consumer": InitialisedStep(
                block_specification=BlockSpecification(
                    block_source="test",
                    identifier="test/fusion_consumer@v1",
                    block_class=FusionConsumerBlock,
                    manifest_class=FusionConsumerManifest,
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
