"""Workload declarations of the two Kafka blocks.

Both blocks carry a HARD "not available on the Roboflow hosted platform"
restriction, the consumer keeps its connection and last record in the block
instance, and the producer's delivery caveat depends on a manifest field that
may hold a selector. The portable hook has to express all of that as conditions
without reading this host's ``GCP_SERVERLESS`` / ``LAMBDA`` flags.

The tests go through the real manifests and the real public introspection API.
No broker is contacted and no block instance is built: one test arms spies on
both module-level Kafka clients and on both block constructors, and the
description is produced anyway.
"""

from typing import Any, Dict, List, Set, Tuple

import pytest
import roboflow_workflows.enterprise_blocks.sinks.kafka_consumer.v1 as consumer_module
import roboflow_workflows.enterprise_blocks.sinks.kafka_producer.v1 as producer_module
from roboflow_workflows.enterprise_blocks.sinks.kafka_consumer.v1 import (
    KAFKA_CONSUMER_PER_REQUEST_PORTABLE_RESTRICTION,
)
from roboflow_workflows.enterprise_blocks.sinks.kafka_consumer.v1 import (
    BlockManifest as ConsumerManifest,
)
from roboflow_workflows.enterprise_blocks.sinks.kafka_consumer.v1 import (
    KafkaConsumerBlockV1,
)
from roboflow_workflows.enterprise_blocks.sinks.kafka_producer.v1 import (
    BlockManifest as ProducerManifest,
)
from roboflow_workflows.enterprise_blocks.sinks.kafka_producer.v1 import (
    KafkaProducerSinkBlockV1,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RestrictionMetadata,
    Runtime,
    RuntimeInputMode,
    Severity,
    WorkOperation,
)
from roboflow_workflows.execution_engine.introspection import blocks_loader
from roboflow_workflows.execution_engine.introspection.workload import (
    describe_workflow_workload,
)

ENTERPRISE_PLUGIN = "roboflow_workflows.enterprise_blocks.loader"
CONSUMER_TYPE = "roboflow_enterprise/kafka_consumer@v1"
PRODUCER_TYPE = "roboflow_enterprise/kafka_producer_sink@v1"
HOSTED_RESTRICTION_CODE = "unavailable_on_hosted_platform"
CONSUMER_STATE_RESTRICTION_CODE = "connection_and_state_rebuilt_per_request"
FIRE_AND_FORGET_CODE = "fire_and_forget_hides_persistence_failures"
HOSTED_PLATFORM_FLAGS = ("GCP_SERVERLESS", "LAMBDA")


def _consumer() -> ConsumerManifest:
    return ConsumerManifest(
        type=CONSUMER_TYPE,
        name="consumer",
        bootstrap_servers="localhost:9092",
        topic="merge-audit",
    )


def _producer(fire_and_forget: Any = True) -> ProducerManifest:
    return ProducerManifest(
        type=PRODUCER_TYPE,
        name="producer",
        bootstrap_servers="localhost:9092",
        topic="merge-audit",
        message="metadata-only probe",
        fire_and_forget=fire_and_forget,
    )


def _kafka_workflow_definition(fire_and_forget: Any = True) -> Dict[str, Any]:
    """The Kafka-only workflow from the merge-audit reproducer."""
    return {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "fire_and_forget",
                "default_value": True,
            }
        ],
        "steps": [
            {
                "type": CONSUMER_TYPE,
                "name": "consumer",
                "bootstrap_servers": "localhost:9092",
                "topic": "merge-audit",
            },
            {
                "type": PRODUCER_TYPE,
                "name": "producer",
                "bootstrap_servers": "localhost:9092",
                "topic": "merge-audit",
                "message": "metadata-only probe",
                "fire_and_forget": fire_and_forget,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "consumer_error",
                "selector": "$steps.consumer.error_status",
            },
            {
                "type": "JsonField",
                "name": "producer_error",
                "selector": "$steps.producer.error_status",
            },
        ],
    }


def _describe_with_enterprise_plugin(definition: Dict[str, Any]) -> Any:
    """Describe a definition with the enterprise plugin registered.

    The plugin list is read from the environment at call time, so every loader
    cache is cleared on the way in AND on the way out - a cached registry must
    not leak into other modules.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("WORKFLOWS_PLUGINS", ENTERPRISE_PLUGIN)
        blocks_loader.clear_caches()
        try:
            return describe_workflow_workload(definition)
        finally:
            blocks_loader.clear_caches()


def _codes(restrictions: Any) -> List[str]:
    items = restrictions.items if isinstance(restrictions, Discovery) else restrictions
    return [restriction.code for restriction in items]


def _portable_axes(restriction: RestrictionMetadata) -> Tuple[Any, ...]:
    condition = restriction.when
    return (
        restriction.severity.value,
        tuple(sorted(item.value for item in (condition.runtimes or ()))),
        tuple(sorted(item.value for item in (condition.step_execution_modes or ()))),
        tuple(sorted(item.value for item in (condition.input_modes or ()))),
    )


def _legacy_axes(restriction: Any) -> Tuple[Any, ...]:
    return (
        restriction.severity.value,
        tuple(sorted(item.value for item in (restriction.applies_to_runtimes or ()))),
        tuple(
            sorted(
                item.value
                for item in (restriction.applies_to_step_execution_modes or ())
            )
        ),
        tuple(
            sorted(item.value for item in (restriction.applies_to_input_modes or ()))
        ),
    )


# ---------------------------------------------------------------------------
# The public API answer for the Kafka-only workflow
# ---------------------------------------------------------------------------


def test_a_kafka_only_workflow_is_described_completely() -> None:
    """The merge-audit reproducer, as the API answers it.

    Before the declarations existed this workflow came back with empty,
    incomplete operations / restrictions / resources and an incomplete model
    inventory.
    """
    description = _describe_with_enterprise_plugin(_kafka_workflow_definition())
    by_node = {step.node_id: step for step in description.steps}
    assert set(by_node) == {"$steps.consumer", "$steps.producer"}

    consumer = by_node["$steps.consumer"]
    assert consumer.block_type == CONSUMER_TYPE
    assert consumer.operations.complete
    assert consumer.operations.items == [
        WorkOperation.EXTERNAL_REQUEST,
        WorkOperation.TEMPORAL_BUFFERING,
    ]
    assert consumer.restrictions.complete
    assert _codes(consumer.restrictions) == [
        CONSUMER_STATE_RESTRICTION_CODE,
        HOSTED_RESTRICTION_CODE,
    ]

    producer = by_node["$steps.producer"]
    assert producer.block_type == PRODUCER_TYPE
    assert producer.operations.complete
    assert producer.operations.items == [WorkOperation.EXTERNAL_REQUEST]
    assert producer.restrictions.complete
    assert _codes(producer.restrictions) == [
        FIRE_AND_FORGET_CODE,
        HOSTED_RESTRICTION_CODE,
    ]

    # known absence of external resources, hence a complete (empty) model
    # inventory for the whole workflow - never a claimed model identity
    for step in description.steps:
        assert step.resources.complete
        assert step.resources.items == []
        assert WorkOperation.MODEL_INFERENCE not in step.operations.items
    assert description.summary.models.complete
    assert description.summary.models.items == []


def test_the_hard_hosted_platform_restriction_reaches_the_public_api() -> None:
    """The legacy HARD note both blocks carry is now structured output."""
    description = _describe_with_enterprise_plugin(_kafka_workflow_definition())
    for step in description.steps:
        hard = [
            restriction
            for restriction in step.restrictions.items
            if restriction.severity is Severity.HARD
        ]
        assert [restriction.code for restriction in hard] == [HOSTED_RESTRICTION_CODE]
        assert hard[0].when.runtimes == [Runtime.HOSTED_SERVERLESS]
        # the runtime axis carries the condition; no host flag is named
        assert hard[0].when.configuration_equals == {}


class _ExplodingKafkaClient:
    """Stands in for the ``confluent_kafka`` module during introspection."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"introspection touched the Kafka client: {name}")


def test_describing_the_workflow_builds_no_client_and_no_block_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compile-time facts only: no constructor runs, no broker is contacted."""

    def _forbid_init(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("introspection constructed a Kafka block instance")

    for module in (consumer_module, producer_module):
        monkeypatch.setattr(module, "confluent_kafka", _ExplodingKafkaClient())
    for block_class in (KafkaConsumerBlockV1, KafkaProducerSinkBlockV1):
        monkeypatch.setattr(block_class, "__init__", _forbid_init)

    # the spies are armed - otherwise this test would pass vacuously
    with pytest.raises(AssertionError):
        consumer_module.confluent_kafka.Consumer({})
    with pytest.raises(AssertionError):
        KafkaProducerSinkBlockV1()

    description = _describe_with_enterprise_plugin(_kafka_workflow_definition())
    assert all(step.operations.complete for step in description.steps)
    assert all(step.restrictions.complete for step in description.steps)
    assert all(step.resources.complete for step in description.steps)


def test_consumer_declares_the_per_instance_connection_caveat() -> None:
    """The cross-run state is declared with the runtimes and the input mode."""
    declared = _consumer().discover_portable_restrictions()
    assert isinstance(declared, list), "nothing about this block is conditional"
    assert _codes(declared) == [
        HOSTED_RESTRICTION_CODE,
        CONSUMER_STATE_RESTRICTION_CODE,
    ]
    state = KAFKA_CONSUMER_PER_REQUEST_PORTABLE_RESTRICTION
    assert state.severity is Severity.SOFT
    assert set(state.when.runtimes) == {
        Runtime.SELF_HOSTED_CPU,
        Runtime.SELF_HOSTED_GPU,
        Runtime.DEDICATED_DEPLOYMENT,
    }
    assert state.when.input_modes == [RuntimeInputMode.IMAGE]


@pytest.mark.parametrize("fire_and_forget", [True, False, "$inputs.fire_and_forget"])
def test_producer_fire_and_forget_branches_through_the_public_api(
    fire_and_forget: Any,
) -> None:
    description = _describe_with_enterprise_plugin(
        _kafka_workflow_definition(fire_and_forget)
    )
    restrictions = {step.node_id: step.restrictions for step in description.steps}[
        "$steps.producer"
    ]
    codes = set(_codes(restrictions))
    # the hard restriction holds whatever the switch turns out to be
    assert HOSTED_RESTRICTION_CODE in codes
    if isinstance(fire_and_forget, bool):
        assert restrictions.complete
        assert (FIRE_AND_FORGET_CODE in codes) is fire_and_forget
    else:
        # the input's default_value is True; the declaration must NOT adopt it
        assert not restrictions.complete
        assert FIRE_AND_FORGET_CODE not in codes
        assert restrictions.unknown_reasons == [
            "fire_and_forget_selector_unresolved:$steps.producer"
        ]


# ---------------------------------------------------------------------------
# Portability and legacy parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gcp_serverless, lambda_runtime",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_portable_declarations_ignore_this_host_hosted_platform_flags(
    monkeypatch: pytest.MonkeyPatch, gcp_serverless: bool, lambda_runtime: bool
) -> None:
    """``run()`` branches on the flags; the declaration must not.

    The flags are patched as module CONSTANTS, which is what the blocks import
    and what ``run()`` genuinely reads.
    """
    baseline = (
        _consumer().discover_portable_restrictions(),
        _producer(True).discover_portable_restrictions(),
        _producer(False).discover_portable_restrictions(),
    )
    for module in (consumer_module, producer_module):
        for flag, value in zip(HOSTED_PLATFORM_FLAGS, (gcp_serverless, lambda_runtime)):
            assert hasattr(module, flag), flag
            monkeypatch.setattr(module, flag, value)
    assert (
        _consumer().discover_portable_restrictions(),
        _producer(True).discover_portable_restrictions(),
        _producer(False).discover_portable_restrictions(),
    ) == baseline


def test_every_legacy_axis_survives_in_the_portable_declaration() -> None:
    """The legacy notes and the portable codes describe the same situations.

    ``fire_and_forget`` keeps its default (``True``), which is the branch the
    class-level legacy declaration describes.
    """
    for manifest, manifest_class in (
        (_consumer(), ConsumerManifest),
        (_producer(), ProducerManifest),
    ):
        legacy = manifest_class.get_restrictions()
        portable = manifest.discover_portable_restrictions()
        assert len(legacy) == len(portable) == 2
        legacy_axes: Set[Tuple[Any, ...]] = {
            _legacy_axes(restriction) for restriction in legacy
        }
        portable_axes: Set[Tuple[Any, ...]] = {
            _portable_axes(restriction) for restriction in portable
        }
        assert legacy_axes == portable_axes
        # the legacy API keeps its human notes and stays a classmethod
        assert all(restriction.note for restriction in legacy)
