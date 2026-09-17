import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.enterprise_blocks.sinks import kafka_common
from inference.enterprise.workflows.enterprise_blocks.sinks.kafka_producer import v1
from inference.enterprise.workflows.enterprise_blocks.sinks.kafka_producer.v1 import (
    BlockManifest,
    KafkaProducerSinkBlockV1,
)

BOOTSTRAP = "broker-1:9092,broker-2:9092"
TOPIC = "vision.detections"
MSK_BOOTSTRAP = (
    "b-1.cluster.abc123.c2.kafka.eu-west-1.amazonaws.com:9098,"
    "b-2.cluster.abc123.c2.kafka.eu-west-1.amazonaws.com:9098"
)


# --------------------------------------------------------------------------------------
# Fake broker / producer standing in for confluent_kafka
# --------------------------------------------------------------------------------------


class FakeMessage:
    def __init__(self, topic: str, partition: int, offset: int):
        self._topic = topic
        self._partition = partition
        self._offset = offset

    def topic(self) -> str:
        return self._topic

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset


class FakeKafkaError:
    def __init__(self, code: int, text: str = "Broker: Unknown topic or partition"):
        self._code = code
        self._text = text

    def code(self) -> int:
        return self._code

    def __str__(self) -> str:
        return self._text


class FakeBroker:
    def __init__(self):
        self.topics = {TOPIC}
        self.creating_topic_replies = 0  # transient "unknown topic" replies to serve
        self.metadata_calls = 0
        self.fail_connect: Optional[Exception] = None
        self.queue_full = False
        self.queue_capacity: Optional[int] = None  # None = unlimited
        self.delivery_error: Optional[str] = None
        self.hang_flush = False  # records stay queued: flush() returns the count
        self.acks_zero = False  # delivery reports carry no offset
        self.construct_error: Optional[Exception] = None
        self.records: List[Dict[str, Any]] = []
        self.producers: List["FakeProducer"] = []


class FakeProducer:
    def __init__(self, broker: FakeBroker, config: Dict[str, Any]):
        self.broker = broker
        self.config = config
        self.pending: List[Any] = []
        self.poll_calls = 0
        self.flush_calls: List[float] = []
        broker.producers.append(self)

    def list_topics(self, topic: str, timeout: float):
        self.broker.metadata_calls += 1
        if self.broker.fail_connect is not None:
            raise self.broker.fail_connect
        if self.broker.creating_topic_replies > 0:
            # what a broker says while auto-creation of the topic is in flight
            self.broker.creating_topic_replies -= 1
            return SimpleNamespace(
                topics={topic: SimpleNamespace(error=FakeKafkaError(3))}
            )
        if topic not in self.broker.topics:
            return SimpleNamespace(
                topics={topic: SimpleNamespace(error=FakeKafkaError(3))}
            )
        return SimpleNamespace(topics={topic: SimpleNamespace(error=None)})

    def produce(self, topic, value=None, key=None, headers=None, on_delivery=None):
        if self.broker.queue_full:
            raise BufferError("Local: Queue full")
        capacity = self.broker.queue_capacity
        if capacity is not None and len(self.pending) >= capacity:
            raise BufferError("Local: Queue full")
        record = {"topic": topic, "value": value, "key": key, "headers": headers}
        self.broker.records.append(record)
        message = FakeMessage(topic, 0, len(self.broker.records) - 1)
        self.pending.append((on_delivery, message))

    def _deliver(self) -> None:
        for callback, message in self.pending:
            if callback is not None:
                if self.broker.acks_zero:
                    message._offset = None
                callback(self.broker.delivery_error, message)
        self.pending = []

    def poll(self, timeout: float) -> int:
        self.poll_calls += 1
        # like librdkafka, delivery reports surface (and free queue slots) on poll,
        # but only once the broker has answered; a hanging broker answers nothing
        if self.broker.hang_flush:
            return 0
        served = len(self.pending)
        self._deliver()
        return served

    def flush(self, timeout: float) -> int:
        self.flush_calls.append(timeout)
        if self.broker.hang_flush:
            return len(self.pending)
        self._deliver()
        return 0


@pytest.fixture
def broker():
    fake_broker = FakeBroker()

    def make_producer(config):
        if fake_broker.construct_error is not None:
            raise fake_broker.construct_error
        return FakeProducer(fake_broker, config)

    fake_module = SimpleNamespace(Producer=make_producer, KafkaError=SimpleNamespace())
    with patch.object(v1, "confluent_kafka", fake_module):
        yield fake_broker


class FakeTokenProvider:
    calls: List[str] = []

    @staticmethod
    def generate_auth_token(region: str):
        FakeTokenProvider.calls.append(region)
        return "signed-token", 1_700_000_000_000


class FailingTokenProvider:
    @staticmethod
    def generate_auth_token(region: str):
        class NoCredentialsError(Exception):
            pass

        raise NoCredentialsError("Unable to locate credentials")


@pytest.fixture
def msk_signer():
    FakeTokenProvider.calls = []
    with patch.object(kafka_common, "MSKAuthTokenProvider", FakeTokenProvider):
        yield FakeTokenProvider


def block(disable_sinks: bool = False) -> KafkaProducerSinkBlockV1:
    return KafkaProducerSinkBlockV1(None, None, disable_sinks=disable_sinks)


def run(instance: KafkaProducerSinkBlockV1, **overrides) -> Dict[str, Any]:
    kwargs = {
        "bootstrap_servers": BOOTSTRAP,
        "topic": TOPIC,
        "message": {"state": "RUNNING"},
        "fire_and_forget": False,
    }
    kwargs.update(overrides)
    return instance.run(**kwargs)


def producer_config(fake_broker: FakeBroker) -> Dict[str, Any]:
    return fake_broker.producers[0].config


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------


def manifest(**overrides) -> BlockManifest:
    data = {
        "type": "roboflow_enterprise/kafka_producer_sink@v1",
        "name": "kafka",
        "bootstrap_servers": BOOTSTRAP,
        "topic": TOPIC,
        "message": "$steps.payload.output",
    }
    data.update(overrides)
    return BlockManifest.model_validate(data)


def test_manifest_defaults() -> None:
    result = manifest()

    assert result.provider == "Self-hosted"
    assert result.username is None and result.password is None
    assert result.aws_region is None
    assert result.key is None and result.headers is None
    assert result.fire_and_forget is True
    assert result.acks == "all" and result.timeout == 5.0


@pytest.mark.parametrize("provider", ["Self-hosted", "AWS MSK"])
def test_manifest_accepts_known_providers(provider: str) -> None:
    assert manifest(provider=provider).provider == provider


def test_manifest_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError):
        manifest(provider="Confluent Cloud")


@pytest.mark.parametrize("acks, expected", [("0", "0"), (1, "1"), ("all", "all")])
def test_manifest_accepts_and_coerces_acks(acks: Any, expected: str) -> None:
    assert manifest(acks=acks).acks == expected


@pytest.mark.parametrize(
    "field, value",
    [
        ("acks", "2"),
        ("acks", True),
        ("acks", "some"),
        ("timeout", 0),
        ("timeout", "soon"),
        ("timeout", float("inf")),
        ("timeout", True),
        ("message", 5),
    ],
)
def test_manifest_rejects_invalid_values(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        manifest(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [("acks", "$inputs.acks"), ("timeout", "$inputs.timeout")],
)
def test_manifest_lets_selectors_bypass_validation(field: str, value: str) -> None:
    assert getattr(manifest(**{field: value}), field) == value


def test_manifest_accepts_dict_and_string_messages() -> None:
    assert manifest(message={"a": 1}).message == {"a": 1}
    assert manifest(message="plain").message == "plain"


def test_manifest_metadata_matches_the_consumer_block() -> None:
    properties = BlockManifest.model_json_schema()["properties"]

    for field in ("username", "password"):
        assert properties[field]["relevant_for"] == {
            "provider": {"values": ["Self-hosted"], "required": False}
        }
    assert properties["aws_region"]["relevant_for"] == {
        "provider": {"values": ["AWS MSK"], "required": False}
    }
    assert properties["password"]["private"] is True
    for field in ("acks", "ssl_ca_location", "timeout"):
        assert properties[field]["additional_section"] is True
    for field in ("bootstrap_servers", "topic", "provider", "message"):
        assert properties[field]["always_visible"] is True


def test_manifest_outputs_compatibility_and_restrictions() -> None:
    assert [o.name for o in BlockManifest.describe_outputs()] == [
        "error_status",
        "message",
    ]
    assert BlockManifest.get_execution_engine_compatibility() == ">=1.4.0,<2.0.0"
    restrictions = [r.to_dict() for r in BlockManifest.get_restrictions()]
    assert len(restrictions) == 1 and restrictions[0]["severity"] == "soft"
    assert restrictions[0]["applies_to_runtimes"] == ["inference_pipeline"]


def test_block_init_parameters() -> None:
    assert KafkaProducerSinkBlockV1.get_init_parameters() == [
        "background_tasks",
        "thread_pool_executor",
        "disable_sinks",
    ]
    assert KafkaProducerSinkBlockV1.get_manifest() is BlockManifest


# --------------------------------------------------------------------------------------
# Publishing
# --------------------------------------------------------------------------------------


def test_confirmed_delivery_publishes_and_reports_position(broker: FakeBroker) -> None:
    # when
    instance = block()
    result = run(
        instance,
        message={"a": 1},
        key="cam-1",
        headers={"source": "inference"},
        fire_and_forget=False,
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Message delivered to partition 0 at offset 0",
    }
    assert broker.records == [
        {
            "topic": TOPIC,
            "value": b'{"a": 1}',
            "key": b"cam-1",
            "headers": [("source", b"inference")],
        }
    ]
    producer = broker.producers[0]
    assert producer.poll_calls == 1
    assert producer.flush_calls == [5.0]


def test_fire_and_forget_returns_without_flushing(broker: FakeBroker) -> None:
    instance = block()
    result = run(instance, fire_and_forget=True)

    assert result == {
        "error_status": False,
        "message": "Message scheduled for delivery",
    }
    producer = broker.producers[0]
    assert producer.flush_calls == []
    assert producer.poll_calls == 1
    assert len(broker.records) == 1


def test_fire_and_forget_delivery_error_is_only_logged(broker: FakeBroker) -> None:
    broker.delivery_error = "Broker: Not enough in-sync replicas"
    instance = block()

    first = run(instance, fire_and_forget=True)
    # the background delivery callback runs later; simulate librdkafka doing so
    broker.producers[0]._deliver()
    second = run(instance, fire_and_forget=True)

    assert first["error_status"] is False and second["error_status"] is False


def test_confirmed_delivery_error_is_returned(broker: FakeBroker) -> None:
    broker.delivery_error = "Broker: Message size too large"

    result = run(block(), fire_and_forget=False)

    assert result["error_status"] is True
    assert "Kafka rejected the message" in result["message"]
    assert "Message size too large" in result["message"]


def test_confirmed_delivery_timeout_is_reported_as_unknown(broker: FakeBroker) -> None:
    broker.hang_flush = True
    instance = block()

    result = run(instance, fire_and_forget=False, timeout=2.5)

    assert result["error_status"] is True
    assert "not acknowledged within 2.5s" in result["message"]
    assert "may still be delivered" in result["message"]
    assert broker.producers[0].flush_calls == [2.5]


def test_full_local_queue_is_reported(broker: FakeBroker) -> None:
    broker.queue_full = True

    result = run(block())

    assert result["error_status"] is True
    assert "local queue is full" in result["message"]


def test_full_queue_recovers_once_delivery_reports_are_served(
    broker: FakeBroker,
) -> None:
    # given a producer whose queue holds one record until poll() serves its report
    broker.queue_capacity = 1
    instance = block()
    first = run(instance, message="m0", fire_and_forget=True)  # occupies the slot

    # when the next run finds the queue full
    second = run(instance, message="m1", fire_and_forget=True)

    # then it polls, frees the slot and publishes instead of failing forever
    assert first["error_status"] is False
    assert second["error_status"] is False
    assert [r["value"] for r in broker.records] == [b"m0", b"m1"]


def test_string_message_is_sent_as_utf8(broker: FakeBroker) -> None:
    run(block(), message="héllo")

    assert broker.records[0]["value"] == "héllo".encode("utf-8")
    assert broker.records[0]["key"] is None
    assert broker.records[0]["headers"] is None


def test_non_serialisable_message_fails_before_publishing(broker: FakeBroker) -> None:
    result = run(block(), message={"bad": {1, 2}})

    assert result["error_status"] is True
    assert "JSON" in result["message"]
    assert broker.records == []


@pytest.mark.parametrize("message", [5, 3.2, None, object()])
def test_unsupported_message_types_fail(broker: FakeBroker, message: Any) -> None:
    result = run(block(), message=message)

    assert result["error_status"] is True
    assert "message must be a string or a dictionary" in result["message"]


def test_headers_must_be_a_dictionary(broker: FakeBroker) -> None:
    result = run(block(), headers=["not", "a", "dict"])

    assert result["error_status"] is True
    assert "headers must be a dictionary" in result["message"]
    assert broker.records == []


def test_header_values_are_stringified(broker: FakeBroker) -> None:
    run(block(), headers={"count": 3, "flag": True, "empty": None})

    assert broker.records[0]["headers"] == [
        ("count", b"3"),
        ("flag", b"True"),
        ("empty", None),
    ]


def test_acks_is_configured_and_record_lifetime_is_left_to_the_client(
    broker: FakeBroker,
) -> None:
    run(block(), acks=1, timeout=2.0)

    config = producer_config(broker)
    assert config["acks"] == "1"
    assert "message.timeout.ms" not in config  # librdkafka default, like any producer
    assert config["bootstrap.servers"] == BOOTSTRAP


def test_timeout_may_change_between_runs(broker: FakeBroker) -> None:
    instance = block()
    run(instance, timeout=2.0)

    result = run(instance, timeout=9.0)

    assert result["error_status"] is False
    assert broker.producers[0].flush_calls == [2.0, 9.0]


@pytest.mark.parametrize(
    "field, value",
    [
        ("message", json.loads('"\\ud800"')),
        ("key", json.loads('"\\udfff"')),
        ("headers", {"h": json.loads('"\\ud800"')}),
    ],
)
def test_text_that_cannot_be_utf8_encoded_is_a_failure_not_an_exception(
    broker: FakeBroker, field: str, value: Any
) -> None:
    result = run(block(), **{field: value})

    assert result["error_status"] is True
    assert "cannot be encoded as UTF-8" in result["message"]
    assert broker.records == []


def test_acks_zero_reports_no_offset(broker: FakeBroker) -> None:
    broker.acks_zero = True
    instance = block()

    result = run(instance, acks="0")

    assert result["error_status"] is False
    assert result["message"] == (
        "Message sent to partition 0 without acknowledgement (acks=0)"
    )


def test_delivery_error_keeps_the_auth_cause(broker: FakeBroker, msk_signer) -> None:
    broker.delivery_error = "Local: Message timed out"
    instance = block()
    run(instance, bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")
    instance._auth_failures.append(RuntimeError("STS blip"))
    broker.delivery_error = "Local: Message timed out"

    result = run(instance, bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")

    assert result["error_status"] is True
    assert "Kafka rejected the message" in result["message"]
    assert "AWS MSK IAM authentication failed" in result["message"]


def test_empty_string_credentials_mean_no_credentials(broker: FakeBroker) -> None:
    result = run(block(), username="", password="")

    assert result["error_status"] is False
    assert producer_config(broker)["security.protocol"] == "PLAINTEXT"


def test_timeout_above_librdkafka_limit_is_rejected(broker: FakeBroker) -> None:
    result = run(block(), timeout=3_000_000)

    assert result["error_status"] is True
    assert "timeout" in result["message"] and "2147483" in result["message"]
    assert broker.producers == []


def test_invalid_client_configuration_is_not_reported_as_unreachable(
    broker: FakeBroker,
) -> None:
    class FakeKafkaError:
        def code(self):
            return "INVALID_ARG"

    broker.construct_error = Exception(FakeKafkaError())
    v1.confluent_kafka.KafkaError = SimpleNamespace(_INVALID_ARG="INVALID_ARG")

    result = run(block())

    assert result["error_status"] is True
    assert "Invalid Kafka client configuration" in result["message"]


def test_producer_is_reused_across_runs(broker: FakeBroker) -> None:
    instance = block()

    run(instance)
    run(instance)
    run(instance)

    assert len(broker.producers) == 1
    assert len(broker.records) == 3


# --------------------------------------------------------------------------------------
# Providers and connection config
# --------------------------------------------------------------------------------------


def test_self_hosted_without_credentials_uses_plaintext(broker: FakeBroker) -> None:
    run(block())

    config = producer_config(broker)
    assert config["security.protocol"] == "PLAINTEXT"
    assert not any(key.startswith("sasl.") for key in config)
    assert "group.id" not in config


def test_self_hosted_with_credentials_uses_scram_over_tls(broker: FakeBroker) -> None:
    run(block(), username="inference", password="s3cret")

    config = producer_config(broker)
    assert config["security.protocol"] == "SASL_SSL"
    assert config["sasl.mechanisms"] == "SCRAM-SHA-512"
    assert config["sasl.username"] == "inference"
    assert config["sasl.password"] == "s3cret"


@pytest.mark.parametrize(
    "credentials", [{"username": "inference"}, {"password": "s3cret"}]
)
def test_one_sided_credentials_fail_before_connecting(
    broker: FakeBroker, credentials: Dict[str, str]
) -> None:
    result = run(block(), **credentials)

    assert result["error_status"] is True
    assert "both username and password" in result["message"]
    assert broker.producers == []


def test_ssl_ca_location_is_passed_through(broker: FakeBroker) -> None:
    run(block(), username="u", password="p", ssl_ca_location="/ca.pem")

    assert producer_config(broker)["ssl.ca.location"] == "/ca.pem"


def test_aws_msk_derives_region_and_uses_oauthbearer(
    broker: FakeBroker, msk_signer
) -> None:
    result = run(block(), bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")

    config = producer_config(broker)
    assert result["error_status"] is False
    assert config["security.protocol"] == "SASL_SSL"
    assert config["sasl.mechanisms"] == "OAUTHBEARER"
    assert callable(config["oauth_cb"])
    assert "sasl.username" not in config and "sasl.password" not in config
    assert msk_signer.calls == ["eu-west-1"]  # signed once up front


def test_aws_msk_explicit_region_wins_over_hostname(
    broker: FakeBroker, msk_signer
) -> None:
    run(
        block(),
        bootstrap_servers="kafka.internal:9098",
        provider="AWS MSK",
        aws_region="us-east-1",
    )

    assert msk_signer.calls == ["us-east-1"]


def test_aws_msk_without_derivable_region_fails_before_connecting(
    broker: FakeBroker, msk_signer
) -> None:
    result = run(block(), bootstrap_servers="kafka.internal:9098", provider="AWS MSK")

    assert result["error_status"] is True
    assert "aws_region" in result["message"]
    assert broker.producers == [] and msk_signer.calls == []


def test_aws_msk_ignores_username_and_password(broker: FakeBroker, msk_signer) -> None:
    result = run(
        block(),
        bootstrap_servers=MSK_BOOTSTRAP,
        provider="AWS MSK",
        username="ignored",
        password="ignored",
    )

    assert result["error_status"] is False
    config = producer_config(broker)
    assert "sasl.username" not in config and "sasl.password" not in config


def test_aws_msk_signer_failure_fails_fast_and_retries(broker: FakeBroker) -> None:
    instance = block()

    with patch.object(kafka_common, "MSKAuthTokenProvider", FailingTokenProvider):
        failed = run(instance, bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")
    with patch.object(kafka_common, "MSKAuthTokenProvider", FakeTokenProvider):
        recovered = run(instance, bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")

    assert failed["error_status"] is True
    assert "AWS MSK IAM authentication failed" in failed["message"]
    assert "no AWS credentials were found" in failed["message"]
    assert recovered["error_status"] is False
    assert len(broker.producers) == 1  # nothing was built while signing failed


def test_transient_token_refresh_failure_is_reported_once(
    broker: FakeBroker, msk_signer
) -> None:
    instance = block()
    run(instance, bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")
    # librdkafka invoked the refresh callback in the background and it failed once
    instance._auth_failures.append(RuntimeError("STS blip"))

    failed = run(instance, bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")
    recovered = run(instance, bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")

    # the record itself was delivered, and the refresh failure rides along once
    assert failed["error_status"] is True
    assert "Message delivered" in failed["message"]
    assert "AWS MSK IAM authentication failed" in failed["message"]
    assert recovered["error_status"] is False
    assert len(broker.producers) == 1
    assert len(broker.records) == 3


def test_aws_msk_requires_signer_package(broker: FakeBroker) -> None:
    with patch.object(kafka_common, "MSKAuthTokenProvider", None):
        result = run(block(), bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")

    assert result["error_status"] is True
    assert "aws-msk-iam-sasl-signer-python" in result["message"]


# --------------------------------------------------------------------------------------
# Lifecycle and failure handling
# --------------------------------------------------------------------------------------


def test_unreachable_broker_reports_error_and_retries_next_run(
    broker: FakeBroker,
) -> None:
    broker.fail_connect = RuntimeError("Local: Broker transport failure")
    instance = block()

    failed = run(instance, timeout=0.5)
    broker.fail_connect = None
    recovered = run(instance, timeout=0.5)

    assert failed["error_status"] is True
    assert "Kafka broker not reachable" in failed["message"]
    assert "transport failure" in failed["message"]
    assert recovered["error_status"] is False
    assert len(broker.producers) == 2
    assert broker.records == [broker.records[0]]  # only the recovered run published


def test_missing_topic_is_reported_once_the_timeout_passes(broker: FakeBroker) -> None:
    result = run(block(), topic="does-not-exist", timeout=0.5)

    assert result["error_status"] is True
    assert "does-not-exist" in result["message"]
    assert broker.records == []
    assert broker.metadata_calls > 1  # kept asking until the budget ran out


def test_topic_still_being_created_is_waited_for(broker: FakeBroker) -> None:
    # given the broker answers "unknown topic" twice while auto-creating it
    broker.creating_topic_replies = 2

    result = run(block(), timeout=5.0)

    assert result["error_status"] is False
    assert broker.metadata_calls == 3
    assert len(broker.records) == 1


def test_non_transient_topic_error_fails_immediately(broker: FakeBroker) -> None:
    class Denied(FakeKafkaError):
        pass

    broker.topics.discard(TOPIC)
    original = FakeProducer.list_topics

    def denied(self, topic, timeout):
        self.broker.metadata_calls += 1
        return SimpleNamespace(
            topics={
                topic: SimpleNamespace(error=Denied(29, "Topic authorization failed"))
            }
        )

    FakeProducer.list_topics = denied
    try:
        result = run(block(), timeout=5.0)
    finally:
        FakeProducer.list_topics = original

    assert result["error_status"] is True
    assert "authorization failed" in result["message"]
    assert broker.metadata_calls == 1


@pytest.mark.parametrize(
    "change",
    [
        {"bootstrap_servers": "elsewhere:9092"},
        {"topic": "other"},
        {"provider": "AWS MSK", "aws_region": "us-east-1"},
        {"username": "u", "password": "p"},
        {"acks": "1"},
    ],
)
def test_changed_connection_parameters_are_rejected(
    broker: FakeBroker, change: Dict[str, Any]
) -> None:
    broker.topics.add("other")
    instance = block()
    run(instance)

    with patch.object(kafka_common, "MSKAuthTokenProvider", FakeTokenProvider):
        result = run(instance, **change)

    assert result["error_status"] is True
    assert "changed between runs" in result["message"]
    assert len(broker.producers) == 1
    assert len(broker.records) == 1


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"timeout": 0}, "timeout"),
        ({"timeout": "soon"}, "timeout"),
        ({"acks": "2"}, "acks"),
        ({"acks": True}, "acks"),
        ({"acks": "ALL"}, "acks"),
        ({"timeout": True}, "timeout"),
        ({"fire_and_forget": "maybe"}, "fire_and_forget"),
        ({"bootstrap_servers": "  "}, "bootstrap_servers"),
        ({"topic": ""}, "topic"),
        ({"provider": "Other"}, "provider"),
    ],
)
def test_selector_resolved_values_are_revalidated_before_connecting(
    broker: FakeBroker, overrides: Dict[str, Any], fragment: str
) -> None:
    result = run(block(), **overrides)

    assert result["error_status"] is True
    assert fragment in result["message"]
    assert broker.producers == []


def test_selector_resolved_values_are_coerced(broker: FakeBroker) -> None:
    instance = block()
    result = run(instance, acks=0, timeout="2", fire_and_forget="false")

    assert result["error_status"] is False
    assert producer_config(broker)["acks"] == "0"
    assert broker.producers[0].flush_calls == [2.0]


@pytest.mark.parametrize("flag", ["GCP_SERVERLESS", "LAMBDA"])
def test_hosted_platform_is_refused(broker: FakeBroker, flag: str) -> None:
    with patch.object(v1, flag, True):
        result = run(block())

    assert result["error_status"] is True
    assert "hosted platform" in result["message"]
    assert broker.producers == []


def test_disabled_sink_publishes_nothing(broker: FakeBroker) -> None:
    result = run(block(disable_sinks=True))

    assert result["error_status"] is False
    assert "disabled" in result["message"]
    assert broker.producers == []


def test_missing_confluent_kafka_is_reported() -> None:
    with patch.object(v1, "confluent_kafka", None):
        result = run(block())

    assert result["error_status"] is True
    assert "confluent-kafka" in result["message"]


def test_close_flushes_with_last_timeout_and_is_idempotent(broker: FakeBroker) -> None:
    instance = block()
    run(instance, fire_and_forget=True, timeout=3.0)
    producer = broker.producers[0]

    instance.close()
    instance.close()

    assert producer.flush_calls == [3.0]
    assert instance._producer is None


def test_dropping_the_instance_flushes_pending_records(broker: FakeBroker) -> None:
    run(
        block(), fire_and_forget=True, timeout=1.5
    )  # instance is unreachable after this

    assert broker.producers[0].flush_calls == [1.5]


def test_close_before_any_run_is_a_noop(broker: FakeBroker) -> None:
    block().close()

    assert broker.producers == []


def test_run_after_close_reconnects(broker: FakeBroker) -> None:
    instance = block()
    run(instance)
    instance.close()

    result = run(instance)

    assert result["error_status"] is False
    assert len(broker.producers) == 2


def test_json_encoding_matches_python_default(broker: FakeBroker) -> None:
    payload = {"state": "RUNNING", "n": 3, "nested": {"ok": True}}

    run(block(), message=payload)

    assert broker.records[0]["value"] == json.dumps(payload).encode("utf-8")
