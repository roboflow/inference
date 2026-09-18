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


class FakeClientError:
    """What librdkafka hands to `error_cb`: a KafkaError with name / code / fatal."""

    def __init__(
        self,
        text: str = "2/2 brokers are down",
        name: str = "_ALL_BROKERS_DOWN",
        code: int = -187,
        fatal: bool = False,
    ):
        self._text = text
        self._name = name
        self._code = code
        self._fatal = fatal

    def name(self) -> str:
        return self._name

    def code(self) -> int:
        return self._code

    def fatal(self) -> bool:
        return self._fatal

    def __str__(self) -> str:
        return self._text


class FakeBroker:
    # `fail_connect` makes every metadata request (`list_topics`, which is also the
    # recovery probe) raise; `creating_topic_replies` makes it answer with a
    # topic-level error. Each request is counted in `metadata_calls`.
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


def block(
    disable_sinks: bool = False, allow_access_to_file_system: bool = False
) -> KafkaProducerSinkBlockV1:
    return KafkaProducerSinkBlockV1(
        None,
        None,
        disable_sinks=disable_sinks,
        allow_access_to_file_system=allow_access_to_file_system,
    )


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
    soft = [r for r in restrictions if r["severity"] == "soft"]
    assert len(restrictions) == 2 and len(soft) == 1
    assert soft[0]["applies_to_runtimes"] == ["inference_pipeline"]


def test_manifest_declares_one_hard_restriction_for_hosted_serverless() -> None:
    # run() refuses when GCP_SERVERLESS or LAMBDA is set; dedicated deployments are not
    # refused, so they must not be listed
    hard = [
        r for r in BlockManifest.get_restrictions() if r.severity is v1.Severity.HARD
    ]

    assert len(hard) == 1
    assert hard[0].applies_to_runtimes == [v1.Runtime.HOSTED_SERVERLESS]
    assert hard[0].applies_to_step_execution_modes is None
    assert hard[0].applies_to_input_modes is None
    assert "error_status=true" in hard[0].note


def test_block_init_parameters() -> None:
    assert KafkaProducerSinkBlockV1.get_init_parameters() == [
        "background_tasks",
        "thread_pool_executor",
        "disable_sinks",
        "allow_access_to_file_system",
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
    result = run(
        block(allow_access_to_file_system=True),
        username="u",
        password="p",
        ssl_ca_location="/ca.pem",
    )

    assert result["error_status"] is False
    assert producer_config(broker)["ssl.ca.location"] == "/ca.pem"


def test_ssl_ca_location_is_refused_without_file_system_access(
    broker: FakeBroker,
) -> None:
    # the default: a hand-constructed block may not read server-side paths
    result = run(
        KafkaProducerSinkBlockV1(None, None),
        username="u",
        password="p",
        ssl_ca_location="/ca.pem",
    )

    assert result["error_status"] is True
    assert "ssl_ca_location" in result["message"]
    assert "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE" in result["message"]
    assert broker.producers == [] and broker.records == []


@pytest.mark.parametrize("ssl_ca_location", [None, ""])
def test_file_system_flag_has_no_effect_without_ssl_ca_location(
    broker: FakeBroker, ssl_ca_location: Optional[str]
) -> None:
    result = run(
        block(allow_access_to_file_system=False), ssl_ca_location=ssl_ca_location
    )

    assert result["error_status"] is False
    assert "ssl.ca.location" not in producer_config(broker)


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


# --------------------------------------------------------------------------------------
# Operator policy for bootstrap servers
# --------------------------------------------------------------------------------------

ALLOW_USER_SERVERS = "KAFKA_WORKFLOWS_SINKS_ALLOW_USER_PROVIDED_BOOTSTRAP_SERVERS"
WHITELISTED_SERVERS = "KAFKA_WORKFLOWS_SINKS_WHITELISTED_BOOTSTRAP_SERVERS"
OPERATOR_SERVERS = ["kafka-internal-1:9092", "kafka-internal-2:9092"]


def test_default_policy_passes_user_bootstrap_servers_through(
    broker: FakeBroker,
) -> None:
    result = run(block(), bootstrap_servers=" Broker-1:9092,broker-2 ")

    assert result["error_status"] is False
    assert producer_config(broker)["bootstrap.servers"] == "Broker-1:9092,broker-2"


def test_allowlist_accepts_user_servers_when_every_entry_is_listed(
    broker: FakeBroker,
) -> None:
    with patch.object(
        kafka_common, WHITELISTED_SERVERS, ["broker-1:9092", "broker-2", "[::1]:9093"]
    ):
        # case, whitespace, a missing port (= 9092) and bracketed IPv6 all normalise
        result = run(
            block(), bootstrap_servers="BROKER-1:9092 , broker-2:9092,[::1]:9093"
        )

    assert result["error_status"] is False
    assert len(broker.records) == 1
    assert (
        producer_config(broker)["bootstrap.servers"]
        == "BROKER-1:9092 , broker-2:9092,[::1]:9093"
    )


def test_allowlist_rejects_unlisted_entry_without_revealing_the_allowlist(
    broker: FakeBroker,
) -> None:
    with patch.object(kafka_common, WHITELISTED_SERVERS, OPERATOR_SERVERS):
        result = run(
            block(), bootstrap_servers="kafka-internal-1:9092,evil.example.com:9092"
        )

    assert result["error_status"] is True
    assert "evil.example.com:9092" in result["message"]
    assert "kafka-internal" not in result["message"]
    assert broker.producers == [] and broker.records == []


def test_allowlist_compares_the_port(broker: FakeBroker) -> None:
    with patch.object(kafka_common, WHITELISTED_SERVERS, ["kafka-internal-1:9092"]):
        result = run(block(), bootstrap_servers="kafka-internal-1:22")

    assert result["error_status"] is True
    assert broker.producers == []


def test_user_servers_disabled_connects_to_operator_servers(
    broker: FakeBroker,
) -> None:
    instance = block()

    # the `inference` logger does not propagate, so caplog cannot see it
    with patch.object(kafka_common, ALLOW_USER_SERVERS, False), patch.object(
        kafka_common, WHITELISTED_SERVERS, OPERATOR_SERVERS
    ), patch.object(kafka_common.logger, "warning") as warning:
        first = run(instance, bootstrap_servers="evil.example.com:9092")
        # a different workflow value is not a connection change: it is ignored
        second = run(instance, bootstrap_servers="other.example.com:9092")

    assert first["error_status"] is False and second["error_status"] is False
    assert len(broker.producers) == 1 and len(broker.records) == 2
    assert producer_config(broker)["bootstrap.servers"] == ",".join(OPERATOR_SERVERS)
    # one warning per block instance, not per run, and without the workflow's value
    assert warning.call_count == 1
    logged = warning.call_args.args[0] % warning.call_args.args[1:]
    assert "operator-configured" in logged
    assert "evil.example.com" not in logged


@pytest.mark.parametrize("allowlist", [None, []])
def test_user_servers_disabled_without_operator_servers_disables_the_block(
    broker: FakeBroker, allowlist: Optional[List[str]]
) -> None:
    with patch.object(kafka_common, ALLOW_USER_SERVERS, False), patch.object(
        kafka_common, WHITELISTED_SERVERS, allowlist
    ):
        result = run(block())

    assert result["error_status"] is True
    assert "disabled on this deployment" in result["message"]
    assert broker.producers == [] and broker.records == []


def test_aws_msk_region_is_derived_from_the_operator_servers(
    broker: FakeBroker, msk_signer
) -> None:
    with patch.object(kafka_common, ALLOW_USER_SERVERS, False), patch.object(
        kafka_common, WHITELISTED_SERVERS, [MSK_BOOTSTRAP.split(",")[0]]
    ):
        result = run(
            block(),
            bootstrap_servers="b-1.other.abc123.c2.kafka.us-east-1.amazonaws.com:9098",
            provider="AWS MSK",
        )

    assert result["error_status"] is False
    # the region of the cluster the client connects to, not of the ignored value
    assert msk_signer.calls and set(msk_signer.calls) == {"eu-west-1"}


# --------------------------------------------------------------------------------------
# Broker problems reported by the client (error_cb)
# --------------------------------------------------------------------------------------


def connected_block(broker: FakeBroker):
    """A block that published one record, plus the `error_cb` its client was given."""
    instance = block()
    first = run(instance, fire_and_forget=True)
    assert first == {"error_status": False, "message": "Message scheduled for delivery"}
    return instance, producer_config(broker)["error_cb"]


def take_broker_down(broker: FakeBroker, deliveries_hang: bool = True) -> None:
    """A real outage: metadata requests fail, so the recovery probe fails too, and
    (unless a test needs delivery reports) queued records get no answer."""
    broker.fail_connect = RuntimeError("Local: Broker transport failure")
    broker.hang_flush = deliveries_hang


def test_client_config_carries_an_error_callback(broker: FakeBroker) -> None:
    _, error_cb = connected_block(broker)

    assert callable(error_cb)


def test_healthy_runs_never_probe_the_broker(broker: FakeBroker) -> None:
    instance, _ = connected_block(broker)
    calls_after_connect = broker.metadata_calls

    with patch.object(kafka_common, "CLIENT_RECOVERY_PROBE_INTERVAL", 0.0):
        for fire_and_forget in (True, False, True):
            result = run(instance, fire_and_forget=fire_and_forget)
            assert result["error_status"] is False

    assert broker.metadata_calls == calls_after_connect


def test_client_reported_outage_is_reported_on_every_fire_and_forget_run(
    broker: FakeBroker,
) -> None:
    instance, error_cb = connected_block(broker)
    calls_after_connect = broker.metadata_calls
    take_broker_down(broker)

    error_cb(FakeClientError("2/2 brokers are down"))
    first = run(instance, fire_and_forget=True)
    # no new callback: the block must keep claiming the problem
    second = run(instance, fire_and_forget=True)

    for result in (first, second):
        assert list(result) == ["error_status", "message"]
        assert result["error_status"] is True
        assert "reported a broker problem" in result["message"]
        assert "2/2 brokers are down" in result["message"]
        assert "_ALL_BROKERS_DOWN" in result["message"]
        assert "queued" in result["message"]
        assert "currently unreachable" in result["message"]
        # the probe's own failure is not what gets reported
        assert "Broker transport failure" not in result["message"]
    # records are still handed to the client so librdkafka can retry them
    assert len(broker.records) == 3
    assert len(broker.producers[0].pending) == 2
    # the first run probed at once and failed; the second is inside the probe interval
    assert broker.metadata_calls == calls_after_connect + 1


@pytest.mark.parametrize("delivery_reports_arrive", [False, True])
def test_transient_client_error_on_a_healthy_cluster_is_never_reported(
    broker: FakeBroker, delivery_reports_arrive: bool
) -> None:
    # librdkafka also reports single-broker disconnects and idle-connection closes
    # through error_cb; the broker answers the immediate probe, so nothing is claimed
    instance, error_cb = connected_block(broker)
    calls_after_connect = broker.metadata_calls
    broker.hang_flush = not delivery_reports_arrive

    error_cb(FakeClientError("broker-2:9092: Disconnected", name="_TRANSPORT"))
    same_run = run(instance, fire_and_forget=True)
    after = run(instance, fire_and_forget=True)

    expected = {"error_status": False, "message": "Message scheduled for delivery"}
    assert same_run == expected and after == expected
    # without a delivery report the immediate probe clears the blip: exactly one
    # probe. With one, the report clears it first and nothing is probed at all.
    probes = 0 if delivery_reports_arrive else 1
    assert broker.metadata_calls == calls_after_connect + probes


def test_each_new_problem_streak_is_probed_at_once(broker: FakeBroker) -> None:
    instance, error_cb = connected_block(broker)
    calls_after_connect = broker.metadata_calls
    broker.hang_flush = True  # no delivery reports: only the probe can clear

    # three blips in a row, all well inside one probe interval
    for blip in range(1, 4):
        error_cb(FakeClientError("broker-2:9092: Disconnected", name="_TRANSPORT"))
        assert run(instance, fire_and_forget=True)["error_status"] is False
        assert broker.metadata_calls == calls_after_connect + blip


def test_client_reported_outage_is_combined_with_a_delivery_timeout(
    broker: FakeBroker,
) -> None:
    instance, error_cb = connected_block(broker)
    take_broker_down(broker)

    error_cb(FakeClientError("2/2 brokers are down"))
    result = run(instance, fire_and_forget=False, timeout=0.5)

    assert result["error_status"] is True
    assert "Delivery not acknowledged within 0.5s" in result["message"]
    assert "2/2 brokers are down" in result["message"]


def test_client_reported_outage_is_combined_with_a_rejected_delivery(
    broker: FakeBroker,
) -> None:
    instance, error_cb = connected_block(broker)
    take_broker_down(broker, deliveries_hang=False)
    broker.delivery_error = "Local: Message timed out"

    error_cb(FakeClientError("2/2 brokers are down"))
    result = run(instance, fire_and_forget=False)

    # a failed delivery report is no proof of life
    assert result["error_status"] is True
    assert "Kafka rejected the message: Local: Message timed out" in result["message"]
    assert "2/2 brokers are down" in result["message"]


def test_confirmed_delivery_clears_the_client_reported_outage(
    broker: FakeBroker,
) -> None:
    instance, error_cb = connected_block(broker)
    take_broker_down(broker)
    error_cb(FakeClientError())
    assert run(instance, fire_and_forget=True)["error_status"] is True
    calls_after_failed_probe = broker.metadata_calls

    # metadata stays down for the rest of the test: only a delivery report can clear
    broker.hang_flush = False
    cleared = run(instance, fire_and_forget=False)
    after = run(instance, fire_and_forget=True)

    # the run that proves the connection healthy reports normally
    assert cleared["error_status"] is False
    assert cleared["message"].startswith("Message delivered to partition 0")
    assert after == {"error_status": False, "message": "Message scheduled for delivery"}
    assert broker.metadata_calls == calls_after_failed_probe


def test_background_delivery_clears_the_client_reported_outage(
    broker: FakeBroker,
) -> None:
    instance, error_cb = connected_block(broker)
    take_broker_down(broker)
    error_cb(FakeClientError())
    assert run(instance, fire_and_forget=True)["error_status"] is True
    calls_after_failed_probe = broker.metadata_calls

    # records flow again and the next poll() serves the queued records' delivery
    # reports; metadata stays down, so only those reports can clear
    broker.hang_flush = False
    cleared = run(instance, fire_and_forget=True)

    assert cleared == {
        "error_status": False,
        "message": "Message scheduled for delivery",
    }
    assert broker.metadata_calls == calls_after_failed_probe


def test_failed_background_delivery_does_not_clear_the_client_reported_outage(
    broker: FakeBroker,
) -> None:
    instance, error_cb = connected_block(broker)
    take_broker_down(broker, deliveries_hang=False)
    broker.delivery_error = "Local: Message timed out"

    error_cb(FakeClientError("2/2 brokers are down"))
    result = run(instance, fire_and_forget=True)

    assert result["error_status"] is True
    assert "2/2 brokers are down" in result["message"]


def test_successful_probe_clears_the_client_reported_outage(
    broker: FakeBroker,
) -> None:
    instance, error_cb = connected_block(broker)
    take_broker_down(broker)
    error_cb(FakeClientError())
    assert run(instance, fire_and_forget=True)["error_status"] is True
    # still down and inside the probe interval: claimed again, without a probe
    calls_before_probe = broker.metadata_calls
    assert run(instance, fire_and_forget=True)["error_status"] is True
    assert broker.metadata_calls == calls_before_probe

    # metadata answers again, records still get no delivery report: only the probe
    # can clear
    broker.fail_connect = None
    with patch.object(kafka_common, "CLIENT_RECOVERY_PROBE_INTERVAL", 0.0):
        cleared = run(instance, fire_and_forget=True)
        after = run(instance, fire_and_forget=True)

    assert cleared == {
        "error_status": False,
        "message": "Message scheduled for delivery",
    }
    assert after["error_status"] is False
    # exactly one probe: once cleared, nothing is probed any more
    assert broker.metadata_calls == calls_before_probe + 1


@pytest.mark.parametrize("failure", ["raises", "topic_error"])
def test_failing_probe_keeps_the_client_reported_outage(
    broker: FakeBroker, failure: str
) -> None:
    instance, error_cb = connected_block(broker)
    broker.hang_flush = True
    error_cb(FakeClientError("2/2 brokers are down"))
    calls_before_probe = broker.metadata_calls
    if failure == "raises":
        take_broker_down(broker)
    else:
        broker.creating_topic_replies = 2

    # the immediate first probe and, with the interval elapsed, a second one
    first = run(instance, fire_and_forget=True)
    with patch.object(kafka_common, "CLIENT_RECOVERY_PROBE_INTERVAL", 0.0):
        result = run(instance, fire_and_forget=True)

    assert first["error_status"] is True
    assert broker.metadata_calls == calls_before_probe + 2
    assert result["error_status"] is True
    # the message still describes the recorded problem, not the probe's failure
    assert "2/2 brokers are down" in result["message"]
    assert "Broker transport failure" not in result["message"]


def test_fatal_client_error_is_never_probed_and_never_cleared(
    broker: FakeBroker,
) -> None:
    instance, error_cb = connected_block(broker)
    calls_after_connect = broker.metadata_calls

    error_cb(FakeClientError("Fatal error: fenced", name="_FATAL", fatal=True))
    # a later non-fatal error must not downgrade it
    error_cb(FakeClientError("2/2 brokers are down"))
    with patch.object(kafka_common, "CLIENT_RECOVERY_PROBE_INTERVAL", 0.0):
        # the fake keeps delivering: not even a delivery report clears a fatal error
        results = [
            run(instance, fire_and_forget=fire_and_forget)
            for fire_and_forget in (True, False, True)
        ]

    for result in results:
        assert result["error_status"] is True
        assert "fatal error" in result["message"]
        assert "Fatal error: fenced" in result["message"]
        assert "Restart the block (pipeline)" in result["message"]
    assert broker.metadata_calls == calls_after_connect


def test_close_resets_the_client_reported_outage(broker: FakeBroker) -> None:
    instance, error_cb = connected_block(broker)
    error_cb(FakeClientError("Fatal error: fenced", fatal=True))
    assert run(instance, fire_and_forget=True)["error_status"] is True

    instance.close()
    result = run(instance, fire_and_forget=True)

    # a rebuilt client starts clean and gets the same tracker as its error_cb
    assert len(broker.producers) == 2
    assert result["error_status"] is False
    assert broker.producers[1].config["error_cb"] is error_cb


def test_error_callback_never_raises(broker: FakeBroker) -> None:
    instance, error_cb = connected_block(broker)
    take_broker_down(broker)

    class Unprintable:
        def __str__(self) -> str:
            raise RuntimeError("no text")

    error_cb(object())
    error_cb(None)
    error_cb(Unprintable())
    result = run(instance, fire_and_forget=True)

    assert result["error_status"] is True
    assert "reported a broker problem" in result["message"]
