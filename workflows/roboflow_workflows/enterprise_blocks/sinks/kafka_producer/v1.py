import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, field_validator
from roboflow_workflows.core_steps.sinks.noop import disabled_sink_response
from roboflow_workflows.enterprise_blocks.sinks.kafka_common import (
    LIBRDKAFKA_LOG_LEVEL,
    PROVIDER_AWS_MSK,
    PROVIDER_SELF_HOSTED,
    ClientErrorTracker,
    ConfigurationError,
    build_connection_config,
    coerce_timeout,
    combine_messages,
    describe_error,
    is_selector,
    pop_auth_failure_message,
    preflight_token,
    resolve_bootstrap_servers,
    wait_for_topic_metadata,
)
from roboflow_workflows.environment import GCP_SERVERLESS, LAMBDA
from roboflow_workflows.execution_engine.entities.base import OutputDefinition
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from typing_extensions import Annotated

try:
    import confluent_kafka
except ImportError:  # pragma: no cover - exercised only on images without the wheel
    confluent_kafka = None

logger = logging.getLogger(__name__)
BOOLEAN_ADAPTER = TypeAdapter(bool)

SELF_HOSTED_RELEVANT = {
    "provider": {"values": [PROVIDER_SELF_HOSTED], "required": False}
}
AWS_MSK_RELEVANT = {"provider": {"values": [PROVIDER_AWS_MSK], "required": False}}
ACKS_VALUES = ("0", "1", "all")

LONG_DESCRIPTION = """
The **Kafka Producer** block publishes one message to an Apache Kafka topic on every
workflow run. Upstream steps decide what the message contains — a JSON dictionary or a
plain string — and the block sends it as-is. This is the write side of Kafka; the **Kafka
Consumer** block is the read side, and both share the same connection settings.

## What is published

- `message`: a dictionary is JSON-encoded, a string is sent as UTF-8 text.
- `key` (optional): Kafka keeps all records with the same key in order on one partition.
  Set it to the camera id so every frame from one stream stays ordered, and so downstream
  consumers can read one camera's records.
- `headers` (optional): string-to-string metadata attached to the record, for example the
  source system or a schema version.

The block never creates topics or chooses partitions; the topic exists already and the
broker assigns the partition from the key.

## Delivery

- `fire_and_forget=true` (default): the record is handed to the client and the run returns
  at once with `error_status=false`. Best for live video. The client keeps retrying
  delivery in the background for up to five minutes, as any Kafka producer does; a record
  that still cannot be delivered is logged, not returned.
- `fire_and_forget=false`: the run waits (up to `timeout`) for the broker's acknowledgement
  and returns the real outcome. Use it when every message matters or when troubleshooting.
  A timeout is reported as delivery unknown: the record may still be delivered.

`acks` controls how many brokers must confirm a write: `all` (default) for durability, `1`
for the partition leader only, `0` for no confirmation.

## Connecting

Choose a `provider`:

- **Self-hosted** — any Kafka cluster you operate. Leave `username` and `password` empty
  for an unauthenticated plaintext listener, or supply both for SASL/SCRAM-SHA-512 over
  TLS. Amazon MSK clusters configured for SASL/SCRAM also use this provider.
- **AWS MSK** — Amazon MSK with IAM access control (port 9098), the authentication mode
  AWS recommends and the only one available on MSK Serverless. No credentials are entered
  in the workflow: the block signs in with the AWS identity present on the machine running
  inference (an attached instance/task role, IAM Roles Anywhere, an assumed-role profile,
  or an SSO login), and that identity needs `kafka-cluster:Connect`, `DescribeCluster`,
  `DescribeTopic` and `WriteData` on the cluster and topic. Tokens are refreshed
  automatically, so a continuous stream never stops for an expired credential. The AWS
  region is derived from the broker address; set `aws_region` only if that fails.

One block instance publishes to a single connection: changing broker, provider,
credentials or `acks` between runs is rejected as a configuration error. Inside an
`InferencePipeline` the connection is opened once and reused for every frame; over the
HTTP API every request builds a fresh block instance and pays one connection.

## Outputs

- `error_status` (boolean): `true` if the run failed, or if confirmed delivery failed.
- `message` (string): what happened — scheduled, delivered, or the failure reason.

When the Kafka client reports a broker problem after connecting (for example all
brokers down), the block checks the broker at once. If the broker does not answer, the
problem is surfaced as `error_status=true` on every run, fire-and-forget included, until
the connection is proven healthy again; records are still queued for delivery meanwhile.

Failures are logged and returned in the outputs; the workflow keeps running and the next
run retries the connection. This block is not available on the Roboflow hosted platform.
Self-hosted servers enable enterprise blocks with `LOAD_ENTERPRISE_BLOCKS=True`.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Kafka Producer",
            "version": "v1",
            "short_description": "Publishes a message to an Apache Kafka topic.",
            "long_description": LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "sink",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-stream",
                "blockPriority": 11,
                "enterprise_only": True,
                "local_only": True,
            },
        }
    )
    type: Literal["roboflow_enterprise/kafka_producer_sink@v1"]

    # --- connection (identical to the Kafka Consumer block) ---
    bootstrap_servers: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Comma-separated list of Kafka broker addresses (`host:port`). "
        "For AWS MSK use the bootstrap string from the console. The server operator can "
        "restrict this value to an allowlist with "
        "`KAFKA_WORKFLOWS_SINKS_WHITELISTED_BOOTSTRAP_SERVERS`, or replace it with that "
        "list by setting "
        "`KAFKA_WORKFLOWS_SINKS_ALLOW_USER_PROVIDED_BOOTSTRAP_SERVERS=False`.",
        examples=[
            "localhost:9092",
            "b-1.cluster.abc123.c2.kafka.us-east-1.amazonaws.com:9098",
            "$inputs.kafka_bootstrap",
        ],
        json_schema_extra={"always_visible": True},
    )
    topic: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Topic to publish to. It must already exist; on brokers with topic "
        "auto-creation enabled a misspelled name silently creates a new topic.",
        examples=["vision.detections", "$inputs.kafka_topic"],
        json_schema_extra={"always_visible": True},
    )
    provider: Literal["Self-hosted", "AWS MSK"] = Field(
        default=PROVIDER_SELF_HOSTED,
        description="Where the cluster lives. **Self-hosted**: any Kafka you operate, "
        "optionally with username/password (SASL/SCRAM-SHA-512 over TLS). **AWS MSK**: "
        "Amazon MSK with IAM access control, signed in with the AWS identity on the "
        "machine running inference; no credentials in the workflow.",
        examples=[PROVIDER_SELF_HOSTED, PROVIDER_AWS_MSK],
        json_schema_extra={"always_visible": True},
    )
    username: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="SASL username. Leave empty for an unauthenticated plaintext listener. "
        "Must be set together with `password`.",
        examples=["inference", "$inputs.kafka_username"],
        json_schema_extra={"relevant_for": SELF_HOSTED_RELEVANT},
    )
    password: Optional[Union[Selector(kind=[SECRET_KIND]), str]] = Field(
        default=None,
        description="SASL password. Provide it as a secret input; it is never stored in "
        "the workflow definition. Must be set together with `username`.",
        examples=["$inputs.kafka_password"],
        json_schema_extra={"private": True, "relevant_for": SELF_HOSTED_RELEVANT},
    )
    aws_region: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="AWS region of the MSK cluster, used to sign the IAM token. Leave "
        "empty to derive it from the broker address "
        "(`*.kafka.<region>.amazonaws.com`).",
        examples=["us-east-1", "$inputs.aws_region"],
        json_schema_extra={"relevant_for": AWS_MSK_RELEVANT},
    )

    # --- what to publish ---
    message: Union[
        Selector(kind=[STRING_KIND, DICTIONARY_KIND]), str, Dict[str, Any]
    ] = Field(
        description="Record value. A dictionary is JSON-encoded; a string is sent as "
        "UTF-8 text. Build the payload with upstream steps.",
        examples=["$steps.payload.output", '{"state": "RUNNING"}'],
        json_schema_extra={"always_visible": True},
    )
    key: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Record key. Records with the same key stay in order on one "
        "partition; use the camera id so each stream's frames stay ordered.",
        examples=["cam-1", "$inputs.camera_id"],
    )
    headers: Optional[Union[Selector(kind=[DICTIONARY_KIND]), Dict[str, str]]] = Field(
        default=None,
        description="Optional string-to-string record headers, for example the source "
        "system or a schema version.",
        examples=[{"source": "inference", "schema": "v1"}, "$inputs.headers"],
    )
    fire_and_forget: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(
        default=True,
        description="`true`: hand the record to the client and return immediately; "
        "delivery failures are only logged. `false`: wait up to `timeout` for the "
        "broker's acknowledgement and return the real outcome.",
        examples=[True, "$inputs.wait_for_ack"],
    )

    # --- advanced ---
    acks: Union[Selector(kind=[STRING_KIND]), Literal["0", "1", "all"]] = Field(
        default="all",
        description="Brokers that must confirm a write: `all` for durability (default), "
        "`1` for the partition leader only, `0` for no confirmation.",
        examples=["all", "1"],
        json_schema_extra={"additional_section": True},
    )
    ssl_ca_location: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Path to a CA certificate bundle on the machine running inference, "
        "for TLS connections whose issuer is not in the system trust store. Requires "
        "local file system access for Workflow blocks "
        "(`ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=True`); otherwise the run "
        "reports an error.",
        examples=["/etc/ssl/certs/ca-certificates.crt"],
        json_schema_extra={"additional_section": True},
    )
    timeout: Union[
        Annotated[float, Field(gt=0, allow_inf_nan=False)],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        default=5.0,
        description="Seconds allowed for connecting and locating the topic on the first "
        "run, and for waiting on the broker's acknowledgement when "
        "`fire_and_forget=false`. Records themselves are retried by the client for up "
        "to five minutes, as with any Kafka producer.",
        examples=[5.0],
        json_schema_extra={"additional_section": True},
    )

    @field_validator("timeout", mode="before")
    @classmethod
    def validate_timeout(cls, value: Any, info: Any) -> Any:
        if is_selector(value):
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{info.field_name} must be a number of seconds")
        return value

    @field_validator("acks", mode="before")
    @classmethod
    def validate_acks(cls, value: Any) -> Any:
        if is_selector(value):
            return value
        # accept the integer spellings JSON editors tend to produce
        if isinstance(value, bool):
            raise ValueError("acks must be one of 0, 1, all")
        if isinstance(value, int):
            value = str(value)
        if value not in ACKS_VALUES:
            raise ValueError("acks must be one of 0, 1, all")
        return value

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            RuntimeRestriction(
                severity=Severity.HARD,
                note=(
                    "On the Roboflow hosted platform every run returns "
                    "`error_status=true` and publishes nothing; no Kafka connection is "
                    "opened from the hosted platform."
                ),
                applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
            ),
            RuntimeRestriction(
                severity=Severity.SOFT,
                note=(
                    "Use fire_and_forget=false to observe delivery failures and avoid "
                    "unbounded buffering when the broker is slower than the stream."
                ),
                applies_to_runtimes=[Runtime.INFERENCE_PIPELINE],
            ),
        ]


def _failure(message: str) -> BlockResult:
    logger.error("Kafka Producer failure: %s", message)
    return {"error_status": True, "message": message}


def _encode_text(text: str, what: str) -> bytes:
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        # lone UTF-16 surrogates (e.g. a malformed "\ud800" in a JSON body) are not
        # characters and have no UTF-8 form; report rather than alter the data
        raise ConfigurationError(
            f"{what} contains characters that cannot be encoded as UTF-8."
        )


def _encode_value(message: Any) -> bytes:
    if isinstance(message, (bytes, bytearray)):
        return bytes(message)
    if isinstance(message, str):
        return _encode_text(message, "message")
    if isinstance(message, (dict, list)):
        try:
            return json.dumps(message).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ConfigurationError(
                f"message could not be JSON-encoded: {describe_error(error)}"
            )
    raise ConfigurationError(
        f"message must be a string or a dictionary, got {type(message).__name__}."
    )


def _encode_key(key: Any) -> Optional[bytes]:
    if key is None:
        return None
    if isinstance(key, (bytes, bytearray)):
        return bytes(key)
    return _encode_text(str(key), "key")


def _encode_headers(headers: Any) -> Optional[List[Tuple[str, bytes]]]:
    if headers is None:
        return None
    if not isinstance(headers, dict):
        raise ConfigurationError(
            f"headers must be a dictionary of strings, got {type(headers).__name__}."
        )
    encoded = []
    for name, value in headers.items():
        if value is None:
            encoded.append((str(name), None))
        elif isinstance(value, (bytes, bytearray)):
            encoded.append((str(name), bytes(value)))
        else:
            encoded.append((str(name), _encode_text(str(value), f"header {name!r}")))
    return encoded


def _coerce_acks(value: Any) -> str:
    if isinstance(value, bool):
        raise ConfigurationError("acks must be one of 0, 1, all.")
    if isinstance(value, (int, float)) and float(value).is_integer():
        value = str(int(value))
    value = str(value).strip()
    if value not in ACKS_VALUES:
        raise ConfigurationError("acks must be one of 0, 1, all.")
    return value


def _is_invalid_configuration(error: BaseException) -> bool:
    # librdkafka reports bad config values as KafkaException(_INVALID_ARG) from the
    # constructor; those are the user's settings, not the broker being unreachable
    code = getattr(getattr(error, "args", [None])[0], "code", None)
    return callable(code) and code() == getattr(
        confluent_kafka.KafkaError, "_INVALID_ARG", object()
    )


class _DeliveryReport:
    """Collects the delivery callback for the record published in this run."""

    def __init__(self, client_errors: Optional[ClientErrorTracker] = None) -> None:
        self.error: Optional[Any] = None
        self.partition: Optional[int] = None
        self.offset: Optional[int] = None
        self.delivered = False
        self._client_errors = client_errors

    def __call__(self, error: Any, message: Any) -> None:
        self.delivered = True
        self.error = error
        if error is None and self._client_errors is not None:
            # the broker took a record: proof of life
            self._client_errors.clear()
        if error is None and message is not None:
            self.partition = message.partition()
            self.offset = message.offset()


def _log_background_delivery(error: Any, message: Any) -> None:
    # fire-and-forget records report here on a later poll(); nothing to return to
    if error is not None:
        logger.error(
            "Kafka Producer: delivery to topic %r failed: %s",
            message.topic() if message is not None else "?",
            error,
        )


def _background_delivery_callback(
    client_errors: ClientErrorTracker,
) -> Callable[[Any, Any], None]:
    # closes over the tracker only, not the block: librdkafka holds the callback for
    # every queued record, and the block must stay collectable meanwhile
    def on_delivery(error: Any, message: Any) -> None:
        _log_background_delivery(error, message)
        if error is None:
            # the broker took a record: proof of life
            client_errors.clear()

    return on_delivery


class KafkaProducerSinkBlockV1(WorkflowBlock):
    def __init__(
        self,
        background_tasks: Optional[BackgroundTasks] = None,
        thread_pool_executor: Optional[ThreadPoolExecutor] = None,
        disable_sinks: bool = False,
        allow_access_to_file_system: bool = False,
    ):
        # gates `ssl_ca_location`, a server-side path chosen by the workflow; False by
        # default so a hand-constructed block is safe
        self._allow_access_to_file_system = allow_access_to_file_system
        # `error_cb` of the client: broker problems librdkafka reports out of band
        self._client_errors = ClientErrorTracker("Kafka Producer")
        self._on_background_delivery = _background_delivery_callback(
            self._client_errors
        )
        self._bootstrap_override_logged = False
        # librdkafka delivers asynchronously on its own network thread, so the shared
        # executor / background tasks are accepted for parity with other sinks but unused
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._disable_sinks = disable_sinks
        self._producer: Optional[Any] = None
        self._connection_identity: Optional[Tuple] = None
        self._auth_failures: List[BaseException] = []
        self._last_timeout: float = 5.0
        self._lifecycle_lock = threading.Lock()

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return [
            "background_tasks",
            "thread_pool_executor",
            "disable_sinks",
            "allow_access_to_file_system",
        ]

    def close(self) -> None:
        with self._lifecycle_lock:
            producer = self._producer
            self._producer = None
            self._connection_identity = None
            if producer is None:
                return
            try:
                # give queued records a bounded chance to leave before dropping the
                # client; on the HTTP path this runs when the request's instance is
                # released and may hold that thread for up to the last `timeout`, which
                # is preferred over silently losing fire-and-forget records
                producer.flush(self._last_timeout)
            except Exception as error:
                logger.error("Failed to flush Kafka producer on close: %s", error)
            finally:
                # after the flush, whose callbacks still belong to the old client
                self._client_errors.reset()

    def __del__(self):
        try:
            self.close()
        except Exception as error:  # pragma: no cover - GC fallback
            logger.error("Failed to close Kafka producer: %s", error)

    def run(
        self,
        bootstrap_servers: str,
        topic: str,
        message: Any,
        provider: str = PROVIDER_SELF_HOSTED,
        username: Optional[str] = None,
        password: Optional[str] = None,
        aws_region: Optional[str] = None,
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        fire_and_forget: bool = True,
        acks: str = "all",
        ssl_ca_location: Optional[str] = None,
        timeout: float = 5.0,
    ) -> BlockResult:
        if self._disable_sinks:
            return disabled_sink_response()
        if GCP_SERVERLESS or LAMBDA:
            return _failure(
                "Kafka Producer is not available on the Roboflow hosted platform."
            )
        if confluent_kafka is None:
            return _failure(
                "Kafka Producer requires the confluent-kafka package in the runtime."
            )
        try:
            timeout = coerce_timeout(timeout, "timeout", allow_zero=False)
            acks = _coerce_acks(acks)
            try:
                fire_and_forget = BOOLEAN_ADAPTER.validate_python(fire_and_forget)
            except ValidationError:
                raise ConfigurationError("fire_and_forget must be a boolean.")
            bootstrap_servers = str(bootstrap_servers).strip()
            topic = str(topic).strip()
            if not bootstrap_servers or not topic:
                raise ConfigurationError("bootstrap_servers and topic must be set.")
            # operator policy; from here on this is where the client connects
            bootstrap_servers = resolve_bootstrap_servers(
                bootstrap_servers, log_override=not self._bootstrap_override_logged
            )
            self._bootstrap_override_logged = True
            auth_failures: List[BaseException] = []
            username = username if username not in (None, "") else None
            password = password if password not in (None, "") else None
            connection_config = build_connection_config(
                provider=provider,
                bootstrap_servers=bootstrap_servers,
                username=username,
                password=password,
                aws_region=aws_region,
                ssl_ca_location=ssl_ca_location,
                auth_failures=auth_failures,
                allow_access_to_file_system=self._allow_access_to_file_system,
            )
            value = _encode_value(message)
            encoded_key = _encode_key(key)
            encoded_headers = _encode_headers(headers)
        except ConfigurationError as error:
            return _failure(str(error))
        identity = (
            bootstrap_servers,
            topic,
            provider,
            username,
            password,
            aws_region,
            ssl_ca_location,
            acks,
        )
        with self._lifecycle_lock:
            self._last_timeout = timeout
            try:
                self._ensure_producer(
                    identity=identity,
                    bootstrap_servers=bootstrap_servers,
                    topic=topic,
                    connection_config=connection_config,
                    auth_failures=auth_failures,
                    acks=acks,
                    timeout=timeout,
                )
            except ConfigurationError as error:
                return _failure(str(error))
            except Exception as error:
                if _is_invalid_configuration(error):
                    return _failure(
                        f"Invalid Kafka client configuration: {describe_error(error)}"
                    )
                return _failure(
                    f"Kafka broker not reachable ({describe_error(error)}). Raise "
                    "'timeout' if the broker needs longer to connect."
                )
            result = self._publish(
                topic=topic,
                value=value,
                key=encoded_key,
                headers=encoded_headers,
                fire_and_forget=fire_and_forget,
                timeout=timeout,
            )
            return self._report_client_problem(result, topic, fire_and_forget)

    # --- lifecycle -----------------------------------------------------------------

    def _ensure_producer(
        self,
        identity: Tuple,
        bootstrap_servers: str,
        topic: str,
        connection_config: Dict[str, Any],
        auth_failures: List[BaseException],
        acks: str,
        timeout: float,
    ) -> None:
        if self._producer is not None:
            if identity != self._connection_identity:
                raise ConfigurationError(
                    "Kafka connection parameters (bootstrap servers, topic, provider, "
                    "credentials or acks) changed between runs; this block publishes only "
                    "to the connection configured on its first run."
                )
            return
        self._auth_failures = auth_failures
        deadline = time.monotonic() + timeout
        try:
            preflight_token(connection_config)
        except Exception:
            self._raise_on_auth_failure()
            raise
        # a new client starts clean: problems of a previous one are not its own
        self._client_errors.reset()
        config = {
            "bootstrap.servers": bootstrap_servers,
            "acks": acks,
            # global client errors (all brokers down, transport failures) arrive only
            # here, served from inside poll() / flush(); produce() keeps succeeding
            "error_cb": self._client_errors,
            # message.timeout.ms is left at librdkafka's default (300s): records survive
            # broker hiccups like any Kafka producer's; `timeout` bounds only connecting
            # and, in confirmed mode, waiting for the acknowledgement
            "log_level": LIBRDKAFKA_LOG_LEVEL,
            **connection_config,
        }
        producer = confluent_kafka.Producer(config)
        try:
            # fail fast on an unreachable broker or a missing topic, but ride out the
            # transient "unknown topic" replies a broker gives while auto-creating one
            wait_for_topic_metadata(producer, topic, deadline)
            self._raise_on_auth_failure()
        except Exception:
            self._raise_on_auth_failure()
            raise
        self._producer = producer
        self._connection_identity = identity

    def _raise_on_auth_failure(self) -> None:
        message = pop_auth_failure_message(self._auth_failures)
        if message is not None:
            raise ConfigurationError(message)

    def _report_client_problem(
        self, result: BlockResult, topic: str, fire_and_forget: bool
    ) -> BlockResult:
        """Add the broker problem the client reported, if one is still recorded, to this
        run's result. Successful delivery reports have already cleared it by now. Runs
        under the lifecycle lock, like publishing."""
        problem = self._client_errors.check(client=self._producer, topic=topic)
        if problem is None:
            return result
        if result["error_status"] or problem.fatal:
            note = None
        elif fire_and_forget:
            note = (
                "The record is queued in the client, which retries delivery for up to "
                "five minutes, but the broker is currently unreachable."
            )
        else:
            note = "This record was acknowledged before the problem was reported."
        # a success text ("scheduled", "delivered") is replaced, a failure text is kept;
        # the tracker logged the problem once, so nothing is logged per frame here
        message = combine_messages(
            result["message"] if result["error_status"] else None,
            problem.message(note),
        )
        return {"error_status": True, "message": message}

    # --- publish -------------------------------------------------------------------

    def _publish(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes],
        headers: Optional[List[Tuple[str, bytes]]],
        fire_and_forget: bool,
        timeout: float,
    ) -> BlockResult:
        producer = self._producer
        report = _DeliveryReport(self._client_errors)
        on_delivery = report if not fire_and_forget else self._on_background_delivery
        try:
            try:
                producer.produce(
                    topic,
                    value=value,
                    key=key,
                    headers=headers,
                    on_delivery=on_delivery,
                )
            except BufferError:
                # a record holds its queue slot until its delivery report has been served,
                # so poll first (the librdkafka idiom) and retry once before giving up
                producer.poll(0)
                producer.produce(
                    topic,
                    value=value,
                    key=key,
                    headers=headers,
                    on_delivery=on_delivery,
                )
        except BufferError:
            return _failure(
                "Kafka producer's local queue is full; the broker is slower than the "
                "stream. Lower the frame rate or use fire_and_forget=false."
            )
        except Exception as error:
            return _failure(f"Failed to publish message: {describe_error(error)}")
        try:
            # services delivery reports of earlier records (the OAuth token refresh
            # runs on librdkafka's own thread and needs no poll)
            producer.poll(0)
            if fire_and_forget:
                auth_message = pop_auth_failure_message(self._auth_failures)
                if auth_message is not None:
                    return _failure(auth_message)
                return {
                    "error_status": False,
                    "message": "Message scheduled for delivery",
                }
            unflushed = producer.flush(timeout)
        except Exception as error:
            auth_message = pop_auth_failure_message(self._auth_failures)
            if auth_message is not None:
                return _failure(auth_message)
            return _failure(f"Failed to publish message: {describe_error(error)}")
        auth_message = pop_auth_failure_message(self._auth_failures)
        if report.error is not None:
            rejected = f"Kafka rejected the message: {report.error}"
            if auth_message is not None:
                rejected = f"{rejected}; {auth_message}"
            return _failure(rejected)
        if report.delivered:
            if report.offset is None:
                # acks=0: the broker sends no acknowledgement, so no offset is known
                delivered = (
                    f"Message sent to partition {report.partition} without "
                    "acknowledgement (acks=0)"
                )
            else:
                delivered = (
                    f"Message delivered to partition {report.partition} at offset "
                    f"{report.offset}"
                )
            if auth_message is not None:
                # the record went out, but a token refresh failed meanwhile: surface it
                # once so the operator sees it before the next refresh is due
                return _failure(f"{delivered}; {auth_message}")
            return {"error_status": False, "message": delivered}
        if auth_message is not None:
            return _failure(auth_message)
        return _failure(
            f"Delivery not acknowledged within {timeout}s ({unflushed} record(s) still "
            "queued); delivery status unknown and the message may still be delivered."
        )
