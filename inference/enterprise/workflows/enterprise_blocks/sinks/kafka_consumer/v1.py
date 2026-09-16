import json
import logging
import threading
import time
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator
from typing_extensions import Annotated

from inference.core.env import GCP_SERVERLESS, LAMBDA
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    Runtime,
    RuntimeInputMode,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.kafka_common import (
    GROUP_ID_PREFIX,
    LIBRDKAFKA_LOG_LEVEL,
    PROVIDER_AWS_MSK,
    PROVIDER_SELF_HOSTED,
    ConfigurationError,
    build_connection_config,
    coerce_non_negative_int,
    coerce_timeout,
    describe_error,
    is_selector,
    pop_auth_failure_message,
    preflight_token,
    time_remaining,
)

try:
    import confluent_kafka
except ImportError:  # pragma: no cover - exercised only on images without the wheel
    confluent_kafka = None

logger = logging.getLogger(__name__)

MAX_MESSAGES_PER_RUN = 1000
MODE_POINTER = "pointer"
MODE_LATEST = "latest"
MODE_SEQUENTIAL = "sequential"
READ_MODE_LATEST = "latest"
READ_MODE_SEQUENTIAL = "sequential"
READ_MODES = (READ_MODE_LATEST, READ_MODE_SEQUENTIAL)
SELF_HOSTED_RELEVANT = {
    "provider": {"values": [PROVIDER_SELF_HOSTED], "required": False}
}
AWS_MSK_RELEVANT = {"provider": {"values": [PROVIDER_AWS_MSK], "required": False}}

LONG_DESCRIPTION = """
The **Kafka Consumer** block reads one message from an Apache Kafka topic on every
workflow run and exposes it to the rest of the workflow as a raw string (`value`) and,
when the message is a JSON object, a parsed dictionary (`payload`). The block does not
interpret the message: downstream steps decide what to do with it, for example a
`continue_if` step that gates the model on a state message, or a sink that forwards it.

## Which message is read

`read_mode` selects how the block moves through the topic; `offset` / `partition` optionally
name a specific record.

- **`latest`** (default) — the newest message wins. On the first run the block seeks to the
  newest record already on the topic, so a freshly started pipeline learns the current
  state without waiting for a new publish. Afterwards each run reads whatever arrived since
  the previous run and keeps only the last message; if nothing arrived, the previously
  returned message is repeated with `is_new=false`. With `key_filter` set, only messages
  with that exact key are considered, including on the first run, which then looks back
  through a bounded window of recent records for that key. With `offset` set, the block
  reads exactly that record regardless of key; supplying the same pointer again returns
  the cached record without touching the broker, and the next run without a pointer
  returns the newest record on the topic again.
- **`sequential`** — every message is delivered, one per run, in the order the broker
  serves them, exactly like a plain Kafka consumer that reads one record at a time. The
  first run starts at the newest record already on the topic (or the newest record for
  `key_filter`); each later run returns the next unread record, or repeats the previous one
  with `is_new=false` when nothing new has arrived. A backlog is worked through one record
  per run. With `offset` set, reading starts at that record and continues from there; the
  same pointer supplied on every run (a literal in the definition) does not reset the
  position, a different pointer does. Positions live in the block instance: an
  `InferencePipeline` keeps them for its whole lifetime, a restart begins again at the
  newest record, and over the HTTP API every request is a fresh instance, so each request
  returns the newest record.

In both modes an offset that is beyond the end of the partition or has been removed by
compaction is reported as an error and the previous outputs are retained.

Ordering follows Kafka's guarantee: records are ordered within a partition, not across
partitions. On a multi-partition topic "newest" means the record the broker delivered last;
to keep one camera's messages strictly ordered, publish them with a key or use a
single-partition topic.

The block never buffers, never commits offsets and never shares partitions with other
consumers: every block instance sees every message, so one pipeline per camera all reading
the same topic each receive the same state. After a restart it reads the newest record
again (or the pointer it is given), never from a stored position.

## Connecting

Choose a `provider`:

- **Self-hosted** — any Kafka cluster you operate. Leave `username` and `password` empty
  for an unauthenticated plaintext listener, or supply both for SASL/SCRAM-SHA-512 over
  TLS. Amazon MSK clusters configured for SASL/SCRAM also use this provider.
- **AWS MSK** — Amazon MSK with IAM access control (port 9098), the authentication mode
  AWS recommends and the only one available on MSK Serverless. No credentials are entered
  in the workflow: the block signs in with the AWS identity present on the machine running
  inference (an attached instance/task role, IAM Roles Anywhere, an assumed-role profile,
  or an SSO login), and that identity needs `kafka-cluster:Connect`, `DescribeTopic`,
  `ReadData`, `DescribeGroup` and `AlterGroup` on the cluster. Tokens are refreshed
  automatically, so a continuous stream never stops for an expired credential. The AWS
  region is derived from the broker address; set `aws_region` only if that fails.

Over the HTTP API every request builds a fresh block instance, so each request pays one
broker connection and `is_new` is always `true`. Inside an `InferencePipeline` the
connection is opened once and reused for every frame.

## Outputs

- `value` (string): raw message value decoded as UTF-8; `null` before any message and
  for a tombstone record (a null value, which deletes a key on a compacted topic).
- `payload` (dictionary): the parsed JSON object, or `{}` when the value is not a JSON
  object.
- `key` (string): the message key, or `null`.
- `is_new` (boolean): `true` only on the run that first returned this record.
- `offset`, `partition` (integer): where the returned record lives, or `null`.
- `error_status` (boolean) and `error_message` (string): broker, credential and input
  problems are reported here and logged; the workflow keeps running and the next run
  retries.

This block is not available on the Roboflow hosted platform. Self-hosted servers enable
enterprise blocks with `LOAD_ENTERPRISE_BLOCKS=True`.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Kafka Consumer",
            "version": "v1",
            "short_description": "Reads a message from an Apache Kafka topic on every run: "
            "the record at a given offset, or the newest one.",
            "long_description": LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-stream",
                "blockPriority": 12,
                "enterprise_only": True,
                "local_only": True,
            },
        }
    )
    type: Literal["roboflow_enterprise/kafka_consumer@v1"]

    # --- connection ---
    bootstrap_servers: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Comma-separated list of Kafka broker addresses (`host:port`). "
        "For AWS MSK use the bootstrap string from the console.",
        examples=[
            "localhost:9092",
            "b-1.cluster.abc123.c2.kafka.us-east-1.amazonaws.com:9098",
            "$inputs.kafka_bootstrap",
        ],
        json_schema_extra={"always_visible": True},
    )
    topic: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Topic to read from.",
        examples=["line1.state", "$inputs.kafka_topic"],
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

    # --- what to read ---
    read_mode: Literal["latest", "sequential"] = Field(
        default="latest",
        description="**latest**: each run returns the newest message, skipping anything "
        "older, for reading a current state. **sequential**: each run returns the next "
        "unread message in order, one per run, like a plain Kafka consumer.",
        examples=["latest", "sequential"],
        json_schema_extra={"always_visible": True},
    )
    key_filter: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Consider only messages whose key equals this value, for example a "
        "camera id when one topic carries state for several cameras. Not applied to a "
        "record read directly by `offset`.",
        examples=["cam-1", "$inputs.camera_id"],
    )
    offset: Optional[
        Union[Selector(kind=[INTEGER_KIND]), Annotated[int, Field(ge=0)]]
    ] = Field(
        default=None,
        description="Pointer to a specific record, as reported to the publisher when it "
        "produced the message. In `latest` mode the block reads exactly that record. In "
        "`sequential` mode reading starts at that record and continues from there. Leave "
        "empty to start from the newest message.",
        examples=[42, "$inputs.state_offset"],
    )
    partition: Union[Selector(kind=[INTEGER_KIND]), Annotated[int, Field(ge=0)]] = (
        Field(
            default=0,
            description="Partition the pointer refers to. Only used together with "
            "`offset`; single-partition topics use 0.",
            examples=[0, "$inputs.state_partition"],
        )
    )

    # --- advanced ---
    ssl_ca_location: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Path to a CA certificate bundle on the machine running inference, "
        "for TLS connections whose issuer is not in the system trust store.",
        examples=["/etc/ssl/certs/ca-certificates.crt"],
        json_schema_extra={"additional_section": True},
    )
    poll_timeout: Union[
        Annotated[float, Field(ge=0, allow_inf_nan=False)],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        default=0.01,
        description="Seconds to wait for a message on each run. The client prefetches "
        "in the background, so a small value is enough once connected; this bounds the "
        "latency the block adds to every frame.",
        examples=[0.01],
        json_schema_extra={"additional_section": True},
    )
    connect_timeout: Union[
        Annotated[float, Field(gt=0, allow_inf_nan=False)],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        default=5.0,
        description="Seconds allowed for connecting to the broker and locating the "
        "topic on the first run, and for fetching a pointed record.",
        examples=[5.0],
        json_schema_extra={"additional_section": True},
    )

    @field_validator("offset", "partition", mode="before")
    @classmethod
    def validate_non_negative_integer(cls, value: Any, info: Any) -> Any:
        if value is None or is_selector(value):
            return value
        if isinstance(value, bool) or type(value) is not int or value < 0:
            raise ValueError(f"{info.field_name} must be a non-negative integer")
        return value

    @field_validator("poll_timeout", "connect_timeout", mode="before")
    @classmethod
    def validate_timeout(cls, value: Any, info: Any) -> Any:
        if is_selector(value):
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{info.field_name} must be a number of seconds")
        return value

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="value", kind=[STRING_KIND]),
            OutputDefinition(name="payload", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="key", kind=[STRING_KIND]),
            OutputDefinition(name="is_new", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="offset", kind=[INTEGER_KIND]),
            OutputDefinition(name="partition", kind=[INTEGER_KIND]),
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="error_message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            RuntimeRestriction(
                severity=Severity.SOFT,
                note=(
                    "The consumer and the last-returned record live in this block "
                    "instance. Over HTTP every request builds a fresh instance, so each "
                    "request pays a broker connection and `is_new` is always True. "
                    "Results are still correct; a long-lived InferencePipeline avoids "
                    "the per-request connection."
                ),
                applies_to_runtimes=[
                    Runtime.SELF_HOSTED_CPU,
                    Runtime.SELF_HOSTED_GPU,
                    Runtime.DEDICATED_DEPLOYMENT,
                ],
                applies_to_input_modes=[RuntimeInputMode.IMAGE],
            ),
        ]


class _Record(NamedTuple):
    partition: int
    offset: int
    key: Optional[str]
    value: Optional[str]
    payload: Dict[str, Any]


def _outputs(
    record: Optional[_Record],
    is_new: bool = False,
    error_message: Optional[str] = None,
) -> BlockResult:
    return {
        "value": record.value if record else None,
        "payload": dict(record.payload) if record else {},
        "key": record.key if record else None,
        "is_new": is_new,
        "offset": record.offset if record else None,
        "partition": record.partition if record else None,
        "error_status": error_message is not None,
        "error_message": error_message,
    }


def _decode(message: Any) -> _Record:
    raw_value = message.value()
    # a null value is a Kafka tombstone (key deleted on a compacted topic): surface it
    # as None rather than an empty string so downstream can tell the two apart
    value: Optional[str] = (
        raw_value.decode("utf-8", errors="replace")
        if isinstance(raw_value, (bytes, bytearray))
        else (None if raw_value is None else str(raw_value))
    )
    raw_key = message.key()
    key = (
        raw_key.decode("utf-8", errors="replace")
        if isinstance(raw_key, (bytes, bytearray))
        else (None if raw_key is None else str(raw_key))
    )
    payload: Dict[str, Any] = {}
    if value is not None:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                payload = parsed
        except ValueError:
            pass
    return _Record(
        partition=message.partition(),
        offset=message.offset(),
        key=key,
        value=value,
        payload=payload,
    )


def _message_key_matches(message: Any, key_filter: Optional[str]) -> bool:
    if key_filter is None:
        return True
    raw_key = message.key()
    if raw_key is None:
        return False
    if isinstance(raw_key, (bytes, bytearray)):
        raw_key = raw_key.decode("utf-8", errors="replace")
    return str(raw_key) == str(key_filter)


class KafkaConsumerBlockV1(WorkflowBlock):
    def __init__(self):
        self._consumer: Optional[Any] = None
        self._connection_identity: Optional[Tuple] = None
        self._topic: Optional[str] = None
        self._partitions: Set[int] = set()
        self._last: Optional[_Record] = None
        # how the consumer is currently positioned: MODE_LATEST / MODE_POINTER /
        # MODE_SEQUENTIAL, or None when connected but not positioned yet
        self._mode: Optional[str] = None
        # pointer already applied as the sequential start position, if any
        self._sequence_pointer: Optional[Tuple[int, int]] = None
        self._auth_failures: List[BaseException] = []
        self._reposition_timeout: float = 5.0
        # offsets the first drain after an assignment must reach per partition
        # (the newest existing record, or the scan window when a key filter is set);
        # polling waits up to connect_timeout for them instead of poll_timeout
        self._catch_up_until: Dict[int, int] = {}
        self._lifecycle_lock = threading.Lock()

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    def close(self) -> None:
        with self._lifecycle_lock:
            consumer = self._consumer
            self._consumer = None
            self._connection_identity = None
            self._topic = None
            self._partitions = set()
            self._mode = None
            self._sequence_pointer = None
            if consumer is None:
                return
            try:
                consumer.close()
            except Exception as error:
                logger.error("Failed to close Kafka consumer: %s", error)

    def __del__(self):
        try:
            self.close()
        except Exception as error:  # pragma: no cover - GC fallback
            logger.error("Failed to close Kafka consumer: %s", error)

    def run(
        self,
        bootstrap_servers: str,
        topic: str,
        provider: str = PROVIDER_SELF_HOSTED,
        username: Optional[str] = None,
        password: Optional[str] = None,
        aws_region: Optional[str] = None,
        read_mode: str = READ_MODE_LATEST,
        key_filter: Optional[str] = None,
        offset: Optional[int] = None,
        partition: int = 0,
        ssl_ca_location: Optional[str] = None,
        poll_timeout: float = 0.01,
        connect_timeout: float = 5.0,
    ) -> BlockResult:
        if GCP_SERVERLESS or LAMBDA:
            return self._failure(
                "Kafka Consumer is not available on the Roboflow hosted platform."
            )
        if confluent_kafka is None:
            return self._failure(
                "Kafka Consumer requires the confluent-kafka package in the runtime."
            )
        try:
            poll_timeout = coerce_timeout(poll_timeout, "poll_timeout", allow_zero=True)
            connect_timeout = coerce_timeout(
                connect_timeout, "connect_timeout", allow_zero=False
            )
            if read_mode not in READ_MODES:
                raise ConfigurationError(
                    f"Unknown read_mode {read_mode!r}; expected one of "
                    f"{', '.join(READ_MODES)}."
                )
            pointer: Optional[Tuple[int, int]] = None
            if offset is not None:
                pointer = (
                    coerce_non_negative_int(partition, "partition"),
                    coerce_non_negative_int(offset, "offset"),
                )
            bootstrap_servers = str(bootstrap_servers).strip()
            topic = str(topic).strip()
            if not bootstrap_servers or not topic:
                raise ConfigurationError("bootstrap_servers and topic must be set.")
            auth_failures: List[BaseException] = []
            connection_config = build_connection_config(
                provider=provider,
                bootstrap_servers=bootstrap_servers,
                username=username,
                password=password,
                aws_region=aws_region,
                ssl_ca_location=ssl_ca_location,
                auth_failures=auth_failures,
            )
        except ConfigurationError as error:
            return self._failure(str(error))
        identity = (
            bootstrap_servers,
            topic,
            provider,
            username,
            password,
            aws_region,
            ssl_ca_location,
        )
        with self._lifecycle_lock:
            # a pointer positions the consumer itself, except the sequential start
            # pointer that was already applied on an earlier run
            applies_pointer = pointer is not None and not (
                read_mode == READ_MODE_SEQUENTIAL and pointer == self._sequence_pointer
            )
            try:
                self._ensure_consumer(
                    identity=identity,
                    bootstrap_servers=bootstrap_servers,
                    topic=topic,
                    connection_config=connection_config,
                    auth_failures=auth_failures,
                    connect_timeout=connect_timeout,
                    key_filter=key_filter,
                    skip_positioning=applies_pointer,
                )
            except ConfigurationError as error:
                return self._failure(str(error))
            except Exception as error:
                return self._failure(
                    f"Kafka broker not reachable ({describe_error(error)}). "
                    "Raise 'connect_timeout' if the broker needs longer to connect."
                )
            self._reposition_timeout = connect_timeout
            if read_mode == READ_MODE_SEQUENTIAL:
                if applies_pointer:
                    return self._read_at_pointer(
                        pointer, connect_timeout, continue_after=True
                    )
                return self._read_next(poll_timeout, key_filter)
            if pointer is not None:
                return self._read_at_pointer(pointer, connect_timeout)
            return self._drain_latest(poll_timeout, key_filter)

    # --- lifecycle -----------------------------------------------------------------

    def _ensure_consumer(
        self,
        identity: Tuple,
        bootstrap_servers: str,
        topic: str,
        connection_config: Dict[str, Any],
        auth_failures: List[BaseException],
        connect_timeout: float,
        key_filter: Optional[str],
        skip_positioning: bool = False,
    ) -> None:
        if self._consumer is not None:
            if identity != self._connection_identity:
                raise ConfigurationError(
                    "Kafka connection parameters (bootstrap servers, topic, provider or "
                    "credentials) changed between runs; this block reads only from the "
                    "connection configured on its first run."
                )
            return
        config = {
            "bootstrap.servers": bootstrap_servers,
            # unique group per instance so every pipeline sees every message;
            # assign() below means the group is never used for rebalancing
            "group.id": f"{GROUP_ID_PREFIX}{uuid4()}",
            "enable.auto.commit": False,
            "auto.offset.reset": "latest",
            "log_level": LIBRDKAFKA_LOG_LEVEL,
            **connection_config,
        }
        self._auth_failures = auth_failures
        try:
            preflight_token(connection_config)
        except Exception:
            self._raise_on_auth_failure()
            raise
        consumer = confluent_kafka.Consumer(config)
        # one budget for the whole first run: metadata plus every watermark lookup
        deadline = time.monotonic() + connect_timeout
        try:
            metadata = consumer.list_topics(
                topic, timeout=time_remaining(deadline, "fetching topic metadata")
            )
            self._raise_on_auth_failure()
            topic_metadata = metadata.topics.get(topic)
            if topic_metadata is None or topic_metadata.error is not None:
                detail = (
                    str(topic_metadata.error)
                    if topic_metadata is not None
                    else "not found"
                )
                raise ConfigurationError(
                    f"Kafka topic {topic!r} is not available ({detail})."
                )
            partitions = sorted(topic_metadata.partitions)
            if not partitions:
                raise ConfigurationError(f"Kafka topic {topic!r} has no partitions.")
            window = self._scan_window(key_filter, len(partitions))
            positions, targets = [], {}
            if not skip_positioning:
                for p in partitions:
                    position, target = self._start_position(
                        consumer,
                        topic,
                        p,
                        window,
                        time_remaining(
                            deadline, f"locating the newest record on partition {p}"
                        ),
                    )
                    positions.append(position)
                    if target is not None:
                        targets[p] = target
                consumer.assign(positions)
            self._catch_up_until = targets
        except Exception:
            try:
                consumer.close()
            except Exception:
                pass
            # a failing token callback surfaces as a generic transport error from
            # librdkafka; report the real cause instead
            self._raise_on_auth_failure()
            raise
        self._consumer = consumer
        self._connection_identity = identity
        self._topic = topic
        self._partitions = set(partitions)
        # positioned at the newest record; the read function claims the mode. None =
        # connected but not positioned yet (a pointer run assigns itself)
        self._mode = None if skip_positioning else MODE_LATEST
        self._sequence_pointer = None

    def _pop_auth_failure_message(self) -> Optional[str]:
        return pop_auth_failure_message(self._auth_failures)

    def _raise_on_auth_failure(self) -> None:
        message = self._pop_auth_failure_message()
        if message is not None:
            raise ConfigurationError(message)

    @staticmethod
    def _scan_window(key_filter: Optional[str], partitions: int) -> int:
        # without a key filter the newest record is the answer; with one, scan a
        # bounded window backwards so the newest record *for that key* is found
        if key_filter is None:
            return 1
        return max(1, MAX_MESSAGES_PER_RUN // max(1, partitions))

    @staticmethod
    def _start_position(
        consumer: Any, topic: str, partition: int, window: int, timeout: float
    ) -> Tuple[Any, Optional[int]]:
        """Position `window` records before the end, plus the newest offset to reach
        (None when the partition is empty)."""
        low, high = consumer.get_watermark_offsets(
            confluent_kafka.TopicPartition(topic, partition), timeout=timeout
        )
        # high is the offset the next record will get; high-1 is the newest record
        if high <= low:
            return confluent_kafka.TopicPartition(topic, partition, high), None
        start = max(low, high - window)
        return confluent_kafka.TopicPartition(topic, partition, start), high - 1

    @staticmethod
    def _next_record_position(
        consumer: Any, topic: str, partition: int, timeout: float
    ) -> Any:
        _, high = consumer.get_watermark_offsets(
            confluent_kafka.TopicPartition(topic, partition), timeout=timeout
        )
        return confluent_kafka.TopicPartition(topic, partition, high)

    def _failure(self, message: str) -> BlockResult:
        logger.error("Kafka Consumer failure: %s", message)
        return _outputs(self._last, is_new=False, error_message=message)

    # --- read paths -----------------------------------------------------------------

    def _read_at_pointer(
        self,
        pointer: Tuple[int, int],
        connect_timeout: float,
        continue_after: bool = False,
    ) -> BlockResult:
        partition, offset = pointer
        if (
            not continue_after
            and self._last is not None
            and (self._last.partition, self._last.offset) == pointer
        ):
            # records are immutable: re-reading the same pointer is redundant. The
            # consumer was not repositioned, so the mode stays whatever it was.
            self._catch_up_until = {}
            return _outputs(self._last, is_new=False)
        if partition not in self._partitions:
            return self._failure(
                f"Partition {partition} does not exist on topic {self._topic!r} "
                f"(partitions: {sorted(self._partitions)})."
            )
        consumer = self._consumer
        deadline = time.monotonic() + connect_timeout
        try:
            low, high = consumer.get_watermark_offsets(
                confluent_kafka.TopicPartition(self._topic, partition),
                timeout=time_remaining(deadline, "locating the pointed record"),
            )
            if offset >= high:
                return self._failure(
                    f"Offset {offset} is out of range for partition {partition}: the "
                    f"newest record is at offset {high - 1}."
                )
            if offset < low:
                return self._failure(
                    f"Offset {offset} on partition {partition} is no longer retained "
                    f"(earliest available offset is {low})."
                )
            # concrete offsets everywhere: the target partition at the pointer, the
            # others at their next record so nothing old is fetched
            consumer.assign(
                [confluent_kafka.TopicPartition(self._topic, partition, offset)]
                + [
                    self._next_record_position(
                        consumer,
                        self._topic,
                        other,
                        time_remaining(deadline, f"positioning partition {other}"),
                    )
                    for other in sorted(self._partitions)
                    if other != partition
                ]
            )
            self._mode = MODE_SEQUENTIAL if continue_after else MODE_POINTER
            self._sequence_pointer = pointer if continue_after else None
            self._catch_up_until = {}
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._failure(
                        f"Timed out fetching offset {offset} from partition {partition}."
                    )
                message = consumer.poll(min(remaining, 0.5))
                if message is None:
                    continue
                if message.error() is not None:
                    return self._failure(
                        f"Kafka error while fetching offset {offset}: {message.error()}"
                    )
                if message.partition() != partition:
                    continue
                if message.offset() < offset:
                    continue
                if message.offset() > offset:
                    return self._failure(
                        f"Offset {offset} on partition {partition} no longer exists "
                        f"(removed by compaction or deletion); the next record is at "
                        f"offset {message.offset()}."
                    )
                return self._accept(message)
        except ConfigurationError as error:
            return self._failure(str(error))
        except Exception as error:
            try:
                self._raise_on_auth_failure()
            except ConfigurationError as auth_error:
                return self._failure(str(auth_error))
            return self._failure(
                f"Failed to read offset {offset} from partition {partition}: "
                f"{describe_error(error)}"
            )

    def _accept(self, message: Any, error_message: Optional[str] = None) -> BlockResult:
        """Make `message` the current record; is_new is False if it is the same record."""
        record = _decode(message)
        is_new = self._last is None or (record.partition, record.offset) != (
            self._last.partition,
            self._last.offset,
        )
        self._last = record
        return _outputs(record, is_new=is_new, error_message=error_message)

    def _poll_one(
        self,
        consumer: Any,
        first_timeout: float,
        key_filter: Optional[str],
        catch_up_deadline: Optional[float],
    ) -> Tuple[Any, Optional[str]]:
        """Poll until one acceptable record arrives or the local queue is empty."""
        error_message: Optional[str] = None
        timeout = first_timeout
        for _ in range(MAX_MESSAGES_PER_RUN):
            if self._catch_up_until:
                remaining = catch_up_deadline - time.monotonic()
                if remaining <= 0:
                    self._catch_up_until = {}
                else:
                    timeout = max(timeout, min(remaining, 0.5))
            message = consumer.poll(timeout)
            timeout = 0
            if message is None:
                if self._catch_up_until:
                    continue
                return None, error_message
            target = self._catch_up_until.get(message.partition())
            if target is not None and message.offset() >= target:
                del self._catch_up_until[message.partition()]
            if message.error() is not None:
                error_message = f"Kafka error while polling: {message.error()}"
                logger.error("Kafka Consumer: %s", error_message)
                continue
            if not _message_key_matches(message, key_filter):
                continue
            return message, error_message
        return None, error_message

    def _read_next(self, poll_timeout: float, key_filter: Optional[str]) -> BlockResult:
        consumer = self._consumer
        try:
            if self._mode != MODE_SEQUENTIAL:
                # first sequential run, or arriving from another positioning: start at
                # the newest record (for the key, when filtered) like the first run
                if self._mode != MODE_LATEST or not self._catch_up_until:
                    self._reposition_at_newest(
                        consumer, self._scan_window(key_filter, len(self._partitions))
                    )
                self._mode = MODE_SEQUENTIAL
            self._raise_on_auth_failure()
            catch_up_deadline = (
                time.monotonic() + self._reposition_timeout
                if self._catch_up_until
                else None
            )
            if self._catch_up_until:
                # starting position: settle on the newest record among the seeds, then
                # continue one record per run from there
                message, error_message, _ = self._poll_until_drained(
                    consumer,
                    poll_timeout,
                    key_filter,
                    catch_up_deadline,
                    stop_when_caught_up=True,
                )
            else:
                message, error_message = self._poll_one(
                    consumer, poll_timeout, key_filter, catch_up_deadline
                )
            error_message = self._pop_auth_failure_message() or error_message
        except ConfigurationError as error:
            return self._failure(str(error))
        except Exception as error:
            try:
                self._raise_on_auth_failure()
            except ConfigurationError as auth_error:
                return self._failure(str(auth_error))
            return self._failure(f"Failed to poll Kafka: {describe_error(error)}")
        if message is not None:
            return self._accept(message, error_message)
        return _outputs(self._last, is_new=False, error_message=error_message)

    def _poll_until_drained(
        self,
        consumer: Any,
        first_timeout: float,
        key_filter: Optional[str],
        catch_up_deadline: Optional[float],
        stop_when_caught_up: bool = False,
    ) -> Tuple[Any, Optional[str], bool]:
        """Poll until the local queue is empty (drained=True) or the per-run cap is hit.
        With stop_when_caught_up, stop as soon as every catch-up target was reached.
        Returns the newest accepted message, any polling error, and whether it drained.
        """
        newest = None
        error_message: Optional[str] = None
        timeout = first_timeout
        for _ in range(MAX_MESSAGES_PER_RUN):
            if stop_when_caught_up and not self._catch_up_until:
                return newest, error_message, True
            if self._catch_up_until:
                # records known to exist have not all been fetched yet: keep waiting
                # in short slices until they arrive or the deadline passes
                remaining = catch_up_deadline - time.monotonic()
                if remaining <= 0:
                    self._catch_up_until = {}
                else:
                    timeout = max(timeout, min(remaining, 0.5))
            message = consumer.poll(timeout)
            timeout = 0
            if message is None:
                if self._catch_up_until:
                    continue
                return newest, error_message, True
            target = self._catch_up_until.get(message.partition())
            if target is not None and message.offset() >= target:
                del self._catch_up_until[message.partition()]
            if message.error() is not None:
                error_message = f"Kafka error while polling: {message.error()}"
                logger.error("Kafka Consumer: %s", error_message)
                continue
            if not _message_key_matches(message, key_filter):
                continue
            newest = message
        return newest, error_message, False

    def _reposition_at_newest(self, consumer: Any, window: int) -> None:
        deadline = time.monotonic() + self._reposition_timeout
        positions, targets = [], {}
        for p in sorted(self._partitions):
            position, target = self._start_position(
                consumer,
                self._topic,
                p,
                window,
                time_remaining(
                    deadline, f"locating the newest record on partition {p}"
                ),
            )
            positions.append(position)
            if target is not None:
                targets[p] = target
        consumer.assign(positions)
        self._catch_up_until = targets

    def _drain_latest(
        self, poll_timeout: float, key_filter: Optional[str]
    ) -> BlockResult:
        consumer = self._consumer
        try:
            if self._mode != MODE_LATEST:
                # leaving pointer mode (or never positioned): back to the newest record
                # on the topic, exactly like a first run
                self._reposition_at_newest(
                    consumer, self._scan_window(key_filter, len(self._partitions))
                )
            self._mode = MODE_LATEST
            self._raise_on_auth_failure()
            newest = None
            error_message: Optional[str] = None
            timeout = poll_timeout
            catch_up_deadline = (
                time.monotonic() + self._reposition_timeout
                if self._catch_up_until
                else None
            )
            for attempt in range(2):
                newest, polling_error, drained = self._poll_until_drained(
                    consumer, timeout, key_filter, catch_up_deadline
                )
                error_message = polling_error or error_message
                if drained or attempt == 1:
                    break
                # more than MAX_MESSAGES_PER_RUN records queued since the last run:
                # skip the backlog and read the newest record directly, so the block
                # never lags behind by reporting an old record as newest
                logger.warning(
                    "Kafka Consumer: over %s records queued since the last run on topic "
                    "%r; skipping to the newest record",
                    MAX_MESSAGES_PER_RUN,
                    self._topic,
                )
                self._reposition_at_newest(consumer, window=1)
                catch_up_deadline = time.monotonic() + self._reposition_timeout
                timeout = 0
            error_message = self._pop_auth_failure_message() or error_message
        except ConfigurationError as error:
            return self._failure(str(error))
        except Exception as error:
            try:
                self._raise_on_auth_failure()
            except ConfigurationError as auth_error:
                return self._failure(str(auth_error))
            return self._failure(f"Failed to poll Kafka: {describe_error(error)}")
        if newest is not None:
            return self._accept(newest, error_message)
        return _outputs(self._last, is_new=False, error_message=error_message)
