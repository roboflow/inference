"""Kafka consumer and producer blocks over explicit, host-owned Kafka clients.

Ports of V1 ``roboflow_enterprise/kafka_consumer@v1`` and
``roboflow_enterprise/kafka_producer_sink@v1``. Neither block creates a Kafka
client. The host constructs the owners, lends them to sessions and closes them::

    host owns (explicit)                         block borrows
    ────────────────────                         ─────────────
    cursor = KafkaConsumerCursor()
    publisher = KafkaPublisher()
    with plan.create_session(resources={
            "kafka_consumer": cursor,            v2/kafka_consumer@v1
            "kafka_publisher": publisher,        v2/kafka_producer_sink@v1
    }) as session:
        session.run(...)                           cursor.read(KafkaRead)
                                                   publisher.publish(KafkaPublish, wait=...)
    # after runs stop and sessions close:
    publisher.close(timeout=5.0)   # stop admission, drain, settle: DeliveryStats
    cursor.close()                 # Consumer.close(), once

Consumer. One ``KafkaConsumerCursor`` is one logical read position: the
connection, the partition assignment, the read mode (latest, sequential or
pointer) and the last returned record. It keeps them across runs, sessions and
processing resets, like V1's block instance inside an ``InferencePipeline``.
Sharing a cursor shares that position: two sessions reading ``sequential``
from one cursor split the stream between them. Readers that must each see
every record need their own cursors. The cursor never commits offsets.

Producer. ``KafkaPublisher.publish`` queues one record in the client.
``fire_and_forget`` returns "Message scheduled for delivery": queued, not
delivered. Delivery reports are served on later ``publish``, ``flush`` and
``close`` calls and stay observable afterwards in ``stats()`` and
``failures()``. A confirmed publish waits up to ``timeout`` for its own
record's report; no record is ever produced twice. ``flush`` drains delivery
reports; it is not a close. ``close`` stops admission, waits for running
calls, flushes, and reports every record still without a report as
``unknown_at_close``, never as delivered.

Lifetime. Engine sessions and processing resets never close an owner, and an
owner holds no thread, timer or hook of its own. The host stops runs, closes
sessions, then closes the publisher and the cursor. To change broker, topic or
credentials, close the old owner and pass a new one with
``session.prepare_update(plan, resources=..., reset=True)``; a new owner starts
without the old one's position, client problems or authentication failures.

Callbacks. Each Kafka client gets its own ``error_cb`` and each record its own
``on_delivery``. Both keep the scrubber of the password they were created
with for as long as the client may call them: during ``close``, after a failed
connection attempt, and after the owner dropped the client.

Unresolved default path. There is no block-owned default client. V1 relied on
``__del__`` and an implicit flush. ``KafkaConsumerCursor.close`` calls
``Consumer.close()``, as V1 did. ``KafkaPublisher.close`` can only flush and
then drop its reference: this port knows no ``Producer`` method that releases
librdkafka's threads and sockets, and confluent-kafka was not available to
inspect. Releasing a producer therefore still depends on the SDK's object
lifetime. A per-operation default producer needs that evidence first.

Differences from V1 (both blocks):

* The host must lend an owner; without one an enabled step fails with
  ``KafkaHostError``. A closed owner also fails the step. For the consumer
  this takes precedence over invalid parameters. A producer step with
  an existing publisher reports invalid parameters without consulting it;
  a missing publisher is checked first and fails even with invalid parameters.
* Parameters of the wrong type or out of range fail compilation, or the step
  when selected; V1 coerced or returned ``error_status=True``. Numeric
  strings, floats, string booleans and non-string keys are no longer coerced.
* ``password`` is a ``Secret``; one without a value fails the step. The supplied
  owners scrub their diagnostic messages, logs and kept failures; the blocks
  scrub their own configuration diagnostics. Record data is never scrubbed.
  Injected ports own their diagnostics after revealing a Secret, including
  their outputs, logs, retained failures and exceptions. Arbitrary host code
  and external library logging are outside this guarantee.
* The fatal-client message asks the host to replace the owner; V1 asked to
  restart the block.
* A confirmed publish waits for its own report, not for the whole queue (V1
  called ``flush``, which with a shared publisher waits for other steps).
* Background delivery failures are kept in ``failures()`` as well as logged.
* The consumer's ``payload`` is a deep copy, so downstream mutation cannot
  reach the retained record. Optional JSON parsing or payload copying that
  exceeds Python's recursion limit yields ``payload={}`` while preserving the
  raw ``value`` and record metadata. This corrects an inherited V1 weakness.

Not verified: confluent-kafka, librdkafka and aws-msk-iam-sasl-signer were not
installed or imported here, and no broker or AWS service was contacted. Tests
use an in-memory stand-in for the ``confluent_kafka`` module and stand-in
token signers with invented credentials.
"""

import copy
import hashlib
import hmac
import importlib
import importlib.util
import ipaddress
import json
import logging
import math
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)
from uuid import uuid4

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)
from roboflow_workflows import environment
from roboflow_workflows.core_steps.common.workload_presets import (
    FIRE_AND_FORGET_RESTRICTION,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    DiscoveryProblem,
    DiscoveryProblemCode,
    Runtime,
    RuntimeInputMode,
    RuntimeRestriction,
    Severity,
    WorkOperation,
    incomplete_discovery,
)
from roboflow_workflows.v2.blocks.sinks.runtime import (
    ERROR_STATUS_OUTPUT,
    MESSAGE_OUTPUT,
    disabled_sink_message,
)
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.secrets import (
    SECRET_KIND,
    Secret,
    reveal_secrets,
    to_secret,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

LOGGER = logging.getLogger(__name__)

CONSUMER_BLOCK_TYPE = "v2/kafka_consumer@v1"
PRODUCER_BLOCK_TYPE = "v2/kafka_producer_sink@v1"

PROVIDER_SELF_HOSTED = "Self-hosted"
PROVIDER_AWS_MSK = "AWS MSK"
PROVIDERS = (PROVIDER_SELF_HOSTED, PROVIDER_AWS_MSK)
MSK_HOST_PATTERN = re.compile(
    r"\.kafka(?:-serverless)?\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?$", re.IGNORECASE
)
# Every client gets its own group, so readers never share partitions.
GROUP_ID_PREFIX = "roboflow-inference-"
# Only warnings and errors from librdkafka itself.
LIBRDKAFKA_LOG_LEVEL = 4
# Port librdkafka applies to a bootstrap entry that names none.
DEFAULT_KAFKA_PORT = "9092"
# While a client-reported problem is recorded, a run may probe topic metadata to
# learn whether the connection recovered: at once for a new problem, then at
# most once per interval, each probe waiting at most the timeout.
CLIENT_RECOVERY_PROBE_INTERVAL = 5.0
CLIENT_RECOVERY_PROBE_TIMEOUT = 1.0
# Errors a broker returns while a topic is still being (auto-)created:
# UNKNOWN_TOPIC_OR_PART and LEADER_NOT_AVAILABLE.
TRANSIENT_TOPIC_ERROR_CODES = (3, 5)
TOPIC_METADATA_RETRY_INTERVAL = 0.2
# librdkafka stores millisecond settings in a signed 32-bit int.
MAX_TIMEOUT_SECONDS = 2_147_483
MAX_MESSAGES_PER_RUN = 1000
# A confirmed publish polls in slices this long, so a report served by another
# thread's poll is noticed promptly.
DELIVERY_POLL_SLICE = 0.1

READ_MODE_LATEST = "latest"
READ_MODE_SEQUENTIAL = "sequential"
READ_MODES = (READ_MODE_LATEST, READ_MODE_SEQUENTIAL)
# How a cursor is positioned: one of the read modes, or at a pointed record.
MODE_POINTER = "pointer"
ACKS_VALUES = ("0", "1", "all")

SCHEDULED_MESSAGE = "Message scheduled for delivery"
CONSUMER_HOSTED_MESSAGE = (
    "Kafka Consumer is not available on the Roboflow hosted platform."
)
PRODUCER_HOSTED_MESSAGE = (
    "Kafka Producer is not available on the Roboflow hosted platform."
)
MSK_SIGNER_MISSING_MESSAGE = (
    "AWS MSK provider requires the aws-msk-iam-sasl-signer-python package."
)
NO_CURSOR_MESSAGE = (
    'Kafka Consumer needs a host-owned KafkaConsumerCursor in resources["kafka_consumer"]; '
    "the host creates it and closes it after sessions stop."
)
NO_PUBLISHER_MESSAGE = (
    'Kafka Producer needs a host-owned KafkaPublisher in resources["kafka_publisher"]; '
    "the host creates it and closes it after sessions stop."
)
CLOSED_CURSOR_MESSAGE = (
    "KafkaConsumerCursor is closed; close it only after its sessions stop running."
)
CLOSED_PUBLISHER_MESSAGE = (
    "KafkaPublisher is closed; close it only after its sessions stop running."
)

SELF_HOSTED_RELEVANT = {
    "provider": {"values": [PROVIDER_SELF_HOSTED], "required": False}
}
AWS_MSK_RELEVANT = {"provider": {"values": [PROVIDER_AWS_MSK], "required": False}}

CONSUMER_HOSTED_RESTRICTION = RuntimeRestriction(
    code="unavailable_on_hosted_platform",
    severity=Severity.HARD,
    note=(
        "On the Roboflow hosted platform every run returns "
        "`error_status=true` with no record, and no Kafka connection is "
        "opened from the hosted platform."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
)
CONSUMER_PER_REQUEST_RESTRICTION = RuntimeRestriction(
    code="connection_and_state_rebuilt_per_request",
    severity=Severity.SOFT,
    note=(
        "The connection, the read position and the last-returned record live "
        "in the host-owned KafkaConsumerCursor. A host that builds a new cursor "
        "for every HTTP request pays a broker connection per request and "
        "`is_new` is always True. A long-lived cursor avoids both."
    ),
    applies_to_runtimes=[
        Runtime.SELF_HOSTED_CPU,
        Runtime.SELF_HOSTED_GPU,
        Runtime.DEDICATED_DEPLOYMENT,
    ],
    applies_to_input_modes=[RuntimeInputMode.IMAGE],
)
PRODUCER_HOSTED_RESTRICTION = RuntimeRestriction(
    code="unavailable_on_hosted_platform",
    severity=Severity.HARD,
    note=(
        "On the Roboflow hosted platform every run returns "
        "`error_status=true` and publishes nothing; no Kafka connection is "
        "opened from the hosted platform."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
)


class KafkaConfigurationError(ValueError):
    """A configuration problem found before or while connecting.

    Reported as ``error_status=True``; the next run tries again.
    """


class KafkaHostError(RuntimeError):
    """The host broke the ownership contract: no owner, or a closed one.

    Raised from ``run``, so the step fails instead of reporting an output.
    """


# --- operator policy and connection settings ---------------------------------


def _normalise_port(port: str) -> str:
    port = port.strip()
    if not port:
        return DEFAULT_KAFKA_PORT

    return str(int(port)) if port.isdigit() else port


def _normalise_host(host: str) -> str:
    host = host.strip().lower()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host

    return f"[{address.compressed}]" if address.version == 6 else address.compressed


def normalise_bootstrap_server(entry: str) -> str:
    """Return the ``host:port`` form compared with the operator's allowlist.

    Whitespace is trimmed, the host lowercased, a missing port read as 9092
    and an IPv6 address bracketed. Text only, no DNS resolution. A malformed
    entry is returned lowercased, so it matches only the same malformed text.

    Args:
        entry: One bootstrap entry, e.g. ``"Broker-1:09092"``.

    Returns:
        The normalized entry, e.g. ``"broker-1:9092"``.
    """
    entry = str(entry).strip()
    if entry.startswith("["):
        host, closing, rest = entry[1:].partition("]")
        if closing and (not rest or rest.startswith(":")):
            return f"{_normalise_host(host)}:{_normalise_port(rest[1:])}"
        return entry.lower()

    if entry.count(":") == 1:
        host, port = entry.split(":")
        return f"{_normalise_host(host)}:{_normalise_port(port)}"

    if entry.count(":") > 1 and not _normalise_host(entry).startswith("["):
        # Several colons, yet not an IPv6 literal (e.g. a `SSL://host:port` spelling).
        return entry.lower()

    # A bare host, or an IPv6 literal without brackets, which cannot carry a port.
    normalized = f"{_normalise_host(entry)}:{DEFAULT_KAFKA_PORT}"

    return normalized


def _split_bootstrap_servers(value: Any) -> List[str]:
    entries: Iterable[Any] = value.split(",") if isinstance(value, str) else value
    return [str(entry).strip() for entry in entries if str(entry).strip()]


def resolve_bootstrap_servers(bootstrap_servers: str, *, log_override: bool) -> str:
    """Apply the operator's bootstrap-server policy, as V1 did.

    ``environment.KAFKA_WORKFLOWS_SINKS_*`` is read on every call. When
    workflow-provided servers are not allowed, the operator's list replaces
    them. Otherwise an operator allowlist, if set, must contain every entry.

    Args:
        bootstrap_servers: The step's comma-separated ``host:port`` list.
        log_override: Log a WARNING when the operator's list replaces it.

    Returns:
        The servers the client connects to.

    Raises:
        KafkaConfigurationError: When the policy forbids the connection. The
            message never echoes the operator's allowlist.
    """
    user_provided_allowed = (
        environment.KAFKA_WORKFLOWS_SINKS_ALLOW_USER_PROVIDED_BOOTSTRAP_SERVERS
    )
    allowlist = environment.KAFKA_WORKFLOWS_SINKS_WHITELISTED_BOOTSTRAP_SERVERS
    operator_servers = (
        _split_bootstrap_servers(allowlist) if allowlist is not None else []
    )
    if not user_provided_allowed:
        if not operator_servers:
            raise KafkaConfigurationError(
                "Kafka blocks are disabled on this deployment: workflow-provided "
                "bootstrap servers are not allowed and the operator configured none."
            )
        if log_override:
            LOGGER.warning(
                "Kafka blocks: the workflow-provided bootstrap_servers value was "
                "replaced by the operator-configured servers (%s), because "
                "KAFKA_WORKFLOWS_SINKS_ALLOW_USER_PROVIDED_BOOTSTRAP_SERVERS is False.",
                ",".join(operator_servers),
            )
        # Exactly what the operator wrote, in the operator's order.
        replaced = ",".join(operator_servers)
        return replaced

    if allowlist is None:
        return bootstrap_servers

    entries = _split_bootstrap_servers(bootstrap_servers)
    if not entries:
        raise KafkaConfigurationError("bootstrap_servers holds no host:port entry.")

    permitted = {normalise_bootstrap_server(entry) for entry in operator_servers}
    for entry in entries:
        if normalise_bootstrap_server(entry) not in permitted:
            raise KafkaConfigurationError(
                f"Bootstrap server {entry!r} is not permitted on this deployment: the "
                "operator restricts the Kafka servers Workflow blocks may connect to."
            )

    return bootstrap_servers


def derive_msk_region(bootstrap_servers: str) -> Optional[str]:
    """Read the AWS region from an MSK broker address.

    Args:
        bootstrap_servers: Comma-separated ``host:port`` list.

    Returns:
        The region of the first ``*.kafka.<region>.amazonaws.com`` host, or
        ``None``.
    """
    for entry in str(bootstrap_servers).split(","):
        host = entry.strip()
        if not host:
            continue
        if host.count(":") == 1:
            host = host.rsplit(":", 1)[0]
        match = MSK_HOST_PATTERN.search(host)
        if match:
            return match.group(1).lower()

    return None


@dataclass(frozen=True)
class KafkaConnection:
    """Validated connection settings of one step; the password stays a Secret.

    Args:
        bootstrap_servers: Servers after the operator's policy.
        topic: Topic name.
        provider: ``"Self-hosted"`` or ``"AWS MSK"``.
        username: SASL username, or ``None``.
        password: SASL password, or ``None``.
        aws_region: The step's ``aws_region`` as given.
        msk_region: Region that signs MSK tokens; ``None`` when self-hosted.
        ssl_ca_location: CA bundle path, or ``None``.
    """

    bootstrap_servers: str
    topic: str
    provider: str
    username: Optional[str]
    password: Optional[Secret] = field(repr=False)
    aws_region: Optional[str]
    msk_region: Optional[str]
    ssl_ca_location: Optional[str]


def kafka_connection(
    bootstrap_servers: str,
    *,
    topic: str,
    provider: str,
    username: Optional[str],
    password: Optional[Secret],
    aws_region: Optional[str],
    ssl_ca_location: Optional[str],
    allow_access_to_file_system: bool,
    log_override: bool,
) -> KafkaConnection:
    """Validate a step's connection settings with V1's rules and messages.

    Args:
        bootstrap_servers: The step's servers, before the operator's policy.
        topic: Topic name.
        provider: ``"Self-hosted"`` or ``"AWS MSK"``.
        username: SASL username, or ``None``.
        password: SASL password, or ``None``. Not revealed here.
        aws_region: Explicit MSK region, or ``None`` to derive it.
        ssl_ca_location: CA bundle path, or ``None``.
        allow_access_to_file_system: Host permission to read
            ``ssl_ca_location`` from the server's file system.
        log_override: Log when the operator's servers replace the step's.

    Returns:
        The validated settings.

    Raises:
        KafkaConfigurationError: With V1's message for the first problem.
    """
    bootstrap_servers = str(bootstrap_servers).strip()
    topic = str(topic).strip()
    if not bootstrap_servers or not topic:
        raise KafkaConfigurationError("bootstrap_servers and topic must be set.")

    bootstrap_servers = resolve_bootstrap_servers(
        bootstrap_servers, log_override=log_override
    )
    if ssl_ca_location and not allow_access_to_file_system:
        # The path is chosen by the workflow and read by the server process.
        raise KafkaConfigurationError(
            "ssl_ca_location needs access to the local file system, which is disabled "
            "on this deployment (ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=False)."
        )

    msk_region = None
    if provider == PROVIDER_SELF_HOSTED:
        if (username is None) != (password is None):
            raise KafkaConfigurationError(
                "Set both username and password for SASL authentication, or neither for "
                "an unauthenticated plaintext listener."
            )
    elif provider == PROVIDER_AWS_MSK:
        msk_region = str(aws_region).strip() if aws_region else None
        if not msk_region:
            msk_region = derive_msk_region(bootstrap_servers)
        if not msk_region:
            raise KafkaConfigurationError(
                "Could not derive the AWS region from bootstrap_servers; set aws_region."
            )
    else:
        raise KafkaConfigurationError(
            f"Unknown provider {provider!r}; expected one of {', '.join(PROVIDERS)}."
        )

    connection = KafkaConnection(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        provider=provider,
        username=username,
        password=password,
        aws_region=aws_region,
        msk_region=msk_region,
        ssl_ca_location=ssl_ca_location or None,
    )

    return connection


def import_confluent_kafka(*, missing_message: str) -> Any:
    """Import the optional confluent-kafka driver at first use.

    Args:
        missing_message: Message naming what to install.

    Returns:
        The ``confluent_kafka`` module.

    Raises:
        KafkaConfigurationError: With ``missing_message`` when it is missing.
    """
    try:
        module = importlib.import_module("confluent_kafka")
    except ImportError:
        raise KafkaConfigurationError(missing_message) from None

    return module


def sign_msk_token(region: str) -> Tuple[str, float]:
    """Default MSK signer: the host's AWS credential chain, via the AWS package.

    Args:
        region: AWS region of the cluster.

    Returns:
        The token and its expiry in epoch milliseconds.

    Raises:
        KafkaConfigurationError: When aws-msk-iam-sasl-signer is missing.
    """
    try:
        signer = importlib.import_module("aws_msk_iam_sasl_signer")
    except ImportError:
        raise KafkaConfigurationError(MSK_SIGNER_MISSING_MESSAGE) from None

    token, expiry_ms = signer.MSKAuthTokenProvider.generate_auth_token(region)

    return token, expiry_ms


def msk_token_callback(
    region: str,
    *,
    failures: List[BaseException],
    signer: Callable[[str], Tuple[str, float]],
) -> Callable[[str], Tuple[str, float]]:
    """Build librdkafka's OAUTHBEARER callback signing MSK IAM tokens.

    librdkafka calls it on its own thread. A failure is kept in ``failures``
    (the latest only) so a run can report it once; a later success clears it.

    Args:
        region: AWS region of the cluster.
        failures: The owner's list of the latest signing failure.
        signer: ``signer(region)`` returning ``(token, expiry_ms)``.

    Returns:
        ``callback(config) -> (token, expiry_seconds)``.
    """

    def callback(_config: str) -> Tuple[str, float]:
        try:
            token, expiry_ms = signer(region)
        except Exception as error:
            failures[:] = [error]
            raise
        failures.clear()
        # The signer reports expiry in epoch milliseconds; librdkafka wants seconds.
        return token, expiry_ms / 1000

    return callback


def _module_available(name: str) -> bool:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        # ValueError: sys.modules holds None for a module blocked from import.
        return False

    return spec is not None


def _client_settings(
    connection: KafkaConnection,
    *,
    auth_failures: List[BaseException],
    signer: Optional[Callable[[str], Tuple[str, float]]],
) -> Dict[str, Any]:
    # Transport and SASL settings derived from the provider; reveals the password.
    if connection.provider == PROVIDER_AWS_MSK:
        if signer is None:
            # Checked without importing; the default signer imports at first use.
            if not _module_available("aws_msk_iam_sasl_signer"):
                raise KafkaConfigurationError(MSK_SIGNER_MISSING_MESSAGE)
            signer = sign_msk_token
        settings: Dict[str, Any] = {
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "OAUTHBEARER",
            "oauth_cb": msk_token_callback(
                connection.msk_region, failures=auth_failures, signer=signer
            ),
        }
    elif connection.username is None:
        settings = {"security.protocol": "PLAINTEXT"}
    else:
        settings = {
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "SCRAM-SHA-512",
            "sasl.username": str(connection.username),
            "sasl.password": connection.password.reveal(),
        }
    if connection.ssl_ca_location:
        settings["ssl.ca.location"] = str(connection.ssl_ca_location)

    return settings


def _preflight_token(settings: Dict[str, Any]) -> None:
    # Sign once before constructing a client: the confluent-kafka constructor
    # blocks for 10s and raises an opaque SASL error when the token callback
    # fails, so a missing AWS credential chain must fail fast with its cause.
    token_callback = settings.get("oauth_cb")
    if token_callback is not None:
        token_callback("")


def _identity(
    connection: KafkaConnection, *, salt: bytes, extra: Tuple[Any, ...] = ()
) -> Tuple[Any, ...]:
    # V1's connection identity. The password is kept as a salted digest, so the
    # identity holds no plaintext credential.
    password = connection.password
    fingerprint = (
        None
        if password is None
        else hmac.new(salt, password.reveal().encode("utf-8"), hashlib.sha256).digest()
    )
    identity = (
        connection.bootstrap_servers,
        connection.topic,
        connection.provider,
        connection.username,
        fingerprint,
        connection.aws_region,
        connection.ssl_ca_location,
        *extra,
    )

    return identity


# --- error reporting ----------------------------------------------------------


def describe_error(error: BaseException) -> str:
    """Return an exception's text, or its type name when the text is empty.

    Args:
        error: The exception.

    Returns:
        The description.
    """
    text = str(error).strip()
    description = text or error.__class__.__name__

    return description


def describe_auth_error(error: BaseException) -> str:
    """Describe an MSK signing failure; an empty credential chain gets V1's text.

    Args:
        error: What the signer raised.

    Returns:
        The description.
    """
    text = describe_error(error)
    # botocore reports an empty credential chain as NoCredentialsError or, via
    # the MSK signer, as an AttributeError on the missing credentials object.
    if "access_key" in text or error.__class__.__name__ in (
        "NoCredentialsError",
        "PartialCredentialsError",
    ):
        return "no AWS credentials were found on this machine"

    return text


def pop_auth_failure_message(failures: List[BaseException]) -> Optional[str]:
    """Report a recorded MSK signing failure once and clear it.

    The token callback mutates ``failures`` on librdkafka's thread; a refresh
    succeeding between a check and the pop must not raise here.

    Args:
        failures: An owner's list of the latest signing failure.

    Returns:
        The message, or ``None`` when no failure is recorded.
    """
    try:
        error = failures.pop()
    except IndexError:
        return None

    failures.clear()
    message = (
        "AWS MSK IAM authentication failed: "
        f"{describe_auth_error(error)}. Check that AWS credentials are available on "
        "this machine and that the identity may connect to the cluster."
    )

    return message


def describe_client_error(error: Any) -> str:
    """Describe what librdkafka passed to ``error_cb``; never raises.

    Args:
        error: Normally a ``KafkaError``; stand-ins may lack its accessors.

    Returns:
        ``"<name>: <text>"`` or just the text.
    """
    try:
        text = str(error).strip() or error.__class__.__name__
    except Exception:
        text = error.__class__.__name__
    label = None
    for accessor in ("name", "code"):
        try:
            label = getattr(error, accessor)()
        except Exception:
            continue
        if label is not None:
            break
    if label is None or str(label) in text:
        return text

    description = f"{label}: {text}"

    return description


def _is_fatal(error: Any) -> bool:
    try:
        return bool(error.fatal())
    except Exception:
        return False


def combine_messages(*messages: Optional[str]) -> Optional[str]:
    """Join the present messages with ``"; "``.

    Args:
        *messages: Messages, some possibly ``None`` or empty.

    Returns:
        The joined text, or ``None`` when none is present.
    """
    present = [message for message in messages if message]
    combined = "; ".join(present) if present else None

    return combined


def _unchanged(text: str) -> str:
    return text


class ClientProblem(NamedTuple):
    """A problem reported by a client's ``error_cb``.

    Args:
        description: What the client reported, scrubbed.
        fatal: The client is unusable.
        since: Monotonic time of the first error of this streak.
        recorded_at: Monotonic time of the latest error.
    """

    description: str
    fatal: bool
    since: float
    recorded_at: float

    def message(self, note: Optional[str], *, now: float) -> str:
        """Return the run's message for this problem.

        Args:
            note: What it means for the returned data, if anything.
            now: Current monotonic time.

        Returns:
            The message.
        """
        age = max(0.0, now - self.since)
        if self.fatal:
            text = (
                f"The Kafka client reported a fatal error {age:.1f}s ago and is "
                f"unusable: {self.description}. Restart the pipeline: the host must "
                "replace its Kafka owner to reconnect."
            )
        else:
            text = (
                f"The Kafka client reported a broker problem, first seen {age:.1f}s "
                f"ago: {self.description}. This is reported on every run until the "
                "connection is proven healthy again."
            )
        message = f"{text} {note}" if note else text

        return message


class ClientErrorTracker:
    """``error_cb`` of one librdkafka client, with V1's recovery probing.

    Global client errors ("all brokers down", transport failures) reach only
    this callback, from inside ``poll()`` / ``flush()``. Harmless errors (one
    broker of several disconnecting) arrive here too, so a new problem is
    checked against the broker on the next ``check``: a successful metadata
    probe drops it unreported. Otherwise it is reported on every run until
    proof of life: ``clear`` (a polled record, a successful delivery report) or
    a later successful probe, at most once per
    ``CLIENT_RECOVERY_PROBE_INTERVAL``. A fatal error is never probed or
    cleared.

    One tracker per client. The client may call it until the client is gone,
    which includes its ``close()`` and any time after the owner dropped it, so
    ``scrub`` is fixed for the tracker's lifetime. A replacement client gets a
    new tracker: an old client's late errors are scrubbed with the old
    client's password and never become the new client's problems.

    Args:
        client_name: Shown in log records.
        scrub: Replaces this client's credentials in a text.
        clock: Monotonic clock.
    """

    def __init__(
        self,
        client_name: str,
        *,
        scrub: Callable[[str], str],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client_name = client_name
        self._scrub = scrub
        self._clock = clock
        self._lock = threading.Lock()
        self._problem: Optional[ClientProblem] = None
        self._generation = 0
        self._last_probe_at = 0.0

    def __call__(self, error: Any) -> None:
        try:
            now = self._clock()
            description = self._scrub(describe_client_error(error))
            with self._lock:
                fatal = _is_fatal(error)
                self._generation += 1
                previous = self._problem
                if previous is not None and previous.fatal:
                    return
                if previous is None:
                    # The first probe is due at once: a healthy cluster answers in
                    # milliseconds, so a blip is never claimed.
                    self._last_probe_at = -math.inf
                self._problem = ClientProblem(
                    description=description,
                    fatal=fatal,
                    since=now if previous is None else previous.since,
                    recorded_at=now,
                )
            if previous is None or fatal:
                LOGGER.error(
                    "%s: the Kafka client reported %s: %s",
                    self._client_name,
                    "a fatal error" if fatal else "a broker problem",
                    description,
                )
        except Exception:  # A callback raising inside librdkafka helps nobody.
            pass

    @property
    def generation(self) -> int:
        """Number of errors reported so far."""
        with self._lock:
            return self._generation

    @property
    def problem(self) -> Optional[ClientProblem]:
        """The recorded problem, or ``None``."""
        with self._lock:
            return self._problem

    def clear(self, if_generation: Optional[int] = None) -> None:
        """Record proof of life.

        Args:
            if_generation: Clear only when no error arrived since this
                generation was read.
        """
        with self._lock:
            problem = self._problem
            if problem is None or problem.fatal:
                return
            if if_generation is not None and if_generation != self._generation:
                return
            self._problem = None
        LOGGER.info(
            "%s: Kafka connection is healthy again after: %s",
            self._client_name,
            problem.description,
        )

    def check(
        self,
        client: Any,
        topic: str,
        *,
        alive_since_generation: Optional[int] = None,
    ) -> Optional[ClientProblem]:
        """End-of-run check: apply proof of life, probe when due, report.

        Args:
            client: The client, for the metadata probe.
            topic: Topic to probe.
            alive_since_generation: Proof of life from this run, valid only if
                no error arrived since this generation was read.

        Returns:
            The problem this run must report, or ``None``.
        """
        if alive_since_generation is not None:
            self.clear(if_generation=alive_since_generation)
        with self._lock:
            problem = self._problem
            due = (
                problem is not None
                and not problem.fatal
                and self._clock() - self._last_probe_at
                >= CLIENT_RECOVERY_PROBE_INTERVAL
            )
            if due:
                self._last_probe_at = self._clock()
            generation = self._generation
        # Not under the lock: the client may serve callbacks while it waits.
        if due and probe_topic_metadata(client, topic):
            self.clear(if_generation=generation)

        current = self.problem

        return current

    def problem_message(self, problem: ClientProblem, note: Optional[str]) -> str:
        """Return ``problem.message(note)`` at this tracker's current time.

        Args:
            problem: The problem.
            note: What it means for the returned data.

        Returns:
            The message.
        """
        message = problem.message(note, now=self._clock())

        return message


def probe_topic_metadata(client: Any, topic: str) -> bool:
    """Tell whether the broker answers a metadata request for ``topic``.

    Args:
        client: A consumer or producer.
        topic: Topic name.

    Returns:
        True when the topic's metadata arrived without an error.
    """
    try:
        metadata = client.list_topics(topic, timeout=CLIENT_RECOVERY_PROBE_TIMEOUT)
        topic_metadata = metadata.topics.get(topic)
        healthy = (
            topic_metadata is not None
            and getattr(topic_metadata, "error", None) is None
        )
    except Exception:
        return False

    return healthy


def _is_transient_topic_error(error: Any) -> bool:
    code = getattr(error, "code", None)
    try:
        return callable(code) and code() in TRANSIENT_TOPIC_ERROR_CODES
    except Exception:
        return False


def _time_remaining(deadline: float, what: str, *, clock: Callable[[], float]) -> float:
    remaining = deadline - clock()
    if remaining <= 0:
        raise TimeoutError(f"timed out {what}")

    return remaining


def wait_for_topic_metadata(
    client: Any,
    topic: str,
    *,
    deadline: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> Any:
    """Return a topic's metadata, riding out its auto-creation.

    Retries until ``deadline`` while the broker reports the topic unknown or
    leaderless, as a stock Kafka client does.

    Args:
        client: A consumer or producer.
        topic: Topic name.
        deadline: Monotonic deadline.
        clock: Monotonic clock.
        sleep: Waits between retries.

    Returns:
        The topic's metadata.

    Raises:
        KafkaConfigurationError: When the topic stays unavailable or the error
            is not transient.
        TimeoutError: When the deadline passed before a request.
    """
    while True:
        metadata = client.list_topics(
            topic,
            timeout=_time_remaining(deadline, "fetching topic metadata", clock=clock),
        )
        topic_metadata = metadata.topics.get(topic)
        error = getattr(topic_metadata, "error", None) if topic_metadata else None
        if topic_metadata is not None and error is None:
            return topic_metadata

        transient = topic_metadata is None or _is_transient_topic_error(error)
        if transient and deadline - clock() > TOPIC_METADATA_RETRY_INTERVAL:
            sleep(TOPIC_METADATA_RETRY_INTERVAL)
            continue

        detail = str(error) if topic_metadata is not None else "not found"
        raise KafkaConfigurationError(
            f"Kafka topic {topic!r} is not available ({detail})."
        )


def _close_quietly(client: Any) -> None:
    try:
        client.close()
    except Exception:
        pass


def _scrubber(*secrets: Optional[Secret]) -> Callable[[str], str]:
    # Scrubs every given secret's plaintext; ignores None.
    revealed = reveal_secrets([secret for secret in secrets if secret is not None])
    return revealed.scrub


# --- consumer -----------------------------------------------------------------


@dataclass(frozen=True)
class KafkaRead:
    """One validated consumer run.

    Args:
        connection: Where and how to connect.
        read_mode: ``"latest"`` or ``"sequential"``.
        key_filter: Only records with this key, or ``None``.
        pointer: ``(partition, offset)`` of a pointed record, or ``None``.
        poll_timeout: Seconds to wait for a record on each run.
        connect_timeout: Seconds for connecting, positioning and a pointed read.
    """

    connection: KafkaConnection
    read_mode: str
    key_filter: Optional[str]
    pointer: Optional[Tuple[int, int]]
    poll_timeout: float
    connect_timeout: float


class KafkaConsumerPort(Protocol):
    """Trusted host reader borrowed by the block; implemented by the cursor.

    The block passes through outputs and exceptions. Implementers own plaintext
    after Secret reveal: use ``reveal_secrets(...).scrub(...)`` on diagnostic
    text before returning, logging, retaining or raising it. Include credentials
    retained by older clients/callbacks where relevant. Do not log raw exceptions
    or expose their traceback/chaining when replacing a sensitive failure.
    Exact plaintext scrubbing does not cover encoded secrets or library logs.
    Preserve record values, payload fields, keys and bytes; these are data,
    not diagnostic text. The example in ``102-kafka/run_demo.py`` shows this
    boundary without requiring an SDK-shaped driver.

    Follow the supplied cursor's error convention: ordinary broker/configuration
    problems return error outputs with the last record; closed/retired host
    ownership raises ``KafkaHostError`` and unavailable credentials raise
    ``SecretUnavailableError`` before external acquisition. Those exceptions
    intentionally fail the step. Unexpected implementation exceptions also
    propagate; the block does not convert or sanitize them.

    Shared callers require serialization by the host or the implementation;
    the block adds no lock. The supplied cursor serializes its own calls. Only
    ``read`` and ``repeat_last`` are required here; lifecycle stays host-owned.
    """

    def read(self, request: KafkaRead) -> Dict[str, Any]:
        """Read per the request and return the block's outputs.

        Args:
            request: The validated run.

        Returns:
            ``value``, ``payload``, ``key``, ``is_new``, ``offset``,
            ``partition``, ``error_status`` and ``error_message``. Broker and
            configuration problems use ``error_status=True`` and a scrubbed
            ``error_message`` with the last record and ``is_new=False``.

        Raises:
            KafkaHostError: For closed/retired host ownership.
            SecretUnavailableError: Before acquisition if a credential is
                unavailable.
        """

    def repeat_last(self, *, error_message: str) -> Dict[str, Any]:
        """Return the outputs of a run that failed before reading.

        Args:
            error_message: The failure, already scrubbed.

        Returns:
            The last record with ``is_new=False`` and the scrubbed error.

        Raises:
            KafkaHostError: When the owner is closed, as ``read`` does.
        """


class _Record(NamedTuple):
    partition: int
    offset: int
    key: Optional[str]
    value: Optional[str]
    payload: Dict[str, Any]


def _outputs(
    record: Optional[_Record],
    *,
    is_new: bool = False,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {}
    if record is not None:
        try:
            payload = copy.deepcopy(record.payload)
        except RecursionError:
            # The optional object view must not destroy an accepted raw record.
            pass

    outputs = {
        "value": record.value if record else None,
        "payload": payload,
        "key": record.key if record else None,
        "is_new": is_new,
        "offset": record.offset if record else None,
        "partition": record.partition if record else None,
        "error_status": error_message is not None,
        "error_message": error_message,
    }

    return outputs


def _decode_text(raw: Any) -> Optional[str]:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")

    return None if raw is None else str(raw)


def _decode(message: Any) -> _Record:
    # A null value is a tombstone (key deleted on a compacted topic): surface it
    # as None rather than an empty string, so downstream can tell them apart.
    value = _decode_text(message.value())
    payload: Dict[str, Any] = {}
    if value is not None:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                payload = parsed
        except (ValueError, RecursionError):
            # Malformed or unrepresentable JSON still has a usable raw value.
            pass
    record = _Record(
        partition=message.partition(),
        offset=message.offset(),
        key=_decode_text(message.key()),
        value=value,
        payload=payload,
    )

    return record


def _message_key_matches(message: Any, key_filter: Optional[str]) -> bool:
    if key_filter is None:
        return True

    raw_key = message.key()
    if raw_key is None:
        return False

    matches = _decode_text(raw_key) == str(key_filter)

    return matches


class KafkaConsumerCursor:
    """Host-owned Kafka consumer: one logical read position across runs.

    The first ``read`` connects with that run's settings and positions the
    consumer; later reads must use the same settings (V1's identity check) and
    continue from the retained position. Reads are serialized. Nothing is
    committed: every cursor uses its own consumer group and ``assign``.

    Sharing a cursor shares its position, mode and last record. The host
    closes it after every borrowing session stopped; ``close`` waits for a
    running read, calls ``Consumer.close()`` once and rejects later reads.

    Args:
        driver: The ``confluent_kafka`` module or a stand-in providing
            ``Consumer`` and ``TopicPartition``; imported at first connect
            when ``None``.
        msk_token_signer: ``signer(region) -> (token, expiry_ms)`` for the
            AWS MSK provider; the default uses aws-msk-iam-sasl-signer and
            the host's AWS credential chain.
        clock: Monotonic clock for deadlines and problem ages.
        sleep: Waits between metadata retries.
    """

    def __init__(
        self,
        *,
        driver: Any = None,
        msk_token_signer: Optional[Callable[[str], Tuple[str, float]]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._driver = driver
        self._signer = msk_token_signer
        self._clock = clock
        self._sleep = sleep
        self._salt = os.urandom(16)
        self._lock = threading.Lock()
        self._closed = False
        # The connected consumer's error_cb; None while no consumer is connected.
        self._client_errors: Optional[ClientErrorTracker] = None
        self._consumer: Optional[Any] = None
        self._identity: Optional[Tuple[Any, ...]] = None
        self._bound_password: Optional[Secret] = None
        self._topic: Optional[str] = None
        self._partitions: Set[int] = set()
        self._last: Optional[_Record] = None
        # How the consumer is positioned: READ_MODE_LATEST, MODE_POINTER,
        # READ_MODE_SEQUENTIAL, or None when connected but not positioned yet.
        self._mode: Optional[str] = None
        # The pointer already applied as the sequential start position, if any.
        self._sequence_pointer: Optional[Tuple[int, int]] = None
        self._auth_failures: List[BaseException] = []
        self._reposition_timeout = 5.0
        # Offsets the first drain after an assignment must reach per partition
        # (the newest existing record, or the scan window with a key filter);
        # polling waits up to connect_timeout for them instead of poll_timeout.
        self._catch_up_until: Dict[int, int] = {}
        self._scrub: Callable[[str], str] = _unchanged

    @property
    def closed(self) -> bool:
        """Whether ``close`` was called."""
        return self._closed

    def read(self, request: KafkaRead) -> Dict[str, Any]:
        """Read one record per V1's modes and return the block's outputs.

        Args:
            request: The validated run.

        Returns:
            The outputs; broker and configuration problems are reported in
            ``error_status`` and ``error_message`` with the last record.

        Raises:
            KafkaHostError: When the cursor is closed.
            SecretUnavailableError: When the password has no value here.
        """
        with self._lock:
            if self._closed:
                raise KafkaHostError(CLOSED_CURSOR_MESSAGE)

            self._scrub = _scrubber(request.connection.password, self._bound_password)
            try:
                outputs = self._read(request)
            finally:
                self._scrub = _unchanged

        return outputs

    def repeat_last(self, *, error_message: str) -> Dict[str, Any]:
        """Return the last record with ``is_new=False`` and ``error_message``.

        Args:
            error_message: Why the run failed before reading; already scrubbed.

        Returns:
            The outputs.

        Raises:
            KafkaHostError: When the cursor is closed; this takes precedence
                over the run's own error, as in ``read``.
        """
        with self._lock:
            if self._closed:
                raise KafkaHostError(CLOSED_CURSOR_MESSAGE)

            outputs = self._failure(error_message)

        return outputs

    def close(self) -> None:
        """Close the consumer once and reject later reads; idempotent.

        Waits for a running read. A failing ``Consumer.close()`` is logged.
        """
        with self._lock:
            if self._closed:
                return

            self._closed = True
            consumer = self._consumer
            scrub = _scrubber(self._bound_password)
            self._unbind()
        if consumer is None:
            return

        try:
            consumer.close()
        except Exception as error:
            LOGGER.error(
                "Failed to close Kafka consumer: %s", scrub(describe_error(error))
            )

    def _unbind(self) -> None:
        # The consumer keeps its own tracker as error_cb, scrubber included,
        # for its close() and anything later.
        self._client_errors = None
        self._consumer = None
        self._identity = None
        self._bound_password = None
        self._topic = None
        self._partitions = set()
        self._mode = None
        self._sequence_pointer = None
        self._catch_up_until = {}

    def _read(self, request: KafkaRead) -> Dict[str, Any]:
        try:
            driver = self._load_driver()
        except KafkaConfigurationError as error:
            return self._failure(str(error))

        pointer = request.pointer
        sequential = request.read_mode == READ_MODE_SEQUENTIAL
        # A pointer positions the consumer itself, except the sequential start
        # pointer already applied on an earlier run.
        applies_pointer = pointer is not None and not (
            sequential and pointer == self._sequence_pointer
        )
        try:
            self._ensure_consumer(request, driver, skip_positioning=applies_pointer)
        except KafkaConfigurationError as error:
            return self._failure(str(error))
        except Exception as error:
            return self._failure(
                f"Kafka broker not reachable ({describe_error(error)}). "
                "Raise 'connect_timeout' if the broker needs longer to connect."
            )

        self._reposition_timeout = request.connect_timeout
        # Errors the client reports from here on arrive during this run's polls.
        errors_before_read = self._client_errors.generation
        if sequential and applies_pointer:
            result = self._read_at_pointer(
                pointer, request.connect_timeout, driver, continue_after=True
            )
        elif sequential:
            result = self._read_next(request.poll_timeout, request.key_filter, driver)
        elif pointer is not None:
            result = self._read_at_pointer(pointer, request.connect_timeout, driver)
        else:
            result = self._drain_latest(
                request.poll_timeout, request.key_filter, driver
            )
        reported = self._report_client_problem(result, errors_before_read)

        return reported

    def _load_driver(self) -> Any:
        if self._driver is None:
            self._driver = import_confluent_kafka(
                missing_message=(
                    "Kafka Consumer requires the confluent-kafka package in the runtime."
                )
            )

        return self._driver

    # --- lifecycle --------------------------------------------------------------

    def _ensure_consumer(
        self, request: KafkaRead, driver: Any, *, skip_positioning: bool
    ) -> None:
        connection = request.connection
        identity = _identity(connection, salt=self._salt)
        if self._consumer is not None:
            if identity != self._identity:
                raise KafkaConfigurationError(
                    "Kafka connection parameters (bootstrap servers, topic, provider or "
                    "credentials) changed between runs; this block reads only from the "
                    "connection configured on its first run."
                )
            return

        # A new client starts clean with its own tracker: problems of a previous
        # client are not its own, and its scrubber stays with it.
        client_errors = ClientErrorTracker(
            "Kafka Consumer", scrub=_scrubber(connection.password), clock=self._clock
        )
        auth_failures: List[BaseException] = []
        self._auth_failures = auth_failures
        settings = _client_settings(
            connection, auth_failures=auth_failures, signer=self._signer
        )
        config = {
            "bootstrap.servers": connection.bootstrap_servers,
            # A unique group per client, so every cursor sees every message;
            # assign() means the group is never used for rebalancing.
            "group.id": f"{GROUP_ID_PREFIX}{uuid4()}",
            "enable.auto.commit": False,
            "auto.offset.reset": "latest",
            "log_level": LIBRDKAFKA_LOG_LEVEL,
            # Global client errors arrive only here, served from inside poll().
            "error_cb": client_errors,
            **settings,
        }
        try:
            _preflight_token(settings)
        except Exception:
            self._raise_on_auth_failure()
            raise

        consumer = driver.Consumer(config)
        # One budget for the whole first run: metadata plus every watermark lookup.
        deadline = self._clock() + request.connect_timeout
        try:
            topic_metadata = wait_for_topic_metadata(
                consumer,
                connection.topic,
                deadline=deadline,
                clock=self._clock,
                sleep=self._sleep,
            )
            self._raise_on_auth_failure()
            partitions = sorted(topic_metadata.partitions)
            if not partitions:
                raise KafkaConfigurationError(
                    f"Kafka topic {connection.topic!r} has no partitions."
                )
            window = self._scan_window(request.key_filter, len(partitions))
            positions, targets = [], {}
            if not skip_positioning:
                for partition in partitions:
                    position, target = self._start_position(
                        consumer,
                        driver,
                        connection.topic,
                        partition,
                        window=window,
                        timeout=_time_remaining(
                            deadline,
                            f"locating the newest record on partition {partition}",
                            clock=self._clock,
                        ),
                    )
                    positions.append(position)
                    if target is not None:
                        targets[partition] = target
                consumer.assign(positions)
            self._catch_up_until = targets
        except Exception:
            # Partial construction: the consumer this attempt created is closed.
            _close_quietly(consumer)
            # A failing token callback surfaces as a generic transport error from
            # librdkafka; report the real cause instead.
            self._raise_on_auth_failure()
            raise

        self._consumer = consumer
        self._client_errors = client_errors
        self._identity = identity
        self._bound_password = connection.password
        self._topic = connection.topic
        self._partitions = set(partitions)
        # Positioned at the newest record; the read function claims the mode.
        # None: connected but not positioned (a pointer run assigns itself).
        self._mode = None if skip_positioning else READ_MODE_LATEST
        self._sequence_pointer = None

    def _raise_on_auth_failure(self) -> None:
        message = pop_auth_failure_message(self._auth_failures)
        if message is not None:
            raise KafkaConfigurationError(message)

    @staticmethod
    def _scan_window(key_filter: Optional[str], partitions: int) -> int:
        # Without a key filter the newest record is the answer; with one, scan a
        # bounded window backwards so the newest record for that key is found.
        if key_filter is None:
            return 1

        window = max(1, MAX_MESSAGES_PER_RUN // max(1, partitions))

        return window

    @staticmethod
    def _start_position(
        consumer: Any,
        driver: Any,
        topic: str,
        partition: int,
        *,
        window: int,
        timeout: float,
    ) -> Tuple[Any, Optional[int]]:
        # Position `window` records before the end; also return the newest
        # offset to reach (None for an empty partition).
        low, high = consumer.get_watermark_offsets(
            driver.TopicPartition(topic, partition), timeout=timeout
        )
        # high is the offset the next record will get; high - 1 is the newest.
        if high <= low:
            return driver.TopicPartition(topic, partition, high), None

        start = max(low, high - window)

        return driver.TopicPartition(topic, partition, start), high - 1

    @staticmethod
    def _next_record_position(
        consumer: Any, driver: Any, topic: str, partition: int, *, timeout: float
    ) -> Any:
        _, high = consumer.get_watermark_offsets(
            driver.TopicPartition(topic, partition), timeout=timeout
        )
        position = driver.TopicPartition(topic, partition, high)

        return position

    def _failure(self, message: str) -> Dict[str, Any]:
        message = self._scrub(message)
        LOGGER.error("Kafka Consumer failure: %s", message)
        outputs = _outputs(self._last, is_new=False, error_message=message)

        return outputs

    def _report_client_problem(
        self, result: Dict[str, Any], errors_before_read: int
    ) -> Dict[str, Any]:
        # A record new to this run came out of poll(): proof of life, unless the
        # client reported an error during the same run (their order is unknown).
        if self._consumer is None:
            return result

        problem = self._client_errors.check(
            self._consumer,
            self._topic,
            alive_since_generation=errors_before_read if result["is_new"] else None,
        )
        if problem is None:
            return result

        if result["is_new"]:
            note = (
                "The returned record is new, but the client reported the problem "
                "during the same run."
            )
        elif self._last is not None:
            note = "The returned record is the last one read and may be stale."
        else:
            note = "No record has been read yet."
        error_message = combine_messages(
            result["error_message"], self._client_errors.problem_message(problem, note)
        )
        reported = {**result, "error_status": True, "error_message": error_message}

        return reported

    # --- read paths ---------------------------------------------------------------

    def _read_at_pointer(
        self,
        pointer: Tuple[int, int],
        connect_timeout: float,
        driver: Any,
        *,
        continue_after: bool = False,
    ) -> Dict[str, Any]:
        partition, offset = pointer
        if (
            not continue_after
            and self._last is not None
            and (self._last.partition, self._last.offset) == pointer
        ):
            # Records are immutable: re-reading the same pointer is redundant. The
            # consumer was not repositioned, so the mode stays whatever it was.
            self._catch_up_until = {}
            return _outputs(self._last, is_new=False)

        if partition not in self._partitions:
            return self._failure(
                f"Partition {partition} does not exist on topic {self._topic!r} "
                f"(partitions: {sorted(self._partitions)})."
            )

        consumer = self._consumer
        deadline = self._clock() + connect_timeout
        try:
            low, high = consumer.get_watermark_offsets(
                driver.TopicPartition(self._topic, partition),
                timeout=_time_remaining(
                    deadline, "locating the pointed record", clock=self._clock
                ),
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
            # Concrete offsets everywhere: the target partition at the pointer, the
            # others at their next record so nothing old is fetched.
            consumer.assign(
                [driver.TopicPartition(self._topic, partition, offset)]
                + [
                    self._next_record_position(
                        consumer,
                        driver,
                        self._topic,
                        other,
                        timeout=_time_remaining(
                            deadline,
                            f"positioning partition {other}",
                            clock=self._clock,
                        ),
                    )
                    for other in sorted(self._partitions)
                    if other != partition
                ]
            )
            self._mode = READ_MODE_SEQUENTIAL if continue_after else MODE_POINTER
            self._sequence_pointer = pointer if continue_after else None
            self._catch_up_until = {}
            while True:
                remaining = deadline - self._clock()
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
        except KafkaConfigurationError as error:
            return self._failure(str(error))
        except Exception as error:
            auth_message = pop_auth_failure_message(self._auth_failures)
            if auth_message is not None:
                return self._failure(auth_message)
            return self._failure(
                f"Failed to read offset {offset} from partition {partition}: "
                f"{describe_error(error)}"
            )

    def _accept(
        self, message: Any, error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        # Make `message` the current record; is_new is False for the same record.
        record = _decode(message)
        is_new = self._last is None or (record.partition, record.offset) != (
            self._last.partition,
            self._last.offset,
        )
        self._last = record
        outputs = _outputs(
            record,
            is_new=is_new,
            error_message=None if error_message is None else self._scrub(error_message),
        )

        return outputs

    def _next_poll_timeout(
        self, timeout: float, catch_up_deadline: Optional[float]
    ) -> float:
        # Records known to exist have not all arrived yet: keep waiting in short
        # slices until they do or the deadline passes.
        if not self._catch_up_until:
            return timeout

        remaining = catch_up_deadline - self._clock()
        if remaining <= 0:
            self._catch_up_until = {}
            return timeout

        extended = max(timeout, min(remaining, 0.5))

        return extended

    def _note_caught_up(self, message: Any) -> None:
        target = self._catch_up_until.get(message.partition())
        if target is not None and message.offset() >= target:
            del self._catch_up_until[message.partition()]

    def _poll_one(
        self,
        consumer: Any,
        first_timeout: float,
        key_filter: Optional[str],
        catch_up_deadline: Optional[float],
    ) -> Tuple[Any, Optional[str]]:
        # Poll until one acceptable record arrives or the local queue is empty.
        error_message: Optional[str] = None
        timeout = first_timeout
        for _ in range(MAX_MESSAGES_PER_RUN):
            timeout = self._next_poll_timeout(timeout, catch_up_deadline)
            message = consumer.poll(timeout)
            timeout = 0
            if message is None:
                if self._catch_up_until:
                    continue
                return None, error_message
            self._note_caught_up(message)
            if message.error() is not None:
                error_message = f"Kafka error while polling: {message.error()}"
                LOGGER.error("Kafka Consumer: %s", self._scrub(error_message))
                continue
            if not _message_key_matches(message, key_filter):
                continue
            return message, error_message

        return None, error_message

    def _poll_until_drained(
        self,
        consumer: Any,
        first_timeout: float,
        key_filter: Optional[str],
        catch_up_deadline: Optional[float],
        *,
        stop_when_caught_up: bool = False,
    ) -> Tuple[Any, Optional[str], bool]:
        # Poll until the local queue is empty (drained=True) or the per-run cap is
        # hit. With stop_when_caught_up, stop once every catch-up target is reached.
        # Returns the newest accepted message, any polling error, and drained.
        newest = None
        error_message: Optional[str] = None
        timeout = first_timeout
        for _ in range(MAX_MESSAGES_PER_RUN):
            if stop_when_caught_up and not self._catch_up_until:
                return newest, error_message, True
            timeout = self._next_poll_timeout(timeout, catch_up_deadline)
            message = consumer.poll(timeout)
            timeout = 0
            if message is None:
                if self._catch_up_until:
                    continue
                return newest, error_message, True
            self._note_caught_up(message)
            if message.error() is not None:
                error_message = f"Kafka error while polling: {message.error()}"
                LOGGER.error("Kafka Consumer: %s", self._scrub(error_message))
                continue
            if not _message_key_matches(message, key_filter):
                continue
            newest = message

        return newest, error_message, False

    def _reposition_at_newest(self, consumer: Any, driver: Any, *, window: int) -> None:
        deadline = self._clock() + self._reposition_timeout
        positions, targets = [], {}
        for partition in sorted(self._partitions):
            position, target = self._start_position(
                consumer,
                driver,
                self._topic,
                partition,
                window=window,
                timeout=_time_remaining(
                    deadline,
                    f"locating the newest record on partition {partition}",
                    clock=self._clock,
                ),
            )
            positions.append(position)
            if target is not None:
                targets[partition] = target
        consumer.assign(positions)
        self._catch_up_until = targets

    def _read_next(
        self, poll_timeout: float, key_filter: Optional[str], driver: Any
    ) -> Dict[str, Any]:
        consumer = self._consumer
        try:
            if self._mode != READ_MODE_SEQUENTIAL:
                # First sequential run, or arriving from another positioning:
                # start at the newest record (for the key, when filtered).
                if self._mode != READ_MODE_LATEST or not self._catch_up_until:
                    self._reposition_at_newest(
                        consumer,
                        driver,
                        window=self._scan_window(key_filter, len(self._partitions)),
                    )
                self._mode = READ_MODE_SEQUENTIAL
            self._raise_on_auth_failure()
            catch_up_deadline = (
                self._clock() + self._reposition_timeout
                if self._catch_up_until
                else None
            )
            if self._catch_up_until:
                # Starting position: settle on the newest record among the seeds,
                # then continue one record per run from there.
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
            error_message = (
                pop_auth_failure_message(self._auth_failures) or error_message
            )
        except KafkaConfigurationError as error:
            return self._failure(str(error))
        except Exception as error:
            auth_message = pop_auth_failure_message(self._auth_failures)
            if auth_message is not None:
                return self._failure(auth_message)
            return self._failure(f"Failed to poll Kafka: {describe_error(error)}")

        if message is not None:
            return self._accept(message, error_message)

        outputs = _outputs(
            self._last,
            is_new=False,
            error_message=None if error_message is None else self._scrub(error_message),
        )

        return outputs

    def _drain_latest(
        self, poll_timeout: float, key_filter: Optional[str], driver: Any
    ) -> Dict[str, Any]:
        consumer = self._consumer
        try:
            if self._mode != READ_MODE_LATEST:
                # Leaving pointer mode (or never positioned): back to the newest
                # record on the topic, exactly like a first run.
                self._reposition_at_newest(
                    consumer,
                    driver,
                    window=self._scan_window(key_filter, len(self._partitions)),
                )
            self._mode = READ_MODE_LATEST
            self._raise_on_auth_failure()
            newest = None
            error_message: Optional[str] = None
            timeout = poll_timeout
            catch_up_deadline = (
                self._clock() + self._reposition_timeout
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
                # More than MAX_MESSAGES_PER_RUN records queued since the last run:
                # skip the backlog and read the newest record directly, so the
                # block never reports an old record as newest.
                LOGGER.warning(
                    "Kafka Consumer: over %s records queued since the last run on "
                    "topic %r; skipping to the newest record",
                    MAX_MESSAGES_PER_RUN,
                    self._topic,
                )
                self._reposition_at_newest(consumer, driver, window=1)
                catch_up_deadline = self._clock() + self._reposition_timeout
                timeout = 0
            error_message = (
                pop_auth_failure_message(self._auth_failures) or error_message
            )
        except KafkaConfigurationError as error:
            return self._failure(str(error))
        except Exception as error:
            auth_message = pop_auth_failure_message(self._auth_failures)
            if auth_message is not None:
                return self._failure(auth_message)
            return self._failure(f"Failed to poll Kafka: {describe_error(error)}")

        if newest is not None:
            return self._accept(newest, error_message)

        outputs = _outputs(
            self._last,
            is_new=False,
            error_message=None if error_message is None else self._scrub(error_message),
        )

        return outputs


# --- producer -----------------------------------------------------------------


@dataclass(frozen=True)
class KafkaPublish:
    """One validated record to publish; an immutable snapshot of the step.

    Args:
        connection: Where and how to connect.
        acks: ``"0"``, ``"1"`` or ``"all"``.
        value: Encoded record value.
        key: Encoded record key, or ``None``.
        headers: ``(name, value)`` pairs, or ``None``.
        timeout: Seconds for connecting and, when confirmed, for the report.
    """

    connection: KafkaConnection
    acks: str
    value: bytes
    key: Optional[bytes]
    headers: Optional[Tuple[Tuple[str, Optional[bytes]], ...]]
    timeout: float


class KafkaPublisherPort(Protocol):
    """Trusted host publisher borrowed by the block; implemented by KafkaPublisher.

    As with ``KafkaConsumerPort``, this implementation owns revealed plaintext
    and sanitizes its diagnostic outputs, logs, retained failures and exceptions.
    The block forwards the returned message and lets exceptions propagate; it
    supplies no automatic sanitizer for custom code or external library logs.
    Use ``reveal_secrets(...).scrub(...)`` before exposing diagnostic text and
    avoid raw exception tracebacks/chaining in sanitized failures. Preserve the
    request's encoded value/key/headers exactly; never scrub record data.

    Ordinary broker/configuration/delivery failures follow the supplied owner's
    convention and return ``(True, scrubbed_message)``. Closed/retired host
    ownership raises ``KafkaHostError``; unavailable credentials raise
    ``SecretUnavailableError`` before external acquisition. These intentionally
    fail the step; unexpected implementation exceptions also propagate.
    Shared callers must be safe through implementation locking or host
    serialization; the block adds no lock. Only ``publish`` is required here,
    with lifecycle and any additional observability owned by the host.
    """

    def publish(self, request: KafkaPublish, *, wait: bool) -> Tuple[bool, str]:
        """Queue one record and, with ``wait``, wait for its delivery report.

        Args:
            request: The validated record.
            wait: Wait up to ``request.timeout`` for the report.

        Returns:
            ``(error_status, message)`` with sanitized diagnostic text. False
            means queued when ``wait=False``, or confirmed when ``wait=True``.
            A timeout may still deliver and must not cause an automatic replay.

        Raises:
            KafkaHostError: For closed/retired host ownership.
            SecretUnavailableError: Before acquisition if a credential is
                unavailable.
        """


@dataclass(frozen=True)
class DeliveryFailure:
    """One record the broker did not take.

    Args:
        topic: Destination topic.
        message: The scrubbed reason.
        background: The step did not wait for this record.
    """

    topic: str
    message: str
    background: bool


@dataclass(frozen=True)
class DeliveryStats:
    """Counts of a ``KafkaPublisher``.

    Args:
        queued: Records the client accepted.
        delivered: Records with a successful delivery report.
        failed: Records with a failed delivery report.
        pending: Queued records without a report yet.
        unknown_at_close: Records still without a report when ``close``
            finished; they may or may not have been delivered.
        late_reports: Reports that arrived after their record was settled as
            unknown; not counted as delivered or failed.
        clients_created: Kafka clients constructed.
        clients_released: Kafka clients the publisher dropped without a close
            call (no verified producer close exists); see the module notes.
    """

    queued: int
    delivered: int
    failed: int
    pending: int
    unknown_at_close: int
    late_reports: int
    clients_created: int
    clients_released: int


def _logged_failure(message: str, *, scrub: Callable[[str], str]) -> Tuple[bool, str]:
    # V1 logged every failed run at ERROR.
    message = scrub(message)
    LOGGER.error("Kafka Producer failure: %s", message)
    return True, message


class _Report(NamedTuple):
    # Terminal status of one record, built whole and then published in one
    # assignment: a waiting publish reads it without the ledger lock, so it
    # never sees a state without its partition, offset or scrubbed reason.
    state: Literal["delivered", "failed", "unknown"]
    partition: Optional[int] = None
    offset: Optional[int] = None
    error: Optional[str] = None


class _Delivery:
    """``on_delivery`` callback of one queued record; settles once."""

    def __init__(
        self,
        ledger: "_DeliveryLedger",
        *,
        topic: str,
        background: bool,
        scrub: Callable[[str], str],
        on_success: Callable[[], None],
    ) -> None:
        self.topic = topic
        self.background = background
        self.scrub = scrub
        self.on_success = on_success
        # None while pending.
        self.report: Optional[_Report] = None
        self._ledger = ledger

    @property
    def settled(self) -> bool:
        return self.report is not None

    def __call__(self, error: Any, message: Any) -> None:
        try:
            self._ledger.settle(self, error=error, message=message)
        except Exception:  # A callback raising inside librdkafka helps nobody.
            # Driver exceptions may contain credentials, including in tracebacks.
            LOGGER.error("Kafka Producer: recording a delivery report failed")


class _DeliveryLedger:
    # Terminal status of every queued record; callbacks run on polling threads.

    def __init__(self, *, kept_failures: int) -> None:
        self._lock = threading.Lock()
        self._pending: Set[_Delivery] = set()
        self._failures: Deque[DeliveryFailure] = deque(maxlen=kept_failures)
        self._counts = dict.fromkeys(
            (
                "queued",
                "delivered",
                "failed",
                "unknown_at_close",
                "late_reports",
                "clients_created",
                "clients_released",
            ),
            0,
        )

    def register(self, delivery: _Delivery) -> None:
        with self._lock:
            self._pending.add(delivery)
            self._counts["queued"] += 1

    def withdraw(self, delivery: _Delivery) -> None:
        # produce() raised: the client never queued the record.
        with self._lock:
            if delivery in self._pending:
                self._pending.discard(delivery)
                self._counts["queued"] -= 1

    def settle(self, delivery: _Delivery, *, error: Any, message: Any) -> None:
        # The report is complete before anything is published. The message
        # accessors are driver calls, so they run outside the lock.
        if error is not None:
            report = _Report("failed", error=delivery.scrub(str(error)))
        elif message is not None:
            report = _Report(
                "delivered", partition=message.partition(), offset=message.offset()
            )
        else:
            report = _Report("delivered")
        with self._lock:
            if delivery.settled:
                self._counts["late_reports"] += 1
                return
            self._pending.discard(delivery)
            delivery.report = report
            if report.state == "delivered":
                self._counts["delivered"] += 1
            else:
                self._counts["failed"] += 1
                self._failures.append(
                    DeliveryFailure(
                        topic=delivery.topic,
                        message=report.error,
                        background=delivery.background,
                    )
                )
        if error is None:
            # The broker took a record: proof of life.
            delivery.on_success()
        elif delivery.background:
            # Fire-and-forget: nothing to return to; V1 logged it.
            LOGGER.error(
                "Kafka Producer: delivery to topic %r failed: %s",
                delivery.topic,
                report.error,
            )

    def abandon_pending(self) -> None:
        with self._lock:
            for delivery in self._pending:
                delivery.report = _Report("unknown")
            self._counts["unknown_at_close"] += len(self._pending)
            self._pending.clear()

    def count(self, name: str) -> None:
        with self._lock:
            self._counts[name] += 1

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def stats(self) -> DeliveryStats:
        with self._lock:
            stats = DeliveryStats(pending=len(self._pending), **self._counts)

        return stats

    def failures(self) -> Tuple[DeliveryFailure, ...]:
        with self._lock:
            failures = tuple(self._failures)

        return failures


class KafkaPublisher:
    """Host-owned Kafka producer with explicit admission, drain and close.

    The first ``publish`` connects with that run's settings and checks the
    topic; later calls must use the same settings and ``acks`` (V1's identity
    check). Calls from several steps, sessions and threads may run at once.

    Every queued record ends in one recorded state: delivered, failed, or
    ``unknown_at_close``. Reports are served on later ``publish``, ``flush``
    and ``close`` calls; there is no background thread. The host closes the
    publisher after every borrowing session stopped.

    Args:
        driver: The ``confluent_kafka`` module or a stand-in providing
            ``Producer`` and ``KafkaError``; imported at first connect when
            ``None``.
        msk_token_signer: ``signer(region) -> (token, expiry_ms)`` for the
            AWS MSK provider; the default uses aws-msk-iam-sasl-signer and
            the host's AWS credential chain.
        kept_failures: Most recent delivery failures kept for ``failures()``.
        clock: Monotonic clock for deadlines and problem ages.
        sleep: Waits between metadata retries.

    Raises:
        ValueError: When ``kept_failures`` is lower than 1.
    """

    def __init__(
        self,
        *,
        driver: Any = None,
        msk_token_signer: Optional[Callable[[str], Tuple[str, float]]] = None,
        kept_failures: int = 100,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        valid = not isinstance(kept_failures, bool) and isinstance(kept_failures, int)
        if not valid or kept_failures < 1:
            raise ValueError(
                f"kept_failures must be an integer >= 1, got {kept_failures!r}"
            )

        self._driver = driver
        self._signer = msk_token_signer
        self._clock = clock
        self._sleep = sleep
        self._salt = os.urandom(16)
        self._ledger = _DeliveryLedger(kept_failures=kept_failures)
        # The connected producer's error_cb; None while no producer is connected.
        self._client_errors: Optional[ClientErrorTracker] = None
        # Admission: running calls are counted; close waits for them.
        self._admission = threading.Condition()
        self._active = 0
        self._closing = False
        # Connection: one thread connects; close releases under the same lock.
        self._connect_lock = threading.Lock()
        self._client: Optional[Any] = None
        self._identity: Optional[Tuple[Any, ...]] = None
        self._bound_password: Optional[Secret] = None
        self._auth_failures: List[BaseException] = []
        self._close_stats: Optional[DeliveryStats] = None

    @property
    def closed(self) -> bool:
        """Whether ``close`` was called; later publishes are rejected."""
        with self._admission:
            return self._closing

    def publish(self, request: KafkaPublish, *, wait: bool) -> Tuple[bool, str]:
        """Queue one record; with ``wait``, wait for its own delivery report.

        Args:
            request: The validated record.
            wait: Wait up to ``request.timeout`` for the report.

        Returns:
            ``(error_status, message)``, with V1's messages, scrubbed.

        Raises:
            KafkaHostError: When the publisher is closed or closing.
            SecretUnavailableError: When the password has no value here.
        """
        self._enter()
        try:
            scrub = _scrubber(request.connection.password, self._bound_password)
            error_status, message = self._publish(request, wait=wait, scrub=scrub)
        finally:
            self._leave()
        outcome = (error_status, scrub(message))

        return outcome

    def flush(self, timeout: float) -> DeliveryStats:
        """Serve delivery reports until none is pending or ``timeout`` passes.

        Draining only: the client stays open. Once ``close`` has started,
        ``flush`` returns the current counts at once without draining and
        without waiting for ``close``. While ``close`` is still draining these
        counts are provisional: records may still be pending, and ``close``
        settles them. Only ``close``'s return value is the final snapshot.

        Args:
            timeout: Most seconds to wait.

        Returns:
            The counts afterwards; provisional while ``close`` is draining.
        """
        with self._admission:
            if self._closing:
                return self.stats()
            self._active += 1
        try:
            self._flush_client(timeout)
        finally:
            self._leave()
        stats = self.stats()

        return stats

    def close(self, timeout: float = 5.0) -> DeliveryStats:
        """Stop admission, drain, settle every record and drop the client.

        Waits for running ``publish`` and ``flush`` calls (each bounded by its
        own timeout), flushes for up to ``timeout``, then counts each record
        still without a report as ``unknown_at_close``. The client is then
        dropped: this port calls no producer close method, because none was
        verified. Idempotent; later calls return the same cached snapshot.
        ``stats()`` stays live, so a report arriving after close shows there
        as ``late_reports``.

        Args:
            timeout: Most seconds the final flush waits.

        Returns:
            The final counts, as a snapshot taken when close completed.
        """
        with self._admission:
            self._closing = True
            self._admission.wait_for(lambda: self._active == 0)
        with self._connect_lock:
            if self._close_stats is not None:
                return self._close_stats

            self._flush_client(timeout)
            self._ledger.abandon_pending()
            if self._client is not None:
                self._ledger.count("clients_released")
            self._client = None
            # The dropped producer keeps its own tracker as error_cb, scrubber
            # included, for as long as it may still call it.
            self._client_errors = None
            self._identity = None
            self._bound_password = None
            self._close_stats = self._ledger.stats()

        return self._close_stats

    def stats(self) -> DeliveryStats:
        """Return the current counts."""
        stats = self._ledger.stats()

        return stats

    def failures(self) -> Tuple[DeliveryFailure, ...]:
        """Return the kept delivery failures, oldest first, scrubbed."""
        failures = self._ledger.failures()

        return failures

    def _enter(self) -> None:
        with self._admission:
            if self._closing:
                raise KafkaHostError(CLOSED_PUBLISHER_MESSAGE)
            self._active += 1

    def _leave(self) -> None:
        with self._admission:
            self._active -= 1
            self._admission.notify_all()

    def _flush_client(self, timeout: float) -> None:
        client = self._client
        if client is None:
            return

        try:
            client.flush(timeout)
        except Exception as error:
            scrub = _scrubber(self._bound_password)
            LOGGER.error(
                "Failed to flush Kafka producer: %s", scrub(describe_error(error))
            )

    def _publish(
        self, request: KafkaPublish, *, wait: bool, scrub: Callable[[str], str]
    ) -> Tuple[bool, str]:
        try:
            client = self._ensure_producer(request)
        except KafkaConfigurationError as error:
            return _logged_failure(str(error), scrub=scrub)
        except Exception as error:
            if self._is_invalid_configuration(error):
                return _logged_failure(
                    f"Invalid Kafka client configuration: {describe_error(error)}",
                    scrub=scrub,
                )
            return _logged_failure(
                f"Kafka broker not reachable ({describe_error(error)}). Raise "
                "'timeout' if the broker needs longer to connect.",
                scrub=scrub,
            )

        outcome = self._produce(client, request, wait=wait, scrub=scrub)
        if outcome[0]:
            outcome = _logged_failure(outcome[1], scrub=scrub)
        # The tracker logs a client problem once; it is not logged per run.
        reported = self._report_client_problem(
            outcome, client, request.connection.topic, background=not wait
        )

        return reported

    def _load_driver(self) -> Any:
        if self._driver is None:
            self._driver = import_confluent_kafka(
                missing_message=(
                    "Kafka Producer requires the confluent-kafka package in the runtime."
                )
            )

        return self._driver

    def _is_invalid_configuration(self, error: BaseException) -> bool:
        # librdkafka reports bad config values as KafkaException(_INVALID_ARG)
        # from the constructor: the user's settings, not an unreachable broker.
        code = getattr(getattr(error, "args", [None])[0], "code", None)
        invalid_arg = getattr(
            getattr(self._driver, "KafkaError", None), "_INVALID_ARG", object()
        )
        try:
            return callable(code) and code() == invalid_arg
        except Exception:
            return False

    def _ensure_producer(self, request: KafkaPublish) -> Any:
        connection = request.connection
        identity = _identity(connection, salt=self._salt, extra=(request.acks,))
        with self._connect_lock:
            if self._client is not None:
                if identity != self._identity:
                    raise KafkaConfigurationError(
                        "Kafka connection parameters (bootstrap servers, topic, provider, "
                        "credentials or acks) changed between runs; this block publishes "
                        "only to the connection configured on its first run."
                    )
                return self._client

            driver = self._load_driver()
            auth_failures: List[BaseException] = []
            self._auth_failures = auth_failures
            settings = _client_settings(
                connection, auth_failures=auth_failures, signer=self._signer
            )
            deadline = self._clock() + request.timeout
            try:
                _preflight_token(settings)
            except Exception:
                self._raise_on_auth_failure()
                raise

            # A new client starts clean with its own tracker: problems of a
            # previous client are not its own, and its scrubber stays with it.
            client_errors = ClientErrorTracker(
                "Kafka Producer",
                scrub=_scrubber(connection.password),
                clock=self._clock,
            )
            config = {
                "bootstrap.servers": connection.bootstrap_servers,
                "acks": request.acks,
                # Global client errors arrive only here, served from inside
                # poll() / flush(); produce() keeps succeeding.
                "error_cb": client_errors,
                # message.timeout.ms stays at librdkafka's default (300s): records
                # survive broker hiccups; `timeout` bounds only connecting and,
                # when confirmed, waiting for the report.
                "log_level": LIBRDKAFKA_LOG_LEVEL,
                **settings,
            }
            client = driver.Producer(config)
            self._ledger.count("clients_created")
            try:
                # Fail fast on an unreachable broker or a missing topic, but ride
                # out the replies a broker gives while auto-creating one.
                wait_for_topic_metadata(
                    client,
                    connection.topic,
                    deadline=deadline,
                    clock=self._clock,
                    sleep=self._sleep,
                )
                self._raise_on_auth_failure()
            except Exception:
                # Partial construction: nothing was queued in this client; it is
                # dropped without a verified close (see the module notes).
                self._ledger.count("clients_released")
                self._raise_on_auth_failure()
                raise

            self._client = client
            self._client_errors = client_errors
            self._identity = identity
            self._bound_password = connection.password

        return client

    def _raise_on_auth_failure(self) -> None:
        message = pop_auth_failure_message(self._auth_failures)
        if message is not None:
            raise KafkaConfigurationError(message)

    def _produce(
        self,
        client: Any,
        request: KafkaPublish,
        *,
        wait: bool,
        scrub: Callable[[str], str],
    ) -> Tuple[bool, str]:
        topic = request.connection.topic
        delivery = _Delivery(
            self._ledger,
            topic=topic,
            background=not wait,
            scrub=scrub,
            on_success=self._client_errors.clear,
        )
        headers = None if request.headers is None else list(request.headers)
        self._ledger.register(delivery)
        try:
            try:
                client.produce(
                    topic,
                    value=request.value,
                    key=request.key,
                    headers=headers,
                    on_delivery=delivery,
                )
            except BufferError:
                # A record holds its queue slot until its report is served, so
                # poll first (the librdkafka idiom) and retry once. BufferError
                # means nothing was queued, so this never duplicates a record.
                client.poll(0)
                client.produce(
                    topic,
                    value=request.value,
                    key=request.key,
                    headers=headers,
                    on_delivery=delivery,
                )
        except BufferError:
            self._ledger.withdraw(delivery)
            return True, (
                "Kafka producer's local queue is full; the broker is slower than the "
                "stream. Lower the frame rate or use fire_and_forget=false."
            )
        except Exception as error:
            self._ledger.withdraw(delivery)
            return True, f"Failed to publish message: {describe_error(error)}"

        try:
            # Serves reports of earlier records (the OAuth refresh runs on
            # librdkafka's own thread and needs no poll).
            client.poll(0)
            if not wait:
                auth_message = pop_auth_failure_message(self._auth_failures)
                if auth_message is not None:
                    return True, auth_message
                return False, SCHEDULED_MESSAGE

            deadline = self._clock() + request.timeout
            while not delivery.settled:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    break
                client.poll(min(remaining, DELIVERY_POLL_SLICE))
        except Exception as error:
            # The record is queued and stays pending: its later report is still
            # counted. It is never produced again here.
            uncertain = (
                f"Message queued, but serving delivery reports failed "
                f"({describe_error(error)}); delivery status unknown and the message "
                "may still be delivered, so publishing it again may duplicate it."
            )
            auth_message = pop_auth_failure_message(self._auth_failures)
            if auth_message is not None:
                return True, f"{uncertain}; {auth_message}"
            return True, uncertain

        outcome = self._confirmed_outcome(delivery, timeout=request.timeout)

        return outcome

    def _confirmed_outcome(
        self, delivery: _Delivery, *, timeout: float
    ) -> Tuple[bool, str]:
        auth_message = pop_auth_failure_message(self._auth_failures)
        # Read once: either None (still pending) or a complete report.
        report = delivery.report
        if report is not None and report.state == "failed":
            rejected = f"Kafka rejected the message: {report.error}"
            if auth_message is not None:
                rejected = f"{rejected}; {auth_message}"
            return True, rejected

        if report is not None and report.state == "delivered":
            if report.offset is None:
                # acks=0: the broker sends no acknowledgement, so no offset is known.
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
                # The record went out, but a token refresh failed meanwhile.
                return True, f"{delivered}; {auth_message}"
            return False, delivered

        if auth_message is not None:
            return True, auth_message

        unknown = (
            f"Delivery not acknowledged within {timeout}s "
            f"({self._ledger.pending_count()} record(s) still queued); delivery "
            "status unknown and the message may still be delivered."
        )

        return True, unknown

    def _report_client_problem(
        self, outcome: Tuple[bool, str], client: Any, topic: str, *, background: bool
    ) -> Tuple[bool, str]:
        # Add the broker problem the client reported, if one is still recorded.
        # Successful delivery reports have already cleared it by now.
        problem = self._client_errors.check(client, topic)
        if problem is None:
            return outcome

        error_status, message = outcome
        if error_status or problem.fatal:
            note = None
        elif background:
            note = (
                "The record is queued in the client, which retries delivery for up to "
                "five minutes, but the broker is currently unreachable."
            )
        else:
            note = "This record was acknowledged before the problem was reported."
        # A success text is replaced, a failure text kept; the tracker logged the
        # problem once, so nothing is logged per frame here.
        combined = combine_messages(
            message if error_status else None,
            self._client_errors.problem_message(problem, note),
        )

        return True, combined


# --- value encoding -----------------------------------------------------------


def _encode_text(text: str, what: str) -> bytes:
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        # Lone UTF-16 surrogates have no UTF-8 form; report rather than alter.
        raise KafkaConfigurationError(
            f"{what} contains characters that cannot be encoded as UTF-8."
        ) from None


def encode_value(message: Any) -> bytes:
    """Encode a record value as V1 did.

    Args:
        message: A string (UTF-8), a dictionary or list (JSON), or bytes.

    Returns:
        The encoded value.

    Raises:
        KafkaConfigurationError: For another type, unencodable text or a
            value JSON cannot encode.
    """
    if isinstance(message, (bytes, bytearray)):
        return bytes(message)

    if isinstance(message, str):
        return _encode_text(message, "message")

    if isinstance(message, (dict, list)):
        try:
            text = json.dumps(message)
        except (TypeError, ValueError) as error:
            raise KafkaConfigurationError(
                f"message could not be JSON-encoded: {describe_error(error)}"
            ) from None
        return text.encode("utf-8")

    raise KafkaConfigurationError(
        f"message must be a string or a dictionary, got {type(message).__name__}."
    )


def encode_key(key: Any) -> Optional[bytes]:
    """Encode a record key as V1 did.

    Args:
        key: ``None``, bytes or text.

    Returns:
        The encoded key, or ``None``.

    Raises:
        KafkaConfigurationError: For text without a UTF-8 form.
    """
    if key is None:
        return None

    if isinstance(key, (bytes, bytearray)):
        return bytes(key)

    encoded = _encode_text(str(key), "key")

    return encoded


def encode_headers(headers: Any) -> Optional[Tuple[Tuple[str, Optional[bytes]], ...]]:
    """Encode record headers as V1 did.

    Args:
        headers: ``None`` or a dictionary; values may be ``None``, bytes or text.

    Returns:
        ``(name, value)`` pairs, or ``None``.

    Raises:
        KafkaConfigurationError: For a non-dictionary or unencodable text.
    """
    if headers is None:
        return None

    if not isinstance(headers, dict):
        raise KafkaConfigurationError(
            f"headers must be a dictionary of strings, got {type(headers).__name__}."
        )

    encoded: List[Tuple[str, Optional[bytes]]] = []
    for name, value in headers.items():
        if value is None:
            encoded.append((str(name), None))
        elif isinstance(value, (bytes, bytearray)):
            encoded.append((str(name), bytes(value)))
        else:
            encoded.append((str(name), _encode_text(str(value), f"header {name!r}")))

    return tuple(encoded)


def coerce_acks(value: Any) -> str:
    """Normalize ``acks`` as V1 did.

    Args:
        value: ``"0"``, ``"1"``, ``"all"`` or the integers 0 and 1.

    Returns:
        ``"0"``, ``"1"`` or ``"all"``.

    Raises:
        KafkaConfigurationError: For any other value.
    """
    if isinstance(value, bool):
        raise KafkaConfigurationError("acks must be one of 0, 1, all.")

    if isinstance(value, (int, float)) and float(value).is_integer():
        value = str(int(value))
    value = str(value).strip()
    if value not in ACKS_VALUES:
        raise KafkaConfigurationError("acks must be one of 0, 1, all.")

    return value


# --- blocks -------------------------------------------------------------------


def _is_selector(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("$")


def _check_timeout(value: Any, name: str, *, allow_zero: bool) -> None:
    # V1's timeout rule; also applied to selected values (the engine reruns it).
    if _is_selector(value):
        return

    seconds = float(value)
    lower_ok = seconds >= 0 if allow_zero else seconds > 0
    if not math.isfinite(seconds) or not lower_ok or seconds > MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"Invalid {name}: {value!r}. It must be a finite number of seconds up to "
            f"{MAX_TIMEOUT_SECONDS}."
        )


def _check_non_negative(value: Any, name: str) -> None:
    if value is None or _is_selector(value):
        return

    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _hosted_platform() -> bool:
    return bool(environment.GCP_SERVERLESS or environment.LAMBDA)


BOOTSTRAP_SERVERS_DESCRIPTION = (
    "Comma-separated list of Kafka broker addresses (`host:port`). For AWS MSK use "
    "the bootstrap string from the console. The server operator can restrict this "
    "value to an allowlist with `KAFKA_WORKFLOWS_SINKS_WHITELISTED_BOOTSTRAP_SERVERS`, "
    "or replace it with that list by setting "
    "`KAFKA_WORKFLOWS_SINKS_ALLOW_USER_PROVIDED_BOOTSTRAP_SERVERS=False`."
)
PROVIDER_DESCRIPTION = (
    "Where the cluster lives. **Self-hosted**: any Kafka you operate, optionally "
    "with username/password (SASL/SCRAM-SHA-512 over TLS). **AWS MSK**: Amazon MSK "
    "with IAM access control, signed in with the AWS identity on the machine running "
    "inference; no credentials in the workflow."
)
USERNAME_DESCRIPTION = (
    "SASL username. Leave empty for an unauthenticated plaintext listener. Must be "
    "set together with `password`."
)
PASSWORD_DESCRIPTION = (
    "SASL password. Prefer a secret from a secrets store step or a workflow input "
    "of kind secret. Must be set together with `username`."
)
AWS_REGION_DESCRIPTION = (
    "AWS region of the MSK cluster, used to sign the IAM token. Leave empty to "
    "derive it from the broker address (`*.kafka.<region>.amazonaws.com`)."
)
SSL_CA_LOCATION_DESCRIPTION = (
    "Path to a CA certificate bundle on the machine running inference, for TLS "
    "connections whose issuer is not in the system trust store. Requires the host "
    "resource `allow_access_to_file_system=True`; otherwise the run reports an error."
)


class KafkaConsumerBlock(Block):
    """Read one message from a Kafka topic on every run, through a borrowed cursor.

    ``read_mode="latest"`` returns the newest record and repeats it with
    ``is_new=False`` when nothing new arrived; ``"sequential"`` returns every
    record in order, one per run. ``offset``/``partition`` point at a record:
    in ``latest`` mode exactly that record, in ``sequential`` mode the start.
    ``key_filter`` keeps only records with that key, except a pointed one. The
    position lives in the host's ``kafka_consumer`` cursor, never in the block.

    Args:
        kafka_consumer: Host-owned cursor; required, never closed here.
        allow_access_to_file_system: Host permission to read
            ``ssl_ca_location``.
    """

    type = CONSUMER_BLOCK_TYPE
    metadata = {
        "name": "Kafka Consumer",
        "port_of": "roboflow_enterprise/kafka_consumer@v1",
        "section": "data_storage",
        "license": "Roboflow Enterprise License",
        "enterprise_only": True,
        "local_only": True,
    }
    outputs = {
        "value": Output(
            STRING_KIND,
            description="Raw value as UTF-8 text; None before any record and for a tombstone.",
        ),
        "payload": Output(
            DICTIONARY_KIND,
            description="The value parsed as a JSON object, or {} when it is not one.",
        ),
        "key": Output(STRING_KIND, description="The record key, or None."),
        "is_new": Output(
            BOOLEAN_KIND,
            description="True only on the run that first returned this record.",
        ),
        "offset": Output(INTEGER_KIND, description="Offset of the record, or None."),
        "partition": Output(
            INTEGER_KIND, description="Partition of the record, or None."
        ),
        "error_status": Output(
            BOOLEAN_KIND,
            description="True when a broker, credential or input problem was reported.",
        ),
        "error_message": Output(STRING_KIND, description="What went wrong, or None."),
    }

    class Params(BlockParams):
        bootstrap_servers: StrictStr | Ref(STRING_KIND) = Field(
            description=BOOTSTRAP_SERVERS_DESCRIPTION,
            examples=["localhost:9092", "$inputs.kafka_bootstrap"],
        )
        topic: StrictStr | Ref(STRING_KIND) = Field(
            description="Topic to read from.", examples=["line1.state"]
        )
        provider: Literal["Self-hosted", "AWS MSK"] = Field(
            default=PROVIDER_SELF_HOSTED,
            description=PROVIDER_DESCRIPTION,
            examples=[PROVIDER_AWS_MSK],
        )
        username: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description=USERNAME_DESCRIPTION,
            examples=["inference"],
            json_schema_extra={"relevant_for": SELF_HOSTED_RELEVANT},
        )
        password: Optional[StrictStr] | Ref(
            SECRET_KIND, STRING_KIND, allow_none=True
        ) = Field(
            default=None,
            description=PASSWORD_DESCRIPTION,
            examples=["$steps.secrets.kafka_password"],
            json_schema_extra={"private": True, "relevant_for": SELF_HOSTED_RELEVANT},
        )
        aws_region: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description=AWS_REGION_DESCRIPTION,
            examples=["us-east-1"],
            json_schema_extra={"relevant_for": AWS_MSK_RELEVANT},
        )
        read_mode: Literal["latest", "sequential"] = Field(
            default=READ_MODE_LATEST,
            description=(
                "**latest**: each run returns the newest message, skipping anything "
                "older. **sequential**: each run returns the next unread message in "
                "order, one per run."
            ),
            examples=[READ_MODE_SEQUENTIAL],
        )
        key_filter: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description=(
                "Consider only messages whose key equals this value. Not applied to "
                "a record read directly by `offset`."
            ),
            examples=["cam-1"],
        )
        offset: Optional[StrictInt] | Ref(INTEGER_KIND, allow_none=True) = Field(
            default=None,
            description=(
                "Pointer to a record. In `latest` mode the block reads exactly that "
                "record; in `sequential` mode reading starts there. Leave empty to "
                "start from the newest message."
            ),
            examples=[42, "$inputs.state_offset"],
        )
        partition: StrictInt | Ref(INTEGER_KIND) = Field(
            default=0,
            description="Partition the pointer refers to; only used with `offset`.",
            examples=[0],
        )
        ssl_ca_location: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = (
            Field(
                default=None,
                description=SSL_CA_LOCATION_DESCRIPTION,
                examples=["/etc/ssl/certs/ca-certificates.crt"],
            )
        )
        poll_timeout: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=0.01,
            description=(
                "Seconds to wait for a message on each run; the client prefetches "
                "in the background, so a small value is enough once connected."
            ),
            examples=[0.01],
        )
        connect_timeout: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=5.0,
            description=(
                "Seconds for connecting and locating the topic on the first run, "
                "and for fetching a pointed record."
            ),
            examples=[5.0],
        )

        # The engine reruns these validators on selected values.
        @field_validator("offset", "partition")
        @classmethod
        def _non_negative(cls, value: Any, info: Any) -> Any:
            _check_non_negative(value, info.field_name)
            return value

        @field_validator("poll_timeout")
        @classmethod
        def _poll_timeout(cls, value: Any) -> Any:
            _check_timeout(value, "poll_timeout", allow_zero=True)
            return value

        @field_validator("connect_timeout")
        @classmethod
        def _connect_timeout(cls, value: Any) -> Any:
            _check_timeout(value, "connect_timeout", allow_zero=False)
            return value

    def __init__(
        self,
        *,
        kafka_consumer: Optional[KafkaConsumerPort] = None,
        allow_access_to_file_system: bool = False,
    ) -> None:
        self._cursor = kafka_consumer
        self._allow_access_to_file_system = allow_access_to_file_system
        self._bootstrap_override_logged = False

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> List[WorkOperation]:
        """Declare broker reads and the position kept across runs.

        Args:
            params: Compiled parameters.

        Returns:
            External request and temporal buffering, as in V1.
        """
        return [WorkOperation.EXTERNAL_REQUEST, WorkOperation.TEMPORAL_BUFFERING]

    @classmethod
    def discover_restrictions(cls, params: BlockParams) -> List[RuntimeRestriction]:
        """Declare V1's hosted-platform and per-request restrictions.

        Args:
            params: Compiled parameters.

        Returns:
            Both restrictions; no field switches either.
        """
        return [CONSUMER_HOSTED_RESTRICTION, CONSUMER_PER_REQUEST_RESTRICTION]

    @classmethod
    def discover_dependent_resources(cls, params: BlockParams) -> List[Any]:
        """Declare no dependent resources, as V1 did.

        Args:
            params: Compiled parameters.

        Returns:
            An empty list.
        """
        return []

    def run(
        self,
        *,
        bootstrap_servers: str,
        topic: str,
        provider: str = PROVIDER_SELF_HOSTED,
        username: Optional[str] = None,
        password: Optional[Union[str, Secret]] = None,
        aws_region: Optional[str] = None,
        read_mode: str = READ_MODE_LATEST,
        key_filter: Optional[str] = None,
        offset: Optional[int] = None,
        partition: int = 0,
        ssl_ca_location: Optional[str] = None,
        poll_timeout: float = 0.01,
        connect_timeout: float = 5.0,
    ) -> Dict[str, Any]:
        """Read one record through the borrowed cursor.

        Args:
            bootstrap_servers: Broker list.
            topic: Topic name.
            provider: ``"Self-hosted"`` or ``"AWS MSK"``.
            username: SASL username or ``None``.
            password: SASL password or ``None``.
            aws_region: MSK region or ``None``.
            read_mode: ``"latest"`` or ``"sequential"``.
            key_filter: Key to match, or ``None``.
            offset: Pointed offset, or ``None``.
            partition: Partition of the pointer.
            ssl_ca_location: CA bundle path, or ``None``.
            poll_timeout: Seconds to wait for a record.
            connect_timeout: Seconds to connect and fetch a pointed record.

        Returns:
            ``value``, ``payload``, ``key``, ``is_new``, ``offset``,
            ``partition``, ``error_status`` and ``error_message``.

        Raises:
            KafkaHostError: Without a cursor, or with a closed one.
            SecretUnavailableError: When the password has no value here.
        """
        if _hosted_platform():
            LOGGER.error("Kafka Consumer failure: %s", CONSUMER_HOSTED_MESSAGE)
            return _outputs(None, error_message=CONSUMER_HOSTED_MESSAGE)

        if self._cursor is None:
            raise KafkaHostError(NO_CURSOR_MESSAGE)

        secret = to_secret(password, name="password")
        scrub = reveal_secrets(secret).scrub
        try:
            connection = kafka_connection(
                bootstrap_servers,
                topic=topic,
                provider=provider,
                username=username,
                password=secret,
                aws_region=aws_region,
                ssl_ca_location=ssl_ca_location,
                allow_access_to_file_system=self._allow_access_to_file_system,
                log_override=not self._bootstrap_override_logged,
            )
        except KafkaConfigurationError as error:
            return self._cursor.repeat_last(error_message=scrub(str(error)))
        finally:
            self._bootstrap_override_logged = True

        request = KafkaRead(
            connection=connection,
            read_mode=read_mode,
            key_filter=key_filter,
            pointer=None if offset is None else (partition, offset),
            poll_timeout=float(poll_timeout),
            connect_timeout=float(connect_timeout),
        )
        outputs = self._cursor.read(request)

        return outputs


class KafkaProducerSinkBlock(Block):
    """Publish one message to a Kafka topic on every run, through a borrowed publisher.

    A dictionary or list ``message`` is JSON-encoded, a string sent as UTF-8.
    ``key`` keeps records with the same key ordered on one partition;
    ``headers`` are string metadata. ``fire_and_forget=True`` returns once the
    record is queued; ``False`` waits up to ``timeout`` for the broker's
    report. Delivery is owned by the host's ``kafka_publisher``.

    Args:
        kafka_publisher: Host-owned publisher; required when enabled, never
            closed here.
        disable_sinks: Host execution policy: skip every publish.
        allow_access_to_file_system: Host permission to read
            ``ssl_ca_location``.
    """

    type = PRODUCER_BLOCK_TYPE
    metadata = {
        "name": "Kafka Producer",
        "port_of": "roboflow_enterprise/kafka_producer_sink@v1",
        "section": "data_storage",
        "license": "Roboflow Enterprise License",
        "enterprise_only": True,
        "local_only": True,
    }
    outputs = {"error_status": ERROR_STATUS_OUTPUT, "message": MESSAGE_OUTPUT}

    class Params(BlockParams):
        bootstrap_servers: StrictStr | Ref(STRING_KIND) = Field(
            description=BOOTSTRAP_SERVERS_DESCRIPTION,
            examples=["localhost:9092", "$inputs.kafka_bootstrap"],
        )
        topic: StrictStr | Ref(STRING_KIND) = Field(
            description=(
                "Topic to publish to. It must already exist; on brokers with topic "
                "auto-creation a misspelled name silently creates a new topic."
            ),
            examples=["vision.detections"],
        )
        message: (
            StrictStr
            | Dict[str, Any]
            | Ref(STRING_KIND, DICTIONARY_KIND, LIST_OF_VALUES_KIND)
        ) = Field(
            description=(
                "Record value. A dictionary or list is JSON-encoded; a string is "
                "sent as UTF-8 text."
            ),
            examples=["$steps.payload.output", '{"state": "RUNNING"}'],
        )
        provider: Literal["Self-hosted", "AWS MSK"] = Field(
            default=PROVIDER_SELF_HOSTED,
            description=PROVIDER_DESCRIPTION,
            examples=[PROVIDER_AWS_MSK],
        )
        username: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description=USERNAME_DESCRIPTION,
            examples=["inference"],
            json_schema_extra={"relevant_for": SELF_HOSTED_RELEVANT},
        )
        password: Optional[StrictStr] | Ref(
            SECRET_KIND, STRING_KIND, allow_none=True
        ) = Field(
            default=None,
            description=PASSWORD_DESCRIPTION,
            examples=["$steps.secrets.kafka_password"],
            json_schema_extra={"private": True, "relevant_for": SELF_HOSTED_RELEVANT},
        )
        aws_region: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description=AWS_REGION_DESCRIPTION,
            examples=["us-east-1"],
            json_schema_extra={"relevant_for": AWS_MSK_RELEVANT},
        )
        key: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description=(
                "Record key. Records with the same key stay in order on one "
                "partition; use the camera id so each stream stays ordered."
            ),
            examples=["cam-1"],
        )
        headers: Optional[Dict[StrictStr, StrictStr]] | Ref(
            DICTIONARY_KIND, allow_none=True
        ) = Field(
            default=None,
            description="Optional string-to-string record headers.",
            examples=[{"source": "inference", "schema": "v1"}],
        )
        fire_and_forget: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "`true`: return once the record is queued; delivery failures are "
                "kept by the host's publisher. `false`: wait up to `timeout` for the "
                "broker's report and return the real outcome."
            ),
            examples=[False],
        )
        acks: (
            Literal["0", "1", "all"] | Literal[0, 1] | Ref(STRING_KIND, INTEGER_KIND)
        ) = Field(
            default="all",
            description=(
                "Brokers that must confirm a write: `all` for durability, `1` for "
                "the partition leader only, `0` for no confirmation."
            ),
            examples=["1"],
        )
        ssl_ca_location: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = (
            Field(
                default=None,
                description=SSL_CA_LOCATION_DESCRIPTION,
                examples=["/etc/ssl/certs/ca-certificates.crt"],
            )
        )
        timeout: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=5.0,
            description=(
                "Seconds for connecting and locating the topic on the first run, "
                "and for waiting on the broker's report when `fire_and_forget=false`. "
                "The client itself retries records for up to five minutes."
            ),
            examples=[5.0],
        )

        @field_validator("timeout")
        @classmethod
        def _timeout(cls, value: Any) -> Any:
            _check_timeout(value, "timeout", allow_zero=False)
            return value

    def __init__(
        self,
        *,
        kafka_publisher: Optional[KafkaPublisherPort] = None,
        disable_sinks: bool = False,
        allow_access_to_file_system: bool = False,
    ) -> None:
        self._publisher = kafka_publisher
        self._disable_sinks = disable_sinks
        self._allow_access_to_file_system = allow_access_to_file_system
        self._bootstrap_override_logged = False

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> List[WorkOperation]:
        """Declare the broker write.

        Args:
            params: Compiled parameters.

        Returns:
            The external-request operation.
        """
        return [WorkOperation.EXTERNAL_REQUEST]

    @classmethod
    def discover_restrictions(
        cls, params: BlockParams
    ) -> Union[List[RuntimeRestriction], Discovery]:
        """Declare V1's hosted-platform and fire-and-forget restrictions.

        Args:
            params: Compiled parameters.

        Returns:
            The restrictions; incomplete when ``fire_and_forget`` is selected.
        """
        if isinstance(params.fire_and_forget, str):
            problem = DiscoveryProblem(
                code=DiscoveryProblemCode.UNRESOLVED_SELECTOR,
                description=(
                    f"Field `fire_and_forget` is set by selector "
                    f"`{params.fire_and_forget}`, which is only known at run time, "
                    "so the restrictions are not fully known."
                ),
                details={
                    "declaration": "restrictions",
                    "field": "fire_and_forget",
                    "selector": params.fire_and_forget,
                },
            )
            discovery = incomplete_discovery([PRODUCER_HOSTED_RESTRICTION], [problem])
            return discovery

        if params.fire_and_forget:
            return [PRODUCER_HOSTED_RESTRICTION, FIRE_AND_FORGET_RESTRICTION]

        return [PRODUCER_HOSTED_RESTRICTION]

    @classmethod
    def discover_dependent_resources(cls, params: BlockParams) -> List[Any]:
        """Declare no dependent resources, as V1 did.

        Args:
            params: Compiled parameters.

        Returns:
            An empty list.
        """
        return []

    def run(
        self,
        *,
        bootstrap_servers: str,
        topic: str,
        message: Any,
        provider: str = PROVIDER_SELF_HOSTED,
        username: Optional[str] = None,
        password: Optional[Union[str, Secret]] = None,
        aws_region: Optional[str] = None,
        key: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
        fire_and_forget: bool = True,
        acks: Union[str, int] = "all",
        ssl_ca_location: Optional[str] = None,
        timeout: float = 5.0,
    ) -> Dict[str, Any]:
        """Publish one record through the borrowed publisher.

        Args:
            bootstrap_servers: Broker list.
            topic: Topic name.
            message: Record value.
            provider: ``"Self-hosted"`` or ``"AWS MSK"``.
            username: SASL username or ``None``; empty means ``None``.
            password: SASL password or ``None``; empty means ``None``.
            aws_region: MSK region or ``None``.
            key: Record key or ``None``.
            headers: Record headers or ``None``.
            fire_and_forget: Return once queued.
            acks: ``"0"``, ``"1"`` or ``"all"``.
            ssl_ca_location: CA bundle path, or ``None``.
            timeout: Seconds to connect and, when confirmed, to wait.

        Returns:
            ``error_status`` and ``message``.

        Raises:
            KafkaHostError: Without a publisher, or with a closed one when the
                parameters are valid.
            SecretUnavailableError: When the password has no value here.
        """
        if self._disable_sinks:
            return {
                "error_status": False,
                "message": disabled_sink_message(by_policy=True),
            }

        if _hosted_platform():
            LOGGER.error("Kafka Producer failure: %s", PRODUCER_HOSTED_MESSAGE)
            return {"error_status": True, "message": PRODUCER_HOSTED_MESSAGE}

        if self._publisher is None:
            raise KafkaHostError(NO_PUBLISHER_MESSAGE)

        secret = to_secret(password, name="password")
        if secret is not None and secret.reveal() == "":
            secret = None
        scrub = reveal_secrets(secret).scrub
        try:
            request = KafkaPublish(
                acks=coerce_acks(acks),
                connection=kafka_connection(
                    bootstrap_servers,
                    topic=topic,
                    provider=provider,
                    username=username or None,
                    password=secret,
                    aws_region=aws_region,
                    ssl_ca_location=ssl_ca_location,
                    allow_access_to_file_system=self._allow_access_to_file_system,
                    log_override=not self._bootstrap_override_logged,
                ),
                value=encode_value(message),
                key=encode_key(key),
                headers=encode_headers(headers),
                timeout=float(timeout),
            )
        except KafkaConfigurationError as error:
            failure = scrub(str(error))
            LOGGER.error("Kafka Producer failure: %s", failure)
            return {"error_status": True, "message": failure}
        finally:
            self._bootstrap_override_logged = True

        error_status, text = self._publisher.publish(request, wait=not fire_and_forget)
        outputs = {"error_status": error_status, "message": text}

        return outputs
