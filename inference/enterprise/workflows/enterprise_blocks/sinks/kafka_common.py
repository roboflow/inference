"""Connection helpers shared by the Kafka Consumer and Kafka Producer enterprise blocks.

Block *versions* stay independent; these helpers hold the provider / credential logic that
must behave identically in every Kafka block: the operator's bootstrap-server policy,
provider-derived librdkafka configuration, AWS MSK IAM token signing, coercion of
selector-resolved values, and tracking of the problems a librdkafka client reports.
"""

import ipaddress
import logging
import math
import re
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple

from inference.core.env import (
    KAFKA_WORKFLOWS_SINKS_ALLOW_USER_PROVIDED_BOOTSTRAP_SERVERS,
    KAFKA_WORKFLOWS_SINKS_WHITELISTED_BOOTSTRAP_SERVERS,
)

try:
    import confluent_kafka
except ImportError:  # pragma: no cover - exercised only on images without the wheel
    confluent_kafka = None

try:
    from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
except ImportError:  # pragma: no cover - exercised only on images without the wheel
    MSKAuthTokenProvider = None

logger = logging.getLogger(__name__)

PROVIDER_SELF_HOSTED = "Self-hosted"
PROVIDER_AWS_MSK = "AWS MSK"
PROVIDERS = (PROVIDER_SELF_HOSTED, PROVIDER_AWS_MSK)
MSK_HOST_PATTERN = re.compile(
    r"\.kafka(?:-serverless)?\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?$", re.IGNORECASE
)
# every block instance gets its own group so pipelines never share partitions
GROUP_ID_PREFIX = "roboflow-inference-"
# only warnings and errors from librdkafka itself
LIBRDKAFKA_LOG_LEVEL = 4
# port librdkafka applies to a bootstrap entry that names none
DEFAULT_KAFKA_PORT = "9092"
# while a client-reported problem is recorded, how often a run may ask the broker for
# topic metadata to find out whether the connection recovered, and for how long; the
# first probe of a problem streak is not delayed by the interval
CLIENT_RECOVERY_PROBE_INTERVAL = 5.0
CLIENT_RECOVERY_PROBE_TIMEOUT = 1.0


class ConfigurationError(ValueError):
    """User-facing configuration problem detected before touching the broker."""


def is_selector(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("$")


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
    """`host:port` form used to compare a bootstrap entry with the operator's allowlist:
    whitespace trimmed, host lowercased, a missing port read as 9092, IPv6 in brackets.
    Text only, no DNS resolution. A malformed entry is returned lowercased, so it can
    match nothing but the same malformed text."""
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
        # several colons, yet not an IPv6 literal (e.g. a `SSL://host:port` spelling)
        return entry.lower()
    # a bare host, or an IPv6 literal without brackets, which cannot carry a port
    return f"{_normalise_host(entry)}:{DEFAULT_KAFKA_PORT}"


def _split_bootstrap_servers(value: Any) -> List[str]:
    entries: Iterable[Any] = value.split(",") if isinstance(value, str) else value
    return [str(entry).strip() for entry in entries if str(entry).strip()]


def resolve_bootstrap_servers(bootstrap_servers: str, log_override: bool = True) -> str:
    """Apply the operator's policy (see KAFKA_WORKFLOWS_SINKS_* in inference.core.env) to
    the workflow-provided `bootstrap_servers` and return the value to hand to librdkafka.
    Raises ConfigurationError when the policy forbids the connection.

    `log_override=False` silences the warning emitted when the workflow-provided value
    is replaced, so a block can log it once per instance instead of once per frame."""
    # module attributes are read on every call so tests can patch them
    user_provided_allowed = KAFKA_WORKFLOWS_SINKS_ALLOW_USER_PROVIDED_BOOTSTRAP_SERVERS
    allowlist = KAFKA_WORKFLOWS_SINKS_WHITELISTED_BOOTSTRAP_SERVERS
    operator_servers = (
        _split_bootstrap_servers(allowlist) if allowlist is not None else []
    )
    if not user_provided_allowed:
        if not operator_servers:
            raise ConfigurationError(
                "Kafka blocks are disabled on this deployment: workflow-provided "
                "bootstrap servers are not allowed and the operator configured none."
            )
        if log_override:
            logger.warning(
                "Kafka blocks: the workflow-provided bootstrap_servers value was replaced "
                "by the operator-configured servers (%s), because "
                "KAFKA_WORKFLOWS_SINKS_ALLOW_USER_PROVIDED_BOOTSTRAP_SERVERS is False.",
                ",".join(operator_servers),
            )
        # exactly what the operator wrote, in the operator's order
        return ",".join(operator_servers)
    if allowlist is None:
        return bootstrap_servers
    entries = _split_bootstrap_servers(bootstrap_servers)
    if not entries:
        raise ConfigurationError("bootstrap_servers holds no host:port entry.")
    permitted = {normalise_bootstrap_server(entry) for entry in operator_servers}
    for entry in entries:
        if normalise_bootstrap_server(entry) not in permitted:
            # the allowlist holds the operator's internal hostnames: never echo it
            raise ConfigurationError(
                f"Bootstrap server {entry!r} is not permitted on this deployment: the "
                "operator restricts the Kafka servers Workflow blocks may connect to."
            )
    return bootstrap_servers


def derive_msk_region(bootstrap_servers: str) -> Optional[str]:
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


def msk_token_callback(
    region: str, failures: List[BaseException]
) -> Callable[[str], Tuple[str, float]]:
    """librdkafka OAUTHBEARER callback signing an MSK IAM token with the host's AWS
    credential chain. Failures are recorded in `failures` (latest only) so the block can
    report them once; a later success clears them."""

    def callback(_config: str) -> Tuple[str, float]:
        try:
            # resolved at call time so tests can patch the module attribute
            token, expiry_ms = MSKAuthTokenProvider.generate_auth_token(region)
        except Exception as error:
            failures[:] = [error]
            raise
        failures.clear()
        # the signer reports expiry in epoch milliseconds; librdkafka wants seconds
        return token, expiry_ms / 1000

    return callback


def build_connection_config(
    provider: str,
    bootstrap_servers: str,
    username: Optional[str],
    password: Optional[str],
    aws_region: Optional[str],
    ssl_ca_location: Optional[str],
    auth_failures: List[BaseException],
    allow_access_to_file_system: bool,
) -> Dict[str, Any]:
    """Transport + SASL settings derived from the provider dropdown; never exposed."""
    if ssl_ca_location and not allow_access_to_file_system:
        # the path is chosen by the workflow and read by the server process
        raise ConfigurationError(
            "ssl_ca_location needs access to the local file system, which is disabled "
            "on this deployment (ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=False)."
        )
    if provider == PROVIDER_SELF_HOSTED:
        if (username is None) != (password is None):
            raise ConfigurationError(
                "Set both username and password for SASL authentication, or neither for "
                "an unauthenticated plaintext listener."
            )
        if username is None:
            config: Dict[str, Any] = {"security.protocol": "PLAINTEXT"}
        else:
            config = {
                "security.protocol": "SASL_SSL",
                "sasl.mechanisms": "SCRAM-SHA-512",
                "sasl.username": str(username),
                "sasl.password": str(password),
            }
    elif provider == PROVIDER_AWS_MSK:
        if MSKAuthTokenProvider is None:
            raise ConfigurationError(
                "AWS MSK provider requires the aws-msk-iam-sasl-signer-python package."
            )
        region = str(aws_region).strip() if aws_region else None
        if not region:
            region = derive_msk_region(bootstrap_servers)
        if not region:
            raise ConfigurationError(
                "Could not derive the AWS region from bootstrap_servers; set aws_region."
            )
        config = {
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "OAUTHBEARER",
            "oauth_cb": msk_token_callback(region, auth_failures),
        }
    else:
        raise ConfigurationError(
            f"Unknown provider {provider!r}; expected one of {', '.join(PROVIDERS)}."
        )
    if ssl_ca_location:
        config["ssl.ca.location"] = str(ssl_ca_location)
    return config


def preflight_token(connection_config: Dict[str, Any]) -> None:
    """Sign once before constructing a librdkafka client: the confluent-kafka constructor
    blocks for 10s and raises an opaque SASL error when the token callback fails, so a
    missing AWS credential chain must fail fast with a clear cause. Raises the signer's
    exception (already recorded in the block's failure list by the callback)."""
    token_callback = connection_config.get("oauth_cb")
    if token_callback is not None:
        token_callback("")


# librdkafka error codes a broker returns while a topic is still being (auto-)created
TRANSIENT_TOPIC_ERROR_CODES = (3, 5)  # UNKNOWN_TOPIC_OR_PART, LEADER_NOT_AVAILABLE
TOPIC_METADATA_RETRY_INTERVAL = 0.2


def is_transient_topic_error(error: Any) -> bool:
    code = getattr(error, "code", None)
    try:
        return callable(code) and code() in TRANSIENT_TOPIC_ERROR_CODES
    except Exception:
        return False


def wait_for_topic_metadata(client: Any, topic: str, deadline: float) -> Any:
    """Return the topic's metadata, retrying until `deadline` while the broker reports
    the topic as unknown or leaderless (what it says while auto-creation is in flight,
    which a stock Kafka client also rides out). Raises ConfigurationError once the
    deadline passes or the error is not transient."""
    while True:
        metadata = client.list_topics(
            topic, timeout=time_remaining(deadline, "fetching topic metadata")
        )
        topic_metadata = metadata.topics.get(topic)
        error = getattr(topic_metadata, "error", None) if topic_metadata else None
        if topic_metadata is not None and error is None:
            return topic_metadata
        transient = topic_metadata is None or is_transient_topic_error(error)
        if transient and deadline - time.monotonic() > TOPIC_METADATA_RETRY_INTERVAL:
            time.sleep(TOPIC_METADATA_RETRY_INTERVAL)
            continue
        detail = str(error) if topic_metadata is not None else "not found"
        raise ConfigurationError(f"Kafka topic {topic!r} is not available ({detail}).")


def time_remaining(deadline: float, what: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"timed out {what}")
    return remaining


def coerce_non_negative_int(value: Any, name: str) -> int:
    # selector-supplied values arrive uncoerced (numeric strings, floats from JSON)
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a non-negative integer.")
    if isinstance(value, float):
        if not value.is_integer():
            raise ConfigurationError(f"{name} must be a non-negative integer.")
        value = int(value)
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ConfigurationError(f"{name} must be a non-negative integer.")
    if number < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer.")
    return number


# librdkafka stores millisecond settings in a signed 32-bit int
MAX_TIMEOUT_SECONDS = 2_147_483


def coerce_timeout(value: Any, name: str, allow_zero: bool) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(
            f"Invalid {name}: {value!r}. It must be a finite number of seconds."
        )
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        seconds = math.nan
    lower_ok = seconds >= 0 if allow_zero else seconds > 0
    if not math.isfinite(seconds) or not lower_ok or seconds > MAX_TIMEOUT_SECONDS:
        raise ConfigurationError(
            f"Invalid {name}: {value!r}. It must be a finite number of seconds up to "
            f"{MAX_TIMEOUT_SECONDS}."
        )
    return seconds


def describe_error(error: BaseException) -> str:
    text = str(error).strip()
    return text or error.__class__.__name__


def describe_auth_error(error: BaseException) -> str:
    text = describe_error(error)
    # botocore surfaces an empty credential chain either as NoCredentialsError or,
    # via the MSK signer, as an AttributeError on the missing credentials object
    if "access_key" in text or error.__class__.__name__ in (
        "NoCredentialsError",
        "PartialCredentialsError",
    ):
        return "no AWS credentials were found on this machine"
    return text


def pop_auth_failure_message(failures: List[BaseException]) -> Optional[str]:
    """Report a recorded MSK token failure once and clear it; librdkafka keeps retrying
    the refresh in the background."""
    # the callback mutates this list on librdkafka's thread; a refresh that succeeds
    # between the emptiness check and the pop must not turn into an exception
    try:
        error = failures.pop()
    except IndexError:
        return None
    failures.clear()
    return (
        "AWS MSK IAM authentication failed: "
        f"{describe_auth_error(error)}. Check that AWS credentials are available on this "
        "machine and that the identity may connect to the cluster."
    )


def describe_client_error(error: Any) -> str:
    """Text for an error handed to a librdkafka `error_cb`. Never raises: test doubles
    and future client versions may pass objects without the KafkaError accessors."""
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
    return f"{label}: {text}"


def _is_fatal(error: Any) -> bool:
    try:
        return bool(error.fatal())
    except Exception:
        return False


def combine_messages(*messages: Optional[str]) -> Optional[str]:
    present = [message for message in messages if message]
    return "; ".join(present) if present else None


class ClientProblem(NamedTuple):
    description: str
    fatal: bool
    since: float  # monotonic time the first error of this streak was reported
    recorded_at: float  # monotonic time of the latest error

    def message(self, returned_data_note: Optional[str] = None) -> str:
        age = max(0.0, time.monotonic() - self.since)
        if self.fatal:
            text = (
                f"The Kafka client reported a fatal error {age:.1f}s ago and is "
                f"unusable: {self.description}. Restart the block (pipeline) to "
                "reconnect."
            )
        else:
            text = (
                f"The Kafka client reported a broker problem, first seen {age:.1f}s "
                f"ago: {self.description}. This is reported on every run until the "
                "connection is proven healthy again."
            )
        return f"{text} {returned_data_note}" if returned_data_note else text


class ClientErrorTracker:
    """`error_cb` of one librdkafka client. Global client errors ("all brokers down",
    transport failures) are delivered only to this callback, served from inside poll() /
    flush(), never returned from them; without it a broker outage after a successful
    connect is invisible.

    The callback also receives harmless errors (one broker of several disconnecting, an
    idle connection being closed), so a new problem is checked against the broker on
    the very next `check()`: if the metadata probe succeeds, the problem is dropped and
    never reported. Otherwise it stays, and is reported on every run, until proof of
    life: `clear()` (a polled record, a successful delivery report) or a later
    successful probe, tried at most once per CLIENT_RECOVERY_PROBE_INTERVAL.
    `error_cb` has no "recovered" event. A fatal error is never probed or cleared."""

    def __init__(self, client_name: str):
        self._client_name = client_name
        self._lock = threading.Lock()
        self._problem: Optional[ClientProblem] = None
        self._generation = 0
        self._last_probe_at = 0.0

    def __call__(self, error: Any) -> None:
        try:
            description = describe_client_error(error)
            fatal = _is_fatal(error)
            now = time.monotonic()
            with self._lock:
                self._generation += 1
                previous = self._problem
                if previous is not None and previous.fatal:
                    return
                if previous is None:
                    # the first probe is due at once: librdkafka also reports
                    # single-broker disconnects and idle-connection closes here, and a
                    # healthy cluster answers in milliseconds, so such a blip is never
                    # claimed; a real outage fails the probe and is claimed from then on
                    self._last_probe_at = -math.inf
                self._problem = ClientProblem(
                    description=description,
                    fatal=fatal,
                    since=now if previous is None else previous.since,
                    recorded_at=now,
                )
            if previous is None or fatal:
                logger.error(
                    "%s: the Kafka client reported %s: %s",
                    self._client_name,
                    "a fatal error" if fatal else "a broker problem",
                    description,
                )
        except Exception:  # a callback raising inside librdkafka helps nobody
            pass

    @property
    def generation(self) -> int:
        """Number of errors reported so far; lets a caller tell whether an error arrived
        during a stretch of work."""
        with self._lock:
            return self._generation

    @property
    def problem(self) -> Optional[ClientProblem]:
        with self._lock:
            return self._problem

    def clear(self, if_generation: Optional[int] = None) -> None:
        """Proof of life. With `if_generation`, only when no error was reported since
        that generation was read."""
        with self._lock:
            problem = self._problem
            if problem is None or problem.fatal:
                return
            if if_generation is not None and if_generation != self._generation:
                return
            self._problem = None
        logger.info(
            "%s: Kafka connection is healthy again after: %s",
            self._client_name,
            problem.description,
        )

    def reset(self) -> None:
        """Forget everything, fatal errors included: the client is gone."""
        with self._lock:
            self._problem = None
            self._last_probe_at = 0.0

    def probe_if_due(self, client: Any, topic: str) -> None:
        """Recovery probe: at once for a new problem streak, then at most once per
        CLIENT_RECOVERY_PROBE_INTERVAL while the streak lasts. Blocks for up to
        CLIENT_RECOVERY_PROBE_TIMEOUT when the broker does not answer. Costs nothing
        while no problem is recorded, and never runs for a fatal error."""
        with self._lock:
            problem = self._problem
            if problem is None or problem.fatal:
                return
            now = time.monotonic()
            if now - self._last_probe_at < CLIENT_RECOVERY_PROBE_INTERVAL:
                return
            self._last_probe_at = now
            generation = self._generation
        # not under the lock: the client may serve callbacks while it waits
        if probe_topic_metadata(client, topic):
            self.clear(if_generation=generation)

    def check(
        self,
        client: Any,
        topic: str,
        alive_since_generation: Optional[int] = None,
    ) -> Optional[ClientProblem]:
        """End-of-run check: apply the caller's proof of life (valid only if no error
        arrived since `alive_since_generation` was read), probe when due (at once for a
        problem first reported during this run), and return the problem this run must
        report, if any."""
        if alive_since_generation is not None:
            self.clear(if_generation=alive_since_generation)
        self.probe_if_due(client, topic)
        return self.problem


def probe_topic_metadata(client: Any, topic: str) -> bool:
    """True when the broker answers a metadata request for `topic` without an error."""
    try:
        metadata = client.list_topics(topic, timeout=CLIENT_RECOVERY_PROBE_TIMEOUT)
        topic_metadata = metadata.topics.get(topic)
        return (
            topic_metadata is not None
            and getattr(topic_metadata, "error", None) is None
        )
    except Exception:
        return False
