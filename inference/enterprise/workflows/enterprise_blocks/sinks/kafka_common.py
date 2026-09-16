"""Connection helpers shared by the Kafka Consumer and Kafka Producer enterprise blocks.

Block *versions* stay independent; these helpers hold the provider / credential logic that
must behave identically in every Kafka block: provider-derived librdkafka configuration,
AWS MSK IAM token signing, and coercion of selector-resolved values.
"""

import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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


class ConfigurationError(ValueError):
    """User-facing configuration problem detected before touching the broker."""


def is_selector(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("$")


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
) -> Dict[str, Any]:
    """Transport + SASL settings derived from the provider dropdown; never exposed."""
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
