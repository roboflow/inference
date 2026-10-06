import logging
import time
from typing import Literal, Optional, Tuple

import requests
from pydantic import BaseModel, Field

from inference_models import configuration as models_configuration
from inference_models.utils.secure_gateway import validate_secure_gateway_url
from inference_server import configuration

logger = logging.getLogger(__name__)

GATEWAY_HEALTH_PATH = "/health"

SecureGatewayHealthStatus = Literal["healthy", "unhealthy", "not_configured"]
SecureGatewayHealthReason = Literal[
    "gateway_error",
    "unexpected_redirect",
    "tls_error",
    "connection_error",
    "timeout",
    "request_error",
]


class SecureGatewayHealthResponse(BaseModel):
    status: SecureGatewayHealthStatus = Field(
        description="healthy: gateway /health answered 2xx. unhealthy: see `reason`. "
        "not_configured: SECURE_GATEWAY is not set on this server.",
        examples=["healthy"],
    )
    reason: Optional[SecureGatewayHealthReason] = Field(
        default=None,
        description="Set only when status is unhealthy. gateway_error: non-2xx answer. "
        "unexpected_redirect: 3xx answer (typically a bare-host SECURE_GATEWAY on a "
        "TLS gateway). tls_error: TLS handshake failed. connection_error: could not "
        "connect. timeout: no answer within the probe timeout. request_error: other "
        "client-side failure.",
    )
    gateway_status_code: Optional[int] = Field(
        default=None,
        description="HTTP status returned by the gateway's /health route, if it answered.",
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Round-trip time of the probe in milliseconds, if the gateway answered.",
    )


def get_secure_gateway_base_url() -> Optional[str]:
    if not models_configuration.SECURE_GATEWAY:
        return None

    gateway_base_url = validate_secure_gateway_url(models_configuration.SECURE_GATEWAY)

    return gateway_base_url


def probe_secure_gateway_health(
    gateway_base_url: Optional[str],
    *,
    timeout: float,
    verify_ssl: bool,
) -> Tuple[int, SecureGatewayHealthResponse]:
    if not gateway_base_url:
        return 404, SecureGatewayHealthResponse(status="not_configured")

    start = time.perf_counter()
    try:
        response = requests.get(
            f"{gateway_base_url}{GATEWAY_HEALTH_PATH}",
            timeout=timeout,
            verify=verify_ssl,
            allow_redirects=False,
            headers={
                "User-Agent": f"roboflow-inference/{configuration.SERVER_VERSION}"
            },
        )
    except requests.exceptions.SSLError as error:
        return 503, _unhealthy("tls_error", error_type=type(error).__name__)
    except requests.exceptions.Timeout as error:
        return 504, _unhealthy("timeout", error_type=type(error).__name__)
    except requests.exceptions.ConnectionError as error:
        return 503, _unhealthy("connection_error", error_type=type(error).__name__)
    except requests.exceptions.RequestException as error:
        return 503, _unhealthy("request_error", error_type=type(error).__name__)

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    if 300 <= response.status_code < 400:
        return 502, _unhealthy(
            "unexpected_redirect",
            gateway_status_code=response.status_code,
            latency_ms=latency_ms,
        )
    if not 200 <= response.status_code < 300:
        return 502, _unhealthy(
            "gateway_error",
            gateway_status_code=response.status_code,
            latency_ms=latency_ms,
        )

    healthy = SecureGatewayHealthResponse(
        status="healthy",
        gateway_status_code=response.status_code,
        latency_ms=latency_ms,
    )

    return 200, healthy


def _unhealthy(
    reason: str,
    *,
    error_type: Optional[str] = None,
    gateway_status_code: Optional[int] = None,
    latency_ms: Optional[float] = None,
) -> SecureGatewayHealthResponse:
    logger.warning(
        "Secure gateway health probe failed (reason=%s, error_type=%s, status_code=%s)",
        reason,
        error_type,
        gateway_status_code,
    )
    unhealthy = SecureGatewayHealthResponse(
        status="unhealthy",
        reason=reason,
        gateway_status_code=gateway_status_code,
        latency_ms=latency_ms,
    )

    return unhealthy
