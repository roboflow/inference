"""Probe of the configured secure gateway, backing `GET /secure-gateway/health`.

Both proxies inference can be configured with - the legacy license server
(Node) and the roboflow secure gateway (FastAPI) - serve an unauthenticated
`GET /health` answering `{"status": "healthy", "timestamp": "<ISO>"}`. The
probe requests that route directly on the gateway host.

It deliberately does NOT go through `wrap_url()`: both proxies unwrap a
self-referential `/proxy?url=<gateway>/health` into
`https://api.roboflow.com/health`, which would test Roboflow's API instead of
the gateway.

The response never identifies the gateway: no URL, no upstream body, no
exception text. Those details go to the server log only.
"""

import time
from typing import Optional, Tuple

import requests

from inference.core.entities.responses.secure_gateway import SecureGatewayHealthResponse
from inference.core.logger import logger
from inference.core.version import __version__

GATEWAY_HEALTH_PATH = "/health"


def probe_secure_gateway_health(
    gateway_base_url: Optional[str],
    timeout: float,
    verify_ssl: bool,
) -> Tuple[int, SecureGatewayHealthResponse]:
    """Probe `{gateway_base_url}/health` and map the outcome onto an HTTP status.

    Never raises. Every outcome is a `(status_code, payload)` pair:
      200 healthy, 404 not_configured, 502 gateway answered non-2xx (3xx
      included), 503 TLS or connection failure, 504 timeout.

    Args:
        gateway_base_url: Output of `get_secure_gateway_base_url()`; None when
            SECURE_GATEWAY is not set.
        timeout: Request timeout in seconds (SECURE_GATEWAY_HEALTH_CHECK_TIMEOUT).
        verify_ssl: TLS verification flag. Mirrors ROBOFLOW_API_VERIFY_SSL so
            the probe fails the same way real proxied calls would.
    """
    if not gateway_base_url:
        return 404, SecureGatewayHealthResponse(status="not_configured")
    start = time.perf_counter()
    try:
        response = requests.get(
            f"{gateway_base_url}{GATEWAY_HEALTH_PATH}",
            timeout=timeout,
            verify=verify_ssl,
            allow_redirects=False,
            headers={"User-Agent": f"roboflow-inference/{__version__}"},
        )
    # Arm order matters: SSLError and ConnectTimeout both subclass
    # ConnectionError, so the specific arms must come before it.
    except requests.exceptions.SSLError as error:
        return 503, _unhealthy(reason="tls_error", error=error)
    except requests.exceptions.Timeout as error:
        return 504, _unhealthy(reason="timeout", error=error)
    except requests.exceptions.ConnectionError as error:
        return 503, _unhealthy(reason="connection_error", error=error)
    except requests.exceptions.RequestException as error:
        return 503, _unhealthy(reason="request_error", error=error)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    if 300 <= response.status_code < 400:
        return 502, _unhealthy(
            reason="unexpected_redirect",
            error=f"HTTP {response.status_code} -> {response.headers.get('Location')!r}",
            gateway_status_code=response.status_code,
            latency_ms=latency_ms,
        )
    if not 200 <= response.status_code < 300:
        return 502, _unhealthy(
            reason="gateway_error",
            error=f"HTTP {response.status_code}",
            gateway_status_code=response.status_code,
            latency_ms=latency_ms,
        )
    return 200, SecureGatewayHealthResponse(
        status="healthy",
        gateway_status_code=response.status_code,
        latency_ms=latency_ms,
    )


def _unhealthy(
    reason: str,
    error: object,
    gateway_status_code: Optional[int] = None,
    latency_ms: Optional[float] = None,
) -> SecureGatewayHealthResponse:
    # Expected diagnostic outcome, not a server fault - warning, not exception.
    # The error text (which may embed the gateway URL) stays in the server log.
    logger.warning("Secure gateway health probe failed (reason=%s): %s", reason, error)
    return SecureGatewayHealthResponse(
        status="unhealthy",
        reason=reason,
        gateway_status_code=gateway_status_code,
        latency_ms=latency_ms,
    )
