from typing import Literal, Optional

from pydantic import BaseModel, Field

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
    """Outcome of probing the configured secure gateway's own `/health` route.

    `status` and `reason` are the client contract. The response deliberately
    carries nothing that identifies the gateway (no URL, no upstream body, no
    exception text) - only the probe verdict and two diagnostics. Every key is
    always present (None when not applicable) so clients can rely on the shape.
    """

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
