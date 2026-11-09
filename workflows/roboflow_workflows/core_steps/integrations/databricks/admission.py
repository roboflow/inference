"""Aggregate Databricks admission before worker launch or destination resolution.

Hosts inject a controller backed by their existing Redis client. The standalone
fallback is process-local and is never accepted for hosted execution. Redis owns
time and performs all rate/concurrency checks atomically. No credentials or raw
workspace identifiers are stored in keys.
"""

import contextlib
import hashlib
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Protocol, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)
_WINDOW_SECONDS = 60
_LEASE_GRACE_SECONDS = 65  # 35 s worker cleanup plus scheduling tolerance.


class AdmissionRejected(RuntimeError):
    """The block must not start because a quota or admission dependency failed."""


@dataclass(frozen=True)
class AdmissionLimits:
    """Operator limits; rates count admitted invocations in a rolling minute."""

    tenant_concurrency: int = 2
    workspace_concurrency: int = 8
    warehouse_concurrency: int = 4
    fleet_concurrency: int = 32
    tenant_rate: int = 30
    workspace_rate: int = 120
    warehouse_rate: int = 60
    fleet_rate: int = 300

    def __post_init__(self) -> None:
        for value in vars(self).values():
            if type(value) is not int or not 1 <= value <= 100000:
                raise ValueError(
                    "Databricks admission limits must be integers in 1..100000"
                )

    @classmethod
    def from_environment(cls) -> "AdmissionLimits":
        """Read the DATABRICKS_ADMISSION_<FIELD> operator settings at startup."""
        defaults = cls()
        values = {
            field: int(os.environ.get(f"DATABRICKS_ADMISSION_{field.upper()}", value))
            for field, value in vars(defaults).items()
        }
        return cls(**values)

    def ordered(self) -> Tuple[int, ...]:
        """Return concurrency then rate limits in the admission key order."""
        return (
            self.fleet_concurrency,
            self.tenant_concurrency,
            self.workspace_concurrency,
            self.warehouse_concurrency,
            self.fleet_rate,
            self.tenant_rate,
            self.workspace_rate,
            self.warehouse_rate,
        )


def _keys(*, tenant: str, hostname: str, http_path: str) -> Tuple[str, ...]:
    # Both SQL endpoint aliases address the same warehouse quota.
    path = http_path.replace("/sql/1.0/endpoints/", "/sql/1.0/warehouses/", 1)
    values = (
        "fleet",
        "tenant:" + tenant,
        "workspace:" + hostname,
        "warehouse:" + hostname + path,
    )
    hashes = [hashlib.sha256(value.encode()).hexdigest() for value in values]
    # One hash slot also supports Redis Cluster's atomic multi-key scripts.
    return tuple(
        f"{{databricks-admission-v1}}:{kind}:{value}"
        for kind in ("active", "rate")
        for value in hashes
    )


class AdmissionBackend(Protocol):
    distributed: bool

    def acquire(
        self,
        *,
        keys: Tuple[str, ...],
        token: str,
        lease_seconds: int,
        limits: Tuple[int, ...],
    ) -> bool: ...

    def release(self, *, keys: Tuple[str, ...], token: str) -> None: ...


class LocalAdmissionBackend:
    """Thread-safe process-local quotas for standalone/self-hosted use only."""

    distributed = False

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, Dict[str, float]] = {}

    def acquire(
        self,
        *,
        keys: Tuple[str, ...],
        token: str,
        lease_seconds: int,
        limits: Tuple[int, ...],
    ) -> bool:
        with self._lock:
            now = time.monotonic()
            self._entries = {
                key: live
                for key, entries in self._entries.items()
                if (
                    live := {
                        member: expiry
                        for member, expiry in entries.items()
                        if expiry > now
                    }
                )
            }
            if any(
                len(self._entries.get(key, {})) >= limit
                for key, limit in zip(keys, limits)
            ):
                return False
            for index, key in enumerate(keys):
                duration = lease_seconds if index < 4 else _WINDOW_SECONDS
                self._entries.setdefault(key, {})[token] = now + duration
            return True

    def release(self, *, keys: Tuple[str, ...], token: str) -> None:
        with self._lock:
            for key in keys[:4]:
                self._entries.get(key, {}).pop(token, None)


_ACQUIRE_SCRIPT = """
local now_parts = redis.call('TIME')
local now = tonumber(now_parts[1]) * 1000 + math.floor(tonumber(now_parts[2]) / 1000)
for i, key in ipairs(KEYS) do
    redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
    if redis.call('ZCARD', key) >= tonumber(ARGV[i + 2]) then
        return 0
    end
end
for i, key in ipairs(KEYS) do
    local ttl = 60000
    if i <= 4 then ttl = tonumber(ARGV[2]) * 1000 end
    redis.call('ZADD', key, now + ttl, ARGV[1])
    local current_ttl = redis.call('PTTL', key)
    if current_ttl < ttl then redis.call('PEXPIRE', key, ttl) end
end
return 1
"""
_RELEASE_SCRIPT = """
for _, key in ipairs(KEYS) do redis.call('ZREM', key, ARGV[1]) end
return 1
"""


class RedisAdmissionBackend:
    """Shared atomic quotas using an existing host-owned Redis connection."""

    distributed = True

    def __init__(self, client: Any) -> None:
        self._client = client

    def acquire(
        self,
        *,
        keys: Tuple[str, ...],
        token: str,
        lease_seconds: int,
        limits: Tuple[int, ...],
    ) -> bool:
        result = self._client.eval(
            _ACQUIRE_SCRIPT, len(keys), *keys, token, lease_seconds, *limits
        )
        return result == 1

    def release(self, *, keys: Tuple[str, ...], token: str) -> None:
        self._client.eval(_RELEASE_SCRIPT, 4, *keys[:4], token)


class DatabricksAdmission:
    """Obtain one lease covering launch, execution, result loading and cleanup."""

    def __init__(self, backend: AdmissionBackend, *, limits: AdmissionLimits) -> None:
        self.backend = backend
        self.limits = limits

    def require_shared_backend(self, *, hosted: bool) -> None:
        """Fail closed on hosted workers without a shared admission backend."""
        if hosted and not self.backend.distributed:
            raise AdmissionRejected(
                "Hosted Databricks execution requires shared Redis admission control"
            )

    @contextlib.contextmanager
    def admit(
        self, *, tenant: str, hostname: str, http_path: str, deadline: float
    ) -> Iterator[None]:
        """Acquire all quotas atomically, without waiting or starting a worker.

        Args:
            tenant: Authenticated workspace ID supplied by the trusted host.
            hostname: Syntax-validated, normalized Databricks workspace hostname.
            http_path: Syntax-validated SQL endpoint path.
            deadline: Absolute monotonic invocation deadline.

        Yields:
            Nothing; the caller may run one worker while the lease is held.

        Raises:
            AdmissionRejected: A limit is reached or the backend is unavailable.
            TimeoutError: The invocation deadline has already expired.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Databricks deadline expired before admission")
        keys = _keys(tenant=tenant, hostname=hostname, http_path=http_path)
        token = uuid4().hex
        try:
            accepted = self.backend.acquire(
                keys=keys,
                token=token,
                lease_seconds=math.ceil(remaining) + _LEASE_GRACE_SECONDS,
                limits=self.limits.ordered(),
            )
        except Exception:
            raise AdmissionRejected(
                "Databricks admission backend is unavailable"
            ) from None
        if not accepted:
            raise AdmissionRejected("Databricks execution limit reached; retry later")
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("Databricks deadline expired during admission")
            yield
        finally:
            try:
                self.backend.release(keys=keys, token=token)
            except Exception:
                # Keep the lease until expiry; do not turn a backend outage into
                # a retry of a successful image import or expose connection details.
                logger.warning(
                    "Databricks admission lease release failed; waiting for expiry"
                )


DEFAULT_ADMISSION = DatabricksAdmission(
    LocalAdmissionBackend(), limits=AdmissionLimits.from_environment()
)


def admission_from_cache(cache: Any) -> DatabricksAdmission:
    """Reuse the host Redis client; a local cache never claims distributed quotas.

    Args:
        cache: Host-selected Workflows cache, optionally exposing a Redis client.

    Returns:
        A shared controller when Redis is available, otherwise the local singleton.
    """
    client: Optional[Any] = getattr(cache, "client", None)
    if client is None:
        return DEFAULT_ADMISSION
    controller = DatabricksAdmission(
        RedisAdmissionBackend(client), limits=DEFAULT_ADMISSION.limits
    )
    return controller
