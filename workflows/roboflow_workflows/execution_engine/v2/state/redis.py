"""Redis backend of managed state (optional ``redis`` package).

Use it through ``ManagedState`` with an explicit address and namespace::

    from roboflow_workflows.execution_engine.v2.state import ManagedState
    from roboflow_workflows.execution_engine.v2.state.redis import RedisStateBackend

    state = ManagedState(
        RedisStateBackend("redis://127.0.0.1:6379/0"), namespace="line-7"
    )

Each operation is one Redis command, so it is atomic on the server:

    operation                    Redis
    get                          GET
    set / set(only_if_absent)    SET / SET NX
    delete                       DEL
    incr                         INCRBY
    compare_and_set              short Lua script: GET, compare, SET or DEL

Failures never fall back to memory and are never retried::

    could not get a connection     StateBackendError         (nothing was sent)
    Redis replied with an error    StateBackendError (or StateTypeError /
                                   StateOverflowError for incr)
    a mutation's reply was lost    StateOutcomeUnknownError  (maybe applied)

``redis`` is imported when a backend is constructed, and the first connection
opens on the first operation. Addresses in messages and ``repr`` never
include credentials.
"""

from typing import Any, Optional
from urllib.parse import urlsplit

from roboflow_workflows.execution_engine.v2.state.errors import (
    StateBackendError,
    StateOutcomeUnknownError,
    StateOverflowError,
    StateTypeError,
)

__all__ = ["RedisStateBackend"]

# ARGV: has_expected, expected, has_new, new.
_COMPARE_AND_SET = """
local current = redis.call('GET', KEYS[1])
if ARGV[1] == '1' then
  if current ~= ARGV[2] then
    return 0
  end
elseif current then
  return 0
end
if ARGV[3] == '1' then
  redis.call('SET', KEYS[1], ARGV[4])
else
  redis.call('DEL', KEYS[1])
end
return 1
"""


class RedisStateBackend:
    """State values in a Redis server, shared by every client of a namespace.

    Args:
        url: Explicit server address: ``redis://host:port/db``,
            ``rediss://...`` (TLS) or ``unix:///path/redis.sock``.
        timeout: Seconds for connecting, for each reply and for waiting on a
            free pooled connection. A stalled server raises instead of
            hanging the workflow.
        max_connections: Pool size; size it to at least the number of threads
            using state at once (pipeline workers plus handler workers).

    Raises:
        ImportError: When the ``redis`` package is not installed.
        ValueError: When ``url`` cannot be parsed or a limit is not positive.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 2.0,
        max_connections: int = 32,
    ) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if max_connections <= 0:
            raise ValueError(f"max_connections must be positive, got {max_connections}")

        try:
            import redis
        except ImportError as error:
            raise ImportError(
                "RedisStateBackend needs the optional 'redis' package: "
                "pip install 'roboflow-workflows[redis]'"
            ) from error

        self._redis = redis
        self._address = _redacted_address(url)
        self._pool = redis.BlockingConnectionPool.from_url(
            url,
            timeout=timeout,
            max_connections=max_connections,
            socket_timeout=timeout,
            socket_connect_timeout=timeout,
            retry_on_timeout=False,
        )
        self._closed = False

    @property
    def address(self) -> str:
        """Server address without credentials, e.g. ``redis://127.0.0.1:6379/0``."""
        return self._address

    def get(self, key: str) -> Optional[str]:
        """Return the stored text of ``key`` or ``None`` (``GET``)."""
        response = self._execute("GET", key, mutation=False)
        stored = None if response is None else response.decode("utf-8")

        return stored

    def set(self, key: str, value: str, *, only_if_absent: bool) -> bool:
        """Store ``value`` (``SET``, with ``NX`` when ``only_if_absent``)."""
        arguments = ("SET", key, value, "NX") if only_if_absent else ("SET", key, value)
        response = self._execute(*arguments, mutation=True)
        stored = response is not None

        return stored

    def delete(self, key: str) -> bool:
        """Remove ``key`` (``DEL``); return whether it existed."""
        response = self._execute("DEL", key, mutation=True)
        existed = response > 0

        return existed

    def incr(self, key: str, amount: int) -> int:
        """Add ``amount`` to the stored integer (``INCRBY``)."""
        result = self._execute("INCRBY", key, str(amount), mutation=True)

        return result

    def compare_and_set(
        self,
        key: str,
        expected: Optional[str],
        new: Optional[str],
    ) -> bool:
        """Replace or delete ``key`` when it equals ``expected`` (Lua script)."""
        response = self._execute(
            "EVAL",
            _COMPARE_AND_SET,
            "1",
            key,
            *_optional_argument(expected),
            *_optional_argument(new),
            mutation=True,
        )
        applied = response == 1

        return applied

    def close(self) -> None:
        """Close pooled connections; later operations raise ``StateBackendError``."""
        self._closed = True
        self._pool.disconnect()

    def __repr__(self) -> str:
        return f"RedisStateBackend({self._address!r})"

    def _execute(self, *arguments: Any, mutation: bool) -> Any:
        command = arguments[0]
        if self._closed:
            raise StateBackendError(f"Redis state backend {self._address} is closed")

        errors = self._redis.exceptions
        try:
            connection = self._pool.get_connection(command)
        except (errors.RedisError, OSError) as error:
            raise StateBackendError(
                f"Redis at {self._address} unavailable, {command} not applied: "
                f"{type(error).__name__}: {error}"
            ) from error

        try:
            connection.send_command(*arguments)
            response = connection.read_response()
        except errors.ResponseError as error:
            raise _reply_error(error, command=command, address=self._address) from error
        except (errors.RedisError, OSError) as error:
            connection.disconnect()
            if mutation:
                raise StateOutcomeUnknownError(
                    f"Redis at {self._address}: {command} was sent but its reply "
                    f"was lost ({type(error).__name__}: {error}); it may or may not "
                    "have been applied and was not retried"
                ) from error
            raise StateBackendError(
                f"Redis at {self._address}: {command} failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        except BaseException:
            connection.disconnect()
            raise
        finally:
            self._pool.release(connection)

        return response


def _optional_argument(value: Optional[str]) -> tuple:
    arguments = ("0", "") if value is None else ("1", value)

    return arguments


def _reply_error(error: Exception, *, command: str, address: str) -> Exception:
    message = str(error)
    if "increment or decrement would overflow" in message:
        mapped: Exception = StateOverflowError(
            "incr would leave the signed 64-bit range"
        )
        return mapped
    if "value is not an integer" in message:
        mapped = StateTypeError("incr needs an integer value")
        return mapped

    mapped = StateBackendError(
        f"Redis at {address} rejected {command}, not applied: {message}"
    )

    return mapped


def _redacted_address(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme:
        raise ValueError("Redis url needs a scheme: redis://, rediss:// or unix://")

    if parts.scheme == "unix":
        address = f"unix://{parts.path}"
        return address

    port = f":{parts.port}" if parts.port is not None else ""
    address = f"{parts.scheme}://{parts.hostname or ''}{port}{parts.path}"

    return address
