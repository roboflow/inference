"""OPC UA writer with typed, immutable requests and explicit connection ownership.

The ``opc_client`` resource is a borrowed operation port: its host implements
``write(target, request)`` and closes it after sessions stop and effects drain.
Without that resource, each effect uses a fresh ``AsyncuaOPCClient``. Its async
client, connection, tasks and event loop all live inside ``asyncio.run``; there
is no sync ThreadLoop, pool, shutdown hook or resource stored on the block.
Fire-and-forget without host ``sink_effects`` executes synchronously.

Only connection establishment is retried (3 attempts, 15ms exponential base by
default). Authentication and server session exhaustion fail immediately. Once
a write is attempted it is never replayed: a failed acknowledgement can mean
that the server applied the value. Failed connects impose a per-block 2s circuit
backoff; exhausted servers impose 30–45s. Unlike V1's singleton pool these
backoffs and the monotonic cooldown do not coordinate independent blocks.

V1 conversion is retained: Boolean strings mean true only for stripped,
case-insensitive 'true' or '1'; other strings mean false. Integer conversion
uses int(), including truncation; Integer aliases Int64; String uses str().
No JSON decoding is performed. V2 deliberately rejects containers, non-finite
numbers, Float overflow, invalid timeouts and malformed namespace indexes
before connection. Params validate literal and selected values alike.

SDK logs on the operation thread are suppressed while using the default
adapter: asyncua debug records can contain endpoints, credentials and values.
Outcomes describe the failing phase without embedding raw driver exceptions.
Tests distinguish stand-ins, inspected SDK source and real driver execution;
none of those imply a real OPC service was exercised.
"""

import asyncio
import contextlib
import hashlib
import importlib
import json
import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, Union

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)
from roboflow_workflows.core_steps.common.workload_presets import (
    COOLDOWN_ACTUAL_RESTRICTION,
    FIRE_AND_FORGET_RESTRICTION,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    DiscoveryProblem,
    DiscoveryProblemCode,
    RuntimeRestriction,
    WorkOperation,
    incomplete_discovery,
)
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    ERROR_STATUS_OUTPUT,
    MESSAGE_OUTPUT,
    BackgroundEffects,
    deliver,
    disabled_sink_message,
)
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.secrets import (
    SECRET_KIND,
    Secret,
    reveal_secrets,
    to_secret,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

BLOCK_TYPE = "v2/opc_writer_sink@v1"
BACKGROUND_WRITE_MESSAGE = "Writing to the OPC UA server in the background task"
ValueType = Literal[
    "Boolean",
    "Double",
    "Float",
    "Int16",
    "Int32",
    "Int64",
    "Integer",
    "SByte",
    "String",
    "UInt16",
    "UInt32",
    "UInt64",
]
LookupMode = Literal["hierarchical", "direct"]
Scalar = Union[StrictStr, StrictBool, StrictInt, StrictFloat]
_INTEGER_RANGES = {
    "SByte": (-128, 127),
    "Int16": (-32768, 32767),
    "Int32": (-(2**31), 2**31 - 1),
    "Int64": (-(2**63), 2**63 - 1),
    "UInt16": (0, 2**16 - 1),
    "UInt32": (0, 2**32 - 1),
    "UInt64": (0, 2**64 - 1),
}
_VALUE_TYPES = set(_INTEGER_RANGES) | {
    "Boolean",
    "Double",
    "Float",
    "String",
    "Integer",
}


def _positive_env(name: str, *, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default

    return value if math.isfinite(value) and value > 0 else default


class OPCSinkError(Exception):
    """Report a safe, actionable OPC failure.

    Args:
        message: Public diagnostic without raw driver text or credentials.
        backoff_seconds: Block-local connection backoff, if connection failed.
    """

    def __init__(self, message: str, *, backoff_seconds: float = 0) -> None:
        super().__init__(message)
        self.backoff_seconds = backoff_seconds


@dataclass(frozen=True, repr=False)
class OPCTarget:
    """Immutable connection settings passed to an operation port.

    Args:
        url: Server endpoint, wrapped as a Secret even for string literals.
        user_name: Optional username, also protected as a Secret.
        password: Optional password.
        timeout: Per-request and connect timeout in seconds; not a total deadline.
        max_retries: Maximum connection attempts, never write attempts.
        retry_backoff_seconds: Exponential connection retry base.
    """

    url: Secret
    user_name: Optional[Secret]
    password: Optional[Secret]
    timeout: int
    max_retries: int
    retry_backoff_seconds: float


@dataclass(frozen=True)
class OPCWrite:
    """Detached scalar write, converted before dispatch.

    Args:
        namespace: Namespace URI or decimal index.
        object_name: Object path or string NodeId identifier.
        variable_name: Variable name; may be empty only in direct mode.
        value: Converted primitive.
        variant_type: Concrete UA type name; Integer has become Int64.
        node_lookup_mode: Hierarchical browse or direct string NodeId.
    """

    namespace: str
    object_name: str
    variable_name: str
    value: Union[str, bool, int, float]
    variant_type: str
    node_lookup_mode: LookupMode


class OPCClient(Protocol):
    """Borrowed operation port; hosts own reuse, synchronization and shutdown."""

    def write(self, target: OPCTarget, request: OPCWrite) -> None:
        """Write once, raising on failure or an unknown acknowledgement.

        Args:
            target: Immutable connection identity and timing settings.
            request: Immutable, converted write.

        Raises:
            OPCSinkError: A safe failure diagnostic. Other exceptions are hidden.
        """
        ...


class _QuietOperationLogs(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.thread_id = threading.get_ident()

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread != self.thread_id


@contextlib.contextmanager
def _quiet_sdk_logs():
    # asyncua 1.1.5 creates these instance loggers during connection setup.
    # Pre-register them so the filter precedes their first debug record.
    for name in (
        "asyncua.client.ua_client.UaClient",
        "asyncua.client.ua_client.UASocketProtocol",
    ):
        logging.getLogger(name)
    loggers = [logging.getLogger("asyncio")]
    loggers.extend(
        item
        for name, item in list(logging.Logger.manager.loggerDict.items())
        if name.startswith("asyncua.") and isinstance(item, logging.Logger)
    )
    loggers.append(logging.getLogger("asyncua"))
    guard = _QuietOperationLogs()
    for logger in loggers:
        logger.addFilter(guard)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(guard)


class AsyncuaOPCClient:
    """Default operation port using asyncua~=1.1.5, imported only on write.

    Each write owns a fresh asyncio loop. ``asyncio.run`` cancels/awaits pending
    tasks, shuts down asynchronous generators and the default executor (including
    DNS workers), then closes the loop. No asyncua.sync ThreadLoop is created.
    Connect attempts each own a fresh Client and disconnect in finally, even
    after partial acquisition. Cleanup failure stops retries and is reported;
    it cannot replace an earlier connect, lookup or write error.
    """

    def write(self, target: OPCTarget, request: OPCWrite) -> None:
        """Acquire, write once and disconnect on an operation-owned event loop.

        Args:
            target: Connection settings, including protected credentials.
            request: Converted write.

        Raises:
            OPCSinkError: Missing driver, acquisition, lookup, write or cleanup failure.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise OPCSinkError(
                "OPC default port requires a synchronous worker thread; supply opc_client in an async host"
            )

        try:
            sdk = importlib.import_module("asyncua")
            errors = importlib.import_module("asyncua.ua.uaerrors")
        except ImportError:
            raise OPCSinkError(
                "OPC UA driver unavailable; use an environment with asyncua~=1.1.5 or supply the host-owned opc_client resource"
            ) from None

        with _quiet_sdk_logs():
            asyncio.run(self._write(sdk, errors, target=target, request=request))

    async def _write(
        self, sdk: Any, errors: Any, *, target: OPCTarget, request: OPCWrite
    ) -> None:
        for attempt in range(target.max_retries):
            client = None
            failure = None
            connecting = True
            retry = False
            cleanup_failed = False
            try:
                client = sdk.Client(target.url.reveal(), timeout=target.timeout)
                client.session_timeout = int(
                    1000 * _positive_env("OPC_SESSION_TIMEOUT_SECONDS", default=60)
                )
                if target.user_name is not None and target.password is not None:
                    user_name = target.user_name.reveal()
                    password = target.password.reveal()
                    if user_name and password:
                        client.set_user(user_name)
                        client.set_password(password)
                await asyncio.wait_for(client.connect(), timeout=target.timeout)
                connecting = False
                await self._write_connected(
                    client, sdk=sdk, errors=errors, request=request
                )
            except OPCSinkError as error:
                failure = error
            except asyncio.CancelledError:
                failure = OPCSinkError(
                    "OPC operation was cancelled; delivery outcome may be unknown; write was not retried"
                )
            except Exception as error:
                if not connecting:
                    failure = OPCSinkError(
                        "OPC write failed; delivery outcome may be unknown. The write was not retried"
                    )
                elif isinstance(error, errors.BadUserAccessDenied):
                    failure = OPCSinkError(
                        "OPC authentication denied; check credentials and server permissions",
                        backoff_seconds=2,
                    )
                elif isinstance(
                    error, (errors.BadTooManySessions, errors.BadMaxConnectionsReached)
                ):
                    failure = OPCSinkError(
                        "OPC server session limit reached; reduce clients or raise the server limit",
                        backoff_seconds=_positive_env(
                            "OPC_SESSION_EXHAUSTION_BACKOFF_SECONDS", default=30
                        )
                        + random.uniform(0, 15),
                    )
                else:
                    retry = attempt + 1 < target.max_retries
                    failure = OPCSinkError(
                        f"OPC connection failed after {attempt + 1} attempt(s); check endpoint, network and server availability",
                        backoff_seconds=2,
                    )
            finally:
                if client is not None:
                    try:
                        await asyncio.wait_for(
                            client.disconnect(), timeout=target.timeout
                        )
                    except (Exception, asyncio.CancelledError):
                        cleanup_failed = True
                        # disconnect() closes the socket in its own finally in
                        # 1.1.5, except when wait_for cancels before it starts.
                        # Its synchronous socket close is safe in either case.
                        try:
                            client.disconnect_socket()
                        except Exception:
                            pass

            if cleanup_failed:
                primary = (
                    str(failure)
                    if failure is not None
                    else "Value was written successfully"
                )
                backoff = failure.backoff_seconds if failure is not None else 0
                raise OPCSinkError(
                    primary
                    + "; OPC cleanup failed; inspect host resources before another operation",
                    backoff_seconds=backoff,
                ) from None
            if failure is None:
                return
            if not retry:
                raise OPCSinkError(
                    str(failure), backoff_seconds=failure.backoff_seconds
                ) from None

            await asyncio.sleep(target.retry_backoff_seconds * (2**attempt))

    async def _write_connected(
        self, client: Any, *, sdk: Any, errors: Any, request: OPCWrite
    ) -> None:
        try:
            nsidx = (
                int(request.namespace)
                if request.namespace.isdigit()
                else await client.get_namespace_index(request.namespace)
            )
        except Exception:
            try:
                count = len(await client.get_namespace_array())
                hint = f"; server advertises {count} namespace(s)"
            except Exception:
                hint = ""
            raise OPCSinkError(
                "OPC namespace lookup failed; check the namespace URI or numeric index"
                + hint
            ) from None

        try:
            if request.node_lookup_mode == "direct":
                identifier = request.object_name
                if request.variable_name:
                    identifier += "/" + request.variable_name
                node = client.get_node(f"ns={nsidx};s={identifier}")
            else:
                components = ["0:Objects"] + [
                    f"{nsidx}:{part}" for part in request.object_name.split("/")
                ]
                components.append(f"{nsidx}:{request.variable_name}")
                node = await client.nodes.root.get_child(components)
        except Exception:
            raise OPCSinkError(
                "OPC node lookup failed; check object_name, variable_name and node_lookup_mode"
            ) from None

        try:
            await node.write_value(
                request.value, getattr(sdk.ua.VariantType, request.variant_type)
            )
        except errors.BadTypeMismatch:
            try:
                expected = (await node.read_data_type_as_variant_type()).name
                if expected not in _VALUE_TYPES:
                    expected = "another UA type"
            except Exception:
                expected = "an unknown UA type"
            raise OPCSinkError(
                f"OPC type mismatch; selected {request.variant_type}, node expects {expected}. Check value_type; write was not retried"
            ) from None
        except Exception:
            raise OPCSinkError(
                "OPC write failed; verify the node and permissions. Delivery outcome may be unknown; write was not retried"
            ) from None


def _converted_value(
    value: Any, *, value_type: str
) -> Tuple[Union[str, bool, int, float], str]:
    if type(value) not in (str, bool, int, float):
        raise ValueError("OPC value must be a scalar string, boolean, integer or float")
    if value_type not in _VALUE_TYPES:
        raise ValueError(
            "OPC value_type is unsupported; choose an advertised UA VariantType"
        )
    variant = "Int64" if value_type == "Integer" else value_type
    try:
        if variant == "Boolean":
            converted = (
                value.strip().lower() in ("true", "1")
                if isinstance(value, str)
                else bool(value)
            )
        elif variant == "String":
            converted = str(value)
        elif variant in _INTEGER_RANGES:
            converted = int(value)
            minimum, maximum = _INTEGER_RANGES[variant]
            if not minimum <= converted <= maximum:
                raise ValueError()
        else:
            converted = float(value)
            if not math.isfinite(converted) or (
                variant == "Float" and abs(converted) > 3.4028234663852886e38
            ):
                raise ValueError()
    except (ValueError, TypeError, OverflowError):
        raise ValueError(
            f"OPC value cannot be converted to {variant} or is out of range"
        ) from None

    return converted, variant


def _outcome(
    error: bool, message: str, *, disabled: bool = False, throttled: bool = False
) -> Dict[str, Any]:
    return {
        "disabled": disabled,
        "throttling_status": throttled,
        "error_status": error,
        "message": message,
    }


class OPCWriterSinkBlock(Block):
    """Write a scalar OPC UA variable; an injected opc_client is always borrowed.

    Args:
        opc_client: Optional caller-owned operation port. Never closed here.
        sink_effects: Host background effects; host closes it after sessions stop.
        disable_sinks: Host execution policy.
        clock: Monotonic clock for instance-local cooldown and circuit backoff.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "OPC UA Writer Sink",
        "port_of": "roboflow_enterprise/opc_writer_sink@v1",
        "section": "industrial",
        "license": "Roboflow Enterprise License",
        "enterprise_only": True,
        "local_only": True,
    }
    outputs = {
        "disabled": Output(
            BOOLEAN_KIND, description="Sink skipped by policy or disable_sink."
        ),
        "throttling_status": Output(
            BOOLEAN_KIND, description="Sink skipped by its per-block cooldown."
        ),
        "error_status": ERROR_STATUS_OUTPUT,
        "message": MESSAGE_OUTPUT,
    }

    class Params(BlockParams):
        url: StrictStr | Ref(SECRET_KIND, STRING_KIND) = Field(
            description="OPC server endpoint; a Secret keeps it out of diagnostics.",
            examples=["opc.tcp://localhost:4840"],
        )
        namespace: StrictStr | Ref(STRING_KIND) = Field(
            description="Namespace URI or decimal UInt16 index.",
            examples=["2", "urn:factory"],
        )
        user_name: Optional[StrictStr] | Ref(
            SECRET_KIND, STRING_KIND, allow_none=True
        ) = Field(default=None, description="Optional authentication username.")
        password: Optional[StrictStr] | Ref(
            SECRET_KIND, STRING_KIND, allow_none=True
        ) = Field(
            default=None,
            description="Optional authentication password; unavailable Secrets fail before dispatch.",
        )
        object_name: StrictStr | Ref(STRING_KIND) = Field(
            description="Object path or direct string identifier.",
            examples=["Line1/Station2"],
        )
        variable_name: StrictStr | Ref(STRING_KIND) = Field(
            description="Variable name; empty is allowed in direct mode.",
            examples=["InspectionSuccess"],
        )
        value: Scalar | Ref(BOOLEAN_KIND, FLOAT_KIND, INTEGER_KIND, STRING_KIND) = (
            Field(
                description="Scalar value; V1 conversions are retained without JSON decoding.",
                examples=[True, "42"],
            )
        )
        value_type: ValueType | Ref(STRING_KIND) = Field(
            default="String",
            description="UA VariantType; Integer aliases Int64.",
            examples=["Int64", "Boolean"],
        )
        timeout: StrictInt | Ref(INTEGER_KIND) = Field(
            default=2,
            description="Positive seconds per request and connection attempt.",
        )
        fire_and_forget: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description="Dispatch to host sink_effects when supplied; otherwise execute synchronously.",
        )
        disable_sink: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False, description="Skip without driver import or Secret reveal."
        )
        cooldown_seconds: StrictInt | Ref(INTEGER_KIND) = Field(
            default=5,
            description="Nonnegative minimum seconds between admitted operations of this block.",
        )
        node_lookup_mode: LookupMode | Ref(STRING_KIND) = Field(
            default="hierarchical",
            description="Browse hierarchical paths or build a direct string NodeId.",
        )
        max_retries: StrictInt | Ref(INTEGER_KIND) = Field(
            default=3,
            description="Positive number of connection attempts; writes are never retried.",
        )
        retry_backoff_seconds: (
            StrictFloat | StrictInt | Ref(FLOAT_KIND, INTEGER_KIND)
        ) = Field(
            default=0.015,
            description="Finite nonnegative exponential retry base in seconds.",
        )

        @field_validator(
            "timeout", "cooldown_seconds", "max_retries", "retry_backoff_seconds"
        )
        @classmethod
        def _timing(cls, value: Any, info: Any) -> Any:
            if isinstance(value, str) and value.startswith("$"):
                return value
            minimum = 1 if info.field_name in ("timeout", "max_retries") else 0
            if value < minimum or not math.isfinite(value):
                raise ValueError(f"{info.field_name} must be finite and >= {minimum}")

            return value

    def __init__(
        self,
        *,
        opc_client: Optional[OPCClient] = None,
        sink_effects: Optional[BackgroundEffects] = None,
        disable_sinks: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = opc_client
        self._sink_effects = sink_effects
        self._disable_sinks = disable_sinks
        self._clock = clock
        self._lock = threading.RLock()
        self._last: Optional[float] = None
        self._backoffs: Dict[bytes, float] = {}

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> List[WorkOperation]:
        """Declare the outbound write.

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
        """Declare cooldown and conditional background failure visibility.

        Args:
            params: Compiled parameters.

        Returns:
            V1 restrictions, incomplete for selected fire_and_forget.
        """
        if isinstance(params.fire_and_forget, str):
            problem = DiscoveryProblem(
                code=DiscoveryProblemCode.UNRESOLVED_SELECTOR,
                description="fire_and_forget is only known at run time",
                details={
                    "declaration": "restrictions",
                    "field": "fire_and_forget",
                    "selector": params.fire_and_forget,
                },
            )
            result = incomplete_discovery([COOLDOWN_ACTUAL_RESTRICTION], [problem])
            return result

        return (
            [COOLDOWN_ACTUAL_RESTRICTION, FIRE_AND_FORGET_RESTRICTION]
            if params.fire_and_forget
            else [COOLDOWN_ACTUAL_RESTRICTION]
        )

    def run(
        self,
        *,
        url: Union[str, Secret],
        namespace: str,
        user_name: Optional[Union[str, Secret]],
        password: Optional[Union[str, Secret]],
        object_name: str,
        variable_name: str,
        value: Any,
        value_type: str = "String",
        timeout: int = 2,
        fire_and_forget: bool = True,
        disable_sink: bool = False,
        cooldown_seconds: int = 5,
        node_lookup_mode: LookupMode = "hierarchical",
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.015,
    ) -> Dict[str, Any]:
        """Validate, snapshot and dispatch one write.

        Args:
            url: Protected endpoint or string.
            namespace: URI or decimal namespace index.
            user_name: Optional protected username.
            password: Optional protected password.
            object_name: Object path or string identifier.
            variable_name: Variable; may be empty for direct lookup.
            value: Primitive value converted with V1 rules.
            value_type: UA type name.
            timeout: Positive seconds per request/connect.
            fire_and_forget: Use host effects when available.
            disable_sink: Skip this step's effect.
            cooldown_seconds: Per-instance monotonic cooldown.
            node_lookup_mode: Direct identifier or hierarchical browse.
            max_retries: Connection attempts, never write attempts.
            retry_backoff_seconds: Connection retry exponential base.

        Returns:
            Disabled, throttling, error flags and a safe message.

        Raises:
            SecretUnavailableError: An enabled, unthrottled credential is unavailable.
        """
        if self._disable_sinks or disable_sink:
            return _outcome(
                False,
                disabled_sink_message(by_policy=self._disable_sinks),
                disabled=True,
            )

        # Serialize admission only. Effect work holds immutable settings, never
        # the engine's mutable selector containers. Reentrant for inline errors.
        with self._lock:
            now = self._clock()
            if self._last is not None and now - self._last < cooldown_seconds:
                return _outcome(False, "Sink cooldown applies", throttled=True)

            try:
                converted, variant = _converted_value(value, value_type=value_type)
                if not isinstance(namespace, str) or not namespace:
                    raise ValueError(
                        "OPC namespace must be a nonempty URI or decimal index"
                    )
                if namespace.isdigit() and not 0 <= int(namespace) <= 65535:
                    raise ValueError("OPC namespace index must be in 0..65535")
                if not object_name or node_lookup_mode not in (
                    "direct",
                    "hierarchical",
                ):
                    raise ValueError("OPC object_name and node_lookup_mode are invalid")
                if node_lookup_mode == "hierarchical" and (
                    not variable_name
                    or any(not part for part in object_name.split("/"))
                ):
                    raise ValueError(
                        "OPC hierarchical lookup requires nonempty object path components and variable_name"
                    )
            except ValueError as error:
                return _outcome(True, str(error))

            target = OPCTarget(
                to_secret(url, name="url"),
                to_secret(user_name, name="user_name"),
                to_secret(password, name="password"),
                timeout,
                max_retries,
                retry_backoff_seconds,
            )
            revealed = reveal_secrets((target.url, target.user_name, target.password))
            identity = hashlib.sha256(json.dumps(revealed.value).encode()).digest()
            self._backoffs = {
                key: until for key, until in self._backoffs.items() if until > now
            }
            if identity in self._backoffs:
                return _outcome(
                    True,
                    "OPC connection backoff applies; retry after the block's connection window",
                )

            request = OPCWrite(
                namespace,
                object_name,
                variable_name,
                converted,
                variant,
                node_lookup_mode,
            )
            client = self._client
            started = threading.Event()

            def effect() -> Tuple[bool, str]:
                started.set()
                try:
                    port = AsyncuaOPCClient() if client is None else client
                    port.write(target, request)
                except OPCSinkError as error:
                    if error.backoff_seconds:
                        with self._lock:
                            self._backoffs[identity] = (
                                self._clock() + error.backoff_seconds
                            )
                    return True, revealed.scrub(str(error))
                except (Exception, asyncio.CancelledError):
                    return (
                        True,
                        "OPC write failed; delivery outcome may be unknown. Check the host port; write was not retried",
                    )

                return False, "Value set successfully"

            try:
                error, message = deliver(
                    effect,
                    fire_and_forget=fire_and_forget,
                    sink_effects=self._sink_effects,
                    label=BLOCK_TYPE,
                )
            except Exception:
                return _outcome(
                    True,
                    "OPC background dispatch failed; check that host effects remain open until sessions stop",
                )

            if started.is_set() or not error:
                self._last = now
            if message == BACKGROUND_MESSAGE:
                message = BACKGROUND_WRITE_MESSAGE

            return _outcome(error, message)
