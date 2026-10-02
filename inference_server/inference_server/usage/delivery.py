"""Hand-over of recorded usage: pending list, queue, sender and shutdown."""

import json
import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from queue import Empty
from threading import Event, Lock, Thread
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Union,
)
from urllib.parse import urlparse

import requests

from inference_server import configuration
from inference_server.platform_http import wrap_url
from inference_server.usage.payload_helpers import (
    APIKey,
    APIKeyHash,
    APIKeyUsage,
    Usage,
    UsagePayload,
    sha256_hash,
    zip_usage_payloads,
)
from inference_server.usage.queues import UNCONFIRMED, PutUnconfirmed

logger = logging.getLogger(__name__)

MAX_PENDING_ROWS = 16384
STOP_TIMEOUT_S = 30.0
COLLECTOR_THREAD_NAME = "usage-collector"
SENDER_THREAD_NAME = "usage-sender"
DROP_LOG_LINE = "Usage rows were dropped because the pending list is full"
UNCONFIRMED_LOG_LINE = "Usage rows were given up because the queue write is unconfirmed"
STEP_FAILED_LOG_LINE = "Usage reporting step failed: %s"
REQUEST_TIMEOUT_S = 1
INFERENCE_VERSION_HEADER = "X-Roboflow-Inference-Version"
ALLOW_CHUNKED_RESPONSE_HEADER = "X-Allow-Chunked"

DetachWindow = Callable[..., bool]
Stored = Union[bool, PutUnconfirmed]


@dataclass(eq=False)
class PendingItem:
    """Rows of one detached window or one closed row, counted once.

    Attributes:
        payload: Rows by API key hash and usage key.
        rows: Number of rows in the payload.
        frames: Sum of their processed frames.
    """

    payload: APIKeyUsage
    rows: int
    frames: int


class _Write(Enum):
    WRITTEN = "written"
    UNCONFIRMED = "unconfirmed"
    EMPTY = "empty"
    REFUSED = "refused"


class _Room(Enum):
    MADE = "made"
    BLOCKED = "blocked"
    FAILED = "failed"


class _SendGate:
    def __init__(self, may_start: Callable[[], bool]) -> None:
        self._may_start = may_start
        self.refused = False

    def __call__(self) -> bool:
        allowed = self.open()
        if not allowed:
            self.refused = True

        return allowed

    def open(self) -> bool:
        return self._may_start()


def pending_item(payload: APIKeyUsage) -> Optional[PendingItem]:
    """Count the rows and frames of a payload.

    Args:
        payload: Rows by API key hash and usage key.

    Returns:
        The payload wrapped for the pending list, or None when it has no row.
    """
    rows = 0
    frames = 0
    for resource_rows in payload.values():
        if not isinstance(resource_rows, dict):
            continue
        for row in resource_rows.values():
            rows += 1
            if isinstance(row, dict):
                frames += row.get("processed_frames", 0) or 0
    if not rows:
        return None

    item = PendingItem(payload=payload, rows=rows, frames=frames)

    return item


def guarded(step: Callable[[], Any]) -> None:
    """Run a step and log, without its text, an exception it raises.

    Args:
        step: Step to run.
    """
    try:
        step()
    except Exception as error:
        logger.error(STEP_FAILED_LOG_LINE, type(error).__name__)


def _always() -> bool:
    return True


@contextmanager
def lock_guard(lock: Any, deadline: Optional[float]) -> Iterator[bool]:
    """Hold a lock for a block, waiting for it at most until a deadline.

    Args:
        lock: Lock to take.
        deadline: Monotonic time after which the lock is no longer waited for;
            None waits without a limit.

    Yields:
        Whether the lock is held; the block must not touch what the lock
        protects when it is not.
    """
    if deadline is None:
        held = lock.acquire()
    else:
        held = lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
    try:
        yield held
    finally:
        if held:
            lock.release()


def ssl_verify_for_endpoint(url: str) -> bool:
    """Tell whether TLS certificates are verified for a usage endpoint.

    Args:
        url: URL that will be requested, after any secure gateway wrapping.

    Returns:
        False only when the host is ``localhost`` or ``127.0.0.1``.
    """
    try:
        hostname = urlparse(url).hostname or ""
    except ValueError:
        return True
    return hostname.lower() not in {"localhost", "127.0.0.1"}


def usage_request_headers() -> Dict[str, Any]:
    """Build the headers sent with every usage request besides authorization.

    Returns:
        ``ROBOFLOW_API_EXTRA_HEADERS`` overridden by the server version and
        chunked-response markers.
    """
    headers = {
        INFERENCE_VERSION_HEADER: configuration.SERVER_VERSION,
        ALLOW_CHUNKED_RESPONSE_HEADER: "true",
    }
    if not configuration.ROBOFLOW_API_EXTRA_HEADERS:
        return headers

    try:
        extra_headers: dict = json.loads(configuration.ROBOFLOW_API_EXTRA_HEADERS)
    except ValueError:
        logger.warning("Could not decode ROBOFLOW_API_EXTRA_HEADERS")
        return headers
    extra_headers.update(headers)

    return extra_headers


def _outbound_row(row: Usage, *, api_key: APIKey) -> Dict[str, Any]:
    outbound_row = {key: value for key, value in row.items() if key != "api_key_hash"}
    stream_session_id = outbound_row.pop("stream_session_id", None)
    if stream_session_id:
        outbound_row["exec_session_id"] = stream_session_id
    outbound_row["api_key"] = api_key

    return outbound_row


def send_usage_payload(
    payload: UsagePayload,
    api_usage_endpoint_url: str,
    hashes_to_api_keys: Optional[Dict[APIKeyHash, APIKey]] = None,
    ssl_verify: bool = False,
    extra_headers: Optional[Dict[str, str]] = None,
    may_post: Optional[Callable[[], bool]] = None,
) -> Set[APIKeyHash]:
    """Post the rows of one payload, one request per API key.

    The payload is left untouched; the posted rows are copies carrying the API
    key instead of its hash.

    Args:
        payload: Rows keyed by API key hash, then by usage key.
        api_usage_endpoint_url: URL the rows are posted to.
        hashes_to_api_keys: API keys by their hash; a hash missing from a
            non-empty mapping is not sent.
        ssl_verify: Whether TLS certificates are verified.
        extra_headers: Headers added to the authorization header.
        may_post: Asked before every request; a request is not started once it
            answers False and the key counts as not accepted.

    Returns:
        API key hashes whose rows were not accepted (anything but HTTP 200, or
        no request started).
    """
    if configuration.LEGACY_OFFLINE_MODE:
        return set(payload.keys())
    hashes_to_api_keys = hashes_to_api_keys or {}
    api_keys_hashes_failed = set()
    for api_key_hash, workflow_payloads in payload.items():
        if hashes_to_api_keys and api_key_hash not in hashes_to_api_keys:
            api_keys_hashes_failed.add(api_key_hash)
            continue
        api_key = hashes_to_api_keys.get(api_key_hash) or api_key_hash
        if not api_key:
            api_keys_hashes_failed.add(api_key_hash)
            continue
        if may_post is not None and not may_post():
            api_keys_hashes_failed.add(api_key_hash)
            continue
        try:
            complete_workflow_payloads = [
                _outbound_row(w, api_key=api_key)
                for w in workflow_payloads.values()
                if "processed_frames" in w
            ]
            if not extra_headers:
                extra_headers = {}
            response = requests.post(
                api_usage_endpoint_url,
                json=complete_workflow_payloads,
                verify=ssl_verify,
                headers={"Authorization": f"Bearer {api_key}", **extra_headers},
                timeout=REQUEST_TIMEOUT_S,
            )
        except Exception:
            api_keys_hashes_failed.add(api_key_hash)
            continue
        if response.status_code != 200:
            api_keys_hashes_failed.add(api_key_hash)
            continue
    return api_keys_hashes_failed


class Delivery:
    """Moves recorded rows from the pending list to the queue and to the platform.

    A row is owned by exactly one stage at a time: the live window of the
    collector, the pending list, the queue, or the sender that took it from the
    queue. A stage gives a row up only after the next one confirmed it, except
    rows counted as dropped and rows counted as unconfirmed, which are not kept.
    """

    def __init__(
        self,
        queue: Any,
        *,
        detach_window: DetachWindow,
        resolve_host_in_background: Callable[[], None],
        host_values: Callable[[Optional[float]], Dict[str, Any]],
        api_keys_hashing_enabled: bool,
    ) -> None:
        """Bind the delivery to its queue and to the collector it serves.

        Args:
            queue: Queue payloads are written to and read from.
            detach_window: Moves the live window of the collector to the
                pending list.
            resolve_host_in_background: Run once by the collector thread before
                its loop.
            host_values: Host description for rows written to the queue.
            api_keys_hashing_enabled: Whether API keys are replaced by their
                hash in rows.
        """
        self.queue = queue
        self._detach_window = detach_window
        self._resolve_host_in_background = resolve_host_in_background
        self._host_values = host_values
        self._api_keys_hashing_enabled = api_keys_hashing_enabled

        self._api_keys_lock = Lock()
        self._hashed_api_keys: Dict[APIKey, APIKeyHash] = {}

        self._pending_lock = Lock()
        self._pending: Deque[PendingItem] = deque()
        self._pending_rows = 0
        self._writing: Optional[PendingItem] = None
        self._last_log: Dict[str, float] = {}
        self.dropped_rows = 0
        self.dropped_frames = 0
        self.unconfirmed_rows = 0
        self.unconfirmed_frames = 0
        self.inline_queue_writes = 0

        self._queue_lock = Lock()

        self._wake = Event()
        self._stopping = Event()
        self._collector_thread: Optional[Thread] = None
        self._sender_thread: Optional[Thread] = None

    @property
    def pending(self) -> List[PendingItem]:
        """Items waiting for a queue write, oldest first."""
        with self._pending_lock:
            items = list(self._pending)

        return items

    @property
    def pending_rows(self) -> int:
        """Rows held by the pending list."""
        with self._pending_lock:
            rows = self._pending_rows

        return rows

    @property
    def started(self) -> bool:
        """Whether the background threads were started."""
        return self._collector_thread is not None

    @property
    def threads(self) -> List[Thread]:
        """The background threads that were started."""
        threads = [
            thread
            for thread in (self._collector_thread, self._sender_thread)
            if thread is not None
        ]

        return threads

    def register_api_key(self, api_key: APIKey) -> APIKeyHash:
        """Remember an API key and return the hash rows carry instead of it.

        Args:
            api_key: API key to register.

        Returns:
            The key itself unless hashing is enabled, otherwise its SHA-256
            hexadecimal digest without the last character.
        """
        with self._api_keys_lock:
            api_key_hash = self._hashed_api_keys.get(api_key)
            if not api_key_hash:
                if self._api_keys_hashing_enabled:
                    api_key_hash = sha256_hash(api_key, length=-1)
                else:
                    api_key_hash = api_key
                self._hashed_api_keys[api_key] = api_key_hash
        return api_key_hash

    def _api_keys_by_hash(self) -> Dict[APIKeyHash, APIKey]:
        with self._api_keys_lock:
            api_keys = {
                api_key_hash: api_key
                for api_key, api_key_hash in self._hashed_api_keys.items()
            }

        return api_keys

    def _running(self) -> bool:
        return not self._stopping.is_set()

    def _has_pending(self) -> bool:
        with self._pending_lock:
            has_pending = bool(self._pending)

        return has_pending

    def _has_room_pending_locked(self, rows: int) -> bool:
        room = self._pending_rows + rows <= MAX_PENDING_ROWS or not self._pending

        return room

    def _log_due_pending_locked(self, line: str) -> bool:
        now = time.monotonic()
        last = self._last_log.get(line)
        due = last is None or now - last >= configuration.TELEMETRY_FLUSH_INTERVAL
        if due:
            self._last_log[line] = now

        return due

    def _drop_oldest_pending_locked(self, rows: int) -> bool:
        dropped = False
        while not self._has_room_pending_locked(rows):
            if self._pending[0] is not self._writing:
                victim = self._pending.popleft()
            elif len(self._pending) > 1:
                writing = self._pending.popleft()
                victim = self._pending.popleft()
                self._pending.appendleft(writing)
            else:
                break
            self._pending_rows -= victim.rows
            self.dropped_rows += victim.rows
            self.dropped_frames += victim.frames
            dropped = True

        return dropped

    def _drop_oldest(self, rows: int) -> bool:
        with self._pending_lock:
            dropped = self._drop_oldest_pending_locked(rows)
            should_log = dropped and self._log_due_pending_locked(DROP_LOG_LINE)
            room = self._has_room_pending_locked(rows)
        if should_log:
            logger.error(DROP_LOG_LINE)

        return dropped or room

    def try_add(self, item: PendingItem) -> bool:
        """Append an item when the bound leaves room for its rows.

        An item larger than the bound alone is admitted into an empty list.

        Args:
            item: Rows to hold.

        Returns:
            True when the item was added.
        """
        with self._pending_lock:
            added = self._has_room_pending_locked(item.rows)
            if added:
                self._pending.append(item)
                self._pending_rows += item.rows
        if added:
            self._wake.set()

        return added

    def add(self, item: PendingItem) -> None:
        """Append an item, making room for it first when the list is full.

        Args:
            item: Rows to hold.
        """
        while not self.try_add(item):
            self.make_room(item.rows, write=True)

    def hand_back(self, payloads: List[APIKeyUsage]) -> None:
        """Put payloads the sender could not deliver in front of the list.

        Args:
            payloads: Payloads in the order they were taken from the queue.
        """
        items = [item for item in map(pending_item, payloads) if item is not None]
        with self._pending_lock:
            for item in reversed(items):
                self._pending.appendleft(item)
                self._pending_rows += item.rows
            dropped = self._drop_oldest_pending_locked(0)
            should_log = dropped and self._log_due_pending_locked(DROP_LOG_LINE)
        if should_log:
            logger.error(DROP_LOG_LINE)
        if items:
            self._wake.set()

    def make_room(
        self,
        rows: int,
        *,
        write: bool,
        deadline: Optional[float] = None,
        may_write: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """Free room for rows by writing the oldest items, else dropping them.

        Args:
            rows: Rows that are about to be added.
            write: Whether queue writes are allowed; without them the oldest
                items are dropped at once.
            deadline: Monotonic time after which no lock is waited for.
            may_write: Asked before every queue write.

        Returns:
            False when nothing was written or dropped and there is no room, or
            when a write was not allowed; True otherwise.
        """
        if write:
            outcome = self._write_oldest(rows, deadline, may_write)
            if outcome is _Room.MADE:
                return True
            if outcome is _Room.BLOCKED:
                return False
        made = self._drop_oldest(rows)

        return made

    def _write_oldest(
        self,
        rows: int,
        deadline: Optional[float],
        may_write: Optional[Callable[[], bool]],
    ) -> _Room:
        if not self._has_pending():
            return _Room.FAILED

        allowed = may_write or _always
        try:
            host = self._host_values(deadline)
            with lock_guard(self._queue_lock, deadline) as held:
                if not held:
                    return _Room.BLOCKED
                while True:
                    with self._pending_lock:
                        room = self._has_room_pending_locked(rows)
                    if room:
                        return _Room.MADE
                    if not allowed():
                        return _Room.BLOCKED
                    result = self._write_front_queue_locked(host)
                    if result is _Write.WRITTEN:
                        self.inline_queue_writes += 1
                    elif result is not _Write.UNCONFIRMED:
                        return _Room.FAILED
        except Exception as error:
            logger.error(STEP_FAILED_LOG_LINE, type(error).__name__)

            return _Room.FAILED

    @staticmethod
    def _complete_rows(payload: UsagePayload, host: Dict[str, Any]) -> None:
        for rows in payload.values():
            for row in rows.values():
                if isinstance(row, dict) and not row.get("hostname"):
                    row["hostname"] = host["hostname"]
                    row["ip_address_hash"] = host["ip_address_hash"]
                    row["is_gpu_available"] = host["is_gpu_available"]

    def _claim_front(self) -> Optional[PendingItem]:
        with self._pending_lock:
            item = self._pending[0] if self._pending else None
            self._writing = item

        return item

    def _settle(self, item: PendingItem, stored: Stored) -> None:
        with self._pending_lock:
            self._writing = None
            if stored is not False:
                self._pending.remove(item)
                self._pending_rows -= item.rows
        if stored is UNCONFIRMED:
            self._give_up_unconfirmed(item)

    def _give_up_unconfirmed(self, item: PendingItem) -> None:
        with self._pending_lock:
            self.unconfirmed_rows += item.rows
            self.unconfirmed_frames += item.frames
            should_log = self._log_due_pending_locked(UNCONFIRMED_LOG_LINE)
        if should_log:
            logger.error(UNCONFIRMED_LOG_LINE)

    def _write_front_queue_locked(self, host: Dict[str, Any]) -> _Write:
        item = self._claim_front()
        if item is None:
            return _Write.EMPTY

        stored: Stored = False
        try:
            self._complete_rows(item.payload, host)
            stored = self._write_payload_queue_locked(item.payload)
        except Exception as error:
            logger.error(STEP_FAILED_LOG_LINE, type(error).__name__)
        finally:
            self._settle(item, stored)
        if stored is UNCONFIRMED:
            return _Write.UNCONFIRMED

        return _Write.WRITTEN if stored else _Write.REFUSED

    def _put(self, payload: Any) -> Stored:
        stored = self.queue.put(payload)
        if stored is UNCONFIRMED:
            return UNCONFIRMED

        return stored is not False

    def _read_budget(self) -> Optional[int]:
        qsize = getattr(self.queue, "qsize", None)
        if qsize is None:
            return None
        batch = getattr(self.queue, "read_batch", 1)
        budget = -(-qsize() // batch)

        return budget

    def _take_raw_queue_locked(
        self, may_read: Optional[Callable[[], bool]] = None
    ) -> List[APIKeyUsage]:
        usage_payloads: List[APIKeyUsage] = []
        budget = self._read_budget()
        reads = 0
        while budget is None or reads < budget:
            if may_read is not None and not may_read():
                break
            if self.queue.empty():
                break
            reads += 1
            try:
                payload = self.queue.get_nowait()
            except Empty:
                break
            if isinstance(payload, list):
                if not payload:
                    break
                usage_payloads.extend(payload)
            elif payload:
                usage_payloads.append(payload)

        return usage_payloads

    def _restore_taken_queue_locked(self, payloads: List[APIKeyUsage]) -> None:
        if not payloads:
            return

        try:
            restored = self._put(payloads)
        except Exception:
            restored = False
        if not restored:
            rows = [pending_item(payload) for payload in payloads]
            with self._pending_lock:
                for item in rows:
                    if item is not None:
                        self.dropped_rows += item.rows
                        self.dropped_frames += item.frames
                should_log = self._log_due_pending_locked(DROP_LOG_LINE)
            if should_log:
                logger.error(DROP_LOG_LINE)

    def _write_payload_queue_locked(self, payload: UsagePayload) -> Stored:
        if not payload:
            return True
        if not self.queue.full():
            stored = self._put(payload)

            return stored

        dumped = self._take_raw_queue_locked()
        try:
            merged = zip_usage_payloads(usage_payloads=[*dumped, payload])
            stored = self._put(merged) if merged else True
        except BaseException:
            self._restore_taken_queue_locked(dumped)
            raise
        if not stored:
            self._restore_taken_queue_locked(dumped)

        return stored

    def enqueue(self, payload: UsagePayload) -> bool:
        """Write a payload to the queue.

        Args:
            payload: Rows by API key hash and usage key.

        Returns:
            True when the queue confirmed it stored the payload.
        """
        with self._queue_lock:
            stored = self._write_payload_queue_locked(payload)

        return stored is True

    def _normalise_legacy_rows(self, payload: Any) -> Any:
        if not isinstance(payload, dict) or not any(
            isinstance(rows, dict)
            and any(isinstance(row, dict) and "api_key" in row for row in rows.values())
            for rows in payload.values()
        ):
            return payload

        normalised: Dict[Any, Dict[Any, Any]] = {}
        for outer_key, rows in payload.items():
            for row_key, row in rows.items():
                key_hash = outer_key
                if isinstance(row, dict) and "api_key" in row:
                    api_key = row["api_key"]
                    row = {
                        name: value for name, value in row.items() if name != "api_key"
                    }
                    if api_key and isinstance(api_key, str):
                        key_hash = self.register_api_key(api_key)
                        row["api_key_hash"] = key_hash
                normalised.setdefault(key_hash, {})[row_key] = row

        return normalised

    def _normalised(self, payload: Any) -> Any:
        try:
            normalised = self._normalise_legacy_rows(payload)
        except Exception as error:
            logger.error(STEP_FAILED_LOG_LINE, type(error).__name__)

            return payload

        return normalised

    def _take_normalised_queue_locked(
        self, may_read: Optional[Callable[[], bool]] = None
    ) -> List[APIKeyUsage]:
        payloads = [
            self._normalised(payload)
            for payload in self._take_raw_queue_locked(may_read)
        ]

        return payloads

    def dump_queue(self) -> List[APIKeyUsage]:
        """Take the payloads out of the queue, reading no more than it held.

        Returns:
            The payloads, rows written by a legacy server carrying the hash of
            their API key instead of the key.
        """
        with self._queue_lock:
            payloads = self._take_normalised_queue_locked()

        return payloads

    def drain_pending(
        self,
        may_continue: Optional[Callable[[], bool]] = None,
        *,
        deadline: Optional[float] = None,
    ) -> bool:
        """Write the items pending now to the queue, oldest first.

        The pass is bounded by the number of items pending when it starts; items
        that arrive while it runs are left for the next call. It stops earlier
        when the pending list runs empty, a queue write is refused, or the
        lock or ``may_continue`` does not allow the next write.

        Args:
            may_continue: Asked before every queue write.
            deadline: Monotonic time after which no lock is waited for.

        Returns:
            False when a write was refused or not allowed before the pass ended,
            True otherwise.
        """
        with self._pending_lock:
            count = len(self._pending)
        if not count:
            return True

        allowed = may_continue or _always
        host = self._host_values(deadline)
        for _ in range(count):
            with lock_guard(self._queue_lock, deadline) as held:
                if not held or not allowed():
                    return False
                result = self._write_front_queue_locked(host)
            if result is _Write.EMPTY:
                break
            if result is _Write.REFUSED:
                return False

        return True

    def send_queued(
        self,
        may_start: Optional[Callable[[], bool]] = None,
        *,
        deadline: Optional[float] = None,
    ) -> bool:
        """Take payloads from the queue and post them, one request per API key.

        The pass is bounded: it reads as many batches as the queue held when it
        started, and stops earlier when the queue runs empty, an empty batch is
        read or ``may_start`` refuses. Rows not read stay queued. The payloads
        read are merged into one payload per execution session, streams apart
        from images, with additional payloads for rows closed by the list
        bound. Rows the platform did not accept go back to the queue while
        ``may_start`` allows queue writes, otherwise to the pending list.

        Args:
            may_start: Asked before every queue read and request.
            deadline: Monotonic time after which no lock is waited for.

        Returns:
            True when no operation was refused by ``may_start``.
        """
        if configuration.LEGACY_OFFLINE_MODE:
            return True

        gate = _SendGate(may_start or _always)
        with lock_guard(self._queue_lock, deadline) as held:
            if not held:
                return False
            payloads = self._take_normalised_queue_locked(gate)
        if not payloads:
            return not gate.refused

        self._deliver(payloads, gate, deadline)

        return not gate.refused

    def _deliver(
        self,
        payloads: List[APIKeyUsage],
        gate: _SendGate,
        deadline: Optional[float],
    ) -> None:
        unsent = payloads
        try:
            unsent = zip_usage_payloads(usage_payloads=payloads)
            self._post_all(unsent, gate)
        finally:
            self._return_unsent(unsent, gate, deadline)

    def _post_all(self, payloads: List[APIKeyUsage], gate: _SendGate) -> None:
        api_usage_endpoint_url = wrap_url(
            configuration.TELEMETRY_API_USAGE_ENDPOINT_URL
        )
        ssl_verify = ssl_verify_for_endpoint(api_usage_endpoint_url)
        hashes_to_api_keys = self._api_keys_by_hash()
        extra_headers = usage_request_headers()

        for payload in payloads:
            api_keys_hashes_failed = send_usage_payload(
                payload=payload,
                api_usage_endpoint_url=api_usage_endpoint_url,
                hashes_to_api_keys=hashes_to_api_keys,
                ssl_verify=ssl_verify,
                extra_headers=extra_headers,
                may_post=gate,
            )
            if api_keys_hashes_failed:
                logger.debug(
                    "Failed to send usage of %s API key(s)",
                    len(api_keys_hashes_failed),
                )
            for api_key_hash in list(payload.keys()):
                if api_key_hash not in api_keys_hashes_failed:
                    del payload[api_key_hash]
            if gate.refused:
                break

    def _requeue(
        self, payload: APIKeyUsage, gate: _SendGate, deadline: Optional[float]
    ) -> bool:
        stored = False
        with lock_guard(self._queue_lock, deadline) as held:
            if held and gate.open():
                try:
                    outcome = self._write_payload_queue_locked(payload)
                except Exception as error:
                    logger.error(STEP_FAILED_LOG_LINE, type(error).__name__)
                else:
                    if outcome is UNCONFIRMED:
                        given_up = pending_item(payload)
                        if given_up is not None:
                            self._give_up_unconfirmed(given_up)
                    stored = outcome is not False

        return stored

    def _return_unsent(
        self,
        unsent: List[APIKeyUsage],
        gate: _SendGate,
        deadline: Optional[float],
    ) -> None:
        handed_back: List[APIKeyUsage] = []
        queue_usable = True
        for payload in unsent:
            if not payload:
                continue
            if queue_usable and gate.open():
                if self._requeue(payload, gate, deadline):
                    continue
                queue_usable = False
            handed_back.append(payload)
        if handed_back:
            self.hand_back(handed_back)

    def _detach_in_background(self) -> None:
        self._detach_window(
            write=True, deadline=None, may_write=self._running, final=False
        )

    def _drain_in_background(self) -> None:
        self.drain_pending(self._running)

    def _send_in_background(self) -> None:
        self.send_queued(self._running)

    def _collector_loop(self) -> None:
        guarded(self._resolve_host_in_background)
        next_window_at = time.monotonic() + configuration.TELEMETRY_FLUSH_INTERVAL
        while not self._stopping.is_set():
            woken = self._wake.wait(max(0.0, next_window_at - time.monotonic()))
            if self._stopping.is_set():
                break
            if woken:
                self._wake.clear()
            if time.monotonic() >= next_window_at:
                guarded(self._detach_in_background)
                next_window_at = (
                    time.monotonic() + configuration.TELEMETRY_FLUSH_INTERVAL
                )
            guarded(self._drain_in_background)

    def _sender_loop(self) -> None:
        while not self._stopping.wait(configuration.TELEMETRY_FLUSH_INTERVAL):
            guarded(self._send_in_background)

    def start(self) -> None:
        """Start the collector and sender threads; a second call does nothing."""
        if self._collector_thread is not None:
            return

        self._collector_thread = Thread(
            target=self._collector_loop, name=COLLECTOR_THREAD_NAME, daemon=True
        )
        self._sender_thread = Thread(
            target=self._sender_loop, name=SENDER_THREAD_NAME, daemon=True
        )
        self._collector_thread.start()
        self._sender_thread.start()

    def stop(self, timeout: float) -> bool:
        """Stop the threads, then hand over everything and make one send pass.

        Args:
            timeout: Seconds the whole call may take.

        Returns:
            True when both threads exited (or none was started), nothing is
            left pending and every request of the final pass was started.
        """
        deadline = time.monotonic() + timeout
        threads = self.threads
        self._stopping.set()
        self._wake.set()
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            guarded(
                lambda: self._detach_window(
                    write=False, deadline=deadline, may_write=None, final=True
                )
            )

            return False

        try:
            complete = self._final_pass(deadline)
        except Exception as error:
            logger.error(STEP_FAILED_LOG_LINE, type(error).__name__)
            complete = False

        return complete

    def _final_pass(self, deadline: float) -> bool:
        def before_deadline() -> bool:
            return time.monotonic() < deadline

        detached = self._detach_window(
            write=True, deadline=deadline, may_write=before_deadline, final=True
        )
        drained = self.drain_pending(before_deadline, deadline=deadline)
        sent = self.send_queued(before_deadline, deadline=deadline)
        complete = detached and drained and sent and not self._has_pending()

        return complete
