"""Per-frame Workflow results from a job's results feed.

The loop follows the platform contract: subscribe, read the WebSocket at `eventsUrl`, and after
any close or refusal subscribe again, so a recovered job (new processor, new token) or a
successor job is followed without the caller noticing more than a `ResultGap`.
"""

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Union

import websocket

from inference_sdk.cloud_streaming.api import CloudStreamingAPI, CloudStreamingError

SUBSCRIBE_RETRY_S = 2.0
SUBSCRIBE_RETRY_STATUSES = {408, 429}
MIN_CONNECT_INTERVAL_S = 1.0
MAX_REFUSAL_BACKOFF_S = 30.0
# The processor pings every 15 s, so a minute of silence means the connection is gone.
READ_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class StreamResult:
    """One processed frame. `timestamp` is when the result was published (ISO 8601, UTC)."""

    job_id: str
    execution_generation: Optional[int]
    seq: int
    frame_id: int
    timestamp: str
    latency_ms: Optional[float]
    outputs: Dict[str, Any]


@dataclass(frozen=True)
class ResultGap:
    """Results `from_seq` to `to_seq` (inclusive) of one execution were not delivered."""

    job_id: str
    from_seq: int
    to_seq: int


class ResultsFeed:
    def __init__(
        self,
        api: CloudStreamingAPI,
        stopped: threading.Event,
        connect: Callable[..., Any] = websocket.create_connection,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._api = api
        self._stopped = stopped
        self._connect = connect
        self._clock = clock
        self._socket = None
        self._lock = threading.Lock()

    def close(self) -> None:
        """Unblock a read in progress; the loop then sees that the stream stopped."""
        with self._lock:
            if self._socket is not None:
                self._socket.close()

    def follow(
        self, job_id: str, on_successor: Callable[[str], None]
    ) -> Iterator[Union[StreamResult, ResultGap]]:
        """Results until the job ends or the stream stops; `on_successor` hears about switches."""
        generation = None
        last_seq = None
        refusals = 0
        last_connect = float("-inf")
        while not self._stopped.is_set():
            subscription = self._subscribe(job_id)
            if subscription is None:
                return
            if subscription.get("ended"):
                successor = subscription.get("successorJobId")
                if not successor:
                    return
                job_id, generation, last_seq = successor, None, None
                on_successor(job_id)
                continue
            if "eventsUrl" not in subscription:
                self._stopped.wait(
                    subscription.get("retryAfterSeconds", SUBSCRIBE_RETRY_S)
                )
                continue
            if subscription.get("executionGeneration") != generation:
                generation, last_seq = subscription.get("executionGeneration"), None
            self._stopped.wait(
                max(0.0, last_connect + MIN_CONNECT_INTERVAL_S - self._clock())
            )
            if self._stopped.is_set():
                return
            last_connect = self._clock()
            url = subscription["eventsUrl"]
            if last_seq is not None:
                url += f"&after={last_seq}"
            try:
                socket = self._connect(
                    url,
                    header=[f"Authorization: Bearer {subscription['accessToken']}"],
                    timeout=READ_TIMEOUT_S,
                )
            except websocket.WebSocketBadStatusException as error:
                if error.status_code == 400:
                    raise CloudStreamingError(
                        "the results feed refused the connection as malformed",
                        status=400,
                    ) from error
                refusals += 1
                self._stopped.wait(_refusal_backoff(refusals, error.status_code))
                continue
            except (OSError, websocket.WebSocketException):
                refusals += 1
                self._stopped.wait(_refusal_backoff(refusals, None))
                continue
            refusals = 0
            with self._lock:
                if self._stopped.is_set():
                    socket.close()
                    return
                self._socket = socket
            try:
                for message in _messages(socket):
                    if self._stopped.is_set():
                        return
                    kind = message.get("type")
                    if kind == "hello":
                        if message.get("executionGeneration") != generation:
                            generation, last_seq = (
                                message.get("executionGeneration"),
                                None,
                            )
                    elif kind == "result":
                        last_seq = message["seq"]
                        result = message["result"]
                        yield StreamResult(
                            job_id=job_id,
                            execution_generation=generation,
                            seq=message["seq"],
                            frame_id=result["frameId"],
                            timestamp=result["timestamp"],
                            latency_ms=result.get("latencyMs"),
                            outputs=result["outputs"],
                        )
                    elif kind == "gap":
                        yield ResultGap(
                            job_id=job_id,
                            from_seq=message["fromSeq"],
                            to_seq=message["toSeq"],
                        )
                    elif kind == "end":
                        break
            finally:
                with self._lock:
                    self._socket = None
                socket.close()

    def _subscribe(self, job_id: str) -> Optional[Dict[str, Any]]:
        while not self._stopped.is_set():
            try:
                return self._api.subscribe_results(job_id)
            except CloudStreamingError as error:
                if not _retryable(error.status):
                    raise
            self._stopped.wait(SUBSCRIBE_RETRY_S)
        return None


def _messages(socket) -> Iterator[Dict[str, Any]]:
    while True:
        try:
            text = socket.recv()
        except (OSError, websocket.WebSocketException):
            return
        if not text:
            return
        yield json.loads(text)


def _retryable(status: Optional[int]) -> bool:
    return status is None or status >= 500 or status in SUBSCRIBE_RETRY_STATUSES


def _refusal_backoff(refusals: int, status: Optional[int]) -> float:
    floor = 5.0 if status == 429 else MIN_CONNECT_INTERVAL_S
    return max(floor, min(MAX_REFUSAL_BACKOFF_S, 2.0**refusals))
