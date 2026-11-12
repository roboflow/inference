import json
from typing import List, Optional

import pytest
import websocket

from inference_sdk.cloud_streaming.api import CloudStreamingError
from inference_sdk.cloud_streaming.results import ResultGap, ResultsFeed, StreamResult


class Clock:
    """A stand-in for the stop event and the monotonic clock: waits advance time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.waits: List[float] = []

    def __call__(self) -> float:
        return self.now

    def is_set(self) -> bool:
        return False

    def wait(self, seconds: float) -> bool:
        self.waits.append(seconds)
        self.now += seconds
        return False


class Api:
    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.subscribed: List[str] = []

    def subscribe_results(self, job_id: str):
        self.subscribed.append(job_id)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class Socket:
    def __init__(self, *messages) -> None:
        self.messages = [json.dumps(m) for m in messages]

    def recv(self) -> str:
        return self.messages.pop(0) if self.messages else ""

    def close(self) -> None:
        pass


class Connector:
    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.urls: List[str] = []

    def __call__(self, url, header, timeout):
        self.urls.append(url)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def feed_url(generation: int, token: str = "token") -> dict:
    return {
        "state": "running",
        "ended": False,
        "executionGeneration": generation,
        "eventsUrl": "wss://processor/events/ws?job=job-1",
        "accessToken": token,
    }


def result(seq: int) -> dict:
    return {
        "type": "result",
        "seq": seq,
        "result": {
            "frameId": seq * 2,
            "timestamp": "t",
            "latencyMs": 12.5,
            "outputs": {"n": seq},
        },
    }


ENDED = {"state": "cancelled", "ended": True}


def follow(
    api: Api, connector: Connector, clock: Clock, switched: Optional[List[str]] = None
):
    feed = ResultsFeed(api, clock, connect=connector, clock=clock)
    return list(
        feed.follow("job-1", [].append if switched is None else switched.append)
    )


def refused(status: int) -> websocket.WebSocketBadStatusException:
    return websocket.WebSocketBadStatusException("refused", status)


def test_reconnects_after_the_last_result_and_resets_the_cursor_for_a_new_execution() -> (
    None
):
    # given
    api = Api(feed_url(1), feed_url(1), feed_url(2), ENDED)
    connector = Connector(
        Socket({"type": "hello", "executionGeneration": 1}, result(0), result(1)),
        Socket(
            {"type": "gap", "fromSeq": 2, "toSeq": 4},
            result(5),
            {"type": "end", "reason": "replaced"},
        ),
        Socket({"type": "hello", "executionGeneration": 2}, result(0)),
    )

    # when
    items = follow(api, connector, Clock())

    # then
    assert [u.rpartition("?")[2] for u in connector.urls] == [
        "job=job-1",
        "job=job-1&after=1",
        "job=job-1",
    ]
    assert items == [
        StreamResult("job-1", 1, 0, 0, "t", 12.5, {"n": 0}),
        StreamResult("job-1", 1, 1, 2, "t", 12.5, {"n": 1}),
        ResultGap("job-1", 2, 4),
        StreamResult("job-1", 1, 5, 10, "t", 12.5, {"n": 5}),
        StreamResult("job-1", 2, 0, 0, "t", 12.5, {"n": 0}),
    ]


def test_follows_a_successor_job_from_a_fresh_cursor() -> None:
    # given
    api = Api(
        feed_url(1), {"ended": True, "successorJobId": "job-2"}, feed_url(1), ENDED
    )
    connector = Connector(Socket(result(7)), Socket(result(0)))
    switched: List[str] = []

    # when
    items = follow(api, connector, Clock(), switched=switched)

    # then
    assert api.subscribed == ["job-1", "job-1", "job-2", "job-2"]
    assert switched == ["job-2"]
    assert "after" not in connector.urls[1]
    assert [(i.job_id, i.seq) for i in items] == [("job-1", 7), ("job-2", 0)]


def test_waits_as_told_while_the_job_has_no_feed() -> None:
    # given
    clock = Clock()
    api = Api(
        {"ended": False, "retryAfterSeconds": 60},
        {"ended": False, "retryAfterSeconds": 2},
        ENDED,
    )

    # when
    items = follow(api, Connector(), clock)

    # then
    assert items == []
    assert clock.waits == [60, 2]


@pytest.mark.parametrize(
    "status, expected_waits",
    [(401, [2.0, 4.0]), (404, [2.0, 4.0]), (429, [5.0, 5.0])],
)
def test_backs_off_and_subscribes_again_after_the_processor_refuses(
    status, expected_waits
) -> None:
    # given
    clock = Clock()
    api = Api(feed_url(1), feed_url(1), feed_url(1), ENDED)
    connector = Connector(refused(status), refused(status), Socket(result(0)))

    # when
    items = follow(api, connector, clock)

    # then
    assert [i.seq for i in items] == [0]
    assert [w for w in clock.waits if w] == expected_waits


def test_keeps_connection_attempts_a_second_apart() -> None:
    # given: the feed closes right away twice
    clock = Clock()
    api = Api(feed_url(1), feed_url(1), ENDED)

    # when
    follow(api, Connector(Socket(), Socket()), clock)

    # then
    assert clock.waits == [0.0, 1.0]


def test_a_malformed_upgrade_fails_the_stream() -> None:
    with pytest.raises(CloudStreamingError):
        follow(Api(feed_url(1)), Connector(refused(400)), Clock())


def test_retries_transient_subscribe_failures_and_raises_the_rest() -> None:
    # given
    clock = Clock()
    transient = Api(
        CloudStreamingError("down", status=503), CloudStreamingError("offline"), ENDED
    )

    # when / then
    assert follow(transient, Connector(), clock) == []
    assert clock.waits == [2.0, 2.0]
    with pytest.raises(CloudStreamingError):
        follow(Api(CloudStreamingError("no", status=403)), Connector(), Clock())
