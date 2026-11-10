import threading
import time
from typing import Dict, List, Optional

import pytest

from inference_sdk.cloud_streaming import stream as stream_module
from inference_sdk.cloud_streaming.api import CloudStreamingError
from inference_sdk.cloud_streaming.stream import CloudStream


class Api:
    def __init__(
        self, jobs: Optional[Dict[str, dict]] = None, held: bool = True
    ) -> None:
        self.jobs = jobs or {}
        self.held = held
        self.calls: List[tuple] = []

    def get_job(self, job_id: str) -> dict:
        return self.jobs[job_id]

    def hold_preview(self, job_id: str) -> bool:
        self.calls.append(("hold_preview", job_id))
        return self.held

    def watch(self, job_id: str, output=None) -> dict:
        self.calls.append(("watch", job_id))
        return {}

    def cancel_job(self, job_id: str) -> dict:
        self.calls.append(("cancel_job", job_id))
        return {}

    def stop_session(self, source_id: str) -> None:
        self.calls.append(("stop_session", source_id))

    def subscribe_results(self, job_id: str) -> dict:
        return {"state": self.jobs[job_id]["state"], "ended": True}


class Publisher:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.error: Optional[BaseException] = None
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True
        self.done.set()


def job(job_id: str = "job-1", purpose: str = "preview", **fields) -> dict:
    return {
        "id": job_id,
        "sourceId": "tmp-1",
        "temporarySessionId": "tmp-1",
        "purpose": purpose,
        "state": "running",
        **fields,
    }


def until(condition) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline, "condition not met"
        time.sleep(0.01)


def test_stop_ends_what_the_stream_started_once() -> None:
    api = Api({"job-1": job()})
    publisher = Publisher()
    stream = CloudStream(
        api, job(), owns_job=True, session_id="tmp-1", publisher=publisher
    )

    stream.stop()
    stream.stop()

    assert [c for c in api.calls if c[0] != "hold_preview"] == [
        ("cancel_job", "job-1"),
        ("stop_session", "tmp-1"),
    ]
    assert publisher.stopped


def test_an_attached_stream_never_cancels_the_job() -> None:
    api = Api({"job-1": job(purpose="deployment")})

    with CloudStream(api, job(purpose="deployment"), owns_job=False):
        pass

    assert api.calls == []


@pytest.mark.parametrize(
    "ended, message",
    [
        (
            job(state="error", lastError={"message": "GPU out of memory"}),
            "GPU out of memory",
        ),
        (job(state="cancelled", stopReason="credit-limit"), "cancelled: credit-limit"),
    ],
)
def test_results_raise_when_the_job_ended_early_but_not_after_a_stop(
    ended, message
) -> None:
    api = Api({"job-1": ended})
    stream = CloudStream(api, ended, owns_job=False)

    with pytest.raises(CloudStreamingError, match=message):
        list(stream.results())
    stream.stop()
    assert list(stream.results()) == []


def test_results_end_quietly_when_the_job_completes() -> None:
    completed = job(state="completed")
    with CloudStream(Api({"job-1": completed}), completed, owns_job=False) as stream:
        assert list(stream.results()) == []


def test_a_preview_that_lost_its_owner_lease_follows_its_successor() -> None:
    api = Api(
        {"job-1": job(state="error", successorJobId="job-2"), "job-2": job("job-2")},
        held=False,
    )

    with CloudStream(api, job(), owns_job=True, session_id="tmp-1") as stream:
        until(lambda: stream.job_id == "job-2")


def test_losing_the_publisher_ends_the_stream_with_its_error() -> None:
    api = Api({"job-1": job()})
    publisher = Publisher()
    stream = CloudStream(
        api, job(), owns_job=True, session_id="tmp-1", publisher=publisher
    )

    publisher.error = OSError("connection reset")
    publisher.done.set()
    until(lambda: ("stop_session", "tmp-1") in api.calls)

    with pytest.raises(
        CloudStreamingError, match="publishing stopped: connection reset"
    ):
        list(stream.results())


def test_a_video_that_ends_stops_the_stream_without_an_error(monkeypatch) -> None:
    monkeypatch.setattr(stream_module, "DRAIN_S", 0)
    api = Api({"job-1": job()})
    publisher = Publisher()
    stream = CloudStream(
        api, job(), owns_job=True, session_id="tmp-1", publisher=publisher
    )

    publisher.done.set()
    until(lambda: ("stop_session", "tmp-1") in api.calls)

    assert list(stream.results()) == []
