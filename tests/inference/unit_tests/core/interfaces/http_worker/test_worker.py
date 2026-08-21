from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

from inference.core.interfaces.http_worker.entities import (
    EVENT_CHECKPOINTED,
    EVENT_DONE,
    EVENT_DOWNLOADING,
    EVENT_FRAME,
    ArtifactTarget,
    TimeBase,
    WorkerPayload,
)
from inference.core.interfaces.http_worker.worker import (
    FramePacket,
    HttpEventPublisher,
    process_frames,
    run_worker,
)


class FakeResult:
    def __init__(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        self.masks = mask[None, ...]
        self.object_ids = np.array([7])
        self.scores = np.array([0.91])
        self.boxes = np.array([[2.0, 2.0, 6.0, 6.0]])
        self.prompt_to_object_ids = {"forklift": [7]}
        self.state_dict = {"ok": True}


class FakeModel:
    def __init__(self) -> None:
        self.prompt_calls = 0
        self.track_calls = 0

    def prompt(self, image, text, clear_old_prompts=True):
        self.prompt_calls += 1
        return FakeResult()

    def track(self, image, state_dict=None):
        self.track_calls += 1
        return FakeResult()


class FakePublisher:
    def __init__(self, stop_after: Optional[int] = None) -> None:
        self.events: List[Tuple[str, Dict[str, Any]]] = []
        self.stop_after = stop_after

    def publish(
        self, event_type: str, payload: Optional[Dict[str, Any]] = None
    ) -> bool:
        self.events.append((event_type, payload or {}))
        if self.stop_after is not None and len(self.events) >= self.stop_after:
            return True
        return False


class FakeWriter:
    def __init__(self) -> None:
        self.chunks: List[Dict[str, Any]] = []
        self.commits: List[Dict[str, Any]] = []

    def checkpoint_chunk(self, **kwargs: Any) -> None:
        self.chunks.append(kwargs)

    def commit_revision(self, **kwargs: Any) -> None:
        self.commits.append(kwargs)


def _frames(count: int) -> List[FramePacket]:
    packets = []
    for index in range(count):
        packets.append(
            FramePacket(
                image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
                frame_index=index,
                pts=index,
                time_base=1 / 30,
                source_time_seconds=index / 30,
                width=8,
                height=8,
            )
        )
    return packets


def _payload() -> WorkerPayload:
    return WorkerPayload(
        session_id="sess-1",
        video_url="https://storage.example/video.mp4",
        class_names=["forklift"],
        artifact=ArtifactTarget(
            app_base_url="https://app.roboflow.com",
            video_id="video-1",
            workspace_id="ws-1",
            dataset_id="ds-1",
            revision_id="rev-1",
            video_time_base=TimeBase(numerator=1, denominator=30),
        ),
        api_key="rf_key",
        events_callback_url="https://serverless.example/sam3/video/sessions/sess-1/internal/events",
        publish_token="pub",
    )


def test_process_frames_checkpoints_every_n_samples_and_commits() -> None:
    publisher = FakePublisher()
    writer = FakeWriter()
    model = FakeModel()

    process_frames(
        frames=_frames(5),
        model=model,
        class_names=["forklift"],
        threshold=0.35,
        revision_id="rev-1",
        publisher=publisher,
        writer=writer,
        video_time_base=TimeBase(numerator=1, denominator=30),
        chunk_sample_size=2,
    )

    assert model.prompt_calls == 1
    assert model.track_calls == 4
    assert len(writer.chunks) == 3
    assert [chunk["chunk_index"] for chunk in writer.chunks] == [0, 1, 2]
    assert [len(chunk["samples"]) for chunk in writer.chunks] == [2, 2, 1]
    assert writer.chunks[0]["track_id"] == "7"
    assert writer.chunks[0]["samples"][0]["geometry"]["rleMask"]["size"] == [8, 8]
    assert len(writer.commits) == 1
    assert writer.commits[0]["sample_count"] == 5
    assert writer.commits[0]["chunk_count"] == 3
    assert writer.commits[0]["track_id"] == "7"

    event_types = [event_type for event_type, _ in publisher.events]
    assert event_types.count(EVENT_FRAME) == 5
    assert EVENT_CHECKPOINTED in event_types
    assert event_types[-1] == EVENT_DONE
    frame_event = next(
        payload for event_type, payload in publisher.events if event_type == EVENT_FRAME
    )
    assert (
        frame_event["serialized_output_data"]["predictions"]["predictions"][0][
            "tracker_id"
        ]
        == 7
    )
    assert frame_event["video_metadata"]["revision_id"] == "rev-1"


def test_process_frames_stop_requested_still_flushes_and_commits() -> None:
    publisher = FakePublisher(stop_after=1)
    writer = FakeWriter()

    process_frames(
        frames=_frames(4),
        model=FakeModel(),
        class_names=["forklift"],
        threshold=0.35,
        revision_id="rev-1",
        publisher=publisher,
        writer=writer,
        video_time_base=TimeBase(numerator=1, denominator=30),
        chunk_sample_size=2,
    )

    assert len(writer.chunks) == 1
    assert writer.chunks[0]["samples"]
    assert writer.commits[0]["sample_count"] == 1
    assert publisher.events[-1][0] == EVENT_DONE
    assert publisher.events[-1][1]["cancelled"] is True


def test_run_worker_with_injected_frames_emits_downloading() -> None:
    publisher = FakePublisher()
    writer = FakeWriter()

    run_worker(
        _payload(),
        publisher=publisher,
        writer=writer,
        model=FakeModel(),
        frames=_frames(1),
        chunk_sample_size=500,
    )

    assert publisher.events[0][0] == EVENT_DOWNLOADING
    assert publisher.events[-1][0] == EVENT_DONE
    assert writer.commits[0]["sample_count"] == 1


def test_http_event_publisher_reads_stop_requested(monkeypatch) -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"stop_requested": True}
    posted = {}

    def fake_post(url, json, headers, timeout):
        posted["url"] = url
        posted["json"] = json
        posted["headers"] = headers
        return response

    monkeypatch.setattr(
        "inference.core.interfaces.http_worker.worker.requests.post",
        fake_post,
    )
    publisher = HttpEventPublisher(
        events_callback_url="https://serverless.example/events",
        publish_token="pub",
        api_key="rf_key",
    )

    assert publisher.publish("frame", {"frame_id": 3}) is True
    assert posted["json"]["publish_token"] == "pub"
    assert posted["json"]["event"]["type"] == "frame"
    assert posted["headers"]["Authorization"] == "Bearer rf_key"
