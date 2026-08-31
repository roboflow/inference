import queue
import threading

import av
import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.interfaces.webrtc_worker.entities import WebRTCWorkerRequest
from inference.core.interfaces.webrtc_worker.sources.file import _decode_worker

MINIMAL_REQUEST_PAYLOAD = {
    "workflow_configuration": {
        "type": "WorkflowConfiguration",
        "workflow_specification": {},
    },
    "webrtc_offer": {"type": "offer", "sdp": "v=0"},
}


def test_request_accepts_video_file_url_and_max_fps() -> None:
    request = WebRTCWorkerRequest(
        **MINIMAL_REQUEST_PAYLOAD,
        video_file_url="https://example.com/video.mp4",
        max_fps=5,
    )

    assert request.video_file_url == "https://example.com/video.mp4"
    assert request.max_fps == 5


def test_request_new_fields_default_to_none() -> None:
    request = WebRTCWorkerRequest(**MINIMAL_REQUEST_PAYLOAD)

    assert request.video_file_url is None
    assert request.max_fps is None


@pytest.mark.parametrize("max_fps", [0, -1])
def test_request_rejects_non_positive_max_fps(max_fps: float) -> None:
    with pytest.raises(ValidationError):
        WebRTCWorkerRequest(**MINIMAL_REQUEST_PAYLOAD, max_fps=max_fps)


def _write_test_video(path: str, frames: int = 30, fps: int = 30) -> None:
    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = 64
    stream.height = 64
    stream.pix_fmt = "yuv420p"
    for i in range(frames):
        image = np.full((64, 64, 3), i % 255, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _run_decode_worker(video_path: str, target_fps=None) -> int:
    frame_queue = queue.Queue(maxsize=100)
    _decode_worker(video_path, frame_queue, threading.Event(), target_fps)
    decoded = 0
    while True:
        item = frame_queue.get_nowait()
        if item is None:
            return decoded
        assert not isinstance(item, dict), f"decode error: {item}"
        decoded += 1


def test_decode_worker_without_target_fps_yields_all_frames(tmp_path) -> None:
    video_path = str(tmp_path / "video.mp4")
    _write_test_video(video_path, frames=30, fps=30)

    assert _run_decode_worker(video_path) == 30


def test_decode_worker_with_target_fps_decimates_frames(tmp_path) -> None:
    video_path = str(tmp_path / "video.mp4")
    _write_test_video(video_path, frames=30, fps=30)

    decoded = _run_decode_worker(video_path, target_fps=10)

    assert 9 <= decoded <= 11
