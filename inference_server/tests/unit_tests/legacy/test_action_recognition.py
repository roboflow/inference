import asyncio
import base64
import inspect
import os
import random
import tempfile
import threading
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from inference_models.models.base.action_recognition import (
    ActionRecognitionPrediction as ModelPrediction,
)
from inference_server import configuration
from inference_server.legacy import action_recognition, video
from inference_server.legacy.action_recognition import (
    classify_video,
    merge_window_segments,
)
from inference_server.legacy.bridge import Route
from inference_server.legacy.entities import (
    ActionRecognitionInferenceResponse,
    ActionRecognitionPrediction,
)
from inference_server.legacy.errors import LegacyHTTPError

_FPS = 10.0
_FRAME_COUNT = 25
_WIDTH, _HEIGHT = 64, 48
SAMPLING = {
    "window_seconds": 1.0,
    "sample_fps": 2.0,
    "min_frames": 2,
    "max_frame_side": None,
    "mode": "sliding_window",
    "max_frames": None,
}
PLANNED_WINDOWS = [(0, 5), (10, 15), (20, 24)]
IMAGE_ERROR_PREFIX = "Could not load input image. Cause: "


def write_clip(path: Path, frame_count: int = _FRAME_COUNT, fps: float = _FPS):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (_WIDTH, _HEIGHT)
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
        frame[:] = (200, 20 + index * 8, 60)
        writer.write(frame)
    writer.release()
    return path


def clip_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def segments_per_window(class_names: List[str]):
    calls = []

    def _segments(action, params):
        calls.append((action, params))
        frames = params["frames"]
        class_name = class_names[min(len(calls) - 1, len(class_names) - 1)]
        return [ModelPrediction(0, len(frames) - 1, class_name)]

    return calls, _segments


class FakeBridge:
    def __init__(self, segments):
        self._segments = segments
        self.calls = []

    async def infer_params_only(self, route, api_key, action, params):
        self.calls.append((action, params))
        if callable(self._segments):
            return self._segments(action, params)
        return self._segments


def action_route(video_sampling=SAMPLING, class_names=("wave", "jump")) -> Route:
    return Route(
        model_id="clips/1",
        registry_id="clips/1",
        task_type="action-recognition",
        action="infer",
        actions={"infer"},
        class_names=list(class_names) if class_names is not None else None,
        video_sampling=video_sampling,
    )


@pytest.fixture
def clip(tmp_path) -> Path:
    return write_clip(tmp_path / "clip.mp4")


@pytest.fixture(autouse=True)
def temp_dir(tmp_path, monkeypatch) -> Path:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    return scratch


def _random_segments(rng: random.Random, sample_count: int, names: List[str]):
    segments = []
    for _ in range(rng.randint(0, 6)):
        start = rng.randint(-2, sample_count + 2)
        end = rng.randint(-2, sample_count + 2)
        segments.append(ModelPrediction(start, end, rng.choice(names)))
    return segments


class TestMergeWindowSegmentsParity:
    @pytest.mark.parametrize("seed", range(40))
    def test_matches_workflows_copy_over_generated_windows(self, seed):
        from roboflow_workflows.utils.action_recognition import (
            merge_window_segments as workflows_merge_window_segments,
        )

        rng = random.Random(seed)
        names = ["wave", "jump", "fall"]
        vocabulary = rng.choice([None, ["wave", "jump"], ["jump"]])
        class_filter = rng.choice([None, ["wave"], ["fall", "wave"]])
        ours: List[ActionRecognitionPrediction] = []
        theirs: list = []
        for _ in range(rng.randint(1, 5)):
            sample_count = rng.randint(0, 6)
            first = rng.randint(0, 100)
            stride = rng.choice([1.0, 2.5, 5.0])
            frame_numbers = [first + int(i * stride) for i in range(sample_count)]
            segments = _random_segments(rng, sample_count, names)
            arguments = dict(
                frame_numbers=frame_numbers,
                segments=segments,
                id_vocabulary=vocabulary,
                stride=stride,
                class_filter=class_filter,
            )
            merge_window_segments(timeline=ours, **arguments)
            workflows_merge_window_segments(timeline=theirs, **arguments)

        assert [entry.model_dump() for entry in ours] == [
            entry.model_dump() for entry in theirs
        ]
        assert all(isinstance(entry, ActionRecognitionPrediction) for entry in ours)


class TestClassifyVideo:
    @pytest.mark.asyncio
    async def test_sends_each_planned_window_in_order_as_params(self, clip):
        calls, segments = segments_per_window(["wave", "jump"])
        bridge = FakeBridge(segments)

        response = await classify_video(
            action_route(),
            "key",
            bridge,
            video_type="base64",
            video_value=clip_base64(clip),
            class_filter=["wave", "jump"],
        )

        assert isinstance(response, ActionRecognitionInferenceResponse)
        assert response.source_fps == _FPS
        assert response.frame_count == _FRAME_COUNT
        assert response.windows_classified == len(PLANNED_WINDOWS)
        assert [action for action, _ in bridge.calls] == ["infer"] * 3
        for (_, params), indices in zip(bridge.calls, PLANNED_WINDOWS):
            assert set(params) == {"frames", "class_names", "fps"}
            assert params["fps"] == SAMPLING["sample_fps"]
            assert params["class_names"] == ["wave", "jump"]
            assert len(params["frames"]) == len(indices)
            assert all(
                frame.shape == (_HEIGHT, _WIDTH, 3) and frame.dtype == np.uint8
                for frame in params["frames"]
            )
            assert [frame[0, 0, 1] for frame in params["frames"]] == pytest.approx(
                [20 + index * 8 for index in indices], abs=6
            )
        assert [entry.model_dump(by_alias=True) for entry in response.timeline] == [
            {"start_frame_idx": 0, "end_frame_idx": 5, "class": "wave", "class_id": 0},
            {
                "start_frame_idx": 10,
                "end_frame_idx": 24,
                "class": "jump",
                "class_id": 1,
            },
        ]

    @pytest.mark.asyncio
    async def test_frames_reach_the_model_in_rgb(self, clip):
        bridge = FakeBridge([])

        await classify_video(
            action_route(),
            None,
            bridge,
            video_type="base64",
            video_value=clip_base64(clip),
            class_filter=None,
        )

        first = bridge.calls[0][1]["frames"][0]
        assert first[0, 0].tolist() == pytest.approx([60, 20, 200], abs=8)
        assert bridge.calls[0][1]["class_names"] is None

    @pytest.mark.asyncio
    async def test_default_sampling_when_metadata_carries_none(self, clip):
        bridge = FakeBridge([])

        response = await classify_video(
            action_route(video_sampling=None),
            None,
            bridge,
            video_type="base64",
            video_value=clip_base64(clip),
            class_filter=None,
        )

        assert response.windows_classified == 1
        assert bridge.calls[0][1]["fps"] == 4.0
        assert len(bridge.calls[0][1]["frames"]) == 10

    @pytest.mark.asyncio
    async def test_model_without_class_list_reports_minus_one(self, clip):
        bridge = FakeBridge([ModelPrediction(0, 1, "a caption")])

        response = await classify_video(
            action_route(class_names=None),
            None,
            bridge,
            video_type="base64",
            video_value=clip_base64(clip),
            class_filter=None,
        )

        assert {entry.class_id for entry in response.timeline} == {-1}
        assert response.timeline[0].class_name == "a caption"

    @pytest.mark.asyncio
    async def test_short_trailing_window_is_skipped(self, clip, monkeypatch):
        monkeypatch.setattr(
            action_recognition, "probe_video", lambda path: (_FPS, _FRAME_COUNT + 10)
        )
        calls, segments = segments_per_window(["wave"])
        bridge = FakeBridge(segments)

        response = await classify_video(
            action_route(),
            None,
            bridge,
            video_type="base64",
            video_value=clip_base64(clip),
            class_filter=None,
        )

        assert [len(params["frames"]) for _, params in bridge.calls] == [2, 2]
        assert response.windows_classified == 2
        assert response.frame_count == _FRAME_COUNT + 10

    @pytest.mark.asyncio
    async def test_timeline_is_sorted_by_start_then_class_id(self, clip):
        bridge = FakeBridge(
            [
                ModelPrediction(1, 1, "jump"),
                ModelPrediction(0, 1, "wave"),
                ModelPrediction(1, 1, "wave"),
            ]
        )

        response = await classify_video(
            action_route(video_sampling=None),
            None,
            bridge,
            video_type="base64",
            video_value=clip_base64(clip),
            class_filter=None,
        )

        keys = [(entry.start_frame_idx, entry.class_id) for entry in response.timeline]
        assert keys == sorted(keys)
        assert keys[0][1] == 0

    @pytest.mark.asyncio
    async def test_duration_cap_is_a_413_with_the_legacy_message(
        self, clip, monkeypatch
    ):
        monkeypatch.setattr(configuration, "MAX_VIDEO_DURATION_SECONDS", 2.0)
        bridge = FakeBridge([])

        with pytest.raises(LegacyHTTPError) as error:
            await classify_video(
                action_route(),
                None,
                bridge,
                video_type="base64",
                video_value=clip_base64(clip),
                class_filter=None,
            )

        assert error.value.status_code == 413
        assert error.value.message == (
            "Video runs 2.5 s. This server classifies at most 2 s in one request. "
            "Send a shorter clip, or raise MAX_VIDEO_DURATION_SECONDS on the server."
        )
        assert bridge.calls == []

    @pytest.mark.asyncio
    async def test_undecodable_clip_is_a_400(self):
        bridge = FakeBridge([])

        with pytest.raises(LegacyHTTPError) as error:
            await classify_video(
                action_route(),
                None,
                bridge,
                video_type="base64",
                video_value=base64.b64encode(os.urandom(4096)).decode("ascii"),
                class_filter=None,
            )

        assert error.value.status_code == 400
        assert error.value.message == f"{IMAGE_ERROR_PREFIX}Video could not be decoded."

    @pytest.mark.asyncio
    async def test_temp_file_is_removed_after_the_windows(self, clip, temp_dir):
        bridge = FakeBridge([])

        await classify_video(
            action_route(),
            None,
            bridge,
            video_type="base64",
            video_value=clip_base64(clip),
            class_filter=None,
        )

        assert list(temp_dir.iterdir()) == []

    @pytest.mark.asyncio
    async def test_gateway_error_propagates_and_removes_the_temp_file(
        self, clip, temp_dir
    ):
        def _boom(action, params):
            raise RuntimeError("model failed")

        bridge = FakeBridge(_boom)

        with pytest.raises(RuntimeError):
            await classify_video(
                action_route(),
                None,
                bridge,
                video_type="base64",
                video_value=clip_base64(clip),
                class_filter=None,
            )

        assert list(temp_dir.iterdir()) == []


_REAL_CAPTURE = cv2.VideoCapture


class _DelegatingCapture:
    released = []
    block = None

    def __init__(self, path):
        self._capture = _REAL_CAPTURE(path)

    def read(self):
        block = _DelegatingCapture.block
        if block is not None:
            started, release = block
            started.set()
            release.wait(10)
        return self._capture.read()

    def release(self):
        _DelegatingCapture.released.append(self)
        self._capture.release()

    def __getattr__(self, name):
        return getattr(self._capture, name)


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancel_during_a_read_closes_the_generator_and_releases_the_capture(
        self, clip, temp_dir, monkeypatch
    ):
        monkeypatch.setattr(video.cv2, "VideoCapture", _DelegatingCapture)
        _DelegatingCapture.released = []
        started, release = threading.Event(), threading.Event()
        opened = []
        real_read_frame_windows = video.read_frame_windows

        def _tracked(*args, **kwargs):
            generator = real_read_frame_windows(*args, **kwargs)
            opened.append(generator)
            _DelegatingCapture.block = (started, release)
            return generator

        monkeypatch.setattr(action_recognition, "read_frame_windows", _tracked)
        encoded = clip_base64(clip)
        task = asyncio.ensure_future(
            classify_video(
                action_route(),
                None,
                FakeBridge([]),
                video_type="base64",
                video_value=encoded,
                class_filter=None,
            )
        )
        try:
            while not started.is_set() and not task.done():
                await asyncio.sleep(0.01)
            released_before = len(_DelegatingCapture.released)
            task.cancel()
            await asyncio.sleep(0.05)
            assert not task.done()
            assert len(_DelegatingCapture.released) == released_before
            _DelegatingCapture.block = None
            release.set()
            with pytest.raises(asyncio.CancelledError) as cancelled:
                await task
        finally:
            _DelegatingCapture.block = None
            release.set()

        assert cancelled.value is not None
        assert len(opened) == 1
        assert inspect.getgeneratorstate(opened[0]) == inspect.GEN_CLOSED
        assert len(_DelegatingCapture.released) > released_before
        assert list(temp_dir.iterdir()) == []


class TestBase64Preparation:
    @pytest.mark.asyncio
    async def test_decode_and_write_run_off_the_event_loop(self, clip, monkeypatch):
        loop_thread = threading.current_thread()
        threads = {}
        real_decode = video._decode_base64_video
        real_write = video._write_payload

        def _decode(value):
            threads["decode"] = threading.current_thread()
            return real_decode(value)

        def _write(handle, payload):
            threads["write"] = threading.current_thread()
            return real_write(handle, payload)

        monkeypatch.setattr(video, "_decode_base64_video", _decode)
        monkeypatch.setattr(video, "_write_payload", _write)

        async with video.video_source_path("base64", clip_base64(clip)) as path:
            assert Path(path).read_bytes() == clip.read_bytes()

        assert set(threads) == {"decode", "write"}
        assert all(thread is not loop_thread for thread in threads.values())
