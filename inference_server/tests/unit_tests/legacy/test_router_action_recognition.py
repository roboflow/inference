import base64
import os
import tempfile
from pathlib import Path

import pytest

from inference_models.models.base.action_recognition import (
    ActionRecognitionPrediction as ModelPrediction,
)
from inference_server import configuration
from inference_server.legacy import action_recognition, video
from tests.unit_tests.legacy.conftest import FakeGateway
from tests.unit_tests.legacy.test_action_recognition import (
    IMAGE_ERROR_PREFIX,
    PLANNED_WINDOWS,
    SAMPLING,
    clip_base64,
    write_clip,
)

MODEL_ID = "clips/1"
VIDEO_URL = "https://example.com/clip.mp4"
FORM_CONTENT_TYPE = {"Content-Type": "application/x-www-form-urlencoded"}
EXPECTED_TIMELINE = [
    {"start_frame_idx": 0, "end_frame_idx": 5, "class": "wave", "class_id": 0},
    {"start_frame_idx": 10, "end_frame_idx": 24, "class": "jump", "class_id": 1},
]


@pytest.fixture(autouse=True)
def temp_dir(tmp_path, monkeypatch) -> Path:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    return scratch


@pytest.fixture
def clip(tmp_path) -> Path:
    return write_clip(tmp_path / "clip.mp4")


def _segments_per_window(class_names=("wave", "jump")):
    calls = []

    def _segments(image, params):
        calls.append(params)
        class_name = class_names[min(len(calls) - 1, len(class_names) - 1)]
        return [ModelPrediction(0, len(params["frames"]) - 1, class_name)]

    return _segments


def _gateway(fake_stat, task_type="action-recognition", **model_info):
    fake_stat[MODEL_ID] = (task_type, "infer", "cosmos3-edge", "cosmos3-edge-2b")
    info = {
        "class_names": ["wave", "jump"],
        "actions": {"infer": {}},
        "model_class_name": "Cosmos3EdgeActionRecognition",
        "video_sampling": SAMPLING,
    }
    info.update(model_info)
    gateway = FakeGateway(
        predictions={(MODEL_ID, "infer"): _segments_per_window()},
        model_info={MODEL_ID: info},
    )

    return gateway


def _infer_calls(gateway):
    return [call for call in gateway.calls if call[0] == "infer"]


def _request(clip, **extra):
    return {
        "model_id": MODEL_ID,
        "api_key": "k",
        "video": {"type": "base64", "value": clip_base64(clip)},
        **extra,
    }


def test_route_classifies_each_window_in_order(legacy_client, fake_stat, clip):
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
        "/infer/action_recognition",
        json=_request(clip, class_filter=["wave", "jump"]),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {
        "timeline",
        "source_fps",
        "frame_count",
        "windows_classified",
        "inference_id",
        "time",
        "resolved_model",
    }
    assert body["source_fps"] == 10.0
    assert body["frame_count"] == 25
    assert body["windows_classified"] == len(PLANNED_WINDOWS)
    assert body["timeline"] == EXPECTED_TIMELINE
    assert body["resolved_model"]["model_id"] == MODEL_ID
    assert body["time"] >= 0
    calls = _infer_calls(gateway)
    assert len(calls) == len(PLANNED_WINDOWS)
    for (_, model_id, action, params, image), indices in zip(calls, PLANNED_WINDOWS):
        assert model_id == MODEL_ID and action == "infer" and image is None
        assert set(params) == {"frames", "class_names", "fps"}
        assert len(params["frames"]) == len(indices)
        assert params["fps"] == SAMPLING["sample_fps"]
        assert params["class_names"] == ["wave", "jump"]


def test_route_stamps_the_request_id(legacy_client, fake_stat, clip):
    response = legacy_client(_gateway(fake_stat)).post(
        "/infer/action_recognition", json=_request(clip, id="req-7")
    )

    assert response.status_code == 200, response.text
    assert response.json()["inference_id"] == "req-7"


def test_route_sends_no_class_filter_as_none(legacy_client, fake_stat, clip):
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
        "/infer/action_recognition", json=_request(clip)
    )

    assert response.status_code == 200, response.text
    assert {call[3]["class_names"] for call in _infer_calls(gateway)} == {None}


def test_route_skips_a_short_trailing_window(
    legacy_client, fake_stat, clip, monkeypatch
):
    monkeypatch.setattr(action_recognition, "probe_video", lambda path: (10.0, 35))
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
        "/infer/action_recognition", json=_request(clip)
    )

    assert response.status_code == 200, response.text
    assert response.json()["windows_classified"] == 2
    assert [len(call[3]["frames"]) for call in _infer_calls(gateway)] == [2, 2]


def test_route_refuses_a_clip_over_the_duration_cap(
    legacy_client, fake_stat, clip, monkeypatch
):
    monkeypatch.setattr(configuration, "MAX_VIDEO_DURATION_SECONDS", 2.0)
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
        "/infer/action_recognition", json=_request(clip)
    )

    assert response.status_code == 413
    assert response.json() == {
        "message": (
            "Video runs 2.5 s. This server classifies at most 2 s in one request. "
            "Send a shorter clip, or raise MAX_VIDEO_DURATION_SECONDS on the server."
        )
    }
    assert _infer_calls(gateway) == []


def test_route_refuses_an_undecodable_clip(legacy_client, fake_stat):
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
        "/infer/action_recognition",
        json={
            "model_id": MODEL_ID,
            "video": {
                "type": "base64",
                "value": base64.b64encode(os.urandom(4096)).decode("ascii"),
            },
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "message": f"{IMAGE_ERROR_PREFIX}Video could not be decoded."
    }


def test_route_refuses_an_unsupported_video_type(legacy_client, fake_stat):
    response = legacy_client(_gateway(fake_stat)).post(
        "/infer/action_recognition",
        json={"model_id": MODEL_ID, "video": {"type": "numpy", "value": "x"}},
    )

    assert response.status_code == 400
    assert response.json() == {
        "message": (
            f"{IMAGE_ERROR_PREFIX}Video type 'numpy' is not supported, expected "
            "one of 'url' or 'base64'."
        )
    }


def test_route_refuses_a_model_of_another_task(legacy_client, fake_stat, clip):
    gateway = _gateway(fake_stat, task_type="object-detection")

    response = legacy_client(gateway).post(
        "/infer/action_recognition", json=_request(clip)
    )

    assert response.status_code == 400
    assert response.json() == {
        "message": f"Model {MODEL_ID!r} is a object-detection model."
    }
    assert _infer_calls(gateway) == []


def test_route_refuses_a_bare_cosmos_reasoner(legacy_client, fake_stat, clip):
    gateway = _gateway(
        fake_stat,
        task_type="vlm",
        model_class_name="Cosmos3EdgeReasoner",
        video_sampling=None,
    )

    response = legacy_client(gateway).post(
        "/infer/action_recognition", json=_request(clip)
    )

    assert response.status_code == 400
    assert response.json() == {
        "message": (
            f"Model {MODEL_ID!r} is a vlm model, loaded as Cosmos3EdgeReasoner; "
            "action recognition needs a model registered for the "
            "action-recognition task."
        )
    }
    assert _infer_calls(gateway) == []


def test_route_without_a_body_is_422(legacy_client, fake_stat):
    response = legacy_client(_gateway(fake_stat)).post(
        "/infer/action_recognition", json={}
    )

    assert response.status_code == 422


def test_catch_all_dispatches_action_recognition_with_class_filter(
    legacy_client, fake_stat, clip
):
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
        f"/{MODEL_ID}?api_key=k&class_filter=wave,%20jump,&confidence=50",
        content=clip_base64(clip).encode("ascii"),
        headers=FORM_CONTENT_TYPE,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["timeline"] == EXPECTED_TIMELINE
    assert body["windows_classified"] == len(PLANNED_WINDOWS)
    assert body["resolved_model"]["model_id"] == MODEL_ID
    calls = _infer_calls(gateway)
    assert [call[3]["class_names"] for call in calls] == [["wave", "jump"]] * 3
    assert all(set(call[3]) == {"frames", "class_names", "fps"} for call in calls)


def test_catch_all_without_class_filter_sends_none(legacy_client, fake_stat, clip):
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
        f"/{MODEL_ID}?api_key=k",
        content=clip_base64(clip).encode("ascii"),
        headers=FORM_CONTENT_TYPE,
    )

    assert response.status_code == 200, response.text
    assert {call[3]["class_names"] for call in _infer_calls(gateway)} == {None}


def test_catch_all_fetches_the_clip_from_the_image_url(
    legacy_client, fake_stat, clip, monkeypatch
):
    monkeypatch.setattr(video, "LEGACY_OFFLINE_MODE", False)
    monkeypatch.setattr(video, "ALLOW_URL_INPUT", True)
    monkeypatch.setattr(configuration, "ALLOW_NON_HTTPS_URL_INPUT", False)
    monkeypatch.setattr(configuration, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
    monkeypatch.setattr(configuration, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", None)
    monkeypatch.setattr(configuration, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", None)
    monkeypatch.setattr(configuration, "VALIDATE_IMAGE_URL_REDIRECTS", False)
    fetched = []

    async def _fetch(url, *, sink, max_bytes, timeout_s, destination_policy):
        fetched.append(url)
        sink(clip.read_bytes())
        return None

    monkeypatch.setattr(video, "fetch_to_sink", _fetch)
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).get(
        f"/{MODEL_ID}?api_key=k&image={VIDEO_URL}&class_filter=wave"
    )

    assert response.status_code == 200, response.text
    assert fetched == [VIDEO_URL]
    assert response.json()["timeline"] == EXPECTED_TIMELINE
    assert [call[3]["class_names"] for call in _infer_calls(gateway)] == [["wave"]] * 3


def test_catch_all_refuses_an_undecodable_clip(legacy_client, fake_stat):
    response = legacy_client(_gateway(fake_stat)).post(
        f"/{MODEL_ID}?api_key=k",
        content=base64.b64encode(os.urandom(4096)),
        headers=FORM_CONTENT_TYPE,
    )

    assert response.status_code == 400
    assert response.json() == {
        "message": f"{IMAGE_ERROR_PREFIX}Video could not be decoded."
    }


def test_catch_all_duration_cap_is_413(legacy_client, fake_stat, clip, monkeypatch):
    monkeypatch.setattr(configuration, "MAX_VIDEO_DURATION_SECONDS", 2.0)

    response = legacy_client(_gateway(fake_stat)).post(
        f"/{MODEL_ID}?api_key=k",
        content=clip_base64(clip).encode("ascii"),
        headers=FORM_CONTENT_TYPE,
    )

    assert response.status_code == 413
    assert response.json()["message"].startswith("Video runs 2.5 s.")


class _RecordingBound(float):
    comparisons = 0

    def __gt__(self, other):
        type(self).comparisons += 1
        return float.__lt__(self, other)


def test_catch_all_action_recognition_skips_detection_parameter_handling(
    legacy_client, fake_stat, clip, monkeypatch
):
    _RecordingBound.comparisons = 0
    monkeypatch.setattr(
        configuration, "CONFIDENCE_LOWER_BOUND_OOM_PREVENTION", _RecordingBound(0.5)
    )
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
        f"/{MODEL_ID}?api_key=k&confidence=50&overlap=50",
        content=clip_base64(clip).encode("ascii"),
        headers=FORM_CONTENT_TYPE,
    )

    assert response.status_code == 200, response.text
    assert response.json()["timeline"] == EXPECTED_TIMELINE
    assert _RecordingBound.comparisons == 0
