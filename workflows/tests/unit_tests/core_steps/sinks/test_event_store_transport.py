import base64
import os
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests
from roboflow_workflows.core_steps.sinks.roboflow import event_store_transport
from roboflow_workflows.core_steps.sinks.roboflow.event_store_transport import (
    EVENT_STORE_SHARED_MEMORY_PROTOCOL,
    post_local_event,
    prepare_local_images,
)
from roboflow_workflows.core_steps.sinks.roboflow.vision_events.v1 import (
    _execute_local_event,
)
from roboflow_workflows.enterprise_blocks.sinks.event_writer.v1 import (
    EventWriterSinkBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.utils.images import encode_image_to_jpeg_bytes

EVENT_STORE_URL = "http://localhost:8001"
EVENTS_URL = f"{EVENT_STORE_URL}/v2/events"


@pytest.fixture
def shm_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("EVENT_STORE_SHM_DIR", str(tmp_path))
    monkeypatch.delenv("EVENT_STORE_SHARED_MEMORY", raising=False)
    monkeypatch.setattr(event_store_transport, "_probe_cache", {})
    return tmp_path


def _identity(directory: Path) -> str:
    st = os.stat(directory)
    return f"{st.st_dev}:{st.st_ino}"


def _response(status_code: int, body: Optional[dict] = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body or {}
    return response


def _health(identity: Optional[str]) -> MagicMock:
    return _response(200, {"status": "ok", "shm_identity": identity})


def _image(seed: int = 0) -> WorkflowImageData:
    numpy_image = np.random.default_rng(seed).integers(
        0, 255, size=(48, 64, 3), dtype=np.uint8
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=f"image-{seed}"),
        numpy_image=numpy_image,
    )


def _shm_files(directory: Path) -> list:
    return sorted(path.name for path in directory.glob("rfevt-*"))


def test_protocol_constant_is_exported() -> None:
    assert EVENT_STORE_SHARED_MEMORY_PROTOCOL == 1


def _send(
    shm_dir: Path,
    output_image: Optional[WorkflowImageData],
    input_image: Optional[WorkflowImageData] = None,
    health=None,
    responses=None,
) -> tuple:
    prepared = prepare_local_images(output_image=output_image, input_image=input_image)
    payload = {"event_schema": "custom", "images": [{"label": "x"}]}
    sent = []

    def record(url, json, **kwargs):
        entry = dict(json["images"][0])
        files = {
            ref["name"]: (shm_dir / ref["name"]).read_bytes()
            for key in ("shmImage", "inputShmImage")
            if (ref := entry.get(key)) is not None
        }
        sent.append((entry, files))
        result = (
            (responses or [_response(201, {"id": "evt-1"})]).pop(0)
            if responses is not None
            else _response(201, {"id": "evt-1"})
        )
        if isinstance(result, BaseException):
            raise result
        return result

    get_kwargs = (
        {"side_effect": [health]}
        if isinstance(health, BaseException)
        else {
            "return_value": (
                health if health is not None else _health(_identity(shm_dir))
            )
        }
    )
    with patch.object(requests, "get", **get_kwargs) as get:
        with patch.object(requests, "post", side_effect=record):
            try:
                response = post_local_event(
                    EVENTS_URL, payload=payload, prepared_images=prepared, timeout=5
                )
            except BaseException as error:
                response = error
    return response, sent, get


def test_shared_memory_used_when_identities_match(shm_dir: Path) -> None:
    output_image, input_image = _image(1), _image(2)

    response, sent, get = _send(shm_dir, output_image, input_image)

    get.assert_called_once_with(f"{EVENT_STORE_URL}/health", timeout=2.0)
    assert response.status_code == 201
    entry, files = sent[0]
    assert set(entry) == {"label", "shmImage", "inputShmImage"}
    for key, image in (("shmImage", output_image), ("inputShmImage", input_image)):
        expected = encode_image_to_jpeg_bytes(image.numpy_image, jpeg_quality=95)
        reference = entry[key]
        assert reference["identity"] == _identity(shm_dir)
        assert reference["size"] == len(expected)
        assert len(reference["name"]) == len("rfevt-") + 32
        assert len(reference["token"]) == 64
        assert files[reference["name"]] == bytes.fromhex(reference["token"]) + expected
    assert sorted(files) == _shm_files(shm_dir)
    for name in files:
        assert (shm_dir / name).stat().st_mode & 0o777 == 0o600


def test_prepare_never_touches_the_network_or_shared_memory(shm_dir: Path) -> None:
    with patch.object(requests, "get") as get:
        prepared = prepare_local_images(_image(1), _image(2))

    get.assert_not_called()
    assert _shm_files(shm_dir) == []
    assert [type(value) for _, _, value in prepared] == [bytes, bytes]


def test_probe_result_is_cached(shm_dir: Path) -> None:
    _, _, first = _send(shm_dir, _image())
    _, _, second = _send(shm_dir, _image())

    first.assert_called_once()
    second.assert_not_called()


@pytest.mark.parametrize(
    "health",
    [
        _health("1:2"),
        _health(None),
        _response(200, {"status": "ok"}),
        requests.exceptions.ConnectionError("refused"),
    ],
)
def test_base64_when_event_store_does_not_share_memory(shm_dir: Path, health) -> None:
    output_image, input_image = _image(1), _image(2)

    _, sent, _ = _send(shm_dir, output_image, input_image, health=health)

    entry, files = sent[0]
    assert entry == {
        "label": "x",
        "base64Image": output_image.base64_image,
        "inputBase64Image": input_image.base64_image,
    }
    assert files == {}
    assert _shm_files(shm_dir) == []


@pytest.mark.parametrize("value", ["false", "0", "off", "disabled", "OFF"])
def test_opt_out_never_uses_shared_memory(
    shm_dir: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("EVENT_STORE_SHARED_MEMORY", value)
    image = _image()

    _, sent, get = _send(shm_dir, image)

    get.assert_not_called()
    assert sent[0][0] == {"label": "x", "base64Image": image.base64_image}


def test_image_with_cached_base64_is_sent_as_base64(shm_dir: Path) -> None:
    cached = _image(1)
    cached_base64 = cached.base64_image

    _, sent, _ = _send(shm_dir, cached, _image(2))

    entry, files = sent[0]
    assert entry["base64Image"] == cached_base64
    assert "shmImage" not in entry
    assert list(files) == [entry["inputShmImage"]["name"]]


def test_low_shared_memory_space_falls_back_to_base64(shm_dir: Path) -> None:
    image = _image()
    jpeg = encode_image_to_jpeg_bytes(image.numpy_image, jpeg_quality=95)
    usage = MagicMock(free=64 * 1024 * 1024 + len(jpeg) - 1)

    with patch.object(event_store_transport.shutil, "disk_usage", return_value=usage):
        _, sent, _ = _send(shm_dir, image)

    assert sent[0][0] == {
        "label": "x",
        "base64Image": base64.b64encode(jpeg).decode("ascii"),
    }
    assert _shm_files(shm_dir) == []


def test_created_response_leaves_files_to_the_event_store(shm_dir: Path) -> None:
    response, sent, _ = _send(shm_dir, _image(1), _image(2))

    assert response.status_code == 201
    assert _shm_files(shm_dir) == sorted(sent[0][1])


@pytest.mark.parametrize("status_code", [400, 413, 500, 507])
def test_failed_response_deletes_files(shm_dir: Path, status_code: int) -> None:
    response, sent, _ = _send(
        shm_dir,
        _image(1),
        _image(2),
        responses=[_response(status_code, {"error": "other"})],
    )

    assert response.status_code == status_code
    assert len(sent) == 1
    assert _shm_files(shm_dir) == []


@pytest.mark.parametrize(
    "error",
    [requests.exceptions.ConnectTimeout(), requests.exceptions.ConnectionError()],
)
def test_request_exception_deletes_files(
    shm_dir: Path, error: requests.RequestException
) -> None:
    response, sent, _ = _send(shm_dir, _image(1), responses=[error])

    assert response is error
    assert len(sent) == 1
    assert _shm_files(shm_dir) == []


def test_read_timeout_leaves_files_to_the_event_store(shm_dir: Path) -> None:
    response, sent, _ = _send(
        shm_dir, _image(1), responses=[requests.exceptions.ReadTimeout()]
    )

    assert isinstance(response, requests.exceptions.ReadTimeout)
    assert _shm_files(shm_dir) == sorted(sent[0][1])


def test_shm_image_unavailable_resends_with_base64(shm_dir: Path) -> None:
    response, sent, _ = _send(
        shm_dir,
        _image(1),
        _image(2),
        responses=[
            _response(400, {"error": "shm_image_unavailable"}),
            _response(201, {"id": "evt-1"}),
        ],
    )

    assert response.status_code == 201
    (first, files), (second, _) = sent
    assert "shmImage" in first and "base64Image" not in first
    assert second == {
        "label": "x",
        "base64Image": base64.b64encode(files[first["shmImage"]["name"]][32:]).decode(
            "ascii"
        ),
        "inputBase64Image": base64.b64encode(
            files[first["inputShmImage"]["name"]][32:]
        ).decode("ascii"),
    }
    assert _shm_files(shm_dir) == []

    _, later, get = _send(shm_dir, _image(3))
    get.assert_not_called()
    assert "shmImage" in later[0][0]


def test_event_writer_sends_shared_memory_references(shm_dir: Path) -> None:
    block = EventWriterSinkBlockV1(background_tasks=None, thread_pool_executor=None)
    with patch.object(requests, "get", return_value=_health(_identity(shm_dir))):
        with patch.object(
            requests, "post", return_value=_response(500, {"detail": "boom"})
        ) as post:
            result = block.run(
                event_ingestion_url=EVENT_STORE_URL,
                event_schema="custom",
                output_image=_image(1),
                input_image=_image(2),
                fire_and_forget=False,
                disable_sink=False,
                request_timeout=5,
            )

    assert result["error_status"] is True
    assert "500" in result["message"]
    image_entry = post.call_args.kwargs["json"]["images"][0]
    assert {"shmImage", "inputShmImage"} <= set(image_entry)
    assert "base64Image" not in image_entry
    assert _shm_files(shm_dir) == []


def test_event_writer_fire_and_forget_sends_the_image_as_it_was_when_queued(
    shm_dir: Path,
) -> None:
    background_tasks = MagicMock()
    block = EventWriterSinkBlockV1(
        background_tasks=background_tasks, thread_pool_executor=None
    )
    image = _image(1)
    expected = encode_image_to_jpeg_bytes(image.numpy_image.copy(), jpeg_quality=95)
    with patch.object(requests, "get", return_value=_health(_identity(shm_dir))):
        block.run(
            event_ingestion_url=EVENT_STORE_URL,
            event_schema="custom",
            output_image=image,
            fire_and_forget=True,
            disable_sink=False,
            request_timeout=5,
        )
    image.numpy_image[:] = 255
    task = background_tasks.add_task.call_args.args[0]
    posted = {}

    def capture(url, json, **kwargs):
        reference = json["images"][0]["shmImage"]
        posted["bytes"] = (shm_dir / reference["name"]).read_bytes()[32:]
        return _response(201, {"id": "evt-1"})

    with patch.object(requests, "get", return_value=_health(_identity(shm_dir))):
        with patch.object(requests, "post", side_effect=capture):
            task()

    assert posted["bytes"] == expected


def test_vision_events_sends_shared_memory_references(shm_dir: Path) -> None:
    sent = []

    def record(url, json, **kwargs):
        entry = json["images"][0]
        sent.append((url, dict(entry), (shm_dir / entry["shmImage"]["name"]).exists()))
        return _response(201, {"id": "evt-7"})

    with patch.object(requests, "get", return_value=_health(_identity(shm_dir))):
        with patch.object(requests, "post", side_effect=record):
            error_status, _, event_id = _execute_local_event(
                event_store_url=EVENT_STORE_URL,
                input_image=_image(2),
                output_image=_image(1),
                prediction=None,
                event_type="custom",
                solution="my-solution",
                event_data={"value": "x"},
                custom_metadata={},
            )

    assert error_status is False
    assert event_id == "evt-7"
    url, entry, existed = sent[0]
    assert url == EVENTS_URL
    assert "base64Image" not in entry and "inputBase64Image" not in entry
    assert entry["label"] == "workflow"
    assert existed


def test_unavailable_fallback_does_not_need_the_shared_memory_file(
    shm_dir: Path,
) -> None:
    image = _image(1)
    expected = encode_image_to_jpeg_bytes(image.numpy_image, jpeg_quality=95)

    def vanish_then_reject():
        for path in shm_dir.glob("rfevt-*"):
            path.unlink()
        return _response(400, {"error": "shm_image_unavailable"})

    class Rejecting:
        def __init__(self):
            self.calls = 0

        def __call__(self, url, json, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return vanish_then_reject()
            self.second = dict(json["images"][0])
            return _response(201, {"id": "evt-1"})

    rejecting = Rejecting()
    prepared = prepare_local_images(image, None)
    payload = {"event_schema": "custom", "images": [{"label": "x"}]}
    with patch.object(requests, "get", return_value=_health(_identity(shm_dir))):
        with patch.object(requests, "post", side_effect=rejecting):
            response = post_local_event(
                EVENTS_URL, payload=payload, prepared_images=prepared, timeout=5
            )

    assert response.status_code == 201
    assert base64.b64decode(rejecting.second["base64Image"]) == expected
