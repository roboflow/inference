import hashlib
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Barrier, Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import pytest

from inference_models.errors import (
    CorruptedModelPackageError,
    ModelPackageAlternativesExhaustedError,
    UnauthorizedModelAccessError,
)
from inference_models.models.auto_loaders import core, model_cache_paths
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.utils import download
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    ModelDependency,
    ModelMetadata,
    ModelPackageMetadata,
)


@pytest.fixture
def artifact_server():
    payload = bytes(range(256)) * 32
    state = SimpleNamespace(
        mode="one-range",
        threaded=True,
        requests=[],
        heads=0,
        lock=Lock(),
        barrier=Barrier(8),
        payload=payload,
        config_downloads=0,
        config_completed=Event(),
        with_config=False,
        head_forbidden=False,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_HEAD(self):
            if urlparse(self.path).path == "/config.json":
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                return

            with state.lock:
                state.heads += 1
            if state.head_forbidden:
                self.send_response(403)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            if urlparse(self.path).path == "/config.json":
                with state.lock:
                    state.config_downloads += 1
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
                state.config_completed.set()
                return

            if state.with_config:
                assert state.config_completed.wait(timeout=10)
            byte_range = self.headers.get("Range")
            if byte_range:
                start, end = map(int, byte_range.removeprefix("bytes=").split("-"))
            else:
                start, end = 0, len(payload) - 1
            with state.lock:
                first_pass = state.heads == 1
            if byte_range:
                state.barrier.wait(timeout=10)
            signature = parse_qs(urlparse(self.path).query)["Signature"][0]
            forbidden = (
                state.mode == "permanent"
                or (state.mode == "expired" and signature == "old-secret")
                or (state.mode == "one-range" and first_pass and start == 0)
                or (state.mode == "all-ranges" and first_pass)
            )
            status = 403 if forbidden else (206 if byte_range else 200)
            with state.lock:
                state.requests.append((signature, byte_range, status))
            self.send_response(status)
            self.send_header("x-goog-request-id", "test-request-123")
            body = b"forbidden" if forbidden else payload[start : end + 1]
            self.send_header("Content-Length", str(len(body)))
            if byte_range and not forbidden:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}/model.safetensors"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@pytest.fixture
def model_load(artifact_server, monkeypatch, tmp_path):
    server = artifact_server
    monkeypatch.setattr(download, "MIN_SIZE_FOR_THREADED_DOWNLOAD", 1)
    monkeypatch.setattr(download, "MIN_THREAD_CHUNK_SIZE", 1)
    monkeypatch.setattr(download, "DISABLE_INTERACTIVE_PROGRESS_BARS", True)
    monkeypatch.setattr(core, "OFFLINE_MODE", False)
    monkeypatch.setattr(core, "OFFLINE_MODE_WARM_UP", False)
    monkeypatch.setattr(core, "ROBOFLOW_API_KEY", None)
    monkeypatch.setattr(model_cache_paths, "INFERENCE_HOME", str(tmp_path))
    package = ModelPackageMetadata(
        package_id="testpackage",
        backend=BackendType.ONNX,
        trusted_source=True,
        package_artefacts=[
            FileDownloadSpecs(
                file_handle="model.safetensors",
                download_url=f"{server.url}?Expires=1&KeyName=old-key&Signature=old-secret",
                md5_hash=hashlib.md5(server.payload).hexdigest(),
            )
        ],
    )
    metadata = ModelMetadata(
        model_id="tenant/canonical/1",
        model_architecture="yolov8",
        model_packages=[package],
        task_type="object-detection",
    )
    state = SimpleNamespace(
        metadata=metadata,
        refreshed_metadata=metadata,
        package=package,
        root=tmp_path,
        provider_calls=[],
        refresh_error=None,
        model=MagicMock(),
    )

    def provider(**kwargs):
        state.provider_calls.append(kwargs)
        if len(state.provider_calls) == 1:
            return state.metadata
        if state.refresh_error is not None:
            raise state.refresh_error
        return state.refreshed_metadata

    monkeypatch.setattr(core, "get_model_from_provider", provider)
    monkeypatch.setattr(
        core, "negotiate_model_packages", lambda **_kwargs: [state.package]
    )
    model_class = MagicMock()
    model_class.from_pretrained.return_value = state.model
    monkeypatch.setattr(core, "resolve_model_class", lambda **_kwargs: model_class)
    state.model_class = model_class

    def load(**kwargs):
        options = dict(
            api_key="test-api-key",
            use_auto_resolution_cache=False,
            auto_resolution_cache=MagicMock(),
            weights_provider_extra_query_params={"service_secret": "test-query"},
            weights_provider_extra_headers={"X-Test-Gateway": "test-header"},
            max_package_loading_attempts=1,
            verbose=False,
        )
        options.update(kwargs)
        return core.AutoModel.from_pretrained("requested-alias/1", **options)

    state.load = load
    return state


@pytest.mark.parametrize(
    "mode", ["one-range", "all-ranges", "expired", "stream", "head"]
)
def test_artifact_403_refreshes_once_and_finishes_download(
    artifact_server, model_load, monkeypatch, caplog, mode
):
    server = artifact_server
    server.mode = mode if mode not in {"stream", "head"} else "one-range"
    if mode == "head":
        server.head_forbidden = True
    if mode == "stream":
        server.threaded = False
        monkeypatch.setattr(download, "MIN_SIZE_FOR_THREADED_DOWNLOAD", 10**9)
    if mode == "expired":
        new_artifact = replace(
            model_load.package.package_artefacts[0],
            download_url=f"{server.url}?Expires=2&KeyName=new-key&Signature=new-secret",
        )
        new_package = replace(model_load.package, package_artefacts=[new_artifact])
        model_load.refreshed_metadata = replace(
            model_load.metadata, model_packages=[new_package]
        )

    assert model_load.load() is model_load.model

    assert len(model_load.provider_calls) == 2
    assert model_load.provider_calls[1] == {
        **model_load.provider_calls[0],
        "model_id": model_load.metadata.model_id,
    }
    statuses = [request[2] for request in server.requests]
    assert statuses.count(403) == (8 if mode in {"all-ranges", "expired"} else 1)
    assert statuses.count(206) == (
        15 if mode == "one-range" else 8 if mode not in {"stream", "head"} else 0
    )
    artifact_path = next(model_load.root.rglob("model.safetensors"))
    assert artifact_path.read_bytes() == server.payload
    assert "old-secret" not in caplog.text
    assert "new-secret" not in caplog.text
    assert "test-api-key" not in caplog.text
    assert "artifact=model.safetensors" in caplog.text
    assert "request_id=test-request-123" in caplog.text
    assert "recovery attempt 1/1" in caplog.text
    if mode not in {"stream", "head"}:
        assert "range=bytes=" in caplog.text


def test_permanent_artifact_403_is_bounded_and_removes_partial_files(
    artifact_server, model_load, caplog
):
    artifact_server.mode = "permanent"

    with pytest.raises(ModelPackageAlternativesExhaustedError) as error:
        model_load.load()

    assert len(model_load.provider_calls) == 2
    assert len(artifact_server.requests) == 16
    assert not list(model_load.root.rglob("model.safetensors"))
    digest = model_load.package.package_artefacts[0].md5_hash
    assert not list(model_load.root.rglob(f"{digest}*"))
    model_load.model_class.from_pretrained.assert_not_called()
    assert "403" in str(error.value)
    assert "old-secret" not in str(error.value) + caplog.text


def test_metadata_denial_is_not_retried(artifact_server, model_load):
    model_load.refresh_error = UnauthorizedModelAccessError("metadata denied")

    with pytest.raises(ModelPackageAlternativesExhaustedError) as error:
        model_load.load()

    assert len(model_load.provider_calls) == 2
    assert len(artifact_server.requests) == 8
    assert error.value.alternatives_errors == [model_load.refresh_error]


@pytest.mark.parametrize(
    "change",
    [
        "model",
        "package",
        "backend",
        "hash",
        "handle",
        "source",
        "trust",
        "architecture",
        "dependencies",
    ],
)
def test_refresh_rejects_changed_identity(artifact_server, model_load, change):
    package = model_load.package
    metadata = model_load.metadata
    if change == "model":
        metadata = replace(metadata, model_id="another/model/1")
    elif change == "architecture":
        metadata = replace(metadata, model_architecture="yolov5")
    elif change == "dependencies":
        metadata = replace(
            metadata,
            model_dependencies=[ModelDependency(name="encoder", model_id="other/1")],
        )
    elif change in {"package", "backend", "trust"}:
        key, value = {
            "package": ("package_id", "other-package"),
            "backend": ("backend", BackendType.TORCH_SCRIPT),
            "trust": ("trusted_source", False),
        }[change]
        package = replace(package, **{key: value})
    else:
        key, value = {
            "hash": ("md5_hash", "0" * 32),
            "handle": ("file_handle", "other.safetensors"),
            "source": (
                "download_url",
                f"{artifact_server.url}/other?Signature=new-secret",
            ),
        }[change]
        package = replace(
            package,
            package_artefacts=[replace(package.package_artefacts[0], **{key: value})],
        )
    model_load.refreshed_metadata = replace(metadata, model_packages=[package])

    with pytest.raises(ModelPackageAlternativesExhaustedError) as error:
        model_load.load()

    assert len(model_load.provider_calls) == 2
    assert len(artifact_server.requests) == 8
    assert isinstance(error.value.alternatives_errors[0], CorruptedModelPackageError)
    model_load.model_class.from_pretrained.assert_not_called()


def test_success_and_cached_artifacts_do_not_refresh(artifact_server, model_load):
    artifact_server.mode = "success"

    model_load.load()
    assert len(model_load.provider_calls) == 1
    assert len(artifact_server.requests) == 8
    model_load.load()
    assert len(model_load.provider_calls) == 2
    assert len(artifact_server.requests) == 8


def test_custom_provider_does_not_refresh_artifact_403(artifact_server, model_load):
    with pytest.raises(ModelPackageAlternativesExhaustedError):
        model_load.load(weights_provider="custom")

    assert len(model_load.provider_calls) == 1
    assert len(artifact_server.requests) == 8


def test_model_constructor_403_does_not_refresh(artifact_server, model_load):
    from requests import HTTPError, Response

    artifact_server.mode = "success"
    response = Response()
    response.status_code = 403
    model_load.model_class.from_pretrained.side_effect = HTTPError(response=response)

    with pytest.raises(ModelPackageAlternativesExhaustedError):
        model_load.load()

    assert len(model_load.provider_calls) == 1


def test_refresh_reuses_completed_verified_file(artifact_server, model_load):
    artifact_server.with_config = True
    config = FileDownloadSpecs(
        file_handle="config.json",
        download_url=artifact_server.url.replace("model.safetensors", "config.json"),
        md5_hash=hashlib.md5(b"{}").hexdigest(),
    )
    model_load.package = replace(
        model_load.package,
        package_artefacts=[config, *model_load.package.package_artefacts],
    )
    model_load.metadata = replace(
        model_load.metadata, model_packages=[model_load.package]
    )
    model_load.refreshed_metadata = model_load.metadata

    model_load.load()

    assert len(model_load.provider_calls) == 2
    assert artifact_server.config_downloads == 1
    assert next(model_load.root.rglob("config.json")).read_bytes() == b"{}"
    assert (
        next(model_load.root.rglob("model.safetensors")).read_bytes()
        == artifact_server.payload
    )


def test_unhashed_artifact_refresh_preserves_download_opt_in(
    artifact_server, model_load, monkeypatch
):
    # Unhashed package metadata/config files use the streaming path in production.
    monkeypatch.setattr(download, "MIN_SIZE_FOR_THREADED_DOWNLOAD", 32 * 1024 * 1024)
    artifact = replace(model_load.package.package_artefacts[0], md5_hash=None)
    model_load.package = replace(model_load.package, package_artefacts=[artifact])
    model_load.metadata = replace(
        model_load.metadata, model_packages=[model_load.package]
    )
    model_load.refreshed_metadata = model_load.metadata

    model_load.load(download_files_without_hash=True)

    assert len(model_load.provider_calls) == 2
    assert (
        next(model_load.root.rglob("model.safetensors")).read_bytes()
        == artifact_server.payload
    )


def test_refresh_bypasses_warm_up_metadata(artifact_server, model_load, monkeypatch):
    monkeypatch.setattr(core, "OFFLINE_MODE_WARM_UP", True)
    monkeypatch.setattr(core, "_maybe_record_warm_up_load", lambda **_kwargs: None)

    model_load.load()

    assert len(model_load.provider_calls) == 2
    assert len(artifact_server.requests) == 16
