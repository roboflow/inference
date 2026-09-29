import hashlib
import json
import logging
import os
import subprocess
import sys

import pytest
import requests
import requests_mock as rm
from inference_models.errors import ModelRetrievalError


def test_configuration_installed_on_import(monkeypatch):
    monkeypatch.setenv("WORKFLOWS_STEP_EXECUTION_MODE", "local")
    from roboflow_workflows.configuration import get_configuration

    import inference_server.workflows.host as host

    assert get_configuration() is host.SERVER_WORKFLOWS_CONFIGURATION


def test_offline_mode_forces_local_step_execution(monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "true")
    monkeypatch.setenv("WORKFLOWS_STEP_EXECUTION_MODE", "REMOTE")
    import inference_server.workflows.host as host

    with pytest.warns(UserWarning):
        configuration = host.build_workflows_configuration()
    assert (
        configuration.engine.step_execution_mode == "local"
        and configuration.platform.offline_mode is True
    )


def test_secure_gateway_with_hosted_remote_target_forces_local(monkeypatch):
    monkeypatch.setenv("SECURE_GATEWAY", "https://gw.example")
    monkeypatch.setenv("WORKFLOWS_STEP_EXECUTION_MODE", "remote")
    monkeypatch.setenv("WORKFLOWS_REMOTE_API_TARGET", "hosted")
    import inference_server.workflows.host as host

    with pytest.warns(UserWarning):
        assert (
            host.build_workflows_configuration().engine.step_execution_mode == "local"
        )


def test_offline_modal_execution_is_refused(monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "true")
    monkeypatch.setenv("WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal")
    import inference_server.workflows.host as host

    with pytest.raises(RuntimeError):
        host.build_workflows_configuration()


def test_invalid_api_key_transport_is_refused(monkeypatch):
    monkeypatch.setenv("WORKFLOWS_REMOTE_API_KEY_TRANSPORT", "carrier-pigeon")
    import inference_server.workflows.host as host

    with pytest.raises(ValueError):
        host.build_workflows_configuration()


def test_postgresql_sink_addresses_reproduce_the_legacy_set_parsing(monkeypatch):
    monkeypatch.setenv("POSTGRESQL_WORKFLOWS_SINK_WHITELISTED_ADDRESSES", "")
    monkeypatch.delenv("POSTGRESQL_WORKFLOWS_SINK_BLACKLISTED_ADDRESSES", raising=False)
    import inference_server.workflows.host as host

    engine = host.build_workflows_configuration().engine
    assert set(engine.postgresql_sink_whitelisted_addresses) == {""}
    assert engine.postgresql_sink_blacklisted_addresses is None


def test_platform_client_posts_with_headers_and_key():
    import inference_server.workflows.host as host

    with rm.Mocker() as m:
        m.post("https://api.roboflow.com/x/y?api_key=k", json={"ok": 1})
        assert host.PLATFORM_CLIENT.post("x/y", api_key="k", payload={"a": 1}) == {
            "ok": 1
        }
        assert m.last_request.headers["x-allow-chunked-response"] == "true"


def test_platform_client_error_raises_legacy_http_error():
    import inference_server.workflows.host as host
    from inference_server.legacy.errors import LegacyHTTPError

    with rm.Mocker() as m:
        m.post(
            "https://api.roboflow.com/x/z?api_key=secret-key",
            status_code=403,
            json={"message": "denied for api_key=secret-key"},
        )
        with pytest.raises(LegacyHTTPError) as exc:
            host.PLATFORM_CLIENT.post("x/z", api_key="secret-key")
    assert exc.value.status_code == 403
    assert "secret-key" not in exc.value.message


def test_platform_client_treats_non_2xx_as_error():
    import inference_server.workflows.host as host
    from inference_server.legacy.errors import LegacyHTTPError

    seen = []
    with rm.Mocker() as m:
        m.post("https://api.roboflow.com/x/c?api_key=k", status_code=304)
        with pytest.raises(LegacyHTTPError) as exc:
            host.PLATFORM_CLIENT.post(
                "x/c", api_key="k", http_errors_handlers={304: seen.append}
            )
    assert exc.value.status_code == 304 and len(seen) == 1


def test_build_api_headers_merge_order(monkeypatch):
    import inference_server.workflows.host as host
    from inference_server import configuration

    monkeypatch.setattr(
        configuration,
        "ROBOFLOW_API_EXTRA_HEADERS",
        json.dumps({"x-allow-chunked-response": "from-extra", "x-extra": "1"}),
    )
    headers = host.PLATFORM_CLIENT.build_api_headers()
    assert headers["x-allow-chunked-response"] == "from-extra"
    assert headers["x-extra"] == "1"
    assert (
        host.PLATFORM_CLIENT.build_api_headers(
            explicit_headers={"x-allow-chunked-response": "from-explicit"}
        )["x-allow-chunked-response"]
        == "from-explicit"
    )


def test_get_workflow_specification_extracts_and_caches():
    import inference_server.workflows.host as host

    payload = {
        "workflow": {
            "id": "wf-internal",
            "config": json.dumps({"specification": {"version": "1.0"}, "other": 1}),
        }
    }
    with rm.Mocker() as m:
        m.get("https://api.roboflow.com/ws/workflows/wf?api_key=k", json=payload)
        assert host.get_workflow_specification("k", "ws", "wf") == {
            "version": "1.0",
            "id": "wf-internal",
        }
        assert host.get_workflow_specification("k", "ws", "wf") == {
            "version": "1.0",
            "id": "wf-internal",
        }
        assert m.call_count == 1


def test_get_workflow_specification_malformed_is_502():
    import inference_server.workflows.host as host
    from inference_server.legacy.errors import LegacyHTTPError

    with rm.Mocker() as m:
        m.get(
            "https://api.roboflow.com/ws/workflows/bad?api_key=k",
            json={"workflow": {"config": json.dumps({"nope": 1})}},
        )
        with pytest.raises(LegacyHTTPError) as exc:
            host.get_workflow_specification("k", "ws", "bad")
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_image_codec_fetches_url_from_plain_worker_thread(monkeypatch):
    import asyncio
    import io
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    import inference_server.workflows.host as host
    from inference_server.legacy.bridge import LoopBridge

    buf = io.BytesIO()
    Image.new("RGB", (4, 3)).save(buf, format="JPEG")

    async def _fetch(urls):
        return [buf.getvalue() for _ in urls], None

    monkeypatch.setattr("inference_server.legacy.bridge.fetch_images_from_urls", _fetch)
    host.GUARDED_IMAGE_CODEC.bind_loop(LoopBridge(asyncio.get_running_loop()))
    with ThreadPoolExecutor(max_workers=1) as pool:
        image = await asyncio.get_running_loop().run_in_executor(
            pool, lambda: host.GUARDED_IMAGE_CODEC.fetch_url("https://a/1.jpg")
        )
    assert image.shape == (3, 4, 3)


def test_bind_image_codec_installs_the_object_from_init_parameters():
    from roboflow_workflows.prototypes.image_codec import (
        get_image_codec,
        reset_image_codec,
    )

    import inference_server.workflows.host as host

    reset_image_codec()
    params = {}
    host.bind_image_codec(params)
    assert (
        params["workflows_core.image_codec"] is host.GUARDED_IMAGE_CODEC
        and get_image_codec() is host.GUARDED_IMAGE_CODEC
    )
    host.bind_image_codec({})
    reset_image_codec()


def test_workspace_resolver_returns_none_on_failure():
    import inference_server.workflows.host as host

    with rm.Mocker() as m:
        m.get("https://api.roboflow.com/?api_key=bad&nocache=true", status_code=401)
        assert host.WORKSPACE_RESOLVER.resolve_workspace("bad") is None


def test_workspace_resolver_returns_workspace():
    import inference_server.workflows.host as host

    with rm.Mocker() as m:
        m.get(
            "https://api.roboflow.com/?api_key=good&nocache=true",
            json={"workspace": "my-ws"},
        )
        assert host.WORKSPACE_RESOLVER.resolve_workspace("good") == "my-ws"


def test_image_codec_refuses_pickled_numpy_and_local_files():
    from roboflow_workflows.errors import WorkflowImageLoadError

    import inference_server.workflows.host as host

    with pytest.raises(WorkflowImageLoadError):
        host.GUARDED_IMAGE_CODEC.load_image({"type": "numpy", "value": b"x"})
    with pytest.raises(WorkflowImageLoadError):
        host.GUARDED_IMAGE_CODEC.ensure_local_file_load_allowed("/tmp/x.jpg")


def test_step_error_handler_maps_unauthorized():
    from roboflow_workflows.errors import ClientCausedStepExecutionError

    import inference_server.workflows.host as host
    from inference_models.errors import UnauthorizedModelAccessError

    with pytest.raises(ClientCausedStepExecutionError) as exc:
        host.step_error_handler("step", UnauthorizedModelAccessError("nope"))
    assert exc.value.status_code == 401


def test_step_error_handler_maps_permission_error_to_401():
    from roboflow_workflows.errors import ClientCausedStepExecutionError

    import inference_server.workflows.host as host

    with pytest.raises(ClientCausedStepExecutionError) as exc:
        host.step_error_handler("step", PermissionError("nope"))
    assert exc.value.status_code == 401
    assert "Roboflow API key" in exc.value.public_message


def test_step_error_handler_maps_lookup_error_to_404():
    from roboflow_workflows.errors import ClientCausedStepExecutionError

    import inference_server.workflows.host as host

    with pytest.raises(ClientCausedStepExecutionError) as exc:
        host.step_error_handler("step", LookupError("ds/1"))
    assert exc.value.status_code == 404
    assert "not existing model" in exc.value.public_message


def test_step_error_handler_maps_legacy_http_error_and_sdk_error():
    from roboflow_workflows.errors import (
        ClientCausedStepExecutionError,
        RuntimeLimitsCausedStepExecutionError,
    )

    import inference_server.workflows.host as host
    from inference_sdk.http.errors import HTTPCallErrorError
    from inference_server.legacy.errors import LegacyHTTPError

    with pytest.raises(ClientCausedStepExecutionError) as exc:
        host.step_error_handler("step", LegacyHTTPError(404, "nope"))
    assert exc.value.status_code == 404
    with pytest.raises(RuntimeLimitsCausedStepExecutionError) as exc:
        host.step_error_handler(
            "step",
            HTTPCallErrorError(description="d", status_code=507, api_message="too big"),
        )
    assert exc.value.status_code == 507
    assert host.step_error_handler("step", ValueError("unmapped")) is None


def test_step_error_handler_maps_feature_deprecated():
    from roboflow_workflows.errors import ClientCausedStepExecutionError
    from roboflow_workflows.prototypes.platform_errors import FeatureDeprecatedError

    import inference_server.workflows.host as host

    with pytest.raises(ClientCausedStepExecutionError) as exc:
        host.step_error_handler("step", FeatureDeprecatedError("thing"))
    assert exc.value.status_code == 410


def test_workflows_platform_bindings_expose_the_module_singletons():
    import inference_server.workflows.host as host

    bindings = host.workflows_platform_bindings()
    assert bindings == {
        "workflows_core.cache": host.WORKFLOWS_CACHE,
        "workflows_core.platform_client": host.PLATFORM_CLIENT,
        "workflows_core.workspace_resolver": host.WORKSPACE_RESOLVER,
        "workflows_core.inner_workflow_spec_resolver": host.inner_workflow_spec_resolver,
    }


def test_configuration_is_installed_before_environment_is_imported():
    code = (
        "import inference_server.workflows.host, "
        "roboflow_workflows.environment as e; "
        "assert e.WORKFLOWS_STEP_EXECUTION_MODE == 'local', "
        "e.WORKFLOWS_STEP_EXECUTION_MODE"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "OFFLINE_MODE": "true",
            "WORKFLOWS_STEP_EXECUTION_MODE": "remote",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_step_error_handler_leaves_key_and_index_errors_unmapped():
    import inference_server.workflows.host as host

    assert host.step_error_handler("step", KeyError("k")) is None
    assert host.step_error_handler("step", IndexError("i")) is None


def _raise_request_error(error):
    def _raise(*args, **kwargs):
        raise error

    return _raise


def _retrieval_error(status_code):
    error = ModelRetrievalError("denied")
    error.status_code = status_code
    return error


def test_platform_client_post_sanitizes_transport_errors(monkeypatch):
    import inference_server.workflows.host as host
    from inference_server.legacy.errors import LegacyHTTPError

    monkeypatch.setattr(
        requests,
        "post",
        _raise_request_error(
            requests.exceptions.ConnectionError("http://x/?api_key=SECRET")
        ),
    )
    with pytest.raises(LegacyHTTPError) as exc:
        host.PLATFORM_CLIENT.post("x/y", api_key="SECRET")
    assert exc.value.status_code == 503 and "SECRET" not in exc.value.message


def test_platform_client_post_maps_timeout_to_504(monkeypatch):
    import inference_server.workflows.host as host
    from inference_server.legacy.errors import LegacyHTTPError

    monkeypatch.setattr(
        requests,
        "post",
        _raise_request_error(requests.exceptions.Timeout("http://x/?api_key=SECRET")),
    )
    with pytest.raises(LegacyHTTPError) as exc:
        host.PLATFORM_CLIENT.post("x/y", api_key="SECRET")
    assert exc.value.status_code == 504 and "SECRET" not in exc.value.message


def test_fetch_workflow_response_sanitizes_transport_errors(monkeypatch):
    import inference_server.workflows.host as host
    from inference_server.legacy.errors import LegacyHTTPError

    monkeypatch.setattr(
        requests,
        "get",
        _raise_request_error(
            requests.exceptions.ConnectionError("http://x/?api_key=SECRET")
        ),
    )
    with pytest.raises(LegacyHTTPError) as exc:
        host._fetch_workflow_response(
            api_key="SECRET",
            workspace_id="ws",
            workflow_id="wf",
            workflow_version_id=None,
        )
    assert exc.value.status_code == 503 and "SECRET" not in exc.value.message


def test_workspace_resolver_does_not_log_the_api_key(monkeypatch, caplog):
    import inference_server.workflows.host as host

    monkeypatch.setattr(
        requests,
        "get",
        _raise_request_error(
            requests.exceptions.ConnectionError("http://x/?api_key=SECRET")
        ),
    )
    with caplog.at_level(logging.WARNING, logger="inference_server.workflows.host"):
        assert host.WORKSPACE_RESOLVER.resolve_workspace("SECRET") is None
    assert caplog.records and "SECRET" not in caplog.text


def test_step_error_handler_unwraps_registry_model_access_error():
    from roboflow_workflows.errors import ClientCausedStepExecutionError

    import inference_server.workflows.host as host

    try:
        raise RuntimeError("denied") from _retrieval_error(403)
    except RuntimeError as error:
        wrapped = error
    with pytest.raises(ClientCausedStepExecutionError) as exc:
        host.step_error_handler("step", wrapped)
    assert exc.value.status_code == 403 and exc.value.block_id == "step"


def test_step_error_handler_unwraps_registry_os_error():
    from roboflow_workflows.errors import ClientCausedStepExecutionError

    import inference_server.workflows.host as host
    from inference_server.legacy.errors import REGISTRY_UNREACHABLE_MESSAGE

    try:
        raise RuntimeError("unreachable") from OSError("down")
    except RuntimeError as error:
        wrapped = error
    with pytest.raises(ClientCausedStepExecutionError) as exc:
        host.step_error_handler("step", wrapped)
    assert exc.value.status_code == 503
    assert exc.value.public_message == REGISTRY_UNREACHABLE_MESSAGE


def test_step_error_handler_leaves_plain_runtime_error_unmapped():
    import inference_server.workflows.host as host

    assert host.step_error_handler("step", RuntimeError("x")) is None


def _local_workflow_path(cache_root, workflow_id):
    return (
        cache_root
        / "workflow"
        / "local"
        / f"{hashlib.sha256(workflow_id.encode()).hexdigest()}.json"
    )


def test_local_workflow_response_reads_regular_file(monkeypatch, tmp_path):
    import inference_server.workflows.host as host
    from inference_server import configuration

    monkeypatch.setattr(configuration, "MODEL_CACHE_DIR", str(tmp_path))
    path = _local_workflow_path(tmp_path, "wf")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"config": "{}"}))
    assert host._local_workflow_response("wf") == {"workflow": {"config": "{}"}}


def test_local_workflow_response_refuses_symlinked_file(monkeypatch, tmp_path):
    import inference_server.workflows.host as host
    from inference_server import configuration

    monkeypatch.setattr(configuration, "MODEL_CACHE_DIR", str(tmp_path))
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({"config": "{}"}))
    path = _local_workflow_path(tmp_path, "wf")
    path.parent.mkdir(parents=True)
    path.symlink_to(target)
    with pytest.raises(FileNotFoundError):
        host._local_workflow_response("wf")


def test_local_workflow_response_refuses_symlinked_directory(monkeypatch, tmp_path):
    import inference_server.workflows.host as host
    from inference_server import configuration

    monkeypatch.setattr(configuration, "MODEL_CACHE_DIR", str(tmp_path))
    real_dir = tmp_path / "elsewhere"
    real_dir.mkdir()
    path = _local_workflow_path(tmp_path, "wf")
    (tmp_path / "workflow").mkdir()
    path.parent.symlink_to(real_dir, target_is_directory=True)
    (real_dir / path.name).write_text(json.dumps({"config": "{}"}))
    with pytest.raises(FileNotFoundError):
        host._local_workflow_response("wf")


def test_local_workflow_response_missing_file(monkeypatch, tmp_path):
    import inference_server.workflows.host as host
    from inference_server import configuration

    monkeypatch.setattr(configuration, "MODEL_CACHE_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        host._local_workflow_response("wf")
