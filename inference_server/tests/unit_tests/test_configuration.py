import os
import subprocess
import sys

import pytest

from inference_server import configuration


def test_app_port_default_is_9001():
    assert configuration.APP_PORT_DEFAULT == 9001


def test_legacy_offline_mode_follows_model_layer_latch_not_server_variable():
    code = (
        "from inference_server import configuration as c; "
        "assert c.OFFLINE_MODE is False, c.OFFLINE_MODE; "
        "assert c.LEGACY_OFFLINE_MODE is True, c.LEGACY_OFFLINE_MODE; "
        "import inference_server.legacy.bridge as b, "
        "inference_server.legacy.common as m; "
        "assert b.LEGACY_OFFLINE_MODE is True and m.LEGACY_OFFLINE_MODE is True"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "OFFLINE_MODE": "false",
            "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START": "true",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _sam3_settings(exec_mode, offline, fine_tuned):
    code = (
        "from inference_server import configuration as c; "
        "print(c.SAM3_EXEC_MODE, c.SAM3_FINE_TUNED_MODELS_ENABLED)"
    )
    env = {
        name: value
        for name, value in os.environ.items()
        if name != "SAM3_FINE_TUNED_MODELS_ENABLED"
    }
    env.update({"OFFLINE_MODE": offline, "SAM3_EXEC_MODE": exec_mode})
    if fine_tuned is not None:
        env["SAM3_FINE_TUNED_MODELS_ENABLED"] = fine_tuned
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


@pytest.mark.parametrize(
    "exec_mode, offline, fine_tuned, expected",
    [
        ("local", "false", None, ["local", "True"]),
        ("local", "false", "false", ["local", "False"]),
        ("local", "true", None, ["local", "True"]),
        ("local", "true", "false", ["local", "False"]),
        ("remote", "false", None, ["remote", "False"]),
        ("remote", "false", "true", ["remote", "True"]),
        ("remote", "true", None, ["local", "True"]),
        ("remote", "true", "false", ["local", "False"]),
    ],
)
def test_sam3_settings_follow_legacy_for_every_mode_offline_combination(
    exec_mode, offline, fine_tuned, expected
):
    assert _sam3_settings(exec_mode, offline, fine_tuned) == expected


def test_hosted_flags_default_to_legacy_values():
    code = (
        "from inference_server import configuration as c; "
        "assert c.LAMBDA is False; "
        "assert c.GCP_SERVERLESS is False; "
        "assert c.ENFORCE_CREDITS_VERIFICATION is False; "
        "assert c.DEDICATED_DEPLOYMENT_ID is None; "
        "assert c.DEDICATED_DEPLOYMENT_WORKSPACE_ID is None; "
        "assert c.DEDICATED_DEPLOYMENT_WORKSPACE_URL is None; "
        "assert c.WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT is None; "
        "assert c.TRANSIENT_ROBOFLOW_API_ERRORS == set(); "
        "assert c.TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES == 3; "
        "assert c.TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL == 1; "
        "assert c.RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API is False; "
        "assert c.ROBOFLOW_SERVICE_SECRET is None; "
        "assert c.ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN is None"
    )
    env = {
        name: value
        for name, value in os.environ.items()
        if name
        not in {
            "LAMBDA",
            "GCP_SERVERLESS",
            "ENFORCE_CREDITS_VERIFICATION",
            "DEDICATED_DEPLOYMENT_ID",
            "DEDICATED_DEPLOYMENT_WORKSPACE_ID",
            "DEDICATED_DEPLOYMENT_WORKSPACE_URL",
            "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT",
            "ROBOFLOW_SERVICE_SECRET",
            "ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN",
            "ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN",
            "TRANSIENT_ROBOFLOW_API_ERRORS",
            "TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES",
            "TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL",
            "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API",
        }
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_hosted_flags_read_legacy_spellings():
    code = (
        "from inference_server import configuration as c; "
        "assert c.LAMBDA is True; "
        "assert c.GCP_SERVERLESS is True; "
        "assert c.ENFORCE_CREDITS_VERIFICATION is True; "
        "assert c.DEDICATED_DEPLOYMENT_WORKSPACE_URL == 'ws-url'; "
        "assert c.WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT == ['a', 'b']; "
        "assert c.ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN == 'legacy-tok'; "
        "assert c.TRANSIENT_ROBOFLOW_API_ERRORS == {502, 503}; "
        "assert c.TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES == 5; "
        "assert c.TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL == 2; "
        "assert c.RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API is True"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "LAMBDA": "True",
            "GCP_SERVERLESS": "true",
            "ENFORCE_CREDITS_VERIFICATION": "1",
            "DEDICATED_DEPLOYMENT_WORKSPACE_URL": "ws-url",
            "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT": " a, b ",
            "ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN": "legacy-tok",
            "TRANSIENT_ROBOFLOW_API_ERRORS": "502, 503",
            "TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES": "5",
            "TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL": "2",
            "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API": "true",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


_URL_RULE_SWITCHES = (
    "ALLOW_NON_HTTPS_URL_INPUT",
    "ALLOW_URL_INPUT_WITHOUT_FQDN",
    "VALIDATE_IMAGE_URL_REDIRECTS",
)


def test_url_rule_switches_default_to_legacy_values():
    code = "from inference_server import configuration as c; " + "; ".join(
        f"assert c.{name} is False" for name in _URL_RULE_SWITCHES
    )
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in _URL_RULE_SWITCHES
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_url_rule_switches_are_read_from_the_environment():
    code = "from inference_server import configuration as c; " + "; ".join(
        f"assert c.{name} is True" for name in _URL_RULE_SWITCHES
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, **{name: "True" for name in _URL_RULE_SWITCHES}},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


_VIDEO_SETTINGS = (
    "MAX_VIDEO_DOWNLOAD_SIZE_MB",
    "VIDEO_DOWNLOAD_TIMEOUT_SECONDS",
    "MAX_VIDEO_DURATION_SECONDS",
)


def _read_video_settings(values):
    code = (
        "from inference_server import configuration as c; "
        "print(repr(c.MAX_VIDEO_DOWNLOAD_SIZE_MB), "
        "repr(c.VIDEO_DOWNLOAD_TIMEOUT_SECONDS), "
        "repr(c.MAX_VIDEO_DURATION_SECONDS))"
    )
    env = {
        name: value for name, value in os.environ.items() if name not in _VIDEO_SETTINGS
    }
    env.update(values)
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def test_video_settings_default_to_legacy_values():
    assert _read_video_settings({}) == ["512", "60.0", "600.0"]


def test_video_settings_are_read_from_the_environment():
    values = {
        "MAX_VIDEO_DOWNLOAD_SIZE_MB": "-1",
        "VIDEO_DOWNLOAD_TIMEOUT_SECONDS": "2.5",
        "MAX_VIDEO_DURATION_SECONDS": "-1",
    }

    assert _read_video_settings(values) == ["-1", "2.5", "-1.0"]


def _read_allow_api_key_from_headers(value):
    env = {
        name: v
        for name, v in os.environ.items()
        if name != "ALLOW_API_KEY_FROM_HEADERS"
    }
    if value is not None:
        env["ALLOW_API_KEY_FROM_HEADERS"] = value
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from inference_server import configuration as c; "
            "print(c.ALLOW_API_KEY_FROM_HEADERS)",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_allow_api_key_from_headers_defaults_to_true():
    assert _read_allow_api_key_from_headers(None) == "True"


def test_allow_api_key_from_headers_is_read_from_environment():
    assert _read_allow_api_key_from_headers("False") == "False"


def _run_with_gateway(code, **env):
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )


def test_secure_gateway_is_the_normalised_model_layer_value():
    code = (
        "from inference_server import configuration as c; "
        "from inference_models import configuration as m; "
        "assert c.SECURE_GATEWAY == m.SECURE_GATEWAY == 'https://gw.example:8443', "
        "c.SECURE_GATEWAY"
    )
    result = _run_with_gateway(code, SECURE_GATEWAY="gw.example:8443")

    assert result.returncode == 0, result.stderr


def test_secure_gateway_is_unset_by_default():
    code = "from inference_server import configuration as c; assert c.SECURE_GATEWAY is None"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"SECURE_GATEWAY", "LICENSE_SERVER"}
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_workflows_host_forces_local_execution_behind_a_legacy_alias_gateway():
    code = (
        "import warnings; warnings.simplefilter('ignore'); "
        "import inference_server.workflows.host as h; "
        "c = h.build_workflows_configuration(); "
        "assert c.engine.step_execution_mode == 'local'"
    )
    env = {k: v for k, v in os.environ.items() if k != "SECURE_GATEWAY"}
    env.update(
        LICENSE_SERVER="https://gw.example",
        WORKFLOWS_STEP_EXECUTION_MODE="remote",
        WORKFLOWS_REMOTE_API_TARGET="hosted",
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


_USAGE_SETTINGS = (
    "API_BASE_URL",
    "METRICS_COLLECTOR_BASE_URL",
    "TELEMETRY_API_USAGE_ENDPOINT_URL",
    "TELEMETRY_FLUSH_INTERVAL",
    "TELEMETRY_QUEUE_SIZE",
    "TELEMETRY_USE_PERSISTENT_QUEUE",
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_SSL",
    "REDIS_TIMEOUT",
)


def _run_with_usage_settings(code, **env):
    base = {
        name: value for name, value in os.environ.items() if name not in _USAGE_SETTINGS
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**base, **env},
        capture_output=True,
        text=True,
    )


def test_usage_settings_default_to_legacy_values():
    code = (
        "from inference_server import configuration as c; "
        "assert c.METRICS_COLLECTOR_BASE_URL == 'https://api.roboflow.com'; "
        "assert c.TELEMETRY_API_USAGE_ENDPOINT_URL == "
        "'https://api.roboflow.com/usage/inference'; "
        "assert c.TELEMETRY_FLUSH_INTERVAL == 10; "
        "assert c.TELEMETRY_QUEUE_SIZE == 10; "
        "assert c.TELEMETRY_USE_PERSISTENT_QUEUE is True; "
        "assert c.REDIS_HOST is None; "
        "assert c.REDIS_PORT == 6379; "
        "assert c.REDIS_SSL is False; "
        "assert c.REDIS_TIMEOUT == 2.0"
    )

    result = _run_with_usage_settings(code)

    assert result.returncode == 0, result.stderr


def test_usage_endpoint_follows_the_api_and_metrics_collector_base_urls():
    code = (
        "from inference_server import configuration as c; "
        "print(c.TELEMETRY_API_USAGE_ENDPOINT_URL)"
    )

    from_api = _run_with_usage_settings(code, API_BASE_URL="https://api.example.com")
    from_collector = _run_with_usage_settings(
        code,
        API_BASE_URL="https://api.example.com",
        METRICS_COLLECTOR_BASE_URL="https://metrics.example.com",
    )

    assert from_api.stdout.strip() == "https://api.example.com/usage/inference"
    assert (
        from_collector.stdout.strip() == "https://metrics.example.com/usage/inference"
    )


def test_usage_settings_read_the_legacy_telemetry_names():
    code = (
        "from inference_server import configuration as c; "
        "assert c.TELEMETRY_API_USAGE_ENDPOINT_URL == 'https://custom.example.com/u'; "
        "assert c.TELEMETRY_FLUSH_INTERVAL == 30; "
        "assert c.TELEMETRY_QUEUE_SIZE == 500; "
        "assert c.TELEMETRY_USE_PERSISTENT_QUEUE is False; "
        "assert c.REDIS_HOST == 'redis.local'; "
        "assert c.REDIS_PORT == 6380; "
        "assert c.REDIS_SSL is True; "
        "assert c.REDIS_TIMEOUT == 0.5"
    )

    result = _run_with_usage_settings(
        code,
        METRICS_COLLECTOR_BASE_URL="https://metrics.example.com",
        TELEMETRY_API_USAGE_ENDPOINT_URL="https://custom.example.com/u",
        TELEMETRY_FLUSH_INTERVAL="30",
        TELEMETRY_QUEUE_SIZE="500",
        TELEMETRY_USE_PERSISTENT_QUEUE="false",
        REDIS_HOST="redis.local",
        REDIS_PORT="6380",
        REDIS_SSL="True",
        REDIS_TIMEOUT="0.5",
    )

    assert result.returncode == 0, result.stderr


def test_usage_flush_interval_and_queue_size_are_clamped():
    code = (
        "from inference_server import configuration as c; "
        "print(c.TELEMETRY_FLUSH_INTERVAL, c.TELEMETRY_QUEUE_SIZE)"
    )

    low = _run_with_usage_settings(
        code, TELEMETRY_FLUSH_INTERVAL="1", TELEMETRY_QUEUE_SIZE="1"
    )
    high = _run_with_usage_settings(
        code, TELEMETRY_FLUSH_INTERVAL="9999", TELEMETRY_QUEUE_SIZE="999999"
    )

    assert low.stdout.strip() == "10 10"
    assert high.stdout.strip() == "300 10000"


_TELEMETRY_SETTINGS = (
    "TELEMETRY_API_USAGE_ENDPOINT_URL",
    "TELEMETRY_FLUSH_INTERVAL",
    "TELEMETRY_USE_PERSISTENT_QUEUE",
    "TELEMETRY_QUEUE_SIZE",
)


def _without_telemetry_variables(monkeypatch):
    for name in list(os.environ):
        if name.upper() in _TELEMETRY_SETTINGS:
            monkeypatch.delenv(name)


def test_telemetry_settings_are_read_whatever_the_case_of_the_variable():
    code = (
        "from inference_server import configuration as c; "
        "assert c.TELEMETRY_API_USAGE_ENDPOINT_URL == 'https://custom.example.com/u'; "
        "assert c.TELEMETRY_FLUSH_INTERVAL == 30; "
        "assert c.TELEMETRY_QUEUE_SIZE == 500; "
        "assert c.TELEMETRY_USE_PERSISTENT_QUEUE is False"
    )

    result = _run_with_usage_settings(
        code,
        telemetry_api_usage_endpoint_url="https://custom.example.com/u",
        Telemetry_Flush_Interval="30",
        telemetry_queue_size="500",
        TeLeMeTrY_UsE_PeRsIsTeNt_QuEuE="false",
    )

    assert result.returncode == 0, result.stderr


def test_telemetry_name_lookup_prefers_the_exact_upper_case_name(monkeypatch):
    _without_telemetry_variables(monkeypatch)
    monkeypatch.setenv("telemetry_queue_size", "700")
    monkeypatch.setenv("TELEMETRY_QUEUE_SIZE", "500")

    assert configuration._telemetry_env_name("TELEMETRY_QUEUE_SIZE") == (
        "TELEMETRY_QUEUE_SIZE"
    )


def test_telemetry_name_lookup_takes_the_first_sorted_variable_otherwise(
    monkeypatch,
):
    _without_telemetry_variables(monkeypatch)
    monkeypatch.setenv("telemetry_queue_size", "700")
    monkeypatch.setenv("Telemetry_Queue_Size", "600")

    assert configuration._telemetry_env_name("TELEMETRY_QUEUE_SIZE") == (
        "Telemetry_Queue_Size"
    )


def test_telemetry_name_lookup_falls_back_to_the_given_name(monkeypatch):
    _without_telemetry_variables(monkeypatch)

    assert configuration._telemetry_env_name("TELEMETRY_QUEUE_SIZE") == (
        "TELEMETRY_QUEUE_SIZE"
    )


def test_telemetry_precedence_of_the_upper_case_name_applies_to_the_settings():
    code = (
        "from inference_server import configuration as c; "
        "print(c.TELEMETRY_QUEUE_SIZE)"
    )

    result = _run_with_usage_settings(
        code, TELEMETRY_QUEUE_SIZE="500", telemetry_queue_size="700"
    )

    assert result.stdout.strip() == "500"


def _workflow_cache_flags(offline, file_cache, single_tenant):
    code = (
        "from inference_server import configuration as c; "
        "print(c.USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS, "
        "c.SINGLE_TENANT_WORKFLOW_CACHE)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "OFFLINE_MODE": offline,
            "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START": offline,
            "USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS": file_cache,
            "SINGLE_TENANT_WORKFLOW_CACHE": single_tenant,
        },
        capture_output=True,
        text=True,
    )


def test_offline_mode_forces_the_workflow_file_cache_and_single_tenant_layout():
    result = _workflow_cache_flags("true", "false", "false")

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True", "True"]
    assert "USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS=False" in result.stderr


def test_online_mode_keeps_the_given_workflow_cache_flags():
    result = _workflow_cache_flags("false", "false", "false")

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False", "False"]


_STREAM_SETTINGS_CODE = (
    "from inference_server import configuration as c; "
    "print(c.ENABLE_STREAM_API, c.STREAM_API_PRELOADED_PROCESSES, "
    "c.STREAM_MANAGER_OPERATIONS_TIMEOUT, c.STREAM_MANAGER_MAX_RAM_MB, "
    "c.STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE, c.VIDEO_SOURCE_BUFFER_SIZE, "
    "c.VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE)"
)

_STREAM_SETTING_NAMES = (
    "ENABLE_STREAM_API",
    "STREAM_API_PRELOADED_PROCESSES",
    "STREAM_MANAGER_OPERATIONS_TIMEOUT",
    "STREAM_MANAGER_MAX_RAM_MB",
    "STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE",
    "VIDEO_SOURCE_BUFFER_SIZE",
    "VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE",
)


def _stream_settings(**overrides):
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in _STREAM_SETTING_NAMES
    }
    env.update(overrides)
    result = subprocess.run(
        [sys.executable, "-c", _STREAM_SETTINGS_CODE],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    return result.stdout.split()


def test_stream_settings_default_to_legacy_values():
    assert _stream_settings() == ["False", "0", "None", "None", "10", "None", "None"]


def test_stream_settings_are_read_under_the_legacy_names():
    settings = _stream_settings(
        ENABLE_STREAM_API="True",
        STREAM_API_PRELOADED_PROCESSES="2",
        STREAM_MANAGER_OPERATIONS_TIMEOUT="1.5",
        STREAM_MANAGER_MAX_RAM_MB="-512",
        STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE="-3",
        VIDEO_SOURCE_BUFFER_SIZE="4",
        VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE="False",
    )

    assert settings == ["True", "2", "1.5", "512.0", "3", "4", "False"]


def test_stream_manager_memory_settings_ignore_unparsable_values():
    settings = _stream_settings(
        STREAM_MANAGER_MAX_RAM_MB="lots", STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE="many"
    )

    assert settings[3:5] == ["None", "10"]
