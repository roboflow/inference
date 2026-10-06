import ast
import logging
import os
from pathlib import Path

import pytest

from inference_server.legacy_env import (
    DERIVED_URL_DEFAULTS,
    LEGACY_DEFAULTS,
    LEGACY_ENV_ALIASES,
    LEGACY_FLAG_ALIASES,
    apply_legacy_env,
)

_PKG_ROOT = Path(__file__).resolve().parents[2] / "inference_server"
_CALL_POINTS = ("app.py", "launcher.py")
_CONFIGURED_PACKAGES = (
    "inference_server",
    "inference_model_manager",
    "inference_models",
    "roboflow_workflows",
    "streamvision",
)
_DERIVATION_INPUTS = (
    "ROBOFLOW_REGION",
    "ROBOFLOW_ENVIRONMENT",
    "PROJECT",
    "ROBOFLOW_API_HOST",
)
_ALL_NAMES = (
    list(LEGACY_ENV_ALIASES)
    + list(LEGACY_ENV_ALIASES.values())
    + list(LEGACY_FLAG_ALIASES)
    + list(LEGACY_FLAG_ALIASES.values())
    + list(LEGACY_DEFAULTS)
    + list(DERIVED_URL_DEFAULTS)
    + list(_DERIVATION_INPUTS)
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in _ALL_NAMES:
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def test_mapping_matches_audit():
    assert LEGACY_ENV_ALIASES == {
        "MAX_ACTIVE_MODELS": "INFERENCE_MAX_ACTIVE_MODELS",
        "PRELOAD_MODELS": "INFERENCE_PRELOAD_MODELS",
        "MEMORY_FREE_THRESHOLD": "INFERENCE_MEMORY_FREE_THRESHOLD",
        "API_KEY": "ROBOFLOW_API_KEY",
        "LEGACY_MMP_LOAD_WAIT_S": "INFERENCE_LEGACY_LOAD_TIMEOUT_S",
        "LEGACY_MMP_INFER_TIMEOUT_S": "INFERENCE_INFER_TIMEOUT_S",
    }
    assert LEGACY_FLAG_ALIASES == {"RUNS_ON_JETSON": "RUNNING_ON_JETSON"}
    assert LEGACY_DEFAULTS == {
        "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES": "True",
        "MAX_IMAGE_URL_REDIRECTS": "30",
        "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM": "True",
    }
    assert DERIVED_URL_DEFAULTS == {"API_BASE_URL": "api", "BUILDER_ORIGIN": "app"}


@pytest.mark.parametrize(
    "region,project,api_base_url,builder_origin",
    [
        (None, None, "https://api.roboflow.com", "https://app.roboflow.com"),
        (
            None,
            "roboflow-staging",
            "https://api.roboflow.one",
            "https://app.roboflow.one",
        ),
        ("eu", None, "https://api.roboflow.eu", "https://app.roboflow.eu"),
        (
            "eu",
            "roboflow-staging",
            "https://api.roboflow-eu.one",
            "https://app.roboflow-eu.one",
        ),
    ],
)
def test_derived_urls_follow_region_and_project(
    clean_env, monkeypatch, region, project, api_base_url, builder_origin
):
    if region is not None:
        monkeypatch.setenv("ROBOFLOW_REGION", region)
    if project is not None:
        monkeypatch.setenv("PROJECT", project)

    apply_legacy_env()

    assert os.environ["API_BASE_URL"] == api_base_url
    assert os.environ["BUILDER_ORIGIN"] == builder_origin


def test_roboflow_environment_overrides_project_for_derived_urls(
    clean_env, monkeypatch
):
    monkeypatch.setenv("PROJECT", "roboflow-staging")
    monkeypatch.setenv("ROBOFLOW_ENVIRONMENT", "prod")

    apply_legacy_env()

    assert os.environ["API_BASE_URL"] == "https://api.roboflow.com"
    assert os.environ["ROBOFLOW_ENVIRONMENT"] == "prod"


def test_unknown_region_falls_back_to_us(clean_env, monkeypatch):
    monkeypatch.setenv("ROBOFLOW_REGION", "mars")

    apply_legacy_env()

    assert os.environ["API_BASE_URL"] == "https://api.roboflow.com"


def test_explicit_api_base_url_wins_over_region(clean_env, monkeypatch):
    monkeypatch.setenv("ROBOFLOW_REGION", "eu")
    monkeypatch.setenv("API_BASE_URL", "https://api.example.test")

    apply_legacy_env()

    assert os.environ["API_BASE_URL"] == "https://api.example.test"
    assert os.environ["BUILDER_ORIGIN"] == "https://app.roboflow.eu"


def test_staging_project_sets_roboflow_environment(clean_env, monkeypatch):
    monkeypatch.setenv("PROJECT", "roboflow-staging")

    apply_legacy_env()

    assert os.environ["ROBOFLOW_ENVIRONMENT"] == "staging"


def test_platform_project_sets_prod_roboflow_environment(clean_env, monkeypatch):
    monkeypatch.setenv("PROJECT", "roboflow-platform")

    apply_legacy_env()

    assert os.environ["ROBOFLOW_ENVIRONMENT"] == "prod"


def test_unset_project_leaves_roboflow_environment_alone(clean_env):
    apply_legacy_env()

    assert "ROBOFLOW_ENVIRONMENT" not in os.environ


def test_api_base_url_reaches_roboflow_api_host_when_unset(clean_env, monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://api.example.test")

    apply_legacy_env()

    assert os.environ["ROBOFLOW_API_HOST"] == "https://api.example.test"


def test_derived_api_base_url_reaches_roboflow_api_host(clean_env, monkeypatch):
    monkeypatch.setenv("ROBOFLOW_REGION", "eu")

    apply_legacy_env()

    assert os.environ["ROBOFLOW_API_HOST"] == "https://api.roboflow.eu"


def test_derived_roboflow_api_host_logs_no_warning(clean_env, caplog):
    with caplog.at_level(logging.WARNING, logger="inference_server.legacy_env"):
        apply_legacy_env()

    assert os.environ["ROBOFLOW_API_HOST"] == "https://api.roboflow.com"
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_explicit_roboflow_api_host_wins_over_api_base_url(clean_env, monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("ROBOFLOW_API_HOST", "https://weights.example.test")

    apply_legacy_env()

    assert os.environ["ROBOFLOW_API_HOST"] == "https://weights.example.test"


@pytest.mark.parametrize("legacy_value", ["true", "True", "1", "yes", "y", "t"])
def test_true_flag_alias_sets_literal_true(clean_env, monkeypatch, legacy_value):
    monkeypatch.setenv("RUNS_ON_JETSON", legacy_value)

    apply_legacy_env()

    assert os.environ["RUNNING_ON_JETSON"] == "True"


@pytest.mark.parametrize("legacy_value", ["False", "false", "0", "no", "", "maybe"])
def test_false_flag_alias_leaves_new_name_unset(clean_env, monkeypatch, legacy_value):
    monkeypatch.setenv("RUNS_ON_JETSON", legacy_value)

    apply_legacy_env()

    assert "RUNNING_ON_JETSON" not in os.environ


def test_flag_alias_does_not_override_explicit_new_name(clean_env, monkeypatch):
    monkeypatch.setenv("RUNS_ON_JETSON", "true")
    monkeypatch.setenv("RUNNING_ON_JETSON", "False")

    apply_legacy_env()

    assert os.environ["RUNNING_ON_JETSON"] == "False"


@pytest.mark.parametrize("legacy_name,new_name", list(LEGACY_ENV_ALIASES.items()))
def test_alias_applied_when_only_legacy_name_is_set(
    clean_env, monkeypatch, legacy_name, new_name
):
    monkeypatch.setenv(legacy_name, "legacy-value")

    apply_legacy_env()

    assert os.environ[new_name] == "legacy-value"


@pytest.mark.parametrize("legacy_name,new_name", list(LEGACY_ENV_ALIASES.items()))
def test_new_name_wins_when_both_are_set(clean_env, monkeypatch, legacy_name, new_name):
    monkeypatch.setenv(legacy_name, "legacy-value")
    monkeypatch.setenv(new_name, "new-value")

    apply_legacy_env()

    assert os.environ[new_name] == "new-value"


@pytest.mark.parametrize("name,value", list(LEGACY_DEFAULTS.items()))
def test_default_applied_when_unset(clean_env, name, value):
    apply_legacy_env()

    assert os.environ[name] == value


@pytest.mark.parametrize("name", list(LEGACY_DEFAULTS))
def test_default_does_not_override_explicit_value(clean_env, monkeypatch, name):
    monkeypatch.setenv(name, "explicit")

    apply_legacy_env()

    assert os.environ[name] == "explicit"


def test_warning_logged_per_mapped_alias(clean_env, monkeypatch, caplog):
    for legacy_name in LEGACY_ENV_ALIASES:
        monkeypatch.setenv(legacy_name, "legacy-value")

    with caplog.at_level(logging.WARNING, logger="inference_server.legacy_env"):
        apply_legacy_env()

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    for legacy_name, new_name in LEGACY_ENV_ALIASES.items():
        assert f"Legacy env {legacy_name} mapped to {new_name}" in messages


def test_no_warning_when_only_new_name_is_set(clean_env, monkeypatch, caplog):
    for new_name in LEGACY_ENV_ALIASES.values():
        monkeypatch.setenv(new_name, "new-value")

    with caplog.at_level(logging.WARNING, logger="inference_server.legacy_env"):
        apply_legacy_env()

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_idempotent(clean_env, monkeypatch):
    for legacy_name in LEGACY_ENV_ALIASES:
        monkeypatch.setenv(legacy_name, "legacy-value")
    monkeypatch.setenv("RUNS_ON_JETSON", "true")
    monkeypatch.setenv("PROJECT", "roboflow-staging")

    apply_legacy_env()
    after_first_call = dict(os.environ)
    apply_legacy_env()

    assert dict(os.environ) == after_first_call


def _imports_configured_package(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        names = [node.module or ""]
    else:
        return False
    return any(
        name == package or name.startswith(package + ".")
        for name in names
        for package in _CONFIGURED_PACKAGES
        if name != "inference_server.legacy_env"
    )


def _is_apply_legacy_env_call(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "apply_legacy_env"
    )


@pytest.mark.parametrize("module_name", _CALL_POINTS)
def test_call_point_applies_legacy_env_before_configured_imports(module_name):
    tree = ast.parse((_PKG_ROOT / module_name).read_text())

    applied = False
    for node in tree.body:
        if _is_apply_legacy_env_call(node):
            applied = True
        if _imports_configured_package(node):
            assert applied, f"{module_name}: import before apply_legacy_env()"
    assert applied, f"{module_name}: apply_legacy_env() never called"
