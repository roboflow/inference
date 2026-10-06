import ast
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_SERVER_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _SERVER_ROOT.parent
_ENTRYPOINTS_ROOT = _SERVER_ROOT / "inference_server" / "entrypoints"
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "docker.inference_server.yml"
_ENTRY_MODULES = ("cpu_http", "gpu_http")
_DOCKERFILE_ENTRY_MODULES = {
    "Dockerfile.cpu": "cpu_http",
    "Dockerfile.gpu": "gpu_http",
    "Dockerfile.jetson.5.1.1": "gpu_http",
    "Dockerfile.jetson.6.0.0": "gpu_http",
    "Dockerfile.jetson.6.2.0": "gpu_http",
}
_SCRIPT_PATH = "/usr/local/bin/run_uvicorn.sh"


def _instructions(dockerfile_name: str) -> list:
    text = (_SERVER_ROOT / "docker" / dockerfile_name).read_text()
    joined = text.replace("\\\n", " ")
    instructions = [
        line.strip()
        for line in joined.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    return instructions


def _is_apply_legacy_env_call(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "apply_legacy_env"
    )


def _imports_inference_server(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        names = [node.module or ""]
    else:
        return False
    return any(
        name == "inference_server" or name.startswith("inference_server.")
        for name in names
        if name != "inference_server.legacy_env"
    )


@pytest.mark.parametrize("module_name", _ENTRY_MODULES)
def test_entry_module_exposes_the_application(module_name):
    import inference_server.app as app_mod

    entry_module = importlib.import_module(
        f"inference_server.entrypoints.{module_name}"
    )
    entry_module = importlib.reload(entry_module)

    assert entry_module.app is app_mod.app


@pytest.mark.parametrize("module_name", _ENTRY_MODULES)
def test_entry_module_applies_legacy_env_before_other_imports(module_name):
    tree = ast.parse((_ENTRYPOINTS_ROOT / f"{module_name}.py").read_text())

    applied = False
    for node in tree.body:
        if _is_apply_legacy_env_call(node):
            applied = True
        if _imports_inference_server(node):
            assert applied, f"{module_name}: import before apply_legacy_env()"
    assert applied, f"{module_name}: apply_legacy_env() never called"


def test_main_passes_configured_host_to_uvicorn(monkeypatch):
    import uvicorn

    import inference_server.app as app_mod
    from inference_server import configuration

    calls = []
    monkeypatch.setattr(
        uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(configuration, "HOST", "203.0.113.7")

    app_mod.main()

    assert len(calls) == 1
    assert calls[0][1]["host"] == "203.0.113.7"


def _configured_host(env: dict) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from inference_server import configuration as c; print(c.HOST)",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    host = result.stdout.strip().splitlines()[-1]

    return host


def test_host_defaults_to_all_interfaces():
    env = {name: value for name, value in os.environ.items() if name != "HOST"}

    assert _configured_host(env) == "0.0.0.0"


def test_host_is_read_from_the_environment():
    env = {**os.environ, "HOST": "127.0.0.1"}

    assert _configured_host(env) == "127.0.0.1"


@pytest.mark.parametrize(
    "dockerfile_name,entry_module", sorted(_DOCKERFILE_ENTRY_MODULES.items())
)
def test_dockerfile_starts_through_the_legacy_entry_command(
    dockerfile_name, entry_module
):
    instructions = _instructions(dockerfile_name)

    entrypoints = [line for line in instructions if line.startswith("ENTRYPOINT")]
    assert entrypoints[-1] == f'ENTRYPOINT ["{_SCRIPT_PATH}", "{entry_module}:app"]'
    assert len(entrypoints) == 1
    assert not [line for line in instructions if line.startswith("CMD")]


@pytest.mark.parametrize("dockerfile_name", sorted(_DOCKERFILE_ENTRY_MODULES))
def test_dockerfile_copies_the_script_and_both_entry_modules(dockerfile_name):
    instructions = _instructions(dockerfile_name)
    copies = [line for line in instructions if line.startswith("COPY")]

    assert f"COPY docker/entrypoint/run_uvicorn.sh {_SCRIPT_PATH}" in copies
    assert f"RUN chmod +x {_SCRIPT_PATH}" in instructions
    workdirs = [line for line in instructions if line.startswith("WORKDIR")]
    assert workdirs[-1] == "WORKDIR /app/inference_server"
    for module_name in _ENTRY_MODULES:
        source = f"inference_server/inference_server/entrypoints/{module_name}.py"
        matching = [
            line
            for line in copies
            if source in line.split() and line.split()[-1] == "/app/inference_server/"
        ]
        assert matching, f"{dockerfile_name}: {module_name}.py is not copied"


@pytest.mark.parametrize("dockerfile_name", sorted(_DOCKERFILE_ENTRY_MODULES))
def test_dockerfile_sets_the_script_environment_defaults(dockerfile_name):
    environment = " ".join(
        line for line in _instructions(dockerfile_name) if line.startswith("ENV")
    ).split()

    assert "HOST=0.0.0.0" in environment
    assert "PORT=9001" in environment
    assert "NUM_WORKERS=1" in environment


def test_workflow_has_swap_input_defaulting_to_false():
    workflow = yaml.safe_load(_WORKFLOW_PATH.read_text())
    triggers = workflow["on"] if "on" in workflow else workflow[True]

    swap = triggers["workflow_dispatch"]["inputs"]["swap"]

    assert swap["type"] == "boolean"
    assert swap["default"] is False
    assert "legacy image names" in swap["description"]


@pytest.mark.parametrize("dockerfile_name", sorted(_DOCKERFILE_ENTRY_MODULES))
def test_dockerfile_installs_inference_server_with_workflows_and_otel(
    dockerfile_name,
):
    instructions = _instructions(dockerfile_name)

    pip_install_lines = [
        line
        for line in instructions
        if "uv pip install" in line or "pip install" in line
    ]
    install_lines = []
    for line in pip_install_lines:
        if "inference_server[" in line:
            install_lines.append(line)

    assert install_lines, f"{dockerfile_name}: no inference_server[ install found"

    for line in install_lines:
        match = re.search(r"inference_server\[([^\]]+)\]", line)
        assert (
            match
        ), f"{dockerfile_name}: could not parse inference_server[ extras in {line}"
        extras_str = match.group(1)
        extras = {extra.strip() for extra in extras_str.split(",")}
        assert (
            "workflows" in extras
        ), f"{dockerfile_name}: workflows extra not found in {extras}"
        assert "otel" in extras, f"{dockerfile_name}: otel extra not found in {extras}"
