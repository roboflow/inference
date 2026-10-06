"""Tests for the standalone isolation probe.

Unit tests exercise the pure helpers (`_child_env`, `_validate_results`,
`run_probe` with a fake child). The end-to-end run is opt-in via the
`WORKFLOWS_ISOLATION_WHEEL` environment variable pointing at a built
`roboflow-workflows` wheel — CI sets this in the standalone job after
building the wheel; the developer flow builds one on demand via the root
`scripts/workflows_isolation_probe.py` launcher.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parents[2] / "scripts" / "workflows_isolation_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("workflows_isolation_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeChild:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


@pytest.mark.slow
def test_workflows_wheel_works_in_isolation(tmp_path) -> None:
    wheel = os.environ.get("WORKFLOWS_ISOLATION_WHEEL")
    if not wheel:
        pytest.skip("WORKFLOWS_ISOLATION_WHEEL not set (expected path to built wheel)")
    assert Path(wheel).is_file(), f"wheel not found: {wheel}"
    argv = [sys.executable, str(PROBE), "--wheel", wheel, "--tensor-mode", "both"]
    # `WORKFLOWS_ISOLATION_FIND_LINKS` is pathsep-separated; the probe merges
    # it into --find-links so locally built SDK/models wheels resolve without
    # a repository-hosted index.
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "[]\n",
        '[{"check": "import_everything", "status": "ok"}]\n',
        "not json at all\n",
        '[{"check": "import_everything"}]\n',
    ],
)
def test_run_probe_never_reads_missing_results_as_success(
    tmp_path, monkeypatch, stdout
) -> None:
    probe = _load_probe()
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: _FakeChild(stdout))
    results = probe.run_probe(Path(sys.executable), tmp_path, tensor_mode=False)
    assert results, "an unusable child must still produce a reported failure"
    assert [r for r in results if r["status"] != "ok"], results


def test_run_probe_accepts_a_complete_result_set(tmp_path, monkeypatch) -> None:
    probe = _load_probe()
    complete = [{"check": name, "status": "ok"} for name in probe.EXPECTED_CHECKS]
    monkeypatch.setattr(
        probe.subprocess, "run", lambda *a, **k: _FakeChild(json.dumps(complete) + "\n")
    )
    assert (
        probe.run_probe(Path(sys.executable), tmp_path, tensor_mode=False) == complete
    )


def test_main_reports_failure_when_the_child_returns_nothing(
    tmp_path, monkeypatch, capsys
) -> None:
    """main() must exit 1 when the child yields no usable results.

    Guards the whole pipeline (arg parsing -> build_venv -> run_probe ->
    exit code), not just run_probe. A silent child that produced no results
    was once tolerated as success; keep the end-to-end regression alive.
    """
    probe = _load_probe()
    wheel = tmp_path / "roboflow_workflows-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"")
    monkeypatch.setattr(
        probe,
        "build_venv",
        lambda venv_dir, wheel_path, find_links: Path(sys.executable),
    )
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: _FakeChild(""))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(probe.tempfile, "mkdtemp", lambda **kwargs: str(scratch))
    monkeypatch.setattr(
        sys, "argv", ["probe", "--wheel", str(wheel), "--tensor-mode", "off"]
    )
    assert probe.main() == 1
    captured = capsys.readouterr().out
    assert "child_process: fail" in captured, captured


@pytest.mark.parametrize(
    "variable, value",
    [
        ("DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER", "extended_roboflow_errors"),
        ("ENABLE_TENSOR_DATA_REPRESENTATION", "True"),
        ("WORKFLOWS_IMAGE_TENSOR_DEVICE", "cuda"),
        ("ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", "False"),
        ("WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"),
        ("ALLOW_WORKFLOWS_FONTS_DOWNLOAD", "True"),
        ("MODEL_CACHE_DIR", "/somewhere/else"),
        ("PYTHONPATH", "/some/checkout/inference"),
    ],
)
def test_child_env_drops_server_only_configuration(
    tmp_path, monkeypatch, variable, value
) -> None:
    """The child must not inherit the host's step-error handler,
    tensor/fonts/model-cache configuration, or a PYTHONPATH pointing at a
    source checkout. The child installs its own configuration and resolves
    everything from the isolated venv.
    """
    probe = _load_probe()
    monkeypatch.setenv(variable, value)
    env = probe._child_env(tmp_path)
    assert variable not in env
    for banned in (
        "DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER",
        "ENABLE_TENSOR_DATA_REPRESENTATION",
        "MODEL_CACHE_DIR",
        "PYTHONPATH",
    ):
        assert banned not in env, f"{banned} leaked into child env"


def test_child_env_activates_enterprise_plugin_canonically(
    tmp_path, monkeypatch
) -> None:
    """Regardless of what the host had set, the child runs with
    `WORKFLOWS_PLUGINS=roboflow_workflows.enterprise_blocks.loader` so
    `load_workflow_blocks()` picks up the enterprise plugin."""
    probe = _load_probe()
    monkeypatch.setenv(
        "WORKFLOWS_PLUGINS", "some.other.plugin,roboflow_workflows.enterprise_blocks"
    )
    env = probe._child_env(tmp_path)
    assert env["WORKFLOWS_PLUGINS"] == "roboflow_workflows.enterprise_blocks.loader"


def test_child_refuses_to_run_with_assertions_stripped(tmp_path) -> None:
    probe = _load_probe()
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            probe.CHILD,
            str(tmp_path),
            "off",
            json.dumps(list(probe.EXPECTED_CHECKS)),
        ],
        capture_output=True,
        text=True,
        env={**dict(PATH=""), "PYTHONOPTIMIZE": "1"},
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == [
        {
            "check": "no_optimize",
            "status": "fail",
            "detail": "child ran with optimization enabled; assert statements are stripped",
        }
    ]
