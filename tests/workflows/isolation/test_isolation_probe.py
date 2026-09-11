import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parents[3] / "scripts" / "workflows_isolation_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("workflows_isolation_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeChild:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="decontamination in progress")
def test_workflows_module_works_in_isolation(bundled_fonts) -> None:
    # `bundled_fonts` (tests/workflows/conftest.py) provisions the assets the
    # probe copies; the probe itself never downloads anything.
    result = subprocess.run(
        [sys.executable, str(PROBE), "--tensor-mode", "both"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "stdout",
    [
        "",  # the child printed nothing at all
        "[]\n",  # the child printed an empty result set
        '[{"check": "import_everything", "status": "ok"}]\n',  # only one check ran
        "not json at all\n",  # the results line is malformed
        '[{"check": "import_everything"}]\n',  # a row without a status
    ],
)
def test_run_probe_never_reads_missing_results_as_success(
    tmp_path, monkeypatch, stdout
) -> None:
    # A child that dies, is truncated or garbles its output must fail the probe.
    # Without this, empty stdout parsed as `[]` and `main()` returned 0.
    probe = _load_probe()
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: _FakeChild(stdout))
    results = probe.run_probe(tmp_path, tensor_mode=False)
    assert results, "an unusable child must still produce a reported failure"
    assert [r for r in results if r["status"] != "ok"], results


def test_run_probe_accepts_a_complete_result_set(tmp_path, monkeypatch) -> None:
    # The guard above must not block the run that finally passes.
    probe = _load_probe()
    complete = [{"check": name, "status": "ok"} for name in probe.EXPECTED_CHECKS]
    monkeypatch.setattr(
        probe.subprocess, "run", lambda *a, **k: _FakeChild(json.dumps(complete) + "\n")
    )
    assert probe.run_probe(tmp_path, tensor_mode=False) == complete


def test_main_reports_failure_when_the_child_returns_nothing(monkeypatch) -> None:
    probe = _load_probe()
    monkeypatch.setattr(probe, "build_tree", lambda target: None)
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: _FakeChild(""))
    monkeypatch.setattr(
        sys, "argv", ["workflows_isolation_probe.py", "--tensor-mode", "off"]
    )
    assert probe.main() == 1


@pytest.mark.parametrize(
    "variable, value",
    [
        ("WORKFLOWS_PLUGINS", "inference.enterprise.workflows.enterprise_blocks"),
        ("DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER", "extended_roboflow_errors"),
        # Since Phase 5 these six reach the child through a WorkflowsConfiguration
        # the child installs itself. An inherited value must not reach it.
        ("ENABLE_TENSOR_DATA_REPRESENTATION", "True"),
        ("WORKFLOWS_IMAGE_TENSOR_DEVICE", "cuda"),
        ("ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", "False"),
        ("WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"),
        ("ALLOW_WORKFLOWS_FONTS_DOWNLOAD", "True"),
        ("MODEL_CACHE_DIR", "/somewhere/else"),
    ],
)
def test_child_env_drops_server_only_configuration(
    tmp_path, monkeypatch, variable, value
) -> None:
    # Server settings the child must not inherit: the plugin list would load
    # enterprise blocks, the step-error handler names a handler the standalone
    # engine does not register, and the rest are configuration the child now
    # installs explicitly.
    probe = _load_probe()
    monkeypatch.setenv(variable, value)
    env = probe._child_env(tmp_path, tensor_mode=False)
    assert variable not in env
    assert "WORKFLOWS_PLUGINS" not in env
    assert "DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER" not in env
    assert "ENABLE_TENSOR_DATA_REPRESENTATION" not in env
    assert "MODEL_CACHE_DIR" not in env


def test_child_refuses_to_run_with_assertions_stripped(tmp_path, monkeypatch) -> None:
    # PYTHONOPTIMIZE strips every `assert` the checks verify with. The parent
    # pins it to "0"; this proves the child's own guard fires if it ever leaks.
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
