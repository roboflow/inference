import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = (
    Path(__file__).resolve().parents[2] / "scripts" / "streamvision_isolation_probe.py"
)


def _load_probe():
    spec = importlib.util.spec_from_file_location("streamvision_isolation_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


class _FakeChild:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


@pytest.mark.slow
def test_streamvision_wheel_works_in_isolation() -> None:
    wheel = os.environ.get("STREAMVISION_ISOLATION_WHEEL")
    if not wheel:
        pytest.skip("STREAMVISION_ISOLATION_WHEEL not set (expected path to a wheel)")
    assert Path(wheel).is_file(), f"wheel not found: {wheel}"

    # The probe merges STREAMVISION_ISOLATION_FIND_LINKS into --find-links itself.
    result = subprocess.run(
        [sys.executable, str(PROBE), "--wheel", wheel, "--webrtc"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "[]\n",
        '[{"check": "distribution_absent", "status": "ok"}]\n',
        "not json at all\n",
        '[{"check": "distribution_absent"}]\n',
        '{"check": "distribution_absent", "status": "ok"}\n',
    ],
)
def test_run_probe_never_reads_missing_results_as_success(
    tmp_path, monkeypatch, stdout
) -> None:
    probe = _load_probe()
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: _FakeChild(stdout))

    results = probe.run_probe(Path(sys.executable), tmp_path, webrtc=False)

    assert [result for result in results if result["status"] == "failed"], results


def test_run_probe_fails_a_complete_result_set_with_nonzero_exit(
    tmp_path, monkeypatch
) -> None:
    probe = _load_probe()
    complete = [
        {"check": name, "status": "ok"} for name in probe.expected_checks(webrtc=False)
    ]
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *a, **k: _FakeChild(json.dumps(complete) + "\n", returncode=1),
    )

    results = probe.run_probe(Path(sys.executable), tmp_path, webrtc=False)

    assert results[-1]["check"] == "child_process"
    assert results[-1]["status"] == "failed"


@pytest.mark.parametrize("webrtc", [False, True])
def test_run_probe_accepts_a_complete_result_set(tmp_path, monkeypatch, webrtc) -> None:
    probe = _load_probe()
    complete = [
        {"check": name, "status": "ok"} for name in probe.expected_checks(webrtc=webrtc)
    ]
    complete[1] = {"check": "import_everything", "status": "skipped", "detail": "x"}
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *a, **k: _FakeChild(json.dumps(complete) + "\n"),
    )

    results = probe.run_probe(Path(sys.executable), tmp_path, webrtc=webrtc)

    assert results == complete
    assert ("webrtc" in [result["check"] for result in results]) is webrtc


def test_run_probe_accepts_a_skipped_manager_roundtrip_without_webrtc(
    tmp_path, monkeypatch
) -> None:
    probe = _load_probe()
    expected = probe.expected_checks(webrtc=False)
    complete = [{"check": name, "status": "ok"} for name in expected]
    complete[expected.index("manager_roundtrip")] = {
        "check": "manager_roundtrip",
        "status": "skipped",
        "detail": "needs --webrtc",
    }
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *a, **k: _FakeChild(json.dumps(complete) + "\n"),
    )

    results = probe.run_probe(Path(sys.executable), tmp_path, webrtc=False)

    assert results == complete
    assert not any(result["status"] == "failed" for result in results)


def test_run_probe_rejects_unexpected_checks(tmp_path, monkeypatch) -> None:
    probe = _load_probe()
    complete = [
        {"check": name, "status": "ok"} for name in probe.expected_checks(webrtc=False)
    ]
    complete.append({"check": "webrtc", "status": "ok"})
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *a, **k: _FakeChild(json.dumps(complete) + "\n"),
    )

    results = probe.run_probe(Path(sys.executable), tmp_path, webrtc=False)

    assert results[-1]["check"] == "results_complete"
    assert results[-1]["status"] == "failed"


def test_run_probe_writes_the_fake_host_and_blocker(tmp_path, monkeypatch) -> None:
    probe = _load_probe()
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: _FakeChild(""))

    probe.run_probe(Path(sys.executable), tmp_path, webrtc=False)

    assert (tmp_path / "host" / "probe_fake_host.py").read_text() == probe.FAKE_HOST
    assert (tmp_path / "host" / "sitecustomize.py").read_text() == probe.BLOCKER


def test_main_exits_one_when_the_child_returns_nothing(
    tmp_path, monkeypatch, capsys
) -> None:
    probe = _load_probe()
    wheel = tmp_path / "streamvision-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(probe, "build_venv", lambda *a, **k: Path(sys.executable))
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: _FakeChild(""))
    monkeypatch.setattr(probe.tempfile, "mkdtemp", lambda **kwargs: str(scratch))
    monkeypatch.setattr(sys, "argv", ["probe", "--wheel", str(wheel)])

    assert probe.main() == 1
    assert "child_process  failed" in capsys.readouterr().out


def test_main_merges_find_links_from_the_environment(tmp_path, monkeypatch) -> None:
    probe = _load_probe()
    wheel = tmp_path / "streamvision-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"")
    received = {}

    def _build_venv(venv_dir, wheel_path, *, find_links, python, webrtc):
        received.update(find_links=find_links, webrtc=webrtc)
        return Path(sys.executable)

    monkeypatch.setattr(probe, "build_venv", _build_venv)
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: _FakeChild(""))
    monkeypatch.setenv("STREAMVISION_ISOLATION_FIND_LINKS", os.pathsep.join("ab"))
    monkeypatch.setattr(
        sys, "argv", ["probe", "--wheel", str(wheel), "--find-links", "c", "--webrtc"]
    )

    probe.main()

    assert received["webrtc"] is True
    assert [Path(link).name for link in received["find_links"]] == ["c", "a", "b"]
    assert all(Path(link).is_absolute() for link in received["find_links"])


@pytest.mark.parametrize(
    "module_name, missing_module, webrtc, expected",
    [
        ("streamvision.stream_manager.manager_app.webrtc", "aiortc", False, "skipped"),
        (
            "streamvision.stream_manager.manager_app.inference_pipeline_manager",
            "av",
            False,
            "skipped",
        ),
        ("streamvision.stream_manager.manager_app.app", "aiortc", False, "skipped"),
        # webrtc installed: nothing should be reported as skipped anymore.
        ("streamvision.stream_manager.manager_app.webrtc", "aiortc", True, "failed"),
        # not on the allowlist, even though the missing module is a webrtc one.
        ("streamvision.camera.video_source", "aiortc", False, "failed"),
        # allowlisted module, but the missing dependency isn't a webrtc one.
        ("streamvision.stream_manager.manager_app.webrtc", "numpy", False, "failed"),
    ],
)
def test_classify_import_error(module_name, missing_module, webrtc, expected) -> None:
    probe = _load_probe()

    assert probe.classify_import_error(module_name, missing_module, webrtc) == expected


@pytest.mark.parametrize(
    "variable",
    ["PYTHONPATH", "PYTHONHOME", "STREAM_MANAGER_PORT", "STREAM_MANAGER_HOST"],
)
def test_child_env_drops_inherited_paths_and_manager_address(
    monkeypatch, variable
) -> None:
    probe = _load_probe()
    monkeypatch.setenv(variable, "/some/checkout")

    env = probe._child_env()

    assert variable not in env
    assert env["PYTHONNOUSERSITE"] == "1"


def test_child_refuses_to_run_with_assertions_stripped(tmp_path) -> None:
    probe = _load_probe()

    proc = subprocess.run(
        [sys.executable, "-c", probe.CHILD, str(tmp_path), str(tmp_path), "off", "[]"],
        capture_output=True,
        text=True,
        env={"PATH": "", "PYTHONOPTIMIZE": "1"},
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1])[0]["status"] == "failed"


def test_blocker_refuses_the_server_package(tmp_path) -> None:
    probe = _load_probe()
    code = (
        probe.BLOCKER
        + "try:\n    import inference\nexcept ModuleNotFoundError as e:\n"
        + "    print(e.name)\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path
    )

    assert proc.stdout.strip() == "inference", proc.stderr


def test_build_venv_runs_pip_outside_the_checkout(tmp_path, monkeypatch) -> None:
    probe = _load_probe()
    calls = []
    monkeypatch.setattr(
        probe.subprocess, "run", lambda argv, **kwargs: calls.append(kwargs)
    )
    monkeypatch.setenv("PYTHONPATH", "/some/checkout")
    (tmp_path / "links").mkdir()

    probe.build_venv(
        tmp_path / "venv",
        tmp_path / "streamvision-0.0.0-py3-none-any.whl",
        find_links=[str(tmp_path / "links")],
        python=sys.executable,
        webrtc=False,
    )

    assert len(calls) == 3
    assert all("PYTHONPATH" not in call["env"] for call in calls)
    assert all(call["cwd"] == tmp_path for call in calls)
