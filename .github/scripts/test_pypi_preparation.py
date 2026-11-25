"""Exercise release preparation with real uv resolution and isolated wheel builds."""

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".release/pypi"))
sys.path.insert(0, str(ROOT / ".github/scripts"))
import candidate
import dependencies
import prepare
from test_pypi_dependencies import _package, _PyPI, _runtime, _wheel
from test_pypi_dependencies import dependencies as planner


def _project(name, *, version="2.0rc1", requirements=()):
    directory = Path(dependencies.PROJECTS[name][0])
    directory.mkdir(exist_ok=True)
    (directory / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "{version}"\n'
        'requires-python = ">=3.10,<3.14"\n'
        f"dependencies = {json.dumps(list(requirements))}\n"
        '[build-system]\nrequires = ["setuptools>=83.0.0", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n'
    )
    module = directory / name.replace("-", "_")
    module.mkdir(exist_ok=True)
    (module / "__init__.py").write_text('VALUE = "prepared candidate"\n')
    return directory


def _publish(wheelhouse, package):
    name, data = _wheel(package)
    (wheelhouse / name).write_bytes(data)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in dependencies.PROJECTS:
        _project(name)
    wheelhouse = tmp_path / "index"
    wheelhouse.mkdir()
    for wheel in Path("/build-wheels").glob("*.whl"):
        (wheelhouse / wheel.name).symlink_to(wheel)
    for name, value in {
        "SOURCE_REVISION": "a" * 40,
        "GITHUB_REPOSITORY": "roboflow/inference",
        "GITHUB_RUN_ID": "100",
        "GITHUB_RUN_ATTEMPT": "1",
        "UV_NO_INDEX": "1",
        "UV_OFFLINE": "1",
        "UV_FIND_LINKS": str(wheelhouse),
        "UV_CACHE_DIR": str(tmp_path / "cache"),
        "UV_PYTHON_DOWNLOADS": "never",
        "PIP_NO_INDEX": "1",
        "PIP_FIND_LINKS": "/build-wheels",
    }.items():
        monkeypatch.setenv(name, value)
    shutil.copyfile(ROOT / "Makefile", "Makefile")
    Path(".release/pypi").mkdir(parents=True)
    shutil.copyfile(
        ROOT / ".release/pypi/build-requirements.txt",
        ".release/pypi/build-requirements.txt",
    )
    return wheelhouse


def _lock(directory):
    return tomllib.loads((directory / "uv.lock").read_text())


def _build(profile):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows" / dependencies.PROJECTS[profile][1]).read_text()
    )
    step = next(
        step
        for step in workflow["jobs"]["build"]["steps"]
        if "uv build" in step.get("run", "") or "make create_" in step.get("run", "")
    )
    # Font provisioning is unrelated to lock resolution; use the actual build
    # recipe and its isolated, constrained backend without downloading fonts.
    command = step["run"].replace(
        "make create_", "make --old-file=download_fonts create_"
    )
    subprocess.run(
        ["bash", "-ec", command],
        cwd=step.get("working-directory", "."),
        check=True,
        capture_output=True,
        text=True,
    )
    directory = Path(candidate.DIRECTORIES[profile])
    wheel = next(directory.glob("*.whl"))
    inventory = dependencies._wheel(wheel.read_bytes(), filename=wheel.name)
    assert inventory["name"] == profile
    assert inventory["version"] == dependencies._catalog()[profile]["version"]
    assert dependencies._edges(inventory) == dependencies._edges(
        dependencies._catalog()[profile]
    )
    return directory


@pytest.mark.parametrize("profile", dependencies.PROJECTS)
@pytest.mark.parametrize(
    "change", ["first-lock", "version", "dependency", "rc-to-final"]
)
def test_preparation_repairs_inputs_before_real_publisher_build(
    workspace, profile, change
):
    _publish(workspace, _package("inference-sdk", "1.0rc1"))
    directory = _project(profile, requirements=["inference-sdk>=1.0rc1"])
    if change != "first-lock":
        subprocess.run(["uv", "lock"], cwd=directory, check=True, capture_output=True)
    original = (directory / "uv.lock").read_bytes() if change != "first-lock" else None
    if change == "version":
        _project(profile, version="2.0", requirements=["inference-sdk>=1.0rc1"])
    if change == "dependency":
        _publish(workspace, _package("inference-sdk", "1.1rc1"))
        _project(profile, requirements=["inference-sdk>=1.1rc1"])
    if change == "rc-to-final":
        # The old RC still satisfies the manifest. A plain `uv lock` leaves it
        # locked, so targeted first-party upgrades are required.
        _publish(workspace, _package("inference-sdk", "1.0"))

    prepare._prepare(profile)
    prepared = (directory / "uv.lock").read_bytes()
    assert prepared != original
    sdk = next(p for p in _lock(directory)["package"] if p["name"] == "inference-sdk")
    assert (
        sdk["version"]
        == {
            "first-lock": "1.0rc1",
            "version": "1.0rc1",
            "dependency": "1.1rc1",
            "rc-to-final": "1.0",
        }[change]
    )
    receipt = json.loads(prepare._paths(profile)[2].read_text())
    assert receipt["original_lock_sha256"] == (
        prepare._sha256(original) if original is not None else None
    )
    _build(profile)
    assert (directory / "uv.lock").read_bytes() == prepared
    candidate._seal(profile)
    candidate._verify(profile, restored=True)


@pytest.mark.parametrize("existing_lock", [False, True])
def test_unpublished_dependency_fails_without_sealing_and_can_retry(
    workspace, existing_lock
):
    profile = "roboflow-workflows"
    directory = Path("workflows")
    if existing_lock:
        subprocess.run(["uv", "lock"], cwd=directory, check=True, capture_output=True)
    original = (directory / "uv.lock").read_bytes() if existing_lock else None
    _project(profile, requirements=["inference-models==2.0rc1"])

    with pytest.raises(subprocess.CalledProcessError):
        prepare._prepare(profile)
    assert not prepare._paths(profile)[2].exists()
    assert not Path("dist").exists()
    assert not Path("release-candidate").exists()
    assert ((directory / "uv.lock").read_bytes() if existing_lock else None) == original
    if not existing_lock:
        assert not (directory / "uv.lock").exists()

    _publish(workspace, _package("inference-models", "2.0rc1"))
    prepare._prepare(profile)
    _build(profile)
    candidate._seal(profile)


@pytest.mark.parametrize("profile", dependencies.PROJECTS)
def test_retry_restores_lock_source_and_identical_built_bytes(
    workspace, monkeypatch, profile
):
    prepare._prepare(profile)
    directory = _build(profile)
    candidate._seal(profile)
    bundle = Path("release-candidate") / profile
    before = candidate._files(bundle)
    inputs = {path: path.read_bytes() for path in prepare._paths(profile)}
    output = candidate._files(directory)
    shutil.rmtree(directory)
    prepare._paths(profile)[1].write_bytes(b"original checkout stale lock")
    prepare._paths(profile)[2].unlink()
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")

    def no_subprocess(*args, **kwargs):
        pytest.fail("A retained retry must not resolve or rebuild")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    candidate._restore(profile)
    candidate._verify(profile, restored=True)
    assert candidate._files(bundle) == before
    assert candidate._files(directory) == output
    assert {path: path.read_bytes() for path in inputs} == inputs
    with pytest.raises(ValueError, match="Retained candidate"):
        prepare._prepare(profile)


@pytest.mark.parametrize(
    "fault", ["lock", "receipt", "metadata", "missing-inputs", "source"]
)
def test_prepared_input_changes_block_restore(workspace, monkeypatch, fault):
    profile = "streamvision"
    prepare._prepare(profile)
    directory = _build(profile)
    candidate._seal(profile)
    shutil.rmtree(directory)
    project, lock, receipt = prepare._paths(profile)
    bundle = Path("release-candidate") / profile
    if fault in {"lock", "receipt"}:
        (bundle / "inputs" / (lock if fault == "lock" else receipt)).write_text(
            "changed"
        )
    elif fault == "metadata":
        project.write_text('name = "changed"')
    elif fault == "missing-inputs":
        (bundle / "inputs" / lock).unlink()
    else:
        monkeypatch.setenv("SOURCE_REVISION", "b" * 40)
    with pytest.raises(ValueError):
        candidate._restore(profile)
    assert not directory.exists()


@pytest.mark.parametrize("fail_preparation", [False, True])
def test_coordinator_prepares_only_after_upstream_publication(
    workspace, monkeypatch, fail_preparation
):
    sdk = _package("inference-sdk", "2.0rc1")
    models = _package("inference-models", "2.0rc1")
    workflows = _package(
        "roboflow-workflows",
        "2.0rc1",
        ["inference-models==2.0rc1", "inference-sdk==2.0rc1"],
    )
    stream = _package("streamvision", "2.0rc1", ["roboflow-workflows==2.0rc1"])
    catalog = {p["name"]: p for p in (sdk, models, workflows, stream)}
    for name in dependencies.PROJECTS:
        _project(name, requirements=catalog[name]["requires_dist"])
    api = _PyPI(monkeypatch, [sdk])
    _publish(workspace, sdk)
    plan = planner._plan(
        _runtime("streamvision==2.0rc1"),
        catalog=catalog,
        registry=planner._Registry(wait_seconds=120),
    )
    events = []

    def dispatch(name, **kwargs):
        events.append(name)
        if fail_preparation and name == "roboflow-workflows":
            # Model an unavailable index file after the publisher was dispatched.
            next(workspace.glob("inference_models-*.whl")).unlink()
        prepare._prepare(name)  # Real resolver cannot see unpublished fixtures.
        output = _build(name)
        candidate._seal(name)
        api.add(catalog[name])
        _publish(workspace, catalog[name])
        shutil.rmtree(output)  # Independent publisher jobs have separate dist/.
        return len(events)

    monkeypatch.setattr(planner, "_dispatch", dispatch)
    if fail_preparation:
        with pytest.raises(subprocess.CalledProcessError):
            planner._coordinate(plan, wait_seconds=120)
        assert events == ["inference-models", "roboflow-workflows"]
        assert not Path("release-candidate/roboflow-workflows").exists()
        assert not Path("stream_vision/uv.lock").exists()
    else:
        planner._coordinate(plan, wait_seconds=120)
        assert events == ["inference-models", "roboflow-workflows", "streamvision"]
        assert set(plan["packages"]) == set(catalog)
        for name in events:
            assert plan["packages"][name]["publisher_run_id"]
            assert plan["packages"][name]["sha256"]
