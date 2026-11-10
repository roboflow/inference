"""Dependency release boundaries; execute inside the release Docker image."""

import copy
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import urllib.error
import zipfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".release/pypi"))
SPEC = importlib.util.spec_from_file_location(
    "pypi_dependencies", ROOT / ".release/pypi/dependencies.py"
)
dependencies = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dependencies)


def _package(name, version, requirements=(), extras=()):
    result = {
        "name": name,
        "version": version,
        "requires_dist": list(requirements),
        "provides_extra": list(extras),
        "requires_python": ">=3.10,<3.14",
    }
    return result


def _wheel(file):
    stem = f"{file['name'].replace('-', '_')}-{file['version']}"
    buffer = io.BytesIO()
    metadata = f"Metadata-Version: 2.4\nName: {file['name']}\nVersion: {file['version']}\nRequires-Python: {file['requires_python']}\n"
    metadata += "".join(f"Requires-Dist: {raw}\n" for raw in file["requires_dist"])
    metadata += "".join(
        f"Provides-Extra: {extra}\n" for extra in file["provides_extra"]
    )
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{file['name'].replace('-', '_')}/__init__.py", "")
        archive.writestr(f"{stem}.dist-info/METADATA", metadata + "\n")
    return f"{stem}-py3-none-any.whl", buffer.getvalue()


class _PyPI:
    def __init__(self, monkeypatch, packages=()):
        self.responses = {}
        self.calls = []
        self.packages = {}
        for file in packages:
            self.add(file)
        monkeypatch.setattr(dependencies.release, "_read_url", self.read)

    def add(self, file):
        name, version = file["name"], file["version"]
        filename, data = _wheel(file)
        url = f"https://files.pythonhosted.org/{filename}"
        remote = {
            "filename": filename,
            "url": url,
            "digests": {"sha256": hashlib.sha256(data).hexdigest()},
            "yanked": False,
        }
        self.responses[url] = data
        self.responses[f"https://pypi.org/pypi/{name}/{version}/json"] = json.dumps(
            {"info": {"name": name, "version": version}, "urls": [remote]}
        ).encode()
        self.packages.setdefault(name, {})[version] = [remote]
        self.responses[f"https://pypi.org/pypi/{name}/json"] = json.dumps(
            {"releases": self.packages[name]}
        ).encode()

    def read(self, url):
        self.calls.append(url)
        value = self.responses.get(url)
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise urllib.error.HTTPError(url, 404, "missing", None, None)
        return value

    def mutate_release(self, name, version, mutate):
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        document = json.loads(self.responses[url])
        mutate(document)
        self.responses[url] = json.dumps(document).encode()


def _registry():
    return dependencies._Registry(wait_seconds=0)


def _runtime(*requirements):
    return [_package("inference", "1.9.0", requirements)]


def test_resolves_actual_transitive_wheel_metadata_in_dependency_order(monkeypatch):
    _PyPI(
        monkeypatch,
        [
            _package("inference-sdk", "1.9.0"),
            _package("inference-models", "0.40.0", extras=["torch-cpu"]),
            _package(
                "roboflow-workflows",
                "0.3.0",
                ["inference-sdk>=1.8", "inference-models>=0.39"],
                ["enterprise"],
            ),
            _package(
                "streamvision",
                "0.2.0",
                ['roboflow-workflows[enterprise]>=0.3; extra == "workflows"'],
                ["workflows"],
            ),
        ],
    )
    plan = dependencies._plan(
        _runtime(
            "inference-sdk==1.9.0",
            "inference-models[torch-cpu]==0.40.0",
            "streamvision[workflows]==0.2.0",
        ),
        catalog={},
        registry=_registry(),
    )
    assert (
        plan["order"].index("inference-models")
        < plan["order"].index("roboflow-workflows")
        < plan["order"].index("streamvision")
    )
    assert plan["order"].index("inference-sdk") < plan["order"].index(
        "roboflow-workflows"
    )
    assert {p["action"] for p in plan["packages"].values()} == {"reuse"}


def test_reuses_compatible_published_version_instead_of_bumping_upstream(monkeypatch):
    _PyPI(monkeypatch, [_package("inference-models", "1.0")])
    plan = dependencies._plan(
        _runtime("inference-models>=1"),
        catalog={"inference-models": _package("inference-models", "2.0")},
        registry=_registry(),
    )
    assert plan["packages"]["inference-models"]["version"] == "1.0"
    assert plan["packages"]["inference-models"]["action"] == "reuse"


def test_selects_missing_exact_same_checkout_version(monkeypatch):
    _PyPI(monkeypatch, [_package("inference-models", "1.0")])
    plan = dependencies._plan(
        _runtime("inference-models==2.0"),
        catalog={"inference-models": _package("inference-models", "2.0")},
        registry=_registry(),
    )
    assert plan["packages"]["inference-models"]["action"] == "publish"


def test_sdk_is_provided_by_parent_not_dispatched_again(monkeypatch):
    _PyPI(monkeypatch)
    plan = dependencies._plan(
        _runtime("inference-sdk==1.9.0rc2"),
        catalog={"inference-sdk": _package("inference-sdk", "1.9.0rc2")},
        registry=_registry(),
    )
    assert plan["packages"]["inference-sdk"]["action"] == "sdk"


def test_backtracks_when_newest_workflows_conflicts_with_exact_sdk(monkeypatch):
    _PyPI(
        monkeypatch,
        [
            _package("inference-sdk", "1.0"),
            _package("roboflow-workflows", "2.0", ["inference-sdk>=2"]),
            _package("roboflow-workflows", "1.0", ["inference-sdk>=1"]),
        ],
    )
    plan = dependencies._plan(
        _runtime("roboflow-workflows>=1", "inference-sdk==1.0"),
        catalog={},
        registry=_registry(),
    )
    assert plan["packages"]["roboflow-workflows"]["version"] == "1.0"


@pytest.mark.parametrize(
    "requirement,local",
    [
        ("inference-models==2", _package("inference-models", "3")),
        ("inference-models[missing-extra]>=1", _package("inference-models", "2")),
        ("inference-models>=2", _package("inference-models", "2.0rc3")),
    ],
)
def test_rejects_unsatisfiable_selection(monkeypatch, requirement, local):
    _PyPI(monkeypatch)
    with pytest.raises(ValueError, match="No compatible"):
        dependencies._plan(
            _runtime(requirement), catalog={local["name"]: local}, registry=_registry()
        )


def test_accepts_rc2_when_requirement_explicitly_allows_prereleases(monkeypatch):
    _PyPI(monkeypatch, [_package("inference-models", "2.0rc2")])
    plan = dependencies._plan(
        _runtime("inference-models>=2.0rc1,<2.1"), catalog={}, registry=_registry()
    )
    assert plan["packages"]["inference-models"]["version"] == "2.0rc2"


def test_optional_dependency_conflict_blocks_plan(monkeypatch):
    _PyPI(
        monkeypatch,
        [
            _package("inference-models", "1"),
            _package(
                "roboflow-workflows",
                "1",
                ['inference-models>=2; extra == "enterprise"'],
            ),
        ],
    )
    with pytest.raises(ValueError, match="No compatible"):
        dependencies._plan(
            _runtime("inference-models==1", "roboflow-workflows==1"),
            catalog={},
            registry=_registry(),
        )


def test_dependency_cycle_is_rejected(monkeypatch):
    _PyPI(
        monkeypatch,
        [
            _package("inference-models", "1", ["roboflow-workflows==1"]),
            _package("roboflow-workflows", "1", ["inference-models==1"]),
        ],
    )
    with pytest.raises(ValueError, match="cycle"):
        dependencies._plan(
            _runtime("roboflow-workflows==1"), catalog={}, registry=_registry()
        )


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_registry_errors_never_authorize_publication(monkeypatch, status):
    api = _PyPI(monkeypatch)
    url = "https://pypi.org/pypi/inference-models/2/json"
    api.responses[url] = urllib.error.HTTPError(url, status, "failure", None, None)
    with pytest.raises((ValueError, urllib.error.HTTPError)):
        dependencies._plan(
            _runtime("inference-models==2"),
            catalog={"inference-models": _package("inference-models", "2")},
            registry=_registry(),
        )


@pytest.mark.parametrize(
    "fault", ["yanked", "missing-wheel", "hash", "identity", "file-404"]
)
def test_existing_corrupt_or_partial_release_is_not_republished(monkeypatch, fault):
    package = _package("inference-models", "2")
    api = _PyPI(monkeypatch, [package])

    def mutate(document):
        if fault == "yanked":
            document["urls"][0]["yanked"] = True
        elif fault == "missing-wheel":
            document["urls"] = []
        elif fault == "hash":
            document["urls"][0]["digests"]["sha256"] = "wrong"
        elif fault == "identity":
            document["info"]["name"] = "different"
        elif fault == "file-404":
            document["urls"][0]["url"] += "missing"

    api.mutate_release("inference-models", "2", mutate)
    with pytest.raises(ValueError):
        dependencies._plan(
            _runtime("inference-models==2"),
            catalog={"inference-models": package},
            registry=_registry(),
        )


def test_direct_url_internal_dependency_is_rejected(monkeypatch):
    _PyPI(monkeypatch)
    with pytest.raises(ValueError, match="Direct URL"):
        dependencies._plan(
            _runtime("inference-models @ https://example.com/model.whl"),
            catalog={},
            registry=_registry(),
        )


def test_coordinator_dispatches_in_order_and_verifies_each_before_next(monkeypatch):
    models = _package("inference-models", "2")
    workflows = _package("roboflow-workflows", "2", ["inference-models==2"])
    api = _PyPI(monkeypatch)
    plan = dependencies._plan(
        _runtime("roboflow-workflows==2"),
        catalog={p["name"]: p for p in [models, workflows]},
        registry=_registry(),
    )
    calls = []

    def dispatch(name, *, version, wait_seconds):
        if name == "roboflow-workflows":
            assert any(
                "files.pythonhosted.org/inference_models" in url for url in api.calls
            )
        calls.append((name, version))
        api.add({models["name"]: models, workflows["name"]: workflows}[name])
        return len(calls)

    monkeypatch.setattr(dependencies, "_dispatch", dispatch)
    assert dependencies._coordinate(plan, wait_seconds=0) == {
        "inference-models": 1,
        "roboflow-workflows": 2,
    }
    assert calls == [("inference-models", "2"), ("roboflow-workflows", "2")]


def test_upstream_failure_blocks_downstream_dispatch(monkeypatch):
    api = _PyPI(monkeypatch)
    catalog = {
        "inference-models": _package("inference-models", "2"),
        "roboflow-workflows": _package(
            "roboflow-workflows", "2", ["inference-models==2"]
        ),
    }
    plan = dependencies._plan(
        _runtime("roboflow-workflows==2"), catalog=catalog, registry=_registry()
    )
    calls = []

    def dispatch(name, **kwargs):
        calls.append(name)
        raise ValueError("publisher failed")

    monkeypatch.setattr(dependencies, "_dispatch", dispatch)
    with pytest.raises(ValueError, match="publisher failed"):
        dependencies._coordinate(plan, wait_seconds=0)
    assert calls == ["inference-models"]
    assert not any("files.pythonhosted.org" in url for url in api.calls)


@pytest.mark.parametrize(
    "fault",
    [
        "failure",
        "cancelled",
        "wrong-sha",
        "wrong-workflow",
        "wrong-event",
        "timeout",
        "moved-ref",
    ],
)
def test_dispatch_run_failure_boundaries(monkeypatch, fault):
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/candidate")
    calls = []

    def github(path, *, payload=None):
        calls.append((path, payload))
        if path.startswith("commits/"):
            return {"sha": "b" * 40 if fault == "moved-ref" else "a" * 40}
        if payload:
            assert payload["return_run_details"] is True
            assert payload["inputs"]["expected_revision"] == "a" * 40
            return {"workflow_run_id": 42}
        return {
            "head_sha": "b" * 40 if fault == "wrong-sha" else "a" * 40,
            "path": (
                "wrong"
                if fault == "wrong-workflow"
                else ".github/workflows/publish.pypi.inference_exp.yml"
            ),
            "event": "release" if fault == "wrong-event" else "workflow_dispatch",
            "status": "in_progress" if fault == "timeout" else "completed",
            "conclusion": fault if fault in {"failure", "cancelled"} else "success",
        }

    monkeypatch.setattr(dependencies, "_github", github)
    with pytest.raises(ValueError):
        dependencies._dispatch("inference-models", version="2", wait_seconds=0)
    if fault == "moved-ref":
        assert len(calls) == 1


def test_child_guard_rejects_moved_revision_before_build(monkeypatch):
    monkeypatch.setenv("SOURCE_REVISION", "new")
    monkeypatch.setenv("EXPECTED_REVISION", "old")
    with pytest.raises(ValueError, match="revision"):
        dependencies._guard("inference-models")


def test_successful_child_without_visible_package_blocks_next_dispatch(monkeypatch):
    _PyPI(monkeypatch)
    models = _package("inference-models", "2")
    workflows = _package("roboflow-workflows", "2", ["inference-models==2"])
    plan = dependencies._plan(
        _runtime("roboflow-workflows==2"),
        catalog={p["name"]: p for p in [models, workflows]},
        registry=_registry(),
    )
    calls = []

    def dispatch(name, **kwargs):
        calls.append(name)
        return 42

    monkeypatch.setattr(dependencies, "_dispatch", dispatch)
    with pytest.raises(ValueError, match="timed out"):
        dependencies._coordinate(plan, wait_seconds=0)
    assert calls == ["inference-models"]


def test_visibility_delay_is_polled_without_redispatch(monkeypatch):
    package = _package("inference-models", "2")
    api = _PyPI(monkeypatch)
    clock = [0]
    monkeypatch.setattr(dependencies.time, "monotonic", lambda: clock[0])

    def sleep(seconds):
        clock[0] += seconds
        api.add(package)

    monkeypatch.setattr(dependencies.time, "sleep", sleep)
    result = dependencies._verify_file(
        {**package, "action": "publish"},
        registry=dependencies._Registry(wait_seconds=20),
    )
    assert result["version"] == "2"
    assert clock[0] == 10


def test_changed_published_metadata_blocks_consumer(monkeypatch):
    package = _package("inference-models", "2")
    api = _PyPI(monkeypatch, [package])
    plan = dependencies._plan(
        _runtime("inference-models==2"), catalog={}, registry=_registry()
    )
    api.add(_package("inference-models", "2", ["inference-sdk>=99"]))
    with pytest.raises(ValueError, match="metadata differs"):
        dependencies._verify_file(
            plan["packages"]["inference-models"], registry=_registry()
        )


def test_registry_hash_checked_again_at_consumer_boundary(monkeypatch):
    api = _PyPI(monkeypatch, [_package("inference-models", "2")])
    plan = dependencies._plan(
        _runtime("inference-models==2"), catalog={}, registry=_registry()
    )
    api.mutate_release(
        "inference-models",
        "2",
        lambda document: document["urls"][0]["digests"].update(sha256="changed"),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        dependencies._verify_file(
            plan["packages"]["inference-models"], registry=_registry()
        )


def test_child_guard_rejects_wrong_version(monkeypatch):
    monkeypatch.delenv("EXPECTED_REVISION", raising=False)
    monkeypatch.setenv("EXPECTED_VERSION", "3")
    monkeypatch.setattr(
        dependencies,
        "_catalog",
        lambda: {"inference-models": _package("inference-models", "2")},
    )
    with pytest.raises(ValueError, match="version"):
        dependencies._guard("inference-models")


def test_successful_dispatch_waits_for_exact_returned_run(monkeypatch):
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v1.9.0rc2")
    calls = []

    def github(path, *, payload=None):
        calls.append(path)
        if path.startswith("commits/"):
            return {"sha": "a" * 40}
        if payload:
            assert payload["ref"] == "refs/tags/v1.9.0rc2"
            return {"workflow_run_id": 123}
        assert path == "actions/runs/123"
        return {
            "head_sha": "a" * 40,
            "path": ".github/workflows/publish.pypi.inference_exp.yml",
            "event": "workflow_dispatch",
            "status": "completed",
            "conclusion": "success",
        }

    monkeypatch.setattr(dependencies, "_github", github)
    assert (
        dependencies._dispatch("inference-models", version="2", wait_seconds=0) == 123
    )
    assert len(calls) == 3


def test_final_verification_is_bound_to_consumer_and_complete_closure(monkeypatch):
    _PyPI(monkeypatch, [_package("inference-models", "1")])
    monkeypatch.setenv("SOURCE_REVISION", "candidate")
    consumers = _runtime("inference-models==1")
    plan = dependencies._plan(consumers, catalog={}, registry=_registry())
    plan["source_revision"] = "candidate"
    assert (
        len(dependencies._verify_plan(plan, registry=_registry(), consumers=consumers))
        == 1
    )
    altered = copy.deepcopy(consumers)
    altered[0]["requires_dist"] = ["inference-models==2"]
    with pytest.raises(ValueError, match="Consumer wheels"):
        dependencies._verify_plan(plan, registry=_registry(), consumers=altered)
    plan["packages"] = {}
    plan["order"] = []
    with pytest.raises(ValueError, match="do not satisfy"):
        dependencies._verify_plan(plan, registry=_registry(), consumers=consumers)


def test_plan_only_cli_never_dispatches_or_uploads(monkeypatch, tmp_path):
    _PyPI(monkeypatch)
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "wheels"
    directory.mkdir()
    filename, data = _wheel(_runtime("inference-models==2")[0])
    (directory / filename).write_bytes(data)
    monkeypatch.setattr(
        dependencies,
        "_catalog",
        lambda: {"inference-models": _package("inference-models", "2")},
    )
    monkeypatch.setattr(
        dependencies,
        "_github",
        lambda *a, **k: pytest.fail("Planning must not call GitHub"),
    )
    result = CliRunner().invoke(
        dependencies._main,
        ["plan", "--directory", str(directory), "--wait-seconds", "0"],
    )
    assert result.exit_code == 0, result.output
    plan = json.loads((tmp_path / "dist/dependency-plan.json").read_text())
    assert plan["packages"]["inference-models"]["action"] == "publish"


def test_workflow_dependency_ownership_and_gate_boundaries():
    def workflow(name):
        return yaml.load(
            (ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader
        )

    main = workflow("publish.pypi.yml")
    coordinator = main["jobs"]["prepare_dependencies"]
    assert coordinator["permissions"] == {"contents": "read", "actions": "write"}
    assert "!cancelled()" in coordinator["if"]
    assert (
        "needs.prepare_dependencies.result == 'success'"
        in main["jobs"]["publish_runtime"]["if"]
    )
    assert "prepare_dependencies" in main["jobs"]["publish_runtime"]["needs"]
    for name, (_, filename) in dependencies.PROJECTS.items():
        config = workflow(filename)
        assert set(config["on"]) == {"workflow_dispatch"}
        assert "environment" not in config["jobs"]["build"]
        steps = config["jobs"]["build"]["steps"]
        guard = next(
            i
            for i, step in enumerate(steps)
            if "dependencies.sh guard" in step.get("run", "")
        )
        gate = next(
            i
            for i, step in enumerate(steps)
            if "dependencies.sh gate" in step.get("run", "")
        )
        upload = next(
            i
            for i, step in enumerate(steps)
            if step.get("uses", "").startswith("pypa/")
        )
        assert guard < gate < upload
        assert "release-evidence/" in steps[gate]["run"]
        assert steps[gate]["if"] == steps[upload]["if"]
        assert "expected_revision" in config["on"]["workflow_dispatch"]["inputs"]


def test_real_runtime_wheels_feed_dependency_plan(monkeypatch, tmp_path):
    # Build real source inventories in an isolated checkout, with no runtime
    # installation and no PyPI mutation. Only the registry transport is faked.
    checkout = tmp_path / "checkout"
    shutil.copytree(
        ROOT,
        checkout,
        ignore=shutil.ignore_patterns(
            ".git", "dist", "build", "*.egg-info", "__pycache__", ".pytest_cache"
        ),
    )
    monkeypatch.chdir(checkout)
    monkeypatch.delenv("EXPECTED_VERSION", raising=False)
    monkeypatch.delenv("PUBLISH", raising=False)
    dependencies.release._run("build-runtime", wait_seconds=0)
    catalog = dependencies._catalog()
    _PyPI(monkeypatch)
    consumers = dependencies._local_wheels(Path("dist/pypi"))
    plan = dependencies._plan(consumers, catalog=catalog, registry=_registry())
    assert {p["name"] for p in consumers} == dependencies.release.RUNTIME
    assert set(plan["packages"]) == dependencies.OWNED
    assert (
        plan["packages"]["inference-sdk"]["version"]
        == dependencies.release._source_version()
    )
    assert plan["packages"]["inference-sdk"]["action"] == "sdk"
    assert plan["packages"]["inference-models"]["version"] == "0.39.8"
    assert plan["packages"]["roboflow-workflows"]["version"] == "0.2.15"
    assert plan["packages"]["streamvision"]["version"] == "0.1.5"
