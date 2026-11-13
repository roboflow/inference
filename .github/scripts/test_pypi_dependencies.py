"""Dependency release boundaries; execute inside the release Docker image."""

import copy
import hashlib
import http.client
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.parse
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


@pytest.fixture(autouse=True)
def _fake_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(dependencies.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        dependencies.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    return clock


def _package(name, version, requirements=(), extras=()):
    result = {
        "name": name,
        "version": version,
        "requires_dist": list(requirements),
        "provides_extra": list(extras),
        "requires_python": ">=3.10,<3.14",
    }
    return result


def _wheel(file, *, code="", timestamp=(2026, 1, 1, 0, 0, 0)):
    stem = f"{file['name'].replace('-', '_')}-{file['version']}"
    buffer = io.BytesIO()
    metadata = f"Metadata-Version: 2.4\nName: {file['name']}\nVersion: {file['version']}\nRequires-Python: {file['requires_python']}\n"
    metadata += "".join(f"Requires-Dist: {raw}\n" for raw in file["requires_dist"])
    metadata += "".join(
        f"Provides-Extra: {extra}\n" for extra in file["provides_extra"]
    )
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            zipfile.ZipInfo(
                f"{file['name'].replace('-', '_')}/__init__.py", date_time=timestamp
            ),
            code,
        )
        archive.writestr(
            zipfile.ZipInfo(f"{stem}.dist-info/METADATA", date_time=timestamp),
            metadata + "\n",
        )
    return f"{stem}-py3-none-any.whl", buffer.getvalue()


class _PyPI:
    def __init__(self, monkeypatch, packages=(), *, complete=True):
        self.responses = {}
        self.calls = []
        self.packages = {}
        for file in packages:
            self.add(file, complete=complete)
        monkeypatch.setattr(dependencies.release, "_read_url", self.read)

    def add(self, file, *, complete=True, **kwargs):
        name, version = file["name"], file["version"]
        filename, data = _wheel(file, **kwargs)
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
        remotes = [remote]
        if complete and name in dependencies.publish.SDISTS:
            stem = f"{name.replace('-', '_')}-{version}"
            buffer = io.BytesIO()
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                metadata = archive.read(f"{stem}.dist-info/METADATA")
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                entry = tarfile.TarInfo(f"{stem}/PKG-INFO")
                entry.size = len(metadata)
                archive.addfile(entry, io.BytesIO(metadata))
            sdist = buffer.getvalue()
            sdist_url = f"https://files.pythonhosted.org/{stem}.tar.gz"
            self.responses[sdist_url] = sdist
            remotes.append(
                {
                    "filename": f"{stem}.tar.gz",
                    "url": sdist_url,
                    "digests": {"sha256": hashlib.sha256(sdist).hexdigest()},
                    "yanked": False,
                }
            )
        self.responses[f"https://pypi.org/pypi/{name}/{version}/json"] = json.dumps(
            {"info": {"name": name, "version": version}, "urls": remotes}
        ).encode()
        self.packages.setdefault(name, {})[version] = remotes
        self.responses[f"https://pypi.org/pypi/{name}/json"] = json.dumps(
            {"releases": self.packages[name]}
        ).encode()

    def read(self, url, **kwargs):
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
    return dependencies._Registry(wait_seconds=20)


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

    def dispatch(name, *, version, **kwargs):
        if name == "roboflow-workflows":
            assert any(
                "files.pythonhosted.org/inference_models" in url for url in api.calls
            )
        calls.append((name, version))
        api.add({models["name"]: models, workflows["name"]: workflows}[name])
        return len(calls)

    monkeypatch.setattr(dependencies, "_dispatch", dispatch)
    assert dependencies._coordinate(plan, wait_seconds=20) == {
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
        dependencies._coordinate(plan, wait_seconds=20)
    assert calls == ["inference-models"]
    assert not any(
        urllib.parse.urlparse(url).hostname == "files.pythonhosted.org"
        for url in api.calls
    )


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

    def github(path, *, payload=None, **kwargs):
        calls.append((path, payload))
        if path.startswith("commits/"):
            return {"sha": "b" * 40 if fault == "moved-ref" else "a" * 40}
        if payload:
            assert set(payload) == {"ref", "inputs"}
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
        dependencies._dispatch("inference-models", version="2", wait_seconds=20)
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
        dependencies._coordinate(plan, wait_seconds=20)
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

    def github(path, *, payload=None, **kwargs):
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
        dependencies._dispatch("inference-models", version="2", wait_seconds=20) == 123
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
        ["plan", "--directory", str(directory), "--wait-seconds", "20"],
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
        assert "github.event.inputs.publish == 'true'" in steps[upload]["if"]
        assert "expected_revision" in config["on"]["workflow_dispatch"]["inputs"]


def test_real_runtime_wheels_feed_dependency_plan(monkeypatch, tmp_path):
    # Build real source inventories in an isolated checkout, with no runtime
    # installation and no PyPI mutation. Only the registry transport is faked.
    checkout = tmp_path / "checkout"
    shutil.copytree(
        ROOT,
        checkout,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "venv",
            ".tox",
            "dist",
            "build",
            "*.egg-info",
            "__pycache__",
            ".pytest_cache",
        ),
    )
    monkeypatch.chdir(checkout)
    monkeypatch.delenv("EXPECTED_VERSION", raising=False)
    monkeypatch.delenv("PUBLISH", raising=False)
    dependencies.release._run("build-runtime", wait_seconds=20)
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
    for name in dependencies.PROJECTS:
        assert plan["packages"][name]["version"] == catalog[name]["version"]


def test_dispatch_payload_has_only_documented_keys(monkeypatch):
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/candidate")

    def github(path, *, payload=None, **kwargs):
        if path.startswith("commits/"):
            return {"sha": "a" * 40}
        if payload is not None:
            assert set(payload) == {"ref", "inputs"}
            return {"workflow_run_id": 42}
        return {
            "head_sha": "a" * 40,
            "path": ".github/workflows/publish.pypi.inference_exp.yml",
            "event": "workflow_dispatch",
            "status": "completed",
            "conclusion": "success",
        }

    monkeypatch.setattr(dependencies, "_github", github)
    assert (
        dependencies._dispatch("inference-models", version="2", wait_seconds=20) == 42
    )


def test_zero_dependency_budget_makes_no_requests(monkeypatch):
    api = _PyPI(monkeypatch, [_package("inference-models", "2")])
    with pytest.raises(ValueError, match="timed out"):
        dependencies._Registry(wait_seconds=0)._file("inference-models", "2")
    assert api.calls == []


def test_cyclic_candidate_backtracks_to_older_release(monkeypatch):
    _PyPI(
        monkeypatch,
        [
            _package("roboflow-workflows", "2", ["inference-models==1"]),
            _package("roboflow-workflows", "1"),
            _package("inference-models", "1", ["roboflow-workflows>=1"]),
        ],
    )
    plan = dependencies._plan(
        _runtime("roboflow-workflows>=1"), catalog={}, registry=_registry()
    )
    assert plan["packages"]["roboflow-workflows"]["version"] == "1"


def test_children_verify_built_contents_after_upload():
    for _, filename in dependencies.PROJECTS.values():
        workflow = yaml.load(
            (ROOT / ".github/workflows" / filename).read_text(), Loader=yaml.BaseLoader
        )
        steps = workflow["jobs"]["build"]["steps"]
        upload = next(
            i
            for i, step in enumerate(steps)
            if step.get("uses", "").startswith("pypa/")
        )
        assert "publish.sh verify" in steps[upload + 1].get("run", "")


@pytest.mark.parametrize(
    "fault", ["source", "reused-sha", "provides-extra", "requirement-extra"]
)
def test_retained_dependency_gates_reject_mutation(monkeypatch, fault):
    package = _package("inference-models", "2", extras=["x"])
    api = _PyPI(monkeypatch, [package])
    consumers = _runtime("inference-models[x]==2")
    plan = dependencies._plan(consumers, catalog={}, registry=_registry())
    monkeypatch.setenv("SOURCE_REVISION", "candidate")
    plan["source_revision"] = "candidate"
    if fault == "source":
        plan["source_revision"] = "other"
    elif fault == "reused-sha":
        api.add(package, timestamp=(2025, 1, 1, 0, 0, 0))
    elif fault == "provides-extra":
        plan["packages"]["inference-models"]["provides_extra"] = ["x", "missing"]
    else:
        plan["packages"]["inference-models"]["requires_dist"] = [
            "inference-models[missing]==2"
        ]
    with pytest.raises(
        ValueError, match="source revision|changed since planning|metadata differs"
    ):
        dependencies._verify_plan(plan, registry=_registry(), consumers=consumers)


@pytest.mark.parametrize("minor", range(10, 14))
def test_every_supported_python_version_is_required(monkeypatch, minor):
    package = _package("inference-models", "2")
    package["requires_python"] = f">=3.10,<3.14,!=3.{minor}.0"
    _PyPI(monkeypatch, [package])
    with pytest.raises(ValueError, match="No compatible"):
        dependencies._plan(
            _runtime("inference-models>=1"), catalog={}, registry=_registry()
        )


@pytest.mark.parametrize("fault", ["yanked", "platform"])
def test_version_search_filters_unusable_wheels(monkeypatch, fault):
    api = _PyPI(
        monkeypatch,
        [_package("inference-models", "1"), _package("inference-models", "2")],
    )
    remote = api.packages["inference-models"]["2"][0]
    if fault == "yanked":
        remote["yanked"] = True
    else:
        remote["filename"] = "inference_models-2-cp312-cp312-linux_aarch64.whl"
    api.responses["https://pypi.org/pypi/inference-models/json"] = json.dumps(
        {"releases": api.packages["inference-models"]}
    ).encode()
    plan = dependencies._plan(
        _runtime("inference-models>=1"), catalog={}, registry=_registry()
    )
    assert plan["packages"]["inference-models"]["version"] == "1"


def test_extra_names_normalize_in_metadata_and_internal_edges(monkeypatch):
    package = _package(
        "roboflow-workflows", "2", ["inference-models[Torch_CPU]>=1"], ["Feature_X"]
    )
    _PyPI(
        monkeypatch,
        [
            _package("inference-models", "1", extras=["torch-cpu"]),
            _package(
                "roboflow-workflows",
                "2",
                ["inference-models[torch-cpu]>=1"],
                ["feature-x"],
            ),
        ],
    )
    actual = dependencies._verify_file(
        {**package, "action": "publish"}, registry=_registry()
    )
    assert dependencies._compatible(
        actual, dependencies._requirements(_runtime("roboflow-workflows[FEATURE_X]>=1"))
    )


def test_self_referencing_extra_is_not_a_cycle(monkeypatch):
    _PyPI(
        monkeypatch,
        [_package("inference-models", "1", ["inference-models[x]>=1"], ["x"])],
    )
    plan = dependencies._plan(
        _runtime("inference-models[x]==1"), catalog={}, registry=_registry()
    )
    assert plan["order"] == ["inference-models"]


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("https://pypi.org", 429, "busy", None, None),
        http.client.HTTPException("incomplete"),
        ConnectionResetError("reset"),
    ],
)
def test_registry_transient_errors_recover_without_new_budget(
    monkeypatch, error, _fake_clock
):
    package = _package("inference-models", "2")
    api = _PyPI(monkeypatch, [package])
    deadlines = []

    def read(url, *, deadline):
        deadlines.append(deadline)
        if len(deadlines) == 1:
            raise error
        return api.read(url)

    monkeypatch.setattr(dependencies.release, "_read_url", read)
    registry = dependencies._Registry(wait_seconds=20)
    assert registry._file("inference-models", "2")["version"] == "2"
    assert deadlines == [20, 20, 20, 20]
    assert _fake_clock[0] == 10


def test_nested_registry_retries_share_deadline_and_stop_before_next_request(
    monkeypatch, _fake_clock
):
    api = _PyPI(monkeypatch, [_package("inference-models", "2")])
    calls = []

    def read(url, *, deadline):
        calls.append((url, deadline))
        _fake_clock[0] += 4
        if urllib.parse.urlparse(url).hostname == "files.pythonhosted.org":
            raise urllib.error.HTTPError(url, 429, "busy", None, None)
        return api.read(url)

    monkeypatch.setattr(dependencies.release, "_read_url", read)
    with pytest.raises(ValueError, match="timed out"):
        dependencies._Registry(wait_seconds=10)._file("inference-models", "2")
    assert len(calls) == 2
    assert [deadline for _, deadline in calls] == [10, 10]
    assert _fake_clock[0] == 10


def test_cached_registry_read_and_resolver_respect_expired_deadline(
    monkeypatch, _fake_clock
):
    api = _PyPI(monkeypatch, [_package("inference-models", "2")])
    registry = _registry()
    registry._file("inference-models", "2")
    before = len(api.calls)
    _fake_clock[0] = registry.deadline
    with pytest.raises(ValueError, match="timed out"):
        registry._file("inference-models", "2")
    with pytest.raises(ValueError, match="timed out"):
        dependencies._plan(
            _runtime("inference-models==2"), catalog={}, registry=registry
        )
    assert len(api.calls) == before


def _built_candidate(monkeypatch, tmp_path, package, **kwargs):
    monkeypatch.setattr(dependencies, "_catalog", lambda: {package["name"]: package})
    directory = tmp_path / "dist"
    directory.mkdir()
    filename, data = _wheel(package, **kwargs)
    (directory / filename).write_bytes(data)
    (directory / "source.tar.gz").write_bytes(b"sdist may coexist")
    return directory


def test_skip_existing_content_race_fails_child_and_stops_downstream(
    monkeypatch, tmp_path
):
    models = _package("inference-models", "2")
    workflows = _package("roboflow-workflows", "2", ["inference-models==2"])
    api = _PyPI(monkeypatch)
    plan = dependencies._plan(
        _runtime("roboflow-workflows==2"),
        catalog={p["name"]: p for p in [models, workflows]},
        registry=_registry(),
    )
    directory = _built_candidate(monkeypatch, tmp_path, models, code="candidate = True")
    api.add(models, code="concurrent_upload = True")
    calls = []
    receipt = tmp_path / "release-evidence/dependency-plan.json"

    def dispatch(name, **kwargs):
        calls.append(name)
        if name == "inference-models":
            dependencies._verify_published(
                name, directory=directory, plan_file=receipt, deadline=20
            )
        else:
            api.add(workflows)
        return 42

    monkeypatch.setattr(dependencies, "_dispatch", dispatch)
    with pytest.raises(ValueError, match="contents differ"):
        dependencies._coordinate(plan, wait_seconds=20)
    assert calls == ["inference-models"]
    evidence = json.loads(
        receipt.with_name("published-content-receipt.json").read_text()
    )
    assert (
        evidence["candidate"]["requires_dist"] == evidence["published"]["requires_dist"]
    )
    assert (
        evidence["candidate"]["content_sha256"]
        != evidence["published"]["content_sha256"]
    )


def test_post_upload_verification_accepts_equivalent_zip_encoding(
    monkeypatch, tmp_path
):
    package = _package("inference-models", "2")
    api = _PyPI(monkeypatch)
    directory = _built_candidate(
        monkeypatch, tmp_path, package, timestamp=(2025, 1, 1, 0, 0, 0)
    )
    api.add(package, timestamp=(2026, 1, 1, 0, 0, 0))
    path = tmp_path / "release-evidence/dependency-plan.json"
    result = CliRunner().invoke(
        dependencies._main,
        [
            "verify-published",
            "--package",
            package["name"],
            "--directory",
            str(directory),
            "--plan-file",
            str(path),
            "--wait-seconds",
            "20",
        ],
    )
    assert result.exit_code == 0, result.output
    receipt = json.loads(path.with_name("published-content-receipt.json").read_text())
    assert receipt["candidate"]["sha256"] != receipt["published"]["sha256"]
    assert (
        receipt["candidate"]["content_sha256"] == receipt["published"]["content_sha256"]
    )


@pytest.mark.parametrize("fault", ["missing", "multiple", "name", "version"])
def test_post_upload_verification_requires_exact_catalog_candidate(
    monkeypatch, tmp_path, fault
):
    package = _package("inference-models", "2")
    api = _PyPI(monkeypatch)
    directory = _built_candidate(monkeypatch, tmp_path, package)
    if fault == "missing":
        next(directory.glob("*.whl")).unlink()
    elif fault == "multiple":
        filename, data = _wheel(_package("inference-models", "3"))
        (directory / filename).write_bytes(data)
    else:
        monkeypatch.setattr(
            dependencies,
            "_catalog",
            lambda: {
                package["name"]: (
                    {**package, "name": "other"}
                    if fault == "name"
                    else {**package, "version": "3"}
                )
            },
        )
        if fault == "name":
            next(directory.glob("*.whl")).unlink()
            filename, data = _wheel(_package("streamvision", "2"))
            (directory / filename).write_bytes(data)
    with pytest.raises(ValueError, match="wheel"):
        dependencies._verify_published(
            package["name"],
            directory=directory,
            plan_file=tmp_path / "receipt.json",
            deadline=20,
        )
    assert api.calls == []


@pytest.mark.parametrize("changed_file", ["wheel", "sdist"])
def test_published_hashes_and_run_id_are_retained_and_rechecked(
    monkeypatch, tmp_path, changed_file
):
    package = _package("inference-models", "2")
    api = _PyPI(monkeypatch)
    consumers = _runtime("inference-models==2")
    plan = dependencies._plan(
        consumers, catalog={package["name"]: package}, registry=_registry()
    )
    monkeypatch.setenv("SOURCE_REVISION", "candidate")
    plan["source_revision"] = "candidate"

    def dispatch(name, **kwargs):
        api.add(package, code="published = True")
        return 42

    monkeypatch.setattr(dependencies, "_dispatch", dispatch)
    path = tmp_path / "plan.json"
    dependencies._coordinate(plan, wait_seconds=20, plan_file=path)
    retained = json.loads(path.read_text())
    entry = retained["packages"][package["name"]]
    assert entry["publisher_run_id"] == 42
    assert entry["sha256"] and entry["content_sha256"]
    assert len(entry["published_files"]) == 2
    dependencies._verify_plan(retained, registry=_registry(), consumers=consumers)
    if changed_file == "wheel":
        api.add(package, code="replaced = True")
    else:
        url = "https://files.pythonhosted.org/inference_models-2.tar.gz"
        altered = io.BytesIO()
        with tarfile.open(
            fileobj=io.BytesIO(api.responses[url]), mode="r:gz"
        ) as source, tarfile.open(fileobj=altered, mode="w:gz") as target:
            for member in source:
                member.mode = 0o755
                target.addfile(member, source.extractfile(member))
        api.responses[url] = altered.getvalue()
        api.mutate_release(
            package["name"],
            package["version"],
            lambda document: document["urls"][1]["digests"].update(
                sha256=hashlib.sha256(altered.getvalue()).hexdigest()
            ),
        )
    with pytest.raises(ValueError, match="changed since planning"):
        dependencies._verify_plan(retained, registry=_registry(), consumers=consumers)


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("https://api.github.com", 429, "busy", None, None),
        urllib.error.HTTPError("https://api.github.com", 503, "busy", None, None),
        http.client.HTTPException("incomplete"),
        ConnectionResetError("reset"),
    ],
)
def test_github_get_retries_transient_failures_with_same_deadline(
    monkeypatch, _fake_clock, error
):
    monkeypatch.setenv("GH_TOKEN", "fixture")
    monkeypatch.setenv("GITHUB_REPOSITORY", "roboflow/inference")
    calls = []

    def open_url(request, *, timeout):
        calls.append((request.get_method(), timeout))
        if len(calls) == 1:
            raise error
        return io.BytesIO(b'{"status": "completed"}')

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", open_url)
    assert dependencies._github("actions/runs/42", deadline=20)["status"] == "completed"
    assert calls == [("GET", 20), ("GET", 10)]
    assert _fake_clock[0] == 10


@pytest.mark.parametrize("status", [401, 403, 404])
def test_github_auth_or_missing_run_fails_without_retry(monkeypatch, status):
    monkeypatch.setenv("GH_TOKEN", "fixture")
    monkeypatch.setenv("GITHUB_REPOSITORY", "roboflow/inference")
    calls = []

    def open_url(request, *, timeout):
        calls.append(timeout)
        raise urllib.error.HTTPError(request.full_url, status, "failure", None, None)

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", open_url)
    with pytest.raises(urllib.error.HTTPError):
        dependencies._github("actions/runs/42", deadline=20)
    assert calls == [20]


def test_ambiguous_dispatch_response_is_never_retried(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "fixture")
    monkeypatch.setenv("GITHUB_REPOSITORY", "roboflow/inference")
    calls = []

    def open_url(request, *, timeout):
        calls.append((request.get_method(), timeout))
        raise ConnectionResetError("response lost")

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", open_url)
    with pytest.raises(ValueError, match="may have started"):
        dependencies._github(
            "actions/workflows/child.yml/dispatches",
            payload={"ref": "candidate", "inputs": {}},
            deadline=20,
        )
    assert calls == [("POST", 20)]


def test_dispatch_polling_does_not_request_after_deadline(monkeypatch, _fake_clock):
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/candidate")
    calls = []

    def github(path, *, payload=None, deadline):
        calls.append((path, deadline))
        if path.startswith("commits/"):
            _fake_clock[0] += 2
            return {"sha": "a" * 40}
        if payload is not None:
            _fake_clock[0] += 2
            return {"workflow_run_id": 42}
        return {
            "head_sha": "a" * 40,
            "path": ".github/workflows/publish.pypi.inference_exp.yml",
            "event": "workflow_dispatch",
            "status": "in_progress",
            "conclusion": None,
        }

    monkeypatch.setattr(dependencies, "_github", github)
    with pytest.raises(ValueError, match="run 42.*timed out"):
        dependencies._dispatch("inference-models", version="2", wait_seconds=10)
    assert len(calls) == 3
    assert {deadline for _, deadline in calls} == {10}
    assert _fake_clock[0] == 10


def test_coordination_uses_one_deadline_and_fresh_final_registry(
    monkeypatch, tmp_path, _fake_clock
):
    models = _package("inference-models", "2")
    workflows = _package("roboflow-workflows", "2", ["inference-models==2"])
    api = _PyPI(monkeypatch)
    consumers = _runtime("roboflow-workflows==2")
    plan = dependencies._plan(
        consumers,
        catalog={p["name"]: p for p in [models, workflows]},
        registry=_registry(),
    )
    monkeypatch.setenv("SOURCE_REVISION", "candidate")
    plan["source_revision"] = "candidate"
    deadlines = []

    def dispatch(name, *, version, deadline):
        deadlines.append(deadline)
        _fake_clock[0] += 4
        api.add({models["name"]: models, workflows["name"]: workflows}[name])
        return len(deadlines)

    monkeypatch.setattr(dependencies, "_dispatch", dispatch)
    dependencies._coordinate(plan, deadline=20, plan_file=tmp_path / "plan.json")
    assert deadlines == [20, 20]
    api.add(models, code="mutated_after_child = True")
    with pytest.raises(ValueError, match="changed since planning"):
        dependencies._verify_plan(
            plan, registry=dependencies._Registry(deadline=20), consumers=consumers
        )


def test_internal_requirement_extras_are_part_of_metadata_identity(monkeypatch):
    expected = _package("roboflow-workflows", "2", ["inference-models[torch-cpu]>=1"])
    actual = _package("roboflow-workflows", "2", ["inference-models[torch-gpu]>=1"])
    _PyPI(monkeypatch, [actual])
    with pytest.raises(ValueError, match="metadata differs"):
        dependencies._verify_file(
            {**expected, "action": "publish"}, registry=_registry()
        )


@pytest.mark.parametrize("payload", [None, {"ref": "candidate", "inputs": {}}])
def test_github_success_after_deadline_is_rejected(monkeypatch, _fake_clock, payload):
    monkeypatch.setenv("GH_TOKEN", "fixture")
    monkeypatch.setenv("GITHUB_REPOSITORY", "roboflow/inference")
    calls = []

    def open_url(request, *, timeout):
        calls.append(timeout)
        _fake_clock[0] = 21
        return io.BytesIO(b'{"workflow_run_id":42}')

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", open_url)
    with pytest.raises(ValueError, match="timed out|may have started"):
        dependencies._github("actions/runs/42", payload=payload, deadline=20)
    assert calls == [20]


def test_retry_rejects_success_returned_after_deadline(_fake_clock):
    def operation():
        _fake_clock[0] = 21
        return "late success"

    with pytest.raises(ValueError, match="timed out"):
        dependencies._retry(operation, deadline=20)


def test_extra_aliases_are_deduplicated_in_internal_edges():
    aliased = _package(
        "roboflow-workflows", "2", ["inference-models[foo_bar,foo-bar]>=1"]
    )
    normalized = _package("roboflow-workflows", "2", ["inference-models[foo-bar]>=1"])
    assert dependencies._edges(aliased) == dependencies._edges(normalized)


@pytest.mark.parametrize("missing", ["sha256", "content_sha256", "publisher_run_id"])
def test_final_verification_requires_retained_publisher_receipt(monkeypatch, missing):
    _PyPI(monkeypatch, [_package("inference-models", "2")])
    consumers = _runtime("inference-models==2")
    plan = dependencies._plan(consumers, catalog={}, registry=_registry())
    monkeypatch.setenv("SOURCE_REVISION", "candidate")
    plan["source_revision"] = "candidate"
    entry = plan["packages"]["inference-models"]
    entry.update(action="publish", publisher_run_id=42)
    del entry[missing]
    with pytest.raises(ValueError, match="Missing retained publisher receipt"):
        dependencies._verify_plan(plan, registry=_registry(), consumers=consumers)


def test_sdk_catalog_reference_is_not_a_publisher_content_requirement(monkeypatch):
    package = _package("inference-sdk", "2")
    api = _PyPI(monkeypatch, [package])
    filename, data = _wheel(package, timestamp=(2025, 1, 1, 0, 0, 0))
    candidate = dependencies._wheel(data, filename=filename)
    api.add(package, code="published_from_accepted_toolchain = True")
    actual = dependencies._verify_file(
        {**candidate, "action": "sdk"}, registry=_registry()
    )
    assert actual["sha256"] != candidate["sha256"]
    assert actual["content_sha256"] != candidate["content_sha256"]


def test_reuse_without_retained_sha_fails_closed(monkeypatch):
    package = _package("inference-models", "2")
    _PyPI(monkeypatch, [package])
    expected = {
        **dependencies._Registry(wait_seconds=20)._file(
            package["name"], package["version"]
        ),
        "action": "reuse",
    }
    del expected["sha256"]
    with pytest.raises(ValueError, match="Missing retained SHA-256"):
        dependencies._verify_file(expected, registry=_registry())


def test_same_checkout_wheel_only_release_is_repaired(monkeypatch):
    package = _package("inference-models", "2")
    _PyPI(monkeypatch, [package], complete=False)
    plan = dependencies._plan(
        _runtime("inference-models==2"),
        catalog={"inference-models": package},
        registry=_registry(),
    )
    assert plan["packages"]["inference-models"]["action"] == "publish"


@pytest.mark.parametrize("present_suffix", [".whl", ".tar.gz"])
def test_same_version_partial_inventory_dispatches_original_child(
    monkeypatch, present_suffix
):
    package = _package("inference-models", "2")
    api = _PyPI(monkeypatch, [package])
    api.mutate_release(
        "inference-models",
        "2",
        lambda document: document.update(
            urls=[
                file
                for file in document["urls"]
                if file["filename"].endswith(present_suffix)
            ]
        ),
    )
    plan = dependencies._plan(
        _runtime("inference-models==2"),
        catalog={"inference-models": package},
        registry=_registry(),
    )
    assert plan["packages"]["inference-models"]["action"] == "publish"
    calls = []

    def dispatch(name, *, version, deadline):
        calls.append((name, version))
        api.add(package)
        return 42

    monkeypatch.setattr(dependencies, "_dispatch", dispatch)
    assert dependencies._coordinate(plan, wait_seconds=20) == {"inference-models": 42}
    assert calls == [("inference-models", "2")]


@pytest.mark.parametrize(
    "catalog", [{}, {"inference-models": _package("inference-models", "3")}]
)
@pytest.mark.parametrize("present_suffix", [".whl", ".tar.gz"])
def test_gate_and_older_partial_versions_cannot_authorize_repair(
    monkeypatch, catalog, present_suffix
):
    api = _PyPI(monkeypatch, [_package("inference-models", "2")])
    api.mutate_release(
        "inference-models",
        "2",
        lambda document: document.update(
            urls=[
                file
                for file in document["urls"]
                if file["filename"].endswith(present_suffix)
            ]
        ),
    )
    with pytest.raises(ValueError, match="No compatible"):
        dependencies._plan(
            _runtime("inference-models==2"), catalog=catalog, registry=_registry()
        )


def test_older_partial_range_selects_genuinely_complete_alternative(monkeypatch):
    api = _PyPI(
        monkeypatch,
        [_package("inference-models", "1"), _package("inference-models", "2")],
    )
    api.mutate_release(
        "inference-models",
        "2",
        lambda document: document.update(urls=document["urls"][:1]),
    )
    plan = dependencies._plan(
        _runtime("inference-models>=1"),
        catalog={"inference-models": _package("inference-models", "3")},
        registry=_registry(),
    )
    assert plan["packages"]["inference-models"]["version"] == "1"
    assert plan["packages"]["inference-models"]["action"] == "reuse"


def test_successful_child_with_only_wheel_still_blocks_dependents(monkeypatch):
    api = _PyPI(monkeypatch)
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
        api.add(models, complete=False)
        return 42

    monkeypatch.setattr(dependencies, "_dispatch", dispatch)
    with pytest.raises(ValueError, match="timed out"):
        dependencies._coordinate(plan, wait_seconds=20)
    assert calls == ["inference-models"]


def test_partial_visibility_is_refreshed_until_complete(monkeypatch):
    package = _package("inference-models", "2")
    api = _PyPI(monkeypatch, [package], complete=False)
    clock = [0]
    monkeypatch.setattr(dependencies.time, "monotonic", lambda: clock[0])

    def sleep(seconds):
        clock[0] += seconds
        api.add(package)

    monkeypatch.setattr(dependencies.time, "sleep", sleep)
    actual = dependencies._verify_file(
        {**package, "action": "publish"},
        registry=dependencies._Registry(wait_seconds=20),
    )
    assert actual["complete"]
    assert len(actual["published_files"]) == 2
    assert clock[0] == 10


def test_sdist_corruption_prevents_reusing_otherwise_valid_wheel(monkeypatch):
    package = _package("inference-models", "2")
    api = _PyPI(monkeypatch, [package])
    api.mutate_release(
        "inference-models",
        "2",
        lambda document: document["urls"][1]["digests"].update(sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        dependencies._plan(
            _runtime("inference-models==2"),
            catalog={"inference-models": package},
            registry=_registry(),
        )


def test_sdist_change_after_planning_blocks_consumer(monkeypatch):
    package = _package("inference-models", "2")
    api = _PyPI(monkeypatch, [package])
    plan = dependencies._plan(
        _runtime("inference-models==2"), catalog={}, registry=_registry()
    )
    url = "https://files.pythonhosted.org/inference_models-2.tar.gz"
    data = api.responses[url]
    altered = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as source, tarfile.open(
        fileobj=altered, mode="w:gz"
    ) as target:
        for entry in source:
            entry.mode = 0o755
            target.addfile(entry, source.extractfile(entry))
    api.responses[url] = altered.getvalue()
    api.mutate_release(
        "inference-models",
        "2",
        lambda document: document["urls"][1]["digests"].update(
            sha256=hashlib.sha256(altered.getvalue()).hexdigest()
        ),
    )
    with pytest.raises(ValueError, match="inventory changed"):
        dependencies._verify_file(
            plan["packages"]["inference-models"], registry=_registry()
        )


def test_equivalent_published_version_spelling_can_be_reused(monkeypatch):
    api = _PyPI(monkeypatch, [_package("inference-models", "1.0.0")])
    api.responses["https://pypi.org/pypi/inference-models/1.0/json"] = api.responses[
        "https://pypi.org/pypi/inference-models/1.0.0/json"
    ]
    plan = dependencies._plan(
        _runtime("inference-models==1.0"), catalog={}, registry=_registry()
    )
    assert plan["packages"]["inference-models"]["version"] == "1.0.0"
    assert plan["packages"]["inference-models"]["action"] == "reuse"


def test_range_skips_fully_yanked_newest_version(monkeypatch):
    api = _PyPI(
        monkeypatch,
        [_package("inference-models", "1"), _package("inference-models", "2")],
    )
    project = json.loads(api.responses["https://pypi.org/pypi/inference-models/json"])
    for file in project["releases"]["2"]:
        file["yanked"] = True
    api.responses["https://pypi.org/pypi/inference-models/json"] = json.dumps(
        project
    ).encode()
    plan = dependencies._plan(
        _runtime("inference-models>=1"), catalog={}, registry=_registry()
    )
    assert plan["packages"]["inference-models"]["version"] == "1"


def test_registry_file_order_does_not_change_retained_plan(monkeypatch):
    package = _package("inference-models", "2")
    api = _PyPI(monkeypatch, [package])
    plan = dependencies._plan(
        _runtime("inference-models==2"), catalog={}, registry=_registry()
    )
    api.mutate_release(
        "inference-models", "2", lambda document: document["urls"].reverse()
    )
    assert dependencies._verify_file(
        plan["packages"]["inference-models"], registry=_registry()
    )["complete"]
