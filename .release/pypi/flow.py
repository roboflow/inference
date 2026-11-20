"""Release preparation, immutable RC evidence, and guarded lockfile commits."""

import ast
import base64
import functools
import hashlib
import http.client
import io
import json
import lzma
import os
import re
import subprocess
import time
import tomllib
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path
from urllib.parse import urlsplit

import click
import dependencies
import release
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version
from versioning import _final_version

STATE = Path("release-state")
LOCKS = tuple(f"{directory}/uv.lock" for directory, _ in dependencies.PROJECTS.values())
FIRST_PARTY = dependencies.OWNED | release.RUNTIME
REPOSITORY = "roboflow/inference"
POLICY = Path(".release/release-policy.json")


class _EvidenceDeadlineExceeded(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def _api_deadline():
    deadline = time.monotonic() + 600
    return deadline


def _api_timeout():
    remaining = _api_deadline() - time.monotonic()
    if remaining <= 0:
        raise _EvidenceDeadlineExceeded(
            "GitHub evidence lookup exceeded its 10-minute budget"
        )

    timeout = min(30, remaining)
    return timeout


def _git(*args, env=None, timeout=None):
    result = subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
    )
    return result.stdout.strip()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _outputs(**values):
    STATE.mkdir(exist_ok=True)
    with (STATE / "outputs").open("a") as stream:
        for name, value in values.items():
            if "\n" in str(value) or "\r" in str(value):
                raise ValueError("Invalid multiline job output")
            stream.write(f"{name}={value}\n")


def _state():
    result = json.loads((STATE / "release.json").read_text())
    return result


def _api(path, *, payload=None):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPOSITORY}/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        data=json.dumps(payload).encode() if payload is not None else None,
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=_api_timeout()) as response:
                result = json.load(response)
            _api_timeout()
            return result
        except urllib.error.HTTPError as error:
            if (
                payload is not None
                or error.code not in {429, 500, 502, 503, 504}
                or attempt == 5
            ):
                raise
        except (TimeoutError, OSError):
            if payload is not None or attempt == 5:
                raise
        time.sleep(min(2**attempt, 16, _api_timeout()))
    raise ValueError("GitHub request exhausted retries")


def _base(version):
    result = _final_version(version)
    return result


def _metadata_version(version):
    parsed = Version(version)
    if parsed.local or parsed.dev is not None or (parsed.pre and parsed.pre[0] != "rc"):
        return version

    result = _base(version)
    return result


def _requirement(raw):
    content, separator, comment = raw.partition("#")
    try:
        requirement = Requirement(content.strip())
    except Exception:
        return raw
    if canonicalize_name(requirement.name) not in FIRST_PARTY:
        return raw
    if requirement.url:
        raise ValueError("First-party release requirements cannot use URLs")
    requirement.specifier = SpecifierSet(
        ",".join(
            (
                f"{specifier.operator}{_metadata_version(specifier.version)}"
                if "*" not in specifier.version
                else str(specifier)
            )
            for specifier in requirement.specifier
        )
    )
    normalized = str(requirement)
    if separator:
        normalized += " #" + comment
    return normalized


def _normalize(value):
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        result = {key: _normalize(item) for key, item in value.items()}
        name = canonicalize_name(str(value.get("name", "")))
        if name in FIRST_PARTY:
            if "version" in result:
                result["version"] = _metadata_version(result["version"])
            if "specifier" in result:
                requirement = Requirement(_requirement(name + result["specifier"]))
                result["specifier"] = str(requirement.specifier)
            # Only first-party distribution identity changes between RC/final.
            result.pop("sdist", None)
            result.pop("wheels", None)
        return result
    if isinstance(value, str):
        return _requirement(value)
    return value


def _canonical(path, data):
    if path in LOCKS:
        return json.dumps(
            _normalize(tomllib.loads(data.decode())), sort_keys=True
        ).encode()
    if Path(path).name == "pyproject.toml":
        project_file = tomllib.loads(data.decode())
        project = project_file.get("project", {})
        if (
            canonicalize_name(project.get("name", "")) in FIRST_PARTY
            and "version" in project
        ):
            project["version"] = _metadata_version(project["version"])
        for key in ("dependencies", "optional-dependencies"):
            if key in project:
                project[key] = _normalize(project[key])
        if "dependency-groups" in project_file:
            project_file["dependency-groups"] = _normalize(
                project_file["dependency-groups"]
            )

        result = json.dumps(project_file, sort_keys=True).encode()
        return result
    if path == "inference/core/version.py":
        tree = ast.parse(data.decode())
        found = 0
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            ):
                node.value = ast.Constant(_base(ast.literal_eval(node.value)))
                found += 1
        if found != 1:
            raise ValueError("Expected exactly one release version assignment")
        return ast.dump(tree, include_attributes=False).encode()
    if path.startswith("requirements/") and path.endswith(".txt"):
        return "\n".join(
            _requirement(line) for line in data.decode().splitlines()
        ).encode()
    if path in {
        f"{directory}/CHANGELOG.md" for directory, _ in dependencies.PROJECTS.values()
    }:
        # Release notes may update an inline first-party pin from RC to final.
        # Keep all prose, third-party versions and actual dependency-floor changes.
        return re.sub(
            r"`([^`\n]+)`",
            lambda match: "`" + _requirement(match[1]) + "`",
            data.decode(),
        ).encode()
    return data


def _file_fingerprints(*, include_locks=True, ref=None):
    if ref is not None:
        entries = _tree_fingerprints(ref, include_locks=include_locks)
        return entries

    entries = []
    for line in _git("ls-files", "--stage", "-z").split("\0")[:-1]:
        metadata, path = line.split("\t", 1)
        mode, _, stage = metadata.split()
        if stage != "0" or mode == "160000":
            raise ValueError(f"Unmerged file or submodule in release source: {path}")
        if path in LOCKS and not include_locks:
            continue
        file = Path(path)
        data = str(file.readlink()).encode() if file.is_symlink() else file.read_bytes()
        entries.append((path, mode, hashlib.sha256(_canonical(path, data)).hexdigest()))
    return entries


def _tree_fingerprints(ref, *, include_locks=True):
    """Hash retained Git blobs without checking out or executing RC source."""
    files = []
    for entry in _git("ls-tree", "-rz", "--full-tree", ref).split("\0")[:-1]:
        metadata, path = entry.split("\t", 1)
        mode, kind, oid = metadata.split()
        if kind != "blob":
            raise ValueError(f"Unsupported object in RC source: {path}")
        if path not in LOCKS or include_locks:
            files.append((path, mode, oid))

    batch = subprocess.run(
        ["git", "cat-file", "--batch"],
        input="".join(oid + "\n" for _, _, oid in files).encode(),
        capture_output=True,
        check=True,
    )
    stream = io.BytesIO(batch.stdout)
    entries = []
    for path, mode, oid in files:
        actual, kind, size = stream.readline().decode().split()
        data = stream.read(int(size))
        if actual != oid or kind != "blob" or stream.read(1) != b"\n":
            raise ValueError(f"Invalid Git blob evidence: {path}")
        entries.append((path, mode, hashlib.sha256(_canonical(path, data)).hexdigest()))
    return entries


def _fingerprint(*, include_locks=True, ref=None):
    entries = _file_fingerprints(include_locks=include_locks, ref=ref)
    result = hashlib.sha256(json.dumps(entries).encode()).hexdigest()
    return result


def _source_difference(receipt):
    if "files" not in receipt:
        return "per-file evidence unavailable in this older RC receipt"

    previous = {path: (mode, digest) for path, mode, digest in receipt["files"]}
    current = {path: (mode, digest) for path, mode, digest in _file_fingerprints()}
    changed = sorted(
        path
        for path in previous.keys() | current.keys()
        if previous.get(path) != current.get(path)
    )
    result = ", ".join(changed) or "policy or receipt metadata"
    return result


def _third_party(lock):
    result = []
    for package in lock.get("package", []):
        if canonicalize_name(package["name"]) not in FIRST_PARTY:
            result.append(_normalize(package))
    return sorted(
        result,
        key=lambda item: (
            item["name"],
            item["version"],
            json.dumps(item, sort_keys=True),
        ),
    )


def _save_locks():
    _write(STATE / "locks.json", {path: Path(path).read_text() for path in LOCKS})


def _restore_locks():
    locks = json.loads((STATE / "locks.json").read_text())
    if set(locks) != set(LOCKS):
        raise ValueError("Unexpected lockfile snapshot inventory")
    for path, data in locks.items():
        tomllib.loads(data)
        Path(path).write_text(data)


def _restore_source():
    state = _state()
    if _git("rev-parse", "HEAD") not in {
        state["trigger_sha"],
        state["source_sha"],
    }:
        raise ValueError("Checkout is not the triggering or generated release commit")
    _restore_locks()
    if "source_tree" in state:
        _git("add", "--", *LOCKS)
        if _git("write-tree") != state["source_tree"] or _git("diff", "--name-only"):
            raise ValueError("Restored source differs from the committed release tree")


def _versions():
    versions = {"inference-sdk": release._source_version()}
    for name, (directory, _) in dependencies.PROJECTS.items():
        versions[name] = tomllib.loads(
            (Path(directory) / "pyproject.toml").read_text()
        )["project"]["version"]
    return versions


def _prepare_lock(package):
    directory, _ = dependencies.PROJECTS[package]
    path = Path(directory) / "uv.lock"
    previous = tomllib.loads(path.read_text())
    project = tomllib.loads((Path(directory) / "pyproject.toml").read_text())["project"]
    commands = ["uv", "lock"]
    versions = _versions()
    # Preserve third-party selections; refresh direct and transitive first-party pins.
    requirements = project.get("dependencies", []) + [
        raw
        for values in project.get("optional-dependencies", {}).values()
        for raw in values
    ]
    selected = {canonicalize_name(Requirement(raw).name) for raw in requirements}
    selected.update(
        canonicalize_name(item["name"]) for item in previous.get("package", [])
    )
    selected.discard(canonicalize_name(project["name"]))
    for name in sorted(selected & FIRST_PARTY):
        commands.extend(["--upgrade-package", f"{name}=={versions[name]}"])
    subprocess.run(commands, cwd=directory, check=True)
    current = tomllib.loads(path.read_text())
    for item in current.get("package", []):
        name = canonicalize_name(item["name"])
        if name in versions and Version(item["version"]) != Version(versions[name]):
            raise ValueError(
                f"{path}: first-party version differs from selected source: {name}"
            )
    if _third_party(previous) != _third_party(current):
        raise ValueError(f"{path}: unexpected third-party dependency change")
    subprocess.run(["uv", "lock", "--check"], cwd=directory, check=True)
    _save_locks()


class _ArtifactRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        if urlsplit(new_url).scheme != "https":
            raise ValueError("Artifact redirects must use HTTPS")
        redirected = super().redirect_request(
            request, response, code, message, headers, new_url
        )
        if redirected and urlsplit(request.full_url).netloc != urlsplit(new_url).netloc:
            redirected.remove_header("Authorization")
        return redirected


def _download_receipt(artifact_id):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip",
        headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}"},
    )
    opener = urllib.request.build_opener(_ArtifactRedirect())
    limit = 8 * 1024 * 1024
    try:
        with opener.open(request, timeout=_api_timeout()) as response:
            payload = response.read(limit + 1)
    except (OSError, http.client.HTTPException) as error:
        _api_timeout()
        raise urllib.error.URLError(error) from error
    _api_timeout()
    if len(payload) > limit:
        raise ValueError("RC receipt artifact is too large")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if archive.getinfo("receipt.json").file_size > limit:
                raise ValueError("Uncompressed RC receipt is too large")
            receipt = json.loads(archive.read("receipt.json"))
    except _EvidenceDeadlineExceeded:
        raise
    except (
        zipfile.BadZipFile,
        zlib.error,
        lzma.LZMAError,
        OSError,
        EOFError,
        RuntimeError,
    ) as error:
        raise ValueError(f"Invalid RC receipt archive: {error}") from error
    finally:
        _api_timeout()
    return receipt


def _evidence_ref(run_id):
    if not re.fullmatch(r"[1-9][0-9]*", str(run_id)):
        raise ValueError("Invalid RC run ID")

    ref = f"refs/tags/rc-evidence/{run_id}"
    return ref


def _retain_rc_source(source_sha, *, run_id, remote, env):
    """Keep the tested commit reachable after the release branch is deleted."""
    ref = _evidence_ref(run_id)
    existing = _git("ls-remote", "--refs", remote, ref, env=env)
    if existing:
        if existing != f"{source_sha}\t{ref}":
            raise ValueError("RC evidence tag already identifies a different commit")
    else:
        _git("push", remote, f"{source_sha}:{ref}", env=env)
    return ref


def _verify_rc_source(receipt):
    """Bind receipt claims to retained source and the workflow trigger."""
    ref = _evidence_ref(receipt["run_id"])
    try:
        _git(
            "fetch",
            "--no-tags",
            f"https://github.com/{REPOSITORY}.git",
            ref,
            timeout=_api_timeout(),
        )
    finally:
        _api_timeout()
    if _git("rev-parse", "FETCH_HEAD^{commit}") != receipt["source_sha"]:
        raise ValueError("RC evidence tag does not match the receipt source")

    _git("merge-base", "--is-ancestor", receipt["trigger_sha"], receipt["source_sha"])
    changed = _git(
        "diff",
        "--no-renames",
        "--name-only",
        receipt["trigger_sha"],
        receipt["source_sha"],
    ).splitlines()
    if not set(changed) <= set(LOCKS):
        raise ValueError(
            "RC source changed files other than generated locks after its trigger"
        )
    if _fingerprint(ref=receipt["source_sha"]) != receipt["fingerprint"]:
        raise ValueError("RC receipt fingerprint does not match its Git source")


def _validate_rc_receipt(receipt, *, run):
    if (
        receipt.get("schema") != 1
        or str(receipt.get("run_id")) != str(run["id"])
        or receipt.get("trigger_sha") != run["head_sha"]
        or receipt.get("repository") != REPOSITORY
        or receipt.get("complete") is not True
        or receipt.get("branch") != run["head_branch"]
        or receipt.get("evidence_ref") != _evidence_ref(run["id"])
        or not re.fullmatch(r"[a-f0-9]{40}", receipt.get("source_sha") or "")
        or not re.fullmatch(r"[a-f0-9]{64}", receipt.get("fingerprint") or "")
        or not re.fullmatch(r"[a-f0-9]{64}", receipt.get("application") or "")
    ):
        raise ValueError("Untrusted or incomplete RC receipt")

    policy = json.loads(POLICY.read_text())
    third_party = {
        path: _third_party(tomllib.loads(Path(path).read_text())) for path in LOCKS
    }
    if receipt.get("policy") != policy or receipt.get("third_party") != third_party:
        raise ValueError(
            "RC source/locks/policy differ at: " + _source_difference(receipt)
        )

    images = receipt.get("images")
    if not isinstance(images, list) or any(
        not isinstance(image, dict) for image in images
    ):
        raise ValueError("Invalid RC image receipts")

    by_key = {image["key"]: image for image in images}
    if len(by_key) != len(images) or set(by_key) != {
        item["key"] for item in policy["images"]
    }:
        raise ValueError("Missing or duplicate RC image receipts")

    smoke = receipt.get("smoke")
    expected_smoke = {f"{name}-py3.12" for name in dependencies.OWNED}
    expected_smoke |= {
        f"{name}-py{version}"
        for name in release.RUNTIME
        for version in policy["python_versions"]
    }
    if not isinstance(smoke, dict) or set(smoke) != expected_smoke:
        raise ValueError("Missing RC package smoke inventories")

    inventories = list(smoke.values())
    for item in policy["images"]:
        image = by_key[item["key"]]
        if (
            image.get("source_sha") != receipt["source_sha"]
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", image.get("digest") or "")
            or not isinstance(image.get("smoke"), dict)
            or set(image["smoke"]) != set(item["platforms"])
        ):
            raise ValueError("Invalid RC image digest, source or smoke platforms")
        inventories.extend(image["smoke"].values())

    for inventory in inventories:
        if (
            not isinstance(inventory, dict)
            or not inventory
            or any(
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
                or not isinstance(version, str)
                or not version
                for name, version in inventory.items()
            )
        ):
            raise ValueError("Invalid RC dependency inventory")
        for version in inventory.values():
            Version(version)


def _find_rc(version):
    fingerprint = _fingerprint()
    rejected = []
    for page in range(1, 6):
        runs = _api(
            "actions/workflows/publish.pypi.yml/runs"
            f"?event=workflow_dispatch&status=success&per_page=100&page={page}"
        )["workflow_runs"]
        for run in runs:
            if run["head_repository"]["full_name"] != REPOSITORY or not re.fullmatch(
                r"release(?:-candidates)?[-/][A-Za-z0-9._/-]+",
                run.get("head_branch") or "",
            ):
                continue
            artifacts = []
            for artifact_page in range(1, 21):
                batch = _api(
                    f"actions/runs/{run['id']}/artifacts?per_page=100&page={artifact_page}"
                )["artifacts"]
                artifacts.extend(batch)
                if len(batch) < 100:
                    break
            else:
                raise ValueError("RC artifact inventory exceeded the pagination limit")
            for artifact in sorted(
                artifacts, key=lambda item: item["id"], reverse=True
            ):
                if artifact["expired"] or not artifact["name"].startswith(
                    "release-receipt-"
                ):
                    continue
                try:
                    receipt = _download_receipt(artifact["id"])
                    if not isinstance(receipt, dict):
                        raise ValueError("RC receipt is not an object")
                    if receipt.get("channel") != "rc" or _base(
                        receipt["version"]
                    ) != _base(version):
                        continue
                    _validate_rc_receipt(receipt, run=run)
                    if receipt["fingerprint"] != fingerprint:
                        raise ValueError(
                            f"RC source differs at: {_source_difference(receipt)}"
                        )
                    _verify_rc_source(receipt)
                    if receipt["application"] != _fingerprint(include_locks=False):
                        raise ValueError(
                            "RC application metadata differs from its source"
                        )
                except subprocess.CalledProcessError as error:
                    rejected.append(
                        f"run {run['id']}: RC Git evidence unavailable or non-ancestral (exit {error.returncode})"
                    )
                    continue
                except (
                    subprocess.TimeoutExpired,
                    urllib.error.URLError,
                    TimeoutError,
                ) as error:
                    _api_timeout()
                    rejected.append(
                        f"run {run['id']}: RC evidence request failed: {error}"
                    )
                    continue
                except (ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
                    rejected.append(f"run {run['id']}: {error}")
                    continue

                _write(STATE / "accepted-rc.json", receipt)
                return receipt
        if len(runs) < 100:
            break
    detail = "; ".join(rejected[:5])
    raise ValueError(
        f"No successful equivalent RC receipt for {version}; complete a matching RC. {detail}"
    )


def _preflight():
    version = release._expected_version()
    if Version(version).is_prerelease:
        raise ValueError("Preflight requires prepared final versions")

    receipt = _find_rc(version)
    click.echo(f"Source and locks match successful RC run {receipt['run_id']}")


def _prepare():
    event = json.loads((STATE / "event.json").read_text())
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise ValueError("Release publication is restricted to roboflow/inference")
    sha = _git("rev-parse", "HEAD")
    version = release._expected_version()
    _base(version)
    channel = "preview"
    if os.environ.get("GITHUB_EVENT_NAME") == "release":
        item = event["release"]
        if event.get("action") != "published" or item["draft"] or item["prerelease"]:
            raise ValueError("Expected a published, non-draft, stable GitHub Release")
        component = r"(?:0|[1-9][0-9]*)"
        if not re.fullmatch(
            rf"v{component}\.{component}\.{component}(?:(?:-|\.)post{component})?",
            item["tag_name"],
        ):
            raise ValueError("Stable tag must be vX.Y.Z, vX.Y.Z-postN or vX.Y.Z.postN")
        if Version(version).is_prerelease or Version(item["tag_name"]) != Version(
            version
        ):
            raise ValueError("Stable tag and package version must match")
        if _git("rev-parse", f"refs/tags/{item['tag_name']}^{{commit}}") != sha:
            raise ValueError("Checkout is not the published tag")
        channel = "stable"
    elif os.environ.get("PUBLISH") == "true":
        if os.environ.get("PACKAGES", "all") == "sdk":
            channel = "sdk"
        else:
            if not Version(version).pre or Version(version).pre[0] != "rc":
                raise ValueError(
                    "Manual runtime publication requires an RC; publish a GitHub Release for stable"
                )
            ref = os.environ.get("GITHUB_REF", "")
            if not re.fullmatch(
                r"refs/heads/release(?:-candidates)?[-/][A-Za-z0-9._/-]+", ref
            ):
                raise ValueError(
                    "RC publication requires a release or release-candidates branch"
                )
            channel = "rc"
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Release source must start clean")
    state = {
        "schema": 1,
        "repository": REPOSITORY,
        "channel": channel,
        "version": version,
        "image_version": (
            event["release"]["tag_name"].removeprefix("v")
            if channel == "stable"
            else version
        ),
        "source_sha": sha,
        "trigger_sha": os.environ["GITHUB_SHA"],
        "branch": os.environ.get("GITHUB_REF", "").removeprefix("refs/heads/"),
        "run_id": os.environ["GITHUB_RUN_ID"],
        "application": _fingerprint(include_locks=False),
        "policy": json.loads(POLICY.read_text()),
        "third_party": {
            path: _third_party(tomllib.loads(Path(path).read_text())) for path in LOCKS
        },
    }
    if channel == "stable":
        receipt = _find_rc(version)
        if (
            receipt["application"] != state["application"]
            or receipt["policy"] != state["policy"]
            or receipt["third_party"] != state["third_party"]
            or receipt["fingerprint"] != _fingerprint()
        ):
            raise ValueError(
                "Stable source/locks/policy differ from RC; complete a new RC: "
                + _source_difference(receipt)
            )
        state["accepted_rc_run_id"] = receipt["run_id"]
    _write(STATE / "release.json", state)
    _save_locks()
    _outputs(
        channel=channel,
        version=version,
        source_sha=sha,
        images=json.dumps({"include": state["policy"]["images"]}),
    )


def _commit_locks():
    state = _state()
    if state["channel"] != "rc":
        _outputs(source_sha=state["source_sha"])
        return
    _restore_locks()
    if _fingerprint(include_locks=False) != state["application"]:
        raise ValueError("Application changed during lock preparation")
    paths = _git("diff", "--name-only").splitlines()
    if not set(paths) <= set(LOCKS) or _git("diff", "--cached", "--name-only"):
        raise ValueError("Only generated lockfiles may be committed")
    branch = state["branch"]
    if not re.fullmatch(r"release(?:-candidates)?[-/][A-Za-z0-9._/-]+", branch):
        raise ValueError("Not a release branch")
    auth = base64.b64encode(
        f"x-access-token:{os.environ['GH_TOKEN']}".encode()
    ).decode()
    env = {
        **os.environ,
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth}",
        "GIT_CONFIG_KEY_1": "safe.directory",
        "GIT_CONFIG_VALUE_1": str(Path.cwd()),
    }
    remote = f"https://github.com/{REPOSITORY}.git"
    _git("fetch", "--no-tags", remote, f"refs/heads/{branch}", env=env)
    remote_sha = _git("rev-parse", "FETCH_HEAD")
    message = f"ci(release): update locks for {state['version']} [skip ci]\n\nRelease-Run: {state['run_id']}"
    if paths:
        _git("add", "--", *paths)
    tree = _git("write-tree")
    if remote_sha != state["source_sha"]:
        if (
            _git("show", "-s", "--format=%P", remote_sha) != state["source_sha"]
            or _git("show", "-s", "--format=%B", remote_sha) != message
            or _git("rev-parse", f"{remote_sha}^{{tree}}") != tree
        ):
            raise ValueError("Release branch changed concurrently; no push performed")
        source_sha = remote_sha
    elif paths:
        _git(
            "-c",
            "user.name=github-actions[bot]",
            "-c",
            "user.email=41898282+github-actions[bot]@users.noreply.github.com",
            "commit",
            "-m",
            message,
        )
        source_sha = _git("rev-parse", "HEAD")
        _git("push", remote, f"HEAD:refs/heads/{branch}", env=env)
    else:
        source_sha = remote_sha
    state["source_sha"] = source_sha
    state["source_tree"] = tree
    state["evidence_ref"] = _retain_rc_source(
        source_sha, run_id=state["run_id"], remote=remote, env=env
    )
    _write(STATE / "release.json", state)
    _outputs(source_sha=source_sha)


def _equivalence():
    state = _state()
    _restore_locks()
    if state["channel"] == "stable":
        receipt = json.loads((STATE / "accepted-rc.json").read_text())
        if _fingerprint() != receipt["fingerprint"]:
            raise ValueError(
                "Prepared stable locks/dependencies differ from the tested RC: "
                + _source_difference(receipt)
            )


def _build_package(package):
    _restore_locks()
    _prepare_lock(package)
    directory, _ = dependencies.PROJECTS[package]
    destination = (STATE / "packages" / package).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if package == "roboflow-workflows":
        subprocess.run(["make", "download_fonts"], check=True)
    command = ["uv", "build", "--no-create-gitignore", "--out-dir", str(destination)]
    if package == "streamvision":
        command.append("--wheel")
    subprocess.run(command, cwd=directory, check=True)
    subprocess.run(["python", ".release/pypi/smoke.py", str(destination)], check=True)


def _check_upload(package):
    state = _state()
    prepared = json.loads((STATE / f"{package}-prepare.json").read_text())
    if (
        state["channel"] == "rc"
        and not Version(_versions()[package]).is_prerelease
        and prepared["upload"]
    ):
        raise ValueError(
            f"RC may reuse an existing stable {package}, but cannot publish a new stable version"
        )


def _receipt():
    state = _state()
    _restore_locks()
    if _fingerprint(include_locks=False) != state["application"]:
        raise ValueError("Application changed after preparation")
    images_by_key = {}
    for path in sorted((STATE / "images").rglob("image-receipt.json")):
        image = json.loads(path.read_text())
        previous = images_by_key.get(image["key"])
        if previous is not None and previous != image:
            raise ValueError("Conflicting image receipts across attempts")
        images_by_key[image["key"]] = image
    images = list(images_by_key.values())
    expected = {image["key"] for image in state["policy"]["images"]}
    if set(images_by_key) != expected or any(
        image["source_sha"] != state["source_sha"] for image in images
    ):
        raise ValueError("Missing or mismatched image publication receipts")
    smoke = {}
    for directory in (STATE / "smoke", STATE / "runtime-smoke"):
        for path in directory.rglob("*.json"):
            inventory = json.loads(path.read_text())
            if path.stem in smoke and smoke[path.stem] != inventory:
                raise ValueError(
                    "Conflicting package smoke inventories across attempts"
                )
            smoke[path.stem] = inventory
    expected_smoke = {f"{name}-py3.12" for name in dependencies.OWNED}
    expected_smoke |= {
        f"{name}-py{version}"
        for name in release.RUNTIME
        for version in state["policy"]["python_versions"]
    }
    if set(smoke) != expected_smoke:
        raise ValueError("Missing package smoke inventories")
    receipt = {
        **state,
        "fingerprint": _fingerprint(),
        "files": _file_fingerprints(),
        "complete": True,
        "packages": json.loads((STATE / "dependency-receipt.json").read_text()),
        "runtime": json.loads((STATE / "runtime-receipt.json").read_text()),
        "images": images,
        "smoke": smoke,
    }
    _write(STATE / "receipt.json", receipt)


def _validate():
    state = _state()
    results = json.loads(os.environ["RELEASE_RESULTS"])
    expected = {
        "rc_" + name.removesuffix(".yml").replace(".", "_")
        for name in state["policy"]["rc_tests"]
    }
    required = "success" if state["channel"] == "rc" else "skipped"
    failed = sorted(
        name for name in expected if results.get(name, {}).get("result") != required
    )
    if results.get("smoke", {}).get("result") != "success":
        failed.append("smoke")
    if state["channel"] == "rc":
        jobs = {}
        for page in range(1, 21):
            batch = _api(
                f"actions/runs/{state['run_id']}/jobs?filter=all&per_page=100&page={page}"
            )["jobs"]
            for job in batch:
                if job["id"] > jobs.get(job["name"], {}).get("id", -1):
                    jobs[job["name"]] = job
            if len(batch) < 100:
                break
        else:
            raise ValueError("RC job inventory exceeded the pagination limit")
        for name in expected:
            suite = [
                job for job in jobs.values() if job["name"].startswith(name + " /")
            ]
            tested = [
                step
                for job in suite
                for step in job.get("steps", [])
                if step["name"].endswith("[release-test]")
            ]
            if (
                not suite
                or any(job["conclusion"] != "success" for job in suite)
                or not tested
                or any(step["conclusion"] != "success" for step in tested)
            ):
                failed.append(name)
        # The run was triggered at the pre-lock SHA. Explicitly report its real
        # results against the generated commit; this creates no workflow run.
        _api(
            "check-runs",
            payload={
                "name": "Release RC validation",
                "head_sha": state["source_sha"],
                "status": "completed",
                "conclusion": "failure" if failed else "success",
                "details_url": f"https://github.com/{REPOSITORY}/actions/runs/{state['run_id']}",
                "output": {
                    "title": "Release RC validation",
                    "summary": (
                        "Failed: " + ", ".join(sorted(set(failed)))
                        if failed
                        else "All required RC suites and retained-wheel smoke tests passed in this run."
                    ),
                },
            },
        )
    if failed:
        raise ValueError("Release validation failed: " + ", ".join(sorted(set(failed))))


@click.command()
@click.argument(
    "operation",
    type=click.Choice(
        [
            "prepare",
            "preflight",
            "restore",
            "lock",
            "build",
            "check-upload",
            "commit-locks",
            "equivalence",
            "validate",
            "receipt",
        ]
    ),
)
@click.option(
    "--package",
    type=click.Choice(sorted(dependencies.PROJECTS)),
)
def _main(operation, *, package):
    STATE.mkdir(exist_ok=True)
    if operation == "prepare":
        _prepare()
    elif operation == "preflight":
        _preflight()
    elif operation == "restore":
        _restore_source()
    elif operation in {"lock", "build", "check-upload"}:
        if not package:
            raise click.ClickException(f"{operation} requires --package")
        if operation == "build":
            _build_package(package)
        elif operation == "check-upload":
            _check_upload(package)
        else:
            _restore_locks()
            _prepare_lock(package)
    elif operation == "commit-locks":
        _commit_locks()
    elif operation == "equivalence":
        _equivalence()
    elif operation == "receipt":
        _receipt()
    elif operation == "validate":
        _validate()


if __name__ == "__main__":
    _main()
