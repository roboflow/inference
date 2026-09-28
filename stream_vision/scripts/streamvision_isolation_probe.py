"""Prove the installed `streamvision` wheel works without the `inference` package.

Creates a throwaway venv outside the checkout, installs the pre-built wheel with
its declared dependencies (local sibling wheels come from `--find-links`), then
runs checks in a child interpreter that refuses every `inference` /
`inference.*` import. The child prints one JSON list of results; the parent
never reads a missing or malformed result as success.
"""

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

LOCAL_WHEELS = ("roboflow_workflows", "inference_models", "inference_sdk")
STATUSES = ("ok", "failed", "skipped")
BASE_CHECKS = (
    "distribution_absent",
    "import_everything",
    "origins_inside_venv",
    "entities_import_light",
    "configure_process",
    "manager_roundtrip",
)
WEBRTC_CHECKS = ("webrtc",)
WEBRTC_MODULES = frozenset({"aiortc", "av"})
# The only streamvision modules that legitimately import aiortc/av at module level.
WEBRTC_IMPORT_ALLOWLIST = frozenset(
    {
        "streamvision.stream_manager.manager_app.webrtc",
        "streamvision.stream_manager.manager_app.inference_pipeline_manager",
        "streamvision.stream_manager.manager_app.app",
    }
)


def classify_import_error(module_name: str, missing_module: str, webrtc: bool) -> str:
    """Classify a module's `ModuleNotFoundError` as skippable or a failure.

    Args:
        module_name: The streamvision module whose import failed.
        missing_module: Root name of the module that could not be found.
        webrtc: Whether the `[webrtc]` extra is installed and checked.

    Returns:
        "skipped" when `webrtc` is absent, `missing_module` is a WebRTC
        dependency, and `module_name` is allowed to need it; else "failed".
    """
    if webrtc or missing_module not in WEBRTC_MODULES:
        return "failed"

    return "skipped" if module_name in WEBRTC_IMPORT_ALLOWLIST else "failed"


BLOCKER = r"""
import sys


class ServerImportBlocker:
    attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "inference" or fullname.startswith("inference."):
            self.attempted.append(fullname)
            raise ModuleNotFoundError(f"blocked: {fullname}", name=fullname)
        return None


sys.meta_path.insert(0, ServerImportBlocker())
"""

FAKE_HOST = r"""
def create_host(**settings):
    raise RuntimeError("the isolation probe never starts a pipeline")
"""

CHILD = (
    BLOCKER
    + f"BLOCKER_SOURCE = {BLOCKER!r}\n"
    + f"WEBRTC_MODULES = {WEBRTC_MODULES!r}\n"
    + f"WEBRTC_IMPORT_ALLOWLIST = {WEBRTC_IMPORT_ALLOWLIST!r}\n"
    + inspect.getsource(classify_import_error)
    + r"""
import asyncio, importlib, importlib.machinery, importlib.metadata, json, os
import pkgutil, signal, socket, subprocess, time, traceback

if sys.flags.optimize:
    print(json.dumps([{"check": "no_optimize", "status": "failed",
                       "detail": "child ran with assertions stripped"}]))
    sys.exit(1)

SCRATCH, HOST_DIR, WEBRTC = sys.argv[1], sys.argv[2], sys.argv[3] == "on"
EXPECTED_CHECKS = json.loads(sys.argv[4])
PREFIX = os.path.realpath(sys.prefix) + os.sep
results = []


class Skipped(Exception):
    pass


def check(name, fn):
    try:
        fn()
        results.append({"check": name, "status": "ok"})
    except Skipped as reason:
        results.append({"check": name, "status": "skipped", "detail": str(reason)})
    except BaseException:  # noqa: BLE001 - report every failure, keep going
        results.append({"check": name, "status": "failed",
                        "detail": traceback.format_exc()})


def fresh(code):
    # A new interpreter, so earlier checks cannot have pre-imported anything.
    proc = subprocess.run([sys.executable, "-c", BLOCKER_SOURCE + code],
                          cwd=SCRATCH, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def distribution_absent():
    for name in ("inference", "inference-core", "inference-gpu", "inference-cpu"):
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise AssertionError(f"distribution {name} is installed")
    assert importlib.machinery.PathFinder.find_spec("inference") is None
    try:
        import inference  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("import inference succeeded")
    ServerImportBlocker.attempted.clear()


def import_everything():
    import streamvision
    infos = list(pkgutil.walk_packages(streamvision.__path__, "streamvision."))
    assert infos, "streamvision has no submodules"
    failures, skipped = {}, {}
    for info in infos:
        try:
            importlib.import_module(info.name)
        except ModuleNotFoundError as error:
            root = (error.name or "").split(".")[0]
            if classify_import_error(info.name, root, WEBRTC) == "skipped":
                skipped[info.name] = error.name
            else:
                failures[info.name] = repr(error)
        except Exception as error:  # noqa: BLE001
            failures[info.name] = repr(error)
    assert not failures, json.dumps({"import_failures": failures}, indent=1)
    attempted = sorted(set(ServerImportBlocker.attempted))
    assert not attempted, json.dumps({"blocked_import_attempts": attempted})
    if skipped:
        raise Skipped(json.dumps({"missing_optional_dependency": skipped}))


def origins_inside_venv():
    outside = sorted(
        name for name, module in list(sys.modules.items())
        if name.split(".")[0] in ("streamvision", "roboflow_workflows")
        and not os.path.realpath(getattr(module, "__file__", None) or "")
        .startswith(PREFIX)
    )
    assert not outside, json.dumps({"outside_venv": outside}, indent=1)
    assert any(name.startswith("streamvision.") for name in sys.modules)


def entities_import_light():
    heavy = fresh(
        "import json, sys\n"
        "import streamvision.stream_manager.api.stream_manager_client\n"
        "import streamvision.stream_manager.manager_app.entities\n"
        "heavy = ('cv2', 'aiortc', 'av', 'streamvision.camera.video_source',"
        " 'streamvision.stream.pipeline')\n"
        "print(json.dumps(sorted(m for m in heavy if m in sys.modules)))\n"
    )
    assert heavy == [], heavy


def configure_process():
    same = fresh(
        "import json\n"
        "from streamvision.stream.configuration import (\n"
        "    StreamsConfiguration, configure_process, get_configuration)\n"
        "configuration = StreamsConfiguration()\n"
        "configure_process(configuration)\n"
        "print(json.dumps(get_configuration() is configuration))\n"
    )
    assert same is True, same


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def manager_roundtrip():
    if not WEBRTC:
        raise Skipped("needs --webrtc")
    from streamvision.stream_manager.api.stream_manager_client import (
        StreamManagerClient,
    )
    port = _free_port()
    env = {**os.environ, "PYTHONPATH": HOST_DIR, "STREAM_MANAGER_HOST": "127.0.0.1",
           "STREAM_MANAGER_PORT": str(port)}
    manager = subprocess.Popen(
        [sys.executable, "-m", "streamvision", "--host-factory",
         "probe_fake_host:create_host"],
        cwd=SCRATCH, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            assert manager.poll() is None, manager.communicate()[1][-4000:]
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                assert time.monotonic() < deadline, "manager never listened"
                time.sleep(0.2)
        client = StreamManagerClient.init(host="127.0.0.1", port=port,
                                          operations_timeout=10)
        response = asyncio.run(client.list_pipelines())
        assert response.status == "success", response
        assert response.pipelines == [], response
        manager.send_signal(signal.SIGTERM)
        _, stderr = manager.communicate(timeout=15)
        assert manager.returncode == 0, (manager.returncode, stderr[-4000:])
        assert "blocked: inference" not in stderr, stderr[-4000:]
    finally:
        if manager.poll() is None:
            manager.kill()
            manager.communicate()


def webrtc():
    import aiortc  # noqa: F401
    import streamvision.stream_manager.manager_app.webrtc  # noqa: F401


CHECKS = {
    "distribution_absent": distribution_absent,
    "import_everything": import_everything,
    "origins_inside_venv": origins_inside_venv,
    "entities_import_light": entities_import_light,
    "configure_process": configure_process,
    "manager_roundtrip": manager_roundtrip,
    "webrtc": webrtc,
}
for name in EXPECTED_CHECKS:
    if name not in CHECKS:
        results.append({"check": name, "status": "failed",
                        "detail": "the child has no implementation for this check"})
        continue
    check(name, CHECKS[name])
print(json.dumps(results))
"""
)


def expected_checks(*, webrtc: bool) -> List[str]:
    """Return the check names the child must report, in order.

    Args:
        webrtc: Whether the `[webrtc]` extra is installed and checked.

    Returns:
        Check names.
    """
    checks = [*BASE_CHECKS, *(WEBRTC_CHECKS if webrtc else ())]

    return checks


def _child_diagnostics(proc: subprocess.CompletedProcess) -> str:
    return (
        f"exit code {proc.returncode}\n"
        f"--- child stderr (tail) ---\n{proc.stderr[-4000:]}\n"
        f"--- child stdout (tail) ---\n{proc.stdout[-2000:]}"
    )


def _validate_results(
    raw: list, proc: subprocess.CompletedProcess, *, expected: List[str]
) -> list:
    results, problems, seen = [], [], set()
    for entry in raw:
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("check"), str)
            and entry.get("status") in STATUSES
        ):
            results.append(entry)
            seen.add(entry["check"])
        else:
            problems.append(f"malformed result entry: {entry!r}")
    missing = [name for name in expected if name not in seen]
    if missing:
        problems.append(f"checks missing from the child's results: {missing}")
    unexpected = sorted(seen - set(expected))
    if unexpected:
        problems.append(f"unexpected checks in the child's results: {unexpected}")
    if problems:
        results.append(
            {
                "check": "results_complete",
                "status": "failed",
                "detail": "\n".join(problems) + "\n" + _child_diagnostics(proc),
            }
        )

    return results


def _child_env() -> dict:
    env = {**os.environ, "PYTHONOPTIMIZE": "0", "PYTHONNOUSERSITE": "1"}
    for variable in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(variable, None)
    for variable in list(env):
        if variable.startswith("STREAM_MANAGER_"):
            env.pop(variable)

    return env


def _write_host_dir(host_dir: Path) -> None:
    host_dir.mkdir(parents=True, exist_ok=True)
    (host_dir / "probe_fake_host.py").write_text(FAKE_HOST)
    # Loaded at startup of the manager subprocess, so it runs under the blocker too.
    (host_dir / "sitecustomize.py").write_text(BLOCKER)


def build_venv(
    venv_dir: Path,
    wheel: Path,
    *,
    find_links: List[str],
    python: str,
    webrtc: bool,
) -> Path:
    """Create a venv outside the checkout and install the wheel with its deps.

    Args:
        venv_dir: Directory of the new venv.
        wheel: The `streamvision` wheel.
        find_links: Local directories searched for sibling wheels.
        python: Interpreter the venv is created from.
        webrtc: Whether to install the `[webrtc]` extra.

    Returns:
        The venv's interpreter.

    Raises:
        ValueError: A find-links directory holds more than one wheel of a package.
    """
    # A checkout on sys.path would let pip report its egg-info as already installed.
    env = _child_env()
    subprocess.run(
        [python, "-m", "venv", "--clear", str(venv_dir)],
        check=True,
        cwd=venv_dir.parent,
        env=env,
    )
    venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
        capture_output=True,
        cwd=venv_dir.parent,
        env=env,
    )

    requirement = f"{wheel.resolve()}[webrtc]" if webrtc else str(wheel.resolve())
    install = [str(venv_python), "-m", "pip", "install", requirement]
    for link in find_links:
        install += ["--find-links", str(link)]
        # Explicit local wheels win over equal-version index candidates.
        for package in LOCAL_WHEELS:
            candidates = sorted(Path(link).glob(f"{package}-*.whl"))
            if len(candidates) > 1:
                raise ValueError(f"Multiple {package} wheels in {link}: {candidates}")
            suffix = "[enterprise]" if package == "roboflow_workflows" else ""
            install.extend(f"{candidate.resolve()}{suffix}" for candidate in candidates)
    subprocess.run(install, check=True, cwd=venv_dir.parent, env=env)

    return venv_python


def run_probe(python: Path, scratch: Path, *, webrtc: bool) -> list:
    """Run the child checks with `python` and return validated results.

    Args:
        python: Interpreter of the probe venv.
        scratch: Scratch directory outside the checkout.
        webrtc: Whether to run the WebRTC checks.

    Returns:
        One `{"check", "status", "detail"?}` dict per check, plus failures for
        a missing or malformed result set.
    """
    expected = expected_checks(webrtc=webrtc)
    host_dir = scratch / "host"
    _write_host_dir(host_dir)
    proc = subprocess.run(
        [
            str(python),
            "-c",
            CHILD,
            str(scratch),
            str(host_dir),
            "on" if webrtc else "off",
            json.dumps(expected),
        ],
        cwd=scratch,
        env=_child_env(),
        capture_output=True,
        text=True,
    )
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        raw = json.loads(line)
        if not isinstance(raw, list):
            raise ValueError(f"results line is {type(raw).__name__}, not a list")
    except ValueError as error:
        return [
            {
                "check": "child_process",
                "status": "failed",
                "detail": f"no/invalid results line ({error})\n{_child_diagnostics(proc)}",
            }
        ]

    results = _validate_results(raw, proc, expected=expected)
    if proc.returncode != 0:
        results.append(
            {
                "check": "child_process",
                "status": "failed",
                "detail": _child_diagnostics(proc),
            }
        )

    return results


def _print_table(results: list) -> None:
    width = max(len(result["check"]) for result in results)
    for result in results:
        print(f"{result['check']:<{width}}  {result['status']}")
    for result in results:
        if result["status"] != "ok":
            print(f"\n--- {result['check']} ({result['status']}) ---")
            print(result.get("detail", "<no detail reported>"))


def main() -> int:
    """Build the probe venv, run the checks and print the result table.

    Returns:
        Process exit code: 1 when any check failed, else 0.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wheel",
        type=Path,
        required=True,
        help="Path to the streamvision wheel to probe.",
    )
    parser.add_argument(
        "--find-links",
        action="append",
        default=[],
        help="Directory with local sibling wheels for pip (repeatable).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter the probe venv is created from.",
    )
    parser.add_argument(
        "--keep-venv",
        action="store_true",
        help="Keep the scratch directory and venv after the run.",
    )
    parser.add_argument(
        "--webrtc",
        action="store_true",
        help="Install the [webrtc] extra and run the WebRTC checks.",
    )
    args = parser.parse_args()
    if not args.wheel.is_file():
        parser.error(f"wheel not found: {args.wheel}")

    # pip runs with a different cwd, so relative find-links must be resolved here.
    find_links = [str(Path(link).resolve()) for link in args.find_links]
    env_links = os.environ.get("STREAMVISION_ISOLATION_FIND_LINKS", "")
    find_links.extend(
        str(Path(part).resolve()) for part in env_links.split(os.pathsep) if part
    )

    scratch = Path(tempfile.mkdtemp(prefix="streamvision-isolation-"))
    try:
        python = build_venv(
            scratch / "venv",
            args.wheel,
            find_links=find_links,
            python=args.python,
            webrtc=args.webrtc,
        )
        results = run_probe(python, scratch, webrtc=args.webrtc)
    finally:
        if args.keep_venv:
            print(f"venv kept at {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)

    _print_table(results)
    failed = any(result["status"] == "failed" for result in results)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
