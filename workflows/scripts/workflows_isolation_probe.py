"""Prove the installed `roboflow-workflows` wheel works standalone.

Creates a throwaway venv OUTSIDE the checkout, installs the pre-built wheel
plus its declared test dependencies (nothing from this checkout is added to
the child's sys.path or PYTHONPATH), then runs eight checks in a child
interpreter that (a) starts in that scratch directory and (b) installs a
meta-path blocker refusing every `inference` / `inference.*` import. The
blocker is load-bearing: it prevents an accidentally-installed server
package or an editable finder from silently satisfying imports the moved
package must never make.

SDK and inference-models are resolved as ordinary declared dependencies of
the roboflow-workflows wheel (or pulled in via --find-links pointing at
locally built wheels for CI); they are NOT injected from the checkout.

Font assets: the wheel already contains fonts (packaged data). One check
verifies offline rendering works against the installed copy, then removes
the font assets from a DISPOSABLE per-run installation to prove the missing-
asset error is surfaced. The developer's shared installation is never
touched: every run uses its own venv or --install-dir.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

PACKAGE = "roboflow_workflows"
ENTERPRISE_LOADER = "roboflow_workflows.enterprise_blocks.loader"

CHILD = r"""
import importlib, json, logging, os, pkgutil, shutil, sys, traceback

if sys.flags.optimize:
    print(json.dumps([{
        "check": "no_optimize",
        "status": "fail",
        "detail": "child ran with optimization enabled; assert statements are stripped",
    }]))
    sys.exit(1)

SCRATCH, TENSOR_MODE = sys.argv[1], sys.argv[2] == "on"
EXPECTED_CHECKS = json.loads(sys.argv[3])
PACKAGE = "roboflow_workflows"
results = []


class ServerImportBlocker:
    # First on sys.meta_path: refuses every `inference` / `inference.*` import
    # and records the attempt, so a `try: import ... except ImportError` inside
    # the moved package is still reported.
    attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "inference" or fullname.startswith("inference."):
            self.attempted.append(fullname)
            raise ImportError(f"blocked: {fullname} is the inference server package")
        return None


sys.meta_path.insert(0, ServerImportBlocker())

# Standalone: this probe IS the host. Configure BEFORE the first
# roboflow_workflows import, because core_steps/loader.py branches on tensor
# at import time and environment.py freezes every constant at its own import.
import dataclasses as _dc
from roboflow_workflows.configuration import (
    configure_process,
    default_configuration,
    resolve_image_tensor_device,
)

_BASE = default_configuration()
configure_process(_dc.replace(
    _BASE,
    tensor=_dc.replace(
        _BASE.tensor,
        representation_enabled=TENSOR_MODE,
        image_tensor_device=resolve_image_tensor_device(TENSOR_MODE),
    ),
    engine=_dc.replace(
        _BASE.engine,
        allow_custom_python_execution=True,
        custom_python_execution_mode="local",
    ),
    fonts=_dc.replace(
        _BASE.fonts,
        allow_download=False,
        model_cache_dir=os.path.join(SCRATCH, "cache"),
    ),
))


def check(name, fn):
    try:
        fn()
        results.append({"check": name, "status": "ok"})
    except BaseException:  # noqa: BLE001 - report every failure, keep going
        results.append({"check": name, "status": "fail", "detail": traceback.format_exc()})


def _package_dir(module):
    file = getattr(module, "__file__", None)
    assert file, f"{module.__name__} has no __file__ (namespace pkg?)"
    return os.path.realpath(os.path.dirname(file))


import roboflow_workflows as _rw_probe  # noqa: E402 - probe of the installed wheel
INSTALL_ROOT = _package_dir(_rw_probe)


def _from_installed_wheel(module):
    file = getattr(module, "__file__", None)
    if not file:
        return False
    return os.path.realpath(file).startswith(INSTALL_ROOT + os.sep) or os.path.realpath(file) == os.path.join(INSTALL_ROOT, "__init__.py")


def assert_only_installed_wheel_loaded():
    # No server module may have been imported; every roboflow_workflows.*
    # module must resolve inside the installed wheel directory.
    foreign = sorted(
        name for name, module in sys.modules.items()
        if (name == PACKAGE or name.startswith(PACKAGE + "."))
        and not _from_installed_wheel(module)
    )
    server = sorted(
        name for name in sys.modules
        if name == "inference" or name.startswith("inference.")
    )
    assert not foreign and not server, json.dumps(
        {"not_from_installed_wheel": foreign, "server_modules_loaded": server},
        indent=1,
    )


def import_everything():
    import roboflow_workflows as workflows
    assert _from_installed_wheel(workflows), workflows.__file__
    # nonempty-source guard: an empty package would make every subsequent
    # check trivially pass. Fail loudly instead of walking nothing.
    module_infos = list(
        pkgutil.walk_packages(workflows.__path__, PACKAGE + ".", onerror=lambda name: None)
    )
    assert module_infos, "roboflow_workflows package has no submodules"
    failures = {}
    for info in module_infos:
        try:
            importlib.import_module(info.name)
        except Exception as error:  # noqa: BLE001
            failures[info.name] = repr(error)
    assert not failures, json.dumps({"import_failures": failures}, indent=1)
    # Enterprise plugin: import the loader (activated via WORKFLOWS_PLUGINS
    # in the child env) and confirm every declared enterprise block imports
    # cleanly. Still guarded by the meta-path server blocker.
    from roboflow_workflows.enterprise_blocks.loader import load_enterprise_blocks
    enterprise = load_enterprise_blocks()
    assert enterprise, "enterprise loader returned no blocks"
    assert_only_installed_wheel_loaded()


def load_blocks():
    from roboflow_workflows.execution_engine.introspection.blocks_loader import (
        describe_available_blocks, load_workflow_blocks,
    )
    loaded = load_workflow_blocks()
    identifiers = {b.identifier for b in loaded}
    assert identifiers, "no blocks loaded"
    # Public block identifiers deliberately keep their pre-extraction names.
    tensor_blur = "inference.core.workflows.core_steps.classical_cv.image_blur.v1_tensor.ImageBlurBlockV1"
    assert (tensor_blur in identifiers) == TENSOR_MODE, (
        TENSOR_MODE, sorted(i for i in identifiers if "image_blur" in i)
    )
    # WORKFLOWS_PLUGINS in the child env pointed at
    # roboflow_workflows.enterprise_blocks.loader, so every enterprise block
    # must be in the loaded set with source == BLOCKS_SOURCE.
    from roboflow_workflows.enterprise_blocks.loader import (
        BLOCKS_SOURCE,
        load_enterprise_blocks,
    )
    # load_enterprise_blocks() returns block CLASSES; loaded specs expose
    # block_class and block_source. Compare on class identity, not identifier.
    enterprise_classes = set(load_enterprise_blocks())
    by_class = {spec.block_class: spec for spec in loaded}
    missing = sorted(
        cls.__name__ for cls in enterprise_classes if cls not in by_class
    )
    assert not missing, json.dumps({"missing_enterprise_blocks": missing}, indent=1)
    wrong_source = sorted(
        cls.__name__ for cls in enterprise_classes
        if getattr(by_class[cls], "block_source", None) != BLOCKS_SOURCE
    )
    assert not wrong_source, json.dumps(
        {"enterprise_blocks_with_wrong_source": wrong_source}, indent=1
    )
    describe_available_blocks(dynamic_blocks=[]).model_dump_json()


def _run(definition, **runtime):
    from roboflow_workflows.execution_engine.core import ExecutionEngine
    engine = ExecutionEngine.init(workflow_definition=definition, init_parameters={})
    return engine.run(runtime_parameters=runtime, serialize_results=True)


def _image_input():
    if TENSOR_MODE:
        import torch
        return torch.zeros((3, 64, 64), dtype=torch.uint8)
    import numpy as np
    return np.zeros((64, 64, 3), dtype=np.uint8)


MODEL_FREE_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowBatchInput", "name": "detections",
         "kind": ["object_detection_prediction"]},
    ],
    "steps": [
        {"type": "roboflow_core/image_blur@v1", "name": "blur",
         "image": "$inputs.image", "blur_type": "gaussian", "kernel_size": 5},
        {"type": "roboflow_core/rich_label_visualization@v1", "name": "label",
         "image": "$steps.blur.image", "predictions": "$inputs.detections",
         "text": "Class", "font_family": "Geist Mono"},
    ],
    "outputs": [
        {"type": "JsonField", "name": "blurred", "selector": "$steps.blur.image"},
        {"type": "JsonField", "name": "labelled", "selector": "$steps.label.image"},
    ],
}
SYNTHETIC_DETECTIONS = {
    "image": {"width": 64, "height": 64},
    "predictions": [{"x": 32, "y": 32, "width": 20, "height": 20,
                     "confidence": 0.9, "class": "isolation", "class_id": 0}],
}


def run_model_free_workflow():
    from roboflow_workflows.execution_engine.core import ExecutionEngine
    engine = ExecutionEngine.init(workflow_definition=MODEL_FREE_WORKFLOW, init_parameters={})
    runtime = {"image": _image_input(), "detections": SYNTHETIC_DETECTIONS}
    raw = engine.run(runtime_parameters=runtime, serialize_results=False)[0]
    assert raw["blurred"].is_tensor_materialised() == TENSOR_MODE, TENSOR_MODE
    assert raw["labelled"].numpy_image.any(), "the label rendered nothing onto the zero canvas"
    serialised = engine.run(runtime_parameters=runtime, serialize_results=True)[0]
    assert "labelled" in serialised and "blurred" in serialised, serialised


def run_workflow_with_host_hook():
    from roboflow_workflows.execution_engine.core import ExecutionEngine

    class _NeutralHost:
        received = None

        def __workflows_bind__(self, init_parameters, step_error_handler):
            _NeutralHost.received = {
                "keys": sorted(init_parameters),
                "handler": step_error_handler,
            }
            init_parameters["workflows_core.model_manager"] = None
            return step_error_handler

    host = _NeutralHost()
    engine = ExecutionEngine.init(
        workflow_definition=MODEL_FREE_WORKFLOW,
        init_parameters={"workflows_core.model_manager": host},
    )
    runtime = {"image": _image_input(), "detections": SYNTHETIC_DETECTIONS}
    result = engine.run(runtime_parameters=runtime, serialize_results=False)[0]
    assert _NeutralHost.received is not None, "the class-level hook never fired"
    assert result["labelled"].numpy_image.any(), "hook path did not reach block execution"


def run_dynamic_block_workflow():
    # Exercises block_scaffolding's generated code — where any exec'd import
    # would need canonical names post-move.
    out = _run(
        {
            "version": "1.0",
            "inputs": [{"type": "WorkflowParameter", "name": "x"}],
            "dynamic_blocks_definitions": [
                {
                    "type": "DynamicBlockDefinition",
                    "manifest": {
                        "type": "ManifestDescription",
                        "block_type": "Doubler",
                        "inputs": {"x": {"type": "DynamicInputDefinition",
                                          "selector_types": ["input_parameter"]}},
                        "outputs": {"y": {"type": "DynamicOutputDefinition", "kind": []}},
                    },
                    "code": {
                        "type": "PythonCode",
                        "run_function_code": "def run(self, x: int) -> BlockResult:\n    return {\"y\": x * 2}\n",
                    },
                }
            ],
            "steps": [{"type": "Doubler", "name": "double", "x": "$inputs.x"}],
            "outputs": [{"type": "JsonField", "name": "y", "selector": "$steps.double.y"}],
        },
        x=21,
    )
    assert out and out[0]["y"] == 42, out


def standalone_logging():
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    # Logger names deliberately retain their historical namespace. This does
    # not import the Inference server, and ordinary stdlib handlers still work.
    tree_logger = logging.getLogger("inference")
    tree_logger.addHandler(handler)
    tree_logger.setLevel(logging.DEBUG)
    run_model_free_workflow()
    assert records, "no log record reached a plain stdlib handler"


def fonts_offline():
    from roboflow_workflows.core_steps.visualizations.common.fonts import (
        resolve_font_path,
    )
    font = resolve_font_path("Geist Mono")
    assert font.is_file() and _from_installed_wheel_path(font), font
    assert (font.parent / "OFL.txt").is_file(), "font licence not packaged next to the font"
    # Delete the assets from THIS disposable install (venv is thrown away by
    # the parent). The developer's shared install is never touched — every
    # run of this probe uses its own venv (see parent's `run_probe`).
    from pathlib import Path as _P
    assets = _P(INSTALL_ROOT) / "core_steps" / "visualizations" / "common" / "fonts" / "assets"
    shutil.rmtree(assets)
    try:
        resolve_font_path("Geist Mono")
    except RuntimeError as error:
        assert "download_fonts" in str(error), error
    else:
        raise AssertionError("a missing packaged font was silently tolerated")


def _from_installed_wheel_path(path):
    return os.path.realpath(str(path)).startswith(INSTALL_ROOT + os.sep)


def no_blocked_import_attempts():
    attempted = sorted(set(ServerImportBlocker.attempted))
    assert not attempted, json.dumps({"blocked_import_attempts": attempted}, indent=1)
    assert_only_installed_wheel_loaded()


CHECKS = {
    "import_everything": import_everything,
    "load_blocks": load_blocks,
    "run_model_free_workflow": run_model_free_workflow,
    "run_workflow_with_host_hook": run_workflow_with_host_hook,
    "run_dynamic_block_workflow": run_dynamic_block_workflow,
    "standalone_logging": standalone_logging,
    "fonts_offline": fonts_offline,  # deletes assets from THIS install; keep after render
    "no_blocked_import_attempts": no_blocked_import_attempts,  # last
}
for name in EXPECTED_CHECKS:
    if name not in CHECKS:
        results.append({"check": name, "status": "fail",
                        "detail": "the child has no implementation for this check"})
        continue
    check(name, CHECKS[name])
print(json.dumps(results))
"""

EXPECTED_CHECKS = (
    "import_everything",
    "load_blocks",
    "run_model_free_workflow",
    "run_workflow_with_host_hook",
    "run_dynamic_block_workflow",
    "standalone_logging",
    "fonts_offline",
    "no_blocked_import_attempts",
)


def _child_diagnostics(proc: subprocess.CompletedProcess) -> str:
    return (
        f"exit code {proc.returncode}\n"
        f"--- child stderr (tail) ---\n{proc.stderr[-4000:]}\n"
        f"--- child stdout (tail) ---\n{proc.stdout[-2000:]}"
    )


def _validate_results(raw: list, proc: subprocess.CompletedProcess) -> list:
    results, problems, seen = [], [], set()
    for entry in raw:
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("check"), str)
            and entry.get("status") in ("ok", "fail")
        ):
            results.append(entry)
            seen.add(entry["check"])
        else:
            problems.append(f"malformed result entry: {entry!r}")
    missing = [name for name in EXPECTED_CHECKS if name not in seen]
    if missing:
        problems.append(f"checks missing from the child's results: {missing}")
    unexpected = sorted(seen - set(EXPECTED_CHECKS))
    if unexpected:
        problems.append(f"unexpected checks in the child's results: {unexpected}")
    if problems:
        results.append(
            {
                "check": "results_complete",
                "status": "fail",
                "detail": "\n".join(problems) + "\n" + _child_diagnostics(proc),
            }
        )
    return results


def _child_env(scratch: Path) -> dict:
    env = {**os.environ}
    # PYTHONPATH is NOT set: the child resolves everything from the venv's
    # site-packages. Any inherited PYTHONPATH is dropped so the checkout's
    # source cannot leak in.
    env.pop("PYTHONPATH", None)
    env["PYTHONOPTIMIZE"] = "0"
    # Never inherit host configuration/plugins/handlers - the child controls
    # every knob itself:
    for variable in (
        "ENABLE_TENSOR_DATA_REPRESENTATION",
        "WORKFLOWS_IMAGE_TENSOR_DEVICE",
        "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS",
        "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE",
        "ALLOW_WORKFLOWS_FONTS_DOWNLOAD",
        "MODEL_CACHE_DIR",
        "WORKFLOWS_PLUGINS",
        "DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER",
    ):
        env.pop(variable, None)
    # Activate the enterprise plugin canonically. The child's load_blocks()
    # asserts every declared enterprise block is present in the loaded set.
    env["WORKFLOWS_PLUGINS"] = ENTERPRISE_LOADER
    return env


def build_venv(venv_dir: Path, wheel: Path, find_links: list) -> Path:
    """Create a venv OUTSIDE the checkout and install the wheel + deps.

    Installs `<wheel>[enterprise]` so the enterprise plugin's declared
    dependencies are resolved through pip - the enterprise blocks must NOT
    silently rely on undeclared imports. `--no-deps` is deliberately NOT
    used; real dependency resolution is part of what the probe verifies.
    --find-links accepts a list of local directories or wheel URLs; use this
    for CI resolution of SDK/models wheels built in the same job.
    """
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    if os.name == "nt":
        python = venv_dir / "Scripts" / "python.exe"
    else:
        python = venv_dir / "bin" / "python"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
        capture_output=True,
    )
    install = [str(python), "-m", "pip", "install", f"{wheel}[enterprise]"]
    for link in find_links:
        install += ["--find-links", str(link)]
        # Explicit local requirements win over equal-version index candidates.
        # Other transitive dependencies still resolve normally from the index.
        for package in ("inference_sdk", "inference_models"):
            candidates = sorted(Path(link).glob(f"{package}-*.whl"))
            if len(candidates) > 1:
                raise ValueError(f"Multiple {package} wheels in {link}: {candidates}")
            install.extend(str(candidate.resolve()) for candidate in candidates)
    subprocess.run(install, check=True)
    return python


def run_probe(python: Path, scratch: Path, tensor_mode: bool) -> list:
    env = _child_env(scratch)
    proc = subprocess.run(
        [
            str(python),
            "-c",
            CHILD,
            str(scratch),
            "on" if tensor_mode else "off",
            json.dumps(list(EXPECTED_CHECKS)),
        ],
        cwd=scratch,
        env=env,
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
                "status": "fail",
                "detail": f"no/invalid results line ({error})\n{_child_diagnostics(proc)}",
            }
        ]
    results = _validate_results(raw, proc)
    if proc.returncode != 0:
        results.append(
            {
                "check": "child_process",
                "status": "fail",
                "detail": _child_diagnostics(proc),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wheel",
        type=Path,
        required=True,
        help="Path to the roboflow-workflows wheel to probe.",
    )
    parser.add_argument(
        "--find-links",
        action="append",
        default=[],
        help="Extra --find-links target for pip (repeatable). Use for local SDK/models wheels.",
    )
    parser.add_argument("--tensor-mode", choices=["off", "on", "both"], default="both")
    parser.add_argument("--keep-tree", action="store_true")
    args = parser.parse_args()
    assert args.wheel.is_file(), f"wheel not found: {args.wheel}"
    find_links = list(args.find_links)
    env_links = os.environ.get("WORKFLOWS_ISOLATION_FIND_LINKS", "")
    if env_links:
        find_links.extend(part for part in env_links.split(os.pathsep) if part)
    modes = {"off": [False], "on": [True], "both": [False, True]}[args.tensor_mode]
    failed = False
    for tensor_mode in modes:
        scratch = Path(tempfile.mkdtemp(prefix="workflows-isolation-"))
        venv_dir = scratch / "venv"
        python = build_venv(venv_dir, args.wheel, find_links)
        for result in run_probe(python, scratch, tensor_mode):
            label = "on" if tensor_mode else "off"
            print(f"tensor={label} {result['check']}: {result['status']}")
            if result["status"] != "ok":
                failed = True
                print(result.get("detail", "<no detail reported>"))
        if args.keep_tree:
            print(f"tree kept at {scratch}")
        else:
            shutil.rmtree(scratch)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
