"""Prove inference/core/workflows works with the server package unimportable.

Copies the module next to EMPTY stub parents (`inference/`, `inference/core/`)
in a scratch tree and runs every check in a child interpreter that (a) puts
that tree first on sys.path and (b) installs a meta-path blocker refusing
every other `inference.*` module. The blocker is load-bearing: the venv's
editable-install finder serves direct children such as
`inference.usage_tracking` from the real checkout even when the parent
package is a stub (verified 2026-09-08), so stub parents alone are not
isolation.

Deviation from the plan text (recorded 2026-09-08): the child's `PYTHONPATH`
is `<scratch tree>` **plus** `<repo>/inference_models`. `inference_models` and
`inference_sdk` are allowed dependencies of workflows and the probe is
specified to run with both installed, but in this checkout `inference_models`
is importable only through that path entry - the venv's editable install
resolves it from a different checkout. Setting `PYTHONPATH` to the scratch
tree alone would silently swap in the other checkout's copy.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE = REPO_ROOT / "inference" / "core" / "workflows"

CHILD = r"""
import importlib, json, logging, os, pkgutil, shutil, sys, traceback

# Every check below verifies with `assert`, which `-O` / PYTHONOPTIMIZE strips.
# A plain `assert sys.flags.optimize == 0` would be stripped too, so the guard
# is an `if` and it runs before anything else - a probe that cannot verify must
# report, not pass quietly. The parent also pins PYTHONOPTIMIZE=0.
if sys.flags.optimize:
    print(json.dumps([{
        "check": "no_optimize",
        "status": "fail",
        "detail": "child ran with optimization enabled; assert statements are stripped",
    }]))
    sys.exit(1)

T, TENSOR_MODE = sys.argv[1], sys.argv[2] == "on"
# The parent owns the check list and hands it over as JSON - one source of
# truth, so a check the parent expects can never silently go unrun.
EXPECTED_CHECKS = json.loads(sys.argv[3])
PACKAGE = "inference.core.workflows"
results = []


def _allowed(name):
    return (
        name in ("inference", "inference.core")
        or name == PACKAGE
        or name.startswith(PACKAGE + ".")
    )


class ServerImportBlocker:
    # First on sys.meta_path: refuses every `inference.*` module outside the
    # copied package - including what the editable finder would otherwise
    # serve - and records the attempt, so a `try: import ... except
    # ImportError` inside workflows is still reported.
    attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "inference" or fullname.startswith("inference.")) and not _allowed(fullname):
            self.attempted.append(fullname)
            raise ImportError(f"blocked: {fullname} is the inference server package")
        return None


sys.meta_path.insert(0, ServerImportBlocker())


def check(name, fn):
    try:
        fn()
        results.append({"check": name, "status": "ok"})
    except BaseException:  # noqa: BLE001 - report every failure, keep going
        results.append({"check": name, "status": "fail", "detail": traceback.format_exc()})


def _inside_copy(file):
    from pathlib import Path
    try:
        Path(file).resolve().relative_to(Path(T).resolve())
        return True
    except ValueError:
        return False


def assert_only_copied_modules_loaded():
    leaked = sorted(
        name for name in sys.modules
        if (name == "inference" or name.startswith("inference.")) and not _allowed(name)
    )
    # Directory containment on resolved paths, not a string prefix - `T-other/`
    # would otherwise pass. A module without __file__ is reported, not skipped.
    foreign = sorted(
        name for name, module in sys.modules.items()
        if (name.startswith(PACKAGE + ".") or name == PACKAGE)
        and not (getattr(module, "__file__", None) and _inside_copy(module.__file__))
    )
    stubs = {
        "inference": os.path.join(T, "inference", "__init__.py"),
        "inference.core": os.path.join(T, "inference", "core", "__init__.py"),
    }
    wrong_stubs = {
        name: getattr(sys.modules.get(name), "__file__", None)
        for name, expected in stubs.items()
        if name in sys.modules and os.path.realpath(sys.modules[name].__file__ or "") != os.path.realpath(expected)
    }
    assert not leaked and not foreign and not wrong_stubs, json.dumps(
        {"leaked_server_modules": leaked, "not_from_copy": foreign, "wrong_stubs": wrong_stubs},
        indent=1,
    )


def import_everything():
    import inference
    # Directory containment on resolved paths, not a string prefix: on macOS
    # `tempfile.mkdtemp` hands back `/var/...` while the child resolves the
    # copy to `/private/var/...`, so a prefix test aborts the walk on the
    # correct tree (probe-code fix, recorded 2026-09-08).
    assert _inside_copy(inference.__file__), f"real package leaked in: {inference.__file__}"
    import inference.core.workflows as workflows
    failures = {}
    for info in pkgutil.walk_packages(
        workflows.__path__,
        PACKAGE + ".",
        onerror=lambda name: failures.setdefault(name, "package import raised"),
    ):
        try:
            importlib.import_module(info.name)
        except Exception as error:  # noqa: BLE001
            failures[info.name] = repr(error)
    assert not failures, json.dumps({"import_failures": failures}, indent=1)
    assert_only_copied_modules_loaded()


def load_blocks():
    from inference.core.workflows.execution_engine.introspection.blocks_loader import (
        describe_available_blocks, load_workflow_blocks,
    )
    identifiers = {b.identifier for b in load_workflow_blocks()}
    assert identifiers, "no blocks loaded"
    # Effective tensor configuration, not just the requested one: the blur
    # block registers from `v1_tensor` only in tensor mode.
    tensor_blur = PACKAGE + ".core_steps.classical_cv.image_blur.v1_tensor.ImageBlurBlockV1"
    assert (tensor_blur in identifiers) == TENSOR_MODE, (
        TENSOR_MODE, sorted(i for i in identifiers if "image_blur" in i)
    )
    describe_available_blocks(dynamic_blocks=[]).model_dump_json()


def _run(definition, **runtime):
    from inference.core.workflows.execution_engine.core import ExecutionEngine
    engine = ExecutionEngine.init(workflow_definition=definition, init_parameters={})
    return engine.run(runtime_parameters=runtime, serialize_results=True)


def _image_input():
    # Tensor mode feeds a raw CHW tensor (deserializers_tensor.py:94 accepts
    # it); numpy mode feeds an ndarray. Same zero canvas either way.
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
        # rich_label needs predictions and a text CHOICE (rich_label/v1.py:136);
        # the synthetic detection's class name is what gets rendered. Never
        # pass empty predictions: the block returns before font resolution
        # (rich_label/v1.py:327).
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
    # Blur, then render a label with a packaged font - a real offline render,
    # in both representations.
    from inference.core.workflows.execution_engine.core import ExecutionEngine
    engine = ExecutionEngine.init(workflow_definition=MODEL_FREE_WORKFLOW, init_parameters={})
    runtime = {"image": _image_input(), "detections": SYNTHETIC_DETECTIONS}
    raw = engine.run(runtime_parameters=runtime, serialize_results=False)[0]
    # Tensor mode must take the tensor path, not the numpy fallback
    # (v1_tensor.py:81): gaussian/5 is supported natively.
    assert raw["blurred"].is_tensor_materialised() == TENSOR_MODE, TENSOR_MODE
    assert raw["labelled"].numpy_image.any(), "the label rendered nothing onto the zero canvas"
    serialised = engine.run(runtime_parameters=runtime, serialize_results=True)[0]
    assert "labelled" in serialised and "blurred" in serialised, serialised


def run_dynamic_block_workflow():
    # Exercises block_scaffolding's generated code - where the exec'd
    # `from inference.core.env import ...` string used to live (Phase 5).
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
    tree_logger = logging.getLogger(PACKAGE)
    tree_logger.addHandler(handler)
    tree_logger.setLevel(logging.DEBUG)
    run_model_free_workflow()
    assert records, "no log record reached a plain stdlib handler"


def fonts_offline():
    from inference.core.workflows.core_steps.visualizations.common.fonts import (
        resolve_font_path,
    )
    font = resolve_font_path("Geist Mono")
    assert font.is_file() and _inside_copy(font), font
    assert (font.parent / "OFL.txt").is_file(), "font licence not packaged next to the font"
    assets = os.path.join(T, "inference", "core", "workflows", "core_steps",
                          "visualizations", "common", "fonts", "assets")
    shutil.rmtree(assets)
    try:
        resolve_font_path("Geist Mono")
    except RuntimeError as error:
        assert "download_fonts" in str(error), error
    else:
        raise AssertionError("a missing packaged font was silently tolerated")


def no_blocked_import_attempts():
    # Runs last: covers imports attempted during block loading and execution,
    # not only during the import walk.
    attempted = sorted(set(ServerImportBlocker.attempted))
    assert not attempted, json.dumps({"blocked_import_attempts": attempted}, indent=1)
    assert_only_copied_modules_loaded()


CHECKS = {
    "import_everything": import_everything,
    "load_blocks": load_blocks,
    "run_model_free_workflow": run_model_free_workflow,
    "run_dynamic_block_workflow": run_dynamic_block_workflow,
    "standalone_logging": standalone_logging,
    "fonts_offline": fonts_offline,  # deletes the assets - keep it after the render
    "no_blocked_import_attempts": no_blocked_import_attempts,  # last
}
# Ordering comes from the parent's EXPECTED_CHECKS, which is authoritative.
for name in EXPECTED_CHECKS:
    if name not in CHECKS:
        results.append({"check": name, "status": "fail",
                        "detail": "the child has no implementation for this check"})
        continue
    check(name, CHECKS[name])
print(json.dumps(results))
"""

# The checks the child must run, in order. `fonts_offline` deletes the copied
# assets so it comes after the render; `no_blocked_import_attempts` is last so
# it sees every attempt. The parent both hands this to the child and validates
# the returned result set against it.
EXPECTED_CHECKS = (
    "import_everything",
    "load_blocks",
    "run_model_free_workflow",
    "run_dynamic_block_workflow",
    "standalone_logging",
    "fonts_offline",
    "no_blocked_import_attempts",
)


def build_tree(target: Path) -> None:
    (target / "inference" / "core").mkdir(parents=True)
    (target / "inference" / "__init__.py").write_text("")
    (target / "inference" / "core" / "__init__.py").write_text("")
    shutil.copytree(
        MODULE,
        target / "inference" / "core" / "workflows",
        ignore=shutil.ignore_patterns("__pycache__"),
    )


def _child_diagnostics(proc: subprocess.CompletedProcess) -> str:
    return (
        f"exit code {proc.returncode}\n"
        f"--- child stderr (tail) ---\n{proc.stderr[-4000:]}\n"
        f"--- child stdout (tail) ---\n{proc.stdout[-2000:]}"
    )


def _validate_results(raw: list, proc: subprocess.CompletedProcess) -> list:
    """Keep only well-formed rows and flag anything the child owed us.

    Missing results must never read as success: an empty list, a truncated run
    and a malformed row all have to surface as a failing `results_complete`
    row, because `main()` decides purely on the rows it is handed.
    """
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


def _child_env(tree: Path, tensor_mode: bool) -> dict:
    env = {
        **os.environ,
        # See the module docstring: the scratch tree first, then this
        # checkout's `inference_models` (an allowed dependency that the venv
        # would otherwise resolve from a different checkout).
        "PYTHONPATH": str(tree) + os.pathsep + str(REPO_ROOT / "inference_models"),
        "MODEL_CACHE_DIR": str(tree / "cache"),  # no previously cached fonts
        # Standalone configuration. Phase 5 decides how the workflows-local
        # default reads these; until then they are the env names the module
        # consumes today. Adjust here if Phase 5 moves them.
        "ENABLE_TENSOR_DATA_REPRESENTATION": "True" if tensor_mode else "False",
        "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": "True",
        "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "local",
        "ALLOW_WORKFLOWS_FONTS_DOWNLOAD": "False",
        # The checks verify with `assert`; an inherited PYTHONOPTIMIZE would
        # strip every one of them and turn the probe green by deleting it.
        "PYTHONOPTIMIZE": "0",
    }
    # Never inherit the server's plugin list (Task 7.1 expands the enterprise
    # plugin into it); the probe loads core blocks only.
    env.pop("WORKFLOWS_PLUGINS", None)
    # Likewise the server's step-error handler: it selects a handler the
    # standalone engine does not register, so an inherited value would fail
    # `ExecutionEngine.init` for a reason that has nothing to do with
    # contamination. The child takes the workflows-local default.
    env.pop("DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER", None)
    return env


def run_probe(tree: Path, tensor_mode: bool) -> list:
    env = _child_env(tree, tensor_mode)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD,
            str(tree),
            "on" if tensor_mode else "off",
            json.dumps(list(EXPECTED_CHECKS)),
        ],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
    )
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        raw = json.loads(line)
        if not isinstance(raw, list):
            raise ValueError(f"results line is {type(raw).__name__}, not a list")
    except ValueError as error:  # includes json.JSONDecodeError
        # No parseable results at all: report the child's own failure rather
        # than crashing on it, and never let silence read as success.
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
    parser.add_argument("--tensor-mode", choices=["off", "on", "both"], default="both")
    parser.add_argument("--keep-tree", action="store_true")
    args = parser.parse_args()
    modes = {"off": [False], "on": [True], "both": [False, True]}[args.tensor_mode]
    failed = False
    for tensor_mode in modes:
        tree = Path(tempfile.mkdtemp(prefix="workflows-isolation-"))
        build_tree(tree)
        for result in run_probe(tree, tensor_mode):
            label = "on" if tensor_mode else "off"
            print(f"tensor={label} {result['check']}: {result['status']}")
            if result["status"] != "ok":
                failed = True
                print(result.get("detail", "<no detail reported>"))
        if args.keep_tree:
            print(f"tree kept at {tree}")
        else:
            shutil.rmtree(tree)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
