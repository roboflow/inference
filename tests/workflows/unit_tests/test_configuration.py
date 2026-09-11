import ast
import dataclasses
import importlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import List

import pytest

# The facade is imported HERE, at collection time, on purpose (round-5 defect 1):
# it binds its 67 constants from `get_configuration()` at its FIRST import, and the
# autouse fixture below resets the registry before every test. If a test body
# performed the first import, the facade would freeze standalone defaults, the
# fixture would restore only `_CONFIGURATION`, and the server/facade parity test
# in tests/inference/.../test_workflows_configuration.py would fail under a
# legitimate restrictive deployment (both access flags False). At collection time
# the process registry still holds what the host installed.
import inference.core.workflows.environment as workflows_environment
from inference.core.workflows import configuration as configuration_module
from inference.core.workflows.configuration import (
    WorkflowsConfiguration,
    configure_process,
    default_configuration,
    describe_configuration_difference,
    ensure_process_configuration_matches,
    get_configuration,
    reset_configuration,
    resolve_image_tensor_device,
)
from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = REPO_ROOT / "inference" / "core" / "env.py"

# Captured at collection time, right after the facade import above and before
# the autouse fixture below ever resets the registry: the configuration the
# host had installed (the server's, after Task 5.2; the sticky standalone
# default in a bare process). The facade must have been bound from THIS.
_INSTALLED_AT_COLLECTION = configuration_module.get_configuration()


@pytest.fixture(autouse=True)
def _isolated_process_configuration():
    # Every test in this file owns the process registry; restore whatever the
    # session had installed so no other suite sees a reset one.
    previous = configuration_module._CONFIGURATION
    reset_configuration()
    yield
    with configuration_module._INSTALL_LOCK:
        configuration_module._CONFIGURATION = previous


# --------------------------------------------------------------------------
# Shape and defaults
# --------------------------------------------------------------------------


def test_configuration_is_frozen_in_every_group() -> None:
    configuration = default_configuration()
    assert len(dataclasses.fields(configuration)) == 9
    total_fields = 0
    for group in dataclasses.fields(configuration):
        value = getattr(configuration, group.name)
        assert dataclasses.is_dataclass(value)
        total_fields += len(dataclasses.fields(value))
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, dataclasses.fields(value)[0].name, "mutated")
    assert total_fields == 67, total_fields


def test_default_configuration_matches_env_pys_empty_environment_defaults() -> None:
    configuration = default_configuration()
    assert configuration.engine.step_execution_mode == "local"
    assert configuration.engine.async_future_result_timeout == 60.0
    assert configuration.engine.max_inner_workflow_depth == 4
    assert configuration.engine.max_inner_workflow_count == 32
    assert configuration.engine.allow_custom_python_execution is True
    assert configuration.engine.custom_python_execution_mode == "local"
    assert configuration.engine.allow_blocks_accessing_local_storage is True
    assert configuration.engine.allow_blocks_accessing_environmental_variables is True
    assert configuration.engine.blocks_write_directory is None
    assert configuration.engine.disabled_block_types == ()
    assert configuration.engine.disabled_block_patterns == ()
    assert configuration.tensor.representation_enabled is False
    assert configuration.tensor.image_tensor_device is None
    assert configuration.tensor.visualisation_validate_owners is False
    assert configuration.tensor.sam_video_mask_representation == "rle"
    assert configuration.tensor.enforce_dense_instance_masks is False
    assert configuration.remote.api_target == "hosted"
    assert configuration.remote.api_key_transport == "both"
    assert configuration.remote.local_inference_api_url == "http://127.0.0.1:9001"
    assert configuration.remote.hosted_detect_url == "https://detect.roboflow.com"
    assert (
        configuration.remote.hosted_classification_url
        == "https://classify.roboflow.com"
    )
    assert (
        configuration.remote.hosted_instance_segmentation_url
        == "https://outline.roboflow.com"
    )
    assert (
        configuration.remote.hosted_semantic_segmentation_url
        == "https://segment.roboflow.com"
    )
    assert configuration.remote.hosted_core_model_url == "https://infer.roboflow.com"
    assert configuration.remote.max_step_batch_size == 1
    assert configuration.remote.max_step_concurrent_requests == 8
    assert configuration.platform.api_base_url == "https://api.roboflow.com"
    assert configuration.platform.offline_mode is False
    assert configuration.platform.secure_gateway is None
    assert configuration.platform.gcp_serverless is False
    assert configuration.fonts.allow_download is True
    assert configuration.fonts.model_cache_dir == "/tmp/cache"
    assert configuration.models.lmm_enabled is False
    assert configuration.models.clip_version_id == "ViT-B-16"
    assert configuration.models.sam3_exec_mode == "local"
    assert configuration.models.sam3_3d_objects_enabled is False
    assert configuration.modal.token_id is None
    assert configuration.modal.workspace_name == "roboflow"
    assert configuration.modal.app_name == "webexec-roboflow-platform"
    assert configuration.modal.executor_idle_ttl_seconds == 1800
    assert configuration.modal.jpeg_quality == 95
    assert configuration.modal.transport == "http"
    assert configuration.modal.ws_connect_timeout_seconds == 30
    assert configuration.modal.ws_read_timeout_seconds == 720
    assert configuration.modal.ws_connection_pool_size == 1
    assert configuration.modal.ws_fail_on_session_loss is False
    assert configuration.modal.ws_idle_release_seconds == 120
    assert configuration.secrets.api_key is None
    assert configuration.secrets.roboflow_internal_service_name is None
    assert configuration.secrets.roboflow_internal_service_secret is None
    assert configuration.debug.output_dir is None


def test_secrets_are_kept_out_of_the_repr() -> None:
    base = default_configuration()
    configuration = dataclasses.replace(
        base,
        secrets=dataclasses.replace(
            base.secrets,
            api_key="SECRET-API-KEY",
            roboflow_internal_service_secret="SECRET-SERVICE-SECRET",
        ),
        modal=dataclasses.replace(base.modal, token_secret="SECRET-MODAL-TOKEN"),
    )
    rendered = repr(configuration)
    assert "SECRET-API-KEY" not in rendered
    assert "SECRET-SERVICE-SECRET" not in rendered
    assert "SECRET-MODAL-TOKEN" not in rendered


# --------------------------------------------------------------------------
# Conflict variants - derived from the INSTALLED value, never hard-coded
# --------------------------------------------------------------------------

# Round-2 defect 4: hard-coded override values ("set representation_enabled=True")
# are equal to the installed value in the mandatory `ENABLE_TENSOR_DATA_REPRESENTATION=True`
# suite run, so `pytest.raises` never fires. Every variant is now derived by
# perturbing whatever is installed, and inequality is asserted first.

CONFLICT_SENTINEL = "phase5-conflict-probe"

GROUP_FIELDS = [
    ("engine", "allow_custom_python_execution"),
    ("tensor", "representation_enabled"),
    ("remote", "api_target"),
    ("platform", "offline_mode"),
    ("fonts", "allow_download"),
    ("models", "sam3_exec_mode"),
    ("modal", "transport"),
    ("secrets", "roboflow_internal_service_name"),
    ("debug", "output_dir"),
]


def perturb(value):
    """A value guaranteed different from `value`, of a compatible kind."""
    if isinstance(value, bool):  # before int - bool IS an int
        return not value
    if value is None:
        return CONFLICT_SENTINEL
    if isinstance(value, str):
        return (
            CONFLICT_SENTINEL
            if value != CONFLICT_SENTINEL
            else CONFLICT_SENTINEL + "-2"
        )
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, tuple):
        return value + (CONFLICT_SENTINEL,)
    raise AssertionError(f"perturb() has no rule for {type(value)!r}")


def variant_of(base: WorkflowsConfiguration, group: str, field: str):
    current = getattr(getattr(base, group), field)
    changed = perturb(current)
    assert changed != current, (group, field, current)
    return dataclasses.replace(
        base, **{group: dataclasses.replace(getattr(base, group), **{field: changed})}
    )


# --------------------------------------------------------------------------
# Registry semantics
# --------------------------------------------------------------------------


def test_get_configuration_falls_back_to_the_standalone_default() -> None:
    assert get_configuration() == default_configuration()


def test_configure_process_accepts_an_equal_but_distinct_object() -> None:
    first = variant_of(default_configuration(), "remote", "api_target")
    configure_process(first)
    configure_process(dataclasses.replace(first))  # equal, different identity
    assert get_configuration() == first


def test_configure_process_accepts_the_identical_object_repeatedly() -> None:
    configuration = default_configuration()
    for _ in range(5):
        configure_process(configuration)
    assert get_configuration() is configuration


@pytest.mark.parametrize(
    "group, field", GROUP_FIELDS, ids=[f"{g}.{f}" for g, f in GROUP_FIELDS]
)
def test_configure_process_refuses_a_conflicting_configuration(group, field) -> None:
    configure_process(default_configuration())
    with pytest.raises(WorkflowEnvironmentConfigurationError) as raised:
        configure_process(variant_of(default_configuration(), group, field))
    assert f"{group}.{field}" in raised.value.public_message


@pytest.mark.parametrize(
    "group, field", GROUP_FIELDS, ids=[f"{g}.{f}" for g, f in GROUP_FIELDS]
)
def test_ensure_process_configuration_matches_refuses_every_group(group, field) -> None:
    configure_process(default_configuration())
    with pytest.raises(WorkflowEnvironmentConfigurationError) as raised:
        ensure_process_configuration_matches(
            variant_of(default_configuration(), group, field)
        )
    assert f"{group}.{field}" in raised.value.public_message


def test_ensure_process_configuration_matches_accepts_an_equal_object_and_refuses_none() -> (
    None
):
    # Round-4 defect 1: the engine calls this only when the key is PRESENT, so
    # a None here is an explicit None - which the resolver would hand to every
    # configuration-consuming block ahead of the registered default.
    configure_process(default_configuration())
    ensure_process_configuration_matches(default_configuration())
    with pytest.raises(WorkflowEnvironmentConfigurationError) as raised:
        ensure_process_configuration_matches(None)
    assert "NoneType" in raised.value.public_message


def test_ensure_process_configuration_matches_rejects_a_foreign_type() -> None:
    # Round-2 defect 1: a value under the DEDICATED key that is not a
    # WorkflowsConfiguration must be a workflows error, not an AttributeError
    # from `describe_configuration_difference`.
    configure_process(default_configuration())
    with pytest.raises(WorkflowEnvironmentConfigurationError) as raised:
        ensure_process_configuration_matches({"threshold": 0.5})
    assert "WorkflowsConfiguration" in raised.value.public_message


def test_describe_configuration_difference_names_group_and_field() -> None:
    base = default_configuration()
    differences = describe_configuration_difference(
        base, variant_of(base, "remote", "api_target")
    )
    assert differences == [f"remote.api_target: 'hosted' != {CONFLICT_SENTINEL!r}"]


def test_describe_configuration_difference_redacts_secret_fields() -> None:
    base = default_configuration()
    changed = dataclasses.replace(
        base, secrets=dataclasses.replace(base.secrets, api_key="SECRET-API-KEY")
    )
    differences = describe_configuration_difference(base, changed)
    assert differences == ["secrets.api_key: differs (redacted)"]
    assert "SECRET-API-KEY" not in "".join(differences)


def test_configure_process_holds_the_install_lock_for_its_whole_critical_section() -> (
    None
):
    """Deterministic proof of mutual exclusion (round-2 defect 8).

    The round-1 barrier test passed 100/100 times with the lock REMOVED,
    because a barrier before the call does not force an interleaving between
    the check and the assignment. This test holds `_INSTALL_LOCK` in the main
    thread and shows a competing `configure_process` in another thread cannot
    proceed until it is released - which is false the moment the `with` block
    is deleted from `configure_process`. Verified against a two-implementation
    probe: locked -> blocked=True, unlocked -> blocked=False.
    """
    started, finished = threading.Event(), threading.Event()
    failures = []

    def _install() -> None:
        started.set()
        try:
            configure_process(default_configuration())
        except BaseException as error:  # noqa: BLE001 - reported, not raised
            failures.append(error)
        finished.set()

    worker = threading.Thread(target=_install)
    with configuration_module._INSTALL_LOCK:
        worker.start()
        assert started.wait(timeout=5), "worker never started"
        # The lock is held here: the worker MUST NOT get through.
        assert not finished.wait(timeout=0.5), (
            "configure_process completed while _INSTALL_LOCK was held - its "
            "critical section is not protected"
        )
    worker.join(timeout=5)
    assert finished.is_set(), "worker did not finish after the lock was released"
    assert not failures, failures


@pytest.mark.parametrize(
    "operation",
    ["get_configuration", "reset_configuration"],
)
def test_the_other_registry_operations_hold_the_lock_too(operation) -> None:
    started, finished = threading.Event(), threading.Event()

    def _call() -> None:
        started.set()
        getattr(configuration_module, operation)()
        finished.set()

    worker = threading.Thread(target=_call)
    with configuration_module._INSTALL_LOCK:
        worker.start()
        assert started.wait(timeout=5)
        assert not finished.wait(timeout=0.5), operation
    worker.join(timeout=5)
    assert finished.is_set()


def test_two_conflicting_installs_end_with_exactly_one_winner() -> None:
    barrier = threading.Barrier(2)
    outcomes = []

    def _install(configuration) -> None:
        barrier.wait(timeout=5)
        try:
            configure_process(configuration)
            outcomes.append("ok")
        except WorkflowEnvironmentConfigurationError:
            outcomes.append("refused")

    first = default_configuration()
    second = variant_of(first, "tensor", "representation_enabled")
    threads = [
        threading.Thread(target=_install, args=(first,)),
        threading.Thread(target=_install, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert sorted(outcomes) == ["ok", "refused"], outcomes
    assert get_configuration() in (first, second)


# --------------------------------------------------------------------------
# resolve_image_tensor_device - DIFFERENTIAL: the COPY against the ORIGINAL
# --------------------------------------------------------------------------

# Round-2 defect 5: the round-1 tests compared the copy against `env.py`'s
# already-materialised module attribute in a subprocess, which (a) never pinned
# `USE_INFERENCE_MODELS` (`env.py:1486` ANDs it into the flag, so a nominal
# "on" case could compare None with None) and (b) could not cover missing torch
# at all, because importing `inference.core.env` needs torch transitively.
#
# The harness below lifts `env.py`'s OWN resolution statement out by AST and
# executes it beside the copy under identical, controlled import conditions.
# No `inference.core.env` import is needed, so every cell - including missing
# torch and cuda-absent - is a true copy-vs-original comparison.


def _original_resolution_block() -> ast.If:
    """`if ENABLE_TENSOR_DATA_REPRESENTATION: try: import torch ...` from env.py."""
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "ENABLE_TENSOR_DATA_REPRESENTATION"
            and any(isinstance(statement, ast.Try) for statement in node.body)
        ):
            for statement in node.body:
                if isinstance(statement, ast.Try) and any(
                    isinstance(inner, ast.Import)
                    and any(alias.name == "torch" for alias in inner.names)
                    for inner in ast.walk(statement)
                ):
                    return node
    raise AssertionError("env.py's device-resolution block was not found")


def _run_original(flag: bool, device: str | None):
    node = _original_resolution_block()
    namespace = {
        "ENABLE_TENSOR_DATA_REPRESENTATION": flag,
        "WORKFLOWS_IMAGE_TENSOR_DEVICE_STR": device,
        "WORKFLOWS_IMAGE_TENSOR_DEVICE": None,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(ENV_PATH), "exec"),
        namespace,
    )
    return namespace["WORKFLOWS_IMAGE_TENSOR_DEVICE"]


@pytest.mark.parametrize(
    "flag, device",
    [(False, None), (False, "cuda"), (True, "cpu"), (True, None)],
    ids=["off-nodevice", "off-device", "on-cpu", "on-autodetect"],
)
def test_the_copy_matches_the_original_resolution(flag, device) -> None:
    pytest.importorskip("torch")
    assert _run_original(flag, device) == resolve_image_tensor_device(flag, device)


def test_the_copy_matches_the_original_when_cuda_is_absent(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original = _run_original(True, None)
    copied = resolve_image_tensor_device(True, None)
    assert original == copied == torch.device("cpu")


def test_the_copy_matches_the_original_when_torch_is_missing() -> None:
    import builtins

    real_import = builtins.__import__

    def _refuse_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch is not installed")
        return real_import(name, *args, **kwargs)

    saved = sys.modules.pop("torch", None)
    builtins.__import__ = _refuse_torch
    try:
        original = _run_original(True, None)
        copied = resolve_image_tensor_device(True, None)
    finally:
        builtins.__import__ = real_import
        if saved is not None:
            sys.modules["torch"] = saved
    assert original is None
    assert copied is None


def test_the_copy_preserves_the_originals_exception_boundary() -> None:
    """`env.py:1535-1547` wraps import + autodetect + materialisation in ONE
    `try`, catching only `ImportError`. The copy must do the same, or a
    `torch.cuda.is_available()` that raises `ImportError` would propagate from
    the copy while `env.py` swallows it."""
    node = _original_resolution_block()
    try_statement = next(s for s in node.body if isinstance(s, ast.Try))
    assert len(try_statement.body) == 3, ast.dump(try_statement)
    assert [type(s).__name__ for s in try_statement.body] == ["Import", "If", "Assign"]
    assert [handler.type.id for handler in try_statement.handlers] == ["ImportError"]

    copy_source = ast.parse(
        Path(configuration_module.__file__).read_text(encoding="utf-8")
    )
    function = next(
        node
        for node in ast.walk(copy_source)
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_image_tensor_device"
    )
    copied_try = next(s for s in ast.walk(function) if isinstance(s, ast.Try))
    assert [type(s).__name__ for s in copied_try.body] == ["Import", "If", "Return"]
    assert [handler.type.id for handler in copied_try.handlers] == ["ImportError"]


def test_the_copy_rejects_an_invalid_device_exactly_as_the_original_does() -> None:
    pytest.importorskip("torch")
    # Neither guards `torch.device(...)`: an invalid string raises RuntimeError.
    with pytest.raises(RuntimeError):
        _run_original(True, "nonsense")
    with pytest.raises(RuntimeError):
        resolve_image_tensor_device(True, "nonsense")


def test_the_resolver_agrees_with_a_real_env_import(tmp_path) -> None:
    """End-to-end cell: a subprocess imports the REAL `inference.core.env` with
    the flag forced on, `USE_INFERENCE_MODELS` pinned (env.py:1486 ANDs it in),
    and asserts the module's own effective flag before comparing devices."""
    pytest.importorskip("torch")
    child = (
        "import json, os\n"
        "from inference.core import env\n"
        "from inference.core.workflows.configuration import resolve_image_tensor_device\n"
        "raw = os.environ.get('WORKFLOWS_IMAGE_TENSOR_DEVICE')\n"
        "print(json.dumps({\n"
        "  'flag': env.ENABLE_TENSOR_DATA_REPRESENTATION,\n"
        "  'equal': resolve_image_tensor_device(env.ENABLE_TENSOR_DATA_REPRESENTATION, raw)\n"
        "           == env.WORKFLOWS_IMAGE_TENSOR_DEVICE,\n"
        "  'device': str(env.WORKFLOWS_IMAGE_TENSOR_DEVICE),\n"
        "}))\n"
    )
    child_env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "inference_models"),
        "ENABLE_TENSOR_DATA_REPRESENTATION": "True",
        "USE_INFERENCE_MODELS": "True",
        "WORKFLOWS_IMAGE_TENSOR_DEVICE": "cpu",
    }
    completed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=REPO_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    # Pinning USE_INFERENCE_MODELS is what makes this cell meaningful: without
    # it the flag can resolve False and both sides trivially compare None.
    assert payload["flag"] is True, payload
    assert payload["device"] == "cpu", payload
    assert payload["equal"], payload


def test_the_resolver_is_none_when_the_flag_is_off() -> None:
    assert resolve_image_tensor_device(False) is None
    assert resolve_image_tensor_device(False, "cuda") is None


# --------------------------------------------------------------------------
# The facade
# --------------------------------------------------------------------------


def test_environment_facade_exports_every_owned_symbol() -> None:
    # Deliberately value-free: this suite runs in BOTH tensor modes and the
    # facade was bound from whatever configuration the process installed. The
    # value contract is pinned server-side against `inference.core.env` by
    # tests/inference/unit_tests/core/interfaces/test_workflows_configuration.py.
    # Uses the MODULE-LEVEL import: a first import inside this body (after the
    # autouse reset) would freeze standalone defaults into the facade.
    exported = {
        name
        for name in vars(workflows_environment)
        if name.isupper() and not name.startswith("_")
    }
    assert len(exported) == 67, sorted(exported)
    assert isinstance(workflows_environment.WORKFLOW_DISABLED_BLOCK_TYPES, list)
    assert isinstance(workflows_environment.WORKFLOW_DISABLED_BLOCK_PATTERNS, list)
    assert isinstance(workflows_environment.ENABLE_TENSOR_DATA_REPRESENTATION, bool)


def test_the_facade_was_bound_from_the_installed_configuration() -> None:
    # Guards the collection-time import above (round-5 defect 1): the facade's
    # constants must come from the configuration installed BEFORE this file's
    # fixture ran, never from the standalone default a reset would install.
    # Two access flags that a restrictive deployment legitimately sets False.
    assert (
        workflows_environment.ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE
        is _INSTALLED_AT_COLLECTION.engine.allow_blocks_accessing_local_storage
    )
    assert (
        workflows_environment.ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES
        is _INSTALLED_AT_COLLECTION.engine.allow_blocks_accessing_environmental_variables
    )
    assert workflows_environment.ENABLE_TENSOR_DATA_REPRESENTATION is (
        _INSTALLED_AT_COLLECTION.tensor.representation_enabled
    )


# --------------------------------------------------------------------------
# The two new modules read no environment - ONE scanner for the whole phase
# --------------------------------------------------------------------------


def environment_reads(tree: ast.AST) -> List[int]:
    """Line numbers of every direct environment read in `tree`.

    ONE rule for the whole phase (round-3 defect 7): a call to `*.getenv(...)`,
    a call on `*.environ` (`os.environ.get(...)`, `.setdefault(...)`, ...) and
    a SUBSCRIPT of `*.environ` (`os.environ["X"]`). Prose - a docstring that
    NAMES `os.environ` - is not a read (round-2 defect 3). Task 5.7's
    `test_no_server_env_imports.py` imports this function, and so does its
    preflight command, so there is exactly one scanner.
    """
    reads = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "getenv":
                reads.append(node.lineno)
            elif (
                isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ"
            ):
                reads.append(node.lineno)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if node.value.attr == "environ":
                reads.append(node.lineno)
    return sorted(reads)


def test_environment_reads_sees_calls_and_subscripts_but_not_prose() -> None:
    source = (
        '"""Values come from the configuration, never from os.environ."""\n'
        "import os\n"
        "A = os.getenv('A')\n"
        "B = os.environ.get('B')\n"
        "C = os.environ['C']\n"
        "os.environ.setdefault('D', '1')\n"
    )
    assert environment_reads(ast.parse(source)) == [3, 4, 5, 6]
    assert environment_reads(ast.parse('"""never from os.environ"""\n')) == []


@pytest.mark.parametrize(
    "module_name",
    ["inference.core.workflows.configuration", "inference.core.workflows.environment"],
)
def test_the_new_module_imports_nothing_from_the_server_and_reads_no_environment(
    module_name,
) -> None:
    # Round-2 defect 3: the round-1 gate was a raw substring scan and failed on
    # the facade's own docstring. This looks at imports, calls and subscripts.
    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    foreign = sorted(
        m
        for m in modules
        if m == "inference"
        or (m.startswith("inference.") and not m.startswith("inference.core.workflows"))
    )
    assert not foreign, (module_name, foreign)
    assert "os" not in modules, (module_name, "os must not be imported")
    assert environment_reads(tree) == [], (module_name, environment_reads(tree))
