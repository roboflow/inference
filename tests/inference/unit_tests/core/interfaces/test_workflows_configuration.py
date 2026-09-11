import ast
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Module level on purpose (round-5 defect 1, D5 rule): the facade binds its 67
# constants at its FIRST import from whatever the registry holds at that
# instant. Importing it here, at collection time, right after `inference.core`
# has installed the server configuration, guarantees the values compared below
# are the server's - never a standalone default frozen in by a later reset.
import inference.core.workflows.environment as workflows_environment
from inference.core import env
from inference.core.interfaces.workflows_configuration import (
    build_configuration_from_env,
    install_workflows_configuration,
    server_workflows_configuration,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.inference.unit_tests.core.interfaces.http.test_http_api import (
    _build_plain_interface,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
WORKFLOW_SPECIFICATION = {"version": "1.0", "inputs": [], "steps": [], "outputs": []}

# (env.py attribute, how the same value is reached on the configuration).
# The drift gate below compares these keys with the facade's exports, so a
# symbol added to one and not the other fails - a hand-typed length assertion
# would not have (round-1 defect 9).
FIELDS = [
    ("WORKFLOWS_STEP_EXECUTION_MODE", lambda c: c.engine.step_execution_mode),
    (
        "WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT",
        lambda c: c.engine.async_future_result_timeout,
    ),
    ("WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH", lambda c: c.engine.max_inner_workflow_depth),
    ("WORKFLOWS_MAX_INNER_WORKFLOW_COUNT", lambda c: c.engine.max_inner_workflow_count),
    (
        "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS",
        lambda c: c.engine.allow_custom_python_execution,
    ),
    (
        "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE",
        lambda c: c.engine.custom_python_execution_mode,
    ),
    (
        "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE",
        lambda c: c.engine.allow_blocks_accessing_local_storage,
    ),
    (
        "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES",
        lambda c: c.engine.allow_blocks_accessing_environmental_variables,
    ),
    ("WORKFLOW_BLOCKS_WRITE_DIRECTORY", lambda c: c.engine.blocks_write_directory),
    ("WORKFLOW_DISABLED_BLOCK_TYPES", lambda c: list(c.engine.disabled_block_types)),
    (
        "WORKFLOW_DISABLED_BLOCK_PATTERNS",
        lambda c: list(c.engine.disabled_block_patterns),
    ),
    ("ENABLE_TENSOR_DATA_REPRESENTATION", lambda c: c.tensor.representation_enabled),
    ("WORKFLOWS_IMAGE_TENSOR_DEVICE", lambda c: c.tensor.image_tensor_device),
    (
        "WORKFLOWS_TENSOR_VISUALISATION_VALIDATE_OWNERS",
        lambda c: c.tensor.visualisation_validate_owners,
    ),
    (
        "WORKFLOWS_SAM_VIDEO_MASK_REPRESENTATION",
        lambda c: c.tensor.sam_video_mask_representation,
    ),
    (
        "WORKFLOWS_ENFORCE_DENSE_INSTANCE_MASKS",
        lambda c: c.tensor.enforce_dense_instance_masks,
    ),
    ("WORKFLOWS_REMOTE_API_TARGET", lambda c: c.remote.api_target),
    ("WORKFLOWS_REMOTE_API_KEY_TRANSPORT", lambda c: c.remote.api_key_transport),
    ("LOCAL_INFERENCE_API_URL", lambda c: c.remote.local_inference_api_url),
    ("HOSTED_DETECT_URL", lambda c: c.remote.hosted_detect_url),
    ("HOSTED_CLASSIFICATION_URL", lambda c: c.remote.hosted_classification_url),
    (
        "HOSTED_INSTANCE_SEGMENTATION_URL",
        lambda c: c.remote.hosted_instance_segmentation_url,
    ),
    (
        "HOSTED_SEMANTIC_SEGMENTATION_URL",
        lambda c: c.remote.hosted_semantic_segmentation_url,
    ),
    ("HOSTED_CORE_MODEL_URL", lambda c: c.remote.hosted_core_model_url),
    (
        "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE",
        lambda c: c.remote.max_step_batch_size,
    ),
    (
        "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS",
        lambda c: c.remote.max_step_concurrent_requests,
    ),
    ("API_BASE_URL", lambda c: c.platform.api_base_url),
    ("OFFLINE_MODE", lambda c: c.platform.offline_mode),
    ("SECURE_GATEWAY", lambda c: c.platform.secure_gateway),
    ("GCP_SERVERLESS", lambda c: c.platform.gcp_serverless),
    ("ALLOW_WORKFLOWS_FONTS_DOWNLOAD", lambda c: c.fonts.allow_download),
    ("MODEL_CACHE_DIR", lambda c: c.fonts.model_cache_dir),
    ("LMM_ENABLED", lambda c: c.models.lmm_enabled),
    ("CLIP_VERSION_ID", lambda c: c.models.clip_version_id),
    ("CORE_MODEL_SAM2_ENABLED", lambda c: c.models.core_model_sam2_enabled),
    ("CORE_MODEL_SAM3_ENABLED", lambda c: c.models.core_model_sam3_enabled),
    ("CORE_MODEL_PE_ENABLED", lambda c: c.models.core_model_pe_enabled),
    ("CORE_MODEL_GAZE_ENABLED", lambda c: c.models.core_model_gaze_enabled),
    ("SAM3_EXEC_MODE", lambda c: c.models.sam3_exec_mode),
    ("SAM3_3D_OBJECTS_ENABLED", lambda c: c.models.sam3_3d_objects_enabled),
    ("FLORENCE2_ENABLED", lambda c: c.models.florence2_enabled),
    ("QWEN_2_5_ENABLED", lambda c: c.models.qwen_2_5_enabled),
    ("QWEN_3_ENABLED", lambda c: c.models.qwen_3_enabled),
    ("QWEN_3_5_ENABLED", lambda c: c.models.qwen_3_5_enabled),
    ("SMOLVLM2_ENABLED", lambda c: c.models.smolvlm2_enabled),
    ("MOONDREAM2_ENABLED", lambda c: c.models.moondream2_enabled),
    ("DEPTH_ESTIMATION_ENABLED", lambda c: c.models.depth_estimation_enabled),
    ("COSMOS3_ENABLED", lambda c: c.models.cosmos3_enabled),
    ("GLM_OCR_ENABLED", lambda c: c.models.glm_ocr_enabled),
    ("MODAL_TOKEN_ID", lambda c: c.modal.token_id),
    ("MODAL_TOKEN_SECRET", lambda c: c.modal.token_secret),
    ("MODAL_WORKSPACE_NAME", lambda c: c.modal.workspace_name),
    ("MODAL_ALLOW_ANONYMOUS_EXECUTION", lambda c: c.modal.allow_anonymous_execution),
    ("MODAL_ANONYMOUS_WORKSPACE_NAME", lambda c: c.modal.anonymous_workspace_name),
    ("WEBEXEC_MODAL_APP_NAME", lambda c: c.modal.app_name),
    (
        "WEBEXEC_MODAL_EXECUTOR_IDLE_TTL_SECONDS",
        lambda c: c.modal.executor_idle_ttl_seconds,
    ),
    ("WEBEXEC_JPEG_QUALITY", lambda c: c.modal.jpeg_quality),
    ("WEBEXEC_TRANSPORT", lambda c: c.modal.transport),
    (
        "WEBEXEC_WS_CONNECT_TIMEOUT_SECONDS",
        lambda c: c.modal.ws_connect_timeout_seconds,
    ),
    ("WEBEXEC_WS_READ_TIMEOUT_SECONDS", lambda c: c.modal.ws_read_timeout_seconds),
    ("WEBEXEC_WS_CONNECTION_POOL_SIZE", lambda c: c.modal.ws_connection_pool_size),
    ("WEBEXEC_WS_FAIL_ON_SESSION_LOSS", lambda c: c.modal.ws_fail_on_session_loss),
    ("WEBEXEC_WS_IDLE_RELEASE_SECONDS", lambda c: c.modal.ws_idle_release_seconds),
    ("API_KEY", lambda c: c.secrets.api_key),
    (
        "ROBOFLOW_INTERNAL_SERVICE_NAME",
        lambda c: c.secrets.roboflow_internal_service_name,
    ),
    (
        "ROBOFLOW_INTERNAL_SERVICE_SECRET",
        lambda c: c.secrets.roboflow_internal_service_secret,
    ),
    ("INFERENCE_DEBUG_OUTPUT_DIR", lambda c: c.debug.output_dir),
]

ROOTS = {
    "inference/core/interfaces/http/http_api.py": 2,
    "inference/core/interfaces/stream/inference_pipeline.py": 1,
    "inference_cli/lib/workflows/local_image_adapter.py": 1,
}
CONFIGURATION_KEY = "workflows_core.configuration"


# --------------------------------------------------------------------------
# Inventory drift gate
# --------------------------------------------------------------------------


def test_the_field_table_matches_the_facade_exports() -> None:
    exported = {
        name
        for name in vars(workflows_environment)
        if name.isupper() and not name.startswith("_")
    }
    tabled = {name for name, _ in FIELDS}
    assert tabled == exported, {
        "missing_from_table": sorted(exported - tabled),
        "missing_from_facade": sorted(tabled - exported),
    }
    assert len(tabled) == 67, len(tabled)


def test_every_name_workflows_imports_from_the_facade_is_exported() -> None:
    # Live AST scan: catches a workflows module importing a name the facade
    # does not define, and stays meaningful after the codemod lands.
    exported = {
        name
        for name in vars(workflows_environment)
        if name.isupper() and not name.startswith("_")
    }
    workflows_root = REPO_ROOT / "inference" / "core" / "workflows"
    requested = set()
    for path in sorted(workflows_root.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_bytes().decode("utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "inference.core.workflows.environment"
            ):
                requested.update(alias.name for alias in node.names)
    assert requested <= exported, sorted(requested - exported)


def test_every_symbol_still_imported_from_env_is_in_the_field_table() -> None:
    # Vacuous once Task 5.7 lands; until then it proves nothing is left behind.
    workflows_root = REPO_ROOT / "inference" / "core" / "workflows"
    deferred = (
        workflows_root / "core_steps" / "sinks" / "roboflow",
        workflows_root / "core_steps" / "integrations" / "roboflow",
    )
    remaining = set()
    for path in sorted(workflows_root.rglob("*.py")):
        if "__pycache__" in str(path) or any(d in path.parents for d in deferred):
            continue
        tree = ast.parse(path.read_bytes().decode("utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "inference.core.env":
                remaining.update(alias.name for alias in node.names)
    assert remaining <= {name for name, _ in FIELDS}, sorted(
        remaining - {name for name, _ in FIELDS}
    )


# --------------------------------------------------------------------------
# Field-by-field parity with inference.core.env
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name, reader", FIELDS, ids=[name for name, _ in FIELDS])
def test_server_configuration_equals_env_field_by_field(name, reader) -> None:
    expected = getattr(env, name)
    actual = reader(build_configuration_from_env())
    assert actual == expected, name
    assert type(actual) is type(expected), (name, type(actual), type(expected))


def test_the_facade_equals_env_field_by_field() -> None:
    # The 151 workflows files read the FACADE, so the facade - not just the
    # builder - is what has to agree with env.py in the running server.
    for name, _ in FIELDS:
        expected = getattr(env, name)
        actual = getattr(workflows_environment, name)
        assert actual == expected, name
        assert type(actual) is type(expected), (name, type(actual), type(expected))


def test_server_workflows_configuration_is_memoised() -> None:
    assert server_workflows_configuration() is server_workflows_configuration()


def test_install_is_idempotent() -> None:
    install_workflows_configuration()
    install_workflows_configuration()
    from inference.core.workflows.configuration import get_configuration

    assert get_configuration() is server_workflows_configuration()


def test_importing_inference_core_installs_before_any_workflows_module_loads() -> None:
    # The ordering guarantee this design rests on: `inference/core/__init__.py`
    # installs the configuration before the constants facade
    # (`inference.core.workflows.environment`) is first imported - the
    # configuration-independent bootstrap modules (`prototypes/platform_errors`,
    # `workflows/configuration`) may load earlier, the facade may not.
    # BOTH flags are pinned - `env.py:1486` ANDs `USE_INFERENCE_MODELS` into
    # the tensor flag (round-3 defect 5) - and the child reports env.py's
    # EFFECTIVE flag, so a gate-induced False can never pass as "both agree".
    script = (
        "import json\n"
        "import inference.core.workflows.environment as environment\n"
        "from inference.core import env\n"
        "print(json.dumps([environment.ENABLE_TENSOR_DATA_REPRESENTATION,"
        " env.ENABLE_TENSOR_DATA_REPRESENTATION]))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT / "inference_models"),
            "ENABLE_TENSOR_DATA_REPRESENTATION": "True",
            "USE_INFERENCE_MODELS": "True",
        },
    )
    assert completed.returncode == 0, completed.stderr
    facade_value, env_value = json.loads(completed.stdout.strip().splitlines()[-1])
    assert (
        env_value is True
    ), "USE_INFERENCE_MODELS is pinned: the effective flag must be on"
    assert facade_value is True
    assert facade_value == env_value


# --------------------------------------------------------------------------
# Composition-root wiring - AST-precise, not "the string appears somewhere"
# --------------------------------------------------------------------------


def _init_parameter_dicts_reaching_engine_init(source: str):
    """For every `ExecutionEngine.init(...)` call, return the set of literal
    init-parameter keys its `init_parameters=` argument carries.

    Resolves the argument to a Name, then collects every `{...}` literal
    assigned to that name - DIRECTLY, or as the first argument of a helper
    call such as Phase 9's `install_workflows_platform_bindings({...})`
    (round-3 defect 1: under R-U both HTTP literals are already wrapped) -
    and every `name["key"] = ...` / `name.setdefault("key", ...)` in the same
    enclosing function. Round-1 defect 7: a substring search anywhere in the
    file passes even when one of the two HTTP roots is missing the key, or
    when the key only appears in a comment.
    """
    tree = ast.parse(source)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def enclosing_function(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                return node
        return None

    def literal_of(value):
        if isinstance(value, ast.Dict):
            return value
        if (
            isinstance(value, ast.Call)
            and value.args
            and isinstance(value.args[0], ast.Dict)
        ):
            return value.args[0]
        return None

    results = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "init"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ExecutionEngine"
        ):
            continue
        argument = next(
            (kw.value for kw in node.keywords if kw.arg == "init_parameters"), None
        )
        assert isinstance(argument, ast.Name), ast.dump(node)
        scope = enclosing_function(node)
        keys = set()
        for inner in ast.walk(scope):
            if isinstance(inner, ast.Assign):
                literal = literal_of(inner.value)
                for target in inner.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == argument.id
                        and literal is not None
                    ):
                        keys.update(
                            k.value for k in literal.keys if isinstance(k, ast.Constant)
                        )
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == argument.id
                        and isinstance(target.slice, ast.Constant)
                    ):
                        keys.add(target.slice.value)
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "setdefault"
                and isinstance(inner.func.value, ast.Name)
                and inner.func.value.id == argument.id
                and inner.args
                and isinstance(inner.args[0], ast.Constant)
            ):
                keys.add(inner.args[0].value)
        results.append(keys)
    return results


def test_the_root_scanner_reads_a_wrapped_and_a_plain_literal_alike() -> None:
    """Round-3 defect 1: under R-U Phase 9 has already wrapped both HTTP
    literals in `install_workflows_platform_bindings({...})`
    (DECONTAMINATION.PLAN.PHASE-9.MD Task 9.3 Step 10), and the round-2 scanner
    returned `[set(), set()]` for that shape. Five synthetic roots: plain
    literal, wrapped literal, subscript assignment, `setdefault`, and a wrapped
    literal WITHOUT the key - which must still be reported missing."""
    source = (
        "def plain():\n"
        "    params = {'workflows_core.configuration': 1}\n"
        "    ExecutionEngine.init(workflow_definition={}, init_parameters=params)\n"
        "def wrapped():\n"
        "    params = install_workflows_platform_bindings({\n"
        "        'workflows_core.api_key': 1,\n"
        "        'workflows_core.configuration': 1,\n"
        "    })\n"
        "    ExecutionEngine.init(workflow_definition={}, init_parameters=params)\n"
        "def assigned(params):\n"
        "    params['workflows_core.configuration'] = 1\n"
        "    ExecutionEngine.init(workflow_definition={}, init_parameters=params)\n"
        "def defaulted(params):\n"
        "    params.setdefault('workflows_core.configuration', 1)\n"
        "    ExecutionEngine.init(workflow_definition={}, init_parameters=params)\n"
        "def missing():\n"
        "    params = install_workflows_platform_bindings({'workflows_core.api_key': 1})\n"
        "    ExecutionEngine.init(workflow_definition={}, init_parameters=params)\n"
    )
    per_call = _init_parameter_dicts_reaching_engine_init(source)
    assert [CONFIGURATION_KEY in keys for keys in per_call] == [
        True,
        True,
        True,
        True,
        False,
    ]


@pytest.mark.parametrize("relative, expected_calls", sorted(ROOTS.items()))
def test_every_engine_call_at_every_root_carries_the_configuration(
    relative, expected_calls
) -> None:
    source = (REPO_ROOT / relative).read_text(encoding="utf-8")
    per_call_keys = _init_parameter_dicts_reaching_engine_init(source)
    assert len(per_call_keys) == expected_calls, relative
    for keys in per_call_keys:
        assert CONFIGURATION_KEY in keys, (relative, sorted(keys))


def test_http_run_route_binds_the_server_configuration(monkeypatch) -> None:
    """Execution proof for HTTP root #1 (`http_api.py:1597`).

    Round-2 defect 2: the AST test alone accepts
    `"workflows_core.configuration": None`. This drives the real route with
    `_build_plain_interface` + `TestClient` - the harness
    `tests/inference/unit_tests/core/interfaces/http/test_http_api.py:1766`
    already provides - and asserts the captured object's IDENTITY.
    """
    import inference.core.interfaces.http.http_api as http_api

    interface, _ = _build_plain_interface(monkeypatch)
    engine = MagicMock()
    engine.run.return_value = []
    execution_engine_mock = MagicMock()
    execution_engine_mock.init.return_value = engine
    monkeypatch.setattr(http_api, "ExecutionEngine", execution_engine_mock)

    with TestClient(interface.app) as client:
        response = client.post(
            "/workflows/run",
            headers={"Authorization": "Bearer header-key"},
            json={"specification": WORKFLOW_SPECIFICATION, "inputs": {}},
        )

    assert response.status_code == 200, response.text
    init_parameters = execution_engine_mock.init.call_args.kwargs["init_parameters"]
    assert init_parameters[CONFIGURATION_KEY] is server_workflows_configuration()


def test_http_validate_route_binds_the_server_configuration(monkeypatch) -> None:
    """Execution proof for HTTP root #2 (`http_api.py:2546`)."""
    import inference.core.interfaces.http.http_api as http_api

    interface, _ = _build_plain_interface(monkeypatch)
    execution_engine_mock = MagicMock()
    monkeypatch.setattr(http_api, "ExecutionEngine", execution_engine_mock)

    with TestClient(interface.app) as client:
        response = client.post(
            "/workflows/validate?api_key=some-key", json=WORKFLOW_SPECIFICATION
        )

    assert response.status_code == 200, response.text
    init_parameters = execution_engine_mock.init.call_args.kwargs["init_parameters"]
    assert init_parameters[CONFIGURATION_KEY] is server_workflows_configuration()


def test_the_pipeline_binds_the_server_configuration(monkeypatch) -> None:
    """Execution proof for the pipeline root (`inference_pipeline.py:751`).

    Round-2 defect 2: the round-1 test patched `inference_pipeline.ExecutionEngine`,
    which does not exist - the name is imported INSIDE `init_with_workflow`
    (`inference_pipeline.py:704`), so the test died with `AttributeError`
    before reaching its assertion. Patching `init` on the DEFINING class works
    regardless of where the name is imported.
    """
    from inference.core.interfaces.stream.inference_pipeline import InferencePipeline

    execution_engine_init = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(ExecutionEngine, "init", execution_engine_init)
    monkeypatch.setattr(
        InferencePipeline, "init_with_custom_logic", MagicMock(return_value=MagicMock())
    )

    InferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification={"version": "1.0"},
        model_manager=MagicMock(),
    )

    init_parameters = execution_engine_init.call_args.kwargs["init_parameters"]
    assert init_parameters[CONFIGURATION_KEY] is server_workflows_configuration()


def test_the_pipeline_preserves_a_caller_supplied_configuration(monkeypatch) -> None:
    """`setdefault`, not assignment: a caller-supplied object must reach the
    engine so the conflict check - not this file - decides whether it is
    acceptable."""
    from inference.core.interfaces.stream.inference_pipeline import InferencePipeline

    supplied = dataclasses.replace(server_workflows_configuration())
    execution_engine_init = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(ExecutionEngine, "init", execution_engine_init)
    monkeypatch.setattr(
        InferencePipeline, "init_with_custom_logic", MagicMock(return_value=MagicMock())
    )

    InferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification={"version": "1.0"},
        model_manager=MagicMock(),
        workflow_init_parameters={CONFIGURATION_KEY: supplied},
    )

    init_parameters = execution_engine_init.call_args.kwargs["init_parameters"]
    assert init_parameters[CONFIGURATION_KEY] is supplied


def _run_cli_root(tmp_path, monkeypatch, init_params=None) -> dict:
    from inference_cli.lib.workflows import local_image_adapter

    image_path = str(tmp_path / "frame.png")
    assert cv2.imwrite(image_path, np.zeros((8, 12, 3), dtype=np.uint8))
    captured = {}

    class _FakeEngine:
        def run(self, runtime_parameters, serialize_results=False):
            return [{"ok": True}]

    def _capturing_init(**kwargs):
        captured.update(kwargs)
        return _FakeEngine()

    monkeypatch.setattr(local_image_adapter.ExecutionEngine, "init", _capturing_init)
    local_image_adapter._run_workflow_for_single_image_with_inference(
        model_manager=MagicMock(),
        image_path=image_path,
        workflow_specification=WORKFLOW_SPECIFICATION,
        workflow_id=None,
        image_input_name="image",
        workflow_parameters=None,
        api_key="test-key",
        thread_pool_executor=MagicMock(),
        max_concurrent_workflows_steps=1,
        workflows_execution_engine_init_params=init_params,
    )
    return captured


def test_the_cli_root_binds_the_server_configuration(tmp_path, monkeypatch) -> None:
    captured = _run_cli_root(tmp_path, monkeypatch)
    assert (
        captured["init_parameters"][CONFIGURATION_KEY]
        is server_workflows_configuration()
    )


def test_the_cli_root_preserves_a_caller_supplied_configuration(
    tmp_path, monkeypatch
) -> None:
    supplied = dataclasses.replace(server_workflows_configuration())
    captured = _run_cli_root(
        tmp_path, monkeypatch, init_params={CONFIGURATION_KEY: supplied}
    )
    assert captured["init_parameters"][CONFIGURATION_KEY] is supplied


def _distinct_from(value):
    """An override guaranteed to differ from the registered default, whatever
    the deployment set it to (round-4 defect 2: a deployment may legitimately
    run with ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=False or
    ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES=False, env.py:1118-1123,
    and the loader forwards those resolved values, loader.py:1594-1596)."""
    if isinstance(value, bool):  # before str/None: bool is the common case
        return not value
    if value is None:
        return "engine-scoped"
    if isinstance(value, str):
        return value + "-engine-scoped"
    raise AssertionError(f"no override rule for {type(value)!r}")


def test_named_init_parameters_still_override_per_engine() -> None:
    """The supported per-engine channel, exercised with the real resolver.

    D1 makes the CONFIGURATION process-wide; these named parameters are how a
    host varies behaviour per engine, and they must keep winning over the
    `REGISTERED_INITIALIZERS` defaults. The default is resolved FIRST and the
    override derived from it, so the test proves precedence under permissive
    and restrictive deployment flags alike.
    """
    from inference.core.workflows.execution_engine.introspection.blocks_loader import (
        load_initializers,
    )
    from inference.core.workflows.execution_engine.v1.compiler.steps_initialiser import (
        retrieve_init_parameter_values,
    )

    initializers = load_initializers()
    for parameter in [
        "api_key",
        "disable_sinks",
        "allow_access_to_file_system",
        "allowed_write_directory",
        "allow_access_to_environmental_variables",
    ]:
        default_value = retrieve_init_parameter_values(
            block_name="step",
            block_init_parameter=parameter,
            block_source="workflows_core",
            explicit_init_parameters={},
            initializers=initializers,
        )
        override = _distinct_from(default_value)
        assert override != default_value, parameter
        resolved = retrieve_init_parameter_values(
            block_name="step",
            block_init_parameter=parameter,
            block_source="workflows_core",
            explicit_init_parameters={f"workflows_core.{parameter}": override},
            initializers=initializers,
        )
        assert resolved == override, parameter


def test_step_execution_mode_override_reaches_the_engine() -> None:
    from inference.core.workflows.execution_engine.v1 import core as engine_core
    from inference.core.workflows.prototypes.block import StepExecutionMode

    resolved = engine_core._retrieve_step_execution_mode(
        init_parameters={"workflows_core.step_execution_mode": StepExecutionMode.REMOTE}
    )
    assert resolved is StepExecutionMode.REMOTE
    fallback = engine_core._retrieve_step_execution_mode(init_parameters={})
    assert fallback == StepExecutionMode(engine_core.WORKFLOWS_STEP_EXECUTION_MODE)
