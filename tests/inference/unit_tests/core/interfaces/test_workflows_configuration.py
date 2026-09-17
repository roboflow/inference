import ast
import dataclasses
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

# Module level on purpose (round-5 defect 1, D5 rule): the facade binds its
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
    (
        "ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES",
        lambda c: c.engine.allow_webhook_sink_to_non_global_addresses,
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
    (
        "WORKFLOWS_INNER_WORKFLOW_REMOTE_TARGET",
        lambda c: c.remote.inner_workflow_remote_target,
    ),
    (
        "WORKFLOWS_INNER_WORKFLOW_REMOTE_DISPATCH_REQUEST_TIMEOUT",
        lambda c: c.remote.inner_workflow_remote_dispatch_request_timeout,
    ),
    (
        "OPENAI_COMPATIBLE_ALLOWED_BASE_URLS",
        lambda c: set(c.remote.openai_compatible_allowed_base_urls),
    ),
    ("API_BASE_URL", lambda c: c.platform.api_base_url),
    ("OFFLINE_MODE", lambda c: c.platform.offline_mode),
    ("SECURE_GATEWAY", lambda c: c.platform.secure_gateway),
    ("GCP_SERVERLESS", lambda c: c.platform.gcp_serverless),
    ("ALLOW_WORKFLOWS_FONTS_DOWNLOAD", lambda c: c.fonts.allow_download),
    ("MODEL_CACHE_DIR", lambda c: c.fonts.model_cache_dir),
    ("LMM_ENABLED", lambda c: c.models.lmm_enabled),
    (
        "WORKFLOWS_VLM_SEGMENTATION_MAX_POLYGON_VERTICES",
        lambda c: c.models.vlm_segmentation_max_polygon_vertices,
    ),
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
    assert len(tabled) == 72, len(tabled)


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


# --------------------------------------------------------------------------
# Field-by-field parity with inference.core.env
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name, reader", FIELDS, ids=[name for name, _ in FIELDS])
def test_server_configuration_equals_env_field_by_field(name, reader) -> None:
    expected = getattr(env, name)
    actual = reader(build_configuration_from_env())
    assert actual == expected, name
    assert type(actual) is type(expected), (name, type(actual), type(expected))


@pytest.mark.parametrize(
    "name, value",
    [
        ("WORKFLOWS_VLM_SEGMENTATION_MAX_POLYGON_VERTICES", 17),
        ("WORKFLOWS_INNER_WORKFLOW_REMOTE_TARGET", "https://deployment.example/v1"),
        ("WORKFLOWS_INNER_WORKFLOW_REMOTE_DISPATCH_REQUEST_TIMEOUT", 12.5),
        ("OPENAI_COMPATIBLE_ALLOWED_BASE_URLS", set()),
        ("OPENAI_COMPATIBLE_ALLOWED_BASE_URLS", {"https://approved.example/v1"}),
    ],
)
def test_server_configuration_preserves_new_settings(monkeypatch, name, value):
    monkeypatch.setattr(env, name, value)
    configuration = build_configuration_from_env()
    assert dict(FIELDS)[name](configuration) == value
    monkeypatch.setattr(
        "inference.core.workflows.configuration.get_configuration",
        lambda: configuration,
    )
    facade = runpy.run_path(workflows_environment.__file__)
    assert facade[name] == value


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
# Composition-root wiring - caller overrides
#
# The default binding at each root is asserted against a REAL engine in
# `test_image_codec_binding.py`; what is left here is the `setdefault`
# contract, with the engine mocked.
# --------------------------------------------------------------------------


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
