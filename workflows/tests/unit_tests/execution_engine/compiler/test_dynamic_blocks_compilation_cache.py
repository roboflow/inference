from copy import deepcopy
from typing import Optional
from unittest import mock

import numpy as np
import pytest
from roboflow_workflows.errors import WorkflowEnvironmentConfigurationError
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.v1.compiler import core as compiler_core
from roboflow_workflows.execution_engine.v1.compiler.core import (
    COMPILATION_CACHE,
    compile_workflow_graph,
)
from roboflow_workflows.execution_engine.v1.core import EXECUTION_ENGINE_V1_VERSION
from roboflow_workflows.execution_engine.v1.dynamic_blocks import (
    block_assembler,
    block_scaffolding,
    modal_executor,
)
from roboflow_workflows.prototypes.observer import NullExecutionObserver

CORE_OBSERVER_KEY = "workflows_core.execution_observer"
CORE_RESOLVER_KEY = "workflows_core.workspace_resolver"


class _Resolver:
    def resolve_workspace(self, api_key: Optional[str]) -> Optional[str]:
        return None


def _workflow(run_function_code: str, *, block_type: str = "CachedBlock") -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": block_type,
                    "inputs": {
                        "image": {
                            "type": "DynamicInputDefinition",
                            "selector_types": ["input_image"],
                        }
                    },
                    "outputs": {"x": {"type": "DynamicOutputDefinition", "kind": []}},
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": run_function_code,
                },
            }
        ],
        "steps": [{"type": block_type, "name": "b", "image": "$inputs.image"}],
        "outputs": [{"type": "JsonField", "name": "x", "selector": "$steps.b.x"}],
    }


CONSTANT_BLOCK_CODE = "def run(self, image):\n    return {'x': 1}\n"
SHARED_STATE_BLOCK_CODE = (
    "CALLS = []\n\n\n"
    "def run(self, image):\n"
    "    CALLS.append(1)\n"
    "    return {'x': len(CALLS)}\n"
)
IMAGE = np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def clean_compilation_cache():
    with COMPILATION_CACHE._cache_lock:
        COMPILATION_CACHE._cache.clear()
        COMPILATION_CACHE._keys_buffer.clear()
    yield
    with COMPILATION_CACHE._cache_lock:
        COMPILATION_CACHE._cache.clear()
        COMPILATION_CACHE._keys_buffer.clear()


@pytest.fixture
def local_mode():
    with mock.patch.object(
        block_assembler, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "local"
    ), mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "local"
    ), mock.patch.object(
        block_assembler, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", True
    ):
        yield


@pytest.fixture
def modal_mode():
    with mock.patch.object(
        block_assembler, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    ), mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    ), mock.patch.object(
        modal_executor, "validate_code_in_modal", return_value=True
    ) as validate_code_in_modal:
        yield validate_code_in_modal


def _step_block(engine: ExecutionEngine):
    step = next(iter(engine._engine._compiled_workflow.steps.values())).step

    return step


def test_local_dynamic_block_compilation_is_cached(local_mode) -> None:
    workflow = _workflow(CONSTANT_BLOCK_CODE)

    first = compile_workflow_graph(workflow_definition=deepcopy(workflow))
    second = compile_workflow_graph(workflow_definition=deepcopy(workflow))

    assert first is second


def test_modal_dynamic_block_compilation_is_not_cached(modal_mode) -> None:
    workflow = _workflow(CONSTANT_BLOCK_CODE)
    init_parameters = {CORE_RESOLVER_KEY: _Resolver()}

    first = compile_workflow_graph(
        workflow_definition=deepcopy(workflow), init_parameters=init_parameters
    )
    second = compile_workflow_graph(
        workflow_definition=deepcopy(workflow), init_parameters=init_parameters
    )

    assert first is not second
    assert modal_mode.call_count == 2


def test_cached_dynamic_block_binds_each_engines_context(local_mode) -> None:
    workflow = _workflow(CONSTANT_BLOCK_CODE)
    first_observer, second_observer = NullExecutionObserver(), NullExecutionObserver()
    first_resolver, second_resolver = _Resolver(), _Resolver()

    first_engine = ExecutionEngine.init(
        workflow_definition=deepcopy(workflow),
        init_parameters={
            "workflows_core.api_key": "first-key",
            CORE_OBSERVER_KEY: first_observer,
            CORE_RESOLVER_KEY: first_resolver,
        },
    )
    second_engine = ExecutionEngine.init(
        workflow_definition=deepcopy(workflow),
        init_parameters={
            "workflows_core.api_key": "second-key",
            "dynamic_workflows_blocks.api_key": "second-dynamic-key",
            CORE_OBSERVER_KEY: second_observer,
            CORE_RESOLVER_KEY: second_resolver,
        },
    )
    first_block, second_block = _step_block(first_engine), _step_block(second_engine)

    first_result = first_engine.run(runtime_parameters={"image": IMAGE})
    second_result = second_engine.run(runtime_parameters={"image": IMAGE})

    assert type(first_block) is type(second_block)
    assert first_block is not second_block
    assert first_block._api_key == "first-key"
    assert second_block._api_key == "second-dynamic-key"
    assert first_block._workspace_resolver is first_resolver
    assert second_block._workspace_resolver is second_resolver
    assert first_block._execution_observer is first_observer
    assert second_block._execution_observer is second_observer
    assert first_result[0]["x"] == 1
    assert second_result[0]["x"] == 1


def test_repeated_init_compiles_dynamic_blocks_once(local_mode) -> None:
    workflow = _workflow(CONSTANT_BLOCK_CODE)

    with mock.patch.object(
        compiler_core,
        "compile_dynamic_blocks",
        wraps=compiler_core.compile_dynamic_blocks,
    ) as compile_dynamic_blocks:
        engines = [
            ExecutionEngine.init(
                workflow_definition=deepcopy(workflow),
                init_parameters={"workflows_core.api_key": "key"},
            )
            for _ in range(20)
        ]

    assert compile_dynamic_blocks.call_count == 1
    block_classes = {type(_step_block(engine)) for engine in engines}
    manifest_classes = {
        type(engine._engine._compiled_workflow.steps["b"].manifest)
        for engine in engines
    }
    assert len(block_classes) == 1
    assert len(manifest_classes) == 1


def test_warm_cache_still_rejects_disallowed_custom_python(local_mode) -> None:
    workflow = _workflow(CONSTANT_BLOCK_CODE)
    init_parameters = {"workflows_core.api_key": "key"}

    with mock.patch.object(
        compiler_core,
        "compile_dynamic_blocks",
        wraps=compiler_core.compile_dynamic_blocks,
    ) as compile_dynamic_blocks:
        ExecutionEngine.init(
            workflow_definition=deepcopy(workflow),
            init_parameters=init_parameters,
        )
        with mock.patch.object(
            block_assembler, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", False
        ):
            with pytest.raises(WorkflowEnvironmentConfigurationError):
                ExecutionEngine.init(
                    workflow_definition=deepcopy(workflow),
                    init_parameters=init_parameters,
                )

    assert compile_dynamic_blocks.call_count == 1
    assert len(COMPILATION_CACHE._cache) == 1


def test_warm_local_cache_is_not_used_in_modal_mode(local_mode) -> None:
    workflow = _workflow(CONSTANT_BLOCK_CODE)
    ExecutionEngine.init(
        workflow_definition=deepcopy(workflow),
        init_parameters={"workflows_core.api_key": "key"},
    )
    warm_key = COMPILATION_CACHE.get_hash_key(
        workflow_definition=workflow,
        execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
    )
    warm = COMPILATION_CACHE.get(key=warm_key)
    assert warm is not None

    with mock.patch.object(
        block_assembler, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    ), mock.patch.object(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    ), mock.patch.object(
        modal_executor, "validate_code_in_modal", return_value=True
    ) as validate_code_in_modal:
        result = compile_workflow_graph(
            workflow_definition=deepcopy(workflow),
            execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
            init_parameters={CORE_RESOLVER_KEY: _Resolver()},
        )

    validate_code_in_modal.assert_called_once()
    assert result is not warm


def test_cached_dynamic_block_shares_module_state(local_mode) -> None:
    """Module-level state of a local block is shared by engines of one definition."""
    workflow = _workflow(SHARED_STATE_BLOCK_CODE, block_type="StatefulCachedBlock")
    results = []

    for _ in range(2):
        engine = ExecutionEngine.init(
            workflow_definition=deepcopy(workflow),
            init_parameters={"workflows_core.api_key": "key"},
        )
        results.append(engine.run(runtime_parameters={"image": IMAGE})[0]["x"])

    assert results == [1, 2]
