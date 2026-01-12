from unittest import mock

import pytest

from inference.core.workflows.core_steps.formatters.expression.v1 import BlockManifest
from inference.core.workflows.errors import (
    DynamicBlockError,
    WorkflowEnvironmentConfigurationError,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
    block_scaffolding,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_scaffolding import (
    assembly_custom_python_block,
    create_dynamic_module,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    PythonCode,
)


def test_create_dynamic_module_when_syntax_error_happens() -> None:
    # given
    init_function = """
def init_fun() -> Dict[str, Any]:
    return {"a": 35}    
"""
    run_function = """
def run_function( -> BlockResult:
    return {"result": a + b}
"""
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        init_function_code=init_function,
        init_function_name="init_fun",
        imports=["import math"],
    )

    # when
    with pytest.raises(DynamicBlockError):
        _ = create_dynamic_module(
            block_type_name="some", python_code=python_code, module_name="my_module"
        )


def test_create_dynamic_module_when_creation_should_succeed() -> None:
    # given
    init_function = """
def init_fun() -> Dict[str, Any]:
    return {"a": 35}    
"""
    run_function = """
def run_function(a, b) -> BlockResult:
    return {"result": a + b}
"""
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        init_function_code=init_function,
        init_function_name="init_fun",
        imports=["import math"],
    )

    # when
    module = create_dynamic_module(
        block_type_name="some", python_code=python_code, module_name="my_module"
    )

    # then
    assert module.init_fun() == {"a": 35}
    assert module.run_function(3, 5) == {"result": 8}


def test_assembly_custom_python_block() -> None:
    # given
    manifest = BlockManifest
    init_function = """
def init_fun() -> Dict[str, Any]:
    return {"a": 6}    
"""
    run_function = """
def run_function(self, a, b) -> BlockResult:
    return {"result": a + b + self._init_results["a"]}
    """
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        init_function_code=init_function,
        init_function_name="init_fun",
        imports=["import math"],
    )

    # when
    workflow_block_class = assembly_custom_python_block(
        block_type_name="some",
        unique_identifier="unique-id",
        manifest=manifest,
        python_code=python_code,
    )
    workflow_block_instance = workflow_block_class()
    execution_result = workflow_block_instance.run(a=3, b=5)

    # then
    assert workflow_block_class.get_init_parameters() == [
        "api_key"
    ], "Expected api_key parameter defined"
    assert (
        workflow_block_class.get_manifest() == BlockManifest
    ), "Expected manifest to be returned"
    assert execution_result == {
        "result": 14
    }, "Expected result of 3 + 5 + 6 (last value from init)"


def test_assembly_custom_python_block_when_init_not_provided() -> None:
    # given
    manifest = BlockManifest
    run_function = """
def run_function(self, a, b) -> BlockResult:
    return {"result": a + b + len(self._init_results)}
    """
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        imports=["import math"],
    )

    # when
    workflow_block_class = assembly_custom_python_block(
        block_type_name="some",
        unique_identifier="unique-id",
        manifest=manifest,
        python_code=python_code,
    )
    workflow_block_instance = workflow_block_class()
    execution_result = workflow_block_instance.run(a=3, b=5)

    # then
    assert workflow_block_class.get_init_parameters() == [
        "api_key"
    ], "Expected api_key parameters defined"
    assert (
        workflow_block_class.get_manifest() == BlockManifest
    ), "Expected manifest to be returned"
    assert execution_result == {
        "result": 8
    }, "Expected result of 3 + 5 + 0 (last value from init)"


def test_assembly_custom_python_block_when_run_function_not_found() -> None:
    # given
    manifest = BlockManifest
    init_function = """
def init_fun() -> Dict[str, Any]:
    return {"a": 6}    
"""
    run_function = """
def run_function(self, a, b) -> BlockResult:
    return {"result": a + b + self._init_results["a"]}
    """
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="invalid",
        init_function_code=init_function,
        init_function_name="init_fun",
        imports=["import math"],
    )

    # when
    with pytest.raises(DynamicBlockError):
        _ = assembly_custom_python_block(
            block_type_name="some",
            unique_identifier="unique-id",
            manifest=manifest,
            python_code=python_code,
        )


def test_assembly_custom_python_block_when_init_function_not_found() -> None:
    # given
    manifest = BlockManifest
    init_function = """
def init_fun() -> Dict[str, Any]:
    return {"a": 6}    
"""
    run_function = """
def run_function(self, a, b) -> BlockResult:
    return {"result": a + b + self._init_results["a"]}
    """
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        init_function_code=init_function,
        init_function_name="invalid",
        imports=["import math"],
    )

    # when
    with pytest.raises(DynamicBlockError):
        _ = assembly_custom_python_block(
            block_type_name="some",
            unique_identifier="unique-id",
            manifest=manifest,
            python_code=python_code,
        )


@mock.patch.object(
    block_scaffolding, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", False
)
def test_run_assembled_custom_python_block_when_custom_python_forbidden() -> None:
    # given
    manifest = BlockManifest
    init_function = """
def init_fun() -> Dict[str, Any]:
    return {"a": 6}    
"""
    run_function = """
def run_function(self, a, b) -> BlockResult:
    return {"result": a + b + self._init_results["a"]}
    """
    python_code = PythonCode(
        type="PythonCode",
        run_function_code=run_function,
        run_function_name="run_function",
        init_function_code=init_function,
        init_function_name="init_fun",
        imports=["import math"],
    )

    # when
    workflow_block_class = assembly_custom_python_block(
        block_type_name="some",
        unique_identifier="unique-id",
        manifest=manifest,
        python_code=python_code,
    )
    workflow_block_instance = workflow_block_class()
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        _ = workflow_block_instance.run(a=3, b=5)
