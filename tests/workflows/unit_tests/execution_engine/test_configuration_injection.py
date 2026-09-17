"""The engine refuses a configuration that contradicts the process one.

`workflows_core.configuration` is a CONSISTENCY ASSERTION (D1): the whole
object is process-wide, so a host that hands one engine a different object is
mis-wired and must be told, not quietly ignored.

Round-2 defect 1 / round-3 defect 4: the check looks ONLY at
`workflows_core.configuration`. A plugin's bare `configuration` init parameter
and any OTHER namespace (`my_plugin.configuration`,
`dynamic_workflows_blocks.configuration`) are existing supported paths
(`steps_initialiser.py:124-133`) and pass through untouched.

Round-3 defect 3 / round-4 defect 1: a callable or an explicit None under
the dedicated key is REFUSED, never invoked or forwarded. `retrieve_init_parameter_values` returns an explicit value unchanged
(`steps_initialiser.py:124`), so a factory the engine had "validated" by calling
it would still reach every block as the function itself.

Round-2 defect 4: every conflicting variant is derived by PERTURBING the
installed value, never hard-coded - the suite is required to run with
`ENABLE_TENSOR_DATA_REPRESENTATION=True USE_INFERENCE_MODELS=True` too, where a
hard-coded `representation_enabled=True` equals the installed value and
`pytest.raises` never fires.
"""

import dataclasses
from typing import List, Literal

import pytest

from inference.core.workflows import configuration as workflows_configuration
from inference.core.workflows import environment as workflows_environment
from inference.core.workflows.configuration import WorkflowsConfiguration
from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_initializers,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
)
from inference.core.workflows.execution_engine.v1.compiler.steps_initialiser import (
    initialise_step,
    retrieve_init_parameter_values,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)
from tests.workflows.unit_tests.test_configuration import perturb, variant_of

TRIVIAL_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "x"}],
    "steps": [],
    "outputs": [{"type": "JsonField", "name": "x", "selector": "$inputs.x"}],
}
CONFIGURATION_KEY = "workflows_core.configuration"

# One field per group, and for each the façade constant that proves an accepted
# override would have changed nothing.
GROUP_FIELDS = [
    (
        "engine",
        "allow_custom_python_execution",
        "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS",
    ),
    ("tensor", "representation_enabled", "ENABLE_TENSOR_DATA_REPRESENTATION"),
    ("remote", "api_target", "WORKFLOWS_REMOTE_API_TARGET"),
    ("platform", "offline_mode", "OFFLINE_MODE"),
    ("fonts", "allow_download", "ALLOW_WORKFLOWS_FONTS_DOWNLOAD"),
    ("models", "sam3_exec_mode", "SAM3_EXEC_MODE"),
    ("modal", "transport", "WEBEXEC_TRANSPORT"),
    ("secrets", "roboflow_internal_service_name", "ROBOFLOW_INTERNAL_SERVICE_NAME"),
    ("debug", "output_dir", "INFERENCE_DEBUG_OUTPUT_DIR"),
]
IDS = [f"{group}.{field}" for group, field, _ in GROUP_FIELDS]


class _ConfigurationConsumerManifest(WorkflowBlockManifest):
    type: Literal["phase5/configuration_consumer@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class _ConfigurationConsumerBlock(WorkflowBlock):
    """A core-sourced block that declares the `configuration` init parameter."""

    def __init__(self, configuration):
        self.configuration = configuration

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["configuration"]

    @classmethod
    def get_manifest(cls):
        return _ConfigurationConsumerManifest

    def run(self, *args, **kwargs):
        return []


def _initialise_consumer(explicit_init_parameters: dict):
    return initialise_step(
        step_manifest=_ConfigurationConsumerManifest(
            type="phase5/configuration_consumer@v1", name="consumer"
        ),
        block_specification=BlockSpecification(
            block_source="workflows_core",
            identifier="workflows_core.ConfigurationConsumer",
            block_class=_ConfigurationConsumerBlock,
            manifest_class=_ConfigurationConsumerManifest,
        ),
        explicit_init_parameters=explicit_init_parameters,
        initializers=load_initializers(),
    ).step


def test_engine_accepts_the_installed_configuration() -> None:
    engine = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={
            CONFIGURATION_KEY: workflows_configuration.get_configuration()
        },
    )
    assert engine is not None


def test_engine_accepts_an_equal_but_distinct_configuration() -> None:
    engine = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={
            CONFIGURATION_KEY: dataclasses.replace(
                workflows_configuration.get_configuration()
            )
        },
    )
    assert engine is not None


@pytest.mark.parametrize("group, field, _constant", GROUP_FIELDS, ids=IDS)
def test_engine_refuses_a_configuration_that_differs_in_any_group(
    group, field, _constant
) -> None:
    conflicting = variant_of(workflows_configuration.get_configuration(), group, field)
    with pytest.raises(WorkflowEnvironmentConfigurationError) as raised:
        ExecutionEngine.init(
            workflow_definition=TRIVIAL_WORKFLOW,
            init_parameters={CONFIGURATION_KEY: conflicting},
        )
    assert f"{group}.{field}" in raised.value.public_message


def test_the_refusal_is_not_bypassed_by_a_warm_compilation_cache() -> None:
    # `compiler/core.py:64` caches by definition + engine version, so a second
    # init of the same workflow skips compilation entirely. The check has to run
    # BEFORE compile_workflow or it would be skipped with it.
    installed = workflows_configuration.get_configuration()
    ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={CONFIGURATION_KEY: installed},
    )
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        ExecutionEngine.init(
            workflow_definition=TRIVIAL_WORKFLOW,
            init_parameters={
                CONFIGURATION_KEY: variant_of(installed, "remote", "api_target")
            },
        )


def test_engine_still_works_without_a_configuration_init_parameter() -> None:
    engine = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW, init_parameters={}
    )
    assert engine is not None


def test_a_plugins_bare_configuration_parameter_is_not_intercepted() -> None:
    """Round-2 defect 1, reproduced against the real functions.

    `_retrieve_init_parameter` (`v1/core.py:89-96`) falls back from
    `workflows_core.configuration` to the BARE name, and a bare explicit
    parameter is how a plugin has always supplied its own `configuration`
    (`steps_initialiser.py:129`). Using the round-1 wording, a plugin passing
    `{"configuration": {"threshold": 0.5}}` reached
    `ensure_process_configuration_matches` and died with
    `AttributeError: 'dict' object has no attribute 'engine'`.
    """
    plugin_value = {"threshold": 0.5}

    engine = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={"configuration": plugin_value},
    )
    assert engine is not None

    resolved = retrieve_init_parameter_values(
        block_name="step",
        block_init_parameter="configuration",
        block_source="my_plugin",
        explicit_init_parameters={"configuration": plugin_value},
        initializers=load_initializers(),
    )
    assert resolved is plugin_value


@pytest.mark.parametrize("namespace", ["my_plugin", "dynamic_workflows_blocks"])
def test_another_namespaces_configuration_parameter_is_not_intercepted(
    namespace,
) -> None:
    """Round-3 defect 4: the round-2 check also reserved
    `dynamic_workflows_blocks.configuration`. A plugin picks its own
    `BLOCKS_SOURCE` (`blocks_loader.py:297`) and may name a parameter
    `configuration`; only `workflows_core.configuration` is ours."""
    plugin_value = {"threshold": 0.5}
    key = f"{namespace}.configuration"

    engine = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW, init_parameters={key: plugin_value}
    )
    assert engine is not None

    resolved = retrieve_init_parameter_values(
        block_name="step",
        block_init_parameter="configuration",
        block_source=namespace,
        explicit_init_parameters={key: plugin_value},
        initializers=load_initializers(),
    )
    assert resolved is plugin_value


def test_an_invalid_value_under_the_dedicated_key_is_a_workflows_error() -> None:
    with pytest.raises(WorkflowEnvironmentConfigurationError) as raised:
        ExecutionEngine.init(
            workflow_definition=TRIVIAL_WORKFLOW,
            init_parameters={CONFIGURATION_KEY: {"threshold": 0.5}},
        )
    assert "WorkflowsConfiguration" in raised.value.public_message


def test_a_callable_under_the_dedicated_key_is_refused_not_invoked() -> None:
    """Round-3 defect 3. The real resolver hands an EXPLICIT value to blocks
    unchanged (`steps_initialiser.py:124`) - shown first - so an engine that
    called the factory to "validate" it would still deliver the function to
    every block. The engine therefore refuses callables outright."""
    installed = workflows_configuration.get_configuration()

    def factory():
        return installed

    resolved = retrieve_init_parameter_values(
        block_name="step",
        block_init_parameter="configuration",
        block_source="workflows_core",
        explicit_init_parameters={CONFIGURATION_KEY: factory},
        initializers=load_initializers(),
    )
    assert resolved is factory, "the resolver does not materialise explicit values"

    with pytest.raises(WorkflowEnvironmentConfigurationError) as raised:
        ExecutionEngine.init(
            workflow_definition=TRIVIAL_WORKFLOW,
            init_parameters={CONFIGURATION_KEY: factory},
        )
    assert "WorkflowsConfiguration" in raised.value.public_message
    assert "function" in raised.value.public_message


def test_an_explicit_none_under_the_dedicated_key_is_refused() -> None:
    """Round-4 defect 1. Presence is decided by the engine (`key in
    init_parameters`), so a None under the key is an EXPLICIT None - and the
    real resolver would deliver exactly that to a block ahead of the registered
    default (shown first). Omission is fine; explicit None is refused."""
    delivered = _initialise_consumer({CONFIGURATION_KEY: None}).configuration
    assert delivered is None, "the resolver prefers an explicit value, even None"

    with pytest.raises(WorkflowEnvironmentConfigurationError) as raised:
        ExecutionEngine.init(
            workflow_definition=TRIVIAL_WORKFLOW,
            init_parameters={CONFIGURATION_KEY: None},
        )
    assert "NoneType" in raised.value.public_message


def test_a_configuration_consuming_block_receives_the_installed_object() -> None:
    """What a block that declares `configuration` actually gets, through the
    real `initialise_step`: the very object under `workflows_core.configuration`
    when the host supplies it, and the loader's registered default
    (`REGISTERED_INITIALIZERS["configuration"]`, Task 5.3) otherwise. In both
    cases a `WorkflowsConfiguration` equal to the installed one - never a
    function (round-3 defect 3)."""
    installed = workflows_configuration.get_configuration()

    explicit = _initialise_consumer({CONFIGURATION_KEY: installed})
    assert explicit.configuration is installed

    defaulted = _initialise_consumer({})
    assert isinstance(defaulted.configuration, WorkflowsConfiguration)
    assert defaulted.configuration == installed
    assert not callable(defaulted.configuration)


@pytest.mark.parametrize("group, field, constant", GROUP_FIELDS, ids=IDS)
def test_a_rejected_value_would_indeed_not_have_been_honoured(
    group, field, constant
) -> None:
    """The reason the refusal is right, made explicit: every field is read from
    a module constant frozen at import time, so an accepted override would have
    changed nothing."""
    installed = workflows_configuration.get_configuration()
    facade_value = getattr(workflows_environment, constant)
    installed_value = getattr(getattr(installed, group), field)
    if constant == "WORKFLOW_DISABLED_BLOCK_TYPES":  # list vs tuple, not used here
        installed_value = list(installed_value)
    assert facade_value == installed_value
    assert perturb(installed_value) != facade_value
