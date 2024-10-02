from typing import Any, Callable, Dict, List, Optional, Union

from inference.core.workflows.errors import (
    BlockInitParameterNotProvidedError,
    BlockInterfaceError,
    UnknownManifestType,
)
from inference.core.workflows.execution_engine.profiling.core import (
    WorkflowsProfiler,
    execution_phase,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
    InitialisedStep,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest


@execution_phase(
    name="steps_initialisation",
    categories=["execution_engine_operation"],
)
def initialise_steps(
    steps_manifest: List[WorkflowBlockManifest],
    available_blocks: List[BlockSpecification],
    explicit_init_parameters: Dict[str, Union[Any, Callable[[None], Any]]],
    initializers: Dict[str, Union[Any, Callable[[None], Any]]],
    profiler: Optional[WorkflowsProfiler] = None,
) -> List[InitialisedStep]:
    available_blocks_by_manifest_class = {
        block.manifest_class: block for block in available_blocks
    }
    initialised_steps = []
    for step_manifest in steps_manifest:
        if type(step_manifest) not in available_blocks_by_manifest_class:
            raise UnknownManifestType(
                public_message=f"Provided step manifest of type ({type(step_manifest)}) "
                f"which is not compatible with registered workflow blocks.",
                context="workflow_compilation | steps_initialisation",
            )
        block_specification = available_blocks_by_manifest_class[type(step_manifest)]
        initialised_step = initialise_step(
            step_manifest=step_manifest,
            block_specification=block_specification,
            explicit_init_parameters=explicit_init_parameters,
            initializers=initializers,
        )
        initialised_steps.append(initialised_step)
    return initialised_steps


def initialise_step(
    step_manifest: WorkflowBlockManifest,
    block_specification: BlockSpecification,
    explicit_init_parameters: Dict[str, Union[Any, Callable[[None], Any]]],
    initializers: Dict[str, Union[Any, Callable[[None], Any]]],
) -> InitialisedStep:
    block_init_parameters = block_specification.block_class.get_init_parameters()
    init_parameters_values = retrieve_init_parameters_values(
        block_name=step_manifest.name,
        block_init_parameters=block_init_parameters,
        block_source=block_specification.block_source,
        explicit_init_parameters=explicit_init_parameters,
        initializers=initializers,
    )
    try:
        step = block_specification.block_class(**init_parameters_values)
    except TypeError as e:
        raise BlockInterfaceError(
            public_message=f"While initialisation of step {step_manifest.name} of type: {step_manifest.type} there "
            f"was an error in creating instance of workflow block. One of parameters defined "
            f"({list(init_parameters_values.keys())}) was invalid or block class do not implement all methods. "
            f"Details: {e}",
            context="workflow_compilation | steps_initialisation",
            inner_error=e,
        ) from e
    return InitialisedStep(
        block_specification=block_specification,
        manifest=step_manifest,
        step=step,
    )


def retrieve_init_parameters_values(
    block_name: str,
    block_init_parameters: List[str],
    block_source: str,
    explicit_init_parameters: Dict[str, Union[Any, Callable[[None], Any]]],
    initializers: Dict[str, Union[Any, Callable[[None], Any]]],
) -> Dict[str, Any]:
    return {
        block_init_parameter: retrieve_init_parameter_values(
            block_name=block_name,
            block_init_parameter=block_init_parameter,
            block_source=block_source,
            explicit_init_parameters=explicit_init_parameters,
            initializers=initializers,
        )
        for block_init_parameter in block_init_parameters
    }


def retrieve_init_parameter_values(
    block_name: str,
    block_init_parameter: str,
    block_source: str,
    explicit_init_parameters: Dict[str, Union[Any, Callable[[None], Any]]],
    initializers: Dict[str, Union[Any, Callable[[None], Any]]],
) -> Any:
    full_parameter_name = f"{block_source}.{block_init_parameter}"
    if full_parameter_name in explicit_init_parameters:
        return explicit_init_parameters[full_parameter_name]
    if full_parameter_name in initializers:
        return call_if_callable(initializers[full_parameter_name])
    if block_init_parameter in explicit_init_parameters:
        return explicit_init_parameters[block_init_parameter]
    if block_init_parameter in initializers:
        return call_if_callable(initializers[block_init_parameter])
    raise BlockInitParameterNotProvidedError(
        public_message=f"Could not resolve init parameter {block_init_parameter} to initialise "
        f"step `{block_name}` from plugin: {block_source}.",
        context="workflow_compilation | steps_initialisation",
    )


def call_if_callable(value: Union[Any, Callable[[None], Any]]) -> Any:
    if callable(value):
        return value()
    return value
