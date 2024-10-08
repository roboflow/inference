from copy import copy
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.profiling.core import (
    WorkflowsProfiler,
    execution_phase,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    InputSubstitution,
)


@execution_phase(
    name="runtime_input_validation",
    categories=["execution_engine_operation"],
)
def validate_runtime_input(
    runtime_parameters: Dict[str, Any],
    input_substitutions: List[InputSubstitution],
    profiler: Optional[WorkflowsProfiler] = None,
) -> None:
    for input_substitution in input_substitutions:
        try:
            step_manifest_copy = copy(input_substitution.step_manifest)
            setattr(
                step_manifest_copy,
                input_substitution.manifest_property,
                runtime_parameters[input_substitution.input_parameter_name],
            )
        except ValidationError as e:
            raise RuntimeInputError(
                public_message=f"Provided input is incompatible with manifest "
                f"of step {input_substitution.step_manifest.name} regarding property "
                f"{input_substitution.manifest_property} provided in input parameter named "
                f"{input_substitution.input_parameter_name}. Details available in inner error object.",
                context="workflow_execution | runtime_input_validation",
                inner_error=e,
            ) from e
