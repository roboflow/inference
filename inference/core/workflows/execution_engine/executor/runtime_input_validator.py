from copy import copy
from typing import Any, Dict, List

from pydantic import ValidationError

from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.compiler.entities import (
    InputSubstitution,
)


def validate_runtime_input(
    runtime_parameters: Dict[str, Any],
    input_substitutions: List[InputSubstitution],
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
