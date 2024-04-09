from copy import copy
from typing import Any, Dict, List

from pydantic import ValidationError

from inference.enterprise.workflows.execution_engine.compiler.entities import (
    InputSubstitution,
)


def validate_runtime_input(
    runtime_parameters: Dict[str, Any],
    input_substitutions: List[InputSubstitution],
) -> None:
    try:
        for input_substitution in input_substitutions:
            step_manifest_copy = copy(input_substitution.step_manifest)
            setattr(
                step_manifest_copy,
                input_substitution.manifest_property,
                runtime_parameters[input_substitution.input_parameter_name],
            )
    except ValidationError as e:
        # TODO: error handling
        raise e
