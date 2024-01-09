from typing import Any, Dict

from inference.enterprise.deployments.complier.execution_engine import execute_graph
from inference.enterprise.deployments.complier.graph_parser import (
    construct_execution_graph,
)
from inference.enterprise.deployments.complier.validator import validate_deployment_spec
from inference.enterprise.deployments.entities.deployment_specs import (
    DeploymentSpecification,
)
from inference.enterprise.deployments.errors import InvalidSpecificationVersionError


def compile_and_execute(
    deployment_spec: dict,
    runtime_parameters: Dict[str, Any],
) -> dict:
    parsed_deployment_spec = DeploymentSpecification.parse_obj(deployment_spec)
    if parsed_deployment_spec.specification.version != "1.0":
        raise InvalidSpecificationVersionError(
            f"Only version 1.0 of deployment specs is supported."
        )
    validate_deployment_spec(deployment_spec=parsed_deployment_spec.specification)
    execution_graph = construct_execution_graph(
        deployment_spec=parsed_deployment_spec.specification
    )
    return execute_graph(
        execution_graph=execution_graph,
        runtime_parameters=runtime_parameters,
        model_manager=None,
    )
