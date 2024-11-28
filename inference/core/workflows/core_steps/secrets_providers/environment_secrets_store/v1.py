import os
from typing import List, Literal, Optional, Type

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import SECRET_KIND
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Environment Secrets Store",
            "version": "v1",
            "short_description": "Fetches secrets from environmental variables",
            "long_description": "TODO",
            "license": "Apache-2.0",
            "block_type": "secrets_provider",
        }
    )
    type: Literal["roboflow_core/environment_secrets_store@v1"]
    variables_storing_secrets: List[str] = Field(
        description="List with names of environment variables to fetch. Each will create separate block output.",
        examples=[["MY_API_KEY", "OTHER_API_KEY"]],
        min_items=1,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="*")]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=variable_name.lower(), kind=[SECRET_KIND])
            for variable_name in self.variables_storing_secrets
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class EnvironmentSecretsStoreBlockV1(WorkflowBlock):

    def __init__(self, allow_access_to_environmental_variables: bool):
        super().__init__()
        self._allow_access_to_environmental_variables = (
            allow_access_to_environmental_variables
        )

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["allow_access_to_environmental_variables"]

    def run(
        self,
        variables_storing_secrets: List[str],
    ) -> BlockResult:
        if not self._allow_access_to_environmental_variables:
            raise RuntimeError(
                "`roboflow_core/environment_secrets_store@v1` block cannot run in this environment - "
                "access to environment variables is forbidden - use self-hosted `inference` or "
                "Roboflow Dedicated Deployment."
            )
        return {
            variable_name.lower(): os.getenv(variable_name)
            for variable_name in variables_storing_secrets
        }
