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

LONG_DESCRIPTION = """
The **Environment Secrets Store** block is a secure and flexible solution for fetching secrets stored as 
**environmental variables**. It is designed to enable Workflows to access sensitive information, 
such as API keys or service credentials, without embedding them directly into the Workflow definitions. 

This block simplifies the integration of external services while prioritizing security and adaptability. You can
use secrets fetched from environment (which can be set by system administrator to be available in self-hosted
`inference` server) to pass as inputs to other steps.

!!! Tip "Credentials security"

    It is strongly advised to use secrets providers (available when running self-hosted `inference` server)
    or workflows parameters to pass credentials. **Do not hardcode secrets in Workflows definitions.**
    
!!! Important "Blocks limitations"

    This block can only run on self-hosted `inference` server, we Roboflow does not allow exporting env
    variables from Hosted Platform due to security concerns. 

#### 🛠️ Block configuration

Block has configuration parameter `variables_storing_secrets` that must be filled with list of
environmental variables which will be exposed as block outputs. Thanks to that, you can
use them as inputs for other blocks. Please note that names of outputs will be lowercased. For example,
the following settings:
```
variables_storing_secrets=["MY_SECRET_A", "MY_SECRET_B"]
```
will generate the following outputs:

* `my_secret_a`

* `my_secret_b`
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Environment Secrets Store",
            "version": "v1",
            "short_description": "Fetch secrets from environmental variables.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "secrets_provider",
            "ui_manifest": {
                "section": "advanced",
                "icon": "far fa-key",
            },
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
                "access to environment variables is forbidden - use self-hosted `inference`"
            )
        return {
            variable_name.lower(): os.getenv(variable_name)
            for variable_name in variables_storing_secrets
        }
